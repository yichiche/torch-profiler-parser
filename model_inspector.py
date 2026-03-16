"""Static model structure inspector for SGLang model files.

Parses model Python source files using the `ast` module (no GPU, no model loading)
and produces a readable nn.Module hierarchy tree + config metadata summary.

Usage:
    python model_inspector.py <model_file.py> [options]

Examples:
    python model_inspector.py /path/to/deepseek_v2.py --list-classes
    python model_inspector.py /path/to/deepseek_v2.py --root DeepseekV2ForCausalLM
    python model_inspector.py /path/to/deepseek_v2.py --config /path/to/config.json -o out.txt

    python3 /home/yichiche/agent-box/profile/model_inspector.py /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py --config /data/deepseek-ai/DeepSeek-R1-0528/config.json
    python3 /home/yichiche/agent-box/profile/model_inspector.py /sgl-workspace/sglang/python/sglang/srt/models/grok.py --config /data/huggingface/hub/xai-org/grok-2/config.json
"""

import argparse
import ast
import gzip
import json
import logging
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModuleAssignment:
    """Represents a `self.xxx = SomeModule(...)` found in __init__."""

    attr_name: str
    class_name: str
    raw_args: str
    line_no: int
    is_conditional: bool = False
    condition_text: str = ""
    # For hybrid dispatch: maps a key (e.g. "attention") to a class name
    dispatch_map: Optional[Dict[str, str]] = None


@dataclass
class ClassInfo:
    """Parsed class with its name, bases, module assignments, and config accesses."""

    name: str
    bases: List[str]
    assignments: List[ModuleAssignment] = field(default_factory=list)
    config_accesses: List[str] = field(default_factory=list)
    line_no: int = 0


@dataclass
class ModuleTree:
    """Tree node for the module hierarchy."""

    class_name: str
    attr_name: str
    children: List["ModuleTree"] = field(default_factory=list)
    multiplicity: str = ""
    is_conditional: bool = False
    condition_text: str = ""
    line_no: int = 0
    # For hybrid dispatch: maps a key (e.g. "attention") to class name
    dispatch_map: Optional[Dict[str, str]] = None


@dataclass
class ConfigMetadata:
    """Wrapper around relevant config.json key-value pairs."""

    data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Known module heuristics
# ---------------------------------------------------------------------------

_MODULE_KEYWORDS = {
    "Linear", "Norm", "Embedding", "Attention", "MoE", "MLP",
    "Processor", "TopK", "Head", "Layer", "Conv", "Pool",
    "Dropout", "BatchNorm", "LayerNorm", "RMSNorm", "GroupNorm",
    "Module", "Sequential", "ModuleList", "Indexer", "Gate",
    "Dispatcher", "Communicator", "Rotary", "Rope",
}

_NON_MODULE_NAMES = {
    "Parameter", "Tensor", "bool", "int", "float", "str", "list", "dict",
    "tuple", "set", "None", "True", "False", "torch", "F", "nn",
    "SiluAndMul", "Stage", "Enum", "LazyValue", "PackWeightMethod",
}

_NON_MODULE_PREFIXES = {"nn.Parameter", "torch."}


def _looks_like_module(class_name: str, known_classes: set) -> bool:
    """Heuristic: does this class name look like an nn.Module subclass?"""
    # Factory calls like get_moe_impl_class()() are likely module constructors
    if class_name.endswith("()"):
        inner = class_name[:-2]
        # Heuristic: factory functions with "moe", "impl", "class" in name
        lower_inner = inner.lower()
        if any(kw in lower_inner for kw in ["moe", "impl", "class", "layer", "module"]):
            return True
    if class_name in _NON_MODULE_NAMES:
        return False
    for prefix in _NON_MODULE_PREFIXES:
        if class_name.startswith(prefix) and class_name != "nn.Module":
            # nn.Parameter, torch.zeros, etc.
            if "Linear" not in class_name and "Norm" not in class_name:
                return False
    # Filter out static/class method calls like Foo.bar_method()
    if "." in class_name and not class_name.startswith("nn."):
        parts = class_name.rsplit(".", 1)
        method = parts[-1]
        # If the last part starts lowercase, it's a method call, not a constructor
        if method and method[0].islower():
            return False
    if class_name in known_classes:
        return True
    for keyword in _MODULE_KEYWORDS:
        if keyword in class_name:
            return True
    # PascalCase heuristic: at least two uppercase letters and no dots (except nn.)
    if class_name.startswith("nn."):
        return True
    upper_count = sum(1 for c in class_name if c.isupper())
    if upper_count >= 2 and "." not in class_name:
        return True
    return False


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _get_call_class_name(node: ast.Call) -> Optional[str]:
    """Extract class name from a Call node.

    Handles:
      - ClassName(...)           -> "ClassName"
      - module.ClassName(...)    -> "module.ClassName"
      - factory()(...)           -> inner call name + "()"
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        value = func.value
        if isinstance(value, ast.Name):
            return f"{value.id}.{func.attr}"
        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            return f"{value.value.id}.{value.attr}.{func.attr}"
    # Factory call: get_moe_impl_class(quant_config)(...)
    if isinstance(func, ast.Call):
        inner = _get_call_class_name(func)
        if inner:
            return f"{inner}()"
    return None


def _get_source_segment(source_lines: List[str], node: ast.AST) -> str:
    """Get a simplified source representation of an AST node."""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _extract_lambda_body_class(node: ast.Call) -> Optional[str]:
    """From make_layers(..., lambda idx, prefix: ClassName(...), ...),
    extract the ClassName from the lambda body."""
    for arg in list(node.args) + [kw.value for kw in node.keywords]:
        if isinstance(arg, ast.Lambda):
            body = arg.body
            if isinstance(body, ast.Call):
                return _get_call_class_name(body)
    return None


def _extract_func_ref_name(node: ast.Call) -> Optional[str]:
    """From make_layers(N, get_layer, ...), extract the function reference name 'get_layer'."""
    if len(node.args) >= 2:
        arg = node.args[1]
        if isinstance(arg, ast.Name):
            return arg.id
    return None


def _find_local_func(body: List[ast.stmt], func_name: str) -> Optional[ast.FunctionDef]:
    """Find a local function definition in a list of statements."""
    for stmt in body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == func_name:
            return stmt
    return None


def _extract_classes_from_func(func_node: ast.FunctionDef) -> List[str]:
    """Extract class names constructed in return statements of a local function."""
    classes = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            name = _get_call_class_name(node.value)
            if name:
                classes.append(name)
    return classes


def _extract_dispatch_dict(tree: ast.Module, dict_name: str) -> Dict[str, str]:
    """Extract a module-level dict mapping string keys to class names.

    Handles patterns like:
        ALL_DECODER_LAYER_TYPES = {
            "attention": Qwen3HybridAttentionDecoderLayer,
            "linear_attention": Qwen3HybridLinearDecoderLayer,
        }
    """
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_name:
                    if isinstance(node.value, ast.Dict):
                        for key, val in zip(node.value.keys, node.value.values):
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                if isinstance(val, ast.Name):
                                    result[key.value] = val.id
                                elif isinstance(val, ast.Attribute):
                                    result[key.value] = ast.unparse(val)
    return result


def _detect_dispatch_in_func(
    func_node: ast.FunctionDef, module_tree: ast.Module
) -> Optional[Dict[str, str]]:
    """Detect if a local function dispatches via a module-level dict.

    Looks for patterns like:
        layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]
        return layer_class(...)
    """
    # Look for: xxx = SOME_DICT[...]
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Subscript):
                subscript_val = node.value.value
                if isinstance(subscript_val, ast.Name):
                    dict_name = subscript_val.id
                    dispatch = _extract_dispatch_dict(module_tree, dict_name)
                    if dispatch:
                        return dispatch
    return None


# ---------------------------------------------------------------------------
# ModelFileParser
# ---------------------------------------------------------------------------

class ModelFileParser:
    """AST-based parser for model Python source files."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(filepath, "r") as f:
            self.source = f.read()
        self.source_lines = self.source.splitlines()
        self.tree = ast.parse(self.source, filename=filepath)
        self.classes: Dict[str, ClassInfo] = {}

    def parse(self) -> Dict[str, ClassInfo]:
        """Parse all classes in the file."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                bases = self._extract_bases(node)
                info = ClassInfo(
                    name=node.name,
                    bases=bases,
                    line_no=node.lineno,
                )
                self.classes[node.name] = info

        # Second pass: extract __init__ assignments now that we know all class names
        known_classes = set(self.classes.keys())
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name in self.classes:
                info = self.classes[node.name]
                init_method = self._find_init(node)
                if init_method:
                    info.assignments = self._extract_init_assignments(
                        init_method.body, known_classes, init_body=init_method.body
                    )
                    info.config_accesses = self._extract_config_accesses(init_method)

        return self.classes

    def _extract_bases(self, node: ast.ClassDef) -> List[str]:
        """Extract base class names."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))
        return bases

    def _find_init(self, class_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
        """Find the __init__ method in a class."""
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return item
        return None

    def _extract_init_assignments(
        self,
        body: List[ast.stmt],
        known_classes: set,
        is_conditional: bool = False,
        condition_text: str = "",
        init_body: Optional[List[ast.stmt]] = None,
    ) -> List[ModuleAssignment]:
        """Walk __init__ body (including if/else) to find self.xxx = ClassName(...)."""
        if init_body is None:
            init_body = body
        assignments = []

        for stmt in body:
            # self.xxx = ClassName(...)
            if isinstance(stmt, ast.Assign):
                assignments.extend(
                    self._process_assign(
                        stmt, known_classes, is_conditional, condition_text,
                        init_body=init_body,
                    )
                )

            # if condition: ... else: ...
            elif isinstance(stmt, ast.If):
                cond_text = ast.unparse(stmt.test)
                assignments.extend(
                    self._extract_init_assignments(
                        stmt.body, known_classes,
                        is_conditional=True,
                        condition_text=cond_text,
                        init_body=init_body,
                    )
                )
                if stmt.orelse:
                    else_text = f"not ({cond_text})"
                    # If orelse is another If (elif), keep its own condition
                    if len(stmt.orelse) == 1 and isinstance(stmt.orelse[0], ast.If):
                        assignments.extend(
                            self._extract_init_assignments(
                                stmt.orelse, known_classes,
                                is_conditional=True,
                                condition_text="",
                                init_body=init_body,
                            )
                        )
                    else:
                        assignments.extend(
                            self._extract_init_assignments(
                                stmt.orelse, known_classes,
                                is_conditional=True,
                                condition_text=else_text,
                                init_body=init_body,
                            )
                        )

        return assignments

    def _process_assign(
        self,
        stmt: ast.Assign,
        known_classes: set,
        is_conditional: bool,
        condition_text: str,
        init_body: Optional[List[ast.stmt]] = None,
    ) -> List[ModuleAssignment]:
        """Process a single assignment statement."""
        results = []

        # Handle make_layers() calls — both tuple unpacking and direct assignment
        call_node = None
        attr_name = None

        if isinstance(stmt.value, ast.Call):
            call_name = _get_call_class_name(stmt.value)
            if call_name == "make_layers":
                call_node = stmt.value
                # Tuple unpacking: self.layers, self.start_layer, self.end_layer = make_layers(...)
                if (
                    len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Tuple)
                ):
                    for elt in stmt.targets[0].elts:
                        if (
                            isinstance(elt, ast.Attribute)
                            and isinstance(elt.value, ast.Name)
                            and elt.value.id == "self"
                        ):
                            attr_name = elt.attr
                            break
                # Direct: self.layers = make_layers(...)
                elif (
                    len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Attribute)
                    and isinstance(stmt.targets[0].value, ast.Name)
                    and stmt.targets[0].value.id == "self"
                ):
                    attr_name = stmt.targets[0].attr

        if call_node and attr_name:
            num_layers_arg = ""
            if call_node.args:
                num_layers_arg = ast.unparse(call_node.args[0])

            # Try lambda body first
            constructed_class = _extract_lambda_body_class(call_node)
            dispatch_map = None

            # Try function reference (e.g. make_layers(N, get_layer, ...))
            if not constructed_class and init_body:
                func_ref = _extract_func_ref_name(call_node)
                if func_ref:
                    local_func = _find_local_func(init_body, func_ref)
                    if local_func:
                        # Check for dispatch dict pattern
                        dispatch_map = _detect_dispatch_in_func(local_func, self.tree)
                        if dispatch_map:
                            # Use first class as representative
                            constructed_class = next(iter(dispatch_map.values()))
                        else:
                            # Direct class construction in return
                            func_classes = _extract_classes_from_func(local_func)
                            if func_classes:
                                constructed_class = func_classes[0]
                                if len(func_classes) > 1:
                                    dispatch_map = {
                                        f"variant_{i}": c
                                        for i, c in enumerate(func_classes)
                                    }

            if constructed_class:
                results.append(ModuleAssignment(
                    attr_name=attr_name,
                    class_name=constructed_class,
                    raw_args=f"make_layers({num_layers_arg}, ...)",
                    line_no=stmt.lineno,
                    is_conditional=is_conditional,
                    condition_text=condition_text,
                    dispatch_map=dispatch_map,
                ))
            return results

        # Handle nn.ModuleList([ClassName(...) for ... in range(...)]) pattern
        if isinstance(stmt.value, ast.Call):
            call_name = _get_call_class_name(stmt.value)
            if call_name == "nn.ModuleList" and stmt.value.args:
                first_arg = stmt.value.args[0]
                if isinstance(first_arg, ast.ListComp) and isinstance(first_arg.elt, ast.Call):
                    inner_class = _get_call_class_name(first_arg.elt)
                    if inner_class and _looks_like_module(inner_class, known_classes):
                        for target in stmt.targets:
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                            ):
                                range_arg = ""
                                if first_arg.generators:
                                    gen = first_arg.generators[0]
                                    if isinstance(gen.iter, ast.Call):
                                        range_arg = ast.unparse(gen.iter)
                                raw_args = f"nn.ModuleList([{inner_class}(...) for ... in {range_arg}])"
                                results.append(ModuleAssignment(
                                    attr_name=target.attr,
                                    class_name=inner_class,
                                    raw_args=raw_args,
                                    line_no=stmt.lineno,
                                    is_conditional=is_conditional,
                                    condition_text=condition_text,
                                ))
                                return results

        # Standard: self.xxx = ClassName(...)
        for target in stmt.targets:
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                continue

            attr_name = target.attr
            value = stmt.value

            if not isinstance(value, ast.Call):
                continue

            class_name = _get_call_class_name(value)
            if not class_name:
                continue

            if not _looks_like_module(class_name, known_classes):
                continue

            raw_args = ast.unparse(value)
            results.append(ModuleAssignment(
                attr_name=attr_name,
                class_name=class_name,
                raw_args=raw_args,
                line_no=stmt.lineno,
                is_conditional=is_conditional,
                condition_text=condition_text,
            ))

        return results

    def _extract_config_accesses(self, func_node: ast.FunctionDef) -> List[str]:
        """Find config.xxx attribute accesses in the function."""
        accesses = set()
        for node in ast.walk(func_node):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "config"
            ):
                accesses.add(node.attr)
        return sorted(accesses)


# ---------------------------------------------------------------------------
# HierarchyBuilder
# ---------------------------------------------------------------------------

class HierarchyBuilder:
    """Builds a module tree from parsed class info."""

    def __init__(self, class_map: Dict[str, ClassInfo]):
        self.class_map = class_map

    def build(self, root_class_name: str) -> ModuleTree:
        """Build the module hierarchy starting from root_class_name."""
        if root_class_name not in self.class_map:
            return ModuleTree(class_name=root_class_name, attr_name="(root)")
        return self._build_node(root_class_name, "(root)", visited=set())

    def _build_node(
        self, class_name: str, attr_name: str, visited: set
    ) -> ModuleTree:
        node = ModuleTree(class_name=class_name, attr_name=attr_name)

        if class_name in visited:
            # Avoid infinite recursion
            return node
        visited = visited | {class_name}

        info = self.class_map.get(class_name)
        if not info:
            return node

        node.line_no = info.line_no

        for assignment in info.assignments:
            child_class = assignment.class_name
            # Strip factory call markers
            resolved_class = child_class.rstrip("()")

            multiplicity = self._detect_multiplicity(assignment)

            if resolved_class in self.class_map:
                child_node = self._build_node(resolved_class, assignment.attr_name, visited)
            else:
                child_node = ModuleTree(
                    class_name=child_class,
                    attr_name=assignment.attr_name,
                )

            child_node.multiplicity = multiplicity
            child_node.is_conditional = assignment.is_conditional
            child_node.condition_text = assignment.condition_text
            child_node.line_no = assignment.line_no
            child_node.dispatch_map = assignment.dispatch_map
            node.children.append(child_node)

        return node

    def _detect_multiplicity(self, assignment: ModuleAssignment) -> str:
        """Detect if this assignment creates multiple layers."""
        if "make_layers" in assignment.raw_args:
            # Try to extract the count argument
            # Pattern: make_layers(config.num_hidden_layers, ...)
            raw = assignment.raw_args
            if "make_layers(" in raw:
                inner = raw.split("make_layers(", 1)[1]
                count_arg = inner.split(",", 1)[0].strip()
                return f"x N ({count_arg})"
        if "ModuleList" in assignment.raw_args:
            import re
            m = re.search(r'range\(([^)]+)\)', assignment.raw_args)
            if m:
                count_arg = m.group(1).rsplit(",", 1)[-1].strip()
                return f"x N ({count_arg})"
            return "x N"
        return ""


# ---------------------------------------------------------------------------
# ConfigParser
# ---------------------------------------------------------------------------

class ConfigParser:
    """Parse HuggingFace config.json and extract relevant keys."""

    RELEVANT_KEYS = [
        "architectures", "model_type",
        "hidden_size", "intermediate_size", "moe_intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
        "num_experts", "shared_expert_intermediate_size",
        "n_group", "topk_group", "topk_method",
        "first_k_dense_replace", "moe_layer_freq",
        "full_attention_interval", "layers_block_type",
        "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
        "q_lora_rank", "kv_lora_rank",
        "head_dim", "partial_rotary_factor",
        "linear_key_head_dim", "linear_value_head_dim",
        "linear_num_key_heads", "linear_num_value_heads",
        "linear_conv_kernel_dim",
        "vocab_size", "max_position_embeddings",
        "rms_norm_eps", "rope_theta", "rope_scaling",
        "routed_scaling_factor",
        "tie_word_embeddings",
    ]

    @classmethod
    def parse(cls, config_path: str) -> ConfigMetadata:
        """Read config.json and extract relevant fields."""
        with open(config_path, "r") as f:
            raw = json.load(f)

        data = {}
        for key in cls.RELEVANT_KEYS:
            if key in raw:
                data[key] = raw[key]

        return ConfigMetadata(data=data)


# ---------------------------------------------------------------------------
# OutputFormatter
# ---------------------------------------------------------------------------

class OutputFormatter:
    """Format and print the module hierarchy and config."""

    def __init__(self, show_line_numbers: bool = False):
        self.show_line_numbers = show_line_numbers

    def format_tree(self, tree: ModuleTree) -> str:
        """Format the module tree as an indented string."""
        lines = []
        lines.append(f"=== Module Hierarchy: {tree.class_name} ===")
        lines.append(tree.class_name)
        self._format_children(tree, lines, prefix="", is_last=True)
        return "\n".join(lines)

    def _format_children(
        self, node: ModuleTree, lines: List[str], prefix: str, is_last: bool
    ):
        children = node.children
        for i, child in enumerate(children):
            is_child_last = (i == len(children) - 1)
            connector = "|- "
            extension = "|   " if not is_child_last else "    "

            label = f"{child.attr_name}: {child.class_name}"
            if child.dispatch_map and len(child.dispatch_map) > 1:
                other_classes = [c for c in child.dispatch_map.values() if c != child.class_name]
                if other_classes:
                    label += f" | {other_classes[0]}"
            if child.multiplicity:
                label += f" [{child.multiplicity}]"
            if child.is_conditional and child.condition_text:
                cond_display = child.condition_text
                # Truncate long conditions
                if len(cond_display) > 60:
                    cond_display = cond_display[:57] + "..."
                label += f" (conditional: {cond_display})"

            if self.show_line_numbers and child.line_no:
                label += f"  [L{child.line_no}]"

            lines.append(f"{prefix}{connector}{label}")

            if child.children:
                self._format_children(
                    child, lines, prefix=prefix + extension, is_last=is_child_last
                )

    def format_config(self, config: ConfigMetadata) -> str:
        """Format config metadata as aligned key=value table."""
        if not config.data:
            return ""
        lines = ["\n=== Config Metadata ==="]
        max_key_len = max(len(k) for k in config.data)
        for key, value in config.data.items():
            val_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            lines.append(f"  {key:<{max_key_len}} = {val_str}")
        return "\n".join(lines)

    def format_profiler_tree(
        self,
        tree: ModuleTree,
        config: Optional[ConfigMetadata] = None,
        max_depth: int = 2,
        class_trees: Optional[Dict[str, ModuleTree]] = None,
    ) -> str:
        """Format as a PyTorch-profiler-style expanded tree with instance indices.

        Expands repeated layers (make_layers) into individual instances with _0, _1, ...
        and resolves conditional branches per-layer using config metadata.
        """
        # Store pre-built class trees for dispatch resolution
        self._class_trees = class_trees or {}

        lines = []
        lines.append(f"=== Module Tree (profiler-style): {tree.class_name} ===")
        lines.append(tree.class_name)

        config_data = config.data if config else {}
        self._profiler_children(tree, lines, prefix="", depth=0, max_depth=max_depth,
                                config_data=config_data, layer_id=None)
        return "\n".join(lines)

    def _profiler_children(
        self,
        node: ModuleTree,
        lines: List[str],
        prefix: str,
        depth: int,
        max_depth: int,
        config_data: Dict[str, Any],
        layer_id: Optional[int],
    ):
        """Recursively format children in profiler style."""
        children = node.children
        # Collect expanded children list (expanding repeated layers)
        expanded: List[Tuple[ModuleTree, Optional[int], bool]] = []  # (node, layer_id, is_expanded)

        for child in children:
            if child.multiplicity and "x N" in child.multiplicity:
                # Expand into individual instances
                num_layers = self._resolve_num_layers(child.multiplicity, config_data)
                if num_layers is not None:
                    if child.dispatch_map:
                        # Hybrid dispatch: resolve class per layer
                        for idx in range(num_layers):
                            resolved = self._resolve_dispatch_for_layer(
                                child, idx, config_data
                            )
                            expanded.append((resolved, idx, True))
                    else:
                        for idx in range(num_layers):
                            expanded.append((child, idx, True))
                else:
                    expanded.append((child, None, False))
            else:
                # Skip conditional variants — resolve which one applies
                expanded.append((child, layer_id, False))

        # De-duplicate conditionals: group by attr_name, pick the right branch
        expanded = self._resolve_conditionals(expanded, config_data)

        for i, (child, lid, is_expanded) in enumerate(expanded):
            is_last = (i == len(expanded) - 1)
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            extension = "    " if is_last else "\u2502   "

            # Build label
            class_name = child.class_name
            if is_expanded and lid is not None:
                label = f"{class_name}_{lid}"
            else:
                label = f"{class_name}_0"

            if self.show_line_numbers and child.line_no:
                label += f"  [L{child.line_no}]"

            lines.append(f"{prefix}{connector}{label}")

            # Recurse if under depth limit and has children
            if depth + 1 < max_depth and child.children:
                self._profiler_children(
                    child, lines, prefix=prefix + extension,
                    depth=depth + 1, max_depth=max_depth,
                    config_data=config_data,
                    layer_id=lid if lid is not None else layer_id,
                )

    def _resolve_num_layers(self, multiplicity: str, config_data: Dict[str, Any]) -> Optional[int]:
        """Resolve 'x N (config.num_hidden_layers)' to an actual integer."""
        # Try to find the config key in the multiplicity string
        if "num_hidden_layers" in multiplicity:
            val = config_data.get("num_hidden_layers")
            if isinstance(val, int):
                return val
        if "num_layers" in multiplicity:
            val = config_data.get("num_layers")
            if isinstance(val, int):
                return val
        # Try generic extraction
        import re
        m = re.search(r'config\.(\w+)', multiplicity)
        if m:
            val = config_data.get(m.group(1))
            if isinstance(val, int):
                return val
        return None

    def _resolve_dispatch_for_layer(
        self,
        child: ModuleTree,
        layer_idx: int,
        config_data: Dict[str, Any],
    ) -> ModuleTree:
        """Resolve a dispatch_map node to the correct class for a given layer index.

        Handles the Qwen3-Next hybrid pattern where full_attention_interval determines
        which layers use "attention" vs "linear_attention".
        """
        dispatch_map = child.dispatch_map
        if not dispatch_map:
            return child

        # Determine layer type key
        layer_type_key = self._get_layer_type_key(layer_idx, config_data, dispatch_map)

        resolved_class = dispatch_map.get(layer_type_key, child.class_name)

        # Create a new ModuleTree with the resolved class
        # We need to look up children from the class_map via the builder
        # For now, just swap the class_name — children will be resolved at recursion time
        resolved = ModuleTree(
            class_name=resolved_class,
            attr_name=child.attr_name,
            children=self._get_children_for_class(resolved_class, child),
            multiplicity="",
            is_conditional=False,
            condition_text="",
            line_no=child.line_no,
            dispatch_map=None,
        )
        return resolved

    def _get_children_for_class(
        self, class_name: str, original: ModuleTree
    ) -> List[ModuleTree]:
        """Get children for a resolved dispatch class.

        If the original node already has children from the hierarchy builder matching
        this class, reuse them. Otherwise return empty (leaf node in tree).
        """
        # If the hierarchy builder already resolved children for the original node's class,
        # and this is the same class, reuse them
        if original.class_name == class_name:
            return original.children
        # For dispatch-resolved nodes, we need the hierarchy builder to have built
        # child trees for all dispatch target classes. This is handled by building
        # a lookup in format_profiler_tree.
        if hasattr(self, '_class_trees') and class_name in self._class_trees:
            return self._class_trees[class_name].children
        return []

    def _get_layer_type_key(
        self, layer_idx: int, config_data: Dict[str, Any], dispatch_map: Dict[str, str]
    ) -> str:
        """Determine the dispatch key for a given layer index.

        Supports:
        - full_attention_interval: every Nth layer is "attention", rest are "linear_attention"
        - layers_block_type: explicit per-layer list from config
        """
        # Check explicit per-layer list
        block_types = config_data.get("layers_block_type")
        if isinstance(block_types, list) and layer_idx < len(block_types):
            return block_types[layer_idx]

        # Check full_attention_interval pattern (Qwen3-Next)
        interval = config_data.get("full_attention_interval")
        if isinstance(interval, int) and interval > 0:
            if (layer_idx + 1) % interval == 0:
                # This maps to "attention" key in ALL_DECODER_LAYER_TYPES
                if "attention" in dispatch_map:
                    return "attention"
                return "full_attention"
            else:
                if "linear_attention" in dispatch_map:
                    return "linear_attention"
                return "linear_attention"

        # Default: first key
        return next(iter(dispatch_map))

    def _resolve_conditionals(
        self,
        items: List[Tuple[ModuleTree, Optional[int], bool]],
        config_data: Dict[str, Any],
    ) -> List[Tuple[ModuleTree, Optional[int], bool]]:
        """For groups sharing the same attr_name, pick the right conditional branch.

        Uses config_data to resolve conditions like `self.is_layer_sparse` based on
        first_k_dense_replace and moe_layer_freq.
        """
        result = []
        i = 0
        while i < len(items):
            child, lid, is_exp = items[i]
            if not child.is_conditional:
                result.append((child, lid, is_exp))
                i += 1
                continue

            # Collect all conditionals with the same attr_name at this position
            group = [(child, lid, is_exp)]
            j = i + 1
            while j < len(items):
                next_child, next_lid, next_exp = items[j]
                if next_child.attr_name == child.attr_name and next_child.is_conditional:
                    group.append((next_child, next_lid, next_exp))
                    j += 1
                else:
                    break

            # Try to resolve which branch applies
            picked = self._pick_conditional(group, config_data)
            result.append(picked)
            i = j

        return result

    def _pick_conditional(
        self,
        group: List[Tuple[ModuleTree, Optional[int], bool]],
        config_data: Dict[str, Any],
    ) -> Tuple[ModuleTree, Optional[int], bool]:
        """Pick the correct conditional branch for a layer.

        For `self.is_layer_sparse` conditions, use first_k_dense_replace + moe_layer_freq.
        """
        if len(group) == 1:
            return group[0]

        child, lid, is_exp = group[0]

        # Try to resolve layer-sparsity condition
        if "is_layer_sparse" in child.condition_text and lid is not None:
            first_k = config_data.get("first_k_dense_replace", 0)
            moe_freq = config_data.get("moe_layer_freq", 1)
            is_sparse = lid >= first_k and moe_freq > 0 and lid % moe_freq == 0
            for g_child, g_lid, g_exp in group:
                cond = g_child.condition_text
                if is_sparse and "is_layer_sparse" in cond and "not" not in cond:
                    return (g_child, g_lid, g_exp)
                if not is_sparse and ("not" in cond or "else" in cond.lower()):
                    return (g_child, g_lid, g_exp)
            # Fallback: pick based on sparse = first branch
            return group[0] if is_sparse else group[-1]

        # For pp_group conditions, default to first rank (most common view)
        if "pp_group.is_first_rank" in child.condition_text:
            for g_child, g_lid, g_exp in group:
                if "not" not in g_child.condition_text:
                    return (g_child, g_lid, g_exp)

        if "pp_group.is_last_rank" in child.condition_text:
            for g_child, g_lid, g_exp in group:
                if "not" not in g_child.condition_text:
                    return (g_child, g_lid, g_exp)

        # Default: first branch
        return group[0]

    def format_class_list(self, classes: Dict[str, ClassInfo]) -> str:
        """Format a list of all classes."""
        lines = ["=== Classes Found ==="]
        for name, info in classes.items():
            bases_str = ", ".join(info.bases) if info.bases else ""
            n_assignments = len(info.assignments)
            line = f"  L{info.line_no:<5} {name}"
            if bases_str:
                line += f"({bases_str})"
            line += f"  [{n_assignments} module assignments]"
            if info.config_accesses:
                line += f"  config: {', '.join(info.config_accesses[:5])}"
                if len(info.config_accesses) > 5:
                    line += f" (+{len(info.config_accesses) - 5} more)"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-detect root class
# ---------------------------------------------------------------------------

def _auto_detect_root(classes: Dict[str, ClassInfo]) -> Optional[str]:
    """Auto-detect the root class (typically *ForCausalLM)."""
    # Prefer classes ending with ForCausalLM that are not subclasses of other local classes
    causal_lm = [
        name for name in classes
        if name.endswith("ForCausalLM")
    ]
    if causal_lm:
        # If there are subclass relationships, pick the base one
        # (e.g. DeepseekV2ForCausalLM over DeepseekV3ForCausalLM)
        for name in causal_lm:
            info = classes[name]
            # Check if any base is NOT another local ForCausalLM
            has_local_parent = any(
                base in classes and base.endswith("ForCausalLM")
                for base in info.bases
            )
            if not has_local_parent:
                return name
        return causal_lm[0]

    # Fallback: look for *ForConditionalGeneration, *Model, etc.
    for suffix in ["ForConditionalGeneration", "Model"]:
        matches = [name for name in classes if name.endswith(suffix)]
        if matches:
            return matches[0]

    return None


def _walk_tree(node: ModuleTree):
    """Yield all nodes in a ModuleTree."""
    yield node
    for child in node.children:
        yield from _walk_tree(child)


# ===========================================================================
# Architecture Diagram v2: Template-based rendering
# ===========================================================================

ARCH_COLOR_MAP = {
    "embedding":    "#1e5fa8",
    "attention":    "#b8860b",
    "norm":         "#4a5568",
    "ffn":          "#2d8659",
    "moe_router":   "#c0392b",
    "expert_pool":  "#6b21a8",
    "projection":   "#0e7490",
    "activation":   "#2d8659",
    "lm_head":      "#0e7490",
    "default":      "#374151",
}

_CATEGORY_PATTERNS = [
    ("embedding",   re.compile(r"Embed|Embedding|RotaryEmb|VocabParallel", re.IGNORECASE)),
    ("attention",   re.compile(
        r"Attention|Attn|MLA|SelfAttn|CrossAttn|MultiHead|FlashAttn", re.IGNORECASE)),
    ("moe_router",  re.compile(r"Router|Gate(?!Proj)|TopK|MoEGate", re.IGNORECASE)),
    ("expert_pool", re.compile(
        r"Expert|MoE(?!Gate)|FusedMoE|MixtralMoE|SharedExpert", re.IGNORECASE)),
    ("norm",        re.compile(r"Norm|RMSNorm|LayerNorm|BatchNorm|GroupNorm", re.IGNORECASE)),
    ("ffn",         re.compile(
        r"MLP|FFN|FeedForward|SiluAndMul|Activation|GatedMLP", re.IGNORECASE)),
    ("projection",  re.compile(
        r"Linear|Proj|GateProj|UpProj|DownProj|QKVParallel|RowParallel"
        r"|ColumnParallel|MergedColumnParallel|QKVParallelLinear", re.IGNORECASE)),
]


def classify_module_category(module_type: str) -> str:
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(module_type):
            return category
    return "default"


def get_category_color(category: str) -> str:
    return ARCH_COLOR_MAP.get(category, ARCH_COLOR_MAP["default"])


# ---------------------------------------------------------------------------
# Semantic block dataclass for template-based diagrams
# ---------------------------------------------------------------------------

@dataclass
class SemanticBlock:
    """A single visual block in the architecture diagram."""
    label: str
    category: str
    annotation: str = ""
    children: List["SemanticBlock"] = field(default_factory=list)
    layout: str = "vertical"       # "vertical" | "parallel" | "expert_grid"
    residual_after: bool = False
    width_fraction: float = 1.0    # 0.5 for half-width parallel blocks
    expert_count: int = 0
    expert_display: int = 6        # how many E1..En to show


@dataclass
class LayerGroupDef:
    """A group of repeated layers with a dashed border."""
    count: int
    label: str
    blocks: List[SemanticBlock] = field(default_factory=list)


@dataclass
class ArchTemplate:
    """Full architecture template for a model."""
    title: str
    subtitle: str = ""
    header_blocks: List[SemanticBlock] = field(default_factory=list)
    layer_groups: List[LayerGroupDef] = field(default_factory=list)
    footer_blocks: List[SemanticBlock] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TraceHierarchyExtractor (kept for fallback / raw tree)
# ---------------------------------------------------------------------------

@dataclass
class ArchNode:
    """Deduplicated architecture tree node (used by fallback renderer)."""
    module_type: str
    category: str
    color: str
    count: int = 1
    annotation: str = ""
    children: List["ArchNode"] = field(default_factory=list)


class TraceHierarchyExtractor:
    """Load a trace file and build a deduplicated ArchNode tree."""

    def __init__(self, trace_path: str):
        self.trace_path = trace_path

    def extract(self) -> List[ArchNode]:
        from trace_module_analyzer import ModuleTreeBuilder, TraceModuleAnalyzer

        data = TraceModuleAnalyzer._load_trace(self.trace_path)
        events = data.get("traceEvents", [])
        prefix = "nn.Module: "
        module_events = [
            e for e in events
            if e.get("cat") == "python_function"
            and str(e.get("name", "")).startswith(prefix)
            and e.get("dur") is not None
        ]
        builder = ModuleTreeBuilder()
        roots = builder.build_from_module_events(module_events)
        if not roots:
            return []
        best_root = max(roots, key=lambda r: r.end - r.ts)
        return [self._deduplicate(best_root)]

    def extract_raw_root(self):
        """Return the raw ModuleNode root (for template matching)."""
        from trace_module_analyzer import ModuleTreeBuilder, TraceModuleAnalyzer

        data = TraceModuleAnalyzer._load_trace(self.trace_path)
        events = data.get("traceEvents", [])
        prefix = "nn.Module: "
        module_events = [
            e for e in events
            if e.get("cat") == "python_function"
            and str(e.get("name", "")).startswith(prefix)
            and e.get("dur") is not None
        ]
        builder = ModuleTreeBuilder()
        roots = builder.build_from_module_events(module_events)
        if not roots:
            return None
        return max(roots, key=lambda r: r.end - r.ts)

    def _deduplicate(self, node) -> ArchNode:
        category = classify_module_category(node.module_type)
        arch = ArchNode(
            module_type=node.module_type,
            category=category,
            color=get_category_color(category),
        )
        if not node.children:
            return arch
        groups: List[Tuple[str, List]] = []
        for child in node.children:
            if groups and groups[-1][0] == child.module_type:
                groups[-1][1].append(child)
            else:
                groups.append((child.module_type, [child]))
        for module_type, members in groups:
            child_arch = self._deduplicate(members[0])
            child_arch.count = len(members)
            arch.children.append(child_arch)
        return arch


# ---------------------------------------------------------------------------
# Template: DeepSeek V2/V3 MoE
# ---------------------------------------------------------------------------

def _build_deepseek_template(cfg: Dict[str, Any], root_type: str,
                             raw_root=None) -> ArchTemplate:
    hs = cfg.get("hidden_size", 0)
    vs = cfg.get("vocab_size", 0)
    nh = cfg.get("num_attention_heads", 0)
    nkv = cfg.get("num_key_value_heads", nh)
    kv_lora = cfg.get("kv_lora_rank")
    inter = cfg.get("intermediate_size", 0)
    moe_inter = cfg.get("moe_intermediate_size", 0)
    n_routed = cfg.get("n_routed_experts") or cfg.get("num_experts", 0)
    n_shared = cfg.get("n_shared_experts", 0)
    top_k = cfg.get("num_experts_per_tok", 0)
    first_k = cfg.get("first_k_dense_replace", 0)
    total_layers = cfg.get("num_hidden_layers", 0)
    max_pos = cfg.get("max_position_embeddings", 0)

    # When config is missing/empty, derive what we can from the trace tree
    _trace_child_types = set()
    if raw_root and hasattr(raw_root, "children"):
        for child in raw_root.children:
            _trace_child_types.add(child.module_type)
            if not total_layers and "DecoderLayer" in child.module_type:
                total_layers = sum(
                    1 for c in raw_root.children
                    if c.module_type == child.module_type
                )
            for gc in getattr(child, "children", []):
                _trace_child_types.add(gc.module_type)

    is_mla = kv_lora is not None and kv_lora > 0
    if not is_mla and any("MLA" in t for t in _trace_child_types):
        is_mla = True
    has_moe_in_trace = any(
        "MoE" in t or "FusedMoE" in t or "Expert" in t
        for t in _trace_child_types
    )

    attn_label = "Multi-head Latent Attention" if is_mla else "Multi-head Attention"
    attn_ann = f"{nh} heads" if nh else ""
    if nkv and nkv != nh:
        attn_ann += f" / {nkv} KV heads"

    model_name = "DeepSeek"
    archs = cfg.get("architectures", [])
    if archs:
        name = archs[0].replace("ForCausalLM", "")
        if "V3" in name or "v3" in root_type:
            model_name = "DeepSeek-V3"
        elif "V2" in name or "v2" in root_type.lower():
            model_name = "DeepSeek-V2"
    elif "v3" in root_type.lower():
        model_name = "DeepSeek-V3"
    elif "v2" in root_type.lower():
        model_name = "DeepSeek-V2"

    # Compute parameter estimate
    params_b = 0
    if hs and total_layers:
        attn_params = 4 * hs * hs
        ffn_params = 3 * hs * inter if inter else 0
        moe_ffn_params = n_routed * 3 * hs * moe_inter if n_routed and moe_inter else 0
        shared_ffn_params = n_shared * 3 * hs * moe_inter if n_shared and moe_inter else 0
        dense_layer_params = attn_params + ffn_params
        moe_layer_params = attn_params + moe_ffn_params + shared_ffn_params
        total_params = first_k * dense_layer_params + (total_layers - first_k) * moe_layer_params
        total_params += vs * hs  # embedding
        params_b = total_params / 1e9

    active_b = 0
    if params_b and n_routed and top_k:
        active_per_expert = 3 * hs * moe_inter if moe_inter else 0
        moe_active_ffn = top_k * active_per_expert + n_shared * active_per_expert
        attn_params = 4 * hs * hs
        dense_ffn = 3 * hs * inter if inter else 0
        active_per_layer = attn_params + moe_active_ffn
        dense_per_layer = attn_params + dense_ffn
        total_active = first_k * dense_per_layer + (total_layers - first_k) * active_per_layer
        total_active += vs * hs
        active_b = total_active / 1e9

    context_k = max_pos // 1000 if max_pos else 0
    subtitle_parts = []
    if params_b > 0:
        subtitle_parts.append(f"{params_b:.0f}B total")
    if active_b > 0:
        subtitle_parts.append(f"{active_b:.0f}B active")
    if context_k:
        subtitle_parts.append(f"{context_k}K context")
    subtitle = " · ".join(subtitle_parts)

    embed_ann = ""
    if hs:
        embed_ann += f"d = {hs:,}"
    if vs:
        embed_ann += f" \u00b7 vocab = {vs:,}"

    # Dense layer blocks
    dense_blocks = [
        SemanticBlock(label="RMSNorm", category="norm"),
        SemanticBlock(label=attn_label, category="attention",
                      annotation=attn_ann, residual_after=True),
        SemanticBlock(label="RMSNorm", category="norm"),
        SemanticBlock(label="Dense FFN", category="ffn",
                      annotation=f"intermediate = {inter:,}" if inter else "",
                      residual_after=True),
    ]

    # MoE layer blocks
    expert_total = n_routed + n_shared
    router_ann = f"Top-{top_k} of {n_routed} routed" if top_k and n_routed else ""
    if n_shared:
        router_ann += f" + {n_shared} shared"

    expert_ffn = SemanticBlock(
        label="EXPERT FFN (SwiGLU)", category="expert_pool", layout="vertical",
        children=[
            SemanticBlock(label="", category="projection", layout="parallel", children=[
                SemanticBlock(label="Gate Projection", category="projection",
                              annotation=f"\u2192 {moe_inter:,}" if moe_inter else "",
                              width_fraction=0.48),
                SemanticBlock(label="Up Projection", category="projection",
                              annotation=f"\u2192 {moe_inter:,}" if moe_inter else "",
                              width_fraction=0.48),
            ]),
            SemanticBlock(label="SiLU Activation", category="activation",
                          annotation="Applied to gate", width_fraction=0.48),
            SemanticBlock(label="Down Projection", category="projection",
                          annotation=f"\u2192 {hs:,}" if hs else "",
                          residual_after=True),
        ],
    )

    expert_grid = SemanticBlock(
        label="", category="expert_pool", layout="expert_grid",
        expert_count=expert_total, expert_display=6,
    )

    moe_blocks = [
        SemanticBlock(label="RMSNorm", category="norm"),
        SemanticBlock(label=attn_label, category="attention",
                      annotation=attn_ann, residual_after=True),
        SemanticBlock(label="RMSNorm", category="norm"),
        SemanticBlock(label="MoE Router", category="moe_router",
                      annotation=router_ann),
        expert_grid,
        expert_ffn,
    ]

    layer_groups = []
    if first_k > 0:
        layer_groups.append(LayerGroupDef(
            count=first_k, label=f"\u00d7{first_k} Dense layers",
            blocks=dense_blocks))
    moe_count = total_layers - first_k if total_layers else 0
    if moe_count > 0 and (n_routed or has_moe_in_trace):
        layer_groups.append(LayerGroupDef(
            count=moe_count, label=f"\u00d7{moe_count} MoE layers",
            blocks=moe_blocks))
    elif moe_count > 0:
        layer_groups.append(LayerGroupDef(
            count=moe_count, label=f"\u00d7{moe_count} layers",
            blocks=dense_blocks))
    if not layer_groups and total_layers:
        if has_moe_in_trace:
            layer_groups.append(LayerGroupDef(
                count=total_layers, label=f"\u00d7{total_layers} layers",
                blocks=moe_blocks))
        else:
            layer_groups.append(LayerGroupDef(
                count=total_layers, label=f"\u00d7{total_layers} layers",
                blocks=dense_blocks))

    # Metadata footer
    meta = {}
    meta["Type"] = "MoE" if (n_routed or has_moe_in_trace) else "Dense"
    if first_k and moe_count:
        meta["Layers"] = f"{first_k}D+{moe_count}M"
    elif total_layers:
        meta["Layers"] = str(total_layers)
    meta["Attention"] = "MLA" if is_mla else "MHA"
    if context_k:
        meta["Context"] = f"{context_k}K"
    if n_routed:
        meta["Experts"] = f"{top_k}/{expert_total}"
    elif has_moe_in_trace:
        meta["Experts"] = "MoE"

    return ArchTemplate(
        title=model_name,
        subtitle=subtitle,
        header_blocks=[
            SemanticBlock(label="Token Embedding", category="embedding",
                          annotation=embed_ann),
        ],
        layer_groups=layer_groups,
        footer_blocks=[
            SemanticBlock(label="RMSNorm", category="norm"),
            SemanticBlock(label="Output Head (LM Head)", category="lm_head",
                          annotation=f"vocab = {vs:,}" if vs else ""),
        ],
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Template: Wan 2.2 Diffusion Transformer
# ---------------------------------------------------------------------------

def _build_wan_template(cfg: Dict[str, Any], root_type: str,
                        raw_root=None) -> ArchTemplate:
    layer_blocks = [
        SemanticBlock(label="Self-Attention", category="attention",
                      annotation="QKV + output projection"),
        SemanticBlock(label="Cross-Attention", category="attention",
                      annotation="Text-conditioned"),
        SemanticBlock(label="MLP", category="ffn",
                      annotation="Feed-forward"),
    ]

    num_blocks = cfg.get("num_hidden_layers", 0)
    if not num_blocks and raw_root and hasattr(raw_root, "children"):
        num_blocks = sum(
            1 for c in raw_root.children
            if "TransformerBlock" in c.module_type or "Block" in c.module_type
        )
    if not num_blocks:
        num_blocks = 40

    return ArchTemplate(
        title="Wan 2.2 Transformer",
        subtitle="Video Diffusion Model",
        header_blocks=[
            SemanticBlock(label="Patch Embedding", category="embedding",
                          annotation="3D patch tokenization"),
            SemanticBlock(label="Time + Text + Image Embedding", category="embedding",
                          annotation="Conditioning signals"),
        ],
        layer_groups=[
            LayerGroupDef(
                count=num_blocks,
                label=f"\u00d7{num_blocks} WanTransformerBlock",
                blocks=layer_blocks,
            ),
        ],
        footer_blocks=[
            SemanticBlock(label="LayerNorm", category="norm"),
            SemanticBlock(label="Output Projection", category="projection"),
        ],
        metadata={
            "Type": "DiT",
            "Blocks": str(num_blocks),
            "Conditioning": "Text + Time",
        },
    )


# ---------------------------------------------------------------------------
# Template matcher
# ---------------------------------------------------------------------------

def match_template(root_type: str, cfg: Optional[Dict[str, Any]] = None,
                   raw_root=None) -> Optional[ArchTemplate]:
    """Auto-detect model type from trace root and build the appropriate template."""
    rt = root_type.lower()
    if cfg is None:
        cfg = {}

    if "deepseek" in rt:
        return _build_deepseek_template(cfg, root_type, raw_root=raw_root)
    if "wantransformer" in rt:
        return _build_wan_template(cfg, root_type, raw_root=raw_root)
    return None


# ---------------------------------------------------------------------------
# ArchDiagramRenderer v2
# ---------------------------------------------------------------------------

class ArchDiagramRenderer:
    """Render an ArchTemplate as a dark-themed matplotlib block diagram."""

    BG_COLOR = "#1a1a2e"
    TEXT_COLOR = "#e0e0e0"
    ARROW_COLOR = "#666666"
    GROUP_BORDER_COLOR = "#555555"
    RESIDUAL_COLOR = "#aaaaaa"
    RESIDUAL_LINE_COLOR = "#555555"

    BLOCK_WIDTH = 4.2
    BLOCK_HEIGHT = 0.55
    BLOCK_GAP = 0.22
    GROUP_PAD_X = 0.35
    GROUP_PAD_TOP = 0.45
    GROUP_PAD_BOTTOM = 0.20
    ANNOTATION_FONT_SIZE = 6.0
    LABEL_FONT_SIZE = 7.5
    GROUP_LABEL_FONT_SIZE = 7.0
    TITLE_FONT_SIZE = 11
    SUBTITLE_FONT_SIZE = 7
    FOOTER_FONT_SIZE = 7.5

    def __init__(self):
        self._y = 0.0

    def render_template_to_png(self, template: ArchTemplate, output_path: str,
                               dpi: int = 200):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        self._y = 0.0
        items = self._layout_template(template)
        total_height = abs(self._y) + 2.0
        fig_width = self.BLOCK_WIDTH + 2 * self.GROUP_PAD_X + 1.6
        fig, ax = plt.subplots(figsize=(fig_width, max(total_height, 5)))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.BG_COLOR)
        ax.set_xlim(-fig_width / 2, fig_width / 2)
        ax.set_ylim(self._y - 0.8, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

        # Title + subtitle
        ax.text(0, 0.85, template.title, ha="center", va="center",
                fontsize=self.TITLE_FONT_SIZE, fontweight="bold",
                color=self.TEXT_COLOR)
        if template.subtitle:
            ax.text(0, 0.55, template.subtitle, ha="center", va="center",
                    fontsize=self.SUBTITLE_FONT_SIZE, color="#999999")

        for item in items:
            self._dispatch_draw(ax, item, mpatches)

        plt.tight_layout(pad=0.2)
        fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

    # --- Layout engine ---

    def _layout_template(self, template: ArchTemplate) -> List[Dict]:
        items: List[Dict] = []
        self._y = 0.2

        for block in template.header_blocks:
            self._layout_block(block, items)
            self._emit_arrow(items)

        for group in template.layer_groups:
            self._layout_group(group, items)
            self._emit_arrow(items)

        for i, block in enumerate(template.footer_blocks):
            self._layout_block(block, items)
            if i < len(template.footer_blocks) - 1:
                self._emit_arrow(items)

        if template.metadata:
            self._y -= self.BLOCK_GAP * 0.5
            items.append({
                "kind": "metadata_footer",
                "y": self._y - 0.4,
                "data": template.metadata,
            })
            self._y -= 0.6

        return items

    def _layout_block(self, block: SemanticBlock, items: List[Dict],
                      width_override: float = 0.0, x_offset: float = 0.0):
        bw = width_override or (self.BLOCK_WIDTH * block.width_fraction)
        bh = self.BLOCK_HEIGHT
        if block.annotation:
            bh += 0.12 * (block.annotation.count("\n") + 1)

        if block.layout == "parallel" and block.children:
            self._layout_parallel(block, items)
            return

        if block.layout == "expert_grid":
            if block.expert_count > 0:
                self._layout_expert_grid(block, items)
            return

        # Blocks with vertical children render as a section header, not a filled box
        is_section_header = block.children and block.layout == "vertical"

        if is_section_header:
            if block.label:
                items.append({
                    "kind": "section_label",
                    "x": x_offset,
                    "y": self._y - 0.15,
                    "label": block.label,
                })
                self._y -= 0.30
            for i, child in enumerate(block.children):
                if child.layout == "parallel":
                    self._layout_parallel(child, items)
                else:
                    self._layout_block(child, items)
                if i < len(block.children) - 1:
                    self._emit_arrow(items)
            return

        block_y = self._y - bh
        items.append({
            "kind": "block",
            "x": x_offset - bw / 2, "y": block_y,
            "w": bw, "h": bh,
            "label": block.label,
            "color": get_category_color(block.category),
            "annotation": block.annotation,
        })
        self._y = block_y - self.BLOCK_GAP

        if block.residual_after:
            res_y = block_y + bh / 2
            items.append({
                "kind": "residual",
                "x": bw / 2 + 0.25 + x_offset,
                "y": res_y,
                "y_top": block_y + bh + self.BLOCK_GAP + self.BLOCK_HEIGHT,
                "y_bottom": block_y,
            })

    def _layout_parallel(self, block: SemanticBlock, items: List[Dict]):
        """Layout children side-by-side."""
        n = len(block.children)
        if n == 0:
            return
        gap = 0.15
        total_w = self.BLOCK_WIDTH
        child_w = (total_w - gap * (n - 1)) / n

        saved_y = self._y
        max_drop = 0.0

        for i, child in enumerate(block.children):
            cx = -total_w / 2 + child_w / 2 + i * (child_w + gap)
            bh = self.BLOCK_HEIGHT
            if child.annotation:
                bh += 0.12
            block_y = saved_y - bh
            items.append({
                "kind": "block",
                "x": cx - child_w / 2, "y": block_y,
                "w": child_w, "h": bh,
                "label": child.label,
                "color": get_category_color(child.category),
                "annotation": child.annotation,
            })
            drop = bh + self.BLOCK_GAP
            if drop > max_drop:
                max_drop = drop

        self._y = saved_y - max_drop

    def _layout_expert_grid(self, block: SemanticBlock, items: List[Dict]):
        """Layout expert grid: E1, E2, ... En, ..., +total."""
        items.append({
            "kind": "expert_grid",
            "y": self._y,
            "expert_count": block.expert_count,
            "expert_display": block.expert_display,
        })
        self._y -= 0.55

    def _layout_group(self, group: LayerGroupDef, items: List[Dict]):
        group_start_y = self._y
        items.append({"kind": "group_start", "y": self._y, "label": group.label})
        self._y -= self.GROUP_PAD_TOP

        for i, block in enumerate(group.blocks):
            self._layout_block(block, items)
            if i < len(group.blocks) - 1:
                self._emit_arrow(items)

        self._y -= self.GROUP_PAD_BOTTOM
        items.append({
            "kind": "group_end",
            "y_start": group_start_y,
            "y_end": self._y,
            "label": group.label,
            "cx": 0.0,
            "width": self.BLOCK_WIDTH + 2 * self.GROUP_PAD_X,
        })
        self._y -= self.BLOCK_GAP * 0.3

    def _emit_arrow(self, items: List[Dict]):
        items.append({
            "kind": "arrow",
            "x": 0.0,
            "y_start": self._y + self.BLOCK_GAP,
            "y_end": self._y + 0.04,
        })

    # --- Drawing primitives ---

    def _dispatch_draw(self, ax, item: Dict, mp):
        kind = item["kind"]
        if kind == "block":
            self._draw_block(ax, item, mp)
        elif kind == "arrow":
            self._draw_arrow(ax, item, mp)
        elif kind == "group_end":
            self._draw_group_border(ax, item, mp)
        elif kind == "residual":
            self._draw_residual(ax, item, mp)
        elif kind == "expert_grid":
            self._draw_expert_grid(ax, item, mp)
        elif kind == "metadata_footer":
            self._draw_metadata_footer(ax, item)
        elif kind == "section_label":
            self._draw_section_label(ax, item)

    def _draw_block(self, ax, item: Dict, mp):
        x, y, w, h = item["x"], item["y"], item["w"], item["h"]
        color = item["color"]
        label = item["label"]
        annotation = item.get("annotation", "")

        box = mp.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=mp.BoxStyle.Round(pad=0.04, rounding_size=0.08),
            facecolor=color, edgecolor="#ffffff20", linewidth=0.7,
        )
        ax.add_patch(box)

        if label:
            text_y = y + h * 0.62 if annotation else y + h / 2
            ax.text(x + w / 2, text_y, label,
                    ha="center", va="center",
                    fontsize=self.LABEL_FONT_SIZE, fontweight="bold",
                    color=self.TEXT_COLOR)

        if annotation:
            ax.text(x + w / 2, y + h * 0.25, annotation,
                    ha="center", va="center",
                    fontsize=self.ANNOTATION_FONT_SIZE,
                    color="#bbbbbb", style="italic")

    def _draw_arrow(self, ax, item: Dict, mp):
        arrow = mp.FancyArrowPatch(
            (item["x"], item["y_start"]), (item["x"], item["y_end"]),
            arrowstyle="-|>", mutation_scale=7,
            color=self.ARROW_COLOR, linewidth=0.8,
        )
        ax.add_patch(arrow)

    def _draw_group_border(self, ax, item: Dict, mp):
        cx, w = item["cx"], item["width"]
        y_start, y_end = item["y_start"], item["y_end"]
        label = item["label"]

        rect = mp.FancyBboxPatch(
            (cx - w / 2, y_end), w, y_start - y_end,
            boxstyle=mp.BoxStyle.Round(pad=0.04, rounding_size=0.12),
            facecolor="none", edgecolor=self.GROUP_BORDER_COLOR,
            linewidth=1.0, linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(cx + w / 2 - 0.05, y_start - 0.05, label,
                ha="right", va="top",
                fontsize=self.GROUP_LABEL_FONT_SIZE,
                color="#888888", fontstyle="italic")

    def _draw_residual(self, ax, item: Dict, mp):
        x, y = item["x"], item["y"]
        circle = mp.Circle(
            (x, y), 0.10,
            facecolor=self.BG_COLOR, edgecolor=self.RESIDUAL_COLOR,
            linewidth=0.8,
        )
        ax.add_patch(circle)
        ax.text(x, y, "+", ha="center", va="center",
                fontsize=7, fontweight="bold", color=self.RESIDUAL_COLOR)

        y_top = item.get("y_top", y + 0.5)
        y_bottom = item.get("y_bottom", y - 0.1)
        bypass_x = x + 0.15
        ax.plot([x, bypass_x, bypass_x, x + 0.10],
                [y_top - 0.05, y_top - 0.05, y, y],
                color="#777777", linewidth=1.0,
                linestyle=":", clip_on=False)

    def _draw_expert_grid(self, ax, item: Dict, mp):
        y = item["y"]
        n_display = item["expert_display"]
        total = item["expert_count"]

        grid_w = self.BLOCK_WIDTH * 0.85
        cell_size = 0.28
        gap = 0.06
        cells_w = n_display * cell_size + (n_display - 1) * gap
        start_x = -cells_w / 2

        border_pad = 0.12
        border = mp.FancyBboxPatch(
            (-grid_w / 2, y - cell_size - border_pad),
            grid_w, cell_size + 2 * border_pad,
            boxstyle=mp.BoxStyle.Round(pad=0.03, rounding_size=0.06),
            facecolor="#6b21a815", edgecolor="#6b21a8",
            linewidth=0.8, linestyle="--",
        )
        ax.add_patch(border)

        for i in range(n_display):
            cx = start_x + i * (cell_size + gap)
            cell = mp.FancyBboxPatch(
                (cx, y - cell_size), cell_size, cell_size,
                boxstyle=mp.BoxStyle.Round(pad=0.02, rounding_size=0.04),
                facecolor="#6b21a8", edgecolor="#ffffff30", linewidth=0.5,
            )
            ax.add_patch(cell)
            ax.text(cx + cell_size / 2, y - cell_size / 2, f"E{i + 1}",
                    ha="center", va="center",
                    fontsize=5.5, fontweight="bold", color=self.TEXT_COLOR)

        dots_x = start_x + n_display * (cell_size + gap)
        ax.text(dots_x, y - cell_size / 2, "\u2026",
                ha="center", va="center",
                fontsize=8, color="#999999")

        plus_x = dots_x + gap + cell_size * 0.6
        plus_cell = mp.FancyBboxPatch(
            (plus_x - cell_size * 0.4, y - cell_size),
            cell_size * 0.8, cell_size,
            boxstyle=mp.BoxStyle.Round(pad=0.02, rounding_size=0.04),
            facecolor="#4a5568", edgecolor="#ffffff30", linewidth=0.5,
        )
        ax.add_patch(plus_cell)
        ax.text(plus_x, y - cell_size / 2, f"+{total}",
                ha="center", va="center",
                fontsize=5, fontweight="bold", color=self.TEXT_COLOR)

    def _draw_metadata_footer(self, ax, item: Dict):
        y = item["y"]
        data = item["data"]
        n = len(data)
        if n == 0:
            return
        total_w = self.BLOCK_WIDTH + 0.4
        cell_w = total_w / n
        start_x = -total_w / 2

        for i, (key, val) in enumerate(data.items()):
            cx = start_x + i * cell_w + cell_w / 2
            ax.text(cx, y + 0.15, key, ha="center", va="center",
                    fontsize=5.5, color="#888888")
            ax.text(cx, y - 0.05, val, ha="center", va="center",
                    fontsize=self.FOOTER_FONT_SIZE, fontweight="bold",
                    color=self.TEXT_COLOR)

    def _draw_section_label(self, ax, item: Dict):
        ax.text(item["x"], item["y"], item["label"],
                ha="center", va="center",
                fontsize=self.LABEL_FONT_SIZE - 0.5,
                fontweight="bold", color="#999999",
                fontstyle="italic")

    # --- Fallback: render raw ArchNode tree (for unknown models) ---

    def render_fallback_to_png(self, roots: List[ArchNode], output_path: str,
                               title: str = "Model Architecture", dpi: int = 200):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        self._y = 0.2
        items: List[Dict] = []
        for root in roots:
            self._layout_archnode(root, items, depth=0)

        total_height = abs(self._y) + 1.5
        fig_width = self.BLOCK_WIDTH + 2 * self.GROUP_PAD_X + 1.6
        fig, ax = plt.subplots(figsize=(fig_width, max(total_height, 4)))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.BG_COLOR)
        ax.set_xlim(-fig_width / 2, fig_width / 2)
        ax.set_ylim(self._y - 0.5, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0, 0.7, title, ha="center", va="center",
                fontsize=self.TITLE_FONT_SIZE, fontweight="bold",
                color=self.TEXT_COLOR)

        for item in items:
            self._dispatch_draw(ax, item, mpatches)

        plt.tight_layout(pad=0.2)
        fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

    def _layout_archnode(self, node: ArchNode, items: List[Dict], depth: int):
        bw = self.BLOCK_WIDTH
        bh = self.BLOCK_HEIGHT
        if node.annotation:
            bh += 0.12

        is_group = node.count > 1 and node.children
        if is_group:
            label = f"\u00d7{node.count} {node.module_type}"
            group_start_y = self._y
            items.append({"kind": "group_start", "y": self._y, "label": label})
            self._y -= self.GROUP_PAD_TOP
            for i, child in enumerate(node.children):
                self._layout_archnode(child, items, depth + 1)
                if i < len(node.children) - 1:
                    self._emit_arrow(items)
            self._y -= self.GROUP_PAD_BOTTOM
            items.append({
                "kind": "group_end",
                "y_start": group_start_y, "y_end": self._y,
                "label": label, "cx": 0.0,
                "width": bw + 2 * self.GROUP_PAD_X,
            })
            self._y -= self.BLOCK_GAP
        else:
            display = node.module_type
            if node.count > 1:
                display = f"{node.module_type} (\u00d7{node.count})"
            block_y = self._y - bh
            items.append({
                "kind": "block",
                "x": -bw / 2, "y": block_y,
                "w": bw, "h": bh,
                "label": display,
                "color": node.color,
                "annotation": node.annotation,
            })
            self._y = block_y - self.BLOCK_GAP
            if node.children:
                self._emit_arrow(items)
                for i, child in enumerate(node.children):
                    self._layout_archnode(child, items, depth + 1)
                    if i < len(node.children) - 1:
                        self._emit_arrow(items)


# ---------------------------------------------------------------------------
# arch_diagram_main — pipeline entry point
# ---------------------------------------------------------------------------

def template_to_text(template: ArchTemplate) -> str:
    """Render an ArchTemplate as a human-readable text diagram."""
    W = 62
    IW = W - 4

    lines: List[str] = []

    def _hbar(char="═"):
        return char * W

    def _box(label: str, annotation: str = "", prefix: str = "  "):
        iw = W - len(prefix) - 2
        lines.append(f"{prefix}┌{'─' * iw}┐")
        lines.append(f"{prefix}│ {label:<{iw - 2}} │")
        if annotation:
            lines.append(f"{prefix}│   {annotation:<{iw - 4}} │")
        lines.append(f"{prefix}└{'─' * iw}┘")

    def _arrow(prefix: str = "  ", mid: int = IW // 2 + 2):
        pad = " " * (mid - 1)
        lines.append(f"{prefix}{pad}│")
        lines.append(f"{prefix}{pad}▼")

    def _residual(prefix: str = "  ", mid: int = IW // 2 + 2):
        pad = " " * (mid - 1)
        lines.append(f"{prefix}{pad}│")
        lines.append(f"{prefix}{pad[:-1]}⊕")
        lines.append(f"{prefix}{pad}│")

    def _gbox(label: str, annotation: str = ""):
        """Block inside a group (wrapped with ║ ... ║)."""
        iw = IW - 4
        lines.append(f"  ║  ┌{'─' * iw}┐║")
        lines.append(f"  ║  │ {label:<{iw - 2}} │║")
        if annotation:
            lines.append(f"  ║  │   {annotation:<{iw - 4}} │║")
        lines.append(f"  ║  └{'─' * iw}┘║")

    def _garrow():
        mid = IW // 2
        pad = " " * (mid - 1)
        lines.append(f"  ║{pad}│{' ' * (IW - mid)}║")
        lines.append(f"  ║{pad}▼{' ' * (IW - mid)}║")

    def _gresidual():
        mid = IW // 2
        pad = " " * (mid - 1)
        lines.append(f"  ║{pad}│{' ' * (IW - mid)}║")
        lines.append(f"  ║{' ' * (mid - 2)}⊕{' ' * (IW - mid + 1)}║")

    def _gblank():
        lines.append(f"  ║{' ' * IW}║")

    def _expert_grid_g(block: SemanticBlock):
        if not block.expert_count:
            return
        n_show = min(block.expert_display, 6)
        cells = [f"  {f'E{i+1}':^4s}  " for i in range(n_show)]
        cells.append(f" {'···':^3s} ")
        cells.append(f" {'×' + str(block.expert_count):^5s} ")
        inner = "│".join(cells)
        top_sep = "┬".join("─" * len(c) for c in cells)
        bot_sep = "┴".join("─" * len(c) for c in cells)
        lines.append(f"  ║  ┌{top_sep}┐ ║")
        lines.append(f"  ║  │{inner}│ ║")
        lines.append(f"  ║  └{bot_sep}┘ ║")

    def _parallel_g(block: SemanticBlock):
        children = block.children
        n = len(children)
        col_w = 26
        row_top = []
        row_lbl = []
        row_ann = []
        row_bot = []
        for c in children:
            row_top.append(f"┌{'─' * (col_w - 2)}┐")
            row_lbl.append(f"│ {c.label:<{col_w - 4}} │")
            if c.annotation:
                row_ann.append(f"│   {c.annotation:<{col_w - 6}} │")
            else:
                row_ann.append(None)
            row_bot.append(f"└{'─' * (col_w - 2)}┘")
        lines.append(f"  ║  {' '.join(row_top)}   ║")
        lines.append(f"  ║  {' '.join(row_lbl)}   ║")
        if any(a is not None for a in row_ann):
            ann_line = " ".join(a if a else f"│{' ' * (col_w - 2)}│" for a in row_ann)
            lines.append(f"  ║  {ann_line}   ║")
        lines.append(f"  ║  {' '.join(row_bot)}   ║")

    def _swiglu_flow_g(block: SemanticBlock):
        """Render the SwiGLU expert FFN with proper parallel-merge flow."""
        children = block.children
        if len(children) < 3:
            _glabel(block.label)
            for i, c in enumerate(children):
                _dump_group_block(c)
                if i < len(children) - 1:
                    _garrow()
            return

        parallel_block = children[0]
        activation_block = children[1]
        down_block = children[2]

        gate_child = parallel_block.children[0] if parallel_block.children else None
        up_child = parallel_block.children[1] if len(parallel_block.children) > 1 else None

        col_w = 26
        g_label = gate_child.label if gate_child else "Gate"
        g_ann = gate_child.annotation if gate_child else ""
        u_label = up_child.label if up_child else "Up"
        u_ann = up_child.annotation if up_child else ""

        _glabel(block.label)
        _garrow()

        lines.append(f"  ║  ┌{'─' * (col_w - 2)}┐ ┌{'─' * (col_w - 2)}┐   ║")
        lines.append(f"  ║  │ {g_label:<{col_w - 4}} │ │ {u_label:<{col_w - 4}} │   ║")
        if g_ann or u_ann:
            ga = f"│   {g_ann:<{col_w - 6}} │" if g_ann else f"│{' ' * (col_w - 2)}│"
            ua = f"│   {u_ann:<{col_w - 6}} │" if u_ann else f"│{' ' * (col_w - 2)}│"
            lines.append(f"  ║  {ga} {ua}   ║")
        lines.append(f"  ║  └{'─' * (col_w - 2)}┘ └{'─' * (col_w - 2)}┘   ║")

        lines.append(f"  ║           │                          │                   ║")
        lines.append(f"  ║           ▼                          │                   ║")

        a_label = activation_block.label
        a_ann = activation_block.annotation or ""
        lines.append(f"  ║  ┌{'─' * (col_w - 2)}┐          │                   ║")
        lines.append(f"  ║  │ {a_label:<{col_w - 4}} │          │                   ║")
        if a_ann:
            lines.append(f"  ║  │   {a_ann:<{col_w - 6}} │          │                   ║")
        lines.append(f"  ║  └{'─' * (col_w - 2)}┘          │                   ║")

        lines.append(f"  ║           │                          │                   ║")
        lines.append(f"  ║           └──────────⊗───────────────┘                  ║")
        lines.append(f"  ║                      │                                   ║")
        _garrow()

        _gbox_inner(down_block.label, down_block.annotation)
        if down_block.residual_after:
            _gresidual()

    def _gbox_inner(label: str, annotation: str = ""):
        iw = IW - 4
        lines.append(f"  ║  ┌{'─' * iw}┐║")
        lines.append(f"  ║  │ {label:<{iw - 2}} │║")
        if annotation:
            lines.append(f"  ║  │   {annotation:<{iw - 4}} │║")
        lines.append(f"  ║  └{'─' * iw}┘║")

    def _glabel(text: str):
        mid = IW // 2
        label_str = f"── {text} ──"
        pad_l = mid - len(label_str) // 2
        pad_r = IW - pad_l - len(label_str)
        lines.append(f"  ║{' ' * pad_l}{label_str}{' ' * pad_r}║")

    def _dump_group_block(block: SemanticBlock):
        if block.layout == "expert_grid":
            _expert_grid_g(block)
        elif block.layout == "parallel" and block.children:
            _parallel_g(block)
        elif block.children and block.layout == "vertical":
            _swiglu_flow_g(block)
        else:
            _gbox(block.label, block.annotation)
            if block.residual_after:
                _gresidual()

    # --- Title ---
    lines.append(_hbar())
    lines.append(f"  {template.title}")
    if template.subtitle:
        lines.append(f"  {template.subtitle}")
    lines.append(_hbar())
    lines.append("")

    # --- Header blocks ---
    for block in template.header_blocks:
        _box(block.label, block.annotation)
        _arrow()

    # --- Layer groups ---
    for group in template.layer_groups:
        lines.append(f"  ╔{'═' * IW}╗")
        lines.append(f"  ║ {group.label:<{IW - 2}} ║")
        lines.append(f"  ╟{'─' * IW}╢")
        _gblank()
        for i, block in enumerate(group.blocks):
            _dump_group_block(block)
            if i < len(group.blocks) - 1:
                if not block.residual_after:
                    _garrow()
                else:
                    _garrow()
        _gblank()
        lines.append(f"  ╚{'═' * IW}╝")
        _arrow()

    # --- Footer blocks ---
    for i, block in enumerate(template.footer_blocks):
        _box(block.label, block.annotation)
        if i < len(template.footer_blocks) - 1:
            _arrow()

    # --- Metadata ---
    if template.metadata:
        lines.append("")
        lines.append(_hbar("─"))
        meta_parts = [f"{k}: {v}" for k, v in template.metadata.items()]
        lines.append("  " + " │ ".join(meta_parts))
        lines.append(_hbar("─"))

    return "\n".join(lines)


def template_to_text_simplified(template: ArchTemplate) -> str:
    """Render an ArchTemplate as a condensed text diagram.

    Layer groups are collapsed into single summary boxes instead of
    expanding all internal blocks (RMSNorm, Attention, MoE, etc.).
    """
    W = 62
    IW = W - 4

    lines: List[str] = []

    def _hbar(char="═"):
        return char * W

    def _box(label: str, annotation: str = "", prefix: str = "  "):
        iw = W - len(prefix) - 2
        max_content = iw - 2
        lines.append(f"{prefix}┌{'─' * iw}┐")
        lines.append(f"{prefix}│ {label[:max_content]:<{max_content}} │")
        if annotation:
            ann_text = f"  {annotation}" if not annotation.startswith(" ") else annotation
            lines.append(f"{prefix}│{ann_text[:iw]:<{iw}}│")
        lines.append(f"{prefix}└{'─' * iw}┘")

    def _arrow(prefix: str = "  ", mid: int = IW // 2 + 2):
        pad = " " * (mid - 1)
        lines.append(f"{prefix}{pad}│")
        lines.append(f"{prefix}{pad}▼")

    def _summarize_group(group: LayerGroupDef) -> Tuple[str, str]:
        """Produce (box_label, annotation) for a collapsed layer group."""
        has_moe = any(
            b.category == "moe_router" or b.layout == "expert_grid"
            for b in group.blocks
        )
        has_attn = any(b.category == "attention" for b in group.blocks)
        has_ffn = any(
            b.category == "ffn" and "Dense" in b.label
            for b in group.blocks
        )

        if has_moe:
            box_label = "MoE Transformer Block"
        elif has_ffn or has_attn:
            box_label = "Dense Transformer Block"
        else:
            parts = group.label.split(" ", 1)
            box_label = parts[1] if len(parts) > 1 else group.label

        skip_attn = has_ffn and not has_moe

        summary_parts = [group.label]
        for block in group.blocks:
            if block.category == "norm":
                continue
            if block.layout == "expert_grid":
                continue
            if block.children and block.layout == "vertical":
                continue
            if block.category == "attention" and block.label and not skip_attn:
                if has_moe:
                    summary_parts.append(block.label)
                else:
                    summary_parts.append(block.annotation or block.label)
            elif block.category == "moe_router" and block.annotation:
                ann = block.annotation
                m = re.match(r"Top-(\d+) of (\d+) routed(?: \+ (\d+) shared)?", ann)
                if m:
                    top_k, routed, shared = m.group(1), m.group(2), m.group(3)
                    total = int(routed) + (int(shared) if shared else 0)
                    ann = f"Top-{top_k}/{total}"
                summary_parts.append(ann)
            elif block.category == "ffn" and block.annotation:
                ann = block.annotation.replace("intermediate = ", "FFN = ")
                summary_parts.append(ann)

        return box_label, " · ".join(summary_parts)

    # --- Title ---
    lines.append(_hbar())
    lines.append(f"  {template.title}")
    if template.subtitle:
        lines.append(f"  {template.subtitle}")
    lines.append(_hbar())
    lines.append("")

    # --- Header blocks ---
    for block in template.header_blocks:
        _box(block.label, block.annotation)
        _arrow()

    # --- Layer groups (collapsed into single summary boxes) ---
    for group in template.layer_groups:
        box_label, annotation = _summarize_group(group)
        _box(box_label, annotation)
        _arrow()

    # --- Footer blocks ---
    for i, block in enumerate(template.footer_blocks):
        _box(block.label, block.annotation)
        if i < len(template.footer_blocks) - 1:
            _arrow()

    # --- Metadata (table format with aligned columns) ---
    if template.metadata:
        lines.append("")
        lines.append(_hbar("─"))
        keys = list(template.metadata.keys())
        vals = list(template.metadata.values())
        col_w = max(10, (W - 2) // len(keys))
        key_line = "  " + "".join(f"{k:<{col_w}}" for k in keys)
        val_line = "  " + "".join(f"{v:<{col_w}}" for v in vals)
        lines.append(key_line)
        lines.append(val_line)
        lines.append(_hbar("─"))

    return "\n".join(lines)


def generate_arch_diagram(trace_path: str, output_png: str,
                          config_path: Optional[str] = None,
                          detailed: bool = False):
    """Generate architecture diagram PNG from a trace file.

    Returns the path to the generated PNG, or None on failure.
    Writes a .txt companion file: simplified by default, detailed with detailed=True.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    extractor = TraceHierarchyExtractor(trace_path)
    raw_root = extractor.extract_raw_root()
    if raw_root is None:
        logger.warning("No module hierarchy found in trace for arch diagram.")
        return None

    root_type = raw_root.module_type

    cfg = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)

    template = match_template(root_type, cfg, raw_root=raw_root)
    renderer = ArchDiagramRenderer()

    if template:
        logger.info("Using template for %s", root_type)
        renderer.render_template_to_png(template, output_png)

        base = output_png.rsplit(".", 1)[0]
        if detailed:
            txt_path = base + ".txt"
            txt = template_to_text(template)
            with open(txt_path, "w") as f:
                f.write(txt + "\n")
            logger.info("Detailed text diagram saved to %s", txt_path)
        else:
            txt_path = base + ".txt"
            txt = template_to_text_simplified(template)
            with open(txt_path, "w") as f:
                f.write(txt + "\n")
            logger.info("Simplified text diagram saved to %s", txt_path)
    else:
        logger.info("No template for %s, using fallback renderer", root_type)
        arch_roots = extractor.extract()
        title = root_type
        archs = cfg.get("architectures", [])
        if archs:
            title = archs[0]
        renderer.render_fallback_to_png(arch_roots, output_png, title=title)

    return output_png


def arch_diagram_main(trace_path: str,
                      config_path: Optional[str] = None,
                      detailed: bool = False):
    """End-to-end pipeline: trace -> arch_diagram/ folder with PNG + TXT."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    trace_dir = os.path.dirname(os.path.abspath(trace_path))
    out_dir = os.path.join(trace_dir, "arch_diagram")
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, "arch_diagram.png")

    print(f"Generating architecture diagram from: {os.path.abspath(trace_path)}")
    result = generate_arch_diagram(trace_path, png_path, config_path=config_path,
                                   detailed=detailed)
    if result is None:
        print("ERROR: Failed to generate diagram.", file=sys.stderr)
        sys.exit(1)

    txt_path = os.path.join(out_dir, "arch_diagram.txt")
    print(f"  PNG: {os.path.abspath(png_path)}")
    print(f"  TXT: {os.path.abspath(txt_path)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Static model structure inspector for SGLang model files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python model_inspector.py deepseek_v2.py --list-classes
  python model_inspector.py deepseek_v2.py --root DeepseekV2ForCausalLM
  python model_inspector.py deepseek_v2.py --config config.json -o out.txt
  python model_inspector.py deepseek_v2.py --show-line-numbers
  python model_inspector.py deepseek_v2.py --profiler-tree --config config.json
  python model_inspector.py deepseek_v2.py --profiler-tree --config config.json --depth 3

  # Architecture diagram from trace file (outputs to arch_diagram/ next to the trace)
  python model_inspector.py --trace /path/to/trace.json.gz --arch-diagram
  python model_inspector.py --trace /path/to/trace.json.gz --arch-diagram --detailed
  python model_inspector.py --trace /path/to/trace.json.gz --arch-diagram --config config.json
""",
    )
    parser.add_argument("model_file", nargs="?", default=None,
                        help="Path to model Python source file (not needed for --arch-diagram)")
    parser.add_argument(
        "--root", metavar="CLASS",
        help="Root class name (default: auto-detect *ForCausalLM)",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="Path to HuggingFace config.json",
    )
    parser.add_argument(
        "--list-classes", action="store_true",
        help="List all classes in the file and exit",
    )
    parser.add_argument(
        "--output", "-o", metavar="PATH",
        help="Save output to file",
    )
    parser.add_argument(
        "--show-line-numbers", action="store_true",
        help="Show source line numbers in the tree",
    )
    parser.add_argument(
        "--profiler-tree", action="store_true",
        help="PyTorch-profiler-style tree with expanded layer instances",
    )
    parser.add_argument(
        "--depth", type=int, default=2, metavar="N",
        help="Max depth for --profiler-tree (default: 2)",
    )
    parser.add_argument(
        "--trace", metavar="PATH",
        help="Path to trace file (.json.gz or .json) for --arch-diagram mode",
    )
    parser.add_argument(
        "--arch-diagram", action="store_true",
        help="Generate architecture block diagram from a trace file",
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Generate detailed text diagram (expanded layer internals). "
             "Default is simplified (collapsed layer groups).",
    )

    args = parser.parse_args()

    # --- Architecture diagram mode ---
    if args.arch_diagram:
        if not args.trace:
            print("Error: --arch-diagram requires --trace <path>", file=sys.stderr)
            sys.exit(1)
        if not os.path.isfile(args.trace):
            print(f"Error: Trace file not found: {args.trace}", file=sys.stderr)
            sys.exit(1)
        arch_diagram_main(args.trace, config_path=args.config,
                          detailed=args.detailed)
        return

    # --- Original static analysis mode ---
    if not args.model_file:
        print("Error: model_file is required (or use --arch-diagram --trace <path>)",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.model_file):
        print(f"Error: File not found: {args.model_file}", file=sys.stderr)
        sys.exit(1)

    # Parse the model file
    file_parser = ModelFileParser(args.model_file)
    classes = file_parser.parse()

    if not classes:
        print("No classes found in the file.", file=sys.stderr)
        sys.exit(1)

    formatter = OutputFormatter(show_line_numbers=args.show_line_numbers)
    output_parts = []

    # --list-classes mode
    if args.list_classes:
        output_parts.append(formatter.format_class_list(classes))
    else:
        # Determine root class
        root_class = args.root
        if not root_class:
            root_class = _auto_detect_root(classes)
            if not root_class:
                print(
                    "Error: Could not auto-detect root class. "
                    "Use --root to specify.",
                    file=sys.stderr,
                )
                sys.exit(1)

        if root_class not in classes:
            print(
                f"Error: Class '{root_class}' not found in {args.model_file}",
                file=sys.stderr,
            )
            print(f"Available classes: {', '.join(classes.keys())}", file=sys.stderr)
            sys.exit(1)

        # Build hierarchy
        builder = HierarchyBuilder(classes)
        tree = builder.build(root_class)

        if args.profiler_tree:
            # Need config for layer expansion
            config = None
            if args.config and os.path.isfile(args.config):
                config = ConfigParser.parse(args.config)

            # Pre-build class trees for all dispatch target classes
            class_trees = {}
            for node in _walk_tree(tree):
                if node.dispatch_map:
                    for cls_name in node.dispatch_map.values():
                        if cls_name not in class_trees:
                            class_trees[cls_name] = builder.build(cls_name)

            output_parts.append(
                formatter.format_profiler_tree(
                    tree, config=config, max_depth=args.depth,
                    class_trees=class_trees,
                )
            )
        else:
            output_parts.append(formatter.format_tree(tree))

    # Config metadata (skip in profiler-tree mode — config is used internally only)
    if args.config and not args.profiler_tree:
        if not os.path.isfile(args.config):
            print(f"Warning: Config file not found: {args.config}", file=sys.stderr)
        else:
            config = ConfigParser.parse(args.config)
            config_output = formatter.format_config(config)
            if config_output:
                output_parts.append(config_output)

    # Emit output
    full_output = "\n\n".join(output_parts) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(full_output)
        print(f"Output saved to {args.output}")
    else:
        print(full_output, end="")


if __name__ == "__main__":
    main()
