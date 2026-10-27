# SPDX-License-Identifier: BSD-3-Clause

"""Edit operations on JSON documents.

Both the agent's intent and the rival's writes are lists of ops. Ops are plain
JSON-serializable dicts so an episode can be logged, replayed and regenerated
from its seed. Every op is semantic: it is applied to whatever document it is
given, which is what a correct merge does after a conflict.
"""

import copy
import json
from typing import Any

Path = list[str]


def get_path(doc: dict, path: Path) -> Any:
    node = doc
    for key in path:
        node = node[key]
    return node


def has_path(doc: dict, path: Path) -> bool:
    node = doc
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def set_path(doc: dict, path: Path, value: Any) -> None:
    node = doc
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def apply_op(doc: dict, op: dict) -> dict:
    """Return a new document with `op` applied."""
    out = copy.deepcopy(doc)
    kind = op["op"]
    path = op.get("path")
    if kind == "set":
        set_path(out, path, copy.deepcopy(op["value"]))
    elif kind == "incr":
        set_path(out, path, get_path(out, path) + op["by"])
    elif kind == "append":
        get_path(out, path).append(copy.deepcopy(op["item"]))
    elif kind == "derive":
        set_path(out, path, get_path(out, op["source"]) * op["factor"])
    elif kind == "rename":
        parent = get_path(out, path[:-1])
        parent[op["to"]] = parent.pop(path[-1])
    elif kind == "set_any":
        target = next((p for p in op["paths"] if has_path(out, p)), op["paths"][0])
        set_path(out, target, op["value"])
    else:
        raise ValueError(f"unknown op {kind!r}")
    return out


def apply_ops(doc: dict, ops: list[dict]) -> dict:
    for op in ops:
        doc = apply_op(doc, op)
    return doc


def dotted(path: Path) -> str:
    return ".".join(path)


def describe(op: dict) -> str:
    """Natural-language instruction for one agent op."""
    kind = op["op"]
    if kind == "set":
        return f"Set `{dotted(op['path'])}` to {json.dumps(op['value'])}."
    if kind == "incr":
        return f"Increase `{dotted(op['path'])}` by {op['by']}."
    if kind == "append":
        return f"Append {json.dumps(op['item'])} to the `{dotted(op['path'])}` list."
    if kind == "derive":
        return (
            f"Set `{dotted(op['path'])}` to {op['factor']} times the current value "
            f"of `{dotted(op['source'])}`."
        )
    if kind == "set_any":
        return (
            f"Set the {op['label']} (currently the key `{dotted(op['paths'][0])}`) "
            f"to {json.dumps(op['value'])}."
        )
    raise ValueError(f"op {kind!r} has no agent-facing description")
