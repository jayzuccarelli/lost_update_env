# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic grading on JSON leaves. No LLM judge.

Let S be the start document, R_i the document after the first i rival writes
that actually landed, R the last of these, and E = intent(R): the merged
document a careful writer would produce. The final document F is compared to
E leaf by leaf (dicts are walked, lists and scalars are leaves, lists compare
as multisets).

    task      = share of leaves where E differs from R that F matches
    rivals    = share of landed rival writes whose changed leaves all match E in F
                (on a leaf both writers touch, the rival's own value also counts)
    integrity = share of untouched leaves (same in S, R and E) still intact in F
    preserve  = rivals * integrity
    reward    = task * preserve

A no-op leaves F == R, so task = 0. A blind overwrite from a stale read loses
a rival write, so rivals = 0 on T1 to T3. Writing only the requested field and
dropping the rest of the document fails integrity.
"""

import json
from typing import Any

from .ops import apply_ops

_MISSING = "<missing>"


def _canon(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, list):
        return "[" + ",".join(sorted(_canon(v) for v in value)) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{json.dumps(k)}:{_canon(v)}" for k, v in sorted(value.items())) + "}"
    return json.dumps(value)


def flatten(doc: Any, prefix: tuple = ()) -> dict[tuple, str]:
    if isinstance(doc, dict) and doc:
        out: dict[tuple, str] = {}
        for key, value in doc.items():
            out.update(flatten(value, prefix + (key,)))
        return out
    return {prefix: _canon(doc)}


def _diff(a: dict, b: dict) -> set:
    return {k for k in a.keys() | b.keys() if a.get(k, _MISSING) != b.get(k, _MISSING)}


def _share(leaves: set, final: dict, target: dict) -> float:
    if not leaves:
        return 1.0
    hits = sum(final.get(k, _MISSING) == target.get(k, _MISSING) for k in leaves)
    return hits / len(leaves)


def grade(start: dict, landed_rival_ops: list[dict], intent: list[dict], final: dict | None) -> dict:
    states = [start]
    for op in landed_rival_ops:
        states.append(apply_ops(states[-1], [op]))
    merged = apply_ops(states[-1], intent)

    f_states = [flatten(s) for s in states]
    f_r, f_e = f_states[-1], flatten(merged)
    f_final = flatten(final) if isinstance(final, dict) else {}

    task_leaves = _diff(f_r, f_e)
    task = _share(task_leaves, f_final, f_e)

    def kept(leaf) -> bool:
        # A leaf both writers touch counts as kept if F holds the rival's value or the merge.
        allowed = {f_e.get(leaf, _MISSING)}
        if leaf in task_leaves:
            allowed.add(f_r.get(leaf, _MISSING))
        return f_final.get(leaf, _MISSING) in allowed

    rival_hits = [
        all(kept(leaf) for leaf in _diff(f_states[i], f_states[i + 1]))
        for i in range(len(landed_rival_ops))
    ]
    rivals = sum(rival_hits) / len(rival_hits) if rival_hits else 1.0

    f_s = f_states[0]
    untouched = {k for k in f_s if f_s[k] == f_r.get(k, _MISSING) == f_e.get(k, _MISSING)}
    integrity = _share(untouched, f_final, f_e) if isinstance(final, dict) else 0.0

    preserve = rivals * integrity
    return {
        "reward": task * preserve,
        "task": task,
        "preserve": preserve,
        "rivals_preserved": rivals,
        "integrity": integrity,
        "rival_writes_landed": len(landed_rival_ops),
    }
