# SPDX-License-Identifier: BSD-3-Clause

"""Scripted policies: the oracle and the reference failure modes.

Each policy takes `call(tool_name, **arguments) -> dict` and the episode (the
oracle reads the intent from it; the agent only ever sees the instruction).
They run unchanged in-process against `LostUpdateGame` or over the wire against
a server, so the same code produces the baselines and the validation oracle.
"""

import json
from typing import Callable

from .ops import apply_ops
from .tasks import Episode

Call = Callable[..., dict]


def _merge(read_result: dict, ep: Episode) -> str:
    return json.dumps(apply_ops(json.loads(read_result["content"]), ep.intent))


def oracle(call: Call, ep: Episode) -> None:
    """Read, conditional update; on 412 re-read, reapply the intent, retry."""
    current = call("read", uri=ep.uri)
    while True:
        result = call("update", uri=ep.uri, content=_merge(current, ep), expected_revision=current["revision"])
        if result.get("ok") or result.get("status") != 412 or "episode_over" in result:
            break
        current = call("read", uri=ep.uri)
    call("submit")


def blind_write(call: Call, ep: Episode) -> None:
    """Read once, apply the change, overwrite unconditionally."""
    current = call("read", uri=ep.uri)
    call("write", uri=ep.uri, content=_merge(current, ep))
    call("submit")


def naive_retry(call: Call, ep: Episode) -> None:
    """Uses the precondition, but on 412 resends the stale content with the new revision."""
    current = call("read", uri=ep.uri)
    content, revision = _merge(current, ep), current["revision"]
    while True:
        result = call("update", uri=ep.uri, content=content, expected_revision=revision)
        if result.get("ok") or result.get("status") != 412 or "episode_over" in result:
            break
        revision = result["current_revision"]
    call("submit")


def noop(call: Call, ep: Episode) -> None:
    call("submit")


POLICIES = {
    "oracle": oracle,
    "blind_write": blind_write,
    "naive_retry": naive_retry,
    "noop": noop,
}
