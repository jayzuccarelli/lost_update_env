# SPDX-License-Identifier: BSD-3-Clause

"""The shared store, the scripted rival writer and the tool semantics.

The store maps a URI to (JSON object, opaque revision). Revisions are compared
byte for byte like a strong HTTP ETag. `update` and `delete` take an
`expected_revision` and fail with a 412-style result that carries the current
revision when it does not match. `write` overwrites unconditionally.

Concurrency is simulated, not real: the rival's i-th scheduled write lands
immediately before the agent's i-th write attempt (update, write, create or
delete) on the target document. Whatever the agent read before that attempt is
now stale. The schedule depends only on the seed and on how many write
attempts the agent has made, so every episode replays exactly.
"""

import hashlib
import json
from typing import Any

from .grader import grade
from .ops import apply_ops
from .tasks import Episode

DEFAULT_MAX_STEPS = 16


def _parse(content: str) -> tuple[dict | None, dict | None]:
    try:
        value = json.loads(content)
    except (TypeError, ValueError) as exc:
        return None, {"ok": False, "status": 400, "error": "invalid_json", "message": str(exc)}
    if not isinstance(value, dict):
        return None, {"ok": False, "status": 400, "error": "invalid_json", "message": "content must be a JSON object"}
    return value, None


class LostUpdateGame:
    def __init__(self, episode: Episode, max_steps: int = DEFAULT_MAX_STEPS):
        self.episode = episode
        self.max_steps = max_steps
        self._rev_counter = 0
        self.store: dict[str, dict[str, Any]] = {
            episode.uri: {"content": episode.doc, "revision": self._next_revision()}
        }
        self.rival_landed: list[dict] = []
        self.write_attempts = 0
        self.steps = 0
        self.done = False
        self.result: dict | None = None
        self.last_seen: dict[str, str | None] = {}
        self.stats = {
            "conflicts_seen": 0,
            "blind_overwrites": 0,
            "write_calls": 0,
            "update_calls": 0,
        }

    # -- helpers -----------------------------------------------------------

    def _next_revision(self) -> str:
        self._rev_counter += 1
        raw = f"{self.episode.seed}:{self.episode.tier}:{self._rev_counter}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _text(self, uri: str) -> str:
        return json.dumps(self.store[uri]["content"], indent=2)

    def _commit(self, uri: str, content: dict) -> str:
        revision = self._next_revision()
        self.store[uri] = {"content": content, "revision": revision}
        return revision

    def _seen(self, uri: str, revision: str | None) -> None:
        self.last_seen[uri] = revision

    def _rival_turn(self, uri: str) -> None:
        """Land the next scheduled rival write before this write attempt."""
        if uri != self.episode.uri:
            return
        self.write_attempts += 1
        if len(self.rival_landed) >= len(self.episode.rival):
            return
        op = self.episode.rival[len(self.rival_landed)]
        self.rival_landed.append(op)
        if uri in self.store:
            self._commit(uri, apply_ops(self.store[uri]["content"], [op]))

    def _tick(self, result: dict) -> dict:
        self.steps += 1
        if not self.done and self.steps >= self.max_steps:
            self.finish()
            result["episode_over"] = "step budget exhausted"
        result["steps_left"] = max(0, self.max_steps - self.steps)
        return result

    def _guard(self) -> dict | None:
        if self.done:
            return {"ok": False, "error": "episode_done", "message": "The episode is over."}
        return None

    # -- tools -------------------------------------------------------------

    def stat(self, uri: str) -> dict:
        if (closed := self._guard()) is not None:
            return closed
        if uri not in self.store:
            return self._tick({"ok": False, "status": 404, "error": "not_found", "uri": uri})
        revision = self.store[uri]["revision"]
        self._seen(uri, revision)
        return self._tick({"ok": True, "uri": uri, "revision": revision, "size": len(self._text(uri))})

    def read(self, uri: str) -> dict:
        if (closed := self._guard()) is not None:
            return closed
        if uri not in self.store:
            return self._tick({"ok": False, "status": 404, "error": "not_found", "uri": uri})
        revision = self.store[uri]["revision"]
        self._seen(uri, revision)
        return self._tick({"ok": True, "uri": uri, "revision": revision, "content": self._text(uri)})

    def update(self, uri: str, content: str, expected_revision: str) -> dict:
        if (closed := self._guard()) is not None:
            return closed
        self.stats["update_calls"] += 1
        self._rival_turn(uri)
        doc, error = _parse(content)
        if error:
            return self._tick(error)
        if uri not in self.store:
            return self._tick({"ok": False, "status": 404, "error": "not_found", "uri": uri})
        current = self.store[uri]["revision"]
        if expected_revision != current:
            self.stats["conflicts_seen"] += 1
            return self._tick(
                {
                    "ok": False,
                    "status": 412,
                    "error": "precondition_failed",
                    "message": "The document changed since that revision. Read it again and reapply your change.",
                    "uri": uri,
                    "current_revision": current,
                }
            )
        revision = self._commit(uri, doc)
        self._seen(uri, revision)
        return self._tick({"ok": True, "uri": uri, "revision": revision})

    def write(self, uri: str, content: str) -> dict:
        if (closed := self._guard()) is not None:
            return closed
        self.stats["write_calls"] += 1
        self._rival_turn(uri)
        doc, error = _parse(content)
        if error:
            return self._tick(error)
        if uri in self.store and self.last_seen.get(uri) != self.store[uri]["revision"]:
            self.stats["blind_overwrites"] += 1
        revision = self._commit(uri, doc)
        self._seen(uri, revision)
        return self._tick({"ok": True, "uri": uri, "revision": revision})

    def create(self, uri: str, content: str) -> dict:
        if (closed := self._guard()) is not None:
            return closed
        self._rival_turn(uri)
        doc, error = _parse(content)
        if error:
            return self._tick(error)
        if uri in self.store:
            return self._tick(
                {
                    "ok": False,
                    "status": 409,
                    "error": "already_exists",
                    "uri": uri,
                    "current_revision": self.store[uri]["revision"],
                }
            )
        revision = self._commit(uri, doc)
        self._seen(uri, revision)
        return self._tick({"ok": True, "uri": uri, "revision": revision})

    def delete(self, uri: str, expected_revision: str) -> dict:
        if (closed := self._guard()) is not None:
            return closed
        self._rival_turn(uri)
        if uri not in self.store:
            return self._tick({"ok": False, "status": 404, "error": "not_found", "uri": uri})
        current = self.store[uri]["revision"]
        if expected_revision != current:
            self.stats["conflicts_seen"] += 1
            return self._tick(
                {
                    "ok": False,
                    "status": 412,
                    "error": "precondition_failed",
                    "uri": uri,
                    "current_revision": current,
                }
            )
        del self.store[uri]
        self._seen(uri, None)
        return self._tick({"ok": True, "uri": uri})

    def submit(self) -> dict:
        if (closed := self._guard()) is not None:
            return closed
        self.steps += 1
        self.finish()
        return {"ok": True, "done": True, **self.result}

    # -- grading -----------------------------------------------------------

    def finish(self) -> dict:
        ep = self.episode
        final = self.store.get(ep.uri, {}).get("content")
        self.result = {
            **grade(ep.doc, self.rival_landed, ep.intent, final),
            **self.stats,
            "steps": self.steps,
            "tier": ep.tier,
        }
        self.done = True
        return self.result

    @property
    def reward(self) -> float:
        return self.result["reward"] if self.result else 0.0
