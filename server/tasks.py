# SPDX-License-Identifier: BSD-3-Clause

"""Seeded episode generation: document, agent intent and rival schedule.

`generate(seed, tier)` is a pure function. The same (seed, tier) always gives
the same document, instruction and rival writes, which is what makes episodes
reproducible and lets the oracle reconstruct the intent without reading any
server state.

Tiers:
    T0: no rival activity (control).
    T1: one rival write to a disjoint field, landing just before the agent's
        first write attempt.
    T2: 2 to 4 rival writes to disjoint fields, one before each of the agent's
        first k write attempts.
    T3: one rival write that overlaps the agent's intent in a compatible way
        (both bump the same counter, both append to the same list, the rival
        changes a value the agent's edit depends on, or the rival renames the
        key the agent must set).
"""

import random
from dataclasses import dataclass

from .ops import Path, apply_ops, describe, get_path

TIERS = ("T0", "T1", "T2", "T3")

PEOPLE = ["ana", "bo", "chen", "dara", "eli", "femi", "gus", "hana", "ivo", "jun"]
ORIGINS = [
    "https://app.example.com",
    "https://admin.example.com",
    "https://docs.example.com",
    "https://beta.example.com",
    "https://status.example.com",
    "https://partners.example.com",
]
FACTS = [
    "prefers short answers",
    "works in UTC+1",
    "allergic to peanuts",
    "has a dog named Miso",
    "is learning Rust",
    "runs on Tuesdays",
    "reads on paper",
    "uses a split keyboard",
    "drinks decaf after noon",
    "keeps notes in markdown",
]
TOPICS = ["travel", "budget", "fitness", "recipes", "garden", "taxes", "music", "books"]
TODO_TEXT = [
    "renew passport",
    "book dentist",
    "fix the flaky test",
    "water the basil",
    "send the invoice",
    "update the README",
    "order printer ink",
    "back up the laptop",
    "call the landlord",
    "review the PR",
]
LABELS = ["home", "work", "urgent", "errand", "later", "health", "money"]
DEVICES = ["laptop", "phone", "tablet", "tv", "watch", "desktop", "speaker"]
LIST_VOCAB = {
    "owners": PEOPLE,
    "allowed_origins": ORIGINS,
    "facts": FACTS,
    "pinned": TOPICS,
    "labels": LABELS,
    "devices": DEVICES,
}


@dataclass(frozen=True)
class Slot:
    """A field an op can target. `kind` is int, bool, str or list."""

    path: tuple[str, ...]
    kind: str
    vocab: tuple = ()


@dataclass(frozen=True)
class Overlaps:
    """Fields used by T3 overlapping edits."""

    counter: tuple[str, ...]
    counter_steps: tuple[int, ...]
    lst: tuple[str, ...]
    derive_target: tuple[str, ...]
    derive_source: tuple[str, ...]
    derive_factor: int
    derive_values: tuple[int, ...]
    rename_path: tuple[str, ...]
    rename_to: str
    rename_label: str


@dataclass
class Family:
    name: str
    uri: str
    doc: dict
    slots: list[Slot]
    overlaps: Overlaps


@dataclass
class Episode:
    seed: int
    tier: str
    family: str
    uri: str
    doc: dict
    intent: list[dict]
    rival: list[dict]
    instruction: str = ""
    overlap_kind: str | None = None


def _service_config(rng: random.Random) -> Family:
    name = rng.choice(["checkout", "search", "billing", "ingest", "notifier", "gateway"])
    doc = {
        "service": name,
        "version": rng.randint(1, 9),
        "api": {
            "rate_limit": rng.choice([100, 200, 250, 500]),
            "timeout_ms": rng.choice([1500, 2000, 3000]),
            "retry_interval_ms": rng.choice([100, 200, 250, 400]),
            "base_path": rng.choice(["/v1", "/v2"]),
        },
        "db": {
            "host": f"db-{rng.randint(1, 4)}.internal",
            "max_conn": rng.choice([16, 32, 48]),
            "pool_timeout_s": rng.choice([5, 10, 30]),
        },
        "features": {
            "beta_ui": rng.random() < 0.5,
            "audit_log": rng.random() < 0.5,
            "dark_launch": rng.random() < 0.5,
        },
        "owners": rng.sample(PEOPLE, 2),
        "allowed_origins": rng.sample(ORIGINS, 2),
    }
    slots = [
        Slot(("api", "rate_limit"), "int"),
        Slot(("api", "base_path"), "str", ("/v1", "/v2", "/v3", "/api")),
        Slot(("db", "host"), "str", tuple(f"db-{i}.internal" for i in range(1, 9))),
        Slot(("db", "pool_timeout_s"), "int"),
        Slot(("features", "beta_ui"), "bool"),
        Slot(("features", "audit_log"), "bool"),
        Slot(("features", "dark_launch"), "bool"),
        Slot(("owners",), "list", tuple(PEOPLE)),
        Slot(("allowed_origins",), "list", tuple(ORIGINS)),
    ]
    overlaps = Overlaps(
        counter=("version",),
        counter_steps=(1, 2),
        lst=("owners",),
        derive_target=("api", "timeout_ms"),
        derive_source=("api", "retry_interval_ms"),
        derive_factor=3,
        derive_values=(150, 300, 350, 600),
        rename_path=("db", "max_conn"),
        rename_to="max_connections",
        rename_label="database connection limit",
    )
    return Family("service_config", f"mem://team/{name}/service.json", doc, slots, overlaps)


def _memory_note(rng: random.Random) -> Family:
    user = rng.choice(PEOPLE)
    doc = {
        "user": user,
        "facts": rng.sample(FACTS, 3),
        "prefs": {
            "theme": rng.choice(["light", "dark"]),
            "language": rng.choice(["en", "it", "es"]),
            "units": rng.choice(["metric", "imperial"]),
            "verbosity": rng.choice(["low", "medium"]),
        },
        "stats": {"sessions": rng.randint(3, 90), "messages": rng.randint(20, 900)},
        "limits": {"summary_k": rng.choice([2, 4, 8]), "context_k": rng.choice([10, 20, 50])},
        "pinned": rng.sample(TOPICS, 2),
        "flags": {"onboarded": rng.random() < 0.5, "beta": rng.random() < 0.5},
    }
    slots = [
        Slot(("prefs", "theme"), "str", ("light", "dark", "system")),
        Slot(("prefs", "units"), "str", ("metric", "imperial")),
        Slot(("prefs", "verbosity"), "str", ("low", "medium", "high")),
        Slot(("stats", "messages"), "int"),
        Slot(("flags", "onboarded"), "bool"),
        Slot(("flags", "beta"), "bool"),
        Slot(("pinned",), "list", tuple(TOPICS)),
        Slot(("facts",), "list", tuple(FACTS)),
    ]
    overlaps = Overlaps(
        counter=("stats", "sessions"),
        counter_steps=(1, 2, 3),
        lst=("facts",),
        derive_target=("limits", "context_k"),
        derive_source=("limits", "summary_k"),
        derive_factor=4,
        derive_values=(3, 5, 6, 7),
        rename_path=("prefs", "language"),
        rename_to="locale",
        rename_label="language preference",
    )
    return Family("memory_note", f"mem://agents/{user}/memory.json", doc, slots, overlaps)


def _todo_list(rng: random.Random) -> Family:
    project = rng.choice(["move", "launch", "garden", "taxes", "trip"])
    texts = rng.sample(TODO_TEXT, 3)
    doc = {
        "project": project,
        "items": [
            {"id": f"t{i + 1}", "text": t, "done": rng.random() < 0.3}
            for i, t in enumerate(texts)
        ],
        "labels": rng.sample(LABELS, 2),
        "counts": {"open": rng.randint(1, 9), "archived": rng.randint(0, 40)},
        "settings": {
            "sort": rng.choice(["due", "priority"]),
            "reminders": rng.random() < 0.5,
            "reminder_lead_min": rng.choice([10, 15, 30]),
            "snooze_min": rng.choice([5, 7, 9]),
            "view": rng.choice(["list", "board"]),
            "week_start": rng.choice(["mon", "sun"]),
            "show_done": rng.random() < 0.5,
        },
        "owner": rng.choice(PEOPLE),
    }
    slots = [
        Slot(("labels",), "list", tuple(LABELS)),
        Slot(("counts", "archived"), "int"),
        Slot(("settings", "reminders"), "bool"),
        Slot(("settings", "view"), "str", ("list", "board", "calendar")),
        Slot(("project",), "str", ("move", "launch", "garden", "taxes", "trip", "q4")),
        Slot(("settings", "week_start"), "str", ("mon", "sun", "sat")),
        Slot(("settings", "show_done"), "bool"),
        Slot(("owner",), "str", tuple(PEOPLE)),
    ]
    overlaps = Overlaps(
        counter=("counts", "open"),
        counter_steps=(1, 2),
        lst=("items",),
        derive_target=("settings", "snooze_min"),
        derive_source=("settings", "reminder_lead_min"),
        derive_factor=2,
        derive_values=(20, 25, 45, 60),
        rename_path=("settings", "sort"),
        rename_to="order_by",
        rename_label="sort order",
    )
    return Family("todo_list", f"mem://projects/{project}/todo.json", doc, slots, overlaps)


def _settings(rng: random.Random) -> Family:
    user = rng.choice(PEOPLE)
    doc = {
        "display": {
            "brightness": rng.choice([40, 60, 80]),
            "theme": rng.choice(["light", "dark"]),
            "font_size": rng.choice([12, 14, 16]),
        },
        "audio": {
            "volume": rng.choice([20, 35, 50]),
            "muted": rng.random() < 0.5,
            "output": rng.choice(["speakers", "headphones"]),
        },
        "notifications": {
            "email": rng.random() < 0.5,
            "push": rng.random() < 0.5,
            "digest_hour": rng.choice([7, 8, 9]),
        },
        "sync": {
            "enabled": rng.random() < 0.5,
            "retry_s": rng.choice([5, 10, 15]),
            "interval_s": rng.choice([60, 120, 300]),
        },
        "devices": rng.sample(DEVICES, 2),
    }
    slots = [
        Slot(("display", "brightness"), "int"),
        Slot(("display", "font_size"), "int"),
        Slot(("audio", "muted"), "bool"),
        Slot(("audio", "output"), "str", ("speakers", "headphones", "hdmi")),
        Slot(("notifications", "email"), "bool"),
        Slot(("notifications", "push"), "bool"),
        Slot(("notifications", "digest_hour"), "int"),
        Slot(("sync", "enabled"), "bool"),
        Slot(("devices",), "list", tuple(DEVICES)),
    ]
    overlaps = Overlaps(
        counter=("audio", "volume"),
        counter_steps=(5, 10, 15),
        lst=("devices",),
        derive_target=("sync", "interval_s"),
        derive_source=("sync", "retry_s"),
        derive_factor=6,
        derive_values=(20, 25, 30, 45),
        rename_path=("display", "theme"),
        rename_to="color_scheme",
        rename_label="display theme",
    )
    return Family("settings", f"mem://users/{user}/settings.json", doc, slots, overlaps)


FAMILIES = (_service_config, _memory_note, _todo_list, _settings)
OVERLAP_KINDS = ("counter", "append", "derive", "rename")


def _new_list_item(rng: random.Random, family: Family, path: Path, doc: dict, taken: set, who: str):
    current = get_path(doc, path)
    if family.name == "todo_list" and list(path) == ["items"]:
        used = {item["text"] for item in current} | taken
        text = rng.choice([t for t in TODO_TEXT if t not in used])
        taken.add(text)
        prefix = "a" if who == "agent" else "r"
        return {"id": f"{prefix}{len(current) + 1}", "text": text, "done": False}
    vocab = LIST_VOCAB[path[-1]]
    choice = rng.choice([v for v in vocab if v not in current and v not in taken])
    taken.add(choice)
    return choice


def _slot_op(rng: random.Random, family: Family, slot: Slot, doc: dict, taken: set, who: str) -> dict:
    path = list(slot.path)
    current = get_path(doc, path)
    if slot.kind == "bool":
        return {"op": "set", "path": path, "value": not current}
    if slot.kind == "str":
        return {"op": "set", "path": path, "value": rng.choice([v for v in slot.vocab if v != current])}
    if slot.kind == "int":
        step = max(1, current // 10)
        if rng.random() < 0.5:
            return {"op": "incr", "path": path, "by": step * rng.randint(1, 5)}
        return {"op": "set", "path": path, "value": current + step * rng.randint(1, 9)}
    if slot.kind == "list":
        return {"op": "append", "path": path, "item": _new_list_item(rng, family, path, doc, taken, who)}
    raise ValueError(slot.kind)


def _overlap(rng: random.Random, family: Family, kind: str) -> tuple[list[dict], list[dict]]:
    """Return (agent_intent, rival_ops) for a T3 overlapping edit."""
    o, doc = family.overlaps, family.doc
    if kind == "counter":
        # Distinct steps, so a stale "+a" can never be mistaken for the rival's "+b".
        a, b = rng.sample(o.counter_steps, 2)
        return (
            [{"op": "incr", "path": list(o.counter), "by": a}],
            [{"op": "incr", "path": list(o.counter), "by": b}],
        )
    if kind == "append":
        taken: set = set()
        mine = _new_list_item(rng, family, list(o.lst), doc, taken, "agent")
        theirs = _new_list_item(rng, family, list(o.lst), doc, taken, "rival")
        return (
            [{"op": "append", "path": list(o.lst), "item": mine}],
            [{"op": "append", "path": list(o.lst), "item": theirs}],
        )
    if kind == "derive":
        current = get_path(doc, list(o.derive_source))
        value = rng.choice([v for v in o.derive_values if v != current])
        return (
            [{"op": "derive", "path": list(o.derive_target), "source": list(o.derive_source), "factor": o.derive_factor}],
            [{"op": "set", "path": list(o.derive_source), "value": value}],
        )
    if kind == "rename":
        path = list(o.rename_path)
        current = get_path(doc, path)
        new_path = path[:-1] + [o.rename_to]
        if isinstance(current, int):
            value = current + max(1, current // 4) * rng.randint(1, 3)
        else:
            vocab = {
                "language": ["en", "it", "es", "fr", "de"],
                "sort": ["due", "priority", "created"],
                "theme": ["light", "dark", "system"],
            }[path[-1]]
            value = rng.choice([v for v in vocab if v != current])
        return (
            [{"op": "set_any", "paths": [path, new_path], "value": value, "label": o.rename_label}],
            [{"op": "rename", "path": path, "to": o.rename_to}],
        )
    raise ValueError(kind)


def _changes(before: dict, ops: list[dict]) -> bool:
    return all(apply_ops(before, [op]) != before for op in ops)


def _valid(ep: Episode) -> bool:
    """Every agent and rival op must change the document it lands on.

    Checked against the start document, the document after all rival writes,
    and each intermediate rival state, so a no-op agent can never get task
    credit and every rival write is observable.
    """
    states = [ep.doc]
    for op in ep.rival:
        nxt = apply_ops(states[-1], [op])
        if nxt == states[-1]:
            return False
        states.append(nxt)
    final_ok = apply_ops(states[-1], ep.intent) != states[-1]
    start_ok = apply_ops(ep.doc, ep.intent) != ep.doc
    each_ok = all(_changes(states[-1], [op]) for op in ep.intent if op["op"] != "set_any")
    stale_differs = True
    if ep.rival:
        # A blind write computed from the start document must not equal the merged result.
        stale_differs = apply_ops(ep.doc, ep.intent) != apply_ops(states[-1], ep.intent)
    return final_ok and start_ok and each_ok and stale_differs


PREAMBLE = (
    "You are editing the shared JSON document `{uri}`. Other agents and people "
    "edit this document too, and may change it at any moment, including between "
    "your read and your write. Keep their changes.\n\n"
    "Task:\n{tasks}\n\n"
    "Tools: stat, read, update (conditional on expected_revision), create, delete, "
    "write (unconditional overwrite), submit. Call submit() when you are done."
)


def resolve_tier(tier, seed: int) -> str:
    if tier is None:
        return ("T1", "T2", "T3")[random.Random(f"tier:{seed}").randrange(3)]
    if isinstance(tier, int) or (isinstance(tier, str) and tier.isdigit()):
        tier = f"T{int(tier)}"
    tier = str(tier).upper()
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    return tier


def generate(seed: int, tier) -> Episode:
    tier = resolve_tier(tier, seed)
    rng = random.Random(f"lost-update:{seed}:{tier}")
    while True:
        family = rng.choice(FAMILIES)(rng)
        overlap_kind = None
        if tier == "T3":
            overlap_kind = rng.choice(OVERLAP_KINDS)
            intent, rival = _overlap(rng, family, overlap_kind)
        else:
            n_rival = {"T0": 0, "T1": 1, "T2": rng.randint(2, 4)}[tier]
            n_agent = rng.randint(1, 2)
            slots = rng.sample(family.slots, n_agent + n_rival)
            taken: set = set()
            intent = [_slot_op(rng, family, s, family.doc, taken, "agent") for s in slots[:n_agent]]
            rival = [_slot_op(rng, family, s, family.doc, taken, "rival") for s in slots[n_agent:]]
        ep = Episode(
            seed=seed,
            tier=tier,
            family=family.name,
            uri=family.uri,
            doc=family.doc,
            intent=intent,
            rival=rival,
            overlap_kind=overlap_kind,
        )
        if _valid(ep):
            break
    tasks = "\n".join(f"- {describe(op)}" for op in intent)
    ep.instruction = PREAMBLE.format(uri=ep.uri, tasks=tasks)
    return ep
