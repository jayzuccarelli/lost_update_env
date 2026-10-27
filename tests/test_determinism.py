# SPDX-License-Identifier: BSD-3-Clause

import json

from lost_update_env.server.game import LostUpdateGame
from lost_update_env.server.lost_update_environment import LostUpdateEnvironment
from lost_update_env.server.policies import POLICIES
from lost_update_env.server.tasks import TIERS, generate
from openenv.core.env_server.mcp_types import CallToolAction


def trace(policy: str, seed: int, tier: str) -> list:
    ep = generate(seed, tier)
    game = LostUpdateGame(ep)
    log = []

    def call(tool, **kw):
        result = getattr(game, tool)(**kw)
        log.append((tool, json.dumps(kw, sort_keys=True), json.dumps(result, sort_keys=True)))
        return result

    POLICIES[policy](call, ep)
    return log


def test_same_seed_same_episode():
    for tier in TIERS:
        for seed in range(50):
            a, b = generate(seed, tier), generate(seed, tier)
            assert (a.doc, a.intent, a.rival, a.instruction) == (b.doc, b.intent, b.rival, b.instruction)


def test_replays_are_byte_identical():
    for tier in TIERS:
        for seed in range(20):
            for policy in POLICIES:
                assert trace(policy, seed, tier) == trace(policy, seed, tier)


def test_seeds_vary_the_episode():
    docs = {json.dumps(generate(seed, "T1").doc, sort_keys=True) for seed in range(50)}
    assert len(docs) > 40


def test_all_t3_overlap_kinds_and_families_are_reachable():
    eps = [generate(seed, "T3") for seed in range(200)]
    assert {e.overlap_kind for e in eps} == {"counter", "append", "derive", "rename"}
    assert {e.family for e in eps} == {"service_config", "memory_note", "todo_list", "settings"}


def test_environment_reset_is_deterministic_and_defaults_to_contended_tier():
    env = LostUpdateEnvironment()
    a = env.reset(seed=11, tier="T2").metadata
    b = env.reset(seed=11, tier="T2").metadata
    assert a == b
    assert {LostUpdateEnvironment().reset(seed=s).metadata["tier"] for s in range(30)} == {"T1", "T2", "T3"}


def test_environment_step_reports_reward_on_submit():
    env = LostUpdateEnvironment()
    meta = env.reset(seed=3, tier="T3").metadata
    ep = generate(3, "T3")

    def call(tool, **kw):
        obs = env.step(CallToolAction(tool_name=tool, arguments=kw))
        call.last = obs
        return obs.result.data

    POLICIES["oracle"](call, ep)
    assert meta["uri"] == ep.uri
    assert call.last.done is True and call.last.reward == 1.0
    assert env.state.tier == "T3" and env.state.seed == 3
