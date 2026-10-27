# SPDX-License-Identifier: BSD-3-Clause

import pytest

from lost_update_env.server.game import LostUpdateGame
from lost_update_env.server.policies import POLICIES
from lost_update_env.server.tasks import TIERS, generate

SEEDS = range(200)


def play(policy: str, seed: int, tier: str) -> LostUpdateGame:
    ep = generate(seed, tier)
    game = LostUpdateGame(ep)
    POLICIES[policy](lambda tool, **kw: getattr(game, tool)(**kw), ep)
    return game


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_scores_one_on_every_seed(tier):
    for seed in SEEDS:
        game = play("oracle", seed, tier)
        assert game.reward == 1.0, (seed, tier, game.result)


@pytest.mark.parametrize("tier", ["T1", "T2", "T3"])
@pytest.mark.parametrize("policy", ["blind_write", "naive_retry"])
def test_lost_update_policies_score_zero_under_contention(policy, tier):
    for seed in SEEDS:
        game = play(policy, seed, tier)
        assert game.reward == 0.0, (seed, tier, game.result)
        assert game.result["rivals_preserved"] == 0.0, "every landed rival write was lost"
        assert game.result["task"] == 1.0 or tier == "T3", "the change itself was applied"


def test_blind_write_is_fine_without_contention():
    for seed in SEEDS:
        assert play("blind_write", seed, "T0").reward == 1.0


@pytest.mark.parametrize("tier", TIERS)
def test_noop_scores_zero(tier):
    for seed in SEEDS:
        assert play("noop", seed, tier).reward == 0.0


@pytest.mark.parametrize("tier", ["T1", "T2", "T3"])
def test_copying_the_rival_scores_zero(tier):
    """Re-read after the rival lands and write that back unchanged: task stays 0."""
    for seed in range(50):
        ep = generate(seed, tier)
        game = LostUpdateGame(ep)
        first = game.read(ep.uri)
        game.update(ep.uri, first["content"], first["revision"])  # rival lands, 412
        while True:
            fresh = game.read(ep.uri)
            if game.update(ep.uri, fresh["content"], fresh["revision"])["ok"]:
                break
        game.submit()
        assert game.reward == 0.0
        assert game.result["preserve"] == 1.0


def test_writing_only_the_requested_field_fails_integrity():
    for seed in range(50):
        ep = generate(seed, "T0")
        game = LostUpdateGame(ep)
        game.write(ep.uri, '{"unrelated": 1}')
        game.submit()
        assert game.reward == 0.0


def test_step_budget_ends_episode_and_grades():
    ep = generate(1, "T1")
    game = LostUpdateGame(ep, max_steps=3)
    for _ in range(3):
        game.stat(ep.uri)
    assert game.done and game.reward == 0.0
    assert game.read(ep.uri)["error"] == "episode_done"
