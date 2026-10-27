# SPDX-License-Identifier: BSD-3-Clause

"""Scripted baselines: mean reward per tier for each reference policy.

    uv run python baselines/run_scripted.py --episodes 50

Runs in-process (no server), deterministic, a few seconds on a laptop CPU.
"""

import argparse
from statistics import mean

from lost_update_env.server.game import LostUpdateGame
from lost_update_env.server.policies import POLICIES
from lost_update_env.server.tasks import TIERS, generate


def run(policy: str, tier: str, episodes: int) -> list[dict]:
    results = []
    for seed in range(episodes):
        ep = generate(seed, tier)
        game = LostUpdateGame(ep)
        POLICIES[policy](lambda tool, **kw: getattr(game, tool)(**kw), ep)
        results.append(game.result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    print(f"Mean reward over {args.episodes} seeds per tier (seeds 0..{args.episodes - 1}).\n")
    print("| policy | " + " | ".join(TIERS) + " |")
    print("|---" * (len(TIERS) + 1) + "|")
    for policy in POLICIES:
        cells = [f"{mean(r['reward'] for r in run(policy, tier, args.episodes)):.2f}" for tier in TIERS]
        print(f"| {policy} | " + " | ".join(cells) + " |")

    print("\nOracle cost per episode (mean tool calls, mean 412s seen):\n")
    print("| tier | tool calls | 412s |")
    print("|---|---|---|")
    for tier in TIERS:
        rows = run("oracle", tier, args.episodes)
        print(f"| {tier} | {mean(r['steps'] for r in rows):.1f} | {mean(r['conflicts_seen'] for r in rows):.1f} |")


if __name__ == "__main__":
    main()
