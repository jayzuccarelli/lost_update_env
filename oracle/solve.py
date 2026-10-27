# SPDX-License-Identifier: BSD-3-Clause

"""Oracle for the Lost Update environment.

Read, conditional update; on a 412, re-read, reapply the intent to the fresh
document and retry. The oracle knows the intent because episodes are a pure
function of (seed, tier): it regenerates the episode locally instead of reading
privileged server state.

    uv run python oracle/solve.py                          # in-process, all tiers
    uv run python oracle/solve.py --url http://localhost:8000 --episodes 5

Exits non-zero if any episode scores below 1.0.
"""

import argparse
import sys

from lost_update_env.server.game import LostUpdateGame
from lost_update_env.server.policies import oracle
from lost_update_env.server.tasks import TIERS, generate


def run_local(seed: int, tier: str) -> float:
    ep = generate(seed, tier)
    game = LostUpdateGame(ep)
    oracle(lambda tool, **kw: getattr(game, tool)(**kw), ep)
    return game.reward


def run_remote(url: str, seed: int, tier: str) -> float:
    from lost_update_env import CallToolAction, LostUpdateEnv

    ep = generate(seed, tier)
    with LostUpdateEnv(base_url=url).sync() as env:
        env.reset(seed=seed, tier=tier)
        last = {}

        def call(tool, **kwargs):
            result = env.step(CallToolAction(tool_name=tool, arguments=kwargs))
            last["reward"] = result.reward
            data = result.observation.result
            return data.get("data", data) if isinstance(data, dict) else data.data

        oracle(call, ep)
        return last["reward"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--tiers", nargs="+", default=list(TIERS))
    args = parser.parse_args()
    worst = 1.0
    for tier in args.tiers:
        rewards = [
            run_remote(args.url, s, tier) if args.url else run_local(s, tier)
            for s in range(args.episodes)
        ]
        worst = min(worst, *rewards)
        print(f"{tier}: mean {sum(rewards) / len(rewards):.3f}  min {min(rewards):.3f}")
    return 0 if worst == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
