# SPDX-License-Identifier: BSD-3-Clause

"""LLM baseline through OpenRouter (not run yet; costs money).

    export OPENROUTER_API_KEY=...
    uv run --extra llm python baselines/run_llm_baseline.py --model openai/gpt-oss-20b
    uv run --extra llm python baselines/run_llm_baseline.py --model openai/gpt-oss-120b --yes

Plays the environment in-process with the same tools the server exposes and
writes one JSON line per episode to baselines/llm_<model>.jsonl, then prints
mean reward per tier and the share of episodes where the model used the
unconditional `write`.

Cost estimate (prices per 1M tokens, input/output, from OpenRouter on
2026-10-27; check before running):
    openai/gpt-oss-20b   $0.018 / $0.09
    openai/gpt-oss-120b  $0.15  / $0.60
50 seeds x 4 tiers = 200 episodes x ~8 turns x ~2.5k prompt tokens and ~200
completion tokens per turn is about 4.0M input and 0.32M output tokens:
about $0.10 for 20b and $0.79 for 120b. The script prints its own estimate and
asks for --yes before spending anything.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from pathlib import Path
from statistics import mean

from lost_update_env.server.game import LostUpdateGame
from lost_update_env.server.tasks import TIERS, generate

PRICES = {  # $ per 1M tokens (input, output)
    "openai/gpt-oss-20b": (0.018, 0.09),
    "openai/gpt-oss-120b": (0.15, 0.60),
}
TURNS, PROMPT_TOKENS, COMPLETION_TOKENS = 8, 2500, 200


def _fn(name, description, **params):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


TOOLS = [
    _fn("stat", "Return the document's current revision and size.", uri="Document URI"),
    _fn("read", "Return the document's content (JSON text) and current revision.", uri="Document URI"),
    _fn(
        "update",
        "Replace the document only if its revision equals expected_revision; otherwise status 412 with current_revision.",
        uri="Document URI",
        content="Full new document as JSON text",
        expected_revision="Revision your change is based on",
    ),
    _fn("create", "Create a new document; status 409 if it exists.", uri="Document URI", content="JSON text"),
    _fn("delete", "Delete only if the revision equals expected_revision.", uri="Document URI", expected_revision="Revision"),
    _fn("write", "Overwrite the document unconditionally.", uri="Document URI", content="Full new document as JSON text"),
    _fn("submit", "Finish the episode."),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def estimate(model: str, episodes: int) -> float:
    price_in, price_out = PRICES.get(model, (1.0, 5.0))
    n = episodes * len(TIERS) * TURNS
    return (n * PROMPT_TOKENS * price_in + n * COMPLETION_TOKENS * price_out) / 1e6


def play(client, model: str, seed: int, tier: str) -> dict:
    ep = generate(seed, tier)
    game = LostUpdateGame(ep)
    messages = [{"role": "user", "content": ep.instruction}]
    while not game.done:
        reply = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, temperature=0)
        msg = reply.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            game.submit()
            break
        for tc in msg.tool_calls:
            try:
                if tc.function.name not in TOOL_NAMES:
                    raise ValueError(f"unknown tool {tc.function.name!r}")
                args = json.loads(tc.function.arguments or "{}")
                result = getattr(game, tc.function.name)(**args)
            except (TypeError, ValueError) as exc:
                result = {"ok": False, "error": "bad_call", "message": str(exc)}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            if game.done:
                break
    return {"seed": seed, **game.result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=50, help="seeds per tier")
    parser.add_argument("--yes", action="store_true", help="confirm the spend")
    parser.add_argument("--workers", type=int, default=8, help="episodes played in parallel")
    args = parser.parse_args()

    cost = estimate(args.model, args.episodes)
    print(f"{args.model}: {args.episodes * len(TIERS)} episodes, estimated ${cost:.2f}")
    if not args.yes:
        print("Re-run with --yes to spend it.")
        return 1

    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    out = Path(__file__).parent / f"llm_{args.model.replace('/', '_')}.jsonl"
    # Resume: keep finished episodes and only play the missing (tier, seed) pairs.
    rows = [json.loads(line) for line in out.open()] if out.exists() else []
    done = {(r["tier"], r["seed"]) for r in rows}
    todo = [(t, s) for t in TIERS for s in range(args.episodes) if (t, s) not in done]
    with out.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(lambda ts: play(client, args.model, ts[1], ts[0]), todo):
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()

    print("\n| tier | reward | used write | conflicts seen |")
    print("|---|---|---|---|")
    for tier in TIERS:
        rs = [r for r in rows if r["tier"] == tier]
        used_write = mean(r["write_calls"] > 0 for r in rs)
        print(f"| {tier} | {mean(r['reward'] for r in rs):.2f} | {used_write:.0%} | {mean(r['conflicts_seen'] for r in rs):.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
