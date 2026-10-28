# /// script
# dependencies = [
#     "trl[vllm]>=1.16.0",
#     "openenv-lost-update-env @ git+https://huggingface.co/spaces/jayzuccarelli/lost_update_env",
# ]
# ///
# SPDX-License-Identifier: BSD-3-Clause

"""GRPO on the Lost Update environment with TRL's `environment_factory`.

    uv run examples/trl_grpo.py --model Qwen/Qwen3-1.7B --env-url http://localhost:8000

Each rollout is one environment session, so set MAX_CONCURRENT_ENVS on the
server to at least num_generations x prompts per step.
"""

import argparse
import json

from lost_update_env import CallToolAction, LostUpdateEnv

PROMPT = "Complete the task using the tools. Your changes must not erase anyone else's."


def make_env_class(env_url: str):
    class SharedDocument:
        def __init__(self):
            self.client = LostUpdateEnv(base_url=env_url).sync().connect()
            self.reward = 0.0

        def reset(self, **kwargs) -> str:
            self.reward = 0.0
            return self.client.reset().observation.metadata["instruction"]

        def _call(self, tool: str, **arguments) -> str:
            result = self.client.step(CallToolAction(tool_name=tool, arguments=arguments))
            if result.done:
                self.reward = result.reward or 0.0
            data = result.observation.result
            return json.dumps(data.get("data", data) if isinstance(data, dict) else data.data)

        def stat(self, uri: str) -> str:
            """Return the document's current revision and size.

            Args:
                uri: Document URI.
            """
            return self._call("stat", uri=uri)

        def read(self, uri: str) -> str:
            """Return the document's JSON content and current revision.

            Args:
                uri: Document URI.
            """
            return self._call("read", uri=uri)

        def update(self, uri: str, content: str, expected_revision: str) -> str:
            """Replace the document only if its revision still equals expected_revision.

            Args:
                uri: Document URI.
                content: The full new document as JSON text.
                expected_revision: The revision your change is based on.
            """
            return self._call("update", uri=uri, content=content, expected_revision=expected_revision)

        def write(self, uri: str, content: str) -> str:
            """Overwrite the document unconditionally.

            Args:
                uri: Document URI.
                content: The full new document as JSON text.
            """
            return self._call("write", uri=uri, content=content)

        def submit(self) -> str:
            """Finish the episode."""
            return self._call("submit")

    return SharedDocument


def reward_fn(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]


def main():
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": PROMPT}]] * (args.max_steps * 4)})
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        environment_factory=make_env_class(args.env_url),
        args=GRPOConfig(
            output_dir="lost-update-grpo",
            max_steps=args.max_steps,
            learning_rate=1e-6,
            num_generations=8,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
