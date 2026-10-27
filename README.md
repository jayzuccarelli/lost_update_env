---
title: Lost Update
emoji: 📝
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
license: bsd-3-clause
short_description: Edit a shared JSON doc while a rival writes to it
tags:
  - openenv
  - agent-environment
  - reinforcement-learning
  - concurrency
  - mcp
---

# Lost Update

An environment where the agent edits a shared JSON document while a scripted rival writes to the same document between the agent's steps. Unlike merge-conflict environments that hand the agent a static file with conflict markers, the conflict here happens live: the agent only learns about it through revision preconditions (`update(..., expected_revision)` fails with a 412 and the current revision) and has to re-read, merge and retry. An unconditional `write` is also on the menu, so the environment measures whether the agent chooses the precondition at all, not just whether it can merge.

## Why

When two writers read the same version of a document and each writes back its own copy, the second write silently erases the first. This is the lost update, and it shows up as soon as agents share a file, a memory store or a todo list. *Position: Multi-Agent Systems Should Prioritize Concurrency Control* (Yang, Li, Ji, Zhang and Jiang, ICML 2026; [arXiv:2608.18092](https://arxiv.org/abs/2608.18092), [ICML poster](https://icml.cc/virtual/2026/poster/67076)) lists it among the core hazards of multi-agent systems, and calls for benchmarks that vary contention and for training agents to be concurrency aware.

The fix is old and well understood. HTTP gives every representation an `ETag`; a client sends `If-Match` with the tag it read, and the server answers `412 Precondition Failed` if someone else wrote first ([RFC 9110, sections 8.8.3, 13.1.1 and 15.5.13](https://www.rfc-editor.org/rfc/rfc9110)). The MCP Filesystems Working Group's [problem statements](https://github.com/modelcontextprotocol/experimental-ext-filesystems/blob/main/docs/problem-statements.md) propose the same for MCP resources: an opaque revision returned by `stat` and `read`, writes conditional on it, and the current revision returned with the precondition error so the agent can re-read and retry. The protocol side is the easy part. What is missing is evidence that models use the precondition and repair their write correctly when it fails, and a place to train that skill. This environment is that place.

## Episode

1. `reset(seed, tier)` builds a document from one of four families (service config, agent memory note, todo list, user settings) and an instruction such as "Increase `stats.sessions` by 2" or "Set `api.timeout_ms` to 3 times the current value of `api.retry_interval_ms`".
2. The agent calls tools. Right before each of its first *k* write attempts (`update`, `write`, `create` or `delete` on the document), the rival commits its next scheduled edit. Whatever the agent read is now stale.
3. The episode ends on `submit()` or when the 16-call budget runs out, and the final document is graded.

Concurrency is simulated, not real: rival writes are seeded and step-interleaved, not wall-clock races. The same seed gives the same document, task and rival schedule, and the same sequence of tool calls gives byte-identical results.

## Tools

| tool | behavior |
|---|---|
| `stat(uri)` | `{revision, size}` |
| `read(uri)` | `{content, revision}`; content is JSON text |
| `update(uri, content, expected_revision)` | writes if the revision matches and returns the new one; otherwise `{status: 412, error: "precondition_failed", current_revision}` and nothing is written |
| `create(uri, content)` | fails with 409 if the URI exists |
| `delete(uri, expected_revision)` | conditional, 412 on mismatch |
| `write(uri, content)` | unconditional overwrite |
| `submit()` | ends the episode and returns the score breakdown |

Revisions are opaque strings compared exactly, like a strong ETag. Note the trap in the 412 response: it carries `current_revision`, so an agent can resend its stale content with the fresh revision and the server will accept it. That is still a lost update, and the grader treats it as one.

## Tiers

| tier | rival activity | what it tests |
|---|---|---|
| T0 | none | control: is the task itself solvable |
| T1 | one write to a different field, before the agent's first write attempt | detect the conflict, re-read, reapply |
| T2 | 2 to 4 writes to different fields, one before each of the agent's first *k* write attempts | retry loops and step budget |
| T3 | one write that overlaps the agent's intent: both bump the same counter, both append to the same list, the rival changes a value the agent's edit depends on, or the rival renames the key the agent must set | merge the intent, not the patch |

If `tier` is not given, it is drawn from the seed over T1 to T3.

## Reward

Deterministic checks on JSON leaves, no LLM judge. With R the document after the rival writes that landed and E the result of applying the agent's intent to R:

- `task`: share of the leaves the intent changes (E vs R) that the final document gets right.
- `preserve`: share of landed rival writes still present, times the share of untouched fields still intact.
- `reward = task * preserve`, in [0, 1].

Doing nothing scores 0 (task 0). A blind overwrite from a stale read scores 0 on T1 to T3 (the rival's write is gone). Copying the rival's version back scores 0 (task 0). Writing only the requested field and dropping the rest scores 0. Lists compare as multisets. The breakdown also reports `conflicts_seen`, `blind_overwrites`, `write_calls`, `update_calls` and `steps`.

## Baselines

Scripted policies, 50 seeds per tier (`uv run python baselines/run_scripted.py`):

| policy | T0 | T1 | T2 | T3 |
|---|---|---|---|---|
| oracle: read, conditional update, on 412 re-read + reapply + retry | 1.00 | 1.00 | 1.00 | 1.00 |
| blind write: read, apply, `write` | 1.00 | 0.00 | 0.00 | 0.00 |
| naive retry: conditional update, on 412 resend stale content with the new revision | 1.00 | 0.00 | 0.00 | 0.00 |
| noop: submit | 0.00 | 0.00 | 0.00 | 0.00 |

The oracle needs 3, 5, 9 and 5 tool calls on average on T0 to T3. Model baselines are not in yet; `baselines/run_llm_baseline.py` runs any OpenRouter model and also reports how often the model reached for `write`.

## Quick start

```bash
uv sync
uv run server                      # serves on :8000
uv run python oracle/solve.py --url http://localhost:8000 --episodes 5
```

```python
from lost_update_env import LostUpdateEnv

with LostUpdateEnv(base_url="http://localhost:8000").sync() as env:
    meta = env.reset(seed=7, tier="T1").observation.metadata
    print(meta["instruction"])
    doc = env.call_tool("read", uri=meta["uri"])
    print(env.call_tool("update", uri=meta["uri"], content=doc["content"], expected_revision=doc["revision"]))
```

From the Hub with `AutoEnv`:

```python
from openenv import AutoEnv

env = AutoEnv.from_env("jayzuccarelli/lost-update-env", base_url="https://jayzuccarelli-lost-update-env.hf.space")
```

## Training with TRL

`GRPOTrainer(environment_factory=...)` turns the public methods of a wrapper class into tools. `examples/trl_grpo.py` has the full script; the core is:

```python
class SharedDocument:
    def __init__(self):
        self.client = LostUpdateEnv(base_url=ENV_URL).sync().connect()

    def reset(self, **kwargs) -> str:
        self.reward = 0.0
        return self.client.reset().observation.metadata["instruction"]

    def update(self, uri: str, content: str, expected_revision: str) -> str:
        """Replace the document only if its revision still equals expected_revision.

        Args:
            uri: Document URI.
            content: The full new document as JSON text.
            expected_revision: The revision your change is based on.
        """
        return self._call("update", uri=uri, content=content, expected_revision=expected_revision)

    # read, stat, write and submit follow the same pattern


def reward_fn(environments, **kwargs):
    return [env.reward for env in environments]


trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=reward_fn,
    train_dataset=dataset,
    environment_factory=SharedDocument,
    args=GRPOConfig(num_generations=8),
)
```

Each rollout holds one session, so set `MAX_CONCURRENT_ENVS` on the server (default 8) to at least the number of parallel rollouts.

## Limits of v0

- The rival is scripted, not a second learner. Making it a second policy is the natural next step.
- One document per episode, and the 412 carries no diff. A "semantic conflict details" variant (tell the agent which fields changed) is a planned knob.
- Rival writes land relative to the agent's write attempts, which models the read-to-write window but not every interleaving.

## Citation

```bibtex
@inproceedings{yang2026concurrency,
  title     = {Position: Multi-Agent Systems Should Prioritize Concurrency Control},
  author    = {Yang, Xin and Li, Letian and Ji, Zimo and Zhang, Terry Jingchen and Jiang, Wenyuan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {arXiv:2608.18092}
}
```

## License

BSD-3-Clause. Author: Eugenio "Jay" Zuccarelli.
