# Real-provider three-turn Console benchmark

This directory retains the reproducible evidence for TASK-19641. The benchmark
mounts the real Console, types through its composer, queues the third prompt
before turn two releases, and uses the real local `load_tools` and confined
`fs_write` path against a credential-free llama.cpp server.

## Prerequisites

- Run from the repository root with the repository development environment
  installed. The examples use `../../.venv/bin/python` from the isolated
  worktree used to collect this evidence; substitute the path to your Python
  environment when reproducing elsewhere.
- Serve
  `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf` through the local
  OpenAI-compatible endpoint `http://127.0.0.1:9099`.
- Keep the listener dedicated to this benchmark. The recorded server reports
  one slot, so samples intentionally run serially.
- Start with an output directory that does not exist or is empty.

The runner refuses non-loopback endpoints, URL credentials, query strings, and
model mismatches. It builds isolated HOME/XDG/config/data/database/workspace and
shadow-repository roots for every child. It does not read user conversations,
use cloud credentials, or retain prompt, response, tool-result, or file bodies.

## Fail-fast preflight

```bash
../../.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --endpoint http://127.0.0.1:9099 \
  --model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  --output-root /tmp/tldw-console-three-turn-preflight \
  --preflight-only
```

The preflight verifies the exact model with a synthetic temperature-zero,
16-token, non-streaming completion. It disables template thinking so the probe
must return visible assistant content.

## Six-conversation smoke

```bash
../../.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --endpoint http://127.0.0.1:9099 \
  --model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  --iterations 1 \
  --output-root /tmp/tldw-console-three-turn-smoke
```

This runs one warmup for each arm, then one measured control/disabled/enabled
block. On the measured host it took about three minutes. Every provider call is
fixed at temperature `0`, a 512-token response cap, streaming enabled, and
reasoning effort `none`. The fixture IDs and hashes in the manifest pin the
three prompts, `local:fs_write` schema, confined mutation, and synthetic
workspace corpus.

## Full 90-sample measurement

```bash
../../.venv/bin/python Tests/Performance/run_console_three_turn_profile.py \
  --endpoint http://127.0.0.1:9099 \
  --model gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf \
  --iterations 30 \
  --output-root Docs/superpowers/qa/console-three-turn-real-provider
```

The full schedule is three warmups plus 90 measured conversations in rotated,
complete three-arm blocks. Budget roughly 45–60 minutes on the measured host;
generation speed and host load dominate elapsed time. Do not edit imported
source while the run is active.

## Evidence inventory

- `real-provider-three-turn.raw.jsonl`: flushed child-start and terminal sample
  records, including content-free timestamps, heartbeat vectors, exact tool
  ownership, token counts, review events, and final resource ownership.
- `real-provider-three-turn.manifest.json`: pinned revisions, model and request
  settings, fixture IDs/hashes, schedule, runtime/dependency versions,
  sanitized llama.cpp properties, host load, and listener CPU/RSS snapshots.
- `real-provider-three-turn.summary.json`: recomputable medians, nearest-rank
  p95 values, paired confidence bounds, validity gates, and verdict.
- `real-provider-three-turn-summary.md`: conservative interpretation separating
  provider time from application-owned latency.

The control is pinned to `5f720a40417eaa78f33619d5cbc82effc470104b`.
Candidate identity is resolved once from `HEAD` before the run begins. Failed,
partial, noisy, or incomplete evidence must not be promoted into a performance
claim.
