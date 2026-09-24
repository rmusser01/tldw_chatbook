# Long-tail fixture captures

Full-raw OpenAI-shaped envelopes captured from real servers on
**2026-09-24** (macOS 26.5.2, arm64, Apple M5 Max) by
`Tests/fixtures/longtail/capture_local.py` and `capture_cloud.py`.

These fixtures are the evidence gate for Phase 2 (ADR-179): the parser
widenings (Task 4), preset allowances (Task 5), and the custom-family swap
(Task 6) may only tolerate what is proven here.

## Captured

| Server | Version | Model | Fixture |
|---|---|---|---|
| ollama | 0.34.4 (Homebrew) | `qwen2.5:0.5b` | `ollama.json` |
| llama-server (llama.cpp) | 0.5.0 (Homebrew, build `b11146`) | qwen2.5-0.5b gguf (ollama blob) | `llama-server.json` |

Each fixture records the FULL RAW bodies — never normalized — of three
rounds (plain chat; one-tool round with `tool_choice: "auto"`; streamed
chat as the ordered SSE `data` payloads including `[DONE]`), plus
`captured_at` and a sanitized `capture_cmd` replay template. Local
fixtures have `models_response: null` (the `GET /models` round is
cloud-only by spec).

llama-server was pointed at the gguf blob already on disk from the ollama
pull (`~/.ollama/models/blobs/sha256-c539…`; override with `LLAMA_GGUF`).
Both servers ran on isolated loopback ports (8901/8902) and were stopped
after capture; no developer-running instance was touched.

## Skipped, with reasons

- **vLLM** — expected skip on this machine: no CUDA runtime exists
  (Apple M5 Max; `nvidia-smi` absent), and the `vllm/vllm-openai` image is
  CUDA-only. The script attempts vLLM only when `nvidia-smi` demonstrably
  succeeds. **Until a Linux capture lands, any vLLM-specific key
  (`stop_reason` as an extra choice key) is UNVERIFIED MEMORY, NOT
  EVIDENCE** and must be treated as provisional in later tasks.
- **LM Studio** — GUI application, unscripted by design; llama-server
  (captured above) represents the server family for evidence purposes.
- **Together / Cerebras / Fireworks** — clean skip: no
  `TOGETHER_API_KEY` / `CEREBRAS_API_KEY` / `FIREWORKS_API_KEY` in the
  capture environment. No cloud fixture exists yet; provider-specific
  allowances derived without a fixture remain provisional (Task 5).

## Headline findings (see the inventory test for the pinned sets)

- **llama-server** sends a `timings` object at the top level of
  non-streaming bodies and on the terminal stream event — an unknown key
  at both the body and stream-event closed levels.
- **ollama** (0.34.4, qwen2.5:0.5b) is clean at every closed allowlist
  level: empty inventory is the pinned, expected outcome.

## Baseline replay under the current strict parser

`test_longtail_fixture_characterization.py`'s replay helpers (Task 4
flips these to acceptance assertions) produce, today:

- ollama plain body: parses to a turn.
- ollama tool-call body: protocol error — its `tool_calls` entries carry
  an `index` member (`id`/`index`/`type`/`function`) the strict tool-call
  shape rejects (a tool-shape finding, not a closed-level key).
- ollama stream: protocol error — terminal event carries
  `finish_reason: "stop"` but **no `usage` anywhere** before `[DONE]`.
- llama-server (all three rounds): protocol error — `timings` outside the
  closed top/event allowlists.

## Replay / re-capture

Re-capture against fresh local servers (starts/stops each server itself):

```bash
python3 Tests/fixtures/longtail/capture_local.py
```

Cloud (writes only the fixtures whose key is present):

```bash
TOGETHER_API_KEY=<key> CEREBRAS_API_KEY=<key> FIREWORKS_API_KEY=<key> \
  python3 Tests/fixtures/longtail/capture_cloud.py
```

Single-round sanitized replay templates are stored per fixture in the
`capture_cmd` field (credentials appear only as the literal `<key>`
placeholder; the scripts build real headers at call time and never store
them). The committed evidence is pinned by
`Tests/LLM_Calls/test_longtail_fixture_characterization.py`.
