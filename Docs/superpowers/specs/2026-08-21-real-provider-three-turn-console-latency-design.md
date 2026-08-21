# Real-provider three-turn Console latency design

Date: 2026-08-21
Task: TASK-19641
Status: proposed

## Context

The Change Review recovery branch has a deterministic mounted regression proving
that turn three reaches its provider while turn two's end snapshot is deliberately
held. That is strong ordering evidence, but it is not a speed measurement. Console
mount profiling also cannot answer this question because it stops at the worker-launch
seam and never performs a real model request or workspace mutation.

The user selected a real-provider comparison rather than a scripted gateway. The
available provider is an OpenAI-compatible `llama-server` on
`http://127.0.0.1:9099`, serving
`gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`. A one-token preflight
completed successfully in 7.90 seconds. The benchmark therefore uses thirty balanced
samples per arm despite an expected 40–50 minute inference run.

## Goals

1. Measure the original three-turn failure path through the mounted production
   Console, real provider gateway, prompt coordinator, durable assistant anchor, real
   local tool mutation, and Change Review finalization.
2. Separate model/network time from application-owned admission, persistence, queue
   release, review, and event-loop latency.
3. Compare the exact legacy dev behavior with Change Review disabled and enabled on
   the candidate branch.
4. Retain raw, continuously flushed evidence from which every summary value can be
   independently recomputed.

## Non-goals

- The benchmark does not tune prompts, model sampling, llama.cpp, GPU placement, or
  token throughput.
- It does not introduce production telemetry callbacks or change runtime behavior.
- It does not use cloud providers, paid requests, personal conversations, the user's
  real workspace, or the user's retained shadow repositories.
- It does not claim that a noisy real-provider result is a regression or improvement
  unless the application-owned intervals and completeness gates support that claim.

## Compared arms

The parent process pins two immutable revisions before measurement and records their
full hashes. The control is a detached worktree at the then-current `origin/dev`; the
candidate is the current branch HEAD. Moving refs during the run cannot change either
target.

Each measured sample is one fresh mounted Console conversation with three sends:

1. **Control / legacy tracked mutation** — pinned dev, its legacy Change Review path,
   and the same turn-two workspace mutation used by the candidate.
2. **Candidate / review disabled mutation** — candidate code, explicit per-workspace
   disabled consent, and the same mutation. This measures the default no-review path.
3. **Candidate / review enabled mutation** — candidate code, explicit enabled consent
   brought to ready state before timing, and the same mutation. This measures the
   asynchronous finalization path.

Every arm runs one untimed warmup and thirty measured samples. Arm order rotates by
iteration (`control, disabled, enabled`; then `disabled, enabled, control`; then
`enabled, control, disabled`) so server temperature, caches, and background host load
cannot systematically favor one arm.

## Provider and tool preflight

The run begins with four fail-closed checks:

1. `GET /v1/models` must identify the expected local model.
2. A fixed temperature-zero short completion must succeed without credentials.
3. A fixed tool request must produce a valid call for the selected confined local
   filesystem mutation tool, and the real tool result must be consumed by the model.
4. The mutation must change only a generated file below the sample's scratch
   workspace; the path and bytes are fixed synthetic values.

The tool-call turn may use a larger fixed output cap than the plain turns. Exact
per-turn prompts, tool schemas, sampling values, output caps, response token counts,
and model identity are recorded. If the model cannot reliably complete this contract,
the run stops as an invalid benchmark instead of substituting a fake tool call.

## Harness architecture

`Tests/Performance/run_console_three_turn_profile.py` is a standalone parent/child
runner. Parent mode owns revision pinning, detached control-worktree lifecycle,
balanced ordering, report validation, and summary construction. Child mode executes
one sample in a fresh process.

The child receives an explicit target root and inserts that root before every other
repository path on `sys.path`. It asserts that the imported `tldw_chatbook` package,
Console screen, and test harness helpers all resolve below that target. This prevents
candidate modules from contaminating control samples.

The child uses the target revision's production-shaped `ConsoleHarness` and test-app
builder, but replaces only external ownership:

- a fresh ChaChaNotes database;
- a fresh workspace registry database;
- a fresh generated workspace and shadow-repository directory;
- an explicit local llama.cpp provider configuration for port 9099;
- fixed synthetic prompts and mutation content;
- timing wrappers around existing provider, prompt-coordinator, persistence, and
  finalization seams.

Wrappers record timestamps and delegate immediately; they do not sleep, reorder, or
change return values. Version-specific discovery is contained in a small target
adapter that detects the legacy tracker or candidate coordinator. Missing seams are a
hard compatibility failure, never silently omitted data.

## Timestamp model

All timestamps use `time.perf_counter_ns()` in one child process and are emitted as
monotonic deltas. Each raw sample contains the following application/provider
boundaries for all three turns:

- composer commit / send requested;
- prompt admitted or queued;
- worker scheduled and started;
- provider request started;
- first provider chunk received;
- provider stream completed;
- assistant message made durable;
- terminal result returned to the prompt coordinator;
- accepted live-turn slot released;
- next queued prompt claimed;
- Change Review baseline started/ready;
- Change Review finalization scheduled/started/completed;
- third-turn provider request started.

Derived metrics are:

- send-to-worker;
- worker-to-provider;
- provider start-to-first-token;
- first-token-to-provider-complete;
- provider-complete-to-assistant-durable;
- assistant-durable-to-turn-release;
- terminal-to-third-worker and terminal-to-third-provider;
- baseline wait and finalization duration;
- complete three-turn wall time.

A 10 ms heartbeat runs on the Textual event loop from the first send through final
settlement. Raw tick lateness is retained; median and p95 event-loop lag are derived
without including process startup or the untimed warmup.

## Isolation and privacy

The parent creates one run directory under an explicit output root. Every sample gets
new `config`, `data`, database, workspace, shadow repository, and cache subdirectories.
The child sets the repository's supported profile/data overrides and verifies after
each sample that no configured path resolved to the user's normal data directories.

Raw evidence contains synthetic arm/sample identifiers, relative workspace aliases,
timestamps, counts, status categories, model identity, token usage, and response
lengths. It does not retain prompt/response bodies, absolute workstation paths, API
keys, headers, environment dumps, tool-result contents, or generated file contents.
Paths in exceptions and stacks are normalized to `$CONTROL`, `$CANDIDATE`, `$RUN`,
`$VENV`, and `$HOME` before they are flushed.

Each JSONL event is written and flushed immediately. A killed or timed-out run retains
its last completed boundary and failure category. Summary generation refuses partial
samples unless they are explicitly counted as failures.

## Statistics and decision rules

The report retains all thirty samples per arm and reports median and nearest-rank p95;
minimum and best-case values are never used for a decision. It also reports failures,
prompt-loss count, tool-contract failures, response-token distribution, and the
provider-latency coefficient of variation so model noise is visible.

The result is valid only when:

- all ninety samples reach the third provider and terminal third assistant;
- all turn-two mutations are confined and observed;
- the disabled arm performs no Change Review snapshot work;
- enabled candidate baselines are ready before measured sends;
- every expected boundary exists in every sample;
- revision/import-path/isolation/privacy validations pass.

The candidate passes the non-regression gate when send-to-worker and event-loop-lag
p95 are no more than 10% worse than the applicable control. The report may claim
critical-path improvement only from application-owned intervals: assistant-durable to
turn release and terminal to third worker/provider. Total conversation wall time is
reported but cannot prove the app became faster because it is dominated by model
inference.

If the provider becomes unavailable, emits inconsistent tool calls, or produces enough
variance to invalidate comparison, the report says `inconclusive` and preserves the
evidence. It never converts an incomplete real-provider run into a pass.

## Testing and review

Test-driven implementation begins with unit tests for:

- balanced arm rotation;
- nearest-rank p95 and median summaries;
- exact target-root import validation;
- required-boundary and ninety-sample completeness validation;
- failure-preserving JSONL writes;
- absolute-path and credential-field rejection;
- decision rules at exact and just-over 10% boundaries.

One three-arm, one-sample smoke validates the mounted path before the long run. The
final thirty-sample evidence is independently recomputed from raw JSONL, privacy
scanned, and reviewed before TASK-19641 can close.

## ADR check

ADR required: no.

This design adds opt-in benchmark tooling and retained evidence. It changes no storage
schema, runtime boundary, provider contract, ownership model, or long-lived UX. ADR-077
remains the governing decision for Change Review consent and asynchronous
finalization.
