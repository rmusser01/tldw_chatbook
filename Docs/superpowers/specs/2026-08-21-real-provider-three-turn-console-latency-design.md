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
samples per arm despite an expected 50–70 minute inference run once the tool-call
round trips and warmups are included.

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
full hashes. The legacy control is exactly
`5f720a40417eaa78f33619d5cbc82effc470104b`, the dev commit on which the recovery
branch was based; the candidate is the current branch HEAD. A preflight must prove the
control still has default-on tracking and synchronous end-snapshot ownership, and must
prove the candidate's disabled/enabled consent and asynchronous finalization
fingerprints. A detached control worktree and the candidate hash remain immutable for
the complete run; moving refs cannot change either target.

Each measured sample is one fresh mounted Console conversation with three sends made
through the real composer/send action. Turn three is not submitted after awaiting turn
two. A content-free wrapper around the durable turn-two assistant transition schedules
the third composer action immediately, while turn two still owns the accepted live-turn
slot. The sample records `third_send_requested` and is valid only when
`third_send_requested < turn_2_release`; tracked-arm samples additionally require the
third send to overlap the real end-snapshot/finalization interval. No harness barrier
delays release to manufacture this ordering.

1. **Control / legacy tracked mutation** — pinned dev, its legacy Change Review path,
   and the same turn-two workspace mutation used by the candidate.
2. **Candidate / review disabled mutation** — candidate code, explicit per-workspace
   disabled consent, and the same mutation. This measures the default no-review path.
3. **Candidate / review enabled mutation** — candidate code, explicit enabled consent
   brought to ready state before timing, and the same mutation. This measures the
   asynchronous finalization path.

All arms begin from the same generated corpus manifest: 1,024 fixed 4 KiB files plus
one fixed 8 MiB tracked blob. Before timing, both tracked arms complete and verify the
same untimed initial snapshot. The disabled arm verifies that no snapshot repository or
operation was created. Turn two then performs the one identical confined mutation.

Every arm runs one untimed warmup and thirty measured samples. Arm order rotates by
iteration (`control, disabled, enabled`; then `disabled, enabled, control`; then
`enabled, control, disabled`) so server temperature, caches, and background host load
cannot systematically favor one arm.

## Provider and tool preflight

The run begins with five fail-closed checks:

1. `GET /v1/models` must identify the expected local model.
2. A fixed temperature-zero short completion must succeed without credentials.
3. A fixed tool request must produce exactly one valid call for the selected confined
   `fs_write` tool, and exactly one follow-up model request must consume the successful
   real tool result.
4. The mutation must change only `measured/turn-two.txt` below the sample's scratch
   workspace to fixed synthetic bytes. Turns one and three must produce no mutating
   calls, and no other tool call or write is permitted.
5. The isolated permission store must contain an `allow` decision for the exact live
   `fs_write` definition/schema hash before timing; every arm must prove the same
   effective permission and must not open an approval flow.

The tool-call turn may use a larger fixed output cap than the plain turns. Exact
per-turn prompts, tool schemas, sampling values, output caps, response token counts,
and model identity are recorded. Synthetic prompt fixtures are published in the runner
source; raw evidence stores only fixture IDs and hashes. If the model cannot reliably
complete this exact contract, the run stops as an invalid benchmark instead of
substituting a fake tool call.

## Harness architecture

`Tests/Performance/run_console_three_turn_profile.py` is a standalone parent/child
runner. Parent mode owns revision pinning, detached control-worktree lifecycle,
balanced ordering, report validation, and summary construction. Child mode executes
one sample in a fresh process.

Before an interpreter imports any target module, the parent supplies a sanitized child
environment with sample-scoped home/config/data/cache/temp paths and no inherited cloud
credentials. The bootstrap asserts those values, then inserts the explicit target root
before every other repository path on `sys.path`. It asserts that the imported
`tldw_chatbook` package, Console screen, and test harness helpers all resolve below
that target. This prevents candidate modules from contaminating control samples and
prevents import-time settings from touching the user's profile.

The child uses the target revision's production-shaped `ConsoleHarness` and test-app
builder, but replaces only external ownership:

- a fresh ChaChaNotes database;
- a fresh workspace registry database;
- a fresh generated workspace and shadow-repository directory;
- an explicit local llama.cpp provider configuration for port 9099;
- fixed synthetic prompts and mutation content;
- timing wrappers around existing provider, prompt-coordinator, persistence, and
  finalization seams.

The child explicitly owns and drains every test-app temporary directory, worker,
provider client, database connection, and detached repository. After each sample, the
parent inventories created files and fails if anything was written outside the sample
and run roots.

Wrappers record timestamps and delegate immediately; they do not sleep, reorder, or
change return values. Version-specific discovery is contained in a small target
adapter with an arm-specific schema. Missing seams fail only when required for that
arm: disabled review events are prohibited, the legacy arm requires synchronous
baseline and end-snapshot boundaries, and candidate enabled requires
schedule/start/completion boundaries.

## Timestamp model

All timestamps use `time.perf_counter_ns()` in one child process and are emitted as
monotonic deltas. Each provider boundary is keyed by `(turn, request_round)` because
turn two must contain one tool-call round and one post-tool-result assistant round.
Each raw sample contains:

- composer commit / send requested;
- prompt admitted or queued;
- worker scheduled and started;
- every provider request-round start;
- every request-round first provider chunk;
- every request-round stream completion;
- the terminal assistant round that triggers the turn-three composer action;
- assistant message made durable;
- third composer send requested;
- terminal result returned to the prompt coordinator;
- accepted live-turn slot released;
- next queued prompt claimed;
- arm-specific review events: prohibited for disabled; legacy baseline start/ready and
  synchronous end start/end; candidate baseline start/ready and finalization
  scheduled/started/completed;
- third-turn provider request started.

Derived metrics are:

- send-to-worker;
- worker-to-provider;
- per-round provider start-to-first-token and first-token-to-complete;
- aggregate provider time per turn, excluding the real local-tool execution interval;
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
new home, `config`, `data`, database, workspace, shadow repository, cache, and temp
subdirectories. The sanitized environment is installed before Python imports target
code. The child verifies the effective paths before import and after teardown; the
parent compares an allowlisted write inventory and rejects any path outside the sample
or run roots.

Raw evidence contains synthetic arm/sample identifiers, relative workspace aliases,
timestamps, counts, status categories, model identity, token usage, and response
lengths. It does not retain prompt/response bodies, absolute workstation paths, API
keys, headers, environment dumps, tool-result contents, or generated file contents.
Paths in exceptions and stacks are normalized to `$CONTROL`, `$CANDIDATE`, `$RUN`,
`$VENV`, and `$HOME` before they are flushed.

Each JSONL event is written and flushed immediately. Provider request rounds have a
120-second deadline, turns a 300-second deadline, and a complete sample a 900-second
deadline. The parent watchdog sends TERM to an overdue child, waits five seconds, then
sends KILL and performs bounded descendant/worktree cleanup. A killed or timed-out run
retains its last completed boundary and failure category. Summary generation refuses
partial samples unless they are explicitly counted as failures.

The run manifest records Python, Textual, pytest, OS, architecture, CPU count, load
average, target hashes, model endpoint/model metadata, sanitized llama-server build and
runtime properties, context/concurrency settings, and listener-process RSS/CPU at each
sample boundary. Absolute model paths and command-line secrets are never retained.

## Statistics and decision rules

The report retains all thirty samples per arm and reports median and nearest-rank p95;
minimum and best-case values are never used for a decision. The rotated iteration is
the pairing block. A deterministic 10,000-resample paired block bootstrap produces a
two-sided 95% interval plus one-sided 95% upper and lower bounds for every
candidate/control p95 ratio. It also reports failures, prompt-loss count,
tool-contract failures, response-token distribution, and provider-latency dispersion
as descriptive context.

The result is valid only when:

- all ninety samples reach the third provider and terminal third assistant;
- every third send is requested before turn-two release, and tracked-arm samples
  overlap the real end-snapshot/finalization interval;
- every turn two has exactly two provider rounds, exactly one allowed `fs_write`, one
  successful confined mutation, and one model follow-up consuming its result;
- the disabled arm performs no Change Review snapshot work;
- both tracked arms complete equivalent initial snapshots before measured sends;
- every boundary required by that arm's schema exists and every prohibited boundary is
  absent;
- revision/import-path/isolation/privacy validations pass.

Both candidate arms pass a send-to-worker or event-loop-lag non-regression gate only
when the paired confidence interval's upper bound for the p95 ratio is at most 1.10.
An arm is a measured regression only when the lower bound is above 1.10; otherwise the
metric is `inconclusive`. The report may claim critical-path improvement only from
application-owned intervals—assistant-durable to turn release and terminal to third
worker/provider—and only when the paired ratio interval's upper bound is below 1.00.
Total conversation wall time is reported but cannot prove the app became faster
because it is dominated by model inference.

If the provider becomes unavailable, violates the exact tool contract, or any
completeness/isolation requirement fails, the overall result is `invalid` and evidence
is preserved. A complete metric whose confidence interval crosses its decision
boundary is `inconclusive`. There is no subjective variance override and an incomplete
real-provider run can never become a pass.

## Testing and review

Test-driven implementation begins with unit tests for:

- balanced arm rotation;
- nearest-rank p95 and median summaries;
- deterministic paired block-bootstrap confidence bounds;
- exact target-root import validation;
- arm-specific required/prohibited boundaries, per-round provider accounting,
  third-send overlap, exact tool contract, and ninety-sample completeness validation;
- failure-preserving JSONL writes;
- absolute-path and credential-field rejection;
- fixed watchdog/termination behavior;
- decision rules with confidence intervals wholly below, crossing, and wholly above
  the 10% boundary.

One three-arm, one-sample smoke validates the mounted path before the long run. The
final thirty-sample evidence is independently recomputed from raw JSONL, privacy
scanned, and reviewed before TASK-19641 can close.

## ADR check

ADR required: no.

This design adds opt-in benchmark tooling and retained evidence. It changes no storage
schema, runtime boundary, provider contract, ownership model, or long-lived UX. ADR-077
remains the governing decision for Change Review consent and asynchronous
finalization.
