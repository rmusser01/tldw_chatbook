# Project-chat UAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline. Steps use checkbox syntax for tracking.

**Goal:** Deliver the reviewed workflow catalog and four deterministic real-app project-chat variants that cannot pass without actual bound-project evidence.

**Architecture:** An ordinary pytest module uses the existing private-profile child runner and starts TldwCli normally. A small stdlib loopback model server scripts only HTTP responses; the real UI, provider adapter, registry, permissions, file tools, and databases execute the journey. Independent assertions and negative controls validate the final answer and provenance.

**Tech Stack:** Python >=3.12, existing pytest/pytest-asyncio/pytest-timeout, Textual 8.x, stdlib HTTP/threading/JSON/SQLite.

**Spec:** Docs/superpowers/specs/2026-10-28-critical-journey-uat-design.md
**Task:** TASK-37270
**ADR required:** no; test composition under ADR-069/174/241/253, no product boundary changes.

## Global Constraints

- Catalog rows other than implemented pilot variants remain explicitly planned.
- No real user profile, secrets, external model, model download, or unrestricted sockets.
- Real startup and visible UI actions; no app factory, forced ready state, controller injection, or manually seeded workspace/chat.
- Four variants: fenced, native, compact recovery, wrong-family listing recovery.
- A fresh random README code is unavailable to the model script until a real tool result supplies it; alias also comes from actual outbound context/schema.
- Targeted verification only; no full repository suite. Preserve current CI budgets.
- Windows/Ubuntu scripted qualification cannot be described as live Fedora qualification.

## Review Focus

- Code only in a tool card/history must fail final-answer checks (Task 1 oracle tests).
- Implicit model fallback or schema omission must fail protocol checks (Task 1 script tests and Task 2 native run).
- Default localhost probes, inherited proxies/credentials, and orphaned workers must not escape the fixture (Tasks 1/2 isolation checks).
- Successful empty scratch/failed README reads must turn the journey red (Tasks 1/2 negative controls).
- Skipped variants, missing receipts, and an overfull CI lane must not produce a false green (Task 3 report tests and measured enrollment).

### Task 1: Minimal provider fixture and evidence oracle

**Files:** create `Tests/uat_project_chat_support.py`; create `Tests/QA/test_project_chat_uat_support.py`.
**Interfaces:** `ProjectChatModel(variant)` consumes only HTTP request payloads and produces OpenAI-compatible scripted messages; `project_chat_server(variant)` owns a numeric loopback listener; `assert_project_evidence(...)` checks expected fixture evidence against independent final message and trace inputs. Keep the functions small; no generic journey DSL.

- [x] Write support tests: reject final text without code even when a tool result has it; reject empty/error read and wrong-root provenance; require actual native schemas; reject unexpected protocol steps; metadata probes cannot consume a generation step.
- [x] Run the focused support tests and observe the intended missing-behavior failures.
- [x] Implement the smallest provider script, bounded HTTP lifecycle, and independent oracle. Keep fixture expected values out of the script.
- [x] Test the owned-port network boundary and clean shutdown; record attempt errors before raising.
- [x] Run the focused support file and scoped lint/format checks.

### Task 2: Real app/UI project-chat pilot

**Files:** create `Tests/QA/test_project_chat_uat.py`; use `Tests/private_profile.py` without broadening its shared behavior.
**Interfaces:** `test_project_chat_uat(variant, request, tmp_path, monkeypatch)` runs in a private-profile child; evidence artifacts are case-specific JSON plus viewport SVG/text in a caller-selected artifact directory.

- [x] Read the created Backlog task and existing authority/startup contracts. Save a preconfigured disposable profile before importing TldwCli; sanitize inherited environment in the parent before the child is launched.
- [x] Start the owned HTTP server, instantiate TldwCli, and await actual Console readiness. Use supported config for optional background work and all permission preconditions.
- [x] Drive visible workspace create/name/folder add/confirm controls, disable the optional interview visibly, create a scoped chat, type the question, click Send, and accept the first-use AGENTS.md disclosure through its visible Proceed button.
- [x] Observe each variant's actual provider schema/result sequence, final owning assistant message, real binding and saved run. Assert the actual rendered assistant answer separately from tool cards; export failure receipts even when a wait fails.
- [x] Run fenced first under the CI dependency venv; repair setup mistakes without replacing production boundaries. Add native, compact recovery, and wrong-family variants after the first path works.
- [x] Run a full negative control with deliberately missing README evidence; require the pilot to fail at the evidence stage, then run unmodified variants green.
- [x] Verify shutdown and artifact retention. Record non-app test substitutions and phase timings.

### Task 3: Gate enrollment, diagnostics, and handoff

**Files:** modify the suitable existing job in `.github/workflows/derived-artifacts.yml`; create a small report verifier only if no existing verifier handles exact variants/receipts; update corresponding `Tests/CI` contract tests and the catalog with commands/evidence.

- [x] Write failing checks for missing/skipped variant and missing receipt if introducing report validation; implement only the exact pilot report contract.
- [x] Measure pilot duration with the lane's dependency set; enroll it without increasing timeouts or disturbing existing shard ordering. Prefer a bounded step in an existing job with measured headroom.
- [x] Ensure `if: always()` uploads synthetic receipts and JUnit even when tests fail; empty or partial reports fail validation.
- [x] Run targeted CI-contract checks, pilot/support tests, changed-file lint/format, and diff hygiene. Run Linux qualification through available existing CI only; state any unperformed environment explicitly.
- [x] Independently review the final branch, repair concrete defects with targeted evidence, and update catalog/task notes with exact node IDs, commands, durations, and limitations. Do not mark the task Done until ACs are actually satisfied.

## Execution record

- 2026-10-28: Design approved; user requested deeper review before implementation. Eight concrete design corrections recorded in the spec. No implementation or passing pilot evidence yet.
- Inline execution selected to keep this small pilot in one context. Existing worktree/branch `codex/uat-project-chat` reused from merged PR; no new checkout needed.

- Qualification found an omitted user action: first-use AGENTS.md disclosure must be accepted before provider entry. Linux run 40801165548 captured the real runtime waiting for that decision and the visible modal. The pilot now asserts its session/source, proves no generation happened before consent, and clicks Proceed. No production consent gate is bypassed.

- Run 40802492563 completed the real fenced read and all three answer views, then exposed an invalid oracle assumption: AgentRunsDB deliberately omits tool arguments/file bodies. Correlate its call IDs/outcome with actual provider history instead. Updated native/fenced private-metadata controls: 2 RED; 56 support/report/enrollment checks GREEN (8.71s). Preserve synthetic evidence before assertions for diagnostic failures.

- Run 40803962773 again passed fenced/native/compact but stopped the wrong-family case at the built-in always-ask approval card. Enable its normal optional-tool config and click the sole list_directory Approve once control; assert the exact pending script call before approval. The real high-risk permission floor remains intact.

- Current targeted validation: 147 CI enrollment/workflow contracts passed in 258.78s on Windows Python 3.12; all eight changed Python files pass Ruff and six new Python files pass format checks. No full repository test sweep was run.

- Qualification at 4c0ce64d6c: Linux run 40805071773 passed all four actual-app cases plus 55 controls (59 passed, 171.593s); the real missing-evidence negative control failed for the intended fs_read/code reason (37.802s) and its verifier passed. Both downloaded report sets pass independently. Full existing CI suites still running; keep the draft and task In Progress until their verdict. Windows is explicitly failed qualification; Fedora/Python 3.14/live model/native terminal are not run. The spec records exact node IDs, timings, commands, and artifacts.

## Authorized follow-up: Windows, streaming, and Strata/PolyStrata

ADR required: no.
ADR path: N/A; direct implementation of ADR-002 (compatible model discovery),
ADR-006 (provider-owned parameter mapping), ADR-012 (credentials), and ADR-179
(provider identity), with existing storage/file authority unchanged.
Reason: add independent external HTTP provider identities through established
registration/configuration/dispatch seams; no engine process management or new
storage/runtime protocol. The heartbeat correction restores the existing SSE
content boundary.

1. Sample the unchanged Windows journey and identify the actual pending wait.
   Keep native path/ACL checks and real startup. Reproduce the cause narrowly,
   fix it at its owner, and qualify all four variants with measured bounded deadlines.
2. Reproduce comment/event/id/retry SSE leakage on the real llama_cpp native-tools
   adapter-to-Console route. Relay data frames only; preserve actual prose,
   reasoning, tools, errors, termination and response ownership.
3. Register separate strata/polystrata provider identities, optional credentials,
   configured URLs/model discovery, actual model context metadata, and metadata-gated
   native tools. Reuse the existing compatible transport with provider_name.
   Keep these identities out of llama.cpp's direct-execution/template assumptions;
   use the upstream reasoning_effort/reasoning_budget_tokens wire parameters.
   No engine installation, model downloads, template trust promotion, or UI layout
   redesign. Add focused configuration/discovery/dispatch/native/stream regressions.
4. Review concrete feedback, validate changes with targeted suites, update measured
   evidence, and keep the PR open until all required checks pass on the final head.

Primary API references verified 2026-10-29 UTC:
- Niko1221/Strata at 46c152a1310f77cd2f0ce2c7cd213962df14fe83.
- VecSzn/PolyStrata at 504ff71d310a4a4d9dfcf9176978f68110814969.
Both use /v1/chat/completions, /v1/models, /health and /props; their OpenAI stream
keepalive is an SSE comment, `: keep-alive`, rather than assistant content.

- Streaming follow-up fixture correction: the real-provider tests
  `test_stopping_a_hosted_stream_closes_its_response` (7 variants) and
  `test_real_local_dispatcher_preserves_native_schema_and_reasoning` (12 variants)
  enter guarded configuration after the per-test sandbox changes its identity.
  Confirm this on the original SSE guard, then opt these exact nodes into the
  existing bootstrap_profile fixture contract. Preserve all production admission
  checks and the tests' stream/reasoning/Stop assertions.

- Windows follow-up narrowing: the full ancestry scan is quadratic but measured
  at roughly 31 ms per deep-root admission; changing that alone would not explain
  the multi-minute first-send stall. The acquisition condition wakes every 10 ms
  and repeats path validation under the coordinator lock. Replace only timed
  polling with the existing notifications; retain fresh validation on entry and
  wake, provenance checks, recursive initialization refusal, and pause cancellation.
  ADR required: no; synchronization repair within ADR-126's existing authority.
  Cover notification/cancellation and mutation during waiting before qualification.
- Keep the module size ratchets: remove the superseded initial LOCAL_PROVIDERS
  literal, preserving the active TOML-derived public mapping and all existing
  entries. Extract the bounded local metadata fetch/parser from the gateway;
  retain gateway cache/TTL/permission ownership. No raised budgets or unrelated
  whitespace trimming.

- Independent follow-up review found and reproduced missing auxiliary keyless
  authority, tool-round reasoning ownership, and local zero-cost classification
  for the new provider keys. Extend the existing mappings and regressions.
  Do not enroll unrelated legacy thinking tests merely to cover these two nodes.
- Streaming review added an active keep-alive-only Stop negative control: simply
  consuming control lines inside one next() prevents the owner from observing
  cancellation. Empty iteration checkpoints preserve invisible control traffic
  while retaining existing owner-thread cleanup; no cross-module interruption
  protocol or authority/lifetime change is introduced.
- The initial LOCAL_PROVIDERS literal is dead: config later derives the public
  mapping from its parsed provider table. Remove that superseded literal instead
  of introducing a defaults module. Keep the bounded metadata HTTP extraction and
  lower the exact module ratchets. Existing metadata tests await the already
  scheduled background refresh and read fingerprinted fixtures explicitly as UTF-8.

- Timing-budget diagnosis: the monitored shallow Windows run reached five provider
  requests, but its fifth request carried the production wall-clock-exhausted
  instruction instead of the required fs_read result. The pilot had set a 180s
  agent budget, versus the product's 86400s default. At 561s it had completed 2698
  native admissions; no authority-open exceeded 1.45s, while initialization waiting
  comprised about 68% of aggregate admission time. This is forward progress under
  full Windows derivation, not a stuck call or an access refusal.
  Correct the Windows-only functional-test budget to 600s for the agent and each
  send-stage wait (including the built-in approval appearing), bounded by 1200s for the whole case. Preserve the measured
  performance limitation and the notification repair. Linux's 90s agent/240s case
  and existing CI job/step budgets remain unchanged. This replaces the initial
  assumption that the original Windows timers were adequate; no security check,
  evidence oracle, Stop behavior, or product default is weakened.

- Windows actual-run RED: fenced and wrong-family reached the disclosure with a
  different session ID than the pilot captured after clicking New chat. A click
  queues the async handler; it does not establish that the new session exists.
  Wait first for the workspace's active chat/composer binding, snapshot existing
  IDs, then require a distinct newly active session with its composer bound before
  pasting/sending. Keep the original disclosure ownership assertion unchanged.


### Follow-up qualification

- Source head `3660a6caff`: all four Windows journeys passed with complete receipts,
  using the real native authority and approval paths. Body times: fenced 743.657s,
  native 571.203s, compact recovery 493.250s, wrong family 509.734s. The aggregate
  verifier accepted exactly four successful cases. Run sequentially; preserve the
  Windows latency limitation and avoid claims from confounded path/timing comparisons.
- Windows full missing-evidence control: one intended failure, no errors/skips,
  399.400s parent / 387.407s body; verifier confirmed the actual altered README
  reached the fifth provider request through fs_read before rejection.
- Linux run 40817423013: 59 passed in 183.711s; negative control one intended failure
  in 38.888s. Both downloaded evidence verifiers passed. The same lane's newly
  enrolled provider/streaming/reasoning regressions also passed.
- Final independent source review found no actionable issues. Qodo could not review
  due to workspace credits and CodeRabbit auto-review skips the dev target; neither
  is treated as external approval. Required final-head CI remains the closeout gate.

### Required inventory follow-up

Run 40817423013 passed every source test lane but failed the persistent-diagnostic
inventory: console_provider_support had one new raw rejected-effort debug message.
Statement-level review against the pin identified that sole addition. Commit
133b18c2cf removes the unnecessary log; two privacy regressions failed before the
fix and the full 48-case Strata file passed afterward. The file's five diagnostic
statements exactly match the pin again, without regenerating the inventory. This
changes no valid reasoning payload or UAT path. Final committed-head CI remains
required before task closeout.

Source closeout: required aggregate and full run 40822404188 passed on
133b18c2cf3f029fc248987ca4a2108c0b4d6b13. All acceptance criteria are verified;
TASK-37270 is Done. The documentation-only closeout retains required CI, and no
merge is authorized.
