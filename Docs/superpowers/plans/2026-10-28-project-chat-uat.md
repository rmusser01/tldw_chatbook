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

- [ ] Write support tests: reject final text without code even when a tool result has it; reject empty/error read and wrong-root provenance; require actual native schemas; reject unexpected protocol steps; metadata probes cannot consume a generation step.
- [ ] Run the focused support tests and observe the intended missing-behavior failures.
- [ ] Implement the smallest provider script, bounded HTTP lifecycle, and independent oracle. Keep fixture expected values out of the script.
- [ ] Test the owned-port network boundary and clean shutdown; record attempt errors before raising.
- [ ] Run the focused support file and scoped lint/format checks.

### Task 2: Real app/UI project-chat pilot

**Files:** create `Tests/QA/test_project_chat_uat.py`; use `Tests/private_profile.py` without broadening its shared behavior.
**Interfaces:** `test_project_chat_uat(variant, request, tmp_path, monkeypatch)` runs in a private-profile child; evidence artifacts are case-specific JSON plus viewport SVG/text in a caller-selected artifact directory.

- [ ] Read the created Backlog task and existing authority/startup contracts. Save a preconfigured disposable profile before importing TldwCli; sanitize inherited environment in the parent before the child is launched.
- [ ] Start the owned HTTP server, instantiate TldwCli, and await actual Console readiness. Use supported config for optional background work and all permission preconditions.
- [ ] Drive visible workspace create/name/folder add/confirm controls, disable the optional interview visibly, create a scoped chat, type the question, click Send, and accept the first-use AGENTS.md disclosure through its visible Proceed button.
- [ ] Observe each variant's actual provider schema/result sequence, final owning assistant message, real binding and saved run. Assert the actual rendered assistant answer separately from tool cards; export failure receipts even when a wait fails.
- [ ] Run fenced first under the CI dependency venv; repair setup mistakes without replacing production boundaries. Add native, compact recovery, and wrong-family variants after the first path works.
- [ ] Run a full negative control with deliberately missing README evidence; require the pilot to fail at the evidence stage, then run unmodified variants green.
- [ ] Verify shutdown and artifact retention. Record non-app test substitutions and phase timings.

### Task 3: Gate enrollment, diagnostics, and handoff

**Files:** modify the suitable existing job in `.github/workflows/derived-artifacts.yml`; create a small report verifier only if no existing verifier handles exact variants/receipts; update corresponding `Tests/CI` contract tests and the catalog with commands/evidence.

- [ ] Write failing checks for missing/skipped variant and missing receipt if introducing report validation; implement only the exact pilot report contract.
- [ ] Measure pilot duration with the lane's dependency set; enroll it without increasing timeouts or disturbing existing shard ordering. Prefer a bounded step in an existing job with measured headroom.
- [ ] Ensure `if: always()` uploads synthetic receipts and JUnit even when tests fail; empty or partial reports fail validation.
- [ ] Run targeted CI-contract checks, pilot/support tests, changed-file lint/format, and diff hygiene. Run Linux qualification through available existing CI only; state any unperformed environment explicitly.
- [ ] Independently review the final branch, repair concrete defects with targeted evidence, and update catalog/task notes with exact node IDs, commands, durations, and limitations. Do not mark the task Done until ACs are actually satisfied.

## Execution record

- 2026-10-28: Design approved; user requested deeper review before implementation. Eight concrete design corrections recorded in the spec. No implementation or passing pilot evidence yet.
- Inline execution selected to keep this small pilot in one context. Existing worktree/branch `codex/uat-project-chat` reused from merged PR; no new checkout needed.

- Qualification found an omitted user action: first-use AGENTS.md disclosure must be accepted before provider entry. Linux run 40801165548 captured the real runtime waiting for that decision and the visible modal. The pilot now asserts its session/source, proves no generation happened before consent, and clicks Proceed. No production consent gate is bypassed.

- Run 40802492563 completed the real fenced read and all three answer views, then exposed an invalid oracle assumption: AgentRunsDB deliberately omits tool arguments/file bodies. Correlate its call IDs/outcome with actual provider history instead. Updated native/fenced private-metadata controls: 2 RED; 56 support/report/enrollment checks GREEN (8.71s). Preserve synthetic evidence before assertions for diagnostic failures.

- Run 40803962773 again passed fenced/native/compact but stopped the wrong-family case at the built-in always-ask approval card. Enable its normal optional-tool config and click the sole list_directory Approve once control; assert the exact pending script call before approval. The real high-risk permission floor remains intact.
