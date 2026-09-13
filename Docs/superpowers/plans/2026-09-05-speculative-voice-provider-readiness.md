# Speculative Voice Provider Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject unavailable-provider voice entry before audio starts and clearly explain later request-preparation failures without losing drafts.

**Architecture:** Reuse ConsoleChatController's bounded provider resolver and the existing view-generation/owner-loop boundaries. Fence readiness with existing store activation and settings revisions, session identity, current selection and runtime config generation. Use a small app-owned typed preparation error and fixed UI/diagnostic categories; do not add retries, failover, services, or audio policy.

**Tech Stack:** Python 3.12, asyncio, Textual, pytest, existing Console runtime.

Spec: `Docs/superpowers/specs/2026-09-05-speculative-voice-provider-readiness-design.md` (approved).
Task: TASK-23175, In Progress.

ADR required: no new ADR
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
Reason: Routine readiness and error-reporting correction within existing authority, controller, UI bridge and audio lifecycle boundaries.

## Constraints and file ownership

Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/speculative-duplex-voice-adr097`, branch `codex/speculative-duplex-voice-adr097`. Preserve every pre-existing dirty trace-ledger/CSS file. Do not run physical audio, generation requests, qualification, route switching, repetitions, soaks or full pytest. Use the main repository `.venv/bin/python` and `.venv/bin/ruff`. Give each pytest run a fresh `--basetemp` from `mktemp -d /private/tmp/tldw-voice-preflight-test.XXXXXX`; the shared pytest retention directory contains unrelated permission-protected old outputs and must not be cleaned up.

Implementation unit (one worker because the lifecycle and failure delivery are coupled):
- `tldw_chatbook/Chat/console_voice_preflight.py` (new, only if a shared typed error/readiness snapshot needs a home).
- `tldw_chatbook/Chat/console_chat_controller.py`: bounded read-only readiness and app-owned preparation failure categories.
- `tldw_chatbook/config.py`: a content-free monotonic generation read, if needed to fence the initial snapshot await without blocking the UI loop.
- `tldw_chatbook/Chat/console_speculative_voice_session.py`: current-attempt failure reporting, safe category diagnostics, existing UI bridge delivery.
- `tldw_chatbook/UI/Console_Modules/hands_free.py`: preflight before factory/worker/STT/audio, late-result ownership, fixed safe notification.
- `Tests/Chat/test_console_voice_preflight.py` and `Tests/UI/test_console_voice_provider_preflight.py` (new focused tests); existing `Tests/UI/test_console_speculative_voice_wiring.py`, `Tests/Chat/test_console_speculative_voice_session.py`, `Tests/integration/test_speculative_voice_pipeline.py`, `Tests/Packaging/test_speculative_voice_development_entry.py` only for directly affected regressions/fixtures.
- `Tests/UI/test_console_voice_native_callback.py`: migrate ready-provider fixtures only for the two copy/late-cleanup notification regressions; do not run other native callback scenarios.

Root integration: source/lint inventories discovered by `rg -n 'console_speculative_voice_session.py|test_console_speculative_voice_wiring.py' scripts Tests/Packaging`; this plan/spec; TASK-23175; relevant live-verification lesson. No edits to the packaged qualification manifest or native artifacts.

## Task 1 — Bounded startup readiness and safe preparation failures

- [x] Baseline: run `../../.venv/bin/python -m pytest -q Tests/UI/test_console_speculative_voice_wiring.py::test_qualified_path_uses_one_view_session_and_runtime_owners`. Passed: 1 test, two existing dependency warnings. Shared pytest retention cleanup also warned after completion; use isolated basetemp for remaining runs.
- [x] Add failing mounted visible-control tests. A blocked or missing-session preflight must leave the switch off, draft unchanged, and factory/worker/STT/audio counters at zero. Ready result must reach the existing speculative session, never legacy. Cover validation exception and timeout with sentinel secret text.
- [x] Add event-controlled race tests for toggle off, teardown, session switch/replacement, provider/model changes and A→B→A. Late success cannot enter; late failure cannot notify/reset a newer session. Avoid sleeps for synchronization and do not call live gateways.
- [x] Run `../../.venv/bin/python -m pytest -q Tests/UI/test_console_voice_provider_preflight.py` and retain the expected failing assertions.
- [x] Implement the smallest controller-side readiness helper: capture active session identity, activation epoch, generation-settings revision, provider selection and config generation; await `_resolve_for_send_bounded(selection)`; validate the stamp before use. Do not call `_capture_and_resolve_turn_execution_context` for startup because library/staged authority is unrelated to availability. Capture runtime config through `get_runtime_config_snapshot` off the UI loop if it can block; use existing `run_if_runtime_config_generation_current` only for a nonblocking final fence. No provider payload preparation occurs in preflight.
- [x] In `_construct_qualified_console_voice`, validate before obtaining `runtime.voice_worker` or calling the session factory. Retain/recheck the readiness stamp after any awaited cold factory load and immediately before construction/entry. Use the existing startup generation to fence off/replacement/teardown. Reject changed selections rather than auto-retrying. Failure/reset copy is app-owned and does not interpolate errors or provider endpoints.
- [x] Add failing controller/effects tests for app-owned session/provider preparation failures, generic unexpected failures, cancellation and stale attempt results. Assert exactly-once draft preservation and one safe explanation for the current failing attempt; no notification for cancelled/stale results. Use real controller preparation with isolated store/gateway where practical, including DeepSeek readiness with a fake response/credential.
- [x] Run `../../.venv/bin/python -m pytest -q Tests/Chat/test_console_voice_preflight.py` and retain expected RED evidence.
- [x] Replace app-owned preparation refusal RuntimeErrors with a narrow typed error carrying only a validated fixed category. Keep unknown exceptions generic. Preserve existing reducer retry suspension and draft ownership. Deliver failure UI via the existing owner-loop bridge, with session/attempt currency checked at delivery, without using coalescible preview delivery for a terminal notice. Never classify arbitrary exception messages or append the transcript a second time.
- [x] Record `attempt_prepare_failed` with fixed error category plus existing exception class; preserve privacy tests with sentinel transcript/credential/exception-message strings. Do not broaden persistent diagnostics fields with free text.
- [x] Run both new files and affected existing voice wiring/session/integration/development-entry files; migrate only fixtures that assumed startup skipped the provider. Do not weaken gate assertions or add a production test bypass.
- [x] Self-review, then independent spec review followed by code-quality review. Resolve actionable issues with regression tests and rerun focused checks before committing only owned files.

## Task 2 — Integration, evidence and handoff

- [x] Update source/lint inventories for any new runtime/test files and this spec/plan as required by the existing inventory policy. Preserve all qualification flags and historical hashes.
- [x] Run exact joined targeted tests for new files, affected existing files, source-manifest/lint inventories and guarded launcher. Record test counts and warnings; do not infer hardware success.
- [x] Run Ruff check and format check on changed Python files, and `git diff --check`. Inspect final diff against the starting spec commit, excluding unrelated dirty work.
- [x] Update TASK-23175 with the scoped acceptance criterion, approved plan and concise implementation notes; keep its overall status In Progress. Record the missed LLM preflight incident and software verification limitation in the relevant lesson. Update plan checkboxes and spec approval state.
- [x] Obtain a final scoped review, commit only this repair, and report exact HEAD, software test evidence, no live rerun and unresolved USB playback fault. Do not launch the app: its existing exact-HEAD launcher will now correctly reject the old commit.

## Completion evidence

- Runtime/test commits: `c7f7e632d1154dfb59dc0c94ab1d60b48990251b` and `bf1e39d733159e36bbfc11d4577cdfcc2bcfb23c`. Independent spec review caught and regression-tested configuration publication during the first snapshot await; the follow-up captures a content-free monotonic baseline before yielding. Spec re-review and code-quality review found no remaining actionable issues.
- Initial joined gate: 192 passed, eight failures confined to older native-notification fixtures that lacked ready-provider setup. Migrated only those fixtures; production readiness stayed enforced. The initial-snapshot race independently produced RED (two failed, one passed), then GREEN with 31 focused cases.
- Final joined gate on `bf1e39d733` with owned inventory/documentation changes: **203 passed, zero skips, 217.72 seconds**, exit 0. Exact scope: speculative UI wiring, speculative session, pipeline integration, guarded development entry, new Chat/UI provider-preflight tests, diagnostics, source digest, lint scope, and only the two native notification/late-cleanup test functions. All 11 changed Python files passed Ruff check and format check; committed-range and working diff checks passed.
- Three existing warnings: requests dependency compatibility, audioop deprecation, and webrtcvad/pkg_resources deprecation. Tests used isolated basetemps and fake provider/native boundaries; no microphone, TTS, external provider requests, qualification, route changes, repetitions, soaks or full suite were run.
- Updated the development guide, source/lint inventories and live-verification incident lesson. ADR-098 remains governing; qualification flags and historical hashes are unchanged. Only TASK-23175 AC15 is completed; overall task stays In Progress. The existing app was not relaunched and still contains older code. USB overflow/playback completion remains unresolved; keep the branch/worktree intact.
