# Textual lifecycle and import follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Complete TASK-36693 through TASK-36698 and deliver one reviewed PR against dev with passing required CI.

**Architecture:** Repair publication and retirement at the existing widget, import-panel, parser and screen owners. Reuse LazyLiveDictationService.abandon, existing canonical Chunking Lab session edits, ValueError import validation and Textual worker cancellation. No shared draft framework, new dependency, storage change or background-capture mode.

**Tech Stack:** Python 3.12, Textual 8.x, existing SQLite services and pytest private-profile harness.

**Spec:** [Approved book audit](../qa/2026-10-24-textual-book-after-pr3228.md), with the four acceptance criteria in each existing TASK-36693 through TASK-36698 record as the binding requirements.

## Global Constraints

- Workers for operations >100ms; do not join processing threads on the UI loop.
- Preserve the existing service, persistence, consent, recovery and file-validation boundaries.
- Screens must not bind terminal-convention keys (ctrl+c/v/x/s/d/z/a/r/w) or shadow the globals ctrl+p/ctrl+q/f1/f6.
- Read backlog/docs/design-language.md and backlog/docs/component-patterns.md before changing UI owners; use existing token-backed components. No stylesheet change is planned.
- Use targeted tests only; no full local suite. Every new regression must execute on the hosted UI or appropriate parser gate.
- Run actual message/worker and filesystem boundaries; fake only bounded audio/provider/catalog I/O. Use fresh private profiles and inspect child JUnit failures/errors.
- Existing source assertions and safety allowances remain intact. Root coordinates the shared UI census and Backlog CLI; implementers own only their assigned source/tests.
- Superpowers requesting-code-review replaces Qodo. Create a PR against dev and address CI/review feedback; merging that new PR is outside this request.

## ADR Check

ADR required: no new ADR expected.
ADR paths: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md; 031-tui-keybinding-and-footer-hint-conventions.md; 033-application-session-state-ownership.md; 118-chunking-lab-local-execution-and-recovery.md; 161-component-pattern-library.md; 176-one-media-generation-core.md; 240-mcp-destination-retirement-tools-hub-in-settings.md.
Reason: Bounded bug fixes directly preserve existing owners/contracts. Re-evaluate before expanding scope into storage, background capture, security policy or new interfaces.

## Review Focus

- A captured callback arriving after teardown must lose authority before it can update either the departed or replacement Speech view (Task 1).
- Ordinary Enter/Space button activation must not rely on the existing forbidden ctrl+d workaround, and unhandled container buttons must retain routing (Task 2).
- A sample edit followed by clearing, a newer sample intent, and a failed read must leave live and canonical Chunking state coherent (Task 3).
- Typing then clearing to the original empty MCP text and Cancel/reopen must invalidate an older file read; rejected Preview must not retain an earlier candidate (Tasks 4 and 5).
- An old result whose UI callback is already queued, explicit worker cancellation, and old failure must not overwrite a newer Image Gen result (Task 6).

## Integration and Delivery

- [ ] Register all new tests on the real hosted gates, preserving existing selectors, 17 UI shards and existing timeouts/safety allowances. Use measured durations to place additions; root alone edits census/contracts.
- [ ] Run combined targeted UI/parser/CI/token/size checks appropriate to modified owners, plus actual required source guards. Attribute any native-platform limitations explicitly.
- [ ] Complete per-task AC/notes/ADR checks and status via Backlog CLI only after evidence and reviews satisfy them; update relevant architecture/QA documentation.
- [ ] Obtain a fresh whole-branch Superpowers review, fix actionable findings, push and create/attach the PR against dev.
- [ ] Inspect all current-head CI and complete remote feedback/thread pages, address failures/findings and repeat affected checks until the delivered head is green and feedback is handled.
## Task 1: TASK-36693 — retire foreground Speech capture

**Files:** Modify tldw_chatbook/UI/Dictation_Window_Improved.py; inspect tldw_chatbook/UI/STTS_Window.py and Audio/dictation_service_lazy.py; create Tests/UI/test_dictation_view_retirement.py.
**Interfaces:** Consumes the existing STTS optional cleanup() and LazyLiveDictationService.abandon() contracts. Produces idempotent DictationWindow cleanup() -> None and retired callback authority; explicit Stop retains its result semantics.
**ADR required:** no. **ADR path:** existing 025 and 033. **Reason:** same widget-owned service and existing nonblocking abandonment.

- [x] Write mounted active-leave/unmount, late-callback, replacement, explicit-Stop, unstarted and repeated-cleanup controls using injected audio I/O and the real processor thread. Ordinary Stop must yield stop_calls=1; active leave must retire capture and processing without a UI-thread join.
- [x] Run the focused cases before implementation and retain the relevant retirement/publication failures; distinguish setup/teardown errors from product failures.
- [x] Revoke callback publication before existing abandonment and before asynchronous settings flush/unmount waits. Use the existing cleanup seam, without creating a background capture owner.
- [x] Run the new module plus relevant existing dictation lifetime/settings controls; inspect child JUnit and run focused Ruff/format checks.
- [x] Self-review and commit only this task's source/tests, then obtain a task-scoped Superpowers review.

## Task 2: TASK-36694 — deliver handled Speech buttons once

**Files:** Modify tldw_chatbook/UI/Dictation_Window_Improved.py; inspect STTS_Window.py; create Tests/UI/test_dictation_button_delivery.py.
**Interfaces:** Consumes real Button.Pressed bubbling and the parent dispatch. Produces no new API; recognized child controls consume their message once and unrelated container controls still bubble.
**ADR required:** no. **ADR path:** existing 025 and 031. **Reason:** routine delivery correction without new keys or routing structure.

- [ ] Write actual Pilot mouse and Enter/Space Start/Stop controls: one Start leaves start_calls=1, stop_calls=0 and listening; one later Stop leaves stop_calls=1 and idle. Cover other handled controls and an unhandled/container control.
- [ ] Run the focused cases against the unchanged dispatch and retain the duplicate-delivery failures.
- [ ] Stop propagation for controls this child handles; preserve unhandled/container delivery. Do not add or pin a reserved terminal chord.
- [ ] Run focused new/existing Speech routing checks and Ruff/format checks.
- [ ] Self-review, commit this task's source/tests and obtain a task-scoped Superpowers review.

## Task 3: TASK-36695 — preserve newer Chunking sample intent

**Files:** Modify tldw_chatbook/UI/Screens/chunking_lab_screen.py and only if needed UI/Chunking_Lab_Modules/sample_region.py; create Tests/UI/test_chunking_lab_sample_import_drafts.py.
**Interfaces:** Consumes existing file_operation, read_sample_file, canonical session edits and recovery. Produces no shared/public API; applicability stays with the screen/sample owner.
**ADR required:** no. **ADR path:** existing 118 and 033. **Reason:** existing sample/recovery ownership, unchanged validation.

- [ ] Write real mounted sample-file/excerpt reads with accepted keyboard input witnessed in editor and session before release; test later text, clear, replacement intent, untouched load and read failure. An untouched load publishes selected file content and provenance; later input stays exact.
- [ ] Run and retain product RED evidence. Build a bounded host or settle existing owners so fixture teardown is independently attributable; do not hide native-worker uncertainty.
- [ ] Fence publication of an older read against current sample intent, including changes while earlier screen messages wait. Retain existing file-size/UTF-8 errors and recovery contracts.
- [ ] Run new controls and relevant existing file/excerpt/recovery tests; inspect child JUnit and Ruff/format checks.
- [ ] Self-review, commit this task's source/tests and obtain a task-scoped Superpowers review.

## Task 4: TASK-36696 — preserve MCP import drafts

**Files:** Modify tldw_chatbook/UI/MCP_Modules/mcp_workbench.py and mcp_profile_form.py; create Tests/UI/test_mcp_import_pending_file_drafts.py.
**Interfaces:** Consumes real EnhancedFileOpen callback, _load_import_file and MCPImportPanel text. Produces owner-local request/panel/draft applicability only; Task 5 retains the same Preview interface.
**ADR required:** no. **ADR path:** existing 033 and 240. **Reason:** existing Settings/panel ownership and filesystem read contract.

- [ ] Write actual picker callback + threaded file-read controls for untouched text, later typing, clear (including returning to initial empty text), and Cancel/reopen replacement.
- [ ] Run against current completion and retain the intended draft/replacement RED evidence.
- [ ] Bind completion to the originating panel and unchanged draft intent; invalidate on replacement/cancel. Preserve ordinary reads and raw invalid user text.
- [ ] Run new controls and relevant existing MCP import/file-picker tests; inspect private child JUnit and focused Ruff/format checks.
- [ ] Self-review, commit this task's source/tests and obtain a task-scoped Superpowers review.

## Task 5: TASK-36697 — recover malformed MCP import Preview

**Files:** Modify tldw_chatbook/MCP/mcp_import.py and only if needed UI/MCP_Modules/mcp_profile_form.py; create Tests/MCP/test_mcp_import_shapes.py and Tests/UI/test_mcp_import_validation_recovery.py.
**Interfaces:** Consumes parse_mcp_servers_json and existing ValueError Preview recovery. Produces field-specific ValueError for malformed args/env containers; preserves supported valid/omitted inputs and spawn/env security rules.
**ADR required:** no. **ADR path:** existing 161 and 240. **Reason:** input-boundary validation within the existing contract.

- [ ] Write parser and mounted Preview controls for args:42, env:["bad"], other invalid container forms, valid/omitted fields, valid-preview followed by rejected text, and correction/recovery under the installed keepalive policy.
- [ ] Run before implementation; retain TypeError/AttributeError or stale-candidate RED evidence at the actual parser/panel boundary.
- [ ] Validate container shapes at the parser boundary and preserve editable source, disabled Import and cleared stale candidates on rejection. Keep existing command/secret filtering intact.
- [ ] Run new and existing parser/spawn-validation/import Preview controls; inspect private child JUnit and Ruff/format checks.
- [ ] Self-review, commit this task's source/tests and obtain a task-scoped Superpowers review.

## Task 6: TASK-36698 — ignore superseded Image Gen demo publications

**Files:** Modify tldw_chatbook/UI/Screens/image_gen_demo_screen.py; inspect UI/image_gen_command_provider.py; create Tests/UI/test_image_gen_demo_request_lifetime.py.
**Interfaces:** Consumes real exclusive thread Worker, _generate/_apply_result and off-thread decode. Produces current-request success/error publication guards; preserves serial generation and the live palette entry.
**ADR required:** no. **ADR path:** existing 176 and 033. **Reason:** existing demo worker/media core, no provider/runtime boundary change.

- [ ] Write actual Generate/worker controls for held OLD success/error after NEW completes, queued OLD UI publication after NEW admission, cancellation, ordinary serial success and newest error. Real adapters stay bounded; exercise decode/publication where available.
- [ ] Run unchanged source and retain stale-publication RED evidence with actual Worker.is_cancelled observed.
- [ ] Check cancellation and current-request authority at UI publication for both success and error; retain off-thread decode and existing status behavior.
- [ ] Run new and existing demo controls, palette reachability check and focused Ruff/format checks.
- [ ] Self-review, commit this task's source/tests and obtain a task-scoped Superpowers review.
