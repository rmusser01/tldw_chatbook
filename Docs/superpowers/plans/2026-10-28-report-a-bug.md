# Report a Bug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one editable, privacy-governed Markdown bug report containing the user's description and retained diagnostic evidence, ready for manual sharing.

**Architecture:** A pure report module owns formatting, redaction, budgets, and final-copy validation. A small collection module coordinates bounded reads through the existing log and Console repositories; the native trace reader governs optional tool fields. One shared modal owns description/review state and client clipboard requests, with thin command-palette, Logs, and About entry points.

**Tech Stack:** Existing Python >=3.12, Textual 8.x, SQLite, Pydantic, asyncio/threading, and existing sanitizers. No new dependencies or schema migrations.

**Spec:** [2026-10-28-report-a-bug-design.md](../specs/2026-10-28-report-a-bug-design.md)

**Backlog:** [TASK-37173](../../../backlog/tasks/task-37173%20-%20Prepare-copyable-bug-reports-from-retained-diagnostics.md)

ADR required: yes

ADR path: [backlog/decisions/252-bug-report-evidence-and-sharing.md](../../../backlog/decisions/252-bug-report-evidence-and-sharing.md)

Reason: Defines a user-controlled export contract across app logs, governed Console evidence, and clipboard delivery. ADR-029, ADR-080, and ADR-097 remain binding.

## Global Constraints

- Console **events, timing, and errors are included by default**; **redacted tool content is optional and off by default**.
- Maximum 500 retained log entries and 500 projected Console events; sources remain current-session logs plus existing selected local Console history.
- Complete rendered/copied report: **60,000 UTF-8 bytes**, including edits, fences, and notices. Section budgets: logs 24,000; trace facts 12,000; optional tools 8,000; exceptions 4,000; authored fields 8,000; environment/inventory/formatting 4,000.
- Source presentation deadline: **five seconds**. Bound database reads and content materialization as well as output; a timed-out thread remains owned until retired.
- Immutable source credential/PII masks precede reporting-time built-in diagnostic redaction. Unverifiable content is unavailable; never use unmasked canonical/legacy content as fallback.
- Tool opt-in selects tool arguments/results only; source message, variant, reasoning, and provider-exchange fields remain excluded. Disclose that tool results can themselves contain private document/conversation excerpts.
- App-session logs are application-wide. Preserve append order, existing trace causal order, unknown timing, timestamp timezone provenance, and missing/truncated-source states.
- Report creation performs no upload, LLM call, new capture/retention, config/domain mutation, session initialization, storage bootstrap, or automatic draft-file write.
- Use only already-existing runtime/session/storage owners. `ensure_session`, `ensure_*` Console factories, the full-history Trace snapshot loader, and whole-call native reconstruction are prohibited shortcuts.
- Keep reporting/trajectory imports deferred; compose canonical forms/dialogs with `$ds-*` tokens. No existing token-value changes, new shell destination, or global shortcut.
- Copy the exact reviewed sanitized bytes. A changed final sanitization updates Review and requires a second press. Oversize manual edits stay visible and block copy with a byte-count error.
- Served/SSH copy targets the user's client. Unacknowledged delivery says **Copy requested**, not **Copied**.
- Execute only focused tests. Repository instructions require explicit user opt-in before a full suite.

## Review Focus

1. Frozen PII masks or ownership metadata missing for a tool source: retain diagnostic metadata, omit the body, and never fall back to an old payload (Task 2).
2. A fork with many inherited segments and an unfinished provider call: bound traversal, respect inherited fences, preserve unknown timing, and expose coverage gaps (Task 2).
3. Multiline credentials, multibyte text, and adversarial Markdown delimiters: sanitize before clipping, preserve report structure, and enforce the final byte limit after edits (Tasks 1 and 3).
4. A stalled database read followed by Close, reopen, or workspace change: keep the draft responsive and reject stale results without abandoning the underlying read (Tasks 2 and 3).
5. Command palette opened above a dirty screen, served-mode copy, or a second Copy while sanitizing: keep original context, guard dismissal, target the client, and deliver at most the reviewed candidate (Tasks 3 and 4).

---

## Workspace and execution setup

The existing native-managed worktree is `C:\Users\GDesktop-1\.codex\worktrees\report-a-bug-design\tldw_tui`, based on dev `798f8fca75a7acba56da5f523329efa52c4a78f0`. It is detached; reuse it. Do not create another worktree or mix in the primary checkout's untracked work. Use the native app's branch workflow if a branch is needed for publishing.

Planning creates only this plan, the ADR, task metadata, and design cross-links. Before implementation, confirm the same scoped baseline with the relevant existing tests below; do not install packages or run broad tests merely to prepare the plan. Use the existing project environment when available. Git commands in this Windows sandbox may need the exact command-local `safe.directory` override; do not change global Git configuration. Set `PYTHONIOENCODING=utf-8` for `backlog-py` output on Windows.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| Create `tldw_chatbook/Utils/bug_report.py` | Immutable report inputs, built-in text sanitization, safe Markdown rendering, budgets, final-copy validation. No Textual or database imports. |
| Create `tldw_chatbook/Utils/bug_report_capture.py` | Existing-owner context resolution, evidence collection, source/task retirement, metadata merge, environment inventory. Heavy Console imports stay inside collection paths. |
| Modify `tldw_chatbook/Logging_Config.py::LogsBufferHandler` | One lock-protected immutable tail reader over the existing records. |
| Modify `tldw_chatbook/Chat/console_trace_repository.py` | Bounded attached-lineage event-tail queries and metadata-only reference reads. |
| Modify `tldw_chatbook/Chat/console_trace_native_reader.py` | Project only selected tool fields using existing immutable semantic/artifact governance. |
| Modify `tldw_chatbook/Chat/console_semantic_revision.py` | A bounded tool-field projection that preserves source masks without loading images, attachments, or unrelated message fields. |
| Modify `tldw_chatbook/DB/ChaChaNotes_DB.py` and `DB/AgentRuns_DB.py` | Focused metadata-tail queries for legacy trajectory facts and agent events; never hydrate full messages, step histories, or result blobs. |
| Create `tldw_chatbook/Widgets/bug_report_dialog.py` | Describe/review UI, evidence controls, frozen preview, exact-copy flow, safe dismissal and focus. |
| Modify `app.py`, `app_command_providers.py`, `UI/Logs_Window.py`, `UI/Screens/settings_screen.py` | One app action and three thin entry points. |
| Modify CSS source/build outputs as needed | Token-backed modal geometry; generated sheets only through `css/build_css.py`. |
| New focused tests listed per task | Behavior, privacy, read bounds, lifetime, UI, and import evidence. |

### Shared report interfaces

In `Utils/bug_report.py`, use frozen dataclasses for internal immutable inputs; validate user form boundaries with the existing `Utils/input_validation.py` conventions and Pydantic rather than introducing an independent validator framework.

- `BugReportDraft(description: str, steps: str = "", expected: str = "", actual: str = "")`.
- `BugReportOptions(include_logs: bool = True, include_trace: bool = True, include_tool_content: bool = False)`.
- `BugReportItem(source: Literal["logs", "exceptions", "trace", "tools"], text: str, event_key: str | None = None, observed_at: str | None = None)`. Items reaching the renderer are already governed; `text` is plain display text, never interpreted markup. `event_key` is internal and is never printed verbatim.
- `BugReportEvidence(environment: tuple[tuple[str, str], ...] = (), logs: tuple[BugReportItem, ...] = (), exceptions: tuple[BugReportItem, ...] = (), trace: tuple[BugReportItem, ...] = (), tools: tuple[BugReportItem, ...] = (), inventory: tuple[str, ...] = ())`.
- `BugReportCopyCheck(state: Literal["ready", "review_required", "oversize", "unavailable"], text: str, byte_count: int, message: str)`. Unsafe raw text must not appear in `message`, repr, or logs; suppress body-field repr throughout these report models.
- `render_bug_report(draft: BugReportDraft, evidence: BugReportEvidence, options: BugReportOptions) -> str`.
- `prepare_bug_report_copy(text: str) -> BugReportCopyCheck`.

In `Utils/bug_report_capture.py`:

- `BugReportContext` freezes `RuntimeIdentity` from `UI/Navigation/screen_state_store.py`, workspace/session/conversation IDs where available, active-session epoch, originating route, and UTC collection-start time. IDs/titles are local selection facts, not automatically exported labels.
- `BugReportCapture` contains that context, immutable `BugReportEvidence`, frozen internal tool-source references, and source collection ranges/sequence watermarks. Internal bodies and source owners suppress repr.
- `capture_bug_report_context(app: TldwCli) -> BugReportContext` performs only existing in-memory context reads on the UI loop.
- `BugReportCollector(app: TldwCli, context: BugReportContext)` captures existing source handles without initializing them. `async collect() -> BugReportCapture`, `async collect_tools(capture: BugReportCapture) -> BugReportCapture`, and `cancel() -> None` own bounded operations and cooperative retirement. The collector retains task handles after presentation timeout and refuses an overlapping retry until that source retires. It never updates widgets directly.

## Task 1: Pure report rendering and final-copy contract

**Files:** Create `Utils/bug_report.py` and `Tests/Utils/test_bug_report.py`.

**Interfaces:** Consumes existing uncapped `log_sanitizer.sanitize_string`; produces all shared report data classes and the two pure functions above. No Console, UI, clipboard, or storage dependency.

- [ ] **Write the failing tests.** Cover `test_default_report_excludes_optional_tool_items`, `test_utf8_budgets_include_fences_and_notices`, `test_multiline_credentials_are_redacted_before_clipping`, `test_missing_sources_are_explicit`, `test_final_copy_preserves_oversize_manual_edits`, and `test_repeat_copy_stabilizes_after_redaction`. Use the real sanitizer and synthetic evidence, not a mock that always returns safe text. Representative assertions:

  ```python
  evidence = BugReportEvidence(tools=(BugReportItem("tools", "PRIVATE_TOOL_SENTINEL"),))
  text = render_bug_report(BugReportDraft("Observed failure"), evidence, BugReportOptions())
  assert "PRIVATE_TOOL_SENTINEL" not in text
  assert len(text.encode("utf-8")) <= 60_000
  oversized = "界" * 20_001
  check = prepare_bug_report_copy(oversized)
  assert check.state == "oversize" and check.text == oversized
  assert check.byte_count == 60_003
  ```

- [ ] **Run the focused tests to establish failure.** `python -m pytest Tests/Utils/test_bug_report.py -q`. Expected: missing new module/interfaces, not an unrelated fixture failure.
- [ ] **Implement the interfaces.** Validate nonblank description and 8,000-byte authored-field budget. Apply reporting-time built-in credential/PII sanitization before source clipping and again before final copy. Account for rendering overhead in each section; drop oldest automatic items with explicit notices while preserving valid UTF-8. Reserve the 4,000-byte metadata/formatting allowance. If authored content itself cannot fit, return a field validation error rather than silently trimming it.
- [ ] **Implement stable formatting and copy checks.** Use the exact `Chatbook bug report v1` marker and source/coverage inventory from the spec. Fence untrusted text safely even when it contains both backticks and tildes; formatting overhead consumes the same limit. `prepare_bug_report_copy` preserves oversize text, returns `review_required` when sanitization changes the candidate, and returns `ready` only for stable sanitized bytes within 60,000. A sanitization exception returns a content-free unavailable result. Never apply the 2,000-character log-record cap to the whole report.
- [ ] **Verify green and commit.** Run the same test file plus `Tests/Utils/test_log_sanitizer.py`. Expect all selected cases to pass. Stage only the new module/tests; commit `feat: render bounded privacy-governed bug reports`.

## Task 2: Bounded evidence collection through existing owners

**Files:** Create `Utils/bug_report_capture.py`, `Tests/Utils/test_bug_report_capture.py`, `Tests/Chat/test_bug_report_trace_evidence.py`; modify `Logging_Config.py`, `Chat/console_trace_repository.py`, `Chat/console_trace_native_reader.py`, `Chat/console_semantic_revision.py`, `DB/ChaChaNotes_DB.py`, `DB/AgentRuns_DB.py`.

**Interfaces consumed:** Task 1 report models. Existing `ConsoleRuntime.chat_store`, `ConsoleChatStore.active_session_id`, `sessions()`, `active_session_epoch()`, session workspace/persisted-conversation facts, app `_current_runtime_identity()`, and already-existing database owners.

**Interfaces produced:** Context/capture/collector above; plus these source-owned public reads:

- `LogsBufferHandler.snapshot_report_tail(limit: int = 500) -> tuple[int, tuple[tuple[str, str, str], ...]]`: watermark and immutable `(level, logger_name, formatted_message)` records copied under `_log_records_lock`.
- `ConsoleTraceRepository.read_report_event_tail(cursor: sqlite3.Cursor, conversation_id: str, *, limit: int = 500, max_segments: int = 64) -> TraceReportEventTail`. Add a frozen repository-local result with `events: tuple[TraceEventRecord, ...]`, `has_older: bool`, and `lineage_truncated: bool`. Preserve attached-owner/fork fences; metadata only.
- `CharactersRAGDB.get_report_trajectory_tail(conversation_id: str, *, limit: int = 500) -> tuple[Mapping[str, object], ...]`: explicit identity/kind/state/timing/provider/model columns; no `payload_json` or message content.
- `AgentRunsDB.get_report_event_tail(conversation_id: str, *, limit: int = 500) -> tuple[Mapping[str, object], ...]`: bounded run/step structural metadata and selected diagnostic error fields. Do not call `list_runs()` to hydrate all steps. Prefer existing columns and bounded SQLite JSON field extraction with malformed/oversized records reported as unavailable; never select complete `steps`, args, result, or payload blobs for the default report.
- `ConsoleTraceNativeReader.read_report_tool_fields(conversation_id: str, events: tuple[TraceEventRecord, ...], *, max_source_bytes: int = 65_536, max_total_source_bytes: int = 262_144, max_fields: int = 500) -> tuple[Mapping[str, object], ...]`: internal source-event key, governed tool name/arguments/result, and explicit field states. This is a selected-field reader, not a call to `read_calls()` or `_reconstruct_request()`.
- `console_semantic_revision.project_semantic_revision_trace_tool_fields(cursor: sqlite3.Cursor, *, revision_id: str, expected_conversation_id: str, policy_id: str, max_source_bytes: int = 65_536) -> dict[str, object]`: validates revision/policy binding, applies the existing immutable mask decisions to supported tool-call/result field paths, and returns only those fields plus omission states. It must never fetch image/attachment BLOBs or unselected provider/reasoning content. Existing general trace-message projection behavior remains unchanged.

- [ ] **Write failing collection/privacy tests using real SQLite.** Cover no Console/session, app-global log labels, concurrent log append, no initialization/writes, metadata-only default reads, immutable mask preservation, capture-time PII Off plus reporting-time built-in PII, missing policy/legacy governance, and tool results containing conversation excerpts. Spy on the forbidden whole-history/native-whole-call methods so the tests fail if they are used. Assert:

  ```python
  assert len(capture.evidence.logs) <= 500
  assert len(capture.evidence.trace) <= 500
  assert capture.evidence.tools == ()
  assert "MASKED_SOURCE_SENTINEL" not in report_text
  assert "Application-wide session logs" in report_text
  assert database_total_changes_after == database_total_changes_before
  assert forbidden_initializer_calls == []
  ```

  Add `test_fork_tail_respects_inherited_fence_and_segment_cap`, `test_missing_policy_has_no_raw_legacy_fallback`, `test_large_omitted_tool_source_is_not_loaded`, `test_tool_projection_never_reads_images_or_unselected_content`, and `test_default_report_does_not_reconstruct_provider_exchange`. A shared in-memory database must use the existing test ownership pattern when worker threads read it; do not accidentally test a separate empty SQLite connection.
- [ ] **Run the two new test files to establish failure.** Expected: missing new interfaces or an assertion showing the unbounded/raw path. Fix fixture/environment failures separately from product behavior.
- [ ] **Implement source-tail queries in their owners.** Each query returns at most 500 candidates plus at most one metadata sentinel for `has_older`; globally merge at most 1,500 metadata candidates into 500 exported events. Visit at most 64 attached lineage segments. Do not sort away source/causal order or read outside inherited fences. Keep fields with unknown timing unknown and note missing parents. Assign report-local identity aliases once per frozen capture and reuse them across generated sections; replace exact known identifiers in log text with the same aliases where present, without guessing scope/correlation from arbitrary prose. No payload reads for excluded tool content. Metadata default tool names may be unavailable if they exist only inside an unverifiable content field; do not bypass governance to fill the label.
- [ ] **Implement selected tool projection.** Reuse the ownership, policy-binding, and immutable mask decisions from `console_semantic_revision.project_semantic_revision_trace_message` in the new bounded tool-field helper, plus corresponding governed artifact decisions in the native reader. The existing full-message helper can load image/attachment data, so do not call it and discard unrelated fields afterward. Validate that references belong to the selected attached lineage before projecting using each verified revision's source conversation and frozen policy, including inherited fork prefixes. Read only the exact selected tool component, never an entire reconstructed request. Check source lengths before materialization; cap one field at 65,536 source bytes and the optional-content operation at 262,144 aggregate source bytes, stopping once the 8,000-byte output budget is filled. Omit an oversized/ambiguous/unverifiable source rather than clipping unredacted bytes or guessing a policy. Do not use provider-verification `_resolve_reference_values`, which deliberately exposes canonical values without viewer masks. Apply the report allowlist and diagnostic sanitizer after governance. Leave older trace facts usable with content marked unavailable.
- [ ] **Implement the collector and cancellation contract.** Freeze source selection on the loop without `ensure_*`; gather through existing read-only owner connections off-loop. Do not bootstrap storage or use AgentRuns write transactions. Enforce scoped query work/deadlines on operation-owned connections without changing global/shared DB timeouts. A five-second presentation deadline returns available evidence with a controlled unavailable code, sets cooperative cancellation, and retains the read task until it exits. Closing or changing runtime/session epoch rejects late results. Tool opt-in revisits only frozen references, not a newer tail; omit purged/changed sources. Failed source messages must not log report bodies or raw exception contents.
- [ ] **Add and run lifetime/bounds regressions.** `test_source_deadline_keeps_read_owned_until_retirement`, `test_retry_waits_for_previous_source_retirement`, `test_workspace_change_rejects_late_capture`, `test_tool_toggle_uses_frozen_event_refs`, `test_nonmonotonic_clocks_preserve_causal_order`, and `test_large_conversation_reads_bounded_metadata_only`. Use held gates and real query instrumentation, not flaky sleeps or a mock asserting that `limit=500` was passed. Verify bounded returned rows/content allocations and controlled partial results; use existing indexes or bounded traversal without a schema change.
- [ ] **Verify green and commit.** Run the new collection/trace test files, `Tests/Chat/test_console_trace_native_reader.py`, `Tests/Chat/test_console_trace_privacy_owners.py`, `Tests/Chat/test_console_trace_pii_masks.py`, and `Tests/test_logs_buffer_single_record_per_emission.py`. Expect all selected cases to pass, with unchanged existing trace exports. Commit only these source/test changes as `feat: collect bounded governed bug report evidence`.

## Task 3: Shared describe/review modal and client copy

**Files:** Create `Widgets/bug_report_dialog.py`, `Tests/UI/test_bug_report_dialog.py`; modify source CSS only if canonical component classes cannot express the required geometry, and update `Tests/Architecture/test_guarded_modal_quit_census.py` for the new guarded close owner. Rebuild affected generated CSS sheets through `css/build_css.py`.

**Interfaces consumed:** Task 1 report functions/models; Task 2 `BugReportCollector` and frozen context/capture.

**Interfaces produced:** `BugReportDialog(app_instance: TldwCli, context: BugReportContext)` subclasses `SafeModalDismissMixin, ModalScreen[None]`; exposes `request_description_focus() -> None` for duplicate invocation. Modal-private handlers own compose/collect/preview/copy/discard; no second report service is introduced.

- [ ] **Write failing Textual interaction tests.** Exercise the real controls and worker completion, not only private methods. Test initial Description focus, blank/oversized authored-field errors, optional evidence controls, tool opt-in off, editable Review, frozen evidence, Back regeneration with a dirty-review discard guard, copy sanitization requiring re-review, oversize manual edits, unavailable clipboard, duplicate Copy, cancellation, and opener focus. Representative assertions:

  ```python
  assert dialog.query_one("#bug-report-tools").value is False
  assert clipboard_payloads == []  # after a Copy that changed the sanitized preview
  assert review.text == expected_sanitized_preview
  assert clipboard_payloads == [review.text]  # after the next stable Copy
  assert len(clipboard_payloads[0].encode("utf-8")) <= 60_000
  ```

  Define the controls with stable IDs `bug-report-description`, `bug-report-steps`, `bug-report-expected`, `bug-report-actual`, `bug-report-logs`, `bug-report-trace`, `bug-report-tools`, `bug-report-inventory`, `bug-report-review`, `bug-report-generate`, `bug-report-back`, `bug-report-copy`, and `bug-report-close` so tests and focus restoration use the same contract.
- [ ] **Run the new UI file to establish failure.** `python -m pytest Tests/UI/test_bug_report_dialog.py -q` must fail for the missing dialog/behavior.
- [ ] **Implement the two-stage modal.** Use canonical labelled form fields, checkbox controls, scrollable body, status area, and docked actions. In Description, validate before Generate; show missing evidence as source states rather than disabling narration. Inventory distinguishes app-global logs from the selected conversation, plus collection range, unknown timing, source omissions, and opt-in disclosure. Review is editable plain text. Freeze evidence and guard regeneration over manual review edits; no automatic refresh overwrites user work. Heavy operations run in exclusive worker groups with `exit_on_error=False`, request generations, and guarded UI delivery. Maintain text-change revisions so a delayed copy result cannot overwrite a newer edit.
- [ ] **Implement exact-copy and dismissal.** Validate/sanitize the exact current review. On changed text, update it and require the second press; on edits while a copy check runs, discard that check and require a fresh request. Serialize copying and never copy a stale candidate. Use `app.copy_to_clipboard` as the existing client route, with **Copy requested** wording when acknowledgement is absent; on driver error leave the selectable text intact. Confirmed native copying may reuse existing installed capability but is unnecessary to add for v1. Follow `SafeModalDismissMixin` cancellation, `confirm_quit_discarding_edits`, eligible opener focus, and the guarded-close census. Retire collection work on close without claiming an uncancellable thread stopped.
- [ ] **Verify layouts and lifetime.** At 60x24 and 120x40, keyboard traversal reaches every field and Copy/Close without hidden actions. Hold collection and copy checks across Close/reopen/runtime change, and verify no stale update or duplicate delivery. Test dirty Escape, backdrop, visible Close, and Ctrl+Q through real app quit paths. Preserve terminal conventions: no screen binding for Ctrl+C or Ctrl+A and no shadow of global Ctrl+P/F1/F6.
- [ ] **Verify green and commit.** Run the modal tests, `Tests/UI/test_console_modal_dismissal.py`, `Tests/Architecture/test_guarded_modal_quit_census.py`, and the targeted design/CSS checks in Task 4 after rebuilding. Commit `feat: add editable bug report review and copy dialog`.

## Task 4: Shared entry points, documentation, and end-to-end evidence

**Files:** Modify `app.py`, `app_command_providers.py`, `UI/Logs_Window.py`, `UI/Screens/settings_screen.py`, `Docs/User_Guide/logs.md`, `Docs/Architecture/app-shell.md`, `Docs/Architecture/console.md`, and the Backlog task. Create `Tests/UI/test_bug_report_entry_points.py` and `Tests/Packaging/test_bug_report_import_deferral.py`; extend existing palette/About tests only where their existing contracts require it.

**Interfaces consumed:** `capture_bug_report_context`, `BugReportDialog`, and its `request_description_focus` method.

**Interface produced:** `TldwCli.action_report_bug() -> None` lazily opens one dialog from captured underlying destination context; duplicate requests focus the already-open dialog. Store only the minimum current-dialog/request ownership needed on the app and clear it when that owner dismisses. The callback runs after palette dismissal so palette route/focus is not captured as the originating destination.

- [ ] **Write failing entry/import tests.** Verify `QuickActionsProvider.search()` and `.discover()` expose **Report a bug…**, Logs and About buttons call the same action, the underlying route/dirty state is preserved, and an uninitialized Console remains uninitialized. In a fresh-process import test, plain app/Logs/About imports do not load the new dialog, trajectory, native reader, or optional heavy dependencies; opening a report without Console evidence does not materialize them either. Imported shared Trace screens supply no automatic local evidence.
- [ ] **Run the focused tests to establish failure.** `python -m pytest Tests/UI/test_bug_report_entry_points.py Tests/Packaging/test_bug_report_import_deferral.py -q`.
- [ ] **Wire the action and entry points.** Add an app action with a deferred modal/capture import; use existing app lifetime/runtime state, not a new service graph or destination. Add matching search/discover records to `QuickActionsProvider`, one button beside Logs' existing copy actions, and one About action in the canonical Settings screen. Preserve Logs filters and underlying screen state. Do not alter old project links or unrelated settings behavior as part of this task.
- [ ] **Document user-visible boundaries and actual integration.** Explain description/review/copy, optional tool content and its excerpt warning, app-wide log scope, source/time/retention limits, current-session-only recovery, manual sharing, and copy-request wording. Add code-anchored app-shell/Console references to the final helpers and bounded native reader. Link ADR-252 and this plan from the task; implementation notes are written only after implementation/verification, not as a premature success claim.
- [ ] **Run focused automated and static checks.** Run the four new feature test groups plus affected palette/About/native-reader/privacy tests; include `Tests/UI/test_design_token_governance.py`, `Tests/UI/test_component_pattern_governance.py`, `Tests/UI/test_css_bundle_sync_guard.py`, `Tests/UI/test_non_obscuring_focus_contract.py`, and `Tests/Packaging/test_exchange_export_trajectory_deferral.py`. Run `python tldw_chatbook/css/build_css.py` before CSS consistency checks if source styling changed. Use the available project formatter/linter on new helpers/tests and changed code; for legacy-file lint debt, compare the same checks at the original baseline and do not mass-reformat unrelated code. Minimum syntax/undefined-name check: `python -m ruff check --select E9,F63,F7,F82` over the changed Python files; run `python -m ruff format --check` on the new modules/tests. Reconcile real failures, then rerun only affected checks.
- [ ] **Perform live verification with a disposable profile.** Follow `backlog/docs/lessons-live-verification.md`, including the supported launch mechanism and app-log evidence; do not redirect Textual's entire stderr to a file and misdiagnose a blank UI. Generate one controlled captured failure with known dummy credentials/PII and one selected Console tool run. Open reporting via palette, Logs, and About; inspect defaults, toggle tool details, edit Review, paste into a local text sink, and compare exact bytes with the reviewed candidate. Exercise no-conversation, missing-traceback, blocked trace read, 60x24 layout, and clipboard-unavailable behavior. Do not upload real report content or make a paid LLM call just to verify reporting. Record commands, observed behavior, evidence paths/hashes, and any platform-specific gap honestly.
- [ ] **Review and close the task.** Self-review against the spec and ADR, then request the whole-branch independent review required by the selected execution workflow; fix substantive findings. Run `git diff --check`, the ADR/task uniqueness/readability guards, and any affected license/security/static checks. Add concise Implementation Notes with final files/decisions, exact test/live evidence, and deviations. Check each acceptance criterion only with evidence, and mark TASK-37173 Done via CLI only after all DoD items pass. No full-suite claim without a user-authorized full sweep. Commit `feat: expose report a bug across application help surfaces` with documentation/task evidence.

## Planning self-review

- Spec coverage: Tasks 1-2 cover contents, limits, privacy, source selection, governance, and ownership; Task 3 covers the complete describe/review/copy/dismiss flow; Task 4 covers discoverability, deferred imports, documentation, and live evidence.
- Interfaces: Shared names/signatures above are reused consistently; no task depends on an undefined later component. Source-reader results stay internal and never become raw report fields.
- Review Focus: Each of the five input/failure classes has explicit tests in its owning task. Deterministic gates and query instrumentation replace timing guesses.
- Proportion: One feature/Backlog task, four reviewable implementation commits. No separate storage, uploader, recorder, settings framework, or generic export abstraction is introduced.
- Current state: Planning artifacts only. Product code, feature tests, and implementation notes have not been written, and no implementation-test result is claimed.

## Execution handoff

Review this plan before product implementation. **Recommended: Native execution in this session**, since the four tasks share the same report contracts and bounded-source lifetime; one implementer can keep those interfaces consistent, followed by an independent whole-branch review. Subagent-driven execution is also available, with a fresh implementer and reviewer per task and a final whole-branch review, at greater context cost. The user's execution-method choice has not yet been supplied.
