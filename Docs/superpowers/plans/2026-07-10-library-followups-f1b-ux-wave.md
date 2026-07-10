# F1b — Home/Library UX wave (post-review fixes) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Anchors are exact at branch head d8d3632b (F1); grep symbols, lines drift.

**Goal:** Implement all 15 findings from the 2026-07-10 sr UX/HCI review of the Home and Library screens (H1–H3, M1–M5, L1–L7), on `claude/library-followups`, folded into PR #592.

**Architecture:** Three sequential tasks: W1 Home canvas coherence, W2 ingest failure classification + queue canvas, W3 Library copy & metadata. W1 lands before W2 (W2 refines the failure copy W1 starts rendering on Home).

## Global Constraints

- Stage only changed files by explicit path; NEVER `git add -A`. Never touch `.claude/settings.local.json`.
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- RED-first for behavior changes; bounded polls in pilots (`range(150)`/`pause(0.02)`); `escape_markup` for any user-controlled text that reaches Button/Static markup.
- Registry mutations are UI-thread-only; worker threads use `self.call_from_thread`.
- Test command: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`.
- CSS changes go in source modules + `./build_css.sh` (commit both).
- No behavior-weakening test edits: re-anchor copy asserts, keep the assertion's intent.

### Task 1: W1 — Home canvas coherence (H1, H2, H3, M1, L7)

**Files:**
- Modify: `tldw_chatbook/Library/library_ingest_jobs.py` (job dataclass ~:114-130; `mark_done` :311; `mark_failed` :344)
- Modify: `tldw_chatbook/Home/active_work_adapter.py` (`_local_ingest_job_items` :557; `_local_recent_work_items` :499; v1-exclusion docstring :560-575)
- Modify: `tldw_chatbook/Home/dashboard_state.py` (`HomeActiveWorkItem` :100; controls builder :300-365; next-action engine ~:185-210; canvas builder :770-840)
- Test: `Tests/Home/` (adapter + dashboard-state units), `Tests/UI/test_home_screen.py` (pilots)

**Interfaces (produced):** `LibraryIngestJob.finished_at_wall: str` (ISO-8601 UTC, "" until terminal); `HomeActiveWorkItem.status_detail: str = ""`.

**H1 — done imports appear in Recent.** Add `finished_at_wall: str = ""` to `LibraryIngestJob`; stamp `datetime.now(timezone.utc).isoformat()` inside `mark_done` AND `mark_failed` (both terminal). In the adapter, extend `_local_recent_work_items` (or a sibling `_local_ingest_recent_items()` merged before the sort) to emit DONE registry jobs (state DONE, not `dismissed`, not `superseded`) as `HomeActiveWorkItem(item_id=f"local:ingest:{job.job_id}", title=<escaped basename, same escape() as :605>, source="Library", status="done", detail_route="library", console_available=False, updated_at=job.finished_at_wall)`. The existing `_HOME_RECENT_WORK_STATUSES` set already contains "done"; keep `_HOME_RECENT_WORK_LIMIT`. Rewrite the v1-exclusion docstring paragraph (:568-575) — it is now stale, including the `updated_at` always-"" claim. RED test: registry with one DONE job → recent items contain it, sorted by `finished_at_wall`.

**H2 — no global flashcards shortcut on an item canvas.** In the controls builder (:353-363), append `home-review-flashcards` ONLY when no real work item is selected: when the selected row is a `HomeActiveWorkItem`, skip it; when nothing is selected or the selected row is the synthetic `HOME_FLASHCARDS_DUE_ROW_ID`, keep it. The builder must learn the selection — thread the selected row id (or detail item) into `build_home_controls` from the canvas builder (:786, :801); the count-only fallback path (:826-834) keeps today's behavior. Re-anchor existing tests that assert the control's unconditional presence.

**H3 — the Next hint must not duplicate the selected item's own recovery.** When the canvas builder resolves `next_action` for a selected item: if `next_action.action_id == "review_failed_work"` and the selected item's status is in the failed set and its `detail_route` equals the action's route, recompute the suggestion with that branch suppressed (add an `exclude: frozenset[str]` parameter to `choose_next_best_action`, default empty) so the engine falls through to its next branch. Count-only path unchanged. RED pilot/unit: failed ingest item selected → canvas `next_action.action_id != "review_failed_work"`; unselected state still suggests it.

**M1 — failure reason on the Home canvas.** Add `status_detail: str = ""` to `HomeActiveWorkItem`. Adapter sets it for FAILED ingest jobs from `job.error`, escaped with the same `escape()` used for titles (it flows into Static/Button markup). Canvas builder (:807-812): when the selected item has a non-empty `status_detail`, insert it as its own line directly after `status_line`, truncated to 140 chars with "…". Unit: failed job with error → canvas lines include the reason.

**L7 — bridge the source/destination mismatch on the flashcards canvas.** :791: `f"{selected.glyph} due for review · Library"` → `f"{selected.glyph} due for review · Study decks in Library"`. Re-anchor asserts.

Run: `Tests/Home/` + `Tests/UI/test_home_screen.py` + `Tests/Library/test_library_ingest_jobs*.py` (registry field). Commit: `fix(home): recent imports, selection-scoped controls, non-duplicate next hint, failure reason line`.

### Task 2: W2 — Ingest failure classification + queue canvas (M4, L4, L5)

**Files:**
- Modify: `tldw_chatbook/Library/library_ingest_jobs.py` (`mark_failed` :344 — new kwarg)
- Modify: `tldw_chatbook/app.py` (runner failure sites :1223, :1249, :1436; `retry_library_ingest_job` :1230)
- Modify: `tldw_chatbook/Library/library_ingest_state.py` (`_build_queue_row` ~:220-266; form-state builder `build_library_ingest_state` :294)
- Modify: `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (form compose ~:46-140; queue rows :162-235)
- Modify: `tldw_chatbook/Home/active_work_adapter.py` + `tldw_chatbook/Home/dashboard_state.py` (Retry gating; control at :295)
- Modify: CSS source module for `.library-ingest-row-action` (+ `./build_css.sh`)
- Test: `Tests/Library/` ingest state/registry files, `Tests/UI/test_library_shell.py` ingest pilots

**Interfaces (consumed):** W1's `HomeActiveWorkItem.status_detail`, `finished_at_wall`. **Produced:** `LibraryIngestJob.permanent: bool = False`; `mark_failed(job_id, *, error, permanent=False)`; `HomeActiveWorkItem.retry_available: bool = True`.

**M4 — permanent (validation-class) failures don't offer Retry.** Registry: `permanent: bool = False` on the job; `mark_failed` gains `permanent: bool = False` kwarg. Runner (`app.py`): classify at the failure sites — a `ValueError` whose message starts with "Unsupported file type" (raised from `local_file_ingestion.py:81`) and missing-file failures (`FileNotFoundError`, or the runner's own pre-check if one exists) set `permanent=True`; everything else stays retryable. `requeue`/`retry_library_ingest_job` must refuse a permanent job (return None) as defense in depth. Queue row state: `can_retry = failed and not permanent` (`_build_queue_row` :266); Dismiss unchanged. Home: adapter sets `retry_available=not job.permanent` on failed ingest items; the controls builder omits `home-retry` (:295) when the selected item has `retry_available=False`. RED tests both layers: permanent failed job → no Retry button in canvas state, no home-retry control; ordinary failure → both present.

**L4 — short reason in the row; supported list moves to the form.** `_build_queue_row` failed line: derive `short_error` = `job.error.split(" Supported types:")[0].rstrip()` (covers the "…: .xyz. Supported types: …" copy; a plain error without the marker passes through whole). Row line uses `short_error`. Form: `build_library_ingest_state` adds `supported_types_line: str` built dynamically from `get_supported_extensions()` (`tldw_chatbook/Local_Ingestion/local_file_ingestion.py`) — uppercase keys, comma-joined, prefixed "Supported: " — NOT hardcoded (A2 lesson). Canvas renders it as a muted quiet-line Static (reuse `library-ingest-quiet-line` class) directly under the Browse… button, always visible. Tests: failed row line lacks "Supported types:"; form state's line contains "TXT" and no "XML".

**L5 — row actions on one line.** In `library_ingest_canvas.py`, wrap each row's action buttons (Open in Library / Retry / Dismiss, :193-225) in a `Horizontal(classes="library-ingest-row-actions")` container. Button ids unchanged (job_id-keyed). CSS source: `.library-ingest-row-actions { height: auto; }` with the existing per-button margin moved/adjusted so the pair sits on one line with a blank line below (preserve the A3 spacing intent documented at :169-177); rebuild `./build_css.sh` and commit both files. Pilot: failed row → Retry and Dismiss share the same `region.y`.

Run: `Tests/Library/` + ingest pilots in `Tests/UI/test_library_shell.py` + `Tests/Home/`. Commit: `fix(library,home): permanent-failure retry gating, short queue reasons, inline row actions`.

### Task 3: W3 — Library copy & metadata (M2, M3, M5, L1, L2, L3, L6)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` (:103), `tldw_chatbook/Widgets/Library/library_media_viewer.py` (:160-184, `_compose_analysis` :334+)
- Modify: `tldw_chatbook/Library/library_media_viewer_state.py` (:195)
- Modify: `tldw_chatbook/Library/library_media_state.py` (+ the screen-side media snapshot mapping in `tldw_chatbook/UI/Screens/library_screen.py` — trace from `build_library_media_canvas_state` callers)
- Modify: `tldw_chatbook/Library/library_rag_state.py` (`row_badge_label` :655-663; run label :514; scope strip)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (handoff button label; modes dict :178-191)
- Test: `Tests/Library/` state units, `Tests/UI/test_library_shell.py`, `Tests/UI/test_library_content_hub.py`

**M2 — rename cross-surface jumps.** "Open in Media" → "Open in Media manager" at BOTH sites (`library_media_canvas.py:103`, `library_media_viewer.py:178`). "Use in Chat" → "Use in Console" in the Library viewer only (`library_media_viewer.py:166`); if the viewer's blocked-notify copy (grep "Use in Chat requires" `app.py:2011` and `Chat/chat_handoff_messages.py`) is reachable from the Library viewer path, update the user-visible string there to match; legacy surfaces (`MediaWindow_v2`, `search_rag_window`) unchanged. Button ids unchanged. Re-anchor label asserts.

**M3 — list metadata parity with the viewer.** Symptom: list rows show bare "document" and the card "Updated: unknown" while the viewer shows "Updated: 13m" for the same record. Trace where the browse-list records are fetched/mapped in `library_screen.py` (the records feeding `LibraryMediaEntry`; `_first_present_text(record, _UPDATED_KEYS)` at `library_media_state.py:144` finds nothing) and make the fetch/mapping carry the same timestamp key the viewer path uses (:3398 checks created_at/last_modified/updated_at). Fix at the query/mapping site; do NOT widen `_UPDATED_KEYS` speculatively. Keep the bare Console-style age (no "ago" suffix — app-wide convention). RED: seeded record with last_modified → list card shows `Updated: <age>`, row secondary "document · <age>".

**M5 — humanize the evidence badge strip.** `row_badge_label` (:655-663): join with " · " not " | "; keep the source-type badge; drop the workspace badge when it is the default "all workspaces"; include citations only when count > 0; map eligibility to nothing when "eligible" and to "excluded from context" when "blocked". Result: "media", "media · 2 citations", "media · excluded from context". Keep the individual badge properties intact (other call sites/tests). Re-anchor strip asserts.

**L1 — state-dependent analysis action.** `_compose_analysis`: the toggle button reads "Add analysis" when the viewer has no analysis text, "Edit analysis" otherwise (mirror the Read-it-later conditional at :172). Id unchanged.

**L2 — verb the handoff button.** The handoff canvas action button (labeled from `LIBRARY_STUDY_HANDOFF_MODES[kind]["action_label"]`, render site near `library_screen.py:2361`) becomes `f"Continue in Study"` for all study handoff kinds — set via the copy dict (add "button_label": "Continue in Study" to each mode; header/purpose/recovery copy still use action_label). Id unchanged.

**L3 — hide internal URLs.** `library_media_viewer_state.py:195`: skip the `URL:` line when the url starts with `local://`. Unit: local:// record → no URL line; https record → line present.

**L6 — trim search chrome.** Run button label "Run Search/RAG" → "Run" (`library_rag_state.py:514`). The scope strip (the `Scope: all local | Notes 4 | Media 3 | Conversations 4` line — grep its builder in `library_rag_state.py`) drops the per-source counts, keeping "Scope: all local sources" (the checkboxes below already carry counts). Re-anchor asserts.

Run: `Tests/Library/` + `Tests/UI/test_library_shell.py` + `Tests/UI/test_library_content_hub.py` + gate16. Commit: `fix(library): destination-honest labels, metadata parity, human evidence badges, search chrome trim`.

## Verification & gate

Combined final gate = F1's set (Tests/Library, Tests/Home, test_library_shell.py, test_home_screen.py, test_destination_shells.py, gate16, test_media_ingest_window_rebuilt.py, phase3 knowledge-entry, test_library_content_hub.py) + import check. Visual QA re-captures (served TUI, seeded HOME): Home failed-ingest canvas (reason line, no flashcards control, changed next hint), Home Recent with a done import, ingest queue (short reason, inline actions, supported-types helper, no Retry on permanent failure), media list card (Updated age), viewer (labels, no local URL), search results (badges, Run), handoff button. Present captures for user approval; push to PR #592; merge only on explicit authorization.
