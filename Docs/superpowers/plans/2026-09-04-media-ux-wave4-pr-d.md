# Media UX fix wave 4 — PR D (set-level analysis path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Library ▸ Media a set-level analysis path: a bulk Analyze action in Select mode, a single "Analyze skipped" action after an import run, honest per-item receipts, no silent overwrites, a Generate button that says why it is off, and an Import-behavior header that shows its analysis state.

**Architecture:** One worker group runs the existing per-item generator (`_generate_library_media_analysis`, `library_screen.py:~41694`, which already resolves the provider via `resolve_ingest_analysis_provider` and persists through `save_analysis_version`) over an ordered id list, reporting into an in-list receipt in the same two-row grammar the delete/dismiss receipts use (PR A). Provider readiness is computed once per gesture and rendered as a disabled-with-reason label (the product's `○` grammar via `library_disabled_action_label`). No new provider plumbing; no change to the five-key summary contract.

**Tech Stack:** Python 3.12, Textual 8.x, pytest + pytest-asyncio; `LibraryProductionCSSHarness`, `ControlledDetailMediaService` / `_flow_app` (`Tests/UI/test_library_media_reader_flow.py`), `_painted` (`Tests/UI/test_library_media_render_fixes.py`), the ingest-run tests under `Tests/UI/test_library_ingest*.py` (grep) for AC#1/#2.

**Spec:** critique #4 priority issue 5 (`.impeccable/critique/2026-09-04T13-50-05Z…md`) and `backlog/tasks/task-28007 - …md` (ACs 1-6 are binding). `task-28018` is parked: its recon records that no in-app Settings control exists for the analysis provider — a Settings-IA decision for the user, not a fix.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-wave4-d`, branch `fix/media-wave4-d`, based on the PR-C head (its receipts, chrome and Trash changes are in the tree). Every command: `cd <worktree> && PYTHONPATH=<worktree> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -p no:cacheprovider`; absolute paths; UI test files in separate processes; compare failures against the base before claiming them (`test_library_shell.py` known failures: `backlog/tasks/task-31249 - …md`).
- No new `logger.*` calls without regenerating `Docs/security/production-diagnostic-inventory.json` (`python scripts/check_persistent_diagnostic_inventory.py --write`, commit it). After any `BUNDLED_CSS` / component TCSS edit: `python -m tldw_chatbook.css.build_css` + `python tldw_chatbook/css/check_bundle_sync.py` (exit 0); commit regenerated files.
- Workers: `run_worker(..., group=<named constant>, exclusive=True, exit_on_error=False)`; never `exclusive=True` without a group; the media action toolbar is at its width floor (task-28025/30043) — a NEW button must go on a row that has budget (the select-mode bulk row `_select_mode_bulk_buttons`, `library_media_canvas.py:399`, after Export/Review; the danger row stays Delete-only) and be proven by a painted-text test at 235x52.
- The five-key media summary contract is frozen; do not touch review-set code or the Find focus token.
- Receipts use PR A's two-row grammar (`.library-media-receipt` / `-copy` / `-actions` classes) with unique ids; copy names counts and the recovery path.
- Commit per task with the trailers `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_011LebG4HPfSniVohbuXkU4n`. Backlog task files are flipped by the controller.

---

### Task 1: Generate says why it is off; the Import-behavior header shows its analysis state (task-28007 AC#5, AC#6)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py` (the Analysis-tab action row: `Button("Regenerate" if self.viewer.analysis else "Generate", id="library-media-analysis-generate", …)` ~:640-660) — accept an `analysis_provider_reason: str` constructor arg; when non-empty render `library_disabled_action_label("Generate", True)` / `"○ Regenerate"`, `disabled=True`, `tooltip=reason`. Thread the arg through the screen's viewer construction and `_sync_library_media_viewer_state`'s compare/assign (the #2351 trap).
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — compute the reason once per viewer sync from `resolve_ingest_analysis_provider(self.app_instance.app_config)` (the same call `handle_library_media_analysis_generate` makes at ~:41667; reuse its resolution → reason mapping, e.g. the resolution's `hint`/`reason` field — read `IngestAnalysisResolution` at `tldw_chatbook/Library/ingest_analysis.py:89`); keep the handler's post-click guard as belt-and-braces.
- Modify: `tldw_chatbook/Library/ingest_capabilities.py:1006-1020` — the collapsed `label="Import behavior"` group summarises its state: `Import behavior · analysis on` / `· analysis off` (derive from the "Analyze after import" toggle value at compose time; grep how the group label reaches the Import canvas widget and where the toggle state lives).
- Test: `Tests/UI/test_library_media_render_fixes.py` (Generate disabled + tooltip on an item with no provider configured — the test app config has none; and enabled when a fake resolution is ready via monkeypatching `resolve_ingest_analysis_provider` in the screen module), `Tests/Library/test_ingest_capabilities.py` (grep; header label carries the state for both toggle values).

**Interfaces:**
- Produces: viewer arg `analysis_provider_reason`; a screen helper `_library_media_analysis_provider_reason() -> str` (used by Task 2's bulk gate too); header label strings `Import behavior · analysis on|off`.

- [ ] Step 1: failing tests (three: Generate disabled+reason; Generate enabled with a ready resolution; header label both states).
- [ ] Step 2: run; confirm each fails (button enabled today; header label bare).
- [ ] Step 3: implement per Interfaces; rebuild CSS only if you touch it.
- [ ] Step 4: run `test_library_media_render_fixes.py`, `test_library_media_reader_flow.py`, `Tests/Library/test_ingest_capabilities.py`, `test_library_shell.py -k "analysis or ingest or import"` (compare to base).
- [ ] Step 5: live tmux 235x52: Analysis tab on an item with no provider → `○ Generate` with tooltip; Import canvas → collapsed header reads `Import behavior · analysis off`.
- [ ] Step 6: commit `feat(library): Generate says why it is off; Import behavior header shows its analysis state (task-28007)`.

---

### Task 2: Bulk Analyze in Select mode with an in-list receipt (task-28007 AC#3, AC#4)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` — `_select_mode_bulk_buttons` (:399) gains `Analyze` (id `library-media-analyze-selected`, class `library-canvas-action`, gated like Export/Review via `_bulk_action_button` and disabled with Task 1's reason when the provider is not ready); a new receipt block (id `library-media-analyze-receipt`, two-row grammar) driven by canvas state fields `analyze_receipt_total`, `analyze_receipt_done`, `analyze_receipt_failed`, `analyze_receipt_running` (add to `LibraryMediaCanvasState` in `tldw_chatbook/Library/library_media_state.py` next to `delete_receipt_count` ~:767/945/1031, defaulted → safe) with copy `Analyzing 3 of 40 · 2 failed` while running and `✓ analyzed · 38 of 40 · 2 failed` when done, actions `Retry failed` and `Dismiss`.
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — handler `handle_library_media_analyze_selected` (Button.Pressed on the new id): snapshot the selection in browse order, exit select mode (like Review-selected, task-31233), start ONE worker in group `_ANALYZE_SELECTED_WORKER_GROUP = "library_media_analyze_selected"` (`exclusive=True, exit_on_error=False`) that iterates the ids, skipping items that already have an analysis unless the gesture was `Analyze (overwrite)` — AC#3: the first press on a selection containing analysed items arms a one-line choice `N already analysed — Skip them | Overwrite` in the receipt row (no modal); calls `_generate_library_media_analysis` per id (await it; catch per-item exceptions → failed count, never abort the run), updates the receipt counts in place after every item (`_sync_library_canvas(self, "media")` or the in-place receipt patch helper if one exists), and on completion leaves the final receipt; `Retry failed` re-runs only the failed ids; `Dismiss` clears the receipt.
- Test: `Tests/UI/test_library_multiselect_media.py` (fake-driven: handler snapshots ids in browse order, exits select mode, starts exactly one worker in the named group; per-item failure increments failed and continues; already-analysed items skipped by default; overwrite path generates them) — use `_media_fake` + `_bind_media_mutation_seams` and a stub `_generate_library_media_analysis` recording calls; `Tests/UI/test_library_media_render_fixes.py` (painted receipt at 235x52 with both actions readable; `Analyze` button visible on the bulk row without pushing Export/Review off-pane — painted-text).

**Interfaces:**
- Consumes: Task 1's `_library_media_analysis_provider_reason()`; PR A's receipt CSS classes; `_generate_library_media_analysis(media_id, …)` (read its signature at ~:41694 — it may take the backing id and a `detail`; adapt the loop to load what it needs via the reading service, off-loop).
- Produces: worker group constant; canvas state fields; receipt ids.

- [ ] Step 1: failing tests (handler contract ×4; painted receipt; painted bulk row).
- [ ] Step 2: run; confirm failures (no handler / no button / no receipt).
- [ ] Step 3: implement; rebuild CSS for the receipt id if any new rule is needed (prefer the existing classes).
- [ ] Step 4: run `test_library_multiselect_media.py`, `test_library_media_render_fixes.py`, `test_library_media_side_by_side.py`, `test_library_shell.py -k "select or analy"` (compare to base); `python scripts/check_persistent_diagnostic_inventory.py` (no pipe).
- [ ] Step 5: live tmux 235x52 with a stub provider (if no real provider is configured, verify the disabled-with-reason state and the receipt copy with a monkeypatched generator in an app-test instead — say which in the report).
- [ ] Step 6: commit `feat(library): bulk Analyze in Select mode with an honest in-list receipt (task-28007)`.

---

### Task 3: "Analyze skipped" after an import run (task-28007 AC#1, AC#2)

**Files:**
- Read: the import-run completion surface in the Library Import canvas (grep `analysis skipped` / `analysis_skipped` under `tldw_chatbook/Library/ingest_*.py` and `tldw_chatbook/Widgets/Library/library_ingest*.py`) — the per-row receipt "analysis skipped: no analysis provider is configured" and the run summary row.
- Modify: the run-summary row gains one action `Analyze N skipped` (id `library-ingest-analyze-skipped`) when the run has analysis-skipped items and the provider is now ready (Task 1's reason is empty); it reuses Task 2's worker over those ids and reports per item in the SAME receipt style as import rows (`✓ analyzed · <title>` / `✕ analysis failed · <title> · <reason>` next to each row, or in the run summary if rows are not individually addressable — say which).
- Test: the ingest-run UI tests (grep the existing "analysis skipped" tests and extend them): the action appears only with skipped items + a ready provider; pressing it runs the worker over exactly the skipped ids; per-item outcomes render in the receipt grammar.

- [ ] Step 1: failing tests (visibility gate ×2; ids passed; outcome rendering).
- [ ] Step 2: run; confirm.
- [ ] Step 3: implement, reusing Task 2's worker and Task 1's reason.
- [ ] Step 4: run the ingest UI test files (separate processes) + `test_library_shell.py -k "ingest or import"` (compare to base).
- [ ] Step 5: live tmux: import two small text files with Analyze-after-import ON and no provider → rows say skipped → configure/stub a provider → `Analyze 2 skipped` → receipts.
- [ ] Step 6: commit `feat(library): one action analyses every import row that was skipped (task-28007)`.
