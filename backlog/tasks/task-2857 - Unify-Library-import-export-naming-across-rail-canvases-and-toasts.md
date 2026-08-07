---
id: TASK-2857
title: 'Unify Library import/export naming across rail, canvases and toasts'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 04:58'
labels:
  - library
  - ux-copy
  - consistency
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-10; extends the 2026-08-04 critique's naming P1; observed at dev
`6ffa56516`).

One flow, five names: rail button "Add content…" → canvas titled "Import media" → Media empty
state "Ingest something to see it here." → button "Start ingest" → toast "Ingest finished — 1
imported". Siblings use "Import note" (Notes toolbar) and "Import…" (Prompts/Skills). On the
export side: rail "Export" → canvas "Export chatbook" ("chatbook" appears nowhere else in the
UI) → media detail action "Open in Media manager" (a surface never named anywhere else).

First-time users wonder whether these are different features; the naming breaks recognition on
every return visit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One verb pair is chosen (recommend Import/Export) and used consistently across rail rows, canvas titles, empty states, buttons and toasts for the same concept
- [x] #2 "chatbook" is either introduced with a one-line explainer where it appears or replaced (e.g. "Export bundle (.zip)")
- [x] #3 "Open in Media manager" names the surface it actually opens using that surface's own name
- [x] #4 A naming inventory in the task notes lists every changed string (rail/canvas/toast/tooltip), and the user guide pages citing these labels are updated or re-stamped
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep-audit every user-facing Library string containing ingest/chatbook/Media manager across Library/*.py, Widgets/Library/*.py, UI/Screens/library_screen.py -- classify each as identifier (leave) vs UI copy (rename), building the full inventory before touching code.
2. Rename to the Import/Export verb pair: rail row 'Add content...' -> 'Import...'; ingest canvas Start button 'Start ingest' -> 'Start import'; guardrail modal copy + 'Start ingest anyway' -> import wording; Media empty-state 'Ingest something...' -> 'Import something...'; toast 'Ingest finished' -> 'Import finished'; queue copy (INGEST_UNAVAILABLE_COPY, QUEUE_EMPTY_COPY, 'Recent ingests', 'Analyze after ingest' checkbox label) -> Import wording.
3. Export side: EXPORT_HEADER_COPY/EXPORT_BUTTON_COPY 'Export chatbook' -> 'Export bundle (.zip)'; success toast 'Exported chatbook to' -> 'Exported bundle to'; default export filename fallback 'chatbook' -> 'bundle'. Leave the Chatbooks product/service identifiers (ChatbookCreator, export_chatbook, create_chatbook, the default artifact name fallback) untouched -- product-level naming, not a Library rail/canvas/toast/tooltip label.
4. 'Open in Media manager' -> 'Open in Library > Media' (task-2851 already retired the legacy Media route alias, so this button now round-trips into Library; correct the stale docstring claiming it 'genuinely navigates away').
5. Add/adjust a test pinning the canonical verb pair agreement (rail button + canvas title + toast all say Import) on the primary import flow; update every existing test assertion hardcoding an old string (Tests/Library + Tests/UI); leave test behavior/coverage otherwise intact.
6. Re-stamp/update Docs/User_Guide/library.md + library/import-and-export.md + library/media-and-conversations.md + library/notes.md for every quoted label that changed.
7. Run targeted pytest + --collect-only over Tests/Library; live-tmux verify the Import flow and Export flow end-to-end.
8. Record the full old->new string inventory in Implementation Notes; commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Chosen verb pair: **Import / Export**. "ingest" survives only in code identifiers (ids,
classes, config keys, function/variable names, DB fields) -- never in a rendered string.
"chatbook" stays as the product/service name in `Chatbooks/`, `ChatbookCreator`,
`export_chatbook`/`create_chatbook`, and the created artifact's own default metadata name
(`_safe_text(form.get("name", ""), "Chatbook", ...)` at `library_screen.py` -- deliberately
untouched: that's the exported bundle's own type-appropriate default title, consistent
elsewhere in Artifacts/Home, not a Library rail/canvas/toast label).

### String inventory (old -> new)

Rail / top button / landing hub / command palette:
- `library_shell_state.py` row title "Add content…" -> "Import…"
- `library_screen.py` rail-top primary button "Add content…" -> "Import…"
- `library_screen.py` landing-hub action button "Add content…" -> "Import…"
- `library_screen.py` footer hint tuple `("i", "add content")` -> `("i", "import content")`
  (`LIBRARY_LANDING_SHORTCUTS`; rendered in the landing footer as "i add content")
- `app.py` `LibraryIngestProvider.COMMANDS` "Library: Add content…" -> "Library: Import…";
  help "Open Library and add content" -> "Open Library and import content"
- `app.py` palette-action toast "Opened Library to ingest content" -> "...to import content"
- `app.py` palette-action error toast "Failed to open Library ingest: {e}" -> "...import: {e}"

Ingest canvas (still named "Import media" -- unchanged, already correct):
- `library_ingest_canvas.py` Start button "Start ingest" -> "Start import"
- `library_ingest_canvas.py` "Recent ingests" collapsible title -> "Recent imports"
- `library_ingest_state.py` `INGEST_UNAVAILABLE_COPY` "Ingest is unavailable in this
  runtime." -> "Import is unavailable in this runtime."
- `library_ingest_state.py` `QUEUE_EMPTY_COPY` "No ingest jobs yet." -> "No import jobs yet."
- `ingest_capabilities.py` checkbox `label="Analyze after ingest"` -> "Analyze after import"
  (generic/Plain-text-HTML type group)
- `library_screen.py` `IngestGuardrailModal` "Some files may fail to ingest:" -> "...import:";
  "Start ingest anyway" -> "Start import anyway"
- `library_screen.py` completion toast "Ingest finished — " -> "Import finished — "
- `library_screen.py` "Cancelling a server ingest is unavailable in this runtime." ->
  "...server import is unavailable..."
- `app.py` per-job progress detail `f"Ingested {source_name}"` -> `f"Imported {source_name}"`
  (rendered in the queue row, e.g. "Ingested report.txt")
- `app.py` empty-folder job error "No files to ingest were found in this folder." ->
  "No files to import were found in this folder."
- `app.py` generic parse-failure fallback "Library ingest parsing failed." (both call sites)
  -> "Library import parsing failed."
- `app.py` broken-pool job error "Library ingest parse pool failed unexpectedly; retry to
  resume." -> "Library import parse pool failed unexpectedly; retry to resume."
- `app.py` server-ingest failure text: "No server backend is configured, so this ingest
  cannot run..." -> "...this import cannot run..."; "The server refused the ingest: {exc}"
  -> "...import: {exc}"; "The server rejected the ingest: {errors[0]}" -> "...import:
  {errors[0]}"; "Could not reach the server to cancel that ingest." -> "...that import."
- **(round 2, review finding)** `library_screen.py:14231` `_resolve_ingest_source`'s
  blank-path warning "Please choose a file to ingest." -> "Please choose a file to
  import." -- fires from `_submit_library_ingest_form` (shared by the Start button and
  Enter-in-path-field); every sibling warning on this form already said "import", this
  one was missed in the first pass. New test:
  `test_submit_with_blank_path_warns_to_import_not_ingest`.
- **(round 3, re-review finding)** `web_clip_request.py:121` `NotAWebClipSource`'s
  message `f"{text!r} is not a web page; a file or media URL is ingested, not clipped."`
  -> "...is imported, not clipped." -- raised when a local file/media URL is submitted to
  the web-clip mapper, caught by `app.py`'s `_submit_web_clip_job` (`except
  NotAWebClipSource as exc: ... mark_failed(job.job_id, error=str(exc), ...)`), rendered
  UNCHANGED into the failed queue row via `short_ingest_error()`/`unwrap_ingest_error()`
  (`library_ingest_state.py`'s `line=f"... {short_error}..."`) -- a genuinely visible,
  in-scope string two hops from its raise site that both round-1 and round-2's sweeps
  missed (neither traced exception messages through `mark_failed(error=str(exc))`).
  New test: `test_failed_row_line_for_a_non_clippable_source_says_imported_not_ingested`
  in `test_library_ingest_state.py`, which calls the REAL `build_web_clip_kwargs`,
  captures the REAL exception text, and renders it through the REAL queue-row builder --
  proving the fix end to end, not just at the raise site.
- **(round 3, re-review finding)** `server_ingest_request.py:73-76` `_ELSEWHERE["article"]`
  -- same `mark_failed` sink via `ServerIngestUnsupported`, caught in `app.py`'s
  `_submit_server_ingest_job`. The leading clause "A web page is clipped rather than
  ingested as a media file" -> "...rather than imported as a media file" (plain user
  copy). "the ingest-jobs API" in the trailing clause is KEPT and justified in a new
  comment: it names the server's real, established endpoint family
  (`ServerMediaReadingService.submit_ingest_jobs`/`list_media_ingest_jobs`/
  `cancel_media_ingest_jobs_batch`, pinned by `SERVER_ACCEPTED_MEDIA_TYPES`'s own comment
  as "Established by submitting to a live server") -- a proper noun for an external
  system, not a Library UI label this task governs, same category as the config-key
  carve-out.

Media browse/viewer:
- `library_media_state.py` `LIBRARY_MEDIA_EMPTY_COPY` "No media in your Library yet. Ingest
  something to see it here." -> "...Import something to see it here."
- `library_media_viewer.py` full-viewer action button "Open in Media manager" -> "Open in
  Library ▸ Media" (AC #3: task-2851 already retired the standalone Media screen this used
  to jump to -- "media" now aliases to "library" in `screen_registry._SCREEN_ALIASES` --
  so the button round-trips into Library's own restored state rather than leaving it; live
  tmux check confirmed no crash and the same viewer state on return. The stale docstring
  claiming it "genuinely navigates away" is corrected to describe the alias.)

Export canvas:
- `library_export_state.py` `EXPORT_HEADER_COPY` "Export chatbook" -> "Export bundle (.zip)"
- `library_export_state.py` `EXPORT_BUTTON_COPY` "Export chatbook" -> "Export bundle (.zip)"
  (AC #2: "chatbook" replaced rather than explained, per the recommended text)
- `library_screen.py` success toast `f"Exported chatbook to {path}"` -> `f"Exported bundle
  to {path}"`
- `library_screen.py` default export filename fallback ("chatbook.zip" when the Export
  name field is empty) -> "bundle.zip" (live check: the field's own date-based default,
  "Library export <date>", normally pre-fills it first, so this fallback only fires when a
  user clears the name -- confirmed correct but rarely exercised)
- Library RAG empty-index recovery copy (`library_local_rag_search_service.py`):
  `next_action` "Ingest content to index it automatically..." -> "Import content...";
  `recovery_action` "Library ingest" -> "Library import" (rendered via
  `DestinationRecoveryState.visible_copy`'s "Recovery: …" line); `disabled_tooltip`
  "...Ingest content to index it automatically..." -> "...Import content..."
- **(round 2, review finding)** `library_screen.py:5861` `_marshal_library_export_failure`
  no-service-seam error "Chatbook export service unavailable." -> "Bundle export service
  unavailable." (mirrors the "Export chatbook"->"Export bundle (.zip)" swap already applied
  to the header/button). Renders into the visible `#library-export-error-line` Static;
  pinned by `test_library_shell_export_submit_missing_service_surfaces_error_and_reenables`
  (3 assertion sites updated to match).
- **(round 2, Minor)** `library_export_state.py:27` comment above
  `EXPORT_HEADER_COPY`/`EXPORT_BUTTON_COPY` declared the OLD "Export chatbook" values
  "binding -- see the F4 plan's Global Constraints"; reworded to say task-2857 superseded
  the F4 plan's wording, so a future editor consulting that plan doesn't revert the copy.

### Deliberately left unchanged (non-Library surfaces / product identifiers / dead code)

- `Chatbooks/`, `ChatbookExportManagementWindow.py`, `Docs/User_Guide/home.md` and
  `index.md`'s "Chatbook artifacts" -- a separate, established surface (Artifacts/Home) that
  genuinely deals in Chatbook-typed artifacts; not a Library label.
- `Watchlists_Modules/items_pane.py` status filter option `("Ingested", "ingested")` -- a
  different feature's own status vocabulary (matches a stored `ingested` DB value), not
  connected to Library's Import flow.
- `Home/active_work_adapter.py` "Opening Library ingest job details." -- Home's own message
  about a Library job, outside "rail, canvases and toasts" scope; flagged as a follow-up
  candidate below rather than expanded into out of scope.
- `TabNavigationProvider.TAB_HELP_TEXT[TAB_INGEST]` = "Switch to content ingestion" (and the
  matching `Constants.py` tab-display-name entry) -- traced via `command_palette_tab_ids()`
  (only primary `NAVIGATION_TABS` get a labeled command; legacy folded routes like `ingest`
  surface only as fuzzy-match alias terms via `_destination_alias_terms`, not this dict) --
  appears to be an unreachable/dead entry; left alone rather than guessed at.
- `SERVER_QUIET_LINE_COPY = "ingest runs on Local"` in `library_ingest_state.py` -- explicitly
  commented "Retired"; confirmed zero call sites (grep), never rendered.
- Deep pipeline error text `f"Failed to ingest {file_type} file: ..."` in
  `Local_Ingestion/local_file_ingestion.py` -- shared by callers well outside Library (web
  clip, preflight, `app.py`), and its exact punctuation is load-bearing for
  `short_ingest_error`'s marker-based split; out of this task's blast radius.
- `known_prefix = f"Chatbook created successfully at {path}"` in `library_screen.py` --
  matches `ChatbookCreator`'s own literal return-value prefix (a different module), not
  Library UI text; changing it would break the stripping logic, not rename a label.
- Config keys/identifiers throughout (`library.ingest_backend`, `library.ingest_options.*`,
  `library.ingest_directory_scan_limit`, every `id="library-ingest-*"`/`#ingest-*` DOM id,
  `IngestGuardrailModal`, `LibraryIngestCanvas`, `LibraryIngestJob`, `INGEST_DUPLICATE_
  PROGRESS_PREFIX`, route id `"ingest"` in `screen_registry._SCREEN_ALIASES`) -- identifiers,
  not UI copy, per the task's own instruction.
- **(round 2 re-sweep)** `Widgets/Library/library_file_notes_git_panel.py` +
  `library_file_notes_workspace.py` -- ~14 occurrences of "Chatbook" used as shorthand for
  *this application* (e.g. "STAGED · by Chatbook", "already staged outside Chatbook") in the
  File Notes ▸ Session Git panel. This is a DIFFERENT feature and a DIFFERENT meaning of the
  word (the app's own name, not the export-bundle format) -- the backlog description's claim
  that "'chatbook' appears nowhere else in the UI" is factually wrong, but renaming this is a
  separate, much larger scope decision (what this app calls itself across an entire feature's
  git-safety copy) that does not belong in an import/export-naming task. Left unchanged;
  recommend a follow-up task if this bothers a future reviewer.
- **(round 2 re-sweep)** `app.py:4403` `HomeControlResult(message="This ingest job can no
  longer be retried.", ...)` -- same category as the already-noted "Opening Library ingest
  job details.": a Home-screen message about a Library job, not a Library rail/canvas/toast.
  Left unchanged for the same reason.
- **(round 2 re-sweep)** False positives ruled out by full-text inspection, not just
  regex: `app.py:1616` `.notify("tldw_chatbook - TUI...")` and
  `library_local_rag_search_service.py:894/918` `pip install "tldw_chatbook[embeddings_rag]"`
  both match only because "chatbook" is a substring of the literal PyPI package name
  `tldw_chatbook`; `Library/server_ingest_status.py:40` `"error": IngestJobState.FAILED` is a
  status-code dict key, not a message. None are UI labels for the Import/Export flow.
- **(round 2 re-sweep method)** Re-swept with a Python AST-adjacent, multi-line-aware scan
  (not single-line grep) of every `Static(`/`Button(`/`Collapsible(`/`Input(`/`.notify(`/
  `_notify_library_*(`/`_marshal_library_export_failure(`/`mark_failed(`/`mark_skipped(`/
  `DestinationRecoveryState(` call site in the three target trees, checking the FULL
  (multi-line) call text rather than the line the function name appears on -- this is what
  the first pass's line-scoped grep missed for both review findings.
- **(round 3 re-review finding)** Both custom-sweep scripts (rounds 1 and 2) had a fixed
  pattern list that never included `raise` statements / exception-message construction --
  a genuinely rendered string can be *two hops* from any of the patterns searched (raised
  as an exception -> caught -> `mark_failed(error=str(exc))`), which is exactly what
  `web_clip_request.py`/`server_ingest_request.py` were. Round 3 abandoned the custom
  script and instead: (a) ran the reviewer's literal `grep -rn -i
  'ingest|chatbook|media manager'` across the three target trees (1234 raw hits) filtered
  to lines containing an actual quoted-string pattern (376 hits -- see the fix report for
  the full evidence), and (b) separately enumerated every `class ...Error`/`class
  ...Exception` and `raise` site in the same three trees, reading each message in full.
  Every one of the 376 filtered hits was individually classified as: already-fixed (rounds
  1-3), an identifier (`id=`, `classes=`, config key, DOM selector, dict key mapped to an
  enum, route id), a docstring/comment (not rendered), an internal `logger.debug`/
  `logger.warning` call, a non-Library surface already documented above (File Notes Git
  panel, Home), or a PyPI-package-name substring false positive. One hit,
  `("Importable files", _is_ingestible)` (`library_screen.py:500`, a file-picker Filters
  label), was found ALREADY correct -- it says "Importable", not "Ingestible"; no change
  needed, noted here so it is not mistaken for an unreviewed hit.

### Tests

Updated every existing assertion pinning an old string (`Tests/Library/test_library_ingest_
state.py`, `test_library_media_state.py`, `test_library_export_execution.py`, `test_library_
shell_state.py`, `test_library_ingest_runner.py`; `Tests/UI/test_library_screen.py`,
`test_library_ingest_canvas.py`, `test_library_ingest_guardrail_modal.py`, `test_library_
multiselect_media.py`, `test_command_palette_providers.py`, `test_product_maturity_phase3_
library_contract_layout.py`, `test_unified_shell_phase6_power_user_replay.py`,
`test_product_maturity_phase6_power_user_replay.py`, `test_screen_footer_hints.py`,
`test_library_shell.py`) -- assertions and docstrings both, never deleted a test.

Extended `test_library_shell.py::test_ingest_cta_uses_one_canonical_label_everywhere`
(previously task-2235's) to also open the canvas and check its header agrees. Added a new
end-to-end test, `test_library_shell_import_verb_pair_agrees_across_rail_canvas_and_toast`,
that submits a real file through the rail button, canvas, and Start button, and asserts the
completion toast text -- pinning "rail says Import, canvas says Import media, Start button
says Start import, toast says Import finished" as one regression.

Targeted run (all touched Library/UI/App test files + `--collect-only -q Tests/Library`):
1076 collected, 703+572 passed across two full sub-suite runs, plus `test_library_shell.py`
full file green -- **9 confirmed pre-existing failures, unrelated to this task, verified
against a throwaway HEAD worktree (git worktree add --detach) so they are not misreported as
introduced by this change**:
`test_do_submit_ingest_persists_options`, `test_faster_whisper_recovery_handler_uses_
explicit_provider`, `test_switch_is_not_offered_when_the_server_seam_cannot_submit`,
`test_submit_confirm_guardrail_calls_submit`, `test_submit_without_warnings_calls_submit`,
`test_submit_clears_the_stale_preflight_summary`, `test_phase6_power_user_release_replay_
exposes_fast_repeat_paths`, `test_landing_footer_advertises_the_landing_keyboard_story`,
`test_options_persist_to_config` -- the first 6 + the 9th share one root cause
(`LibraryScreen.__new__`/`object.__new__`-constructed fixtures never set
`_library_ingest_preflight_generation`, added by an unrelated task-2011-era commit); the 7th
is a Home-screen assertion unrelated to Library; the 8th is a footer-text test missing the
`AppFooterStatus.GLOBAL_HINTS` suffix another commit added. None of these fixtures were
touched by this task. Recommend a follow-up backlog task to fix the shared fixture helper.

**Round 2 (review fixes).** Added `Tests/UI/test_library_ingest_guardrail_modal.py::
test_submit_with_blank_path_warns_to_import_not_ingest` -- calls
`_submit_library_ingest_form()` with a blank path and asserts
`_notify_library_ingest_warning` was called once with "Please choose a file to import.",
no job submitted, no modal pushed. Updated the 3 assertion sites in
`test_library_shell_export_submit_missing_service_surfaces_error_and_reenables` (the poll
loop's break condition, the `_library_export_error` check, and the rendered
`#library-export-error-line` Static's text) from "Chatbook export service unavailable." to
"Bundle export service unavailable." Re-ran: the new test (1 passed), the full guardrail
modal file (8 passed, the same 3 pre-existing failures above, unrelated), the export-error
test alone (1 passed), the file's whole export section (`-k export`, 24 passed), and the
combined export+ingest slice of `test_library_shell.py` + `test_library_ingest_state.py` +
`test_library_export_execution.py` (219 passed, 3 pre-existing deselected). Zero new
failures.

**Round 3 (re-review fixes).** No existing test pinned either exact exception-message
string (checked: `grep -rn "is not a web page\|rather than ingested as a media file\|
NotAWebClipSource\|ServerIngestUnsupported" Tests/` -- the two exception classes are
raised-and-caught in several tests, but only one asserts any message substring
(`test_plain_web_page_is_not_a_jobs_api_source` checks `"web page" in
str(excinfo.value).lower()`, which still holds after the rename -- reran it to confirm).
Added `Tests/Library/test_library_ingest_state.py::
test_failed_row_line_for_a_non_clippable_source_says_imported_not_ingested` -- calls the
real `build_web_clip_kwargs("/tmp/notes.txt", options={})`, captures the real
`NotAWebClipSource` message, asserts it says "imported"/not "ingested", then feeds it into
a `FAILED` `LibraryIngestJob` and renders it through `build_library_ingest_state` /
`_build_queue_row`, asserting the rendered `row.line` also says "imported, not clipped"
and never "ingested" -- covering the full raise -> `mark_failed` -> queue-row-render path
the re-review flagged, not just the raise site. Re-ran:
- the new test alone: 1 passed
- `Tests/Library/test_library_ingest_state.py` (full file, includes the new test): 134
  passed
- `Tests/Library/test_web_clip_request.py` + `test_server_ingest_request.py` (both files
  whose exceptions changed): 51 passed
- `test_web_clip_request.py` + `test_server_ingest_request.py` + `test_library_ingest_
  state.py` + `test_library_ingest_runner.py` + `Tests/App/test_submit_library_ingest_
  job.py` + `Tests/integration/test_library_ingest_flow.py` (deselecting the one
  pre-existing failure): 296 passed, 1 deselected
- `--collect-only -q Tests/Library`: 1077 collected (1076 + the one new test)

Zero new failures anywhere across all three rounds.

### Live verification (tmux, socket `sddT5lib<rand>`, scratch profile `/tmp/sddT5`)

Walked Import end-to-end: command palette "Library: Import…" / "Open Library and import
content" -> rail-top "Import…" button -> canvas header "Import media" -> options fold
"Analyze after import: off, ..." -> "Start import" button -> real file imported -> queue
"No import jobs yet." before, done row after. Walked Export: rail "Export" -> canvas header
"Export bundle (.zip)" -> submit button "Export bundle (.zip)" -> Choose destination dialog
(pre-filled date-based name, so the "bundle.zip" fallback wasn't exercised in this run, but
the fallback code path was confirmed by reading). Opened the imported item's full viewer and
confirmed the action row reads "Open in Library ▸ Media"; pressed it and confirmed no crash
and the same Library state on return (matching the documented alias round-trip).

### Modified files

`tldw_chatbook/app.py`, `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Library/
{ingest_capabilities,library_export_state,library_ingest_state,library_local_rag_search_
service,library_media_state,library_shell_state}.py`, `tldw_chatbook/Widgets/Library/
{library_ingest_canvas,library_media_canvas,library_media_viewer}.py`; the 15 test files
listed above; `Docs/User_Guide/library.md` + `library/{import-and-export,media-and-
conversations,notes}.md` (re-stamped `4acb17a0b`).

Round 2 landed inside this same file set -- no new files: `library_screen.py` (2 more
string fixes), `library_export_state.py` (comment), `Tests/UI/test_library_shell.py`
(3 assertion sites) and `Tests/UI/test_library_ingest_guardrail_modal.py` (1 new test).

Round 3 touched two files new to this task: `tldw_chatbook/Library/web_clip_request.py`
(1 string fix) and `tldw_chatbook/Library/server_ingest_request.py` (1 string fix + 1
justifying comment); plus `Tests/Library/test_library_ingest_state.py` (1 new test + the
`pytest` import it needed).
<!-- SECTION:NOTES:END -->
