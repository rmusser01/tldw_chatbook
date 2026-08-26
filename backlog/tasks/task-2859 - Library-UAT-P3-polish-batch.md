---
id: TASK-2859
title: Library UAT P3 polish batch
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 21:58'
labels:
  - library
  - polish
  - uat-2026-08-06
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 P3 findings, critique snapshot
`.impeccable/critique/2026-08-07T01-01-42Z__tldw-chatbook-ui-screens-library-screen-py.md`,
observed at dev `6ffa56516`. One polish pass; none block tasks individually.

1. Conversations canvas has no title header (top row is "Export… / Select"; siblings show
   "Name (n)"); its filter input renders below the empty-state text instead of above.
2. Rail gloss "Prompts (0) — AI asks" is cryptic; a freshly created prompt is stamped "legacy".
3. Export quality caption describes the non-selected option ("quality: thumbnail ▸" captioned
   with the "original copies full media files…" explanation).
4. Ingest queue summary "3 done — in queue" self-contradicts (done vs in queue).
5. Details disclosure: clicking the "Details" label does nothing (only the ▸ chip toggles);
   content wraps mid-unit ("Prompts 144.0 / KB"); "Status" renders as a bare heading with no
   value; DB sizes exclude -wal files (reported 4.0 KB while the WAL held ~4 MB).
6. File/folder pickers are dev-flavored: raw sizes ("30624"), second-precision timestamps,
   $HOME default with no recent/suggested locations.
7. Canvas title grammar drifts: "Media (3)" vs "Library Collections"/"Library Search/RAG"; rail
   says "Search / RAG", canvas says "Search/RAG".
8. sort: defaults differ across siblings (Notes/Prompts "Newest", Skills "Name") with no system.
9. Skill editor copy "Not applied in v1 — shown for SKILL.md round-tripping only" is
   internal-version talk.
10. Search results lack a "N results for 'query'" headline; media evidence snippet text sits
    flush against the card border (missing left pad).
11. The toast panel overlaps the ingest queue area.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each numbered item is fixed, or declined with a one-line reason in the notes
- [x] #2 Copy changes keep the DESIGN.md voice (plain language, labels carry meaning)
- [x] #3 Touched surfaces are re-verified live at 170×50
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify all 11 findings at HEAD (dev 023a04a48 + tasks 1-2) via code reads and a live tmux pass before touching anything -- two Library UAT arcs landed since the critique snapshot.
2. For each item, decide fix vs decline; TDD the fixes (RED test against current behavior, implement, GREEN), preferring rendered-geometry assertions for anything layout-visible.
3. Live-verify every touched surface at 170x50 via tmux (palette nav, SGR clicks by character column).
4. Run targeted test suites for every touched module plus a Tests/Library --collect-only sweep as collection sanity.
5. Regenerate the CSS bundle (build_css.py) and confirm check_bundle_sync.py for the one CSS change (item 10 snippet padding).
6. Backlog hygiene: 11-line disposition table in notes, AC checkboxes, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Disposition table (11 items, re-verified at HEAD before touching anything):

1. FIXED — Conversations canvas now opens with a "Conversations (N)" title Static matching the sibling "Name (n)" pattern; its filter Input moved above the empty-state/status text (was below).
2. FIXED — rail gloss "Prompts (0) — AI asks" reworded to "Prompts (0) — reuse" (plain language; two longer drafts silently dropped at the rail's real 170x50 width — F-015's "whole or not at all" rule); a fresh prompt's artifact-status line now reads "text format" via a new definition_state_display_label() map, not the internal "legacy" value.
3. FIXED — the export quality helper line is now per-option (thumbnail/compressed/original each get their own honest caption) instead of one fixed "original copies full media files…" sentence shown regardless of selection.
4. FIXED — ingest queue tally reads "This queue: N done" (a leading scope label) instead of the self-contradicting trailing "N done — in queue" suffix.
5. FIXED (3 of 4 sub-parts; 1 declined) — clicking the "Details" LABEL now toggles the section (previously only the chip did; root cause was a genuine Textual trap — Message._sender inherits the CURRENT dispatch context's active_message_pump, so Button.press() called from an ancestor's own click handler gets misattributed as "parent is sender" and MessagePump._on_message stops the Pressed message's bubble one hop early; fixed by resetting the contextvar around the press() call). DB sizes now include -wal/-shm sidecars (get_formatted_db_size_with_wal). The "144.0 KB" mid-unit wrap is fixed by dropping the space entirely, not by a non-breaking space — NBSP does NOT work here: Rich's own word-wrap splitter's `\s` regex matches U+00A0 same as an ordinary space (verified directly against rich._wrap at the rail's actual width). DECLINED: the bare "Status" heading is one of three deliberate, already-tested parallel group headers (Status/Workspace/Actions, see test_library_details_section_renders_grouped_headers_and_drops_policy_prose) — singling it out for a value would break that parallelism, and the rows beneath it (Source/counts/DB sizes) are not actually valueless once you read past the header.
6. FIXED (bounded; 1 sub-part declined) — the vendored fspicker's DirectoryEntry._size/_mtime now show human-readable sizes ("29.9 KB" not "30624") and minute-precision timestamps (no seconds). DECLINED: $HOME default + recent/suggested locations is a real feature (persisted state + new UI), too large for a vendored-library polish item — needs its own task.
7. FIXED — every canvas title now matches "Name (n)" (Collections: "Library Collections" -> "Collections (N)"); Search/RAG canvas: "Library Search/RAG" -> "Search / RAG", matching the rail row's own spacing. The separate, cross-app "Library Search/RAG" evidence-provenance label (OWNER_LIBRARY_RAG / Console staged-evidence source=) was NOT touched -- different vocabulary, same string by coincidence.
8. DECLINED — Skills has no timestamp field to sort by (installed skills aren't authored/dated content the way Notes/Prompts are); alphabetical is the defensible default for a smaller, install-based list. Unifying would require plumbing a new field, out of scope for a polish pass.
9. FIXED — skill editor's Model override hint reworded from "Not applied in v1 — shown for SKILL.md round-tripping only." to "Not used when running this skill — kept so saving doesn't lose the value."
10. FIXED — Evidence region gets a new "N results for 'query'." headline above the result cards (tracks the currently-visible, scope-filtered rows -- deliberately independent of the separate "Evidence · top N" heading, which task-8's own test pins as NOT row-count-driven); the snippet's left padding now matches its title/badges siblings (CSS-only, `.library-rag-result-snippet { padding: 0 1; }`).
11. ALREADY FIXED at HEAD (by task-2130) — live-verified at 170x50 with a single import and again with 6 stacked imports/toasts: the toast's right anchor plus the existing `Toast { margin-bottom: 2 }` rule keeps it clear of the ingest queue column and the canvas's bottom border in every capture.

Tests: 9 new/modified test files, ~50 new test cases, all TDD'd RED->GREEN where behavior changed (destination_rail click-bubbling fix, size-format space removal, queue-line wording, quality captions, fspicker formatting, rag results count line + snippet padding via a dedicated LibraryHarness CSS-bundle-loading test). Full targeted run: 657 passed / 1 pre-existing ambient failure (test_action_library_skill_back_honors_dirty_guard, A/B-confirmed identical at clean HEAD, unrelated file). test_library_shell.py in full: 367 passed. Tests/Library --collect-only: 1118 collected, 0 errors.

CSS: one bundle change (.library-rag-result-snippet padding), source-of-truth edited in components/_agentic_terminal.tcss, bundle regenerated via build_css.py, check_bundle_sync.py green.

Docs: Docs/User_Guide/library.md + library/{collections,search-and-rag,skills,import-and-export,media-and-conversations}.md updated with new copy and fresh "Verified against dev @ 023a04a48" stamps.

Notable lesson (destination_rail.py): calling Button.press() from an ancestor widget's own bubbled-Click handler silently breaks message propagation one hop early unless the active_message_pump contextvar is reset to the button first -- Textual's own "parent is sender" bubble-stop optimization misfires when the message is constructed while executing AS the parent. Documented at length in the fix's docstring and in the destination_rail test file.

Notable lesson (Rich text wrapping): a non-breaking space (U+00A0) does NOT prevent Rich's plain-Static word-wrap from splitting at it -- rich._wrap.words() tokenizes on a Unicode-aware `\s` regex that treats U+00A0 as ordinary whitespace. Removing the space entirely is the only fix verified to work at every relevant terminal width; confirmed directly against rich._wrap, not just visually.

Notable lesson (CSS testing): DestinationHarness (used by most of test_library_content_hub.py) hosts screens under a bare Textual App with no CSS_PATH -- only widget-level DEFAULT_CSS Python blocks load, never the app bundle (css/tldw_cli_modular.tcss). A bundle-only CSS rule (e.g. the snippet padding fix) is invisible to rendered-geometry assertions there; LibraryHarness (test_library_shell.py) is the one Library harness that sets CSS_PATH to the real bundle and must be used for that class of check.
<!-- SECTION:NOTES:END -->
