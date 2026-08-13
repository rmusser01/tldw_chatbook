---
id: TASK-15458
title: Library media viewer: in-place match navigation instead of full-document re-parse
status: In Progress
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit — plausibly the single worst click in the app: the media viewer holds the entire document as one Markdown widget (`Widgets/Library/library_media_viewer.py:148`), and next/prev match (`library_screen.py:23476`), mode toggle (`:23431`), and search submit (`:23362`) each perform a full-document markdown re-parse plus a whole-screen remount — multi-second per click on a long transcript on constrained hardware. Separately, the media panel's content search re-parses the document twice per keystroke via `Markdown.update("")` followed by `Markdown.update(content)` (`Widgets/Media/media_viewer_panel.py:1000-1025`, reached from `:1223/:1266`), with no debounce.

Fix direction: keep the Markdown widget mounted and move match-highlight and scroll-to-match to in-place updates; drop the empty pre-update; debounce the search box. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Match navigation does not re-parse the document or remount the screen (evidence)
- [ ] #2 Search-while-typing performs at most one deferred re-render per debounce window and never the double update(\"\")/update(content) parse
- [ ] #3 Match highlighting and scroll behavior preserved (tests); click latency before/after on a long document recorded
<!-- AC:END -->

## Implementation Plan

1. Add mounted tests and a focused `LibraryMediaContentSearchControls` widget whose match-only updates preserve navigation identity and focus.
2. Add mounted tests and a lazy `LibraryMediaContentBody` that stores Raw search state, mounts each selected mode at most once, and applies latest-request-wins visibility.
3. Integrate the focused children through `LibraryMediaViewer` and replace Library screen recomposes with narrow query, match-index, mode, and post-layout scroll synchronization.
4. Add mounted legacy-panel tests and a generation-guarded 250 ms content-search debounce; remove the empty Markdown cache-busting update.
5. Record deterministic before/after latency and parse/identity evidence, run focused and full verification, complete rendered keyboard UAT, and close the task documentation.

Detailed plan: `Docs/superpowers/plans/2026-08-12-library-media-viewer-inplace.md`

ADR required: no

ADR path: N/A

Reason: this applies existing Textual widget ownership and timer patterns without changing persistence, security, dependencies, services, or cross-module contracts.
