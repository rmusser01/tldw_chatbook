---
id: TASK-14825
title: >-
  Ingest consistency drift: install commands, retry labels, state vocabulary,
  picker headers
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
updated_date: '2026-08-12 21:12'
labels:
  - library
  - ingest
  - copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 of the 2026-08-10 re-critique — consistency defects accumulated across the two shipped arcs, each captured live. Individually small; together they are why the Consistency heuristic dropped from 3 to 2.

1. **Two install-command forms for one dependency on one screen**: the preflight warning says `pip install -e ".[ebook]"` while the queue row for the same missing dep says `Install with: pip install tldw_chatbook[ebook]`.
2. **Three retry labels coexist**: the per-row `Retry`, the batch `Retry this batch`, and the footer's `r retry last batch`.
3. **State vocabulary drift in the queue**: `2 running` and `2 parsing` describe the same in-flight items on adjacent lines, and item rows use a third form (`● parsing`).
4. **The supported list is written two ways on one screen**: the intro ends `…plain text files, web pages.` while the unsupported-file error ends `…plain text files, web pages (by URL).`
5. **Picker column headers misaligned with their own columns** (measured by column index): the `Size` header occupies cols 185–188 while values right-align to 186; the `Modified` header starts at col 201 while its dates start at 191 — the header sits 10 columns right of its column's left edge, visually labelling the HH:MM half. The header row also sits outside the bordered list box. This is inside the header row task-3304 added.
6. **Mid-sentence detail fragment**: `✗ failed · empty_file.txt · is empty; there was nothing to ingest.` — the basename-echo strip removes the subject, leaving a sentence starting with "is".
7. **Collapsed titles advertise values of disabled fields**: `Images — Extract text (OCR): on` while the control itself reads `— needs OCR backend installed`; `build_type_group_title` filters on emptiness and default-ness but never consults `field_disabled_state`.
8. Raw Python exceptions reach row lines (`Failed to ingest pdf file: 'NoneType' object has no attribute 'FileDataError'`), and nested tool prefixes stack four deep before the actual cause.
9. The consent copy says "Press **Start** again" even when the user armed it with Enter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One install-command form is used wherever a missing dependency is reported
- [ ] #2 Retry affordances share one vocabulary across row, batch and footer
- [x] #3 The queue describes an in-flight job with one word across tally, batch header and row
- [ ] #4 The supported-format list has one wording, from one source
- [x] #5 Picker column headers align with the values they label
- [ ] #6 A failure detail reads as a complete sentence regardless of basename stripping
- [x] #7 Collapsed panel titles do not advertise values of fields that are disabled
- [ ] #8 A raw exception repr never reaches a queue row line; the user-facing cause comes first
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Split the nine findings by owning file. Items 5 and 7 live in files this agent owns (``Third_Party/textual_fspicker``, the canvas); the rest live in ``Library/library_ingest_state.py``, ``Local_Ingestion/*`` and ``UI/Screens/library_screen.py``, which a sibling agent held for the same arc.
2. #5: measure the header/data column offset in CELLS from the compositor (character indices lie -- the listing's icon is one character, two cells), fix both contributing insets, and pin it with AND without a scrollbar.
3. #7: consult ``field_disabled_state`` in ``build_type_group_title`` so a gated field contributes no ``name: value`` pair.
4. Report the remaining seven with file, symbol and the change to make.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Done here (#5, #7).**

**#5 Picker headers.** Measured from compositor strips in CELL columns:
the ``Size`` header ended one cell right of its values with no scrollbar
and THREE cells right with one. Two insets stacked, and the original
header padding compensated for only the first: ``DirectoryNavigation``'s
``border: blank`` (1 cell each side) plus ``OptionList``'s own default
``padding: 0 1`` (1 more each side), then the vertical scrollbar (2 cells,
right only) the moment the list overflows -- shipped as "cosmetic and
accepted", which a user reads as a broken table. Fix in
``base_dialog.py``'s ``DEFAULT_CSS``: header ``padding: 0 4 0 2`` and
``scrollbar-gutter: stable`` on ``DirectoryNavigation``, so the gutter is
reserved whether or not the scrollbar shows and one static header can be
right in both states. Test is parameterised over a 3-file and a 60-file
directory with a precondition asserting each fixture really exercises its
scrollbar case.

**#7 Collapsed titles.** ``build_type_group_title`` now runs
``field_disabled_state`` per field and skips disabled fields entirely, so
``Enable OCR: off`` can no longer sit above a control reading "— needs the
docext engine". Mutation: removing the skip restores ``PDF engine: …,
Enable OCR: off`` in a fully gated group's title.

**Not done here -- root cause is in a sibling-held file (verbatim
handoff):**
- #1 two install-command forms: the preflight form is
  ``OptionalFeatureInfo.source_install_command`` (``pip install -e
  ".[x]"``); the queue-row form comes from ``ImportError`` strings in
  ``Local_Ingestion/PDF_Processing_Lib.py`` (:80, :238),
  ``Book_Ingestion_Lib.py`` (:148, :371, :489, :1662) and
  ``web_article_ingestion.py`` (:142), which all hand-write ``pip install
  tldw_chatbook[x]``. Route them through
  ``OptionalFeatureInfo.source_install_command`` (or strip the command
  from the exception and let the row renderer add the canonical one).
- #2 three retry labels: row ``"Retry"``
  (``library_ingest_canvas.py``, ``library-ingest-retry-{job_id}``), batch
  ``LIBRARY_INGEST_RETRY_LABEL = "Retry this batch"``
  (``library_ingest_state.py``:~570) and the footer binding description
  ``"retry last batch"`` (``library_screen.py`` BINDINGS). Changing only
  the row one makes it worse -- these must move together.
- #3 ``running`` vs ``parsing``: ``library_ingest_state.py``:1797 and
  :1858 emit ``"{n} running"`` while row lines use ``"● parsing"``
  (:1255). Pick one verb for tally, batch header and row.
- #4 supported list written twice: ``SUPPORTED_FORMATS_COPY``
  (``library_ingest_state.py``:76) ends "web pages (by URL)." while
  ``INGEST_INTRO_WHAT_COPY`` (:240) composes from ``_TYPE_GROUP_LABELS``
  and ends "web pages.". One source.
- #6 mid-sentence fragment: ``_strip_basename_echo``
  (``library_ingest_state.py``:395) removes the leading basename and
  leaves "is empty; there was nothing to ingest." Re-attach a subject
  (or only strip when what remains still parses as a sentence).
- #8 raw exception reprs in row lines: the row-line/error path in
  ``library_ingest_state.py`` (``unwrap_ingest_error`` and its callers) --
  ``'NoneType' object has no attribute 'FileDataError'`` reached a row,
  and nested tool prefixes stack four deep before the cause.
- #9 "Press **Start** again" when armed by Enter:
  ``INGEST_START_CONFIRM_PREFIX`` (``library_ingest_state.py``:815). The
  copy names a control the user did not use.

Files touched: ``Third_Party/textual_fspicker/base_dialog.py``,
``Widgets/Library/library_ingest_canvas.py``,
``Tests/UI/test_fspicker_listing_columns.py``,
``Tests/UI/test_library_ingest_canvas.py``,
``Docs/User_Guide/library/import-and-export.md``.

TASK-15702 follow-up: the remaining queue-vocabulary drift is resolved. Batch headers now use the neutral phase label active and enumerate the actual queued, parsing, and writing states instead of appending a second derived N running tally. Rows and queue totals keep those exact state names.
<!-- SECTION:NOTES:END -->
