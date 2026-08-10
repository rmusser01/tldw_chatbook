---
id: TASK-14820
title: >-
  Ingest forecast and consent must come from one truthful computation
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
labels:
  - library
  - ingest
  - ux
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P1 of the 2026-08-10 re-critique (23/40; snapshot `.impeccable/critique/2026-08-10T20-43-44Z__chatbook-widgets-library-library-ingest-canvas-py.md`). The ingest surface's central promise — the commit-point forecast — is wrong by roughly half on the archetypal mixed folder, and it contradicts the consent line two rows above it.

`commit_summary_line` computes `will_import = supported_total - will_match` and counts only empty files as failures; a file whose type group has UNMET `required_features` (no pdf/audio/ebook/OCR tooling installed) is still counted as "will import", even though the preflight has already emitted a warning naming that exact missing dependency. The inline consent line uses a completely separate computation (`count_warning_affected_files`), so the two numbers are derived independently and disagree on screen.

Observed live in two independent sessions on different fixtures: `15 will import` sitting two rows above `⚠ Press Start again to import anyway — 7 files may fail`, delivering `8 imported · 5 skipped · 8 failed`; and `10 will import · 3 will skip · 2 will fail` delivering `1 done · 3 skipped · 10 failed · 1 matched`. The optimistic number is the one that persists on screen.

This is the forecast→receipt honesty loop that two prior critiques named as this surface's signature strength. It is now its central defect: the arcs made the forecast more detailed without making it more truthful.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The commit forecast and the inline consent line are derived from ONE computation and can never state different numbers for the same staged selection
- [ ] #2 Files whose type group has unmet required tooling are forecast as failures (not imports), with the reason distinguishable from the empty-file reason (e.g. "N need tooling, M empty")
- [ ] #3 A mixed folder staged on an install lacking optional backends produces a forecast whose import/skip/fail counts match the actual receipt tally
- [ ] #4 The forecast remains visible (not blanked) while a gate blocks Start, so a blocked user does not lose the numbers they were reasoning about
<!-- AC:END -->
