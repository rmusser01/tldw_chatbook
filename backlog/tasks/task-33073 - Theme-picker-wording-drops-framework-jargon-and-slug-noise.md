---
id: TASK-33073
title: Theme picker wording drops framework jargon and slug noise
status: Done
assignee: []
created_date: '2026-09-27 18:00'
labels:
  - settings
  - theme
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P3. The group is titled 'TEXTUAL' (the framework's name), duplicate names show slug suffixes, and dialogs show file slugs while the list shows display names. Evidence and file:line causes: Docs/superpowers/qa/2026-09-27-theme-critique-3/findings.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The built-in group has a user-facing title
- [x] #2 Dialogs show the display name alongside the file name
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rename the picker's 'TEXTUAL' group to 'BUILT-IN' and use the same user-facing origin word wherever the origin is shown.
2. build_catalog disambiguates duplicate display names by origin word when that is unique, slug otherwise.
3. Rename/Delete/Export dialogs show the display name with the file name; update pinning tests/docs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Picker group 'TEXTUAL' -> 'BUILT-IN'; theme_catalog.ORIGIN_LABELS maps textual -> 'built-in' and is used for the 'overrides ...' marker and the preview title's origin word (one-token edits in Lane A's picker file). build_catalog tells duplicate display names apart by origin word when unique (Solarized Dark · built-in / · shipped), falling back to the id within one origin. Rename/Export prompt titles and Delete/Replace/Overwrite confirmations show "'Display Name' (file.toml)" via _dialog_label/dialog_label (resolved through the one scan, R12). Tests + user guide updated.

P3 review fixes (M3/M4): the card title repeated the origin for disambiguated names ('Solarized Dark · built-in  ·  dark · built-in'); it now uses _bare_name(entry), which strips the ' · <origin>' suffix. The filter matches that bare name and the id, not origin words: matching 'built-in'/'shipped' would make short queries ('i', 'p') list whole groups, and the headings already group by origin. Test: test_shared_name_shows_origin_once_and_filter_ignores_origin_words (fails on each half separately).
<!-- SECTION:NOTES:END -->
