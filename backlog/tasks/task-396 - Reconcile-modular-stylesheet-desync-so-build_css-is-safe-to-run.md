---
id: TASK-396
title: Reconcile modular stylesheet desync so build_css is safe to run
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Hit concretely while fixing the Stats screen CSS (2026-07-20): running `tldw_chatbook/css/build_css.py` produces a `tldw_cli_modular.tcss` ~238 lines SMALLER than the checked-in one. The built file contains content whose sources are gone — e.g. a whole `components/splash_viewer.css` module block with no such file under `css/components/`, plus `#settings-category-list` rules and a `margin` divergence in settings styles. Until reconciled, any full rebuild silently sweeps live styles away, so fixes must be hand-applied to BOTH the component source and the built file (as the Stats geometry fix was). Reconcile by either restoring the missing module sources from the built file and re-adding them to the build inputs, or confirming the extra content is dead and deleting it deliberately — then rebuild and diff to byte-parity so `build_css.py` is trustworthy again. Consider a CI guard that rebuilds and diffs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Running build_css.py reproduces the checked-in tldw_cli_modular.tcss byte-for-byte (or with documented, intended diffs only)
- [ ] #2 Every module block present in the built file has a corresponding source file in the css/ tree
- [ ] #3 No visual regression on the screens whose styles currently exist only in the built file (Settings category list, splash viewer)
<!-- AC:END -->
