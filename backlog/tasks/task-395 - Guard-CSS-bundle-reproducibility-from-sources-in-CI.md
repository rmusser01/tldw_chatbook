---
id: TASK-395
title: Guard CSS bundle reproducibility from sources in CI
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 18:35'
labels: [css, ci, tech-debt]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
778f75813 hand-edited the generated `tldw_cli_modular.tcss` instead of a source module, and the desync shipped unnoticed for a day — every rebuild (including the app's boot-time mtime rebuild) silently stripped live Settings splash/theme styling until the PR #723 resync. A CI check in the style of `backlog-guard.yml` should fail any PR touching `tldw_chatbook/css/**` where running `build_css.py` produces a bundle that differs from the committed one, ignoring the `Generated:` timestamp line (a naive byte compare always fails on it). Note the check must run even while general CI stays in the intentionally-cancelled state — model it on the standalone backlog-guard workflow, which does run.

Raised by review on PR #723 (Qodo suggestion 2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A PR that edits a `.tcss` source without regenerating the bundle (or edits the bundle directly) fails the check with a message naming the drifted module
- [x] #2 The comparison ignores the `Generated:` timestamp line so a faithful rebuild passes
- [x] #3 The check runs on PRs touching `tldw_chatbook/css/**` and post-merge on dev, like backlog-guard
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
New standalone `.github/workflows/css-bundle-guard.yml` (modeled on backlog-guard,
so it runs even while general CI is intentionally-cancelled) triggers on PRs
touching `tldw_chatbook/css/**` and post-merge on dev/main (AC#3). It runs
`tldw_chatbook/css/check_bundle_sync.py` — a stdlib-only, non-destructive checker
that rebuilds the bundle into a TEMP file (via `build_css.build_css(css_dir, out)`),
strips the `Generated:` timestamp line from both (AC#2), and reports drift. On
drift it names the specific MODULE block(s) via the `/* ===== MODULE: X ===== */`
markers (AC#1) and exits 1 with `::error::` annotations. build_css/check are
imported as sibling modules (no package __init__), so no `pip install` is needed.
Verified: in-sync repo → exit 0; a source edited without rebuilding → exit 1
naming `utilities/_overrides.tcss`. 3 unit tests (timestamp-ignore, module-naming,
committed-bundle-in-sync) + existing css_build_integrity green.
<!-- SECTION:NOTES:END -->
