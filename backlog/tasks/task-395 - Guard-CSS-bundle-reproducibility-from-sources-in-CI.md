---
id: TASK-395
title: Guard CSS bundle reproducibility from sources in CI
status: To Do
assignee: []
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
- [ ] #1 A PR that edits a `.tcss` source without regenerating the bundle (or edits the bundle directly) fails the check with a message naming the drifted module
- [ ] #2 The comparison ignores the `Generated:` timestamp line so a faithful rebuild passes
- [ ] #3 The check runs on PRs touching `tldw_chatbook/css/**` and post-merge on dev, like backlog-guard
<!-- AC:END -->
