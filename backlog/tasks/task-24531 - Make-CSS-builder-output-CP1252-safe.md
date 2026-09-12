---
id: TASK-24531
title: Make CSS builder output CP1252 safe
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-30 02:23'
labels:
  - css
  - windows
  - portability
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-29-windows-css-builder-output-portability-design.md
documentation:
  - Docs/superpowers/plans/2026-08-29-windows-css-builder-output-portability.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow the complete CSS generation entry point to run on strict Windows CP1252 standard output without weakening substantive build failures or changing generated stylesheets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every direct CSS-builder progress and completion message is encodable by a strict CP1252 output stream and remains readable
- [x] #2 A full scratch-tree build succeeds through strict CP1252 output even when the checkout path contains a non-CP1252 character
- [x] #3 The integration proof observes all four build phases, verifies every expected generated artifact and manifest exists, and proves distinctive source CSS reaches the generated bundle
- [x] #4 Generated CSS ordering, hashing, manifest staleness semantics, source-race detection, missing-module failures, and output-preservation behavior remain unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Adapt the existing builder integration test to strict CP1252 output and a non-representable checkout path.
2. Replace direct builder output with ASCII-only numeric phase messages.
3. Verify manifest, staleness, fail-loud, and generated-artifact semantics.
4. Complete task evidence and self-review.

Detailed plan: Docs/superpowers/plans/2026-08-29-windows-css-builder-output-portability.md
ADR required: no
ADR path: N/A
Reason: portable build-script presentation only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made every direct CSS-builder phase message ASCII-only and path-free while retaining internal module/path handling and all substantive build failures. Adapted the existing real-main integration test to strict CP1252 stdout under a checkout-漢 scratch path, with explicit UTF-8 file I/O, four phase assertions, all generated artifact/manifest checks, distinctive bundle content, and unchanged staleness controls. Modified only tldw_chatbook/css/build_css.py and Tests/UI/test_css_staleness_manifest.py; no generated stylesheet or manifest changed. TDD evidence: the strict-stream test failed before production edits with UnicodeEncodeError on the checkmark and passed after the minimal output change. Fresh focused verification passed 66 tests. Direct print literals passed ASCII validation; Ruff check/format, compileall, and git diff --check passed. Independent specification and code-quality reviews approved after adding explicit UTF-8 to the new test file operations. Full repository suite not run under targeted-test policy. ADR required: no. Lessons learned: none added; the portability contract is captured by the approved regression.
<!-- SECTION:NOTES:END -->
