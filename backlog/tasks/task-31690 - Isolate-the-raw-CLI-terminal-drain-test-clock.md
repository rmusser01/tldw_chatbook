---
id: TASK-31690
title: Isolate the raw CLI terminal drain test clock
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:31'
updated_date: '2026-09-05 18:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the finite fake drain clock local to the raw CLI module instead of replacing the standard-library monotonic clock used by pytest and other threads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The exit-versus-dead-worker test retains its exact three clock values, five queue calls, exit code and truncation assertions without changing the global clock.
- [x] #2 The complete raw CLI process test file passes with scoped static checks and no runtime changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add an explicit assertion that the drain clock injection leaves time.monotonic unchanged and reproduce RED. 2. Replace only raw_cli.time with a tiny existing SimpleNamespace clock seam, preserving the exact finite iterator and all output/queue assertions. 3. Run the complete raw CLI process file and scoped static checks. ADR required: no. ADR path: N/A. Reason: test-only monkeypatch isolation, no executor timing or process policy change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced only raw_cli.time with the existing SimpleNamespace seam rather than monkeypatching time.monotonic globally. Added clock-identity assertion, which failed against original setup; exact three fake clock values, five queue reads, exit23 and truncation assertions stay unchanged. Full52case process file:51passed1nativeWindows skip15.95s (/private/tmp/tldw-review-raw-cli-process-final-20260905.xml). Ruff, changed-range format, diff whitespace and self-review passed. No executor timing changes/new ADR.
<!-- SECTION:NOTES:END -->
