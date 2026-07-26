---
id: TASK-661
title: Handle unsupported files in ingest pre-flight instead of raising
status: Done
assignee: []
created_date: '2026-07-26 03:26'
updated_date: '2026-07-26 03:38'
labels:
  - ingest
  - bug
  - p1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-flight analysis crashes for any unsupported file, and for any folder containing even one, replacing the whole summary with a raw Python error string. The user loses the file count, size and per-type breakdown, and the missing-tooling guardrail can no longer fire. Common everyday files such as images, subtitles and JSON are enough to trigger it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Analyzing a single unsupported file returns a result instead of raising
- [x] #2 Analyzing a folder of supported files plus one unsupported file returns the full summary for the supported ones
- [x] #3 Unsupported files are reported to the user in their own summary line, not as an error string
- [x] #4 Tooling warnings for the supported groups in a mixed folder still reach the guardrail
- [x] #5 No raw exception text is shown in the pre-flight area for this case
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for a single unsupported file and a realistic mixed folder
2. Give the unsupported sentinel a name and make the capability lookup handle it
3. Skip the sentinel where pre-flight collects tooling warnings
4. Re-run the original repro
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
get_type_group returns an 'unsupported' sentinel by design so the summary can count those files separately, and the state and canvas layers already implemented that path. The sentinel is deliberately not a key of the capability table, so get_tooling_warnings raised KeyError on it -- aborting the whole analysis. Any folder holding a .json, .jpg or .srt next to the content was enough, which is close to every real folder.

The sentinel is now a named constant, get_tooling_warnings returns no warnings for it (installing something cannot make those files ingestible), and analyze_path skips it when collecting warnings rather than relying on the callee. The previously unreachable 'N unsupported files will be recorded as a failure' summary line now renders.

Verified on the original repro: a folder of 4 mixed files now reports 4 files, 537 bytes, a pdf/generic/unsupported breakdown and the 3 PDF tooling warnings that feed the guardrail, where it previously produced only "Pre-flight analysis failed: 'unsupported'".

Changed: tldw_chatbook/Library/ingest_capabilities.py, tldw_chatbook/Library/ingest_preflight.py, Tests/Library/test_ingest_preflight.py
<!-- SECTION:NOTES:END -->
