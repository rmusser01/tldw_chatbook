---
id: TASK-842
title: Semantic chunking depends on an undeclared NLTK download
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 02:02'
updated_date: '2026-07-27 02:31'
labels:
  - chunking
  - packaging
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The semantic chunking method needs NLTK tokeniser data that is fetched at runtime rather than declared as a dependency. Where that data is absent the method raises a lookup error naming an NLTK resource, which reads as an internal fault rather than a missing optional asset. Observed while exercising every chunking method: the same call succeeded in one environment and failed in another purely because of the cached download.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Semantic chunking either has its data available or explains what to install
- [x] #2 The failure is not presented as an internal error
- [x] #3 Other chunking methods are unaffected when the data is absent
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Semantic chunking now degrades usefully instead of raising when its tokeniser corpus is missing.

ROOT CAUSE, deeper than first diagnosed. _ensure_nltk handled nltk being ABSENT but not nltk being PRESENT WITHOUT ITS DATA: it bound the real tokeniser, which then raised LookupError deep inside a chunking call. Underneath that sat a second fault -- ensure_nltk_data checked for and downloaded "punkt", but nltk >= 3.9 reads "punkt_tab". Reproduced with a corpus containing exactly what that download produces: readiness reported True and the very next call still raised Resource 'punkt_tab' not found. Naming a resource was the mistake; WHICH one nltk wants is version-dependent.

THE FIX. Readiness is now decided by whether the tokeniser TOKENISES: probe once at bind time, and on failure attempt both corpora and re-probe, letting the probe -- not the download's own verdict -- decide. An nltk too old to know punkt_tab reports failure for it while being perfectly usable, so the download return value cannot be trusted. The unusable verdict is latched, since nltk stays None on that path and every chunking call would otherwise re-probe, re-download and re-warn.

A SEPARATE DEFECT FOUND WHILE VERIFYING. The semantic path's fallback split on NEWLINES, so ordinary single-paragraph prose came back as one chunk holding the whole document -- technically not raising while silently not chunking. Its LookupError branch also retried with language="english" uncaught, which is the crash shape originally reported. Both now route to the built-in sentence split.

VERIFICATION, AND A CORRECTION TO MY OWN. The earlier claim of verifying via NLTK_DATA pointed at an empty directory was a FALSE POSITIVE: NLTK_DATA appends to nltk.data.path rather than replacing it, so those runs kept finding the real corpus at ~/nltk_data and exercised the working tokeniser, never the fallback. With nltk.data.path[:] restricted for real, the first honest run failed at once -- and after the fix returns 4 chunks. All four guards are mutation-checked: removing the probe, the download, the latch, or the probe-based readiness each fails exactly one test and nothing else.

Files: Chunking/Chunk_Lib.py, Tests/RAG/test_chunking_service.py, Tests/Utils/test_startup_polish_regressions.py, backlog/docs/lessons-live-verification.md.
<!-- SECTION:NOTES:END -->
