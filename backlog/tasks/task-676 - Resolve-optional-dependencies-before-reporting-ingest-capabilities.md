---
id: TASK-676
title: Resolve optional dependencies before reporting ingest capabilities
status: Done
assignee: []
created_date: '2026-07-26 03:26'
updated_date: '2026-07-26 03:56'
labels:
  - ingest
  - bug
  - p1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ingest capability layer reads the optional-dependency registry without ever resolving it, and lazy checking is the default, so every optional feature reports as missing. Users are told to install packages they already have, and every per-type advanced option is permanently disabled, making the PDF engine, OCR, transcription model, language, timestamps, diarization and e-book extraction controls impossible to change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Capability checks reflect what is actually installed rather than an unresolved registry
- [x] #2 Advanced options for an installed feature are editable
- [x] #3 Advanced options for a genuinely missing feature remain disabled
- [x] #4 No tooling warning is shown for a feature that is installed
- [x] #5 Dependency resolution happens once rather than per field render
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for unchecked-registry probing, warnings and probe caching
2. Stop treating a registry False as authoritative
3. Resolve umbrella feature flags without importing
4. Separate sibling-field gating from dependency gating
5. Live-verify the warnings and the option controls
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DEPENDENCIES_AVAILABLE is pre-seeded with every key False and only filled in when something resolves it, which under the default lazy mode never happened -- ensure_dependencies_checked still has zero call sites. _is_installed treated that placeholder as authoritative, so every optional feature read as missing.

Only a True in the registry is authoritative now; anything else falls through to a memoised find_spec probe. Umbrella flags (pdf_processing, ebook_processing, audio_processing, video_processing) are not package names, and the authoritative checks establish them with a real __import__ of torch/chromadb/transformers -- far too expensive for a render path -- so this module mirrors their package lists explicitly and probes without importing. Verified to agree with the authoritative resolution on all four.

The drift guard added for that duplication then exposed a second defect: depends_on conflated 'this package must be installed' with 'this sibling checkbox must be on'. chunk_size and chunk_overlap declared depends_on='chunk', which resolved through the installed-feature lookup, so both inputs were disabled permanently even with chunking switched on. Sibling gating is now a separate enabled_when field.

Live-verified: the three false PDF warnings are gone and the PDF engine select renders as an active dropdown instead of greyed out.

Changed: tldw_chatbook/Library/ingest_capabilities.py, tldw_chatbook/Widgets/Library/library_ingest_canvas.py, Tests/Library/test_ingest_capabilities.py, Tests/UI/test_library_ingest_canvas.py
<!-- SECTION:NOTES:END -->
