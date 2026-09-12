---
id: TASK-21138
title: 'Attached file as a chunking source - scope/staging decision'
status: To Do
assignee: []
created_date: '2026-08-23'
updated_date: '2026-08-23'
labels:
  - chunking
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Program-close follow-up closing the last dangling thread of #1's spec §11
(`Docs/superpowers/specs/2026-08-18-chunking-engine-parity-design.md`, "Follow-up
tasks to file"). §11 filed two attachment items:

1. **Attachment text extraction** — DONE. Absorbed by TASK-19576, whose shared
   `_extract_local_ingest_text` helper (backed by `parse_local_file_for_ingest`
   off-thread) replaced all five `Utils/file_handlers.py` placeholder returns
   (PDF / Document / Ebook handlers), so an attached file now yields real text.
2. **"Attached file as a chunking source. Once extraction works, let the
   sub-project #4 tool address an attached (non-ingested) file, including the
   staging/caching decision for where extracted text lives."** — never
   dispositioned. #4 declined it as a non-goal
   (`Docs/superpowers/specs/2026-08-22-chunking-agent-tools-design.md` §3:
   "**Attached-but-not-ingested files** — follow-ups already filed (#1's §11:
   attachment extraction, attached-file chunking source)"; §2 source surface:
   "ingested Library media only").

With the Chunking Parity & Agent Tools program closed 6-of-6, the chunk-source
half still needs a decision, not an implementation: should the `library_*`
chunk-reading tools (`library_get_media_structure`, `library_get_media_chunk`,
`library_list_chunk_specs`, `library_rechunk_media`) — or any other tool
surface — be able to address an attached, non-ingested file; and if yes, where
does the extracted text stage (per-call extraction vs a conversation-scoped
cache vs reuse of the TASK-19576 path)? Or is this out of scope for good? This
task exists to record that decision either way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] The scope decision is recorded (in-scope / out / defer) together with the staging answer for where extracted attachment text would live — per-call extraction, a named cache/store, or N/A-because-out
- [ ] If in-scope (or deferred with intent to build): the tool surface is named — extending a specific `library_*` tool vs a new attachment-addressable surface — and an implementation task is filed; if out-of-scope: the rationale explicitly cites #4's §3 non-goal and no child task is filed
- [ ] The disposition (executed decision or explicit decline) is captured in this task's Implementation Notes so #1's §11 thread is closable without re-deriving the history
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read #4's §2 source-surface constraints (ingested-Library-media-only model, media-id addressing) and TASK-19576's extraction path (`_extract_local_ingest_text`) to ground the options
2. Weigh per-call extraction cost vs a staging cache: attachment lifetime vs media-record lifetime, re-extraction on repeat reads, and whether `library_*` schemas can address non-ingested files without breaking the ingested-only contract
3. Record the decision + staging answer in this task's notes; file an implementation task only if the answer is "build it"; otherwise close with the #4-non-goal rationale
<!-- SECTION:PLAN:END -->
