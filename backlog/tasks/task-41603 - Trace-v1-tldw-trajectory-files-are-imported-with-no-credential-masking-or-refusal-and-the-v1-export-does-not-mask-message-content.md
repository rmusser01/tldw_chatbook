---
id: TASK-41603
title: "Trace v1 (tldw-trajectory) files are imported with no credential masking or refusal, and the v1 export does not mask message content"
status: To Do
assignee: []
created_date: 2026-11-17 19:44
labels:
- follow-up
- security
- trace
priority: high
references:
- tldw_chatbook/Chat/trajectory_import.py
- tldw_chatbook/Chat/trajectory_export.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
dependencies:
- TASK-41133
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the round-15 trace review (TASK-41133, 2026-11-17). The problem predates round 15. This task covers the version-1 (`tldw-trajectory`) format only. Version-2 (`tldw-trace`) imports are handled on the trace branch (owner decision 2026-11-17, "Re-mask on import").

**Import** (read at 212f015a91):
- `load_imported_trace` (`trajectory_import.py:1099`) sends every document whose `format` is not `tldw-trace` to `_v1_snapshot` (`:1103`, `:1121`).
- `_v1_snapshot` (`:141-170`) validates the document and maps its messages straight into the snapshot the Trace view renders. Nothing masks credentials and nothing refuses them.
- `_has_credential_material` is imported (`:46`), but only the v2 path calls it (`:939`).
- So a v1 file that carries a key in a message, a title or a tool payload shows it on screen after Import.

**Export:**
- `build_trajectory_export` (`trajectory_export.py:1274`) writes message `content` (`_MESSAGE_KEYS`, `:100-108`) and the conversation title verbatim (`:1355-1363`).
- It only shortens tool payloads to previews when `include_payloads` is false (`:1347-1348`).
- It has no final credential check. `build_trace_export` has one (`:1261`).
- No product path calls the v1 builder at this commit: the Export dialog builds v2 (`trace_export_dialog.py:386`). Only tests and API callers reach it.

**Related.** TASK-41510 and TASK-41133 cover the v2 masking rules, not the v1 format.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Importing a v1 file whose messages, title or tool payloads carry credential-shaped values never shows those values: they are masked by the current rules, as v2 imports are, or the import is refused with a message that says why
- [ ] #2 The v1 export either masks credential-shaped values in message content and titles, or is removed if nothing in the product needs it; the notes record which and why
- [ ] #3 Tests pin the import and the export, each red before the fix
<!-- AC:END -->
