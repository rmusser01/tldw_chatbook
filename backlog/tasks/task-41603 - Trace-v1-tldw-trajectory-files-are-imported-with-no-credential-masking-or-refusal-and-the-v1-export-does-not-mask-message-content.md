---
id: TASK-41603
title: "Trace v1 (tldw-trajectory) files are imported with no credential masking or refusal, and the v1 export does not mask message content"
status: Done
assignee:
- '@claude'
updated_date: 2026-11-18 04:03
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
- [x] #1 Importing a v1 file whose messages, title or tool payloads carry credential-shaped values never shows those values: they are masked by the current rules, as v2 imports are, or the import is refused with a message that says why
- [x] #2 The v1 export masks credential-shaped values across the whole payload (message content, titles, tool payloads with or without `include_payloads`, their previews, variant replies and compaction records), as a Trace v2 bundle masks them; the builder stays (coordinator's widening, review r16tr-rev1)
- [x] #3 Tests pin the import and the export, each red before the fix
<!-- AC:END -->

## Implementation Plan

1. Base check (45b1949ca3): `_v1_snapshot` maps a validated v1 document straight into the snapshot; `build_trajectory_export` writes message `content` and the title verbatim.
2. Import: `_v1_snapshot` masks the validated document with `trajectory_export._mask_credential_material` (the function `_validate_v2` re-masks with) before mapping; a row's `payload_json` text that holds a JSON object is parsed first so its values are masked as v2 payload values are. No refusal (owner decision 2026-11-17, "Re-mask on import").
3. Export: `build_trajectory_export` masks each message's `content` and the conversation `title` with the same function. The v1 builder stays (decision in the brief).
4. Tests, red at base: the must-mask and must-keep table tests also import a v1 document holding each row in a message, the title and a tool payload; a v1 export test seeds a conversation holding credentials in a message and the title.

## Implementation Notes

A v1 (`tldw-trajectory`) import is masked by the rules a v2 import re-masks with, and the v1 export masks message content and the title.

- **Base (45b1949ca3).** As described: `_v1_snapshot` mapped the validated document straight into the snapshot; `build_trajectory_export` wrote `content` and `title` verbatim.
- **AC#1 (mask, not refuse).** `_v1_snapshot` masks the validated document with `trajectory_export._mask_credential_material` before mapping (owner decision 2026-11-17, "Re-mask on import"). A row's `payload_json` that is JSON text holding an object is parsed first (`_json_object`), so a tool payload is masked as a v2 payload dict is (keys and values), and the derived record gets the masked dict. Masking runs inside the existing mapping `try`, so a failure there (a recursion error on a file nested over ~1000 levels in an unknown field, which imported at base) is a `TrajectoryImportError`, never a crash.
- **AC#2 (mask; builder kept).** Round 1 masked each message's `content` and the conversation `title`; AC#2 was widened in review r16tr-rev1 (below). The builder stays: the decision came with the brief, and only tests and API callers reach it today.
- **User Guide.** `console/semantic-trace-capture.md`: opening an older `tldw-trajectory` file masks credential-shaped values as it loads.
- **Tests.**
  - Import: `test_every_secret_in_the_must_mask_table_is_masked` and `test_every_look_alike_in_the_must_keep_table_is_kept_and_imports` (laned nodes) also import a v1 file holding each row in a message, the title and a tool payload (`_v1_imported`): the preview equals the masked text's preview, the payload equals the masked text, and the raw text appears nowhere the import keeps; look-alikes come through unchanged.
  - Export: `Tests/Chat/test_trajectory_export.py::test_export_masks_credentials_in_message_content_and_the_title` (real temp `CharactersRAGDB`; every must-mask row masked as the bundle masks it, every must-keep row kept, the title masked). Not on a PR lane: the file runs only in the full suite (proposed follow-up).
- **Red evidence.** At base: 267 failed (every must-mask row, `token=DONOTUSE... != token=[credential redacted]`), 103 must-keep passed; the export test failed (`'token=DONOTUSEabc123def456' == 'token=[credential redacted]'`). Must-keep v1 pin under a temporary edit that re-applies the older simple-label rule to v1 message content: 8 failed. Payload parsing under a temporary edit that masks `payload_json` as text: 252 failed.
- **Runs** (run.sh, `-n 4`, load 57-96): `test_trace_v2_collaboration.py`, `test_trajectory_export.py`, `test_trajectory_import.py`, `test_trajectory_import_bounds.py`: 716 passed. The other callers (`test_assistant_generation_state_roundtrip`, `test_console_staged_evidence_trace_export`, `test_console_voice_ephemerality`, `test_library_preparation`, `test_thinking_privacy_surfaces`, `Tests/UI/test_trajectory_import_ui.py`, `test_trace_export_ui.py`, `test_trace_responsive.py`, `Tests/Docs/test_user_guide_contract.py`, `test_user_guide_conflict_markers.py`): 159 passed, 2 failed with `RecoveryRequired: raw_source_selection_changed` (the known local trap); the same 2 fail with the base sources.
- Files: `tldw_chatbook/Chat/trajectory_import.py`, `tldw_chatbook/Chat/trajectory_export.py`, `Tests/Chat/test_trace_v2_collaboration.py`, `Tests/Chat/test_trajectory_export.py`, `Docs/User_Guide/console/semantic-trace-capture.md`.

**Review r16tr-rev1 (fix round 1).** AC#2 widened by the coordinator first: the whole v1 payload.
- **Export.** `build_trajectory_export` runs `_mask_credential_material` over the finished payload, as `build_trace_export` checks its own: message content, the title, tool payloads with or without `include_payloads` and their `result_preview`/`args_preview`, variant replies and compaction records. A row's `payload_json` holding a JSON object is masked as a parsed object (keys and values, as a v2 payload) and serialized again only when something in it changed, so a payload with nothing to mask stays as written (`test_include_payloads_opt_in_keeps_verbatim` unchanged). `_json_object` moved to `trajectory_export` (import uses it from there).
- **Test.** `test_export_masks_credentials_in_message_content_and_the_title` is parametrized over `include_payloads` and adds a tool call holding secrets in its arguments and result, a variant reply and a compaction `failure_reason`. The schema's CHECK allows only reason codes in `failure_reason`, so the test injects an older or foreign row at `ConsoleContextRepository.list_auxiliary_attempts`. Its fixture now closes the DB, so the node passes `--storage-lease-guard`. Red before: both cases named all four leaking channels (`tool payload or previews`, `variant reply`, `compaction record`, `whole payload`); green after.
- **Docs.** `Docs/Features/Trajectory-View.md` no longer says credential-bearing bundles fail closed: they import masked (TASK-41133), v1 files are masked as they load, a quoted error value is masked and plain text, and the v1 export masks its whole payload.
- **Lane.** The export node joins the recovered-suites step (`derived-artifacts.yml`, `RECOVERED_SUITES_TARGETS`) in the round's lane commit.
- **Runs** (load 26-35, `-n 4`): the covering set (trace and trajectory Chat files, `test_log_sanitizer.py`, the other v1 callers, trace UI, Docs, CI workflow contracts): 1347 passed, 2 failed with the `RecoveryRequired` trap (the same 2 as round 1, red with base sources).
