---
id: TASK-16319
title: 'Trajectory export: versioned JSON trace format and writer'
status: In Progress
assignee: []
created_date: '2026-08-15 13:53'
updated_date: '2026-08-15 17:24'
labels:
  - trajectory
  - export
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users export a conversation's trajectory (trace) to a shareable, self-contained file, building on the Console trajectory view (task-16311..16315, ADR-066). Purpose: sharing/debugging agent runs outside the local DB. Export folds the same inputs as the projection (messages incl. usage_json, sidecar rows, variant sets where available, compaction records) into one versioned JSON document. Privacy: tool payloads may contain file contents -- export defaults to redacted payload previews, full payloads only behind an explicit opt-in flag. ADR required: yes (export format is a data contract) -- create before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Export produces a schema-versioned single-file JSON trace of one conversation,Tool payloads redacted by default; full payloads only with explicit opt-in flag,Import validator round-trips the exported file (task-2 seam),Export of a conversation lacking sidecar rows still succeeds (legacy fallback),Unit tests cover format, redaction, and edge cases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. ADR-067 export format contract (versioned JSON, redaction default). 2. Chat/trajectory_export.py: build/write/validate; validator is the import seam. 3. Round-trip tests (export->validate->derive_trajectory renders), redaction, legacy no-sidecar, malformed/version rejection, atomic write. 4. Task notes + Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Approach: `tldw_chatbook/Chat/trajectory_export.py` is pure (no Textual/widget imports) and exposes three seams per ADR-067 §6: `build_trajectory_export(db, conversation_id, *, include_payloads=False, variant_sets=())`, `write_trajectory_export(path, payload)` (pretty JSON, sibling mkstemp + `os.replace`, temp cleanup on failure), and `validate_trajectory_export(payload)` returning the normalized payload — the task-16320 import seam.
- Build reads the same accessors the live projection uses: `get_conversation_by_id`, `get_messages_for_conversation(include_image_data=False)` (image blobs never exported; only id/sender/content/timestamp/parent_message_id/usage_json per message), `get_trajectory_rows` (all `TrajectoryRowRead` fields), `get_conversation_active_leaf`, and compaction records via `ConsoleContextRepository.list_auxiliary_attempts(limit=500)` filtered to `purpose == "conversation_compaction"`.
- Redaction default (ADR-067 §3): tool_call/tool_result `payload_json` becomes `{"name", "result_preview" (<=120 chars, single line), "args_preview" (<=120 chars or null), "redacted": true}`; unparseable payloads degrade to empty previews rather than leaking. `include_payloads=True` keeps `payload_json` verbatim; the document-level `redacted` flag records the mode.
- Validator enforces format marker, `version == 1` (higher -> `TrajectoryExportError` naming the version), required sections/types (`exported_at`, `redacted`, `conversation.id`, `messages` with all six keys, `trajectory_rows` with the load-bearing row keys), fills optional sections (`compaction_records`, `variants`, `active_leaf_message_id`) to `[]`/`None`, and ignores additive fields (ADR-067 §2).
- Gotcha found: the DB layer returns `datetime` objects for timestamp columns; a `_jsonable` coercion (datetime -> ISO string) makes payloads JSON-serializable AND keeps `_parse_timestamp` working on import rendering.
- Tests (`Tests/Chat/test_trajectory_export.py`, real temp `CharactersRAGDB` with sidecar rows + real aux-attempt rows): round trip build -> validate -> write/read -> `derive_trajectory` re-renders the exact ledger (user/assistant/tool_call/tool_result/user/assistant/compaction, usage + variants carried), redaction default/opt-in, legacy no-sidecar export, malformed rejections naming the offending field, atomic write with no temp leftovers. 36/36 pass with `Tests/Chat/test_trajectory_projection.py`; ruff clean.
- Files: added `tldw_chatbook/Chat/trajectory_export.py`, `Tests/Chat/test_trajectory_export.py`. Commit: 7c794e6.
- ADR check: ADR-067 (pre-existing, linked) is the format contract; no new ADR needed.
