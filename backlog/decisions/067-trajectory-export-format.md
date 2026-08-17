# ADR-067: Trajectory Export Format

- Status: Accepted
- Date: 2026-08-15
- Related: ADR-066 (trajectory view + sidecar), task-16813 (export), task-16320 (import)

## Context

Users need to share or archive a conversation's trajectory (trace) as a
single portable file — for debugging agent runs, attaching to issues, or
viewing on another machine (task-16320 renders imports read-only). The data
lives across `messages` (with `usage_json`), the local-only
`message_trajectory_metadata` sidecar, `conversations.active_leaf_message_id`,
compaction attempts (`console_context_repository.list_auxiliary_attempts`),
and process-local variant sets.

## Decision

1. **Format**: one JSON document per conversation:
   `{"format": "tldw-trajectory", "version": 1, "exported_at": <ISO-8601 UTC>,
   "redacted": <bool>, "conversation": {id, title, created_at},
   "active_leaf_message_id": <str|null>, "messages": [...non-deleted rows with
   id/sender/content/timestamp/parent_message_id/usage_json...],
   "trajectory_rows": [...sidecar rows as-is...], "compaction_records":
   [...], "variants": [...] (only when exported from a live session; may be
   empty)}`. Everything the existing projection (`Chat/trajectory.derive_trajectory`)
   needs to render the exact TrajectoryScreen view travels in the file.
2. **Versioning**: `version` is an integer starting at 1. Readers must accept
   the exact versions they were built for and reject higher ones with an
   actionable error; additive fields within a version are allowed and must be
   ignored by older readers.
3. **Redaction is the default**: tool payloads may contain file contents.
   Unless the caller passes an explicit opt-in, `payload_json` is replaced by
   `{"name": ..., "result_preview": <first 120 chars>, "args_preview": <first
   120 chars or null>, "redacted": true}`; `"redacted": true` at the document
   level marks the mode. Full payloads (`payload_json` verbatim) only on
   explicit opt-in.
4. **No secrets**: the file contains conversation content and metadata only —
   never API keys, config, or provider credentials.
5. **Export is a user-initiated egress of local-only data**: the sidecar is
   local-only in the DB (ADR-066); exporting copies it out of the machine by
   explicit action, which is the feature's purpose. Import (task-16320) never
   writes imported data back into local tables — imports render read-only.
6. **Implementation seam**: `Chat/trajectory_export.py` is pure (no Textual):
   `build_trajectory_export(...) -> dict`, `write_trajectory_export(path, payload)`
   (atomic tmp+rename), and `validate_trajectory_export(payload)` which
   returns normalized data or raises `TrajectoryExportError` — the same
   validator is the import seam, so export/import can never drift apart.

## Alternatives considered

- **Reuse `document_generator.py` export formats (md/html/pdf)**: those export
  chat transcripts for humans; the trajectory needs a machine-round-trippable
  projection input, i.e. structured JSON.
- **Two files (trace + manifest)**: single-file is the shareability
  requirement; a manifest adds failure modes for no gain.
- **SQLite snapshot file**: opaque to humans and diff tools; JSON matches the
  "shareable artifact" goal and validates cheaply.

## Consequences

- `version: 1` is now a public contract; breaking changes require a bump and
  reader rejection logic.
- Redaction means shared default exports lose full tool output; the opt-in is
  the escape hatch, and the document-level `redacted` flag keeps receivers
  honest.
- Legacy conversations (no sidecar rows) export fine — empty
  `trajectory_rows` with messages present is valid; the projection's legacy
  fallback applies on import rendering.
