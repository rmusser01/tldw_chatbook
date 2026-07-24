# Console Multiple Attachments per Message (TASK-217) — Design

- **Date:** 2026-07-13
- **Status:** Approved pending user spec review
- **Scope anchor:** TASK-217 — multiple attachments per Console message: staging, payloads, persistence (with the DB schema decision), transcript. Builds on #621/#626/#628.

## Decisions (user-approved)

| Decision | Choice |
|---|---|
| Schema | New `message_attachments` table (v18→v19, repo migration pattern) **holding positions ≥ 1 only** — the legacy `messages.image_data`/`image_mime_type` columns remain the storage for attachment #0 (design revision during review: the originally proposed backfill+mirror stored every first attachment twice, forever; positions-≥1 gives zero duplication, an empty-table migration, and legacy readers untouched by construction). **Trade-off (amended post-QA):** the legacy columns carry no filename column, so `display_name` persists for positions ≥ 1 only; position 0 keeps the pre-existing mime·size resume label (legacy columns carry no filename) after a DB conversation resume. The zero-duplication design stands — only the filename-on-resume benefit is scoped to positions ≥ 1. |
| Staging UX | Count summary + clear-all: 1 attachment → today's exact label; N → `📎 N files` with all names in the tooltip; ✕ clears ALL; re-attach APPENDS (replaces today's replace-on-reattach). Per-item removal deferred. |
| Caps & rendering | Staging cap **5** per message (6th attach → toast); transcript shows one chip line PER attachment; inline rendering (TASK-215 window/cache) stays message-keyed and renders the FIRST image only. |
| Model | List-first + mirrored scalars: `ConsoleChatMessage.attachments: tuple[MessageAttachment, ...]` is source of truth; existing scalar image fields auto-mirror `attachments[0]` through a single store setter path (no split-brain); `ConsoleChatSession.pending_attachments: list[PendingAttachment]`. |

## Schema (v18 → v19)

`DB/migrations/chachanotes_v18_to_v19_message_attachments.sql` + `_migrate_from_v18_to_v19` (established pattern; current version verified v18):

```sql
CREATE TABLE message_attachments (
  message_id   TEXT    NOT NULL REFERENCES messages(id) ON DELETE CASCADE ON UPDATE CASCADE,
  position     INTEGER NOT NULL CHECK (position >= 1),
  data         BLOB    NOT NULL,
  mime_type    TEXT    NOT NULL,
  display_name TEXT    NOT NULL DEFAULT '',
  PRIMARY KEY (message_id, position)
);
CREATE INDEX idx_message_attachments_message ON message_attachments(message_id);
```

(Composite PK, no surrogate id, no sync/version columns — matching the repo's bare-association-table precedent, `conversation_keywords`. Migration implementation shape verified: SQL as a `_MIGRATE_V18_TO_V19_SQL` class attribute run via `executescript` with the version bump inside the script, plus the `DB/migrations/*.sql` documentation mirror; the fresh-create path applies `_apply_schema_v4` then the migration chain, so one definition covers fresh and upgrading DBs.)

- **Positions ≥ 1 only** (CHECK-enforced): position 0 lives in the legacy message columns. Single-attachment messages never touch this table.
- **No backfill**: the table starts empty; pre-migration data is already complete (every existing message has ≤ 1 attachment, in the legacy columns).
- **Deletion semantics (verified against the DB's patterns):** messages soft-delete (`deleted` flag + sync-trigger transitions) — attachment rows stay put and are unreachable through normal readers; hard deletes cascade via the FK (`PRAGMA foreign_keys = ON` confirmed; `ON DELETE CASCADE` precedent confirmed at messages/conversations FKs). Console's `store.delete_message` is local-transcript-only today; no new deletion pathways added.
- **No sync triggers (deliberate, verified against the per-entity trigger pattern):** Sync v2 does not carry images (TASK-220); BLOBs don't belong in `sync_log` json payloads. TASK-220 designs attachment sync separately.

## Components

### Model — `Chat/console_chat_models.py`

- `@dataclass(frozen=True) MessageAttachment(data: bytes | None, mime_type: str, display_name: str, position: int)` (`data` None only for metadata-only restore states, mirroring today's scalar semantics).
- `ConsoleChatMessage.attachments: tuple[MessageAttachment, ...] = ()` — source of truth. The existing scalar fields (`image_data`, `image_mime_type`, `attachment_label`) become a maintained mirror of `attachments[0]`; ALL mutation flows through one store helper (`_set_message_attachments`) that enforces the mirror, so every existing reader (payload builder scalar fallback, chips, rehydration, Save Image, serialization allowlist) stays correct during and after the transition.

### Store — `Chat/console_chat_store.py`

- `ConsoleChatSession.pending_attachments: list[PendingAttachment]` (replaces the single field). `add_pending_attachment` (appends; raises/returns cap signal at 5), `clear_pending_attachments`, `pending_attachments()` accessor; the Phase-1 single accessors reimplemented over the list (`pending_attachment()` → first item or None) so #628's paste/Alt+V call sites keep working until migrated in the same PR.
- `append_message(..., attachments: Sequence[MessageAttachment] = ())` supersedes the scalar image kwargs internally (scalar kwargs kept, converted to a one-element tuple — additive back-compat for tests and any straggler call sites).

### Persistence — `Chat/chat_persistence_service.py` + `DB/ChaChaNotes_DB.py`

- Service `create_message`/`update_message_content` gain `attachments: Sequence[...] | None`; the adapter writes attachment #0 into the legacy columns and positions ≥ 1 into the table, in ONE `transaction()`. Update semantics extend the #621 fix: attachments untouched unless explicitly provided (None = don't touch, mirroring the image-kwargs-omitted rule).
- New DB methods: `set_message_attachments(message_id, rows)` (delete+insert within the caller's transaction) and `get_attachments_for_messages(message_ids) -> dict[message_id, list[row]]` (single `WHERE message_id IN (...)` query — the resume path batch-fetches; no N+1).
- Rehydration/read: full tuple = legacy columns (position 0) + table rows (≥ 1), assembled in the adapter; pre-migration rows need no special casing (table simply has no rows for them). **Amended (post-QA honesty finding):** the legacy columns carry no filename column, so the position-0 entry always rehydrates with an empty `display_name` (mime·size resume label, matching pre-existing #626 behavior); only positions ≥ 1 rehydrate with the real `display_name` persisted in the table.

### Screen / composer / transcript

- Attach flows (picker, path-paste, Alt+V) call `add_pending_attachment`; at cap → toast "Attachment limit reached (5 per message)." and drop.
- Composer `set_pending_attachment_label`: unchanged for one; `📎 N files` + tooltip listing names for N (label building stays screen-side; composer API unchanged).
- Send: all pendings → `MessageAttachment` tuple (position = stage order) → message; clear-all after append (same placement as today's single clear).
- Payload builder: image budget counts **images** — walk messages newest-first, take each message's images in position order until `max_history_images` is exhausted; partially-budgeted messages include what fits; text parts always included.
- Transcript: one `🖼 name · size` chip line per attachment (order by position); TASK-215 image row unchanged (first image only — the render cache/window stay message-keyed; multi-image inline rendering is an explicit non-goal, follow-up if wanted). **Amended (post-QA honesty finding):** `name` is the real `display_name` for a live-session attachment and for any resumed attachment at position ≥ 1; a message resumed from a persisted DB conversation has no filename for position 0 (legacy columns carry no filename column), so that chip falls back to the mime·size label instead — pre-existing #626 behavior, unchanged by this feature.
- Save Image: saves ALL attachments of the message (existing collision-safe naming per file), one summary toast ("Saved N images to …").
- Screen-state serialization: allowlist gains `attachment_labels: list[str]` metadata (names only — the no-raw-bytes constraint holds); restore rehydrates bytes via the batch fetch.

## Edge cases

- Mixed stage (image + .md): text files inline into the draft at attach time (unchanged); only attachment-mode items join the pending list.
- Vision gate: any staged image + non-vision model blocks send (predicate over the list; unchanged copy).
- Metadata-only restore (DB unavailable): tuple of dataless attachments; chips render from names; payload skips dataless images (existing behavior generalized).
- Cap interactions: multi-path drop respects remaining capacity ("Attached first N of M dropped files." when truncated by the cap).
- Editing text of a multi-attachment message: attachments untouched (None-means-don't-touch rule).
- Payload size: the pre-existing exposure (base64 images × `max_history_images` budget) is unchanged by this feature, but multi-attach makes it easier to reach; acknowledged here rather than silently inherited — a payload-size guard would be its own follow-up if it ever bites.

## Testing

1. Migration: fresh-create at v19 + upgrade-from-v18 fixture (table exists, empty; FK cascade on hard delete; CHECK rejects position 0).
2. Store: mirror-invariant tests (scalars always == attachments[0] through every mutation path), cap enforcement, clear-all, single-accessor compatibility.
3. Persistence: round-trip N attachments (legacy columns hold #0, table holds rest); update-without-attachments preserves; batch fetch shape.
4. Payload: budget-across-messages matrix (multi-image messages, partial budgets, dataless skips).
5. Mounted: append staging + count label + cap toast + clear-all; chips per attachment; Save-Image-all; send clears.
6. Legacy image regression gate untouched and green (legacy chat never reads the new table).
7. QA captures + user visual gate.

## Out of scope

Per-item pending removal; multi-image inline rendering; Sync v2 attachment sync (TASK-220); Chatbook export (TASK-221); legacy chat multi-attach; config-driven caps (TASK-222 — the 5-cap is a named constant).

## Key file touch list

| File | Change |
|---|---|
| `DB/ChaChaNotes_DB.py` + `DB/migrations/…v18_to_v19…sql` | v19 table + accessors + batch fetch |
| `Chat/console_chat_models.py` | `MessageAttachment`, `attachments` tuple |
| `Chat/console_chat_store.py` | pending list, mirror setter, append_message |
| `Chat/chat_persistence_service.py` | attachment-aware create/update/read |
| `Chat/console_chat_controller.py` | image-counting budget, send staging |
| `UI/Screens/chat_screen.py` | staging flows, cap toast, labels, Save-all, serialization |
| `Widgets/Console/console_transcript.py` | chips per attachment |
