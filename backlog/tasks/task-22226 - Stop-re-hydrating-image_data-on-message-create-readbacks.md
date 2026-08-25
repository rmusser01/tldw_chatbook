---
id: TASK-22226
title: >-
  Stop re-hydrating image_data on message-create readbacks
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
labels:
  - database
  - chat
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22226).

Pre-existing shape (the select list grew since the pin). `Chat/chat_persistence_service.py:
1324-1382` re-reads the just-written message up to three times per create (feedback +
citation paths), each via `get_message_by_id` (`DB/ChaChaNotes_DB.py:10965`) which
hydrates the `image_data` BLOB — MBs copied per image message persist.

## Acceptance Criteria

- [x] Create-path readbacks use a projection without BLOB columns, or reuse already-known values
- [x] Measured before/after on an image message persist
- [x] No change to what callers receive (shape-compatible)

## Implementation Plan

1. Caller census of `get_message_by_id` (done before touching it): its semantics
   stay unchanged; add a narrow sibling instead (TASK-22206 precedent:
   `get_message_tree_rows_for_conversation` — same select list minus the
   `image_data` BLOB, replaced by a `has_image` 0/1 flag).
2. Consumer-fields audit of each create-path readback in
   `Chat/chat_persistence_service.py::create_message`:
   - `:1346` (citation+feedback), `:1352` (citation revision), `:1382`
     (legacy feedback) read ONLY `version` (a DB-normalized value —
     keep reading it from the DB, just without the BLOB).
   - `:1324` (citation retry-identity readback) genuinely needs the
     `image_data` BYTES for the byte-equality identity check
     (`_verify_citation_message_retry`), and on the normal create path it
     returns None (row absent) with zero BLOB cost — left on
     `get_message_by_id` deliberately.
3. Add `CharactersRAGDB.get_message_by_id_without_blob(message_id)`:
   `get_message_by_id`'s exact select list with
   `(image_data IS NOT NULL) AS has_image` in place of `image_data`.
4. Red-first probe as a permanent test under `Tests/Chat/` (config isolation
   only applies there): a row_factory wrapper on the live thread connection
   counting materialized `image_data` bytes during `create_message` with a
   1 MB image — nonzero on base, ~0 after.
5. Switch `:1346`/`:1352`/`:1382` to the narrow reader; also
   `get_message_version` (`:134`), the per-send settle/continuation version
   reconcile that re-reads just-written rows through the same BLOB-hydrating
   method (same readback family, same contract: returns only `version`).
   Update the one test that monkeypatches `get_message_by_id` to drive it.
6. Measure per-create wall time before/after with a 1 MB image.
7. Targeted suites + `--collect-only` sweep, tee'd; `./scripts/preflight.sh`;
   mutation test (point one readback back at `get_message_by_id` → probe reds);
   failure walk (narrow readback returning None mid-path == base behavior).

## Implementation Notes

Added `CharactersRAGDB.get_message_by_id_without_blob` (`DB/ChaChaNotes_DB.py`,
directly after `get_message_by_id`): the exact `get_message_by_id` select list
with `(image_data IS NOT NULL) AS has_image` replacing the `image_data` BLOB —
the TASK-22206 narrow-projection precedent applied to the by-id shape.
`get_message_by_id` itself is untouched (semantics preserved for its ~15 other
production callers, several of which — Console image viewer,
`Character_Chat_Lib.get_message_by_id`'s documented contract, store snapshot
rebuild — genuinely need the bytes).

Switched four version-only readbacks in `Chat/chat_persistence_service.py` to
the narrow reader:
- `create_message` citation+feedback site, citation `message_revision` site,
  and legacy feedback site — each consumed ONLY `version` (the DB-normalized
  optimistic-lock input). The citation revision is additionally re-validated
  against the messages row inside `write_prepared` itself, so a wrong version
  cannot commit silently.
- `get_message_version` — the per-send settle/continuation reconcile that
  re-reads just-written rows through the same BLOB-hydrating method (same
  readback family; contract `int | None` unchanged). One existing test's
  monkeypatch seam updated accordingly.

Deliberately NOT switched: the citation retry-identity readback (`existing_message`)
— `_verify_citation_message_retry` byte-compares `image_data`, so it needs the
real BLOB; on the normal create path that call returns None (row absent) at
zero BLOB cost.

Evidence (tees in session scratchpad):
- Red-first probe (`Tests/Chat/test_chat_persistence_create_readbacks.py`,
  row-factory BLOB-byte counter on the live connection, 1 MB image):
  before 1,048,593 B/1 row (legacy+feedback), 2,097,186 B/2 rows
  (citation+feedback), 1,048,593 B/1 row (citation-only) → after 0 B on all
  three; probe self-check proves the counter can go red.
- Wall time (40-create medians, 1 MB image): legacy+feedback 7.50 → 7.20 ms,
  citation+feedback 11.89 → 11.63 ms; run-to-run median spread ~1 ms, so the
  wall-time delta is inside the noise floor — the eliminated cost is the
  1–2 MB/create readback copy itself (byte metric above), the insert dominates.
- Mutation tests: pointing the legacy site back at `get_message_by_id` reds
  exactly the legacy probe; pointing the citation revision site back reds both
  citation probes.
- Failure walk: a None readback mid-path raises the same TypeError at the same
  subscript as base (temp Tests/ probe, run then deleted).
- Suites: 197 passed (persistence service + create-readbacks + message feedback
  + citation trace repository), 143 passed/1 failed (ChaChaNotes DB +
  continuation/durable/dispatch consumers) — the 1
  (`test_visible_discard_projects_explicit_checkpoint_clear`) fails
  IDENTICALLY on base 76f130138 (stale sync-payload expectation missing
  `assistant_generation_state`), 294 passed (console chat store + parent
  persist), `--collect-only` sweep: 59,451 collected, 28 errors all
  missing-optional-dep (numpy/TTS/audio), pre-existing.
- Pre-existing dev reds confirmed by base-vs-change A/B with identical
  commands: `test_console_terminal_citation_persistence.py` (2 failures,
  identical FAILED sets) and `test_console_local_citation_boundary.py`
  (mass-red + run-killing hang in `test_citation_repair_late_chunk…`*,
  byte-identical progress pattern on base) — not caused by this change.
- `./scripts/preflight.sh` green after reviewing the one added diagnostic
  (interpolates only the message UUID + wrapped DB error, the sibling
  statement's exact shape) and regenerating the inventory.

Files: `tldw_chatbook/DB/ChaChaNotes_DB.py`,
`tldw_chatbook/Chat/chat_persistence_service.py`,
`Tests/Chat/test_chat_persistence_create_readbacks.py` (new),
`Tests/Chat/test_chat_persistence_service.py` (monkeypatch seam),
`Docs/security/production-diagnostic-inventory.json` (regenerated).
