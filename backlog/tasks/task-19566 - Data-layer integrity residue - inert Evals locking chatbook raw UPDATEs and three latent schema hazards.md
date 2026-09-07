---
id: TASK-19566
title: >-
  Data-layer integrity residue — inert Evals locking, chatbook import raw
  UPDATEs, and three latent schema hazards
status: To Do
assignee: []
created_date: '2026-08-21 20:16'
labels:
  - db
  - data-integrity
  - tech-debt
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 3 (data layer & schema integrity) —
its **F6, F8, F9, F11, F12**. Grouped as one triage batch: each is a real
integrity defect, none is an active user-facing incident, and each carries an
honest reachability qualifier that must survive into the fix. All re-verified
at this branch base.

**F8 — Evals optimistic locking is inert (CONFIRMED).** Five tables in
`DB/Evals_DB.py` declare a `version` column (35 mentions of `version` in the
file), but **`expected_version` appears zero times** and **no `UPDATE` carries
`AND version = ?`**. It has the shape of concurrency control and provides
none — which is worse than having none, because reviewers will read it as
protection.

**F9 — chatbook import writes raw UPDATEs that bypass ChaChaNotes versioning
(CONFIRMED).** `Chatbooks/chatbook_importer.py:650` (`UPDATE messages SET
variant_of = ?…`) and `:676` (`UPDATE conversations SET
active_leaf_message_id = ?…`): no version bump, no `client_id`, no
`last_modified`. Unlike Media, ChaChaNotes has **no guard trigger** to catch
this, so the rows silently desynchronise from the sync log. Sharpest detail:
`deleted = ?` is set **from untrusted archive data** on an existing row — an
imported chatbook can mark a user's existing conversation deleted.

**F6 — two Media UPDATEs are guaranteed `IntegrityError` (CONFIRMED, but
UNREACHABLE).** The Media `BEFORE UPDATE` guard trigger requires `version + 1`;
`Chunking/chunking_interop_library.py:449` and `:471` (and
`DB/Client_Media_DB_v2.py:8261`) do not supply it, so they cannot succeed.
**Honest reachability, carried through from the lane: the calling widget has no
production importer, and `mark_media_as_processed` has no callers.** Frame this
as **wire-or-retire**, not "fix the bug": decide whether this code is meant to
be reachable. If it is, fix it and wire it; if it is not, delete it. Do not
repair dead code and leave it dead.

**F11 — two databases run with `foreign_keys` OFF (CONFIRMED, LATENT).**
`DB/Library_Ingest_Jobs_DB.py` and `DB/RAG_Indexing_DB.py` never enable the
pragma. Latent today because neither declares any foreign keys — but the next
schema change silently gets no enforcement.

**F12 — a cascade that would take the whole chat history (CONFIRMED schema, NO
LIVE TRIGGER).** `character_cards → conversations → messages` with
`ON DELETE CASCADE` (`ChaChaNotes_DB.py:470, 525`) and foreign keys ON. One
hard `DELETE` of a character card would remove the user's entire chat history
for it. **No such hard delete exists outside `Tests/` today** — the app soft-
deletes. This is a landmine for a future change, not a present bug; the value
here is the guard, not a schema change.

## Acceptance Criteria

- [ ] Evals optimistic locking either works — `expected_version` supplied and
      `AND version = ?` on the UPDATEs, with a conflict test — or the `version`
      columns and their appearance of protection are removed
- [ ] The chatbook importer goes through the versioned write path: version
      bump, `client_id`, `last_modified` set on every row it touches
- [ ] `deleted` is never set on an existing row from archive-supplied data; an
      imported chatbook cannot mark a user's existing conversation deleted
- [ ] A guard (trigger or test) fails if a ChaChaNotes UPDATE bypasses
      versioning, matching the protection Media already has
- [ ] The two Media UPDATEs are resolved as **wire-or-retire** with the
      decision recorded — repaired *and* reachable, or deleted
- [ ] `Library_Ingest_Jobs_DB` and `RAG_Indexing_DB` enable `foreign_keys`, or
      a comment records why they deliberately do not
- [ ] A test fails if a hard `DELETE` of a character card becomes reachable
      from production code, so the cascade cannot be armed unnoticed
