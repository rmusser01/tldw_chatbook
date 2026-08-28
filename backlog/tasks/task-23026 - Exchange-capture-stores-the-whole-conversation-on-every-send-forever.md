---
id: TASK-23026
title: >-
  Exchange capture stores the whole conversation on every send, forever
status: Done
assignee:
  - '@claude'
created_date: '2026-08-27'
labels:
  - storage
  - database
  - privacy
priority: high
---

## Description

`messages_payload` is on `CAPTURE_REQUEST_ALLOWLIST`, so every send persists a blob containing the
entire conversation so far. The blob grows **2.8 KB at turn 1 -> 145.4 KB at turn 200**, totalling
**15.40 MB for a single 200-turn conversation**. Capture is **on by default**.

There is **no retention path**. The only purge is user-invoked and filtered to
`capture_detail = 'full'`; nothing hard-deletes conversations or messages in production, so the
`ON DELETE CASCADE` never fires and soft-deleted conversations keep their blobs indefinitely.

This is a storage finding, not a latency one - write cost is 0.05-0.20 ms. It matters because the
database it bloats is the one boot migrations and backups walk.

## Acceptance Criteria

- [x] A long conversation does not accumulate a full copy of itself per turn - store a reference, a delta, or a bounded excerpt
- [x] Existing oversized captures are reclaimable without the user knowing to run a manual purge
- [x] Whatever capture retains is still sufficient for the debugging the feature exists to support - say what that is
- [x] Growth re-measured over 200 turns after the change
- [x] `omitted_keys` behaviour is reviewed: it currently omits only `api_key`, while the payload carries the user's entire conversation

## Evidence

Measured with the exact production kwargs shape (`_chat_api_kwargs_from_prepared`), reproduced
independently at 15.35 MB. An earlier probe reported this feature **clean** at 0.2 KB per turn - it
had been built from a hand-made input rather than the real caller's kwargs, and was refuted.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.

## Implementation Plan

1. **Baseline measurement first**, through the REAL production path (not hand-made kwargs): drive
   `ConsoleProviderGateway.stream_chat` with a fake `chat_api_call_fn` for 200 growing turns and
   record per-turn `capture_to_blob` size + total, under Safe (the default).
2. **Bound Safe capture's history retention** in the pure builder (`console_exchange_capture.py`):
   under `CaptureDetail.SAFE`, `build_request_capture` keeps the newest
   `CAPTURE_SAFE_HISTORY_TAIL_ROWS` rows of `messages_payload` verbatim (the turn's delta plus
   immediate context) and replaces each older prefix row IN PLACE with a small fingerprint row
   (role + origin tag preserved, `capture_elided: True`, content replaced by
   `[... N chars, sha256:<16 hex>]`). The elision is idempotent (already-elided rows pass through
   untouched — the export path re-runs the builder over stored requests) and is surfaced through
   the existing `omitted_keys` visibility mechanism the Inspector already renders. FULL captures
   are untouched: Full is the deliberate, consent-gated, purgeable verbatim mode (ADR-092).
   Apply the same bounding to the llama.cpp wire-literal branch's `messages` list.
3. **Reclaim existing oversized captures without user action**: schema v52 -> v53 migration that
   walks `message_exchanges` rows with `capture_detail = 'safe'`, applies the identical trim to
   each stored blob via a pure `trim_safe_capture_blob` helper, and rewrites only rows that
   changed — inside the migration chain's single outer transaction (atomic; SIGKILL rolls back to
   v52 and the deterministic trim re-runs), re-enterable via the guarded version bump, per-row
   corrupt blobs skipped (never brick the DB). Full rows byte-untouched.
4. **Tests**: pure-builder elision/idempotency/inventory pins, a bounded-growth pin through the
   real gateway path, migration tests (historical bootstrap at v52, content-hash equality of
   everything not deliberately trimmed, `integrity_check`, SIGKILL interrupt-safety, re-entry,
   corrupt-blob skip, Full untouched), each proven to fail against a deliberately broken
   implementation (mutation results in Notes).
5. **Re-measure** the same 200-turn run after the change; update the user guide's Safe/Full
   retention paragraph; state the `omitted_keys` privacy stance below; preflight; commit.

## Implementation Notes

**Approach.** Bounded excerpt + per-row references, applied at the one pure seam every
capture flows through (`build_request_capture`), plus a one-time automatic trim of
already-stored blobs in the schema migration chain.

- **Safe (the default) is now bounded.** `elide_safe_history_rows` in
  `Chat/console_exchange_capture.py`: the newest `CAPTURE_SAFE_HISTORY_TAIL_ROWS = 8`
  `messages_payload` rows (this turn's delta plus immediate context) persist verbatim;
  every older row is replaced IN PLACE by a content-free fingerprint row — `role` and
  `__tldw_ephemeral_origin` tag preserved, `capture_elided: True`, content replaced by
  `[conversation history elided by capture policy -- N chars, sha256:<16 hex>]`. The
  digest is sha256 over the row's canonical sanitized JSON — exactly the form the earlier
  capture whose tail the row was new in retained — so row identity is verifiable across
  turns and against the transcript without re-storing bodies. Idempotent (structural
  `capture_elided` key, never content matching), because `console_exchange_export.py`
  re-runs the builder over stored requests. The llama.cpp wire-literal branch
  (`capture_wire_payload` in `console_provider_gateway.py`) bounds its `messages` list
  identically, path-prefixed `wire_payload.messages`. **Full is untouched**: it is the
  explicit, consent-gated, purgeable verbatim mode (ADR-092), and the existing
  user-invoked purge already covers it.
- **What Safe still answers** (AC 3): "what did this call hand the provider adapter" —
  full system prompt, tools, sampling/routing params, response, usage, per-call status,
  the complete payload shape (row count/order/roles/sizes stay truthful in the
  Inspector's Messages section), the turn's NEW content verbatim, and a verifiable
  fingerprint for each elided history row. Elided bodies remain recoverable from the
  transcript, from the earlier capture where the row was in the tail (hash-checkable), or
  by opting a conversation into Full for a debugging session.
- **Reclaim** (AC 2): ChaChaNotes v52→v53 (`_migrate_from_v52_to_v53`) walks
  `message_exchanges WHERE capture_detail='safe'` and rewrites each blob through the pure
  `trim_safe_capture_blob` (returns `None` when nothing changes; never touches Full;
  per-row undecodable blobs are skipped, so one corrupt row cannot brick the DB).
  DML-only Python step — no DDL, no `.sql`, no `VALID_TABLES`/index-census entries.
  Correctness evidence in `Tests/DB/test_chachanotes_v53_safe_capture_trim.py`: value
  identity for everything not deliberately trimmed, `PRAGMA integrity_check` clean,
  re-entry fixed-point, in-process failure mid-walk rewinds blobs AND version stamp
  together, and a real-SIGKILL child (v47-backfill technique) rolls back to v52 and then
  converges byte-identically with an uninterrupted control run. Schema v53 swept against
  all 808 refs and every worktree — no collision (this programme has collided four
  times).
- **omitted_keys stance** (AC 5, recorded beside the allowlist in the module): capture
  exists to answer "what was sent", so conversation content is the subject and is
  retained BY DESIGN — governed by the user-controllable Safe/Full detail policy, not
  the allowlist. Credentials never persist at any level; project-instruction bodies
  never persist under Safe; and every withholding — dropped kwarg, redacted instruction
  body, elided history range (`messages_payload[0..N].content (conversation history
  elided)`) — is named in `omitted_keys`, which the Inspector already renders verbatim.
- **Growth re-measured** (AC 4), real production path (`stream_chat` →
  `_chat_api_kwargs_from_prepared` → `build_request_capture` → `capture_to_blob`), same
  inputs both arms: turn-1 blob 0.9 KB → 0.9 KB; turn-200 blob **217.1 KB → 10.1 KB**;
  200-turn total **21.33 MB → 1.48 MB** (14.4x). Residual growth is the ~25 B/row
  compressed fingerprint index plus the bounded verbatim tail.
- **Readers audited**: Inspector (`console_conversation_inspector.py` — renders rows as
  collapsibles + JSON; fingerprint rows render as ordinary rows, counts stay truthful),
  export projections (`console_exchange_export.py` — idempotency pinned), store/flush
  (`console_chat_store.py` — opaque blob pass-through), loader (`chat_screen.py` —
  `capture_from_storage`, unchanged). Capture failure still never blocks a send
  (existing gateway pin `test_never_break_send_when_build_request_capture_raises` covers
  the new code path).
- **Mutation results** (all reverted): M1 disable elision → 6 tests red (and exposed a
  non-discriminating growth guard whose "lorem" filler compressed ~100x — filler made
  semi-incompressible, guard proven red-capable); M2 re-fingerprint elided rows → red;
  M3 redaction re-measures fingerprints → red; M4 migration trim no-op → 3 migration
  tests red; M6 digest not over row body → red; M7 wire elision dropped → red; M8 trim
  touches Full → red.
- **Tests**: +23 new (13 pure, 5 gateway, 4 migration, 1 updated version pin);
  touched-surface total 567 green (169 core + 398 gateway/inspector). Full
  `Tests/DB/ + Tests/ChaChaNotesDB/`: 1931 passed; 7 pre-existing dev reds
  (A/B-identical failure sets on pristine base c4e52794e2: 1x v27 character-authority
  column pin broken by v52's `thinking_history_policy`, 3x sync_log_retention, 2x v47
  fts backfill, 1x provider_continuation) — not adopted.
- **Modified**: `Chat/console_exchange_capture.py`, `Chat/console_provider_gateway.py`,
  `DB/ChaChaNotes_DB.py` (v53), `Docs/User_Guide/console/context-and-rag.md` (Safe/Full
  retention paragraph now matches the code — the shipped text already promised "bounded"
  Safe retention the code did not honor), tests as above.
- **Diagnostic inventory**: +2 `logger.info` rows in `ChaChaNotes_DB.py` reviewed via
  `check_persistent_diagnostic_inventory.py --statements --since 5d9b4bec5a` before
  `--write`: both are the migration chain's established start/finish pattern,
  interpolating only `db_path_str` and two integer counters — no capture content, user
  text, or secrets (capture blobs are deliberately never logged, even on the
  corrupt-row skip path).
