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

## Approved Design

- [Console Safe Capture Retention design](../../Docs/superpowers/specs/2026-08-27-console-safe-capture-retention-design.md)
- [ADR-096: Bound Safe exchange-capture history](../decisions/096-console-safe-capture-retention.md)

ADR required: yes

ADR path: `backlog/decisions/096-console-safe-capture-retention.md`

Reason: the task changes durable capture retention, privacy semantics, and the ChaChaNotes data migration contract.

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

**Approach.** Implemented to the approved design (ADR-096 /
`Docs/superpowers/specs/2026-08-27-console-safe-capture-retention-design.md`): a bounded
diagnostic excerpt plus ONE content-free aggregate marker, applied at the one pure seam
every capture flows through (`build_request_capture`), plus a one-time automatic
compaction of already-stored blobs in the schema migration chain. A first
implementation used per-row sha256 fingerprints; the owner converged it to ADR-096,
which forbids retaining any digest of elided text (a digest is a guess-verification
oracle for omitted private content) — the fingerprints were removed entirely.

- **Safe (the default) is now bounded** (`compact_safe_history_rows` in
  `Chat/console_exchange_capture.py`): the retained set is first-`system` row ∪
  last-`user` row ∪ final eight physical rows (deduplicated, original order, values
  untouched; non-mapping rows eligible only via the tail). Everything else is
  represented by one versioned marker at the first omitted position carrying ONLY:
  kind/version discriminator, original row count, omitted row count, normalized
  omitted-role counts (system/user/assistant/tool/other — unknown/missing/non-string
  roles count toward `other` and their raw values never reach the marker), and the
  retained rows' original positions. No content, snippets, per-row lengths, hashes,
  IDs, or timestamps. Idempotent by strict structural marker recognition (exact key
  set + types): re-projection through the export path is a fixed point, an input
  marker never disables compaction of surrounding rows, malformed lookalikes are
  ordinary rows. Ordering per ADR-096: allowlist → endpoint identity → instruction
  redaction → credential/binary sanitization → compaction → shared budget. The
  llama.cpp wire-literal branch (streaming AND its stream→complete fallback) compacts
  its `messages` list through the same helper. **Full is untouched** (explicit,
  consent-gated, purgeable verbatim mode — ADR-092).
- **What Safe still answers** (AC 3, per the design): what initial system framing was
  in effect, what current user request drove the call, and what immediate
  assistant/tool loop surrounded it — plus full system prompt, tools, sampling
  params, response, usage, status, and honest counts of what was elided. It is no
  longer an exact historical record; users who need that choose Full before the send.
- **Elision visibility** (AC 5): the stable paths `messages_payload.history` /
  `wire_payload.messages.history` fold into `omitted_keys` (rendered on the
  Inspector's existing "Omitted by capture policy" line; stable so repeated
  projection cannot create duplicate or ever-changing strings), and the Exchange
  tab's Messages title now reads "Messages (N sent; M elided by capture policy)"
  from the marker — the compacted physical count alone would under-state the send.
  omitted_keys stance recorded beside the allowlist: conversation content is the
  capture's subject, retained by design under the Safe/Full policy; credentials
  never persist; instruction bodies never persist under Safe; every withholding is
  named.
- **Reclaim** (AC 2): ChaChaNotes v52→v53 (`_migrate_from_v52_to_v53`) keyset-pages
  Safe `message_exchanges` rows in bounded batches (100 rows), rewrites only blobs
  the pure `trim_safe_capture_blob` changed, inside the migration chain's outer
  immediate transaction. Only the `CaptureUnavailableError`/`CaptureCorruptError`
  family is a per-row skip (unreadable rows stay byte-identical and are counted;
  the version may still advance); any unexpected error aborts and rolls back blobs
  AND version stamp together. Diagnostics are aggregate-only (examined/compacted/
  skipped + exception class), never content. Full, small, and already-compacted
  blobs stay byte-identical. Evidence
  (`Tests/DB/test_chachanotes_v53_safe_capture_trim.py`): value-identity for
  everything not deliberately compacted, `PRAGMA integrity_check` clean, re-entry
  fixed point, in-process mid-walk failure rolls back to a working v52, and a
  real-SIGKILL child (v47-backfill technique) rolls back then converges
  byte-identically with an uninterrupted control run. Migration probe on a
  200-turn/21.25 MB historical fixture: open+migrate **537.9 ms**, Python peak
  allocation 21.4 MB (bounded by the batch), blob bytes 21.25 MB → **0.99 MB**.
  The db FILE size is unchanged until SQLite reuses the freed pages — logical
  reclaim only; no VACUUM, and no forensic-erasure claim (ADR-096).
- **Growth re-measured** (AC 4), real production path (`stream_chat` →
  `_chat_api_kwargs_from_prepared` → `build_request_capture` → `capture_to_blob`),
  same inputs all arms: per-turn blob 0.9 KB at turn 1, then **plateaus flat at
  5.2 KB** (marker + 8-row tail) from turn 50 through turn 200 — cumulative growth
  is linear. 200-turn totals: **21.33 MB before → 1.01 MB after** (the interim
  fingerprint design measured 1.48 MB and still grew per-row; the aggregate marker
  is O(1)).
- **Readers audited**: Inspector (renders the marker row naturally; Messages title
  now surfaces sent/elided counts — pinned in
  `Tests/UI/test_console_conversation_inspector.py`), export projections
  (`console_exchange_export.py` — fixed-point pinned; Safe export cannot
  reconstruct omitted rows), store/flush and loader (opaque blob pass-through,
  unchanged). Capture failure still never blocks a send (existing gateway pin
  covers the new path).
- **Mutation results** (all reverted, each killed by a named test): N1 compaction
  disabled → 17 red across pure/gateway/migration/inspector; N2 marker recognition
  removed → fixed-point + input-marker tests red; **N3 digest reintroduced
  "silently" (sha256 key added AND the frozen key set extended in the same edit) →
  killed by `test_marker_shape_guard_a_digest_cannot_be_reintroduced_silently`**
  (only-string-is-the-kind-discriminator + no-"sha256"-anywhere assertions); N5
  input marker disables compaction → red; N6 last-user retention rule removed →
  red; N7 stored-blob compaction no-op → 3 migration tests red; N9 marker
  validator accepts key supersets → malformed-lookalike test red. Earlier
  fingerprint-round mutations also exposed a non-discriminating growth guard
  ("lorem" filler compressed ~100x) — all history fixtures now use
  semi-incompressible hex-word filler.
- **Tests**: 15 pure-contract tests, 5 gateway-path tests, 4 migration tests, 1
  Inspector title pin, 1 updated schema-version pin. Touched-surface totals: 170
  (capture/export/store/controller/policy/exchanges/thinking/atomicity) + 398
  (gateway + inspector suites) green. Known pre-existing dev reds (A/B-identical
  on pristine base): 7 in Tests/DB//Tests/ChaChaNotesDB (v27 authority pin broken
  by v52's `thinking_history_policy`, 3x sync_log_retention, 2x v47 fts backfill,
  1x provider_continuation) — not adopted.
- **Modified**: `Chat/console_exchange_capture.py`, `Chat/console_provider_gateway.py`,
  `DB/ChaChaNotes_DB.py` (v53), `Widgets/Console/console_conversation_inspector.py`
  (honest Messages title), `Docs/User_Guide/console/context-and-rag.md`, tests as
  above. Schema v53 swept against all remote refs and every worktree at bump time
  and re-swept before each commit — no collision.
- **Diagnostic inventory**: the migration's `logger.info` rows reviewed via
  `check_persistent_diagnostic_inventory.py --statements` before `--write`: the
  established start/finish pattern, interpolating only `db_path_str` and integer
  counters — no capture content, user text, or secrets; the failure wrap logs the
  exception CLASS only (ADR-096: exception values may contain decoded content).
