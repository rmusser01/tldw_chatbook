---
id: TASK-21130
title: >-
  TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
labels:
  - performance
  - tts
  - migrations
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21130).

`TTS/profile_schema.py:1300-1316` (`_migration_reference_snapshot`) pulls `wav_bytes` for every
row, and is called at :1439 and :1468 with the first snapshot still held - up to ~1 GB peak at
profile-store open under the 512 MiB store bound (profile_reference_types.py:21), against the
module's own 256 KiB streaming norm. Swap-thrash on constrained hardware at exactly the moment
TTS opens.

## Acceptance Criteria

- [x] The migration compares projections without wav_bytes (sibling profile_migration_candidate.py:320 already does) using hashes; peak memory during v3->v4 is bounded and asserted by a test with synthetic large references
- [x] Migration outcomes byte-identical for the existing fixtures
- [x] **The same defect is fixed on the path the live repository actually
      upgrades through** (`_step_candidate`), not only on the call sites the
      finding cited - see "Where the finding was wrong" below
- [x] The climb stays atomic, re-enterable and interrupt-safe: proved by a
      SIGKILL-mid-transaction test, a failed-attempt re-entry test, and
      `PRAGMA integrity_check` after every arm
- [x] Every new assertion is killed by at least one deliberately broken
      implementation (12-mutation matrix)

## Where the finding was wrong

The finding cited `profile_schema.py:1439/1468` (`_run_migrations`) as the
boot-path cost. Measured truth: `TTSProfileRepository._worker_initialize_store`
**never** lets `open_profile_store` migrate a populated store - a store below
the current version is routed to `_worker_publish_migrated_store` ->
`migrate_profile_store_to_candidate` -> `step_profile_migration_candidate`, and
only then reopened at v4 (profile_repository.py:1636-1638). The live upgrade a
user actually pays for therefore ran through `_step_candidate`
(profile_migration_candidate.py:413/431), which carried the **identical**
double-snapshot - measured at the same 966 MiB peak. Fixing only the cited
lines would have left the whole user-visible cost in place. Both sites are
fixed; the finding's diagnosis was right and its call-site attribution was
incomplete.

The AC's prescribed shape ("compare projections without wav_bytes using
hashes") was verified rather than assumed before it was implemented, because
dropping the payload from the comparison removes the only direct check on the
BLOBs. It holds, but only as a three-link chain that has to be kept intact -
documented at `_migration_reference_evidence` and pinned by mutation arms M2
and M3.

## Implementation Plan

1. Build the measurement + differential harness first, at the merge base:
   a v3 store of real canonical WAV references at the 512 MiB store bound,
   migrated in its own process, reporting wall time, `tracemalloc` peak, RSS
   high-water and a content hash of every reference column.
2. Baseline it on pinned dev `f49956038` in a second worktree.
3. Replace the payload-carrying projection with the sibling's payload-free
   evidence, as one shared definition used by both migration paths.
4. Re-measure interleaved against the same pinned base; prove the peak stops
   scaling with the table by holding total bytes constant and varying the
   per-reference size.
5. Add tests for peak scaling, payload-free evidence, byte identity,
   payload-only and metadata-only mutation rejection, corrupt-payload and
   poisoned-digest rejection, re-entry after failure, SIGKILL mid-transaction,
   already-migrated and fresh/empty stores.
6. Run every new test against deliberately broken implementations and require
   each to go red.

## Implementation Notes

`_migration_reference_snapshot` selected `wav_bytes` for every row and was
called twice with the first result still referenced, so peak allocation was
2x the whole reference table. It is replaced by
`_migration_reference_evidence`, which projects every column **except** the
payload and substitutes the transcript with its UTF-8 length + digest. The
sibling `_compact_reference_evidence` in `profile_migration_candidate` now
delegates to it, so both migration paths share one definition, and
`_step_candidate` - the path the live repository upgrades through - uses it
too.

Byte-for-byte identity is preserved by transitivity instead of retention:
`_validate_migration_reference_rows` streams each BLOB and re-derives
`sha256(wav_bytes)` against the stored `sha256` column on **both** sides of
the climb, and that column travels verbatim in the evidence. `blob_before ==
sha_before`, `sha_before == sha_after`, `sha_after == blob_after`. Mutation
arms M2/M3 exist precisely to keep both ends of that chain from being deleted
by a later change.

Measured on an M-series laptop, 22 references x 23.04 MB = 483 MiB (the
512 MiB `MAX_REFERENCE_TOTAL_BYTES` bound), 5 interleaved runs per arm,
each arm in its own process, A/B against merge-base `f49956038`:

| path | metric | before | after |
|---|---|---|---|
| `open_profile_store` | tracemalloc peak | **966.9 MiB** | **88.0 MiB** (11.0x) |
| `open_profile_store` | RSS growth (median) | 1,218 MB | 262 MB (4.6x) |
| `open_profile_store` | wall (median) | 5.92 s | 5.49 s |
| `step_profile_migration_candidate` (live path) | tracemalloc peak | **966.9 MiB** | **88.0 MiB** |
| `step_profile_migration_candidate` | wall (median) | 11.09 s | 10.05 s |

The second snapshot is gone rather than relocated: holding total reference
bytes constant at 483 MiB and changing only the per-reference size, the base
peak is unchanged (1,013.8 MB at 22x23 MB vs 1,014.3 MB at 220x2.3 MB - it
tracks the TOTAL) while the fixed peak falls with it (92.2 MB -> 9.7 MB - it
tracks the LARGEST SINGLE reference). The residual 88 MiB is the pre-existing
per-row payload validation (`read_reference_blob` + canonical-WAV re-encode),
now bounded by `MAX_REFERENCE_CANONICAL_BYTES` (32 MiB) instead of
`MAX_REFERENCE_TOTAL_BYTES` (512 MiB).

Migration outcome is byte-identical: the content hash over all twelve
reference columns including `wav_bytes` is the same value before and after
migration, and the same value in the base and fixed arms, for both stores.
`PRAGMA integrity_check` is clean in every arm.

Fifteen new test cases in `Tests/TTS/test_profile_reference_storage.py`
(4,090 -> 4,105 passing in `Tests/TTS/`). The one red,
`test_app_lifecycle_shutdown_drains_all_owners_in_authority_order`, fails with
a byte-identical `AttributeError: 'types.SimpleNamespace' object has no
attribute '_shutdown_actor_pack_import'` on pristine `f49956038` - pre-existing
dev red, filed separately.

Mutation matrix (12 arms, each run against the whole file): M1 restore the
BLOB projection, M2/M3 delete either end of the payload-identity chain, M4
delete the evidence comparison, M5 blind the transcript digest, M6 detach the
candidate projection, M7 commit before validation, M8 remove the transaction,
M9 rewrite payloads during the migration, M10 migrate an already-current
store, M11 skip the evidence for pre-v3 sources, M12 remove the failure
rollback. Every new test is killed by at least one arm. M12 killed nothing:
the explicit `connection.rollback()` is redundant with the connection close
that follows it under `sqlite3` - pre-existing belt-and-braces, no new claim
depends on it. M3 initially killed nothing either; the poisoned-digest test
was corrected to require that the migration is never *entered*, which isolates
the link it claims to guard.

`Docs/security/production-diagnostic-inventory.json` was regenerated for
pre-existing drift in `UI/Screens/library_screen.py` (a file this task does
not touch). All eight drifted rows were read first: they are pure line
re-wraps with byte-identical message text and arguments, no new interpolation
of user content, secrets or paths. The same preflight failure reproduces on
pristine `f49956038`.

Modified: `tldw_chatbook/TTS/profile_schema.py`,
`tldw_chatbook/TTS/profile_migration_candidate.py`,
`Tests/TTS/test_profile_reference_storage.py`,
`Docs/security/production-diagnostic-inventory.json`.
