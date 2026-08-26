# ADR-065: Active Ingest Source Admission and Override

Status: Accepted
Date: 2026-08-13
Related Task: TASK-208
Supersedes: N/A

## Decision

Library ingest uses a default-on, active-only source admission guard. A source
already represented by a `QUEUED`, `PARSING`, or `WRITING` job for the same
backend is refused before submission. The user may override that refusal for one
unchanged request through the Library's inline two-press Start confirmation.

Active identity is `(backend origin, canonical source)`. Filesystem paths use
side-effect-free lexical normalization with platform case semantics. HTTP(S)
identity normalizes scheme, host, default port, empty root path, and fragment
only; it preserves path bytes and query order. Folder admission is atomic across
the existing bounded expansion.

The screen previews admission from the direct source or the candidate paths
already produced by background preflight; it performs no new filesystem scan.
The app repeats one outer guard immediately before local job creation or remote
submission, then routes admitted folder members through a private seam that
cannot re-enter the guard. The override is an immutable scope containing an
opaque deterministic candidate-set digest/count and the bounded active job IDs
the user consented to. The app re-expands and accepts that scope only when the
candidate identity is exact and every current active match is covered. Expected
refusal carries that privacy-safe scope plus bounded job ID/state references so a
late authoritative refusal can re-arm without exposing paths. The override is
explicit, reason-specific, one-shot, and never persisted.
If the bounded active-ID list is incomplete, the scope records that fact and
cannot authorize an override; truncation is therefore fail-closed.

## Context

The registry permits multiple jobs with the same `source_path`. Repeated clicks,
Enter key repeat, or a second submission while long parsing is still underway can
therefore create duplicate active work and duplicate receipts.

Adjacent mechanisms solve different problems. Preflight and the writer detect
some content already stored in Library; `overwrite_existing` controls historical
persistence; terminal history supports retry and audit. None is an active
admission rule, and using them as one would either require blocking work at the
UI boundary or prevent legitimate re-ingestion of changed files.

ADR-014 places Library ingest coordination at the app/service boundary and keeps
submitted lifecycle authority out of widgets. The guard must therefore be visible
to the screen for consent but authoritative at the app boundary.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Silently return the existing registry job | It makes refusal look like successful queue creation, encourages incorrect form clearing/receipts, and provides no honest override contract. |
| Add a database idempotency key or partial unique index | Session-active safety does not require durable uniqueness; persistence adds migration, expiry, retry, and cleanup policy while blocking legitimate changed-file re-ingestion. |
| Compare all historical jobs | Terminal jobs are evidence, not locks. Their source may have changed, and users must be able to re-import deliberately. |
| Hash content before admission | Hashing adds filesystem/database work, overlaps existing Library content-match behavior, and cannot cover remote URLs without a fetch. |
| Use a modal confirmation | The established inline two-press grammar preserves form context, keyboard parity, focus, and scroll with less interruption. |
| Skip only matching members of a folder | A partially submitted batch makes the cleared form and queue receipt unable to explain which original members were silently omitted. |

## Consequences

Accidental duplicate active work is stopped by default across button and keyboard
entry points. Local and Server remain independent scopes. Terminal history never
blocks a new run, and intentional duplicate active work remains possible through
one visible, deliberate second press.

The source normalizer is not a media identity. It performs no symlink resolution,
filesystem stat, content read, hash, database lookup, network request, redirect
resolution, query sorting, or tracking-parameter removal.

The app performs the binding check before the first local append or remote call.
Added, removed, or changed folder members and newly active unmatched job IDs
invalidate a supplied scope. A consented active job's ordinary lifecycle
transition remains covered, and terminal matches may disappear without blocking
normal submission. Expected refusal creates no failed job and transfers no
external-model resource. The screen's preview is explanatory defense in depth,
not authority.

Consent fingerprints the candidate-set digest/count, tooling affected-file count,
and stable matching job IDs, not lifecycle states. Ordinary queued/parsing/writing
progress preserves the second press; candidate membership, terminal membership,
request, backend, option, warning, or affected-count changes invalidate it. Focus
movement alone does not. A tooling-only confirmation cannot authorize a duplicate
that was not part of its armed reason set.

The inline instruction remains complete in the fixed one-row gate at the minimum
supported Library geometry. Its text, not warning color or a glyph, communicates
the active-import state and the second-press action.

No schema, dependency, setting, persistent preference, or historical cleanup is
introduced. ADR-014 remains authoritative for ingest ownership, service authority,
and recovery.

## Links

- [TASK-208 design](../../Docs/superpowers/specs/2026-08-13-task-208-active-ingest-source-admission-design.md)
- [ADR-014: Library Ingest Service Authority and Recovery](014-library-ingest-service-authority-and-recovery.md)
- [TASK-208](../tasks/task-208%20-%20Optional-source_path-dedup-for-ingest-submissions-idempotency.md)
