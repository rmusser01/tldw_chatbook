# ADR-218: File Notes replica retention bounds

Status: Proposed 2026-10-04 (wave-5 reslice of the superseded TASK-399
B-phase; numbering follows the wave-5 brief's "209+" provenance note and sits
above dev's shipped ceiling ADR-217)
Date: 2026-10-04
Related Task: [TASK-34382](../tasks/task-34382%20-%20File%20Notes%20replica%20retention%20policy.md)
Amends: [ADR-029](029-file-notes-disk-authority.md) — not superseded. ADR-029
deferred "quotas and recovery tooling until demonstrated need"; the unbounded
growth its Context did not anticipate is now demonstrated, and this ADR sets
the replica-side retention bounds inside ADR-029's unchanged authority model.

## Context

ADR-029's replica stores, outside the selected folder, the latest observed
bytes of every supported file plus coalesced pre-edit checkpoints for
protected paths and deletion snapshots for every delete. Nothing ever removes
any of it:

- `revisions` rows accumulate once per editing session per protected file,
  forever (a long-lived protected note edited weekly grows ~52 rows/year);
- a tombstoned `files` row and its `delete` revision outlive the deletion by
  years even though the user has visibly moved on.

Disk remains authoritative and the replica can always be rebuilt from it, so
unbounded retention is not a correctness problem — it is a storage and
hygiene problem inside the owner-secured `file_notes.sqlite` that ADR-029
fixed as the single replica.

## Decision

The replica enforces retention for one root in a single transaction, invoked
by the File Notes workspace when the selected root changes (after the scan,
before the Recently-deleted listing is read) and when the workspace session
ends. Retention failure is logged and never blocks a root change, a shutdown,
or any user action.

1. **Per-note checkpoint cap.** Each path keeps at most the
   `MAX_REVISIONS_PER_NOTE = 50` most recent `pre_edit` revisions (by
   parsed UTC `created_at`, insertion order as tiebreak); older ones are
   deleted. Protected paths are capped too — protected saves are the only
   checkpoint writer, so exempting them would leave exactly the growth
   this ADR exists to bound unbounded (PR #3016 review round). `delete`
   revisions are not counted against the cap — their lifetime is the
   tombstone expiry below.
2. **Tombstone expiry.** A tombstoned row whose `deleted_at` is older than
   `RECOVERY_EXPIRY_DAYS = 30` days is dropped together with its FTS row.
   **Exactly one most-recent tombstone in the root is never evicted**,
   whatever its age, chosen by parsed instant with a deterministic
   tie-break, so the user's last deletion always remains restorable and a
   tied greatest timestamp cannot exempt every peer (PR #3016 review
   round).
3. **Revision expiry.** Revisions older than the same 30-day cutoff are
   deleted, except revisions of protected paths and the preserved
   tombstone's own `delete` revision — identified by row identity, not
   timestamp equality, because the deletion writer records `created_at`
   and `deleted_at` independently (PR #3016 review round).
4. **Protected paths are exempt from expiry, not from the cap.** Exact-match
   and component-bounded prefix protections (`protected_paths`) both count,
   per ADR-029's protection semantics; the per-note checkpoint cap above
   still applies to protected paths.
5. **Fail-safe timestamps.** Retention compares parsed UTC timestamps;
   a value that cannot be parsed is kept, never evicted. Stored timestamps
   use mixed `Z`/`+00:00` (and caller-supplied offset) ISO spellings across
   the codebase's history, so comparison happens in Python, not in SQL
   string ordering.

## Alternatives Considered

- **No retention (status quo).** Rejected: the replica grows without bound
  precisely for the users ADR-029's protections exist for.
- **Configurable bounds.** Rejected for now: one fixed, generous bound per
  class is the first release's answer (mirrors ADR-029's own "sufficient for
  the first release" stance); knobs arrive with demonstrated need.
- **Whole-database vacuum / rebuild-from-disk on startup.** Rejected: heavier
  than the bounded sweeps, touches data the user may still want (recent
  tombstones), and rebuilds do not address checkpoint accumulation.
- **Evicting protected paths' revisions by age.** Rejected: protection is
  ADR-029's explicit pre-edit safety commitment; expiring it silently would
  hollow the guarantee. (The per-note *cap* still applies — see Decision 4
  and the PR #3016 review round, which aligned the rules with this ADR's
  own Consequences.)

## Consequences

- The replica's recovery horizon for unprotected paths is 30 days and the
  most recent deletion; protected paths keep their full checkpoint history up
  to 50 sessions.
- The task-34381 revisions read-path (history listing, verify, export,
  restore) simply reflects retention: expired revisions stop being listed.
- No schema change: bounds live in module constants and are enforced with
  the existing tables.
- If a user re-selects a root after months away, the first root change prunes
  the stale tombstones before the Recently-deleted list is read, so the list
  never names entries the policy just expired.
