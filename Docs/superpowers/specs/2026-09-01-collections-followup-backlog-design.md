# Collections Follow-up Backlog Design

**Date:** 2026-09-01

**Status:** Approved for backlog creation

**Repositories:** `tldw_chatbook`, `tldw_server`

**Foundation:** TASK-18919 and ADR-107

## Goal

Turn the four known post-reader opportunities into an ordered, atomic backlog without reopening
TASK-18919 or implying that unsupported Server behavior is safe. Cross-repository capabilities are
implemented and attested by `tldw_server` before Chatbook enables them. Broader Collections
management is split by workflow so each implementation task can ship in one PR. The legacy
recovery lifecycle begins with a decision task because migration and retirement have materially
different data-loss, rollback, and compatibility consequences.

## Approaches considered

### Selected: atomic cross-repository tasks

Create two Server capability tasks, two corresponding Chatbook integration tasks, a Chatbook
tracking parent with three workflow children, and one Chatbook ADR decision task. This produces
nine records whose ownership and completion evidence are unambiguous.

### Rejected: six coarse tasks

One task per requested line plus two Server prerequisites would make templates, digests, and
import/export one multi-PR implementation unit. It would also mix the legacy lifecycle decision
with an implementation whose desired behavior is not yet known.

### Rejected: duplicate all management work in both repositories

The Server already exposes output-template, Reading digest, and Reading import/export contracts.
Creating speculative Server rewrites for those workflows would duplicate existing product surfaces.
Chatbook tasks should consume and capability-check the existing contracts, and should create a
Server follow-up only if implementation discovery finds a concrete contract gap.

## Task structure

### `tldw_server` capability prerequisites

#### S1. Add atomic revision-guarded hard delete for Reading items

The permanent-delete mutation must accept an exact expected revision or equivalent precondition and
enforce it in the same transaction as deletion. A stale precondition returns a documented conflict
without deleting the item. The operation remains user-scoped, removes capture-owned children, and
does not delete linked external Media or Notes. A versioned docs-info capability is advertised only
when this contract is active. SQLite and PostgreSQL concurrency, authorization, not-found,
conflict, cascade, and diagnostic-privacy behavior are covered.

#### S2. Add coherent scoped tag and domain aggregates for Reading items

Expose bounded, deterministic, fully pageable tag and domain aggregate results for an explicitly
documented Reading scope. Aggregate rows and exact aggregate totals must be evaluated coherently
with the accepted search/status/favorite/date/tag/domain filters; the contract must state how a
facet's own active filter affects its values. The result is user-scoped and avoids returning private
capture content or sensitive URL components. A versioned docs-info capability is advertised only
when the endpoint and its SQLite/PostgreSQL snapshot guarantees are active.

### `tldw_chatbook` capability integrations

#### C1. Enable capability-gated Server capture hard delete

Reference S1 as an external prerequisite rather than using a cross-repository Backlog dependency
identifier. Chatbook keeps Server hard delete visibly unavailable until the exact capability is
positively established. It sends the loaded capture revision with the destructive request, preserves
the item on conflict or unknown outcome, refreshes authoritative state, and removes the row only
after confirmed deletion. Confirmation and cleanup semantics continue to follow ADR-055 and
ADR-107, and linked Media or Notes are never presented as deletion targets.

#### C2. Present complete capability-gated Server tag and domain facets

Reference S2 as an external prerequisite. Until attested aggregates exist, retain the current typed
filters and never label suggestions from returned rows as complete facets. When supported, expose
bounded, searchable tag and domain facet values with exact counts, deterministic paging, complete-
scope filtering, generation fencing, and explicit loading/empty/error/retry states. Source changes
must discard prior-authority facet state, and narrow layouts must preserve the adaptive reader.

### `tldw_chatbook` Collections-management program

#### C3. Tracking parent: complete Server Collections management workflows

Track three independent children and close only when all three are Done. The parent describes the
cohesive outcome but contains no implementation plan of its own.

#### C3a. Manage Server Collections output templates

Add bounded browse, detail, create, edit, and safe delete for the Server output-template contract
used by Collections workflows. Capability and authorization failures remain explicit; Local mode
does not claim template parity. Validation, conflicts, pagination, and destructive confirmation are
covered by service and mounted tests.

#### C3b. Manage Server Reading digest schedules and outputs

Add bounded schedule browse/detail/create/edit/enable-disable/delete plus output history and run
status using the existing Reading digest contracts. Timezone and recurrence values are presented in
human terms, destructive actions are confirmed, and unavailable worker/scheduler capabilities are
truthful. Local mode remains unsupported unless a separate Local design is approved.

#### C3c. Manage full Server Reading import and export workflows

Add bounded import admission and job progress/recovery, plus export format/scope controls and safe
artifact delivery using the existing Reading contracts. The UI distinguishes accepted, running,
partially successful, failed, cancelled, and unknown outcomes; it does not load unbounded files or
leak paths. Round-trip and production-shaped mounted coverage verify Pocket/Instapaper-compatible
workflows. Local capture import/export is not inferred from the Server contract.

### `tldw_chatbook` legacy lifecycle decision

#### C4. Decide migration or retirement of legacy generic Collections recovery

Inventory actual v1 data, recovery usage, compatibility promises, and supported downgrade paths.
Compare at least: retained export-only recovery, explicit user-approved migration into captures,
and retirement after a defined release/notice window. The task produces a new accepted ADR that
defines authority mapping, identity and canonical-URL collisions, membership handling, consent,
backup/export requirements, rollback, retention, telemetry/privacy boundaries, and removal gates.
No legacy data is mutated or deleted in this decision task. Atomic implementation tasks are created
only after the ADR selects an outcome.

## Ordering and references

- S1 and S2 are independent Server tasks.
- C1 references S1 and remains blocked on its attested capability.
- C2 references S2 and remains blocked on its attested capability.
- C3a, C3b, and C3c are independent children of C3 and depend on completed TASK-18919 behavior.
- C4 depends on TASK-18919 and references ADR-107; it requires a new ADR but no code change.
- Chatbook tasks reference the eventual Server task file or repository URL rather than declaring a
  same-repository dependency on a potentially colliding numeric task ID.

## Backlog quality rules

- Each implementation record describes user-visible or externally verifiable outcomes, not a file
  edit sequence.
- Every task is sized for one PR; the tracking parent is explicitly non-implementing.
- Capability tasks fail closed until positive versioned evidence exists.
- No task claims Local/Server parity where one authority lacks the feature.
- No task references an uncreated future task ID.
- Implementation plans and final ADR decisions are added only when their tasks enter progress.

## ADR assessment

- S1: a Server ADR check is required during planning because it changes a destructive API contract;
  an existing Server ADR may be linked if it already governs the exact atomic-delete lifecycle.
- S2: a Server ADR check is required during planning because it adds a public aggregate contract;
  reuse an applicable existing ADR or create one then.
- C1 and C2: no new Chatbook ADR is expected because they implement ADR-107's existing fail-closed
  capability boundary; planning must still record the check.
- C3a, C3b, and C3c: planning must assess whether existing Server ownership contracts fully govern
  the long-lived UI workflows; no storage decision is made by these backlog records.
- C4: a new ADR is mandatory and is the task's primary deliverable.

