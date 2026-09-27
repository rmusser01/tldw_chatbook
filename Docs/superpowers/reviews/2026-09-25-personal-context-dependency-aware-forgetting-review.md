# Dependency-aware Personal Context forgetting review

Date: 2026-09-25
Task: TASK-25907.6
Status: User endorsed design; follow-up technical review complete; accepted design direction only

Design: [Dependency-aware forgetting](../specs/2026-09-25-personal-context-dependency-aware-forgetting-design.md)
Decision: [Accepted design ADR-186](../../../backlog/decisions/186-dependency-aware-personal-context-forgetting.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)

## Deliverable and evidence boundary

The proposed design inventories independent retention owners, distinguishes
archive/reversible deletion/immediate forgetting/source deletion/local removal
and delete-everywhere, and specifies registered-lineage suppression and durable
cross-owner cleanup. It adds exact review, encrypted keyed family controls,
source-owned no-memory marks, retirement/purge fences, owner acknowledgements,
remote/offline/restore outcomes and synthetic future acceptance cases.
It proposes an unlinked-profile first release for newly captured V2 claims with
qualified lineage/owner coverage. No deletion, migration, production schema,
worker, provider call or real-profile access occurred under this design task.

The user approved the preceding evidence/temporal design by requesting
continuation; TASK-25907.5 is Done and ADR-185 accepts that design direction.
Its runtime/schema rollout is still unimplemented and separately scoped.

## Independent review and fixes

A fresh read-only reviewer inspected `df9afcc39b..86a3985fdc` against all six
criteria, ADR-102/182/185 and referenced native owners. It reported no Critical
or Minor finding and two Important design gaps. The parent resolved both in
the design; a targeted reread of those amended contracts is recorded below.

| Finding | Resolution in the written contract |
| --- | --- |
| Important: Console captures and semantic trace copies were missing from inventory | Name Safe/Full capture, live exchange/blob cache, persistence queue, semantic artifact/root/maintenance and reconstruction/export owners. Require native cleanup, late-flush/normalization fences and qualification of shared retained roots. Safe capture may preserve the profile-bearing first system row. |
| Important: cancelling a publisher had no durable linearization/acknowledgement | Specify a common cross-process retirement admission gate, lock order, reserved/publishing/terminal ticket states, owner-transaction receipts, drain/review recheck and ambiguous-crash recovery. A publishing ticket must finish or durably prove rollback/no retry; notification/TTL cannot permit the forget fence. |

Parent self-review additionally inventories app-managed llama.cpp snapshot
binaries/working copies under ADR-119. Their missing claim lineage and uncertain
writer state are eligibility gates; delivered external requests being outside
recall does not exempt app-owned copies. The design also fixes a concrete
canonical HMAC selector projection per reviewed scope, separates source-owner
marks from the coordinator's atomic fence, and keeps device-only operations
under peer-local controls without shared private-record receipts.

The targeted read-only reread confirmed both Important findings resolved and
found no new material contradiction. It also confirmed the adjacent managed-cache
coverage gate and exact HMAC projection remain consistent. The documents are
ready for user review as a proposed design. The reviewer did not run checks or
claim adapter, erasure, gate/handoff or server conformance.

## Acceptance mapping

| Criterion | Design evidence |
| --- | --- |
| 1: Actual artifact owners | Inspected owner inventory and capture/trace/cache qualification; V1 unknown coverage is explicit |
| 2: Distinct operations | Operations table, 24-hour Undo evidence and source/manual-content retention choices |
| 3: New IDs/restart/replay/jobs | Source object-wide marks, per-scope keyed family selectors, admission/publication gate and restore checks |
| 4: ADR/crash/privacy/offline | ADR-186, durable phase/ticket protocol, private versus portable controls and honest peer acknowledgement |
| 5: Synthetic cases | Mixed sources/manual Notes, promoted/recreated proposals, captures/late normalization, last-check/commit race, stale snapshots/Sync/recovery and unmanaged exports |
| 6: Reviewed bounded design only | Proposed unlinked V2 release gates, independent review/fixes and scoped documentation boundary checks |

## Verification and limitations

Native scoped checks at the first `dfc252def8` checkpoint passed across the 10
owned documents: all 188 local
Markdown links and 13 task reference/documentation paths resolve; the original
59 child criteria remain unchanged; all 11 family IDs are unique and original
dependencies point backward. TASK-25907.5 was Done and TASK-25907.6 then remained
In Progress with its six technical criteria checked. The new spec/ADR contain
no placeholders; whitespace and the documentation-only diff boundary pass.
The final offline allocation check inspected 585 refs and 84 worktrees without
a competing ADR-186; no network fetch or open-PR audit was performed, so the
integration-time allocation recheck remains required.
No runtime tests or full sweep are required by this documentation-only task.
The reviewer did not judge runtime correctness, actual physical erasure, server
conformance, real profile/database behavior or provider outcomes. Future
adapters, fixtures and synthetic race/crash cases are requirements, not passing
runtime evidence. No evidence/forgetting capability is declared shipped by
finishing these design documents.

Manual source bodies/instruction files remain independent unless separately
reviewed. Unsupported legacy lineage, shared trace roots, managed cache
uncertainty and unavailable owner acknowledgements block a complete managed
scope claim. Offline remote decryption, delivered provider data, arbitrary
exports/backups and forensic/WAL remnants cannot be promised erased.

TASK-25907.6 is Done after the user endorsed the written design subject to the
follow-up review below. ADR-186 accepts design direction only, with allocation
provisional until integration-time branch/PR checks.
All work was executed natively in the existing isolated worktree. The separate
unstaged evaluation task/roadmap addition was preserved outside this task's
commits. No push, PR, merge or destructive operation was performed.

## User-requested follow-up review

The user endorsed the written design and asked for issues/problems/improvements
before continuing. A bounded independent suppression/release review found no
unresolved material issue and suggested clarifying exact fresh authorship.
Parent native source inspection found two additional Important gaps and resolved
them before continuation:

- AgentRunsDB steps/terminal recovery and segmented filesystem run logs are now
  explicit managed owners. `Agents/agent_service.py`'s `_safe_run_log_content`
  and `on_record`, plus `Agents/run_log.py`'s writer/search surfaces, can preserve
  ordinary profile facts despite credential/path sanitization. Exact whole-unit
  review, native maintenance, late writer/recovery and read/export fences, file
  authority and unknown historical coverage are now required.
- A prepared checkpoint or queued handoff no longer counts as delivered data.
  `Chat/console_provider_gateway.py`'s adapter-entry gate already distinguishes
  cancellation from adapter ownership. The future retirement gate must qualify
  that final boundary, cancel/retire pending payloads and retries, and report
  already-entered work as potentially begun/uncertain rather than confirmed
  delivery. App-owned copies remain cleanup participants.

Fresh authorship now admits only the exact reviewed new record/version and
authorized input; family/source controls still reject replay and later automatic
versions. Synthetic cases cover audit/log units, queued dispatch and that exact
exception. One targeted independent reread confirmed the amendments introduce
no material contradiction; no runtime/readiness claim was made.

Follow-up native checks passed: 190 local Markdown links and 13 task
reference/documentation paths resolve across the 10 owned documents. All 59
original child criteria are unchanged; 11 family IDs are unique and original
dependencies point backward. Tasks .5/.6 are Done with six checked criteria
each. The new spec/ADR have no placeholders; whitespace and the documentation
only boundary pass. Independent evaluation task/roadmap bytes are unchanged. No production
code, fixtures, real data, provider calls or permission defaults changed.
