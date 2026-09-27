# Foreground source-inspection design review

Date: 2026-09-26
Task: TASK-25907.15
Status: User-endorsed design accepted after verified inline and independent review fixes; design only.

Scope: [specification](../specs/2026-09-26-personal-context-foreground-source-inspection-design.md)
and [Accepted design ADR-191](../../../backlog/decisions/191-foreground-personal-context-source-inspection-authority.md).

## Findings resolved in the written contract

- Profile inspection and agent grants did not supply conversation access. Added
  independent trusted-host source admission before body read, exact open-session
  membership and native namespace checks. Current DTO validation cannot mint it.
- A user-role row did not attest authored assertion. Text matching now leaves
  origin, capture trust, semantic support and approval explicitly unverified.
- A new reader lock alone would miss existing mutations and revocations. Require
  actual writer coverage, a final publication gate and visible-result invalidation.
- A fresh snapshot could later become stale through another process. State
  observation-time semantics separately from revocation safety; require a shared
  revocation gate or unsupported deployment. No continuously verified badge.
- Current V1 cannot contain the full binding. Runtime planning requires a real
  qualified containing-record admission path; never assemble one from legacy IDs
  or add an unconsumed resolver under a fictional authority wrapper.
- Worker locks could accidentally cross a queued UI callback. The final UI phase
  now acquires its own nonblocking leases, checks bounded metadata only and
  performs no full-text hashing/decryption; slow or unavailable authority fails
  closed under the 100 ms UI budget.
- Full-body hashing and rendering were unbounded. Set explicit stored/decoded
  limits, span limit and deadline; no truncated verification or active markup.

## Evidence and limits

At the initial drafting checkpoint, native reads checked profile Settings inspection,
agent profile composition,
source locator/citation namespace contracts, Console lifecycle ownership and
ADR-185 plus the completed source audit. That initial pass was an inline document review, not a production authority
test; the later independent review and confirmation are recorded below. No app, real
profile, keyring, provider, server or network was accessed.

At allocation time available Git object paths and 82 existing worktrees showed
no TASK-25907.15; maximum ADR was 190. No fetch was performed. These numbers need
a fresh integration check. Native scoped validation passed with
`python3 /private/tmp/check_memory_source_inspection_design.py`: five owned
document paths, 68 valid local links, 16 unique family IDs, six written-contract
criteria checked and one user-review criterion pending. The guard checked all
older owned task bytes against HEAD, backward-only dependencies, proposed ADR
and In Progress status, placeholder/whitespace rules, zero changed runtime paths
and exact independent task/roadmap-suffix SHA-256 preservation. `git diff --check`
passed. The helper is a local verification receipt, not a shipped test suite;
no runtime tests were necessary for this document-only checkpoint.

## User-requested technical review and verified corrections

The user endorsed the concrete contract and requested review before continuing.
One fresh read-only reviewer inspected `b8bee0039a..50d20996f6` plus scoped native
owners; no code, index, branch, profile, keyring, app, tests or network activity.
It found no Critical issue, one Important and one Minor; both were verified and
fixed in the written contract:

- **Important — fresh final snapshots.** ChaChaNotes_DB.py's transaction context
  borrows a caller-owned transaction (`_enter_transaction`, around line 24010),
  so a final read could retain an old WAL snapshot. Source and profile probes now
  require fresh committed owner-managed snapshots, reject borrowed transactions,
  and qualify late two-connection edits/deletes plus an already-active read.
- **Minor — narrow read-only version acquisition.**
  `console_semantic_revision.py:ensure_current_revision` lazily creates metadata;
  its envelope hydrates image BLOBs, attachments and sidecars. Inspection now
  requires existing metadata, no lazy writes, and tiny-text/huge-unrelated-payload
  controls plus UTF-8/NUL byte guards before hydration.

Root independently verified and fixed five related gaps:

1. Settings owns the foreground provenance surface, while ChatScreen's suspend
   preserves its runtime/controller. The route is now an app-owned coordinator
   from the foreground Settings inspector to that surviving source owner. Source
   owner replacement still invalidates; Settings suspension clears the quote.
2. Generic message getters fetch excess fields and log source IDs on errors.
   Require a narrow parameterized owner query without those logging paths.
3. ChaChaNotes's normal connection timeout is 15 seconds, beyond the 5-second
   inspection budget. Require inspection-owned busy/progress/interrupt controls;
   actual worker capacity remains held until cleanup, including waiter cancellation.
4. Installed Rich strips only a small control subset. A native synthetic
   `Text` probe retained ESC and bidi Cf in all 50 supplied codepoints. This
   demonstrates retained input, not a terminal exploit. Explicit display-only
   escaping, backslash disambiguation and compositor controls are now required.
5. Proposal/authority expiry can occur without a writer event. Add final/use
   clock checks and a one-shot clearing timer, with no source auto-refresh.

Canonical binding metadata still needs retirement/disclosure controls even when
no quote is retained. The revised entry gates explicitly retain that prerequisite
and the unresolved device-only model/tool gap. These are runtime qualification
requirements, not a claim that any current gate is deployed.

The reviewer assessed the contract as ready for design acceptance with those
clarifications, not ready for runtime. Its bounded read-only confirmation pass
verified both findings resolved and found no remaining Critical or Important
issue. Root incorporated all material findings;
no deferred finding remains. No implementation-plan or new source resolver was
introduced. This acceptance applies to design direction only.

## Revised native validation

`python3 /private/tmp/check_memory_source_inspection_review.py` passed after
all verified findings were incorporated: 68 local links, 16 unique family IDs,
seven defined criteria, accepted design-only ADR/spec status, concrete resolved
requirements and qualification groups, previous owned task byte equality and
exact independent task/roadmap-suffix hashes. `git diff --check` passed. The
first revised run preceded tracker closeout (six checked criteria); the final
run qualifies all seven checked criteria and CLI Done/Implementation Notes.
No runtime path changed, and no application test pass is claimed. The native
Rich probe was synthetic and inspected retained codepoints only.

## Next checkpoint

Define the separately scoped local V2 containing-record admission and its
metadata retirement/disclosure boundary before a quotation implementation plan.
The source-inspection plan must include a real mounted caller and every
qualification group above. Native execution preference persists.
