# Personal Context versioned evidence and temporal changes review

Date: 2026-09-25
Task: TASK-25907.5
Status: Technical review complete; user approved written design, 2026-09-25

Design: [Versioned evidence and temporal changes](../specs/2026-09-25-personal-context-versioned-evidence-and-temporal-changes-design.md)
Decision: [Accepted design ADR-201](../../../backlog/decisions/201-versioned-profile-evidence-and-temporal-claims.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)

## Delivered design

A proposed V2 manifest/record/proposal contract keeps exact evidence, support,
user approval, confidence, salience and effective validity separate. Bindings
name a source authority, representation/version, codepoint span and digests;
imported IDs remain inert. V1 schemas and canonical bytes remain unchanged,
with unknown legacy metadata and an explicit capability/activation gate.
Correction, real change, supersession and workspace exceptions carry distinct
reviewed effects. The document adds dated transitions and a Unicode span
vector, retention/disclosure rules and synthetic future acceptance examples.
No production model, schema, database, tool, Sync behavior or fixture changed.

## Independent review and disposition

The initial fresh read-only review inspected draft range `2f2495a024..8fa9c5a151`
against all six criteria and ADR-102/182/024. It found no Critical issue, three
Important decisions and two Minor clarity gaps. A targeted reread of the amended
passages confirmed all five findings addressed at the design level, with no
remaining Important contradiction. This is readiness for user review, not an
implementation or server-conformance certificate.

| Finding | Final design disposition |
| --- | --- |
| Important: inline binding metadata survived source-policy revocation | Immediate use/transport fence; sanitized successor; retirement of restricted history/outbox/recovery/Undo/caches; evidence-retirement epoch and peer acknowledgement; stale replay and restore barriers. Approved text can survive, but no fabricated approval or automatic inference upgrade. |
| Important: an older client could inject a V1 global claim while ignoring its V2 exception | Profile-wide V2 manifest/required semantics; active consumers must upgrade or be explicitly disabled before cutover. Unknown required semantics fence the entire profile, including V1 claims. Opaque retention alone is insufficient. |
| Important: closing a prior interval had no immutable representation | Canonical typed edge effect and explicit transition instant; atomic admission compares target heads; historical projection changes without rewriting prior bytes. Stale admission differs from a later accepted successor. |
| Minor: assessment might transfer to changed evidence with unchanged wording | Assessments bind both claim digest and complete binding digest; source/version/span substitution requires reassessment. |
| Minor: examples lacked fully concrete inputs | Exact Unicode codepoint/UTF-8/digest vector and dated alternative change/correction histories. |

Parent self-review also named required support/known-validity conditions,
held workspace candidates after a global-head change, rejected imported labels
as local intent, bounded retirement receipt pages, and made the example's UTC
boundary explicitly user-confirmed. It also distinguished new/modified relation
admission from preserving an already accepted unchanged historical edge. It states that structured contradiction
holds do not promise detection of every contradiction in free text.

## Acceptance mapping

| Criterion | Reviewable evidence |
| --- | --- |
| 1: Authority-bound sources/version/spans and access | Shared-core contract, exact source binding and evidence inspection sections; Unicode and wrong-authority examples |
| 2: Separate presence/support/approval/confidence/salience/validity | Data concepts, meaning table, eligibility rules and quoted/attached-material examples |
| 3: Corrections/supersession/exceptions without newest-wins | Canonical effect table, atomic admission rules, dated alternative histories and conflict examples |
| 4: Truthful V1, migration, schemas/fixtures/server obligations | V1 migration/compatibility section, profile-wide activation fence and required conformance families |
| 5: Retention/disclosure and edited/missing/inaccessible/conflicting cases | Derivatives and privacy retirement sections, source replacement/revocation/offline examples |
| 6: Reviewed design only | Independent review above and scoped documentation-only boundary check |

## Verification and practical limits

Scoped checks passed: ten family task files retain the same 59 child criteria
and backward dependencies; the task-ID guard finds no duplicate/invalid family
paths. All 157 local Markdown links across the six owned documentation/tracker
files resolve. No TODO/TBD/FIXME markers remain. The concrete Unicode codepoint
span, exact UTF-8 bytes and both SHA-256 values were independently recomputed;
the temporal edge JSON parses. The owned diff is documentation/tracker-only and
`git diff --check` passes. No runtime test or canonical V2 conformance result is
implied by these checks.

No runtime tests or full suite were run for this documentation-only task.
No real profile, source body or model provider was accessed. The reviewer did
not judge runtime correctness, actual deletion guarantees, server conformance,
production migration success or live UI behavior. Synthetic examples are
future regression requirements, not passing runtime tests. Source-policy
retirement cannot promise immediate remote offline erasure or deletion of
unmanaged exports/backups; outstanding peer acknowledgements remain visible.

After presentation of the reviewed contract, the user requested continuation
on 2026-09-25. ADR-201 accepts this design direction and TASK-25907.5 is Done.
No schema/runtime rollout is approved by completing a design-only task.
The native executor has made no implementation plan for V2, no push, PR or merge.
The final offline allocation check found no competing ADR-201 path across
584 local refs and 83 registered worktrees. No network fetch or open-PR scan
was performed; the number stays provisional until integration-time checks
against then-current branches and open PRs.
