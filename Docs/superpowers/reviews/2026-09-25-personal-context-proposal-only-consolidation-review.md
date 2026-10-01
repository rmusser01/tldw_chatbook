# Personal Context proposal-only consolidation review

Date: 2026-09-25
Task: TASK-25907.8
Status: Reviewed design accepted after requested follow-up

Design: [Consolidation contract](../specs/2026-09-25-personal-context-proposal-only-consolidation-design.md)
Decision: [Accepted design ADR-188](../../../backlog/decisions/188-opt-in-proposal-only-memory-consolidation.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)

## Deliverable and boundary

A design-only opt-in new-source workflow produces canonical Personal Context proposals through exact evidence, source enrollment, native disclosure and durable accounting/recovery. No fourth authoritative store, self-approval, goal mutation or background procedural writes. The first future slice uses Run now, a qualified owned on-device closed extraction adapter, new native user evidence and finite foreground batch review. Accepted designs do not enable runtime work.

The user approved TASK-25907.7's disclosure design. Its ADR-203, task and review now record accepted design direction, with a future-amendment link from ADR-102. No existing permissions, schema, provider routes or model calls changed.

## Native inspection and review findings

Native shell inspection confirmed the existing CREATE/UPDATE/ARCHIVE/PROMOTE operations, USER acceptance and exact current-version checks; five-per-root/25-per-session process-local proposal quotas; syncable CREATE defaults and body-less proposal Sync branches; and the separate durable automatic-work owners and foreground ADR-106 lesson boundaries. These facts were read in source, without provider execution or real-profile access.

A fresh read-only reviewer checked the proposed spec/ADR against all seven task criteria and those native contracts from base `55bce55109`. Its identified issues and written resolutions:

| Finding | Resolution |
| --- | --- |
| Important: candidate-slot numbers cannot recover only the unfinished work from a lost transient response after partial publication | Publish the entire validated batch of at most five proposals, exact operation receipt, dependency edges and durable counts in one all-or-none Personal Context owner transaction. Before COMMIT an uncertain attempt pauses with no partial proposals; after COMMIT recover the complete receipt without a model call. Foreground acceptance remains separate per item. No durable response cache was added. |
| Minor: correction wording could imply a temporal change_from relation | Use the exact ADR-201 correction relation/effect and supporting span; do not infer change_from or transition_at from chronology. |

Parent self-review corrected the ADR-106 link, made native GOAL kind exclusions explicit, added stable source-coverage identities across re-enrollment/catch-up/pipeline changes, and refreshed a stale historical roadmap status. Retired restricted coverage cannot permit replay; suppression or refused unknown coverage wins.

The reviewer performed a targeted reread of the fixes and confirmed no Critical, outstanding Important or actionable Minor finding remains. All seven criteria are covered as a written contract ready for user review. This verdict establishes no runtime qualification.

## Acceptance mapping

| Criterion | Written contract evidence |
| --- | --- |
| 1: Existing owners | Canonical proposals, native operational receipts/counters, separate foreground Notes/Agent Lessons; no Dreams/fact cache |
| 2: New revisions and lifecycle | Watermarked enrollment, qualified feed, bounded enumeration/queue, terminal checkpoint holes, stable coverage, whole-batch publication, uncertain restart/lock/cancel recovery |
| 3: Proposal authority | Supported CREATE, exact correction UPDATE, archival and reviewed duplicate guidance; no MERGE invention, acceptance, direct write, promotion or GOAL mutation |
| 4: Evidence/disclosure/suppression | Distinct consolidation purpose, pre-read/final-entry/post-response/COMMIT checks, current source and target authority, registered derivatives/cache/log recovery |
| 5: Opt-in and bounds | Per-run/rolling daily calls, tokens and publications; zero paid/network calls; independent currency fields; exact batch review; quiet stable outcomes and root evidence deduplication |
| 6: Evaluation | Separate at least 40 synthetic source families, frozen independent labels/splits, manual comparison, useful precision/recall, duplicate/rejection/cost reporting and native healthy/failing cases |
| 7: Release gate | Shipped verified evidence/forgetting/disclosure/source-feed/receipt/budget/recovery controls required; design-only completion never enables work |

## Verification and limits

This document records a contract review, not runtime readiness. Qualification cases and synthetic thresholds are future release requirements, not executed tests or reported model scores. No production file, worker, scheduler, schema, migration, real profile, provider, permission, deletion, push, PR or merge changed. At this initial checkpoint written user approval was pending; TASK-25907.8 stayed In Progress and ADR-188 Proposed. The follow-up acceptance below supersedes that status.

Native scoped documentation checks passed across 18 owned documents: 136 local Markdown links and 52 task reference/documentation paths resolve; all 59 original child criteria are unchanged, 11 family IDs are unique and original dependencies point backward. At this initial checkpoint tasks .5/.6/.7 were Done and .8 retained In Progress pending written approval. Whitespace and the documentation-only boundary pass, and independent evaluation task/roadmap bytes are unchanged. Allocation checked 559 available refs and 77 worktrees with no competing ADR-188. No fetch/open-PR allocation check occurred; its number remains provisional until integration. No runtime tests or full-suite run was warranted for this documentation-only checkpoint.

## User-requested follow-up review and acceptance

The user endorsed the written design and asked for an issues/improvements review before continuing. One bounded read-only follow-up examined enrollment, publication-budget ownership and unprocessable extraction against native source and ledger contracts. Three Important written-contract gaps were verified and fixed:

| Gap | Accepted correction |
| --- | --- |
| Cross-owner enrollment cannot have one implied atomic watermark transaction | Inactive immutable intent, source-owner watermark/authority receipt, matching coordinator acknowledgement and current-epoch activation. The source transaction defines the cutoff; ambiguous recovery never silently recaptures. |
| Daily publications were described as charged in both ledger and profile transactions | Automatic-work ledger reserves finite publication slots before the call; native ticket remains claimed through profile batch COMMIT. Profile receipt records the exact actual count and trusted publication timestamp; ledger consumes it once and releases only proven unused capacity. Unsettled capacity does not age out. |
| Oversize/malformed/truncated output had no exact checkpoint outcome | Named deferred extraction holds the checkpoint for exact foreground retry or skip; rejects the whole invalid batch, never repairs with another call or mislabels failure as no change. Known usage settles; unknown usage stays held. |

Self-review added trusted capture provenance (stored/imported role=user is insufficient), compact native-assembled candidate/evidence envelopes within unchanged caps, and kept old-history catch-up outside the first slice. A targeted reviewer reread confirmed all three fixes consistent, with no remaining contradiction. These are specification results, not shipped capability or model quality evidence.

Fresh follow-up native checks passed: 18 owned documents, 136 local Markdown links, 52 task paths, all 59 original child criteria unchanged, 11 unique family IDs, backward dependencies, documentation-only boundaries and whitespace. Independent evaluation additions remain byte-for-byte preserved. The local allocation scan covered 559 available refs and 78 worktrees; no fetch/open-PR check. TASK-25907.8 is Done and ADR-188 accepts design direction after this requested review. No worker, schema, model call, real profile, permission or deletion changed.
