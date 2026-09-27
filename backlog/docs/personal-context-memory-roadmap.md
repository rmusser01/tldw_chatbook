# Personal Context memory roadmap and tracker

Updated: 2026-09-27
Status: Closed native V2 barrier complete and verified; V2 activation, retirement/disclosure and server rollout remain gated
Foundation: [TASK-25907](../tasks/task-25907%20-%20Cross-session-persistent-memory-for-the-agent.md) — Done, @codex

- [Written design](../../Docs/superpowers/specs/2026-09-25-personal-context-memory-evolution-design.md)
- [Accepted ADR-182](../decisions/182-personal-context-memory-evolution.md)
- [Pre-implementation review](../../Docs/superpowers/reviews/2026-09-25-personal-context-memory-preimplementation-review.md)
- [Baseline execution plan](../../Docs/superpowers/plans/2026-09-25-personal-context-memory-baseline.md)
- [Baseline execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-memory-baseline-execution-review.md)
- [Provenance execution plan](../../Docs/superpowers/plans/2026-09-25-personal-context-provenance.md) — complete; native execution retained
- [Provenance execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-provenance-execution-review.md)
- [Next Send selection execution plan](../../Docs/superpowers/plans/2026-09-25-personal-context-next-send-selection.md) — complete; native execution retained
- [Next Send selection execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-next-send-selection.md)
- [Local retrieval execution plan](../../Docs/superpowers/plans/2026-09-25-personal-context-local-retrieval.md) — complete; native execution retained
- [Local retrieval execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-local-retrieval-execution-review.md)
- [Versioned evidence and temporal design](../../Docs/superpowers/specs/2026-09-25-personal-context-versioned-evidence-and-temporal-changes-design.md) — accepted design; no runtime change
- [Accepted design ADR-185](../decisions/185-versioned-profile-evidence-and-temporal-claims.md)
- [Dependency-aware forgetting design](../../Docs/superpowers/specs/2026-09-25-personal-context-dependency-aware-forgetting-design.md) — accepted design; no deletion performed
- [Accepted design ADR-186](../decisions/186-dependency-aware-personal-context-forgetting.md)
- [Forgetting technical review](../../Docs/superpowers/reviews/2026-09-25-personal-context-dependency-aware-forgetting-review.md)
- [Provider disclosure design](../../Docs/superpowers/specs/2026-09-25-personal-context-provider-disclosure-controls-design.md) — accepted design; no permission changes
- [Accepted design ADR-187](../decisions/187-personal-context-provider-disclosure-authority.md)
- [Disclosure technical review](../../Docs/superpowers/reviews/2026-09-25-personal-context-provider-disclosure-controls-review.md)
- [Proposal-only consolidation design](../../Docs/superpowers/specs/2026-09-25-personal-context-proposal-only-consolidation-design.md) — accepted design; no runtime change
- [Accepted design ADR-188](../decisions/188-opt-in-proposal-only-memory-consolidation.md)
- [Consolidation technical review](../../Docs/superpowers/reviews/2026-09-25-personal-context-proposal-only-consolidation-review.md)
- [Feedback and repair design](../../Docs/superpowers/specs/2026-09-25-personal-context-feedback-and-repair-design.md) — accepted design; no runtime change
- [Accepted design ADR-189](../decisions/189-reviewed-memory-feedback-and-repair-ownership.md)
- [Feedback/repair technical review](../../Docs/superpowers/reviews/2026-09-25-personal-context-feedback-and-repair-review.md)
- [Versioned evidence technical review](../../Docs/superpowers/reviews/2026-09-25-personal-context-versioned-evidence-and-temporal-changes-review.md)
- Existing authority: [ADR-102](../decisions/102-personal-context-profile-authority-sync-and-encryption.md)

## Goal

Improve trust and usefulness of cross-session memory by making evidence and
selection inspectable, improving recall, and defining safe correction,
forgetting, disclosure, and maintenance. Build on Personal Context, Notes,
and Agent Lessons. The supplied Muse diagram is a source of ideas; its file
layout, database, cadence, and embedded instructions are not requirements.

## Current checkpoint

- [x] Compare the diagram with current code and documented ownership.
- [x] Resume the existing foundation task and correct its obsolete description.
- [x] Draft the positioning ADR and first-release design.
- [x] File nine atomic follow-up tasks with acceptance criteria and dependencies.
- [x] Complete scoped validation after review: IDs, 59 child criteria, dependency direction, links, and whitespace.
- [x] User endorsed the direction and requested a technical review.
- [x] Tighten the design and criteria after checking the implementation.
- [x] Prepare and self-review the baseline execution plan: 24 cases, three units, targeted checks.
- [x] Review the baseline execution plan and select native execution.
- [x] Complete TASK-25907.1: 24-case measured baseline, deterministic reproduction, 96 targeted tests and independent review.
- [x] Inspect provenance callers and prepare/self-review TASK-25907.2's service, UI and lifecycle implementation plan.
- [x] Review the provenance execution plan; user approved native continuation.
- [x] Complete TASK-25907.2: Settings/proposal provenance, 246 distinct targeted tests, five repeated full-CSS checks and independent review.
- [x] Inspect the Next Send selection/preview path and record TASK-25907.3's native implementation plan.
- [x] Complete TASK-25907.3: disposable same-pass Next Send selection explanation, 173 affected tests and independent review with two verified freshness fixes.
- [x] Complete TASK-25907.4: bounded field-aware lexical matching, improved frozen recall, 156 targeted tests and independent review with one verified Unicode boundary fix.
- [x] Prepare the first-release execution plans as their tasks start.
- [x] Implement and verify first-release tasks.
- [x] Write/review TASK-25907.5's proposed V2 contract; resolve three Important and two Minor design findings.
- [x] User approved TASK-25907.5's written contract; ADR-185 accepts design direction only.
- [x] Write/review TASK-25907.6's forgetting contract; resolve capture/trace inventory and publication-cancellation findings.
- [x] User endorsed TASK-25907.6 subject to review; audit/log and queued-dispatch gaps resolved; ADR-186 accepts design direction only.
- [x] Write/review TASK-25907.7; resolve model-cache state and indirect tool/publication egress findings.
- [x] User explicitly approved TASK-25907.7; ADR-187 accepts design direction only and current permissions are unchanged.
- [x] Inspect native proposal/accounting owners and write TASK-25907.8’s bounded new-source consolidation contract.
- [x] Review TASK-25907.8; resolve whole-batch crash-recovery and correction-relation findings; scoped document/tracker checks pass.
- [x] User endorsed TASK-25907.8 subject to follow-up review; resolve enrollment, quota-owner and malformed-output gaps; ADR-188 accepts design direction only.
- [x] Inspect feedback/Notes/Companion/Dreams owners and draft TASK-25907.9’s foreground repair contract.
- [x] Review TASK-25907.9; separate issue resolution from linked change approval; native scoped checks and targeted reviewer confirmation pass.
- [x] User explicitly approved TASK-25907.9’s written contract; ADR-189 accepts design direction only.
- [x] Complete all nine original tasks while retaining implementation/design-only boundaries.

This is a program plan and status index. Task files are the source of truth
for status and acceptance criteria. Design-only tasks do not represent shipped
features. The completed baseline adds synthetic evaluation code and documentation.
The completed provenance slice adds read-only metadata inspection in My Profile
and proposal review. Source ownership, provider grants and permanent background
jobs remain unchanged.

## Task tracker

| Task | Outcome | Delivery type | Status | Prerequisites |
| --- | --- | --- | --- | --- |
| [TASK-25907.1](../tasks/task-25907.1%20-%20Establish-a-synthetic-Personal-Context-memory-evaluation-baseline.md) | Establish a synthetic Personal Context memory evaluation baseline | First release | Done — 96 targeted tests, reproducible evidence | TASK-25907 |
| [TASK-25907.2](../tasks/task-25907.2%20-%20Explain-existing-Personal-Context-provenance.md) | Explain existing Personal Context provenance | First release | Done — 246 distinct targeted tests; one deferred minor recovery issue | TASK-25907 |
| [TASK-25907.3](../tasks/task-25907.3%20-%20Explain-Personal-Context-selection-in-Next-Send.md) | Explain Personal Context selection in Next Send | First release | Done — 173 affected tests; stale-publication races fixed | TASK-25907 |
| [TASK-25907.4](../tasks/task-25907.4%20-%20Improve-local-Personal-Context-retrieval-relevance.md) | Improve local Personal Context retrieval relevance | First release | Done — 156 targeted tests; one deferred ranking-test gap | TASK-25907, TASK-25907.1 |
| [TASK-25907.5](../tasks/task-25907.5%20-%20Design-versioned-Personal-Context-evidence-and-temporal-changes.md) | Design versioned Personal Context evidence and temporal changes | Design only | Done — reviewed design accepted; no runtime change | TASK-25907 |
| [TASK-25907.6](../tasks/task-25907.6%20-%20Design-dependency-aware-Personal-Context-forgetting.md) | Design dependency-aware Personal Context forgetting | Design only | Done — reviewed design accepted; no deletion or runtime change | TASK-25907, TASK-25907.5 |
| [TASK-25907.7](../tasks/task-25907.7%20-%20Design-Personal-Context-provider-disclosure-controls.md) | Design Personal Context provider disclosure controls | Design only | Done — reviewed design explicitly accepted; no permission change | TASK-25907 |
| [TASK-25907.8](../tasks/task-25907.8%20-%20Design-opt-in-proposal-only-memory-consolidation.md) | Design opt-in proposal-only memory consolidation | Design only | Done — accepted design after follow-up review; no runtime change | TASK-25907, TASK-25907.1, TASK-25907.5, TASK-25907.6, TASK-25907.7 |
| [TASK-25907.9](../tasks/task-25907.9%20-%20Design-reviewed-memory-feedback-and-repair-tracking.md) | Design reviewed memory feedback and repair tracking | Design only | Done — reviewed design explicitly accepted; no runtime change | TASK-25907, TASK-25907.5 |

## Execution sequence

### First release: inspect and retrieve existing memory

1. Establish the synthetic baseline through real service, tool, and snapshot
   entry points. Freeze development and held-out cases; distinguish authorization,
   relevance, and expected context packing. Record current misses honestly.
2. Add a read-only provenance view to My Profile and proposal review. Existing
   references remain labelled as unverified legacy metadata; no source quote
   is fabricated or resolved across an unknown authority. Missing edit/inference
   history stays unknown even when the stored reason mentions user approval.
3. Explain selected records and eligible override/budget omissions in the
   disposable Next Send preview using its actual selection pass. Reject stale
   owner/request results and invalidate on expiry as well as revision changes.
4. Improve field-aware lexical matching and compare results with the baseline.
   Preserve authorization, hard priorities, workspace overrides, and budgets.

Each item is independently reviewable and testable. These changes use existing
canonical fields; they require no shared schema migration, external memory
backend, model call, or persistent search index.

### Later design work: strengthen memory guarantees

- **Evidence and temporal change:** bind evidence to source authority/version;
  distinguish explicit statements, inference, approval, correction, and scoped
  exception; define migration and shared-core compatibility.
- **Forgetting:** inventory every owner and derivative; separate Undo from
  immediate forgetting, prevent re-extraction, and specify crash recovery and
  honest remote acknowledgement.
- **Provider disclosure:** separate syncability from permission to disclose to
  local or remote model providers across all request paths.
- **Consolidation:** after those contracts, design an opt-in, budgeted process
  that works only on new eligible evidence and produces reviewable proposals.
- **Feedback and repair:** retain explicit, scoped corrections and resolution
  evidence through Personal Context and Agent Lessons, without speculative
  psychological profiling or a competing Dreams subsystem.

Later designs must identify a bounded implementation release before code work
is filed. Their current deliverable is the reviewed contract, not a promise
that the full capability will ship in one PR.
Consolidation cannot be enabled just because those designs are complete: its
implementation needs shipped and verified evidence, forgetting/suppression,
disclosure, and recovery controls.

## Decision register

| Decision | Proposed answer | Review state |
| --- | --- | --- |
| Durable ownership | Preserve Personal Context, Notes, and Agent Lessons; no fourth store | Existing owner direction retained |
| First-release evidence | Inspect existing metadata; label unresolved references and edit history honestly | Accepted in ADR-182 |
| First-release recall | Local deterministic lexical ranking; measure relevance independently of eligibility | Accepted in ADR-182 |
| First-release context | Preserve the 12 KiB / ten-percent budgets and existing priority groups | Existing behavior retained |
| Rich source quotations | Authority-bound exact evidence and compatibility | Accepted design ADR-185; unimplemented |
| Cross-owner forgetting and provider disclosure | Native lifecycles and independent destination/purpose controls | Accepted design ADR-186/187; unimplemented |
| Background consolidation | Opt-in, new-signal-driven, bounded, and proposal-only | Accepted design ADR-188; unimplemented |
| Reflection/repair | Approved facts and verified procedures; issue resolution separate from change approval | Accepted design ADR-189; unimplemented |

## Acceptance evidence

Use synthetic data and targeted runs only. A full test sweep requires explicit
user opt-in. Every implementation task includes its own regression evidence;
no aggregate completion claim substitutes for a working production caller.

| Area | Evidence needed |
| --- | --- |
| Recall | Frozen development/held-out relevant-record labels; production calls; precision/recall@K, reciprocal rank, empty-result false positives; separate context-selection expectations |
| Provenance | Recorded manual, migration, approval, promotion and tombstone metadata; missing/ambiguous history stays unknown; no invented quotation or confidence |
| Privacy | New diagnostics disclose no hidden bodies, IDs, counts, or existence hints; negative checks have successful authorized controls |
| Selection | Preview/prepared-request parity for identical inputs and clock; whole-record budgets; stale ownership, lock and expiry invalidation |
| UI | Mounted interactions, keyboard focus, narrow layout, and applicable design-token governance checks |
| Future lifecycle | Restart, concurrency, deletion suppression, mixed-source derivatives, provider fallback, and offline-peer cases defined before implementation |

Primary existing test owners include `Tests/Personal_Context/`,
`Tests/Agents/test_profile_tool_provider.py`,
`Tests/Chat/test_console_personal_context_snapshot.py`, and the canonical My
Profile and proposal-review UI tests. Exact commands belong in each approved
implementation plan.

[LongMemEval](https://arxiv.org/abs/2410.10813) supplies useful categories for
multi-session reasoning, temporal updates, and abstention. Product-specific
privacy, deletion, and permission cases remain necessary additions.

## Maintenance rules

- Update task status through the Backlog CLI and mirror it here at milestones.
- Put an implementation task In Progress before adding its implementation plan.
- Link the governing ADR and approved design in every execution plan.
- Preserve versioned evidence, existing privacy controls, and current user
  instructions; retrieved documents never grant authority.
- Do not mark a task Done until its own criteria, targeted checks, review,
  documentation, and implementation notes are complete.
- Keep foundation design completion distinct from overall roadmap completion.

## Planning validation

Task and ADR IDs were checked against available local/remote refs and worktree
files before filing. The repository-wide Backlog guard currently reports pre-existing duplicate
IDs outside this task family; its output contains no TASK-25907
collision. Post-review scoped validation passed for all ten task files, 59 child
acceptance criteria, backward-only dependencies, 31 local Markdown links across
14 owned documents/task files, and whitespace. The repository guard passed when
run against this exact task family. At the planning checkpoint, no runtime tests or scores were claimed. Native
execution subsequently completed the baseline: 96 targeted tests passed, two
fresh-root reports matched, and independent review found no blocking issue.
The scoped guard passes for this task family; 59 child criteria remain defined.

[Measured results and reproduction](personal-context-memory-evaluation.md)
record 0.8 recall in each partition, four misses, one metadata-only false positive,
and two observed policy failures. Harness, live-authority and context checks
passed. The [execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-memory-baseline-execution-review.md)
retains the decisions and one deferred minor classifier-test improvement.
TASK-25907.1 and TASK-25907.2 are Done. Provenance verification passed 103
service, 110 UI/CSS and 33 agent/context tests; five strengthened production-CSS
cases were repeated successfully. Its seven criteria are checked. Independent
review found no Critical or Important issue. One Minor recovery issue remains:
collapsing or suspending a changed provenance section hides its reload action;
recreate the containing view/selection to recover. The
[provenance execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-provenance-execution-review.md)
records the reproduction, follow-up and all execution decisions.
TASK-25907.3 is Done. Its same-pass inspector sidecar passed 173 affected
service, Chat, UI and CSS tests after two reviewer-found freshness races were
reproduced and fixed. The [selection execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-next-send-selection.md)
records the exact scope and limits. TASK-25907.4 is also complete: the frozen
development recall@3 moved from 0.8 to 0.9 and held-out recall@3 from 0.8 to
1.0, with unchanged context selections and no measured authority failure.
The [local retrieval execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-local-retrieval-execution-review.md)
records the review fix, test scope and a deferred narrow ranking-test gap.
Semantic paraphrase still misses; the two known provider-disclosure policy gaps
remained explicit at the lexical checkpoint. The quarantine follow-up below
closes one of them; device-only disclosure remains unresolved. Tasks .5/.6/.7 now have accepted evidence, forgetting and
disclosure designs, while .8 now has an accepted consolidation contract after follow-up review
and .9 has an accepted reviewed feedback/repair contract. These design checkpoints change no production schema,
source access, provider permission or runtime behavior.

The disclosure design is accepted; consolidation has an accepted reviewed contract after three follow-up lifecycle fixes. Native source/document checks do not qualify the future runtime safeguards. Independent generated-answer evaluation additions remain outside this checkpoint.

## Known limits carried into implementation planning

- The stored provenance cannot prove the current wording was directly stated,
  accepted unchanged, or independently verified.
- ADR-102's broad `device_only` promise exceeds observed model/tool filtering.
  Provider-disclosure design must resolve that mismatch explicitly.
- TASK-25907.11 removes the unscoped quarantine hint from model serialization.
  Internal authority revisions still track quarantine; this does not establish
  timing invariance or broader destination/purpose disclosure enforcement.
- Existing profile selection reads a repository snapshot before scope filtering.
  Local matching does not eliminate that cost or provide semantic retrieval.
- No generated-answer-quality, automatic consolidation, or end-to-end forgetting
  guarantee is delivered by the first release.

## Completed follow-up: remove the unscoped quarantine signal

[TASK-25907.11](../tasks/task-25907.11%20-%20Remove-unscoped-quarantine-signals-from-model-context.md)
is **Done** after the user approved its
[native implementation plan](../../Docs/superpowers/plans/2026-09-25-personal-context-quarantine-signal-removal.md).
Unsupported quarantine state no longer changes model content, packing, token
estimates or permitted selection rows, and cannot create an empty-record block.

The [execution review](../../Docs/superpowers/reviews/2026-09-25-personal-context-quarantine-signal-removal-review.md)
records 175 distinct targeted passes, a strengthened empty-state control,
scoped static checks and one resolved Minor review finding. The separate
[quarantine report](../../Docs/superpowers/reviews/evidence/personal-context-memory/quarantine-signal-v1.json)
reproduced byte-for-byte on fresh roots. Retrieval summaries and all case
selections are unchanged; only h11's quarantine observation and context size
change. The device-only d10 gap and broader V2 controls remain unimplemented.

Private quarantine maintenance and native authority revisions remain unchanged.
No V1 migration, grant, provider execution, real-profile access, Sync change,
UI/app launch or memory background job was added. Original evidence and the
independent evaluation addition are preserved; the branch stays local.

## Completed foundation: exact text-span identity

[TASK-25907.12](../tasks/task-25907.12%20-%20Add-exact-text-span-digests-to-shared-profile-core.md)
is **Done** after the user endorsed its
[native implementation plan](../../Docs/superpowers/plans/2026-09-26-personal-context-exact-span-digests.md)
subject to review. The pure shared-core function returns immutable named
SHA-256 identities for exact UTF-8 source text and a half-open codepoint span.

Plan review tightened the API to built-in str/int types and meaningful
negative-control RED checks. The unvalidated calculation produced 14 failed
controls and 16 passes before validation; the final affected run passed 181
unique cases (30 new, 151 existing V1/shared-core). Whole-file Ruff/format,
native import ownership and the documented example passed. The
[execution review](../../Docs/superpowers/reviews/2026-09-26-personal-context-exact-span-digests-review.md)
records review attribution and the fixed pytest custom-value naming trap;
independent implementation review found no remaining findings.

[API documentation](personal-context-exact-span-digests.md) explains strict
input/error semantics, exact normalization/line-ending behavior, empty spans,
source/span identity separation and caller-owned resource limits. A digest
pair does not establish source access, current version, support or approval.
V1 consumers/exports, canonical schemas/fixtures, permissions, grants and
historical evaluation evidence remain unchanged. Source resolution, complete
V2 bindings, temporal admission, retirement/forgetting, disclosure and server
qualification remain future work; the device-only disclosure gap remains.

## Native source-readiness checkpoint

[TASK-25907.13](../tasks/task-25907.13%20-%20Audit-native-source-readiness-for-exact-memory-evidence.md)
is **Done** after auditing the conversation, Notes, sync-log and citation owners before adapter work.
The [source-readiness report](personal-context-source-readiness-audit.md) distinguishes
stable semantic revision identity from conditional sanitized historical text,
mutable Notes heads and governed message-owned citation snapshots.

The native selection passed 132 distinct checks. Two stale raw-deletion fixtures
now respect the semantic guard: tracked message deletion uses its coordinator,
a real migrated pre-ledger row supplies legacy FK-cascade coverage, and an
additional tracked-cascade test proves rejection and rollback. Runtime guards,
source retention, schema, V1 bytes and grants are unchanged.

The recommended next slice is a read-only current-message adapter with trusted
owner authorization, exact representation/version/span checks and publication
fencing. Historical snapshots, V2 integration, forgetting, disclosure and
consolidation remain separate gated work. No source adapter is shipped by this
audit. Independent review found no major issue; its Minor reproducibility finding
was resolved. All eight criteria and scoped tracker/document checks are complete.

## Completed foundation: owner-version binding identity

[TASK-25907.14](../tasks/task-25907.14%20-%20Add-bounded-owner-version-evidence-bindings-to-shared-profile-core.md)
is **Done** after the user endorsed the component subject to review and native
execution. The [reviewed specification](../../Docs/superpowers/specs/2026-09-26-personal-context-owner-version-evidence-binding-design.md)
and [native plan](../../Docs/superpowers/plans/2026-09-26-personal-context-owner-version-evidence-binding.md)
produce one explicit shared-core model with 18 required fields and a complete
binding digest. Digest admission validates a detached full field snapshot;
unknown or malformed unchecked copies cannot bypass it.

Requested design review caught the fixture packaging omission. Both synthetic
fixture copies and the component are now verified from an actual offline-built
wheel, with no installation or dependency/version change. The final affected
native run passed 339 distinct cases (158 new, 181 existing); whole-file Ruff
and format passed. [API documentation](personal-context-owner-version-evidence-binding.md)
explains strict scalars, Unicode/span/time semantics, unknown-key rejection,
unsafe-copy revalidation and the raw-JSON/structured-error caveats.

The [execution review](../../Docs/superpowers/reviews/2026-09-26-personal-context-owner-version-evidence-binding-review.md)
records RED/GREEN receipts and one bounded independent review with no findings.
The reviewer separately verified fixed digests and malformed-copy rejection;
it did not rerun the root's full targeted selection. All eight criteria and
scoped documentation/tracker checks are complete; ADR-185 applies directly.

This component provides data identity, never source access, support, approval
or guaranteed historical text. Current V1 core/fixture/schema/consumer bytes,
grants and source retention are unchanged. A future current-message adapter
still needs trusted source-owner authorization and publication fencing. Full
V2 admission, server qualification, forgetting/disclosure controls, consolidation
and the device-only gap remain separate work. The branch/worktree stay local;
independent generated-answer evaluation is preserved.

## Accepted interface design: foreground source inspection

[TASK-25907.15](../tasks/task-25907.15%20-%20Design-foreground-Personal-Context-source-inspection-authority.md)
is complete as a **design-only** task after user endorsement and follow-up review. Its [specification](../../Docs/superpowers/specs/2026-09-26-personal-context-foreground-source-inspection-design.md),
[Accepted design ADR-191](../decisions/191-foreground-personal-context-source-inspection-authority.md)
and [technical review and corrections](../../Docs/superpowers/reviews/2026-09-26-personal-context-foreground-source-inspection-review.md)
define a trusted foreground action for an independently authorized open local
conversation. Profile grants and imported/legacy IDs do not grant source reads.
Exact matching remains distinct from origin, support, approval and continuing
truth. Quote access needs real V2 containing-record admission, bounded reads,
publication/revocation fencing and a mounted native caller before it can ship.
Review resolved fresh-snapshot, Settings/Console ownership, worker-drain,
expiry, narrow-read and terminal-control gaps. A device-only flag alone does not
qualify binding metadata disclosure or retirement. No resolver or UI change is
deployed; local V2 containing-record admission is the next prerequisite.

## Completed native correction: device-only canonical agent reads

[TASK-25907.16](../tasks/task-25907.16%20-%20Exclude-device-only-records-from-current-agent-profile-reads.md)
implements the approved conservative deny for device-only canonical records at
agent view/context/search/get eligibility and agent target mutation/proposal
boundaries. Hidden workspace overrides, relevance and budgets cannot change
permitted selection or add omission hints. Manual owner inspection/management,
V1 storage/schema, Sync controls, permissions and grants stay unchanged.
[ADR-102](../decisions/102-personal-context-profile-authority-sync-and-encryption.md)
and [ADR-187](../decisions/187-personal-context-provider-disclosure-authority.md)
govern this bounded privacy correction; no new ADR/schema or grant was added.

The [execution review and native evidence](../../Docs/superpowers/reviews/2026-09-26-personal-context-device-only-read-review.md)
records 278 distinct targeted cases, clean affected-file Ruff/format,
expected RED failures, successful syncable/manual controls and independent
read-only review with no actionable findings. Old successful agent fixtures now
use syncable controls; a baseline shortest-candidate defect was fixed only for
positive probe selection. Frozen labels and measured scoring remain unchanged.
No full suite, real profile/keyring, provider, server or network was used.

The [native before report](../../Docs/superpowers/reviews/evidence/personal-context-memory/device-only-preflight-v1.json)
and [native after report](../../Docs/superpowers/reviews/evidence/personal-context-memory/device-only-v1.json)
retain the byte-identical 24-case fixture. Only d10 changes: device-only is absent
from search/context, while historical selection mismatch stays visible and raw
recall declines from 1.0 to 0.5. Synthetic disclosure checks passing do not
qualify full application disclosure or generated-answer effectiveness.

This covers current canonical agent reads and target mutations. Previously
disclosed history/caches, queued payloads, derivatives, qualified on-device
enrollment, V2 admission, source inspection and full metadata retirement/disclosure
remain separate work. Independent TASK-25907.10 task and roadmap suffix remain
byte-for-byte preserved and unstaged. The branch/worktree stay local.

## Completed readiness audit: native V2 admission and cutover

[TASK-25907.17](../tasks/task-25907.17%20-%20Audit-native-V2-memory-admission-and-cutover-readiness.md)
records the [source-backed readiness matrix and qualification sequence](personal-context-v2-admission-readiness-audit.md).
Native inspection confirms distinct V1 canonical models, strict export/context
snapshots, integer schema attention and existing local retirement/Undo controls.
SQLite schema 8 and Sync V2 transport do not supply Profile V2 semantics.
Exact-span/binding components provide identity, not containing-claim admission
or source access. Per-object quarantine is distinct from the strict snapshot
failure path and from profile-wide consumer compatibility.

All 36 existing targeted native checks passed, including structural/semantic
fixtures, repository/boot rejection, mocked client compatibility attention,
positive V1 reconciliation, encrypted recovery, tombstones and bounded Undo.
The audit's future qualification matrix covers activation, atomic admission,
legacy migration, binding-metadata retirement, disclosure and foreground
publication races. It does not claim any V2 gate was tested or deployed.

The recommended next deliverable is a reviewed concrete V2 canonical data
contract with fixed digest/semantic/byte fixtures, kept inactive until native
admission, consumer retirement, metadata privacy and required companion-server
conformance qualify. A local-only flag or resolver over legacy IDs cannot waive
those gates. ADR-102, ADR-185, ADR-186, ADR-187 and ADR-191 apply; no new policy,
schema, storage, grant, migration, resolver, provider or UI change was made.
Scoped self-review, links/source anchors and unchanged prior task/runtime/core/
fixture/independent follow-up bytes passed. Native branch/worktree retained;
no full sweep, real profile/keyring/provider/server, push, PR, merge or fetch.

## Accepted design: concrete V2 canonical contract

[TASK-25907.18](../tasks/task-25907.18%20-%20Specify-the-inactive-V2-canonical-memory-data-contract.md)
is complete as a design-only task after explicit user approval of the [written V2 contract](../../Docs/superpowers/specs/2026-09-26-personal-context-v2-canonical-data-contract-design.md)
and [Accepted design ADR-192](../decisions/192-personal-context-v2-canonical-data-contract.md).
The draft specifies manifest/record/proposal fields, strict/default-deny policy,
claim meaning and complete binding digests, version-bound approval, support,
validity/relations, sanitized privacy holds and future conformance cases.
V1 schemas/payloads and the published 18-field binding stay unchanged.
Identity/scope enter the meaning digest to prevent transplanted attribution;
audience-purpose pairs avoid accidental disclosure widening. The first evidence
form excludes excerpts, Notes and captured representations until separately
versioned components qualify.

Only documentation is delivered. Native checks cover the illustrative meaning
digest, links and byte/identity preservation; they do not qualify V2 runtime.
The recommended next implementation unit, after written review and planning,
is an inactive shared-core library with schemas and independent fixed fixtures.
Native admission/migration, consumer retirement, metadata privacy, disclosure,
source inspection and server conformance remain separately reviewed gates.
TASK-25907.18 is Done and ADR-192 accepts design direction only. Independent TASK-25907.10
task and roadmap suffix remain unchanged. No provider or source access, real
profile/keyring, full suite, remote refresh, push, PR or merge is authorized here.

## First V2 implementation: inactive claim meaning

[TASK-25907.19](../tasks/task-25907.19%20-%20Implement-the-inactive-V2-claim-meaning-component.md)
delivers typed validity/relations and the exact complete meaning projection/digest
through an explicit inactive shared-core submodule. The [component notes](personal-context-v2-claim-meaning.md)
record fixed conformance resources, native RED/GREEN and an actual offline wheel
installation. The [implementation plan](../../Docs/superpowers/plans/2026-09-26-personal-context-v2-claim-meaning.md)
implements accepted ADR-192 directly. All six software criteria are met;
independent review found two boundary gaps, fixed through observed RED/GREEN
regressions. Final targeted verification passes 418 cases plus affected-file
static checks. TASK-25907.19 is Done for this inactive component only;
no runtime consumer or default V1 API uses it.

V1 dispatch/schemas/fixtures and the existing complete binding stay unchanged.
Full V2 aggregates/dialect, native admission/migration, retirement/disclosure
controls, source inspection and server conformance remain separate units.
Native inline execution preference persists; no new mode approval is required.
Independent .10 task/suffix remain untouched.

## Completed inactive V2 canonical aggregates

[TASK-25907.20](../tasks/task-25907.20%20-%20Implement-inactive-V2-canonical-profile-aggregates.md)
delivers all three explicit V2 aggregates, strict attribution/disclosure/lifecycle
shapes, fresh validated canonical/hash/integrity helpers, duplicate-aware JSON,
separate structural/required-semantic schemas and fixed full aggregate fixtures.
The [component notes](personal-context-v2-aggregates.md) and
[native implementation plan](../../Docs/superpowers/plans/2026-09-27-personal-context-v2-aggregates.md)
record the data-only boundary under accepted ADR-192. Root/V1 dispatch, scopes,
payloads and published binding/meaning components remain unchanged; Python>=3.12.

A fresh independent review found three Important boundary problems: per-call
strict/extra overrides could coerce/drop forbidden input, and the dynamic dialect
required root declarations in460 nested subschemas. Observed RED/GREEN fixes
check contextual exact shapes before parsing (including absent payload defaults)
and correctly scope the dialect document requirement. Final728 targeted library
cases pass with affected py312 static checks, actual offline wheel target install,
full offline dialect validation and independent fixed byte/hash/HMAC oracles.
No Critical/Minor findings, no second review; all6737 prior core/application/test
bytes and independent .10 work are preserved except owned packaging declarations.

Native admission/migration and all-consumer qualification or retirement,
metadata retirement/forgetting, destination/purpose disclosure, source inspection,
and required companion-server conformance remain separate gates before V2 use.
Consolidation, feedback/repair and generated-answer effectiveness remain future
work. This completes the inactive shared profile library contract only; no native
schema/storage/Sync/grant/UI/source/provider/server activation, full app sweep,
real profile/keyring, network, push, PR or merge. Native branch/worktree retained.

## Accepted native V2 compatibility and admission

[TASK-25907.21](../tasks/task-25907.21%20-%20Specify-native-V2-profile-compatibility-and-atomic-admission.md)
is complete after explicit user approval of the [written native design](../../Docs/superpowers/specs/2026-09-27-personal-context-native-v2-admission-design.md)
and [accepted design ADR-193](../decisions/193-native-v2-profile-compatibility-and-admission.md).
The design places one current compatibility barrier at service and repository
boundaries, including individual getters and all mutation/replay paths. Exact
native admission is separate from structural validation, source truth, human
review and model consent. Unsupported semantics block the whole profile,
including V1 records; offline status or a capability declaration is no retirement
or qualification receipt.

The proposed sequence starts with an integrated native barrier/codec with all
production V2 permit paths closed. Qualified retirement/publication, disclosure,
exact admission, companion-server/cohort conformance and foreground cutover/source
inspection follow as independent units. No local-only exception or fourth fact
store. The written design includes positive/failure/race qualification cases;
none are claimed as passed V2 native runtime evidence. The user explicitly approved the written design on 2026-09-27;
ADR-193 accepts design direction only. The native Unit A plan and closed barrier candidate are now implemented under TASK25907.22; its required-check closeout is complete. No V2 qualification, activation or storage/AAD migration follows design approval.

## Native Unit A barrier complete and verified

[TASK-25907.22](../tasks/task-25907.22%20-%20Add-a-closed-native-V2-profile-compatibility-barrier.md)
is Done with the [native execution plan](../../Docs/superpowers/plans/2026-09-27-personal-context-native-v2-barrier.md)
under accepted ADR-193. One implementation/review unit adds explicit V1/V2
validation and a repository/service/bootstrap barrier that keeps every production
V2 permit closed. The plan covers individual reads, incoming canonical writes,
transactional first-link paths, local metadata/Undo/outboxes, app/status/context/
tools and existing interview/export/recovery/Sync consumers. Real encrypted
SQLite positive and denial controls protect existing V1 behavior.

The closed barrier is implemented and has one fresh final read-only review. Its Important known-V1 quarantine finding was fixed with observed RED/GREEN controls; narrow known-V1 corruption quarantines, while V2, unsupported versions, extra fields and ambiguous privacy data remain unavailable without omission or SQL changes. Real first-link ordering, encrypted ingress, stale state, rollback, app/context/tool/interview/export/recovery/Sync owner controls are covered.

Final native Python 3.12.11 verification: 925 passes plus one Pilot harness failure in the 926-case owner selection, repaired and covered by a final 34/34 affected Library/closure rerun. Barrier/memory checks pass 176/176 with all 68 baseline tests and unchanged timeouts; import/package checks pass 15/15 and raw Settings 13/13. Budgets remain 641/660 app, 973/973 UI-ready and 498/500 + 377,190/378,740 LOC preimport. Ruff check/format passes 14 unit files; seven legacy UI/test files add no lint or formatter debt against immutable HEAD. TASK-25907.22 is Done with all eight AC checked. [Component evidence and remaining gates](personal-context-native-v2-barrier.md) records exact receipts and practical limits.

All production V2 permit paths remain closed. Retirement/publication, disclosure, exact admission, foreground source, storage/AAD migration and companion-server/cohort cutover remain independent prerequisites. Shared profile-library schemas/fixtures/contracts and the independent generated-answer task/suffix remain unchanged.
