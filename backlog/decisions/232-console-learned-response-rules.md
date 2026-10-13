# ADR-232: Native learned Console response rules

Date: 2026-10-03
Status: Accepted by user on 2026-10-03 after written-spec review
Related Task: [TASK-34354](../tasks/task-34354%20-%20Validate-native-response-rule-definitions-and-deterministic-outcomes.md), with remaining atomic implementation tasks linked from the plan.
Spec: [Console learned response rules](../../Docs/superpowers/specs/2026-10-03-console-response-rules-design.md)
Plan: [Console response rules implementation](../../Docs/superpowers/plans/2026-10-03-console-response-rules.md), accepted with requested audit; Native execution selected on 2026-10-03
Extends: [ADR-163](163-expanded-console-hook-runtime.md) at the shared continuation admission boundary
Follows: [ADR-029](029-local-private-data-boundary.md), [ADR-092](092-console-chat-fork-copy-and-authority-boundary.md), [ADR-126](126-complete-local-backup-and-recovery.md), [ADR-197](197-console-hook-configuration-review.md), and [ADR-210](210-console-region-ownership.md)

## Context

The user approved a Console `/omfg <problem>` workflow that drafts a native structured rule, tests it against a reported mistake and acceptable behavior, activates it for the current Chat, and corrects later completed responses from existing work. The user chose completed-response checking and explicit Workspace/global promotion, then requested two design review passes before a written spec.

The current hook runtime does not provide this contract: Stop exposes status, forbids required response checks and admits follow-ups only through an owned hook outcome. Native learned rules cannot manufacture that provenance. Regenerate creates a sibling branch rather than retaining the source answer and already completed work.

Rules also need local ownership, stable revisions, bounded model evaluation, truthful failures, privacy-preserving promotion and recovery that cannot replay uncertain actions. Placing activation in synchronized conversation metadata would violate the approved device-local scope.

## Decision

1. Introduce versioned, non-executable response-rule definitions and explicit Chat/Workspace/profile-global bindings. Chat is the default scope. Supported native text predicates and bounded semantic criteria contain no generated code, arbitrary regex, tool permissions or approval exemptions.
2. Keep one profile-owned local rule store under the existing private SQLite owner, outside conversation metadata, sync, server payloads, Chat export and handoff. Add a real migration during implementation. Temporary Chat rules remain memory-owned; durable activation follows successful commit.
3. Foreground `/omfg` may automatically activate a successfully tested, still-current native candidate for that Chat. A failed, stale, unsupported or unverifiable candidate stays inactive. Preserve recorded-versus-synthetic evidence and label success as tested against examples.
4. Inspect eligible completed user-facing Console text answers through the shared runtime, independently of whether hooks are enabled or a view is attached. Keep generation/persistence outcome separate from assessment lifecycle and pass/violation/couldn't-verify verdicts. Preserve answers and finite cancellation/deadline behavior.
5. Extend the existing queue owner's machine-follow-up admission interface with an authenticated host-native correction source. Do not fabricate a Stop event, HookResult or grant, create another scheduler, or resurrect retired run authority. Combine native feedback and valid hook proposals into one admission decision per settlement; preserve controlling vetoes, required gates, current permissions, custody, deduplication and recovery.
6. Automatically triggered repairs inherit the accepted live turn's remaining limits. The initial repair requested by manual `/omfg` uses a fresh current user-authorized operation anchored to the earlier exchange. Both retain prior answers/tool results and apply normal review to additional actions. Two native correction turns share the existing continuation limits and yield to user input and Stop.
7. Pin source response/branch, rule revisions, effective scope and current authority at snapshot, commit and admission. Edits, disable/delete, branch/Workspace changes, cancellation and closure invalidate affected stale work. A result cannot reactivate a disabled rule or schedule repair after its authority is retired.
8. Resolve the same logical rule at Chat, Workspace, then global precedence. Allow explicit Chat exclusions distinct from disabling the source. Promotion pins the exact reviewed revision without broadening applicability or copying private fixtures; later local edits do not silently update broader bindings. Separate Chat forks omit source Chat bindings/evidence/repair state; ordinary inherited bindings still apply.
9. Use sensitive, tool-free auxiliary calls through the existing pinned provider gateway. Validate closed results and supplied evidence references; missing/uncertain/truncated evidence cannot establish success or absence. Bound requests, evaluations and output, account actual helper usage, and introduce no hidden model substitution.
10. Register local records with existing Backup/Recovery ownership. Normal reopen restores committed rules but never starts a pending check or correction. Backup restoration imports definitions/history inactive and transfers no live admission authority; exact internal rollback follows existing owner rules. Permanent source deletion removes private evidence and source bindings without deleting independently promoted definitions.
11. Reuse one manager through `/rules`, visible Chat settings and canonical Settings scope entry points. Follow the token/component language, preserve failed answers with attributed correction follow-ups, and keep actual Stop available throughout checking and repair. Executable hook consent under ADR-197 remains unchanged.
12. Retain helper resource reservations until actual provider-worker/transport completion, independently of caller cancellation. Bound unsettled work, use finite remaining-time transport limits without hidden adapter retries, and account late usage only once to its original owner. Keep required postevent settlement before assessment; shared feedback acceptance atomically commits the ordinary dispatch checkpoint and deduplication receipt.

## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Generate external hook scripts | Adds executable review, isolation and maintenance beyond the user's selected native-rule approach. |
| Save complaints as Agent Lessons or prompt reminders only | Does not provide explicit response evaluation and bounded corrective feedback; Agent Lessons retain their existing approval boundary. |
| Add assistant text and mandatory checks directly to Stop | Changes existing event/effect/failure semantics and still conflates native rule evaluation with external hook provenance. |
| Call regenerate for correction | Forks before the source answer and can repeat work instead of continuing from existing results. |
| Let each rule start its own repair loop | Multiplies automatic work, loses user priority and permits conflicting rules to oscillate. |
| Store active rules in synchronized Chat settings | Broadens device-local activation and couples local operational state to portable Chat metadata. |
| Assume check failures pass or retry automatically | Conceals missing evidence/provider failures and spends further budget without a confirmed violation. |
| Restore pending corrections or activation automatically from backup | Confuses historical records with current execution authority and can replay uncertain model/tool work. |

## Consequences

This feature requires a genuine local schema change, a bounded response-assessment owner and a narrow shared queue interface extension. It does not require a new dependency, scheduler, executable hook type or tool permission owner.

Checks can add latency and model usage and cannot guarantee generalization from examples or prevent already executed tool effects. Those limits are reflected in applicability, three-valued outcomes, finite budgets and honest UI copy. Semantic quality needs behavioural cases in addition to transport/storage tests.

The spec defines activation, revision precedence, scope exclusions, privacy, recovery, usage accounting and targeted verification. ADR-163's existing hook protocol remains authoritative; this proposal extends only the documented host composition/admission boundary. Product implementation begins only after written-spec and implementation-plan review.

## Written-spec review refinements

The requested audit clarified actual provider-resource lifetime, independent applicability and mixed verdicts, original-task scope of generated feedback, calibration without invented tool evidence, reuse/freshness of earlier settled results, stable settlement identities and atomic acceptance, inherited correction counters and existing whole-packet limits. These are implementation contracts for the approved response-only workflow, not new permission or scheduler owners.

## Runtime budget handoff

Actual primary AgentService settlement publishes its remaining allowance through an optional body-free callback independently of external hooks. The Console queue pins that result to the accepted custody identity; repairs narrow the captured agent budget and retain the absolute parent deadline through dispatch. Genuine hook budget records use that same custody identity and can further refuse work. Direct continuation and retry roots carry their actual allowance into shared native repair admission. No synthetic hook lifecycle is created to obtain a budget.

## Implementation review clarifications

Promotion retains sanitized host-labelled calibration outcomes independently of private source-row foreign keys. It copies no fixture bodies or generated explanations. Deleting temporary source messages drops their private drafts and assessments while independently pinned definitions remain adoptable with unavailable original-source provenance. Source removal therefore cannot silently fabricate a new original example or break Save.

Capacity checks resolve known saved and live scope combinations under the serialized mutation boundary before activation, enabling or promotion. Logical overrides and exclusions count once. The runtime still reports couldn't verify if a later scope combination exceeds the limit; it never checks only a prefix.

Global management belongs to the current profile and can be assembled before a Console controller or Chat exists. Later execution binds the real Console controller; Settings does not create a substitute execution owner. Management writes recheck profile/scope ownership inside their worker mutation boundary. Testing edits remains a Chat operation.

Native repair acceptance is explicitly machine initiated for prompt-history exclusion. After reopen, the original task can be reconstructed only from inert host-recorded acceptance ancestry on the actual active branch. Historical receipts do not recreate assessment, queue or execution authority and cause no replay.

## Latest-dev migration integration

The publication branch integrates `origin/dev` at `1b12df2757`. Its installed v75-to-v76 notes FTS repair remains unchanged; native response rules now install in v76-to-v77. Restricted recovery retains exact historical v75 and v76 catalogs plus both guarded dictionary-trigger variants. The complete validated v75 catalog already contains the installed guarded notes trigger, so its restricted v75-to-v76 step advances only the version stamp. The following installed declarations create the v77 local-only records on the disposable candidate. Imported content never supplies migration SQL. Fresh and chain-migrated trigger inventories pin the actual response-rule cleanup body.

## Renumbering and current-dev integration

Original source ADR-219 at 73823fde3a4e records the user-approved learned-rules design. The older ADR-219 Console chat destinations keeps its identity; this younger decision becomes ADR-232, preserving its dated approval and implementation history. Tasks34360-34362 become34805-34807 because their older current-dev claimants keep the original IDs.

The historical v77 publication paragraph above describes that earlier source only. Current integration appends the next actual migration (initial reviewed basis v80 to v81), retains all shipped/native predecessor recovery variants and accepts installed SQL only. Helper custody begins before off-loop preparation and remains through actual preparation/transport settlement; cancelled preparation cannot dispatch a provider. Voice adoption classifies all six new message locators as forbidden residue. Heavy rule services load on first eligible use while stored active rules still govern the first send and global management remains available before any Chat. These amendments implement the existing approved boundaries; they add no provider, scheduler, permission owner or dependency.

## Concrete current-owner implementation (2026-10-12)

Verified private START13b204cb uses ChaChaNotes80. Response rules append installed
`chachanotes_v80_to_v81_response_rules.sql`; all four v80 variants gain exact
successors, retaining versions75–80 and shipped/native distinctions. Only validated
selected-import candidates disable activation; exact rollback retains bytes. Six
rule message locators join voice forbidden residues. Seven tables and three query
/index pins join installed inventories. Qualify the exact private candidate
transform call/purpose with arbitrary-owner/site/path negatives, no broad bypass.

One helper reservation covers physical preparation and transport. Cancellation
seals dispatch/results but retains capacity until both settle. Preparation without
dispatch records no provider usage; dispatched unknown usage remains unknown and
late usage is charged once to its original owner. Current endpoint normalization
and ordinary auxiliary defaults remain.

First eligible generation assembles heavy services before its completion hold;
stored active rules govern that first send and global management works before a
Chat. ADR227 Goal allowance-root/automatic context, actual provider-worker custody
and Personal Context disclosure/currentness continue to narrow rule work. Finite
store workers reuse the native connection retirement owner; shutdown retains
unsettled work without closing borrowed handles.

Measure critical safety cases inside existing required CI capacity and pin actual
executable targets with missing/comment-only/suppression negatives. Preserve all
inherited job caps/processes/commands/aggregation; nearly-full Fast is not a default
destination. Actual model/external-hook/terminal evidence stays separately labelled.


2026-10-12 current Goal ruling (ADR227/232): assess only the exact coordinator-issued, accepted Goal iteration handoff pinned through native parent identity and message version. A genuine live GoalIterationAuthorization is required; arbitrary SYSTEM rows, copied metadata and shape-compatible objects confer no rule authority. Recheck Goal/binding/source/Stop currentness. Completed Goal runs expose no existing same-native-run repair seam: preserve the assessment and explicitly refuse automatic repair rather than admitting an ordinary queued turn. No second native run, Goal attempt, machine receipt, allowance or repeated tool work; keep record_outcome identity guard and inherited absolute limits unchanged. Phase12 exposed the genuine source_changed assessment defect; qualify passing, violating and revoked/fake input cases. This enforces inherited ownership and leaves settled-Goal automatic repair unsupported, documented for independent review. ADR required: yes (existing amendment), backlog/decisions/232-console-learned-response-rules.md.


2026-10-12 connected trace lifetime ruling: phase24 reproduced startup GC removing newly admitted user/assistant semantic revisions before provider call reservation, leaving an accepted repair retained for recovery. Apply existing ADR097 decision16 and ADR232: preserve the live provenance owned by current durable dispatch checkpoints through call reservation, without new storage, a retention owner, timing suppression or weaker GC authorization. Map: durable turn acceptance -> canonical user/assistant revisions -> captured request provenance -> ConsoleTraceBoundaryFactory reservation; TraceGarbageCollector mark/sweep must respect the current accepted owner during the gap. Prove actual ordinary-send and native-repair survival with real collector/admission/reservation, and orphan/revoked/current-version/epoch-change reclamation or refusal controls. Costs: a wrong missing root deletes live provenance; an overbroad root leaks orphan retention. Keep existing graph epochs, native handles, deletion grants, privacy/fork/purge and caps. ADR required: yes, existing ADR097/232 amendment; no new ADR ID.


2026-10-12 automatic GC epoch qualification: phase29 real worker/collector reproduced a completed epoch4 result indefinitely reused at epoch6 after checkpoint settlement because physical compaction was threshold-deferred. Root approved invalidating pending_gc_result only when its marked_epoch differs from current_graph_epoch, returning through existing collection. Preserve unchanged-epoch reuse, actual compactor eligibility, unknown/incomplete result failure handling and all scheduling/thresholds. Goal deliberately freezes Capture Off under its genuine SYSTEM handoff; qualify real collection before actual coordinator-owned provider entry and rule assessment without introducing USER checkpoints or new retained roots. Cost if wrong: repeated collection during unchanged idle state or stale logical collection hiding orphan reclamation. Existing ADR097/232 apply; no new persistent declarations.

2026-10-13 canonical ID reconciliation: this unmerged Rules decision was source ADR-219, then review-branch ADR-228. Accepted dev PR3093 now carries the distinct SQLite store-kernel ADR-228. Rules becomes ADR-232 after the current all-ref/worktree census; decision semantics and dated approval/implementation history are preserved, and its current links are repaired.
