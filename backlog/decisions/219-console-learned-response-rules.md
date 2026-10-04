# ADR-219: Native learned Console response rules

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
