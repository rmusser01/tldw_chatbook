# ADR-226: Console polling and full-state reconciliation

Date: 2026-10-08
Status: Proposed; source review only. No product implementation selected or authorized by this document.
Task: [TASK-34563.33](../tasks/task-34563.33%20-%20Plan-Console-polling-and-full-state-reconciliation.md)
Plan: [Polling and reconciliation](../../Docs/superpowers/plans/2026-10-08-console-polling-reconciliation-plan.md)
Extends: ADR-126 and [ADR-225](225-console-send-preparation-and-io-ownership.md); retains ADR-097 budgets.

Renumbering provenance: this unmerged Console proposal used number 223; PR #3050's rebase preserves dev's landed persistent embedding content-hash cache decision at that number.

## Context and measured problem

The d221 quiet Send witness records 43/48/39 main-thread configuration-admission entries under full Console synchronization. Their entry intervals precede the callback bodies and include nested native observation. Main settings reads are cache hits. The 0.2-second transcript timer starts during Preparing/custody, awaits the entire UI synchronization, then decides whether to stop. Its original ancestry directly accounts for 7/11/7 entries (.811/1.280/.781 seconds); absent async origins prevent assigning all entries to polling. The artifact is incomplete and does not establish ordinary native cleanup or an optimization saving.

Full sync does more than paint: it publishes provider/workspace selection and the agent-runtime gate, materializes roleplay projections, issues pending persistence, reconciles attachment, and updates transcript/tabs/attention/fleet state. A redraw fingerprint occurs after the expensive live admissions. Existing request booleans collapse overlapping calls but do not express a difference between routine display demand and required reconciliation.

## Proposed decision

1. First establish original request origins and real transition controls. This is a bounded prerequisite, not permission to implement a guessed dirty token. In particular, session settings revision does not cover in-place app configuration changes to the runtime gate. A control which manually invokes full sync cannot qualify the producer-to-next-Send path.
2. Preserve `_sync_native_console_chat_ui()` as the FULL default for existing explicit callers. Propose one named routine-poll entry that reuses the existing transcript, session-tab, checked control-bar and mode-bar owners. Do not make the timer transcript-only.
3. Share the existing sync exclusion, teardown/maintenance/attach-visit checks and trailing full-replay owner. A full request during a narrow pass dominates and survives to one trailing original full reconciliation. A routine request during a full pass can be absorbed by that full pass. Deferred/failed work cannot complete attachment or clear the pending full request.
4. A routine pass refreshes current presentation and retains only explicitly classified existing helper effects under their original owners. Transcript refresh can durably acknowledge unseen terminal receipts; tab/manual-read publication is not automatically pure. Task 1 must preserve those independent authority boundaries or escalate before the effects. Disposable display evidence cannot authorize provider/controller/store mutations. Missing session/settings, changed owner or membership, unavailable prerequisite, unresolved/custom behavior, or a pending full reconciliation retains the original full route. An observation fingerprint is never permission.
5. Preserve explicit full transitions for settings/provider/runtime configuration, workspace/session activation, attach/resume, trace recovery, rewind, Stop and terminal settlement. Roleplay identity changes and repair remain live. Pending roleplay persistence drain cannot be dropped merely because its session/name key matches.
6. Preserve the original poll-needed predicate across viewed/background runs, Preparing custody, wake-delivery gaps, queued review publication and console-run workers. When that predicate becomes false, one successful current FULL pass and a post-await owner/poll-needed recheck must precede timer stop, persisted-browser invalidation and fleet-survivor handoff. Deferred entry or newly pending work keeps polling/replay alive. Unclassified speech-context/H3 completion effects remain on the original full route until their triggers are proved. Exact rendered-row acknowledgement and owner checks remain with their existing functions. A viewed turn settling while background work keeps polling active still requires its full terminal transition; the stop-edge full pass is not a substitute.
7. Do not introduce a global cache, generic scheduler, new profile-wide generation, permission proxy or another lifecycle owner. Do not change durability, saved-turn failure policy, native scope lifetimes, worker retirement, deadlines or acceptance budgets.

## Review gate and unresolved proof obligations

This is a proposed routing contract, not a claim that existing invalidation signals are complete. Before any production timer is narrowed, the plan must establish: (a) every supported settings/runtime change reaches fresh full publication before its dependent real action without manual test sync; (b) tab/transcript helpers do not create authoritative state on a routine route, or promote to full before that effect; (c) pending roleplay drain/repair remains scheduled; (d) full requests arriving across awaits survive; (e) terminal/background/attention/attachment behavior remains exact. Unknown shapes stay full. If those properties cannot be established with existing owners, return a bounded revised design to review rather than inventing a signature or weakening the checks.

## Alternatives

| Alternative | Decision |
| --- | --- |
| Another settings cache | Rejected for this measured route: observed settings reads already hit their cache. |
| Hoist roleplay key equality ahead of admission | Not selected: the key omits source/profile/incarnation/pause and the dispatcher first starts pending drain. A separately proved effect-free route is a possible later option. |
| Pass checked display projection through both live callbacks | Rejected: it changes authority for core and roleplay publication. |
| Replace the poll with transcript-only refresh | Rejected: loses tabs, review/run/fleet markers and terminal handoff effects. |
| Reduce polling frequency or loosen deadlines | Not selected: hides work while reducing immediacy; does not establish unchanged behavior. |
| New invalidation graph, queue, generic pipeline or worker for the whole old body | Not selected: adds ownership or merely moves repetition. Reuse existing finite owners and coalescers. |

## Qualification

Use original-callback causal controls and real mounted transitions before selecting the narrow route. Then run only the targeted controls in the plan and sequential original native/real-provider measurements on saved integrated sources. Count reductions, diagnostics and completed replies do not establish the under-one-second overhead or 100 ms input/render contract. Keep whole-phase attempts distinct from unique files or dispatch-only work, and physical process retirement distinct from app-owned native completion. No app, tests or native probes ran in this source-planning lane.

## Accepted bounded manual-Preparing amendment (2026-10-08)

The integration owner selected the sole-stock-unpromoted-manual-Preparing branch in TASK-34563.39 following source review and original mounted Task34563.38 qualification. Only this bounded route is accepted for implementation; the general polling proposal and its wider evidence gate remain Proposed. The implementation and original-callback/native qualification are pending.

Use the completed-current-FULL source witness and exact received-intent/task/claim/input/attachment ownership described in the plan amendment. Compare controller selection with the detached receipt selection and keep the canonical live runtime gate, exact bridge and stock callbacks aligned. Recheck after retained awaits; changes and pending FULL/roleplay effects retain original FULL. Preserve the complete original helper sequence, explicit callers, wake/background/queue/viewless/custom/terminal routes and final timer settlement. A routine success never completes attachment or grants mutation authority.

The manual action independently performs checked capture and submits its completed request configuration. Runtime wake and compatibility producers can use a cached controller runtime gate, and missing-settings selection can use cached provider fields; they therefore remain FULL. This concrete action boundary replaces a guessed general selection-equivalence signature for the first implementation. ADR126 native authority/retirement, ADR225 custody/fresh action capture and the original performance budgets remain unchanged.


## Integrated implementation status (2026-10-08)

The bounded manual-Preparing route above is implemented in PR #3023 and retained by PR #3049. TASK-34601 L2 additionally implements the user-approved routine live-poll cadence in `UI/Console_Modules/poll_cadence.py`: between full passes, publish session tabs, transcript and the run chip through their original owners; request full reconciliation on owner/membership uncertainty, pending full demand, observed settlement, the stop edge and the two-second cadence. Explicit callers retain full reconciliation. A Preparing-narrowed pass never records a completed full reconciliation. The same existing exclusion/replay owner retains a full request arriving across a light pass.

This records the selected bounded implementation, not completion of every general proposal above. The wider source/transition proof obligations and under-100-ms/under-one-second targets remain open. Integrated native polling/reconciliation/source-boundary controls pass in the 92-case integration receipt, and independent source review covers late-full and viewed/background terminal handling. Exact final-source cross-platform and real-provider qualification remain pending; prior c504 CI results cannot qualify PR #3049 additions. TASK-34601 and the optimization ledger retain those evidence limits.


## Bounded context-publication experiment (2026-10-09)

TASK-34601 OPT10 classifies only ConsoleContextReadSnapshot's disposable result-publication callback as an additional caller of the existing guarded manual-Preparing display route. This is within the approved latency simplification work; it does not accept the broader proposed reconciliation design. The plan is Docs/superpowers/plans/2026-10-09-console-context-publication-routing.md. Exact source/receipt eligibility and every retained display helper remain unchanged; other explicit callers remain FULL. No context key, TTL, native owner, permission or durability boundary changes.

Unlike a routine timer tick, a newly published context result cannot be absorbed by an already-running pass that may have consumed older data. During sync overlap, call the originally captured FULL callback so its existing pending-demand flag guarantees a trailing pass. Otherwise enter the existing guarded route without an intervening await. Legacy/custom callbacks preserve their captured no-argument contract. Source changes, promotion, deferral and teardown retain the original fallback and retirement behavior.

Qualification must include real context publication under held manual Preparing, overlapping publication after old data was consumed, original display consumers, and existing drift/full-replay controls. Adopt only after sequential whole-Send evidence or an independently justified correctness improvement. The bounded route is now implemented and retained after51 integrated local controls, six strengthened publication controls and sequential ABBA/BA timing. Confirmation warm means are3.014297s baseline and2.799411s candidate; the initial baseline15.515540s outlier remains unexplained and is not used to claim a56.3% gain. See the plan for complete receipts, source limits and pending remote/physical-terminal acceptance. This does not accept the broader proposed reconciliation design.

## Bounded recovered-image publication experiment (2026-10-09)

TASK34601 OPT92 proposes reusing the existing live-poll routing for changed recovered-image lookup completion only. The catalog result must still publish even when empty to release transient/remote-image suppression. Original metadata/payload reads, cancellation retirement and other image action callbacks remain unchanged. Plan: Docs/superpowers/plans/2026-10-09-console-recovered-image-publication.md.

The production callback keeps FULL for custom FULL implementations, overlap and every existing poll_wants_full_sync refusal (unproven/changed owner, deferred/pending demand, no live work, settlement and cadence). A result arriving during a pass may follow that pass's consumption of old rows; it must request trailing FULL through the existing owner. Otherwise enter the original light display pass without an intervening await. No new cache, scheduler, authority, cadence or lifetime contract. Adoption requires targeted original-body controls and sequential matched whole-Send evidence; the broader reconciliation proposal remains separate.


OPT92 disposition: archived without adoption. Original-body controls pass, but interleaved timing does not establish a repeatable whole-Send gain: the stable A2 baseline and B1/B2 candidate warm means are4.700/4.731s; slower A1/A3 samples dominate the favorable aggregate. Product and candidate-only tests were restored. Full receipts and limitations remain in the plan.

## Context presentation dependency correction (2026-10-09)

TASK34601 OPT10 corrects the existing disposable context memo key. Only the original presentation reader and held-echo getter qualify for replacing run status/active-run ID with their actual held-compaction echo input. Custom or replaced callbacks keep conservative lifecycle invalidation. Exact owner/repository/workspace/revision/config fences, the one-second expiry, native retirement, and fresh live action/dispatch reads remain unchanged. Source drift changes the key and refuses old publication. Plan: Docs/superpowers/plans/2026-10-09-console-context-presentation-dependencies.md. Implemented after six intended original-source failures, then qualified by57 integrated cases and two rendered-feedback controls. Retained for stale-result/source-drift correctness. Warm timing is worse4.108-to6.057s amid large unchanged-baseline variation; no whole-Send speed benefit or causal regression is established. Full receipts and limitations remain in the plan. This does not accept the broader reconciliation proposal.
