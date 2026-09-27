# Opt-in proposal-only Personal Context consolidation

Date: 2026-09-25
Task: TASK-25907.8
Status: Accepted design direction after requested follow-up review; no runtime implementation
Decision: [Accepted design ADR-188](../../../backlog/decisions/188-opt-in-proposal-only-memory-consolidation.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)

## Purpose and alternatives

Turn newly enrolled conversation evidence into a small reviewable batch of memory proposals. Preserve Personal Context as the fact owner, Notes as the document owner, and Agent Lessons as the approved procedure owner. The Muse diagram supplies ideas about consolidation and evidence; its instructions, files, hourly cadence and Dreams subsystem are not requirements.

Three alternatives were compared:

| Approach | Benefit | Limitation |
| --- | --- | --- |
| Continue foreground manual proposals only | Existing authority and lifecycle; least additional machinery | Misses useful facts across sessions; remains available as the control workflow |
| Opt-in bounded processing of new source revisions | Measurable assistance with exact evidence and user acceptance | Requires qualified source, proposal, budget and recovery owners; chosen future direction |
| Hourly global rereading and free-form reflection | Broad unattended coverage | Repeated evidence, cost, authority drift and forgetting races; rejected |

The first future slice offers an explicit Run now operation on enrolled new evidence. An idle wake may later coalesce the same durable jobs through existing automatic-work owners; it does not create extra budget or source authority. No timer is required. No latency promise applies while the app is closed. This design is off by default and is not an implementation plan or authorization to call a provider.

## Existing boundaries and missing capabilities

- [ADR-102](../../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md) and [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md) retain canonical memory ownership and foreground user authority.
- [ADR-185](../../../backlog/decisions/185-versioned-profile-evidence-and-temporal-claims.md), [ADR-186](../../../backlog/decisions/186-dependency-aware-personal-context-forgetting.md) and [ADR-187](../../../backlog/decisions/187-personal-context-provider-disclosure-authority.md) define future evidence, retirement/suppression and model disclosure. Their accepted designs do not supply shipped safeguards.
- [Proposal service](../../../tldw_chatbook/Personal_Context/proposal_service.py) supports CREATE, UPDATE, ARCHIVE and PROMOTE. Acceptance requires a USER actor and exact current base versions. Its five-per-root-turn/25-per-session quota is process-local; it is not a durable daily budget. CREATE currently defaults to syncable, agent-visible proposed records.
- [Repository](../../../tldw_chatbook/Personal_Context/repository.py) owns proposal publication and resolution. Existing Sync paths can include proposals with no proposed_record, including ARCHIVE. Future local-only proposal custody must be enforced independently of record presence.
- [ADR-106](../../../backlog/decisions/106-human-reviewed-agent-lesson-promotion.md) requires each Agent Lesson mutation to have an exact foreground approval consumed by the Notes owner. Abandoned or rejected previews remain ephemeral. Consolidation cannot persist hidden lesson drafts or create a competing procedure queue.
- [ADR-131](../../../backlog/decisions/131-durable-agent-budget-accounting.md), [ADR-134](../../../backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md) and [ADR-135](../../../backlog/decisions/135-fleet-completion-delivery-and-crash-recovery.md) provide native durable automatic-work admission, counters, physical-work leases and conservative recovery. Their budget units do not guarantee billed currency.

Operational source cursors, job identities, counters and content-free receipts extend existing native owners. They are not a fourth fact store, a searchable extraction cache or a duplicate profile. No current V1 bytes or schema are changed here. Subsequent implementation tasks must version native storage and the shared contract, with migrations and compatibility gates.

## Explicit enrollment and eligible sources

The foreground user enables the feature for one unlocked profile, selects exact native conversation/workspace source bindings, reviews limits, and enrolls a qualified owned on-device destination for the distinct purpose **consolidation**. Source access, agent PROPOSE authority, model disclosure and feature enrollment are independent requirements. A conversation, summary, interview or embedding grant is insufficient. ADR-188 proposes adding consolidation to ADR-187's unshipped purpose vocabulary; if that vocabulary has shipped, use a subsequent version.

Enrollment is a staged native intent/receipt/activation handshake, not a cross-database atomic write. The coordinator durably records one immutable enrollment ID, source binding and current profile/feature/grant epochs in an inactive state. The source owner atomically captures and persists its watermark, source-authority revision and exact enrollment receipt. That source transaction defines the cutoff; later acknowledgement never moves it. The coordinator acknowledges that same receipt and activates only if all reviewed epochs and bindings are still current. For multiple bindings, each has its own receipt; work stays inactive until every required receipt is acknowledged. Crash recovery resolves the original receipt or pauses; it never silently recaptures a later watermark. Orphan/ambiguous enrollment or missing/compacted feed coverage refuses processing until exact foreground recovery. Revocation invalidates pending intents and source-owner enrollment handles. Existing history is excluded. A later, separately qualified foreground catch-up preview may enroll a finite exact range; it cannot silently backfill all history. The first slice admits only finalized, newly committed user-authored message revisions from these bindings. A source owner must distinguish direct literal user statements from quotations, attachments, code, imported material, assistant text and tool output. Trusted capture provenance and structured source parts establish role; a stored sender/role=user, imported message field, substring match or model-provided attribution is insufficient. Mixed text whose direct-user span cannot be established by the native capture contract is deferred for foreground review. Such supplied material is data, never instructions or independent user claims. Ambiguous attribution, unsupported time transitions and unresolved authority abstain; they do not become assertions through repetition.

Every source identity binds owner, object, committed version, exact span, root lineage and current source/control authority under ADR-185. Edits supply new revisions but are not automatically new truths. Source-level no-memory marks and automatic-family suppression cover edited copies as ADR-186 specifies; a new message ID does not evade them. Existing permitted records can be comparison context, never a new trigger or independent supporting source. Summary paraphrases, prior proposals and copied quotations share their original roots and do not increase confidence or evidence count. A manually authored independent record remains independent.

Preflight excludes unauthorized scopes, user-only inputs, retired/deleted versions, unknown legacy lineage, no-memory sources, expired evidence and unqualified destinations. Excluded data must not enter ranking, summaries, payloads or existence/count diagnostics. Operational identities may be retained only where their native owner permits it; source revocation/forgetting retires restricted handles and derivative edges. A generic unavailable state cannot expose hidden locators.

## Native source feed, queue and checkpoint

The durable source-owner feed must expose monotonically ordered committed revision events with current heads and authority. UI notifications are wake hints, not the source of truth. Owners lacking a qualified feed remain ineligible. Native enumeration advances its cursor only after every event in the page has an admitted job or a permissible content-free terminal exclusion receipt. Queue overflow pauses enumeration before unseen events; there is no silent cursor jump.

Separate enumeration from the contiguous processing checkpoint. A hole or pending job prevents the terminal checkpoint passing that event. A bounded page contains at most 16 revisions; at most 64 admitted pending revisions are retained per profile. Missed wakes enumerate bounded pages from the last durable cursor, never reread the entire conversation. Backlog notices expose only authorized operational state. Revoked enrollment pauses existing work; ordinary disabling does not delete canonical proposals or grant replay.

A job binds profile/scope, enrolled source bindings and exact revisions, enrollment epoch, disclosure/control epochs and pipeline version. A separate stable source-coverage/operation identity prevents re-enrollment, catch-up, pipeline changes or new random job IDs from republishing already terminal work. Changed policy alone is not new evidence. Restricted identities retired by forgetting cannot enable replay: native family suppression or a refused unknown-coverage state takes precedence. Its keyed identity is local encrypted operational metadata, never a public hash of private facts. Native unique admission prevents two processes admitting the same work. One active consolidation job per profile is the ceiling; automatic-work capacity and manual reservations still apply. Cross-process access to the same custody must share a qualified transactional coordinator or refuse consolidation. This is not a machine-wide guarantee over unrelated runtimes.

| State | Owner action and recovery |
| --- | --- |
| Admitted | Encrypted permitted references only; no copied source bodies. Revalidate before reading. |
| Prepared | Persist an attempt identity and reserve cost/physical-call admission before dispatch; no durable prompt/response cache. |
| Accepted | Actual adapter entry may have begun; retain budget reservation and physical-work lease until native settlement. |
| Response validated | Still transient; publish only through exact proposal-owner checks and receipts. |
| Published/no change/excluded | Content-free native terminal receipt permits contiguous checkpoint progress. |
| Deferred extraction | Oversize input or completed malformed/truncated/cap-invalid output: content-free failure reason, no proposal and no no-change receipt. Holds the checkpoint until exact foreground retry or reviewed skip. Known actual spend is settled; unknown spend stays held. |
| Review required | Uncertain dispatch/spend or crash; no automatic replay. Holds the checkpoint and outstanding reservation until reviewed owner recovery. |

Preparation and acceptance are native runtime states, not assertions made by the model. A job row or queued payload is not evidence that a model call was delivered. A typed proven pre-entry refusal may be retried after fresh checks; every physical attempt still receives its own admission. Restart with ambiguous prepared/accepted work becomes review required. The user may abandon through a reviewed native recovery operation; it does not refund unknown spend or erase suppression.

## Before call, after call and publication

1. Validate unlocked ownership, enrollment, exact current source versions/spans, scope/PROPOSE ceiling, suppression/purge epochs, required comparison-record heads, destination/purpose grants and budgets before source reads or payload preparation.
2. Reserve up to five publication slots in the automatic-work ledger before the call, alongside physical-call/token admission. Bind its immutable single-use publication ticket to exact job/profile/control epochs and the reserved ceiling. A claimed ticket stays reserved through owner publication/recovery; release requires a proven never-entered owner refusal or a qualified exact terminal receipt. Build a bounded closed extraction request. No agent loop, child runs, general tool catalog, web, MCP, skill, shell, publication or fallback route is available. Document contents and model output cannot change the trusted origin, budget chain or operation ceiling.
3. Revalidate the ADR-186 retirement and ADR-187 destination gate at qualified adapter entry. No network wait is held inside the gate. Retried or changed destination requests require fresh admission; first-release offloading and remote fallbacks are disabled.
4. After the call, validate the structured candidate against exact current source authority/version/span, current targets/manifest, grants and all retirement epochs. A stale or suppressed response is discarded, including temporary candidate buffers. A complete valid source quote proves what was said, not whether the assertion is true.
5. Publish the complete validated batch of at most five proposals, its unique job/ticket operation receipt, dependency edges and exact actual publication count in one Personal Context owner transaction, with a commit-time retirement fence. This publication is all-or-none; a quota/conflict or invalidated required input publishes none. Each finite batch has exact proposal identities in the native receipt; there is no partially committed candidate-slot state. The native publication gate binds the live reserved ticket through COMMIT and receipt, with no expiry/release race; Personal Context checks ticket identity, ceiling and unique operation alongside its own quota. The automatic-work ledger then consumes the owner receipt once in its own transaction: charge the actual publication count and release only the proven unused slot remainder. No-change/never-published terminal receipts settle zero publications; they do not refund a delivered model call. Ambiguous publication holds the full slot reservation until receipt reconciliation. Ticket revocation/recovery uses the existing common native gate, not mutable UI counters. There is no pretend cross-database atomic commit.

Managed prompt cache and live slot state follow ADR-187 and [ADR-119](../../../backlog/decisions/119-llamacpp-prompt-cache-snapshot-ownership.md): unknown prior lineage refuses use; qualified execution requires a reviewed clean owned process/slot or complete current-policy lineage. A new conversation does not clean model state. No implicit reset, cache deletion or forensic RAM-erasure promise is introduced.

A crash after batch commit but before checkpoint acknowledgement recovers the complete batch from the content-free exact operation receipt and does not call the model or republish. A crash after dispatch/response but before committed receipt is uncertain; no partial proposals exist, native accounting stays charged/held and requires review. A reviewed retry of that uncommitted whole attempt needs new current admission and spend allowance; it cannot reinterpret a committed batch as unfinished or recreate it. Existing proposals resolved/forgotten in the interim retain only permissible suppression/idempotency receipts and cannot be resurrected. Receipt content and retention qualify under ADR-186. Exactly-once physical provider execution is not promised.

Lock, cancellation, deadline, deletion or changed authority stop new reads, adapter entry and publication. Best-effort worker buffer cleanup does not guarantee Python-memory erasure. If adapter entry has already begun, exposure is reported honestly as begun/uncertain; cancellation cannot retract it. The native physical owner retains capacity and unsettled spend until actual exit/settlement, rather than releasing on waiter cancellation. Unlock revalidates pending jobs and does not replay uncertain work. Forgetting invalidates queued references, buffers, proposals, receipts/edges where restricted, captures/logs and later recovery under ADR-186's owner inventory.

## Canonical proposal operations and review

| Candidate | Canonical result |
| --- | --- |
| Supported durable fact/preference | CREATE with exact evidence and proposed status |
| Explicit correction | UPDATE proposal against an exact current record/base version; ADR-185 exact correction relation/effect and supporting span; no change_from/transition_at is inferred from chronology |
| Duplicate merge suggestion | Foreground guidance linking exact authorized record versions, then individually reviewed UPDATE/ARCHIVE proposals. No new MERGE enum or atomic multi-record merge promise. |
| Archival suggestion | ARCHIVE proposal bound to exact current target/version and evidence; no silent deletion |
| Ambiguous, unsupported, unchanged or repeated claim | No proposal; permissible content-free terminal outcome |

Consolidation never calls accept_proposal, direct_update, PROMOTE or a user-goal mutation path. Native candidate validation excludes CREATE with RecordKind.GOAL and UPDATE/ARCHIVE/merge guidance targeting a GOAL; model classification cannot bypass the current target kind. Ambiguous goal-like changes abstain for separate foreground work. It cannot impersonate ActorType.USER, revise standing goals or write instruction files. Existing schema-valid content alone is not an authority check. Every operation must pass the native PROPOSE ceiling and versioned evidence validation.

The future proposal envelope has its own custody, disclosure and lineage restrictions, including body-less ARCHIVE operations. First-release automatic proposals are device-only and model-deny. Pending storage, outbox, first-link Sync, staging, review, export and receipts enforce those controls; current syncable defaults and missing-proposed_record branches are not reused. Source restrictions intersect at proposal publication. Acceptance does not silently widen these controls, grant an audience or make restricted inline source metadata portable. Any desired widening uses a separate exact foreground review permitted by ADR-185/187. Conflict/expiry/suppression invalidates approval.

Foreground batch review shows at most five authorized proposals, each with exact proposal version, operation, target version, supported source evidence and honest uncertainty. Batch acceptance names this finite set and consumes independent owner preconditions for each item. Partial conflicts are reported per item; no all-or-nothing multi-object transaction is implied. Editing changes the reviewed payload/version. A blanket approve future suggestions option is excluded. Rejection, expiry and supersession follow existing content-shredding resolution plus future evidence/receipt restrictions; no rejected prose training store is created.

Procedural Notes or Agent Lessons are outside the first slice. A later explicit foreground handoff can show an ephemeral suggestion and reuse the exact Notes/ADR-106 preview-and-approval contract. It cannot queue durable lesson drafts in Personal Context or keep rejected previews. This preserves all three native owners without a new Dreams store.

## Opt-in limits and quiet behavior

These conservative first-release ceilings can be narrowed by the user. Zero call allowance disables calls. Lower existing service/automatic-work limits remain binding; Run now is charged to the same consolidation counters as any later idle wake.

| Limit | Per run | Per rolling 24-hour profile window |
| --- | --- | --- |
| Physical model calls | 1, no automatic retry | 4 |
| Prepared input | 4,096 tokens, including comparison context and instructions | Included in admission reservation |
| Maximum output | 1,024 tokens | Included in admission reservation |
| Admission budget tokens | At most 5,120 | 20,480 |
| Published proposals | 5 | 20 |
| Active deadline | 60 seconds; actual worker retains lease if cancellation has not finished | No lease release by deadline alone |
| Paid/network calls | 0 | 0 |

Token estimation must be adapter-qualified, cover the complete prepared request and reserve its input plus maximum output before physical entry. Unknown tokenizer or unenforceable output bound refuses work. Body/schema byte limits and bounded paging apply independently of tokens. Pack whole revisions into a smaller bounded cohort when necessary; never cut a statement/span. A single revision that cannot fit whole becomes deferred extraction, holds the processing checkpoint, and needs an exact foreground retry under unchanged caps or a reviewed terminal skip naming that revision. It is neither silent exclusion nor no change. At most 16 source revisions and five candidate slots enter a run. Return compact candidate wording/action and allowlisted transient source handles/spans; native owners assemble canonical IDs, exact evidence bindings, controls and envelope metadata after validation. The model cannot invent source authority or role. Length-truncated, malformed, schema/byte-invalid or excess-candidate output rejects the complete batch to deferred extraction, without salvage, an extra repair call or automatic retry. The same exact foreground retry/skip rules apply; a proven completed call settles known actual usage, while genuine uncertainty stays reserved. Quiet no-change requires a complete schema-valid explicit no-change result. Final output cannot increase the slots or override their schema limits.

Durable per-profile counters and rolling reservation timestamps extend the native automatic-work ledger. Wake, app restart, route change, pipeline upgrade and a second process do not reset them. Expired settled reservations age out after a trusted 24-hour interval; uncertain/unsettled reservations remain held regardless of age. Native clock continuity is qualified; backward or unexplained forward discontinuity pauses admission for recovery. Timezone/calendar changes grant no allowance. Reservations include every physical attempt and survive lock/cancel. Unknown usage is never settled as zero; above-reservation usage stops subsequent work and is reported honestly. Publication-slot quota is authoritatively reserved and settled in the automatic-work ledger, counting used plus reserved slots against the same window. Personal Context stores the exact actual count with its batch receipt; it does not mutate a ledger counter inside its own transaction. Settled publications enter the rolling window at the trusted proposal-owner COMMIT time carried by the receipt, not the earlier reservation/model-call time or later acknowledgement time. Coordinator recovery reconciles the immutable ticket/receipt before more admission. Unknown publication slots remain held regardless of window age; a second process cannot spend them.

Budget units, measured provider usage, elapsed time and monetary spend are separate fields. This slice permits no paid/network provider, so configured monetary allowance is zero for model-service spend; local electricity/hardware cost is not measured. A later paid release needs separately reviewed destination/rate/currency revision, conservative monetary reservation for full prepared input and maximum output, uncertainty holds and native settlement. Estimates cannot establish a hard provider billing guarantee. No current price or exchange-rate assumption is embedded here.

No eligible work means zero calls. A validated no-change result makes a permissible terminal receipt and no notification. One content-free authorized review notice may announce a newly usable pending batch; lock/revocation invalidates it. Stable pending work is quiet. Repeated failures produce one actionable authorized recovery state rather than repeated summaries, alert spam or unbounded retries. Scheduler defaults that announce every successful run are not suitable for this job.

## Synthetic evaluation and first-release qualification

Keep the frozen 24-case retrieval baseline unchanged. Build a separate synthetic consolidation corpus with at least 40 independently labelled root-source families, each spanning multiple sessions where useful. Freeze family-disjoint development/held-out splits before tuning. Labels cover the exact useful action, admissible span/version/scope, independent roots, and required abstention; output from the proposing model cannot supply its own truth labels. Separate generated-answer effectiveness remains a different roadmap task.

Compare manual/no-consolidation with the bounded proposal workflow on identical permitted source families and disclosure policy. Record the manual proposals and review effort, automatic supported useful proposals, useful labelled opportunities missed, duplicate publication rate, human rejection rate, per-call prepared/output usage, retained unknown reservations, elapsed time and local resource use. Rejection is not synonymous with falsehood, and a quiet system producing no proposals cannot pass usefulness on abstention alone. Report denominators and case counts; small synthetic samples are not production precision estimates. Any future model execution needs its own authorized execution budget and qualified local model; this task calls none.

First-release synthetic gates, registered before execution:

- Zero forbidden disclosure, self-approval, stale/suppressed publication, unauthorized diagnostics or Notes/goal mutation across the owner tests below; successful allowed controls must reach each native entry point.
- Exact valid evidence for every published supported proposal. Independently labelled supported/useful precision at least 90%, with at least ten useful held-out opportunities and at least one valid published CREATE, correction UPDATE and ARCHIVE. Report useful-opportunity recall without claiming a large-sample guarantee.
- Zero duplicate publication for the same operation identity under retry/restart; semantic duplicate rate at most 10% of published proposals in the frozen held-out set. Report human rejection separately; do not tune to a low rejection rate by hiding ambiguous outputs.
- Every physical attempt, uncertain attempt and publication is reconciled within stated caps or blocks further admission. Healthy no-source/no-change controls stay quiet. Misleading cancellation refunds or elapsed-window release of uncertain spend fail qualification.

Future targeted native qualification cases:

| Case | Required outcome |
| --- | --- |
| Fresh permitted literal fact | One exact supported proposal, visible for foreground review; healthy gate control succeeds |
| Correction vs newer unrelated statement | Exact UPDATE relation only for explicit supported correction; no inferred transition |
| Duplicate and repeated assistant summary | Guidance or abstention; original source counted once; no independent echo evidence |
| Quoted instructions/attachment/goal rewrite | No instruction execution, self-approval, goal mutation or procedural write |
| Old history, unenrolled workspace, edited no-memory source | No admission or disclosure; hidden identities/counts absent |
| Oversize source, full queue, missing feed event | Deferred exact revision/paused enumeration, checkpoint held until foreground retry/skip; no truncated evidence or cursor skip |
| Imported role=user or unattributable mixed text | Native capture provenance required; no inferred direct-user origin; healthy native direct capture succeeds |
| Crash between enrollment intent, source receipt and activation | Same immutable enrollment/watermark receipt or inactive refusal; no silently advanced enrollment baseline |
| Publication ticket/receipt settlement crash or delayed release | Used plus reserved slot ceilings hold across DBs; exact idempotent receipt settlement, no early release |
| Malformed/length-truncated/excess-candidate output | No partial publication, model repair or no-change receipt; deferred checkpoint, exact reviewed retry/skip and honest spend |
| Pending hole and missed wake | Durable bounded enumeration; terminal checkpoint stays before unresolved event |
| Revoke/lock/delete before read, before entry, after response, at COMMIT | Native owner fence refuses stale work; healthy controls prove correct entry was reached |
| Already entered provider and canceled waiter | Honest exposure/uncertainty; actual owner retains lease and spend |
| Unknown cache/live slot, forwarding local endpoint, changed route | Refuse; ordinary loopback/new-conversation labels do not qualify |
| Summary grant instead of consolidation grant | Refuse; exact consolidation enrollment healthy control succeeds |
| Model-generated web/MCP/shell/child request | No such capability in closed extraction; refusal cannot be bypassed by output |
| Body-less ARCHIVE, first-link/outbox/export | Explicit local proposal custody enforced even without proposed_record |
| Batch conflict, edited approval, policy widening | Exact per-item checks; no stale apply or implied grant |
| Crash after commit before coordinator acknowledgement | Recover exact owner receipt once; no model replay/duplicate |
| Crash after dispatch before receipt, interrupted batch transaction | Review required/held accounting; zero partial proposals before commit, complete exact receipt after commit, no candidate-slot replay |
| Forgotten proposal/receipt during recovery | Native suppression wins; no resurrected source, proposal or secret locator |
| Restart/second process/window discontinuity/unknown usage | Same durable limits; uncertain reservations never age out or refund |
| Rejected lesson handoff | Foreground ephemeral ADR-106 contract; no hidden draft/outcome store |

## Release boundary and sequence

The first implementation remains disabled until **shipped and verified** ADR-185 evidence/current-source authority, ADR-186 retirement/suppression and owner recovery, ADR-187 destination/purpose/capture/log/cache disclosure controls, and this design's source-feed/receipt/budget contracts qualify through native entry points. Completed design tasks, checked boxes, generic scheduler support or saved credentials are not capabilities. Legacy unknown records/history/model state are excluded; no V1 inference or silent permission migration enables work.

Only new qualified V2 inputs, newly enrolled native direct-user source revisions, new local canonical proposals and an explicitly qualified owned on-device extraction adapter enter the first slice. No remote/offloaded/shared execution, general tools, background Notes/Agent Lessons writes, old-history mass backfill or scheduled hourly reflection. Feature enrollment is a foreground action separate from model/source consent. Runtime recovery/unlock cannot auto-enable it.

Later implementation must be broken into atomic native-owner tasks: versioned purpose/proposal custody and unique publication receipts; source enrollment/feed/checkpoint admission; durable narrow budget/recovery accounting; closed qualified extraction and foreground review; and synthetic/live local qualification. These are work boundaries, not yet-created task references or authorization to start them. The user endorsed the written contract subject to review; the requested follow-up resolved three native lifecycle gaps and its targeted reread found no remaining contradiction. [ADR-188](../../../backlog/decisions/188-opt-in-proposal-only-memory-consolidation.md) accepts design direction only; no runtime rollout or provider permission is approved. No worker, schedule, model call, schema migration, real-profile processing or data deletion occurred in this design task.
