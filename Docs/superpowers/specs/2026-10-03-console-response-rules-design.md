# Console learned response rules

Date: 2026-10-03
Status: Draft for written-spec review; conversational reviews and written-spec audit incorporated
ADR: [ADR-219](../../../backlog/decisions/219-console-learned-response-rules.md) (Proposed)
Related Backlog task: N/A; no implementation task has been created or authorized by this design-only request.
Implementation plan: Not yet written; written-spec approval precedes implementation planning.

## 1. Purpose and agreed intent

Let a user turn a complaint about a Console agent's answer into a native response rule through `/omfg <problem>`. Test whether the rule catches the reported mistake, activate a successful rule for the current Chat, and give the agent specific feedback when later completed answers violate it.

The agreed first version uses native structured rules, completed-response checks, and corrections continuing from existing work. Saved Chat rules survive reopening on this device; temporary Chat rules live in memory. Promotion to a Workspace or the current profile's global scope is an explicit user action. Streaming content remains visible, failed answers remain in history, and corrections appear as follow-ups. A passing check means only that the active rules passed.

Response checks cannot prevent or undo an already executed tool action. A complaint requiring pre-dispatch protection is reported as unsupported by this response-rule feature and leaves its candidate inactive. Behavioral claims in an answer can nevertheless be assessed using permitted execution evidence.

## 2. Reviewed development basis

The initial review inspected local `dev` at `01a2020981c6197e5cd9945e5287567ad977edfe` and remote `dev` at `0001eba40419859ce39ed4952f0f8df7b40639bd`. A refreshed remote review at `81c7c94f48` additionally examined the backup/admission changes; those do not change the response or queue contracts below.

Existing owners to extend rather than duplicate:

- `Chat/console_command_grammar.py`, command suggestions/help, and `UI/Console_Modules/` own native composer commands and UI delegation.
- `Chat/console_runtime.py` owns application-lifetime Console services, including viewless execution. New UI callbacks must be declared in `CONSOLE_VIEW_HOOK_SLOTS` as well as screen wiring.
- `Chat/console_provider_gateway.py` owns `AuxiliaryCompletionRequest` and `complete_auxiliary`: pinned, sensitive, tool-free provider calls with returned usage and no normal Chat history or capture.
- `Chat/console_chat_controller.py` owns normal send/continue/retry/regenerate execution. Regenerate forks a sibling; learned-rule repairs must instead continue after existing work.
- `Chat/console_prompt_queue_coordinator.py` owns continuation admission. Its current API consumes an owned Stop event/outcome; native assessments cannot fabricate that provenance.
- `Agents/hooks_v2/` owns existing hook effects, requirements and resource limits. Stop is not a required response-content guard and currently exposes status rather than an assistant body.
- The private profile's existing SQLite owner and Backup/Recovery participants own durable data and restore admission. Console session settings metadata is not a local-only rule store.

Applicable governance: [ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md), [ADR-092](../../../backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md), [ADR-126](../../../backlog/decisions/126-complete-local-backup-and-recovery.md), [ADR-163](../../../backlog/decisions/163-expanded-console-hook-runtime.md), [ADR-197](../../../backlog/decisions/197-console-hook-configuration-review.md), and [ADR-210](../../../backlog/decisions/210-console-region-ownership.md). UI composition follows [design-language.md](../../../backlog/docs/design-language.md) and [component-patterns.md](../../../backlog/docs/component-patterns.md).

## 3. User workflow

1. `/omfg <problem>` captures the latest eligible, completed user-facing text answer on this Chat's active branch and its originating request. Empty arguments show usage without submitting an ordinary agent prompt. No eligible answer produces an honest refusal. During generation, rule drafting, assessment or repair on this Chat, the command refuses and preserves the draft; the user can wait or Stop.
2. Display drafting/testing progress, with cancellation available through the existing visible Stop action. Draft at most three candidates. Each candidate must identify applicability, a supported detector and corrective feedback.
3. Test against the actual reported response, a synthetic paraphrased violation, a synthetic correction and a relevant synthetic acceptable example. Retain the distinction between recorded and synthetic evidence. The detector must catch both violation cases and accept both controls. Unsupported, unverifiable, stale or exhausted candidates remain inactive and inspectable.
4. A successful, still-current candidate activates automatically for this Chat after durable commit, or in memory for a temporary Chat. This foreground `/omfg` action authorizes this bounded native learning operation; no executable hook consent is manufactured.
5. Show a compact summary of applicability, detection and correction, labelled **Tested against examples**. Start the initial correction as a new user-authorized operation anchored to the original exchange, retaining its answer and completed tool results.
6. Inspect subsequent completed text answers under the effective rule set. Confirmed violations produce one combined correction request; pass and couldn't-verify outcomes do not create repair turns.
7. `/rules` opens the same manager as the visible **Response rules** entry in Chat settings. Users inspect definitions, evidence and outcomes, edit/test candidates, disable, delete or explicitly promote rules. No new permanent composer button is required.

The builder does not automatically rewrite an already active rule after a later violation or false positive. A user edit or a new learning request owns such changes.

## 4. Components and interfaces

### 4.1 Rule store

One profile-owned service provides immutable revisions, local scope bindings, source-linked validation evidence and assessment/correction records. Mutations use optimistic revision checks and ordinary SQLite transactions; no transaction spans a model call. Public rule data cannot supply executable code, authority tokens, event identities or permission decisions.

The logical data contracts are:

| Record | Required content |
| --- | --- |
| Rule revision | Stable logical rule ID, immutable revision, title, applicability, detector kind and criteria, corrective feedback, schema version, origin and creation timestamp. |
| Scope binding | Chat/Workspace/global identity, exact rule revision, enabled or exclusion state, binding revision and activation provenance. |
| Validation | Exact candidate digest, detector/model identity, case types and results, source message/version references, fixture ownership and tested timestamp. |
| Assessment | Host assessment ID, accepted parent turn and message/version, effective-rule digest, evidence completeness, per-rule results and lifecycle state. |
| Correction receipt | Host operation/chain identity, parent assessment, accepted follow-up identity, correction count, source attribution and durable acceptance state. |

Rule state (draft, testing, active, disabled) is separate from response verdict and execution status. Deletion is a store operation with source cleanup and pending-work invalidation, not a model-authored state transition.

The host assigns all rule/revision, assessment, operation and settlement identities. The shared admission receipt has a uniqueness constraint on `(operation_id, parent_turn_id, settlement_id)` across native and hook contributors. Retried callbacks cannot admit a second follow-up; a separately initiated manual learning request has a fresh operation identity, not a reused historical admission token.

Assign and retain the settlement identity once for its accepted parent operation, rather than minting it on each callback. Follow-up acceptance commits the user-role feedback, assistant owner, ordinary dispatch checkpoint and shared receipt in the existing acceptance transaction. A conflicting receipt rolls the competing acceptance back. Assessment records alone cannot authorize dispatch or replace the existing uncertain-dispatch recovery owner.

### 4.2 Builder

The builder takes an immutable learning snapshot: original request, eligible response, permitted evidence, current provider resolution, source versions and store/scope revisions. It returns a validated candidate or a bounded failure. The runtime owns acceptance and activation; a model result cannot activate itself.

Use the existing auxiliary gateway for drafting and model-based validation. Helper requests contain quoted, labelled data rather than injecting the complaint, response or rule as trusted system authority. Helper output is schema-validated and never exposes tools. A cancelled or stale request cannot activate a late candidate.

Tests against generated controls demonstrate discrimination on examples, not proven generalization. The UI and stored provenance must not describe synthetic corrected examples as actual successfully executed work.

Model-based validation receives opaque case IDs and permitted case inputs, without expected verdicts or labels such as "bad response" or "corrected example". The host compares returned classifications with its separately retained expectations. This prevents the validator from passing by repeating supplied answers.

All calibration cases use the same actual permitted execution evidence; synthetic text cannot add invented successful tool results. An acceptable synthetic correction may acknowledge missing work without claiming it was performed. The builder also validates that applicability and corrective feedback address the complaint within the original task, rather than introducing unrelated actions. Snapshot content is explicitly allowlisted: provider resolution, credentials, authority handles and internal project-instruction bodies are host-only and never serialized into model input, fixtures or UI previews.

### 4.3 Evaluator

The evaluator takes a frozen response snapshot and exact effective rule revisions. Supported deterministic predicates are literal inclusion, literal exclusion and required Markdown headings, with explicit case handling. No generated expressions, scripts, arbitrary regex or dynamic imports are accepted. Unsupported deterministic proposals may be drafted as semantic criteria only when response-time evaluation is appropriate.

Semantic applicability and semantic criteria share one bounded model batch where possible. A deterministic result requiring semantic applicability remains pending until that applicability is established. A confidently inapplicable rule is skipped; uncertain applicability is couldn't verify.

Results use a closed shape identifying each exact rule revision, pass/violation/couldn't-verify verdict, bounded explanation and references to supplied evidence. Missing, duplicated, unknown or conflicting entries invalidate the affected batch. Output excerpts must match the supplied source; invented evidence references cannot establish a verdict.

Applicability is separately `applicable`, `inapplicable` or `unknown`. An applicable result has one of the three verdicts; an inapplicable result has no verdict and is skipped; unknown applicability requires couldn't verify. Aggregate a response as violation if any valid check confirms one, retaining other unavailable checks; otherwise use couldn't verify if any check is unavailable, pass if at least one applicable check passed and none failed, or **No applicable rules** if every rule was skipped. Never display skipped or unavailable checks as passes. Valid deterministic results survive a failed semantic batch, but a detector with unknown semantic applicability cannot establish a violation by itself.

Permission-gated evidence comes from actual dispatch/result owners, including settled/not-started/uncertain provenance. Evidence snapshots carry a host-owned completeness flag. Truncation, unavailable private material, missing stage contributions or uncertain execution cannot be interpreted as proof of absence or success. For example, a test-passing claim cannot be accepted solely because the answer or another model says tests passed.

Include relevant settled evidence from the original task and its correction chain on the current branch, not only tools executed in the latest reply. Match a claim to what ran, its outcome and its known task/work revision. A later mutation cannot turn an earlier passing test into evidence that the new state passed. If freshness cannot be established, an unqualified current-state claim is couldn't verify; an accurately qualified historical claim can still pass. This evidence projection introduces no new retrieval authority or judge tool calls.

### 4.4 Correction coordinator

The Console queue owner accepts a distinct typed native correction proposal backed by a current host assessment. It extends its existing admission contract without allowing native callers to forge a HookResult, Stop event or permission grant. It continues to own serialization, current authority, budgets, durable custody and uncertain-dispatch recovery.

Manual `/omfg` repair is a fresh user-authorized operation using current eligible configuration and permissions. It references the historical exchange without resurrecting its retired run, expired budget or old Stop event. Automatically triggered repairs instead inherit the current accepted turn's remaining limits. Both preserve existing tool results and retain normal permission review for any additional action.

The host feedback wrapper identifies the complaint and rule guidance as bounded repair context for the existing user task, not authority for a new task. Do not enable agent/tool mode, broaden scope or bypass an approval because a generated rule requests it. When the current mode cannot finish missing work, retain an honest unresolved outcome or correct the answer's claim.

### 4.5 Auxiliary request resource ownership

Builder and evaluator calls use one runtime-owned request lease through the existing gateway and provider-work machinery. Caller cancellation and a logical deadline retire acceptance immediately, but do not imply that a synchronous provider thread or transport has stopped. Retain the actual worker/transport completion handle, its source identity and capacity reservation until physical settlement; a cancelled asyncio task alone is not evidence of cleanup.

Bound unsettled native-rule helper work to four requests application-wide and one per Chat, subject to tighter existing provider limits. Do not allocate another provider thread pool. Capacity exhaustion leaves a candidate inactive or a response couldn't verify without blocking ordinary user sends. A cancelled call keeps its reservation and cannot be replaced by another helper for that Chat until actual settlement. Apply finite transport timeouts bounded by the remaining helper allowance, with zero adapter retries for these calls; only the host's explicit candidate-attempt loop may retry learning.

Late content cannot commit an assessment, activate a rule or schedule repair. The source operation's retained usage-accounting owner may record actual late usage exactly once, independently of retired content-acceptance authority; it must never land in a newly selected Chat/profile or be charged twice. Shutdown seals acceptance before bounded cleanup and retains unresolved-work accounting rather than claiming the thread was killed.

## 5. Completed-response lifecycle

Execution and assessment are independent:

1. Normal execution settles required postevent gates and tool results, then saves the completed assistant answer through its existing owners. Only a successfully settled eligible text response enters assessment. A save or controlling gate failure follows ordinary recovery; it is not repaired by the rule subsystem.
2. Before releasing final completion signals or admitting the next queued turn, the runtime captures the answer, source versions, evidence and effective rules. An eligible turn with no effective rules follows its existing path with no auxiliary request.
3. Mark the assessment pending and project **Checking rules**. Keep the visible Stop action live. Run checks outside SQLite transactions and blocking mutation locks.
4. Commit a current assessment or mark it couldn't verify/cancelled/stale. Preserve the generated answer regardless of checker availability. Do not convert a checker failure into provider failure or falsely report a pass.
5. Resolve existing Stop hooks at their existing owning settlement boundary. Aggregate valid hook proposals and native correction feedback before making one machine-follow-up admission decision for this settlement. Preserve their separate provenance within the combined request.
6. Existing continuation vetoes, controlling admission failures, required postevent gates, current-authority checks, cancellation, closure and user priority remain effective. Native feedback cannot bypass a refusal that controls this shared automatic-work decision.
7. Admit at most one follow-up for this settlement, or retain an unresolved violation and return control. Emit final user-visible completion once the assessment and existing owned settlement work finish, with its assessment qualification.

Rule checks cover normal Send, queued/background Console text sends, Retry, Continue and Regenerate through shared runtime paths. They do not inspect auxiliary drafting/judging, individual tool messages, partial/cancelled/error streams, subagent-internal responses or media-generation output. All eligibility and helper-origin distinctions are host-owned, not model-provided labels. Switching the viewed Chat does not redirect an assessment to another Chat or cancel otherwise authorized background work.

## 6. Correction limits and priority

- At most two native correction turns per operation/chain, counting the initial `/omfg` repair. The rule set shares this limit; it is not two attempts per rule.
- Native repair also shares the existing maximum of three admitted continuation turns and 120 elapsed seconds with any hook-created work in that chain. Other current provider/tool/automatic-work limits can stop it earlier.
- All confirmed violations in a response form one feedback packet. Contradictory rules cannot start competing recursive loops; unresolved conflicting feedback is reported when bounded repair cannot satisfy it.
- A combined native/hook follow-up consumes one shared continuation turn and, when it includes native repair, one native correction turn. Hook-only work consumes the shared count without resetting or incrementing the native count. Persist inherited counts with acceptance; new assessment IDs or rule revisions cannot reset the chain's limits.
- Preserve existing whole-message limits: native feedback is at most 4 KiB UTF-8 and the combined hook/native body at most 8 KiB, including attribution wrappers. Overflow refuses that machine follow-up with a visible limit outcome. Do not drop rules, split it into extra turns or silently truncate instructions; prior task context remains in normal history rather than being duplicated into this packet.
- A corrected answer is checked again under a current rule snapshot. Only a confirmed further violation can consume another native correction turn. Couldn't-verify results do not justify another automatic attempt.
- Foreground input takes priority. A newly arrived user prompt fences a pending automatic repair; it is not retained behind the user's work. If evaluation is cancelled to release the Chat, the assessment stays honestly incomplete.
- If user work is already waiting before automatic model-check admission, do not start that helper call. Retain any completed deterministic results, record the deferred/cancelled semantic assessment honestly and release the Chat through the existing queue owner.
- Stop cancels learning/checking and pending repair admission promptly. Cleanup remains owned until resource settlement; a late callback cannot restart work. Closing the Chat or changing its branch similarly invalidates affected proposals.
- A pending assessment is bound to Chat, branch/source response versions, effective rule revisions and execution authority. Revalidate these at result commit and actual follow-up admission. Disabling or deleting a rule invalidates affected work immediately, even if persistence later fails.

## 7. Scope, editing and promotion

Global means the current local profile, not all application accounts or devices. Rules add response preferences and feedback; they never expand tools, file access, approval exemptions or permission profiles.

Resolve bindings for each logical rule with explicit precedence: Chat, then current Workspace, then profile-global. The selected binding pins one immutable revision. A Chat exclusion masks that logical rule, including later inherited revisions, until explicitly removed; **Disable here** and **Disable at source** are distinct actions. Exact duplicate bindings do not multiply checks.

Editing detection criteria or applicability creates an inactive candidate revision, which must pass validation before the selected binding changes. Existing bindings keep their pinned revision until that change is explicitly committed. Editing feedback alone still creates a new reviewed revision and invalidates pending work using the old binding, but may reuse unchanged detector validation with explicit provenance.

Reusing detector validation requires an identical detector/applicability digest, evaluator protocol/version and available validation provenance. Corrective feedback is excluded from judge input so a feedback edit cannot secretly change the detector. The new feedback still receives a complaint/task-scope check and explicit user review in the editor; missing original evidence prevents automatic reuse and requires new examples.

Promotion previews the exact definition, current applicability and destination scope, and creates a binding pinned to that reviewed revision. It does not silently generalize the condition. Private validation fixtures and source Chat/message bodies are not copied into the broader scope. The promoted rule can retain body-free validation provenance; that does not imply it was tested in every newly eligible Chat.

A separate Chat fork starts without source Chat-specific bindings, calibration evidence or pending corrections. Applicable Workspace/global bindings are resolved normally and displayed. Branches inside the same Chat retain Chat-level rules. Conversation movement between Workspaces recomputes the effective set and invalidates stale assessments.

## 8. Persistence, privacy and recovery

Persist definitions, bindings, evidence links, assessments and correction receipts as local-only sidecar records under the existing private profile SQLite owner. Add a real schema migration and increment the then-current schema version; version 75 is the reviewed basis, not a reserved next version. These records are outside conversation metadata, sync outboxes, server payloads, normal Chat export and handoff manifests.

Durable activation succeeds only after its transaction commits. A failed activation stays inactive with a recoverable draft and truthful save error. Temporary Chat state is memory-owned. Saving a temporary Chat remaps rule/source identities and commits durable rule state consistently with normal Chat adoption; failure leaves its live temporary state intact without claiming durable activation.

Soft-deleted Chats retain disabled, inaccessible Chat records for normal undo. Permanent Chat deletion removes its bindings and private source/calibration data, while independently promoted definitions remain. Source message deletion removes its private evidence; surviving rules display that their original evidence is unavailable rather than retaining hidden copies. Expired/orphaned references cannot break normal Chat browsing.

Explicit Chat undo may restore the prior binding state only when the referenced definition remains eligible and unchanged; it never restores pending assessment or correction admission.

Normal reopen on the same admitted profile restores committed active definitions and historical outcomes. Startup does not launch pending checks or corrections. Durable acceptance receipts and existing uncertain-dispatch recovery determine whether work was accepted; a lost response never authorizes automatic replay of model or tools.

Register the new local owner with existing Backup/Recovery admission, drain, backup and restore transformations. A backup may retain operational definitions and historical records; restoration imports them inactive, with no live activation, assessment lease, correction admission token or executable authority. Exact internal rollback retains the pre-operation owner state under existing recovery rules rather than pretending to be a new import. Do not invent drive-number/inode identity as a device authorization shortcut.

Stored complaints, fixtures and user-viewable feedback are private content. Ordinary logs and metadata diagnostics contain identifiers, counts, outcome/reason codes and timing only. Promotion requires inspection of the definition itself because its criteria or feedback may contain private terms even when fixtures are omitted. Provider evidence disclosure uses the existing permitted projection and privacy boundaries.

## 9. Bounds, models and usage

Initial host-owned defaults:

| Limit | Default |
| --- | --- |
| Candidate attempts | 3; one draft call and at most one model-validation batch per attempt |
| Whole learning operation | 120 seconds; each helper call at most 30 seconds, shortened by remaining allowance |
| Completed-response assessment | 30 seconds total, shortened by remaining inherited limits |
| Effective rules | At most 16 per response after scope resolution |
| Model assessment batches | At most one per response; all applicable semantic criteria must fit |
| Rule definition | At most 8 KiB serialized; explicit bounded fields and shallow predicate shape |
| Helper output | At most 4,096 tokens, further limited by gateway/model/current automatic-work caps |
| Unsettled native-rule helper work | 4 application-wide; 1 per Chat, retained through physical cleanup |
| Native/combined follow-up body | 4 KiB / 8 KiB UTF-8 including attribution; refuse overflow |
| Native correction turns | 2 per chain, within the existing shared continuation limits |

Activation/promotion reports a capacity refusal before committing a change that would exceed a known target Chat's limit. If later scope combination exceeds the limit, the response assessment reports couldn't verify; it never silently checks only a prefix. Criteria, fixtures and evidence are never silently shortened into a different rule or a supposed complete evidence set. A batch that cannot fit the pinned context/output allowance is couldn't verify.

Within the definition cap, titles are at most 120 characters; semantic applicability is at most 1,024 UTF-8 bytes; criteria and feedback are each at most 2,048 UTF-8 bytes. A deterministic predicate has one supported operation, at most 16 nonempty literal strings or eight headings, and an explicit boolean case flag; each literal/heading is at most 256 UTF-8 bytes. Unknown fields and nested expression trees are rejected. Each synthetic fixture is at most 8 KiB. The recorded response is referenced at its exact immutable version and must fit the captured context allowance without pretending a shortened body is the complete original.

Use the current eligible Chat provider/model for new learning and the accepted turn's eligible resolution for automatic checking. No hidden judge-model substitution or provider failover is introduced. Unsupported native structured output may use ordinary auxiliary text followed by strict host parsing; malformed output still fails validation. Changing configuration during a call requires current-authority revalidation before accepting its result.

Feed actual returned auxiliary usage into existing Chat/profile usage accounting with distinct learning/checking attribution. Unknown provider usage remains unknown rather than zero. Helpers do not become conversation messages or leak their bodies through tracing. Deterministic-only, inapplicable and no-rule responses make no unnecessary model call.

## 10. Native UI contract

Use existing token-backed status, callout, dialog, form and action patterns. No new visual literals or feature-local token inventions; any needed token is added centrally. Edit stylesheet source modules and rebuild the generated bundle only during implementation.

Composer commands register once in grammar/help/suggestions. The rules manager is reachable through `/rules` and Chat settings and shows effective scope, inherited/excluded status, exact revision, applicability, detector and feedback. Workspace/global management reuses this manager through canonical Settings scope entry points, not deprecated Settings parallels.

Compact transcript/runtime states are **Drafting rule**, **Testing rule**, **Checking rules**, **Rule violation**, **Couldn't verify**, and **Correction limit reached**. A failed answer remains readable; a correction is a visibly attributed follow-up. The violation detail offers **Disable here** and **Manage rules**. Failed drafts remain inspectable. Passing outcomes say **Passed active rules**, without claiming universal correctness. All-inapplicable assessments say **No applicable rules**; mixed results expose unavailable checks alongside confirmed violations. Resource/capacity or feedback-size limits show their specific reason rather than a generic failed provider message.

Stop and new user input must remain reachable at narrow terminal widths during every helper/assessment/repair phase. Do not add a terminal-convention keybinding or advertise an unimplemented action. Switching or detaching a view does not strand runtime work or its cancellation controls.

## 11. Required verification

Use targeted verification only unless the user explicitly requests a full sweep.

1. Drive `/omfg` through the actual composer grammar/dispatcher with eligible, missing, incomplete, active-run, empty and cancelled cases. Verify suggestions and help resolve to the implemented action.
2. Exercise public builder/evaluator boundaries with independent labelled cases: original mistake, paraphrase, acceptable correction, unrelated acceptable response, quotation false positives, unsupported tool-time complaints and injection-like data. Separate scripted transport correctness from model-behavior evaluation.
3. Prove all three verdicts, inapplicability and mixed-result aggregation, strict result shape, unsupported structured output, missing/truncated/uncertain evidence and actual evidence-backed claims. Fully deterministic detector/applicability and no-rule controls must avoid auxiliary calls; semantic applicability uses its bounded batch. Include reuse of prior settled evidence, stale passing tests after a mutation, and synthetic examples unable to invent tool success.
4. Use real SQLite for activation/version races, write failure, migration/reopen, temporary-to-saved adoption, scope precedence/exclusion, promotion, source deletion, permanent Chat deletion, restore-inactive behavior and body-free ordinary diagnostics.
5. Use the real queue admission/custody path with native feedback plus existing hook proposals, vetoes, pending requirements, source limits and no-hooks configuration. Prove one follow-up and correct source attribution; no fabricated hook provenance.
6. Hold evaluation deterministically while sending user input, pressing the actual Stop button, disabling/editing a rule, changing Workspace/branch, deleting the source or closing/detaching the view. Prove stale results cannot commit or schedule repair and cleanup retains ownership.
7. Cover Send, Retry, Continue, Regenerate and queued/viewless Console text execution with the same checks. Verify failed/partial/helper/subagent/media results do not enter the assessment path.
8. Prove corrections retain prior tool results and use normal permission review for additional actions without expanding the original task or enabling agent mode, share two native attempts and existing chain caps across mixed native/hook turns, refuse packet overflow and do not replay after cancellation, restart or ambiguous acceptance. Fail the acceptance transaction after feedback insertion to prove no orphan assistant/checkpoint/receipt is committed.
9. Verify actual helper usage accounting, context/output bounds and unknown-usage reporting. Hold a real provider worker beyond caller cancellation, verify its reservation remains occupied, capacity cannot grow with repeated cancellation, transport retries remain zero, and late results are inert while usage is charged at most once to the source owner. Inspect ordinary logs and model inputs for private host-only payloads using controlled fixtures.
10. Mount the real Console and use actual commands/actions at 80x24 and 120x35 under token-backed styles, including Stop after generation while a check is pending. Run callback slot-set, design-token and generated-CSS governance checks for touched UI work. Run authored-file lint/format/static checks and documentation link/whitespace checks.

No application tests or live-provider evidence are claimed by this documentation-only spec.

## 12. Scope and next gate

The first version does not add executable/prompt hooks, prevention of tool effects, streaming interruption, an Agent Lessons mutation, instruction-file promotion, cross-device activation, autonomous active-rule rewriting, a new scheduler or a separate judge-provider setting.

ADR required: yes

ADR path: `backlog/decisions/219-console-learned-response-rules.md`

Reason: New local-only storage and recovery behavior, a response-assessment lifecycle boundary, scope semantics, and a native correction interface shared with existing hook continuations.

This written spec and Proposed ADR require user review before `writing-plans`. An implementation plan and its execution method must then be reviewed before product code or schema changes. No implementation tasks are referenced before their creation.
