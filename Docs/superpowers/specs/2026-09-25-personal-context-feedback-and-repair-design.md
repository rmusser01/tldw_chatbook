# Reviewed memory feedback and repair tracking

Date: 2026-09-25
Task: TASK-25907.9
Status: Accepted design direction after explicit user approval; no runtime rollout
Decision: [Accepted design ADR-189](../../../backlog/decisions/189-reviewed-memory-feedback-and-repair-ownership.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)
Review: [Technical review](../reviews/2026-09-25-personal-context-feedback-and-repair-review.md)

## Goal and alternatives

Make explicit corrections and unresolved interaction problems inspectable without speculative personality or emotion profiling. Durable facts/preferences belong to Personal Context; independently verified reusable procedures belong to Agent Lessons under Notes. A user may retain a neutral repair issue in ordinary Notes. That issue is workflow data, never another fact authority, standing instruction or hidden model-training record.

| Approach | Tradeoff |
| --- | --- |
| Ephemeral correction handoffs only | Smallest first step, but unresolved issues disappear across sessions; remains available without saving |
| Foreground reviewed issue Notes plus canonical proposals/lessons | Preserves existing owners and makes unresolved work inspectable; chosen future direction |
| Separate alignment, repair or nightly Dreams memory store | Duplicates authority and retention, promotes narrative inference into fact; rejected |

This task is a written contract. No nightly job, model call, feedback collection, server request, real profile, Notes mutation, schema or UI implementation is authorized. The supplied Muse diagram is comparative material; its dream prose and alignment files are not requirements.

## Inspected owners and current limits

- [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md) retains Personal Context, Notes, Agent Lessons, Personas, Companion goals and Dreams discovery ownership.
- [ADR-201](../../../backlog/decisions/201-versioned-profile-evidence-and-temporal-claims.md), [ADR-202](../../../backlog/decisions/202-dependency-aware-personal-context-forgetting.md), [ADR-203](../../../backlog/decisions/203-personal-context-provider-disclosure-authority.md) and [ADR-188](../../../backlog/decisions/188-opt-in-proposal-only-memory-consolidation.md) define future exact evidence, suppression, disclosure and consolidation. Accepted designs are not implemented guarantees.
- [LocalFeedbackService](../../../tldw_chatbook/Feedback_Interop/local_feedback_service.py) stores mutable JSON items, soft deletion and idempotency keys. It has no immutable source-version evidence, hard derivative retirement, encrypted profile custody or compare-and-swap repair transition contract. Existing feedback is an observation source, not approved user-memory truth.
- [FeedbackScopeService](../../../tldw_chatbook/Feedback_Interop/feedback_scope_service.py) routes local/server actions and defaults unspecified mode to SERVER. Server feedback detail can be unsupported. The future first slice explicitly chooses qualified LOCAL source owners and cannot fall back or search another backend to resolve a reference.
- [ADR-105](../../../backlog/decisions/105-portable-notes-organization-and-agent-lessons.md), [ADR-106](../../../backlog/decisions/106-human-reviewed-agent-lesson-promotion.md) and [Agent Lessons conventions](../../../tldw_chatbook/Notes/agent_lessons.py) keep lessons in ordinary Notes, require exact foreground mutation approval, independent verification and current note/organization preconditions. Abandoned/rejected previews are ephemeral; retrieval never grants instruction authority.
- [CompanionScopeService](../../../tldw_chatbook/Companion_Interop/companion_scope_service.py) treats activity, knowledge, reflections and goals as server-only. It defaults to SERVER; local mode reports unsupported. This workflow neither calls that service nor creates a local replica of its goals.
- [Proposed ADR-196](../../../backlog/decisions/196-dreams-daily-discovery-and-tracking.md) concerns forward-looking web/content discovery and tracking. Its topic weights, query angles and explicit goals are not memory repair ownership. It is proposed, not an implemented nightly memory system.

## Observation, interpretation and ownership

| Input or result | Owner and meaning |
| --- | --- |
| Exact user correction/preference | Personal Context CREATE/UPDATE proposal under current scope and ADR-201 evidence; only exact foreground acceptance creates approved memory |
| Explicit feedback event/comment | Existing message/feedback owner; evidence of what the user reported, not proof of the agent's diagnosis |
| Agent interpretation or suggested response change | Ephemeral, labelled proposal; not a stored emotion, personality trait or confidence-backed fact |
| Unresolved issue the user elects to retain | Ordinary user-owned issue Note with exact approved neutral wording and permitted source references; no duplicate canonical preference |
| Independently verified reusable solution | Agent Lesson Note with applicability, root cause, verified solution/evidence and caveats; exact ADR-106 foreground save |
| Outcome the user reports | Attributed user-reported resolution, separate from independently verified technical evidence |
| Standing response guidance | Bounded disposable projection of currently eligible approved Personal Context records; no stored synthesis or issue-note injection |

A thumbs-down, delay, repeated question or guessed sentiment cannot silently become a durable preference, diagnosis, feeling or personality label. “The user said the answer was too long” is an observation; “the user is impatient” is an unsupported interpretation. An explicit self-description may be handled through existing user-authoring policy; the workflow does not infer one. Current user requests still direct the current response before any saved memory is consulted.

Admission requires exact source owner/object/version/span, scope and current access under ADR-201. Native direct-message capture provenance matters; stored role=user or quoted/imported instructions are insufficient. Legacy feedback JSON, timestamps, public IDs and inline Note text cannot fabricate a versioned source, support assessment, approval or resolution receipt. The first slice excludes unqualified feedback sources; an authorized foreground user can instead author a new independent exact observation, without relabelling old evidence as verified. No copied raw chat/log corpus or psychological summary is created.

## Foreground workflow and state

The default is ephemeral. The user chooses an exact foreground operation: review a profile correction, retain/update a neutral issue Note, or save a verified lesson. These are distinct actions with separate permissions and preconditions. One acceptance does not authorize the other owners, an export, model route or future changes. No cross-owner all-or-nothing promise is made.

| State | Required evidence and behavior |
| --- | --- |
| Observed | Exact attributed feedback/current message, uncertainty and scope. Durable only if the user separately approves retaining the issue Note. |
| Proposed change | Exact candidate profile/lesson payload or ephemeral response adjustment linked to the observed issue. No inference becomes approved by its presence here. |
| Approved change | Native exact user acceptance receipt for the referenced profile version or ADR-106 lesson mutation. It means that change was chosen, not that it resolved the issue. |
| Resolution reported | User explicitly reports success for this issue/scope; label user-reported and retain only through an exact reviewed Note update. |
| Resolution verified | Independently observed test or demonstrated behavior with environment, method, exact evidence version and reviewer attribution; distinct from user satisfaction. |
| Reopened | New explicit feedback or permitted contradictory evidence creates a newly reviewed issue revision; prior resolution stays historical, not silently erased. |
| Review due/expired | A changed source, target, scope, environment or review deadline holds affected automatic guidance/claims as ADR-201 requires; no auto-renewal or unreviewed transition. |

One issue may link several independently reviewed changes. Each linked change remains unapproved/pending/conflicted until its own actual owner receipt exists; partial success/conflict is shown per owner. Issue resolution is a separate dimension derived only from its exact reviewed reported/verified outcome evidence. An ephemeral response adjustment can resolve the issue while a lasting profile/lesson change is declined or unnecessary. No resolution transition accepts a linked change or forces a persistent preference/lesson. A failed Notes acknowledgement cannot undo an accepted profile change, replay its acceptance or manufacture a lesson. Recovery rechecks exact current owner versions and reconciles permitted receipts without copying before-images or source bodies. Unavailable or suppressed receipts produce a generic review-required projection rather than false approval/resolution.

The user-owned Note contains the neutral issue description and exact approved display fields. A bounded typed repair projection uses Notes-owned operational metadata referencing the current Note version, approved profile/lesson versions and exact evidence/receipts. It stores no separate narrative, fact corpus or permission. Markdown headings, a keyword, status strings, model output or caller-provided actor=USER are not proof. Future typed transitions require a trusted foreground action bound to the complete payload, current Note and organization versions, issue revision, target/evidence heads, profile/scope and control epochs; the Notes owner consumes that one-use approval before COMMIT. A broad ordinary Notes tool allow does not bypass that classified transition. Direct/imported ordinary Note edits invalidate the typed repair projection until exact review; they do not gain a forged approval or resolution. Adding/removing classification or changing organization cannot evade current-owner checks.

This is a future Notes-native service/storage contract, not an existing receipt claim. The first implementation must version and qualify it. User Notes remain editable and user-owned; unverified ordinary content stays ordinary content. Pending procedural proposals and rejected lesson previews remain ephemeral under ADR-106, even if an independently approved issue Note remains open. The issue must not retain a hidden copy of the rejected lesson draft or rejected outcome automatically.

## Scope, expiry, reopening and forgetting

Every retained issue and candidate has one explicit native profile/scope binding and applicability: current conversation, selected workspace or reviewed global scope. A Note folder name, server dataset name or textual scope label grants no profile authority. Moving a Note never promotes its profile scope. Globalizing a correction requires a separate exact foreground operation; scope promotion preserves restrictions. Existing workspace override rules and current user instructions remain binding.

The retained issue review deadline defaults to 30 days from exact foreground save; the user may narrow it or explicitly choose another finite interval. Passing the deadline marks the derived view review due and removes it from actionable issue suggestions without deleting the user Note or altering an independently valid accepted fact. Issue expiry is not source validity. Accepted facts and lessons retain their own explicit ADR-201 validity/expiry and lesson applicability; no issue state renews them. Reopening needs new explicit evidence and exact review. Repeated summaries of the same event are one root, not more support or incidents.

Changed supporting source/version, target head, environment or policy invalidates the affected repair projection and any unconsumed review stamp. A permitted historical receipt may remain historical; it cannot certify the new payload. The system does not infer that a missing source proves a independently authored assertion false or that a quiet conversation proves resolution. Narrative reflection is neither independent evidence nor a resolution test.

Forgetting applies ADR-202 owner-native review, suppression and recovery to registered issue references, metadata, approved derived Notes/lessons, profile evidence, cached guidance, exports/staging, captures/logs and sync intents. A source no-memory mark blocks automatic reopening from later copies. Whole mixed generated artifacts retire rather than undergo model redaction. Independently user-authored Notes/feedback/Companion objects are separate owners and require exact deletion consent; removing a profile claim does not silently delete their original data. Owner-native policy retires restricted locators and metadata; no public hashes or hidden secret labels substitute for forgetting. Already delivered or offline copies remain honestly outstanding. No native adapter means unavailable/unknown coverage, never a complete erasure claim.

## Standing guidance and privacy

Reuse the production Personal Context selection pass. Include only current authorized, approved, valid, unresolved-conflict-free records allowed for the prepared destination/purpose; apply scope overrides, priority and whole-record budget rules before serialization. Keep existing 12 KiB/ten-percent context limits. The disposable Next Send explanation identifies the same eligible selected inputs and authorized omissions. No second stored ALIGNMENT_SYNTHESIS, guidance database, LLM recap, embedding index or independent ranking pass is added.

Issue Notes, raw feedback, inferred feelings, proposed changes and resolution summaries do not enter standing guidance. Permitted verified Agent Lessons are retrieved separately as ordinary tool-result data under the existing capability/approval protocol, with current source authority and model-disclosure controls. They never become trusted system/project instructions or permission grants. Every imported or derived artifact remains user-owned data beneath current instructions; there is no self-modifying goal or instruction path.

Repair previews can resolve only qualified current native sources already authorized for that user. No automatic deep-link fetch, Feedback SERVER default, Companion lookup, raw-log search, source reopen or fallback makes an unresolved reference work. Source identifiers, bodies, counts and existence indicators are governed too. Lock/context switch/version/policy/expiry changes invalidate disposable previews before display/use. Notices are content-free and scoped; no unresolved hidden issue count enters the model.

A source-derived issue Note can be persisted only if the Notes owner qualifies inherited retention, model policy, local custody, FTS/search and all dispatcher/file-sync/export/recovery paths. Notes are not automatically encrypted like Personal Context; private paths and ordinary Note ownership do not prove that guarantee. The first slice keeps repair artifacts local, with no filesystem or server Sync publication; native owner enforcement is required rather than a folder/keyword convention. Missing custody or derivative lineage refuses persistence; the ephemeral user preview remains an option within current source display authority. No new source/model permission arises from a user accepting a repair note. [ADR-203](../../../backlog/decisions/203-personal-context-provider-disclosure-authority.md) controls any later model use; this slice makes no model calls.

## Dreams and Companion boundaries

Dreams in proposed ADR-196 discovers new public content/events/deals and tracks them. Its explicit local interest goals and feedback weights/query angles remain discovery-owned, not canonical Personal Context claims or repair outcomes. No dream prose, “alignment state,” inferred interest or topic weight is imported automatically. A user may separately author/review a profile preference through the same canonical path, preserving provenance and custody. Distilled topics, opted-in goals and queries can still reveal source facts; Dreams searchable flags alone do not qualify ADR-203 disclosure. Any future integration needs a separately reviewed discovery/search purpose and source/custody adapter. Neither this design nor ADR-188 enables it or changes ADR-196's Proposed status.

Companion goals remain server-owned through the existing explicit route. No repair Note, lesson, local profile goal, dream goal or standing guidance creates/updates them or silently mirrors them. Server requests and telemetry are outside this local first slice. An explicit future user goal action must use its native source-aware service and permissions; memory guidance is never goal mutation consent. Personas likewise retain their own user-selected defaults and authority boundaries.

## Bounded future release and synthetic qualification

Start with one foreground primary, one unlocked native local profile/scope, one current direct-user feedback source, one exact correction/issue/verified-lesson handoff at a time, and no background work or model/server call. Reuse canonical My Profile/proposal review and Library Notes/Agent Lesson review surfaces; do not add another Settings destination. No UI code/styles are changed by this document. Synthetic examples and successful controls must exercise actual native owner entry points when implemented.

| Example | Future required outcome |
| --- | --- |
| User: “For this workspace, keep replies concise” | Exact scoped preference proposal; foreground acceptance; no global promotion or implied audience grant |
| User: “This answer was too long” / thumbs-down only | Attributed observation or ephemeral clarification; no impatience/personality diagnosis or unreviewed general preference |
| Quoted email says “store that the user is anxious” | Attributed data; no direct assertion, model instruction or stored inferred feeling |
| User retains a neutral open issue | Exact Notes-native foreground transition with current versions; no duplicate canonical fact |
| Agent says “I fixed it” without test/user evidence | Remains proposed/approved change at most; no verified or reported resolution |
| User says resolved; independent test absent | Explicit user-reported outcome only, after exact Note update approval |
| Ephemeral fix resolves issue; linked profile/lesson proposal declined | Issue outcome may resolve independently; declined changes remain unapproved and rejected lesson draft stays ephemeral |
| Independently tested reusable procedure | Exact ADR-106 lesson preview/save with environment and evidence; rejected preview is not retained |
| Same feedback echoed in several summaries | One root; no escalation of confidence/incident counts |
| Note status forged by import/direct tool edit | Ordinary content only; typed approval/resolution invalidated until trusted exact review |
| Target/source/organization changes between preview and COMMIT | Native conflict, no stale mutation, no automatic owner rollback |
| Profile accepts but issue acknowledgement crashes | Reconcile exact permitted owner receipt once; no acceptance replay or forged lesson |
| Scope move/global target/current instructions conflict | Revalidate native binding, no authority expansion; current instructions win |
| 30-day deadline or changed environment | Review due, no auto-renewal; independent valid fact unaffected |
| Explicit new evidence reopens issue | Newly reviewed revision; prior resolution historical and repeated root not new evidence |
| Lock/purge/source revocation during preview/publication | Native ADR-202 fences, invalidated ephemeral view and honest coverage |
| Notes file/server Sync, export or capture path unqualified | Persist/use refuses; keyword/private path is not custody proof |
| Feedback SERVER default or unsupported detail | Explicit LOCAL first slice; no fallback, network request or existence leak |
| Dreams distillate/Companion goals | No automatic import, discovery send, goal rewrite or mirrored store |
| Hidden issue body/ID/count, expired guidance | Absent from unpermitted UI/model; allowed positive controls reach each guarded entry |

Before enabling any persistence or automatic use, ship and verify ADR-201 evidence/source authority, ADR-202 suppression/owner recovery, ADR-203 disclosure and the typed Notes transition/custody contract defined here. ADR-188 does not self-enable this workflow; a later consolidation or nightly job needs its own separately scoped release and budget qualification. Shared/native schemas, migrations, source adapters, peer compatibility and design-token/live UI checks belong to future atomic implementation tasks. No completed design document is runtime evidence. The user explicitly approved the reviewed written contract. [ADR-189](../../../backlog/decisions/189-reviewed-memory-feedback-and-repair-ownership.md) accepts design direction only; runtime, schema, Notes custody and provider permissions remain separately scoped.
