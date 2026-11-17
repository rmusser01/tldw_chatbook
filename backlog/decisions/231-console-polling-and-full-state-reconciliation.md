# ADR-231: Console polling and full-state reconciliation

Date: 2026-10-08
Status: Proposed; source review only. No product implementation selected or authorized by this document.
Task: [TASK-34563.33](../tasks/task-34563.33%20-%20Plan-Console-polling-and-full-state-reconciliation.md)
Plan: [Polling and reconciliation](../../Docs/superpowers/plans/2026-10-08-console-polling-reconciliation-plan.md)
Extends: ADR-126 and [ADR-230](230-console-send-preparation-and-io-ownership.md); retains ADR-097 budgets.

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

The manual action independently performs checked capture and submits its completed request configuration. Runtime wake and compatibility producers can use a cached controller runtime gate, and missing-settings selection can use cached provider fields; they therefore remain FULL. This concrete action boundary replaces a guessed general selection-equivalence signature for the first implementation. ADR126 native authority/retirement, ADR230 custody/fresh action capture and the original performance budgets remain unchanged.

## Renumbering and current integration

Original source decision at c504ff3b1bda is preserved above with its original approval/Proposed status. The older current-dev ADR222 HTTP-session and ADR223 embedding-cache decisions keep their identities; younger Send-preparation222 and polling223 become230 and231 after the all-ref/worktree census. Existing34406/34563 child task IDs and historical qualification remain. The general polling proposal stays Proposed; only its explicitly accepted sole-stock-manual-Preparing amendment is selected. Current integration composes resident acknowledgement/captured-draft/custody/human-decision/Goal/Personal Context/response-rule owners, not a parallel ledger, cache or scheduler. Actual current native, original callback, final saved-source and rendered-input/transport evidence must qualify the original bounds before completion; old source counts or server-frame emission do not establish that proof.


## Current-owner integration amendment (2026-10-13)

The accepted narrow route excludes Goal, automatic/machine/hook followups, Rules Checking/repair, pending helper cleanup, and all non-manual producers. Explicit False from FULL means deferred reconciliation: no timer stop or survivor handoff. Completed-current FULL, exact unpromoted manual receipt/task/claim/draft/attachments/visit/membership/gate/bridge, post-await checks and trailing FULL dominance remain required. This amendment selects no general polling cache.


### PR3050 reconciliation boundary — 2026-10-31

Retain the current configuration projection cadence, explicit FULL request
accounting, trailing FULL dominance and durable-commit maintenance hold. The
older separate two-second light-poll implementation is not replayed over those
owners. Pressed fallback custody starts the existing transcript poll only for its
original attached generation; a failed presentation handoff cannot undo already
accepted custody. This reconciliation does not qualify the broader polling or
whole-path latency proposal.

### Promoted manual preparation display amendment — 2026-11-15

Selected for PR3534 under the owner's resumed latency work and six-second merge
gate; implementation and native qualification remain pending. The wider polling
proposal remains Proposed. A stock full-app Windows Send takes38.259/55.197/
54.747s with the original coordinator; diagnostic periodic deferral measures
16.367/19.647/15.186s on the same source. Host variation prevents a precise saving
claim, but the comparison justifies reducing repeated display I/O on this route.

Keep the accepted unpromoted branch intact. A promoted manual branch may narrow
only while the sole original custody task still owns the current preparation
and its own existing SendStart has not ended at provider entry or completion.
Use the controller's task-to-preparation map and the store's current preparation;
promotion replaces the received claim and successful save changes draft revisions.
Do not weaken the received-claim predicate or invent revision offsets. The
completed-FULL witness must cover relevant resident input/policy revisions:
changed inputs require one original FULL pass before later unchanged polls narrow.
Do not extend the separate three-second synchronous display-I/O exception.

Within existing sync exclusion, retain speech/H3 reconciliation, draft display,
current tabs, original transcript publication/acknowledgement and resident Send
feedback. On this qualified promoted branch, defer routine retrieval-summary,
character-browser, readiness/config and rail/inspector refreshes. Transcript
context presentation may reuse an already available same-owner display without
scheduling refresh; unavailable or changed ownership restores FULL. This evidence
is display-only and cannot authorize capture, persistence or provider dispatch.
Mark work owed using the existing cadence and return deferred so polling remains
live. No new scheduler, cache, authority token or preparation owner is introduced.

Recheck qualification after retained awaits. Pending FULL, changed settings,
runtime/Canvas publication, roleplay persistence/identity changes, human decisions,
other active sessions, custom callbacks, cancellation and terminal settlement
retain FULL. Explicit FULL dominates and retains its trailing replay. Preserve
the original final FULL, post-await stop predicate and survivor handoff. A narrow
pass cannot complete attachment or terminal settlement. Qualify real Enter/button
and producer transitions with held original callbacks before release; diagnostics
or count reductions alone do not satisfy the six-second or feedback contracts.

### Completed promoted display publication — 2026-11-15

Within the accepted sole-stock promoted manual route, a completed original tabs
and transcript publication may suppress their unchanged routine pipelines. Reuse
one complete transcript gate, extracted from the existing implementation, rather
than creating a second partial invalidation key. Retain the exact mounted widget,
acknowledgement row, deletion arm, run activity, effective memory and current
visible citation counts. The latter is already an explicit token invalidator.
Only successful publication on the current widget installs the token; failed,
cancelled, detached or abandoned attempts retain the previous completed token.

The completed FULL witness also retains existing controller lifecycle,
presentation publication and fleet-unseen revisions. Capture before publication
and qualify again after all retained work. Missing, changed or custom/unproved
publication takes the original route. Preserve speech/H3, draft reconciliation,
resident feedback, settings/project/workspace attention, archive recovery and
manual-read scheduling. Always run the original recovery-region synchronization:
continuation warnings have no complete display revision.

Retain the existing fifteen-call transcript backstop. A short pass advances its
existing counter once; the forced pass takes the current qualified promoted long
branch and increments through the original transcript function. No timer or
budget changes. Short passes mark cadence owed and return False, keeping polling
and attachment reconciliation live. Explicit FULL, changed sources, provider
entry, cancellation, background activity and terminal handoff remain original.
This publication is display evidence only, never authority for actions or I/O.

### Runtime-owned display reader — 2026-11-16

Apply ADR356's runtime-registration rule to PR3534's refresh-free display read.
The composed Context/spend controller owns one reader of the existing
ConsoleContextReadSnapshot. Its normal read records the actual snapshot, config
projection and attachment visit. Reuse requires the same registered resource and
controller, current screen/controller/store/session, unchanged attachment and a
current published value, checked again after reading. Copies of registrations do
not create owners. Transparent wrappers do not change resource ownership.

Keep custom constructors without a registered resource on their original live
callback route. Remove the newly added function/code/default attestation rather
than relocating it. No new cache, scheduler, native owner, permission or cadence
is introduced; explicit FULL and unavailable/stale display fallback remain.
Task3r in the Windows Send plan supplies the mounted ownership/currentness and
existing feedback/FULL qualification. The broader polling proposal stays Proposed.

Resident FULL source/publication observation helpers live with the Console display
projections. Composer run controls retain preparation and custody eligibility.
This implementation move adds no cache, lifecycle, authority or timing exception;
completed-publication and source-currentness conditions remain unchanged.

### Optional busy-chat prefetch deferral, not selected — 2026-11-16

Task3y experimented with deferring implicit half-life config prefetch during
same-session runtime custody, retaining every expiry, cold/input/tokenizer,
handoff, retry and explicit FULL warm-up check. Thirteen mounted controls passed.
Quiet Windows Enter measured7.298/5.661/4.885s; Send button8.040/7.631/6.865s.
The experiment did not establish a whole-Send speed benefit and was removed.
Keep the original scheduling contract; the patch and original-owner regression
remain in task artifacts for possible later review. These timings do not isolate
a causal regression, and no longer display lifetime was considered acceptable.
