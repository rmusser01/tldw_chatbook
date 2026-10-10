# Console polling and full-state reconciliation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove repeated live-state preparation from routine Console display polling only after proving that authoritative transitions and terminal/background behavior retain their original contracts.

**Architecture:** Keep the original full-sync entry for existing explicit callers. Propose one narrow routine-poll entry using existing transcript/tab/control/mode owners and the existing full-sync exclusion/replay machinery. First prove the request and invalidation map; no incomplete dirty signature or disposable display proof may authorize live state.

**Tech Stack:** Python >=3.12, Textual 8.x, existing asyncio/worker and SQLite ownership; no new dependency.

**Spec:** [Approved Send architecture](../specs/2026-10-06-console-send-preparation-architecture-design.md), with proposed [ADR-226](../../../backlog/decisions/226-console-polling-and-full-state-reconciliation.md).

**Task:** [TASK-34563.33](../../../backlog/tasks/task-34563.33%20-%20Plan-Console-polling-and-full-state-reconciliation.md).

**Status:** Integration owner reviewed the source plan and authorized Phase 1 source/test preparation only. [TASK-34563.34](../../../backlog/tasks/task-34563.34%20-%20Verify-Console-polling-reconciliation-boundaries.md) owns the first two focused controls. Product narrowing and native qualification remain pending; Phase 1 evidence and integration-owner review are required before Task 2. ADR226 remains Proposed.

ADR required: yes.
ADR path: backlog/decisions/226-console-polling-and-full-state-reconciliation.md (Proposed).
Reason: a new polling-demand/invalidation contract crosses UI and live-state boundaries; ADR-126 authority and ADR-225 finite owners remain unchanged.

## Global constraints

- Preserve saved-turn failure behavior: stop Send and retain its draft; existing temporary chats remain available.
- Keep required durable acceptance, permissions, consent/trace/context facts before their dependent effects.
- No native lease across an await or approval wait; no cached mapping grants file or provider authority.
- Preserve current-source, exact session/workspace/app/database identity, maintenance, cancellation and physical retirement checks.
- Existing budgets remain: actual input/render feedback within 100 ms and ordinary app overhead under one second to adapter entry. Never redefine acceptance to a smaller phase.
- Only targeted tests; native timing runs sequentially under the integration owner, with quiet saved-source windows for measurements. No full sweep or concurrent app/probe in this lane.
- No polling-interval/deadline changes, storage schema/mode changes, new cache/queue/framework, or module-count-only restructuring.
- Exclude separately owned recovery-rail, workspace-attention-generation, Buddy, llama setup, progress-modal and Skills surface fixes. Preserve their integrated results without duplicating that work.

## Evidence and scope

At saved d221, full sync enters live config before both core and roleplay callbacks. Four main-thread groups contain 43/48/39 entries with entry intervals totaling 2.796/4.142/2.986 seconds. These intervals end at first yield/pre-yield return and are not total UI-body time or promised savings. Poll ancestry directly accounts for 7/11/7 entries; other async origins are unresolved. All 19/25/21 observed main settings reads hit cache. The 0.2-second timer starts on Preparing receipt and custody commit, always runs full sync, then checks whether it may stop.

The d221 measurement is incomplete, has unobserved native returns, and does not prove ordinary app-native cleanup. Latest action fixes in the integration lane leave this full-sync/poll boundary unchanged. Source anchors below refer to d221 symbols/lines, not a claim that later line numbers match. The integration checkout is the implementation baseline; do not copy this planning branch's older Screen over it.

## Effect and invalidation map

| Trigger/owner | Required behavior and order | Route / source anchor |
| --- | --- | --- |
| Initial/changed provider, settings, workspace or runtime gate | Build current selection, publish workspace/controller fields and current `app.app_config['console']['agent_runtime']`/bridge before dependent preparation. In-place config changes must be observed; settings revision alone is insufficient. | FULL; `chat_screen.py:_sync_console_chat_core_state` (d221:9484). |
| Session/workspace activation | Capture outgoing draft, align workspace, switch controller session, refresh target retrieval scope, await full sync, then focus/manual read acknowledgement. Revalidate ownership across awaits. | FULL; `session.py:_activate_native_console_session` region d221:3141; workspace activation regions 5871/6493. |
| Attach/resume/recovery | Complete current-visit full reconciliation before `finish_view_reconciliation` and ordinary timers. Deferred false cannot complete attach; ordered Resume must not create a competing default session. Trace-recovery callback stays full. | FULL; Screen15625, Session6740. |
| Settings/identity publication and roleplay repair | Existing explicit settings paths request full refresh. Identity generation clears the key; forced repair bypasses equality. Preserve pending persistence drain even when current session/name are unchanged; materialize projections before transcript publication. | FULL/live effect; Screen4292/5550 and4552/4581/4620/4688. |
| Ordinary active poll | Publish current transcript, session tabs, run/queue/review/fleet markers, checked controls and mode chip with their existing current-source checks. Classify existing helper effects: terminal receipt acknowledgement can write storage, and tab/manual-read effects retain their own authority or escalate before effect. Do not invoke live core/roleplay preparation merely for unchanged presentation. | Proposed narrow entry after proof; transcript17105, tabs17686, checked controls20783, mode17415. |
| Missing settings/session, changed store/session/membership or unresolved callback | Do not permit a supposedly narrow tab helper to ensure/create settings or publish to a replaced owner. Escalate before the effect; retain the original full behavior when equivalence is unknown. | FULL fallback; tab helper has awaits and is not inherently read-only. |
| Maintenance/busy/late full request | Preserve in-progress exclusion, current attachment visit and pending full replay. A full request during a narrow await dominates. Clear only the request consumed by a completed owned pass; retain failed/deferred demand. | Existing coalescers; Screen17507/17669-17684 and20682/20743. |
| Active/background work and custody | Keep polling for background runs, Preparing/accepted custody, wake scheduling gap, queued review publication and console-run workers even when viewed session is idle. | Original poll-needed predicate and timer17812/17817; receipt18219; wiring792. |
| Terminal reply/Stop/rewind | Full transition/replay must settle live state; exact rows render before terminal receipt acknowledgement under its original durable owner. This includes a viewed turn completing while a background/custodied run keeps polling active; it cannot wait for the stop-edge full pass. Preserve accepted owners and changed-composer/visible-session guards. | FULL transitions; transcript17105, poll17817, rewind19200, Stop19380; attention owner remains original. |
| Final poll/browser/fleet | When the original poll-needed predicate becomes false, require one successful current FULL pass; then recheck owner and poll-needed after its awaits. Only then invalidate browser rows, stop the timer and hand off to the survivor tick. Deferred full entry or new wake/review/custody keeps polling/replay alive. | Poll tail17851-17865; `fleet.py`138/196/313. |
| Speech/H3 and other unclassified reconciliation | Preserve speech-context and image-completion reconciliation, which precede core sync today. Establish their original completion/trigger path before narrowing; any pending or unclassified effect retains FULL. Do not silently omit these callbacks. | Full sync prelude; original speech and image owners. |
| Readiness result changed | Preserve its existing full request; display evidence is only presentation provenance. Retain cold/changed owner deferral and one existing checked read owner. | `console_spend_projection.py:440`; no whole-live-callback proof substitution. |

Existing identity/settings/attachment/repair generations are useful signals, not a complete source-authority witness. The roleplay key omits profile path/generation, object incarnation and pause. Direct in-memory changes and supported custom routes must be covered explicitly or keep FULL.

## Proposed routing API and ownership

- Keep `async ChatScreen._sync_native_console_chat_ui()` as the FULL default for every existing explicit caller. Preserve its completion/deferred behavior and attach consumers; avoid broad call-site migration.
- Proposed private addition: `async ChatScreen._sync_console_poll_display_ui() -> bool`. Only the original timer's `_poll_transcript` requests it after the phase-1 gate. `True` means that current owned presentation pass completed; it never substitutes for full view reconciliation or durable acceptance.
- Reuse `_console_sync_in_progress`, the existing pending FULL `_console_sync_requested`, current visit, maintenance/teardown guards and `_console_control_bar_replay_whole_sync`. Do not create a second scheduler, task map or generic demand queue.
- Routine demand while FULL is active may be absorbed by that already-inclusive pass. A FULL request during narrow work sets the existing full-request flag and must get a trailing original full pass. A narrow completion must not clear a late FULL flag; failure/defer retains its existing owner and replay path.
- Call original `_sync_console_native_session_tabs`, `_sync_native_console_transcript`, `_sync_console_control_bar` through its checked wrapper and `_sync_console_mode_bar` in their established order. Do not call an under-config renderer directly. Retain existing finite snapshot/workspace-build optimizations only where their ownership and publication validity remain equivalent.
- On the stop edge, await a successful current FULL pass and recheck owner/poll-needed before original browser invalidation, timer stop and survivor handoff. Maintenance, failed entry, a successor view or newly pending work retains the timer and existing replay demand. The survivor timer's own interval/final paint remain unchanged.
- The narrow label describes its request source, not a promise of I/O-free helpers. Classify and retain existing transcript acknowledgement/tab/manual-read effects under their original independent owners; no checked display projection authorizes their writes. If that separation cannot be proved, escalate before effect.
- A terminal transition while polling stays active for another run requires its own full reconciliation; the stop-edge final full is additional protection, not the only terminal trigger.
- Pending roleplay drain/repair and terminal work are not disposable display. Phase 1 must select an already-owned trigger for them, or the relevant poll remains FULL. Do not move drain behind tuple equality or use a new cache to infer it is unnecessary.
- This draft intentionally does not define a guessed all-input dirty token. Phase 1 records the concrete trigger/guard predicates and only then freezes the implementation details of the narrow route. If existing owners cannot express complete invalidation, return that precise gap for a bounded revised design before product edits.

## File ownership

| Lane | Planned files / responsibility |
| --- | --- |
| Integration owner | `tldw_chatbook/UI/Screens/chat_screen.py`: shared routing/exclusion, existing poll callback and terminal tail. Own final integration, saved-head tests and all native/real-provider runs. |
| Transition audit | `UI/Console_Modules/session.py`, `workspace.py`, `wiring.py`, `fleet.py`, `console_spend_projection.py`: read-only mapping initially. Edit only a demonstrated missing existing trigger under an explicit scope revision; do not broadly rewrite these modules. |
| Verification | `Tests/UI/test_console_poll_reconciliation.py` (proposed focused controls), selected existing tests below; diagnostic original request-origin observation only. No parallel native app launches. |
| Planning owner | This plan, ADR226, task34563.33 and `Docs/Development/console-optimization-review-list.md` OPT60. |

## Review focus

- In-place configuration/runtime-flag change must affect the next real action without manual full sync.
- Same-ID replacement or maintenance during an await must not publish stale rows or falsely finish attachment.
- A late FULL request must survive a narrow pass; a deferred FULL pass must replay once through its original owner.
- Viewed-idle background/custody/wake/review work must keep progressing and settle without another input.
- Pending roleplay persistence and terminal attention/browser/survivor effects must not disappear behind an unchanged fingerprint.

## Task 1: Establish causal request and transition controls

**Files:** create `Tests/UI/test_console_poll_reconciliation.py`; use the exact existing controls in the table below. No production timer routing change in this task.

**Interfaces:** consumes original timer/full-sync bodies, actual mounted Console, settings publication, runtime custody, real store/provider dispatch, and existing original-body diagnostic instrumentation. Produces observed request origins, the complete trigger map and explicit RED cases; adds no production API.

- [ ] Add `test_stable_preparing_poll_publishes_without_repeating_full_preparation`: hold a real configured preparation/provider boundary, observe multiple natural timer callbacks and their task/caller origins; assert transcript/tabs progress and count original core/roleplay preparation. Its desired count reduction must be RED on the original full-poll route for the stated reason. Do not replace measured callbacks or use elapsed sleep as completion evidence.
- [ ] Add `test_settings_change_reaches_next_send_without_manual_full_sync`, including current runtime gate and in-place app-config change: drive the real settings publication/next Send and assert actual provider/bridge selection. Do not invoke full sync from the test to manufacture freshness.
- [ ] Add `test_full_request_during_poll_await_replays_with_current_owner`, covering source/profile/session/store replacement and pause/resume. Hold an original publication/read; require stale refusal, no false attach completion, retained full demand and current replay. A forced-discarded pending full request must turn it red.
- [ ] Add `test_poll_terminal_transition_settles_without_another_input`: complete an actual received/custodied turn and require final rows/tab/attention, browser invalidation and timer/survivor transition; include pre-run custody, pending-review, and a viewed-terminal/background-still-active case. That mixed-run case must establish the full terminal transition and exact durable attention outcome while the timer correctly remains armed. A transcript-only replacement or viewed-idle-only stop predicate must turn it red.
- [ ] Run only these nodes and necessary named baseline controls in the integration native lane. Distinguish expected causal RED from fixture/setup failures. Record source revision, origin coverage, actual original effects and cleanup after fixture teardown.
- [ ] **Gate:** resolve supported in-place configuration changes, pending roleplay drain/repair, tab ensure/create/manual-read effects, original durable terminal-receipt acknowledgement and terminal escalation while other work remains active. Freeze exact predicates in this plan/ADR and obtain integration-owner review before Task 2. No source-only equivalence claim can satisfy this gate.

## Task 2: Implement the reviewed narrow polling route

**Files:** `tldw_chatbook/UI/Screens/chat_screen.py`; the focused control file. Existing controller/module trigger files remain outside ownership unless Task 1 demonstrates a specific missing trigger and the plan is revised first.

**Interfaces:** produces `_sync_console_poll_display_ui() -> bool` as described above; consumes existing full sync and checked presentation owners. Existing explicit caller API remains FULL.

- [ ] Add routing tests that fail if routine overlap invents a second owner, if FULL is downgraded/lost during an await, or if a narrow result completes attachment.
- [ ] Implement the reviewed route using existing coalescers and original ordered narrow bodies; preserve the complete original polling predicate and terminal tail.
- [ ] Keep unknown/missing/changed/effectful states on FULL. Preserve current live gates and supported custom callbacks; do not move a native context across an await.
- [ ] Run the new causal controls GREEN and the named original controls below, with deadlines unchanged. Review the diff specifically for hidden native/store effects in functions called from the narrow path.
- [ ] Save the implementation only after source/lint/format and targeted runtime evidence; one integration owner handles collisions and sequential runs.

## Targeted original controls and coverage limits

All paths below are under `Tests/`; each is a targeted node, not authorization for a full suite.

| File | Original nodes / what they establish |
| --- | --- |
| `Backup_Recovery/test_console_config_sync_lifetime.py` | `test_full_console_sync_defers_live_state_until_fresh_config_entry` (all four core/roleplay lock cases); `test_console_refresh_retries_after_native_pause_without_stale_state`. Original admission/refresh, with presentation callbacks substituted. |
| `UI/test_console_checked_display_scope.py` | `test_default_config_sync_still_enters_real_native_scope`; `test_actual_concurrent_saved_generation_invalidates_display`. Display proof remains distinct from live authority. |
| `UI/test_console_readiness_config_projection.py` | `test_readiness_config_publication_rejects_changed_owner`; `test_fresh_default_handoff_rejects_actual_session_owner_swap`. Checked-result publication fences, not physical-read lifetime. |
| `Performance/test_console_tab_sync_source_ownership.py` | `test_console_tab_publication_source_ownership` cases `same_id_session_replacement_during_surface_await`, `store_replacement_during_surface_await`, `active_change_during_surface_await`, `replay_keeps_drain_owned`. Original extracted bodies; complement with mounted cases. |
| `UI/test_console_parallel_runs.py` | `test_transcript_sync_timer_keeps_ticking_for_background_run_while_viewed_idle`. Actual timer and rendered state. |
| `UI/test_console_fleet_wake_ui_freshness.py` | `test_wake_reply_reaches_the_viewed_transcript_without_a_switch`; `test_wake_turn_in_a_nonviewed_session_flips_the_tab_glyph_off_running`; `test_poll_survives_the_wake_scheduling_gap_then_stops_after`. Preserve real publication; the gap case seeds a pending record. |
| `UI/test_console_fleet_survivor_tick.py` | `test_survivor_elapsed_advances_with_no_other_interaction`; `test_the_tick_stops_itself_with_one_final_settle_paint`. Actual timer/rendering, controlled bridge. |
| `UI/test_console_session_settings.py` | `test_console_identity_refresh_request_dispatches_without_transcript_tick`; `test_real_inactive_console_tab_activation_dispatches_identity_refresh`; `test_roleplay_repair_marker_retries_partial_then_consumes`; `test_roleplay_writer_cleanup_waits_for_owner_acceptance`. Real store/plans, persistence doubles. |
| `UI/test_console_turn_attention.py` | `test_successful_full_row_refresh_acknowledges_after_exact_mount`; `test_outgoing_refresh_cannot_ack_after_successor_claims_runtime`. Exact receipt/view ownership. |
| `UI/test_console_attach_completion.py` | `test_mounted_original_refresh_replays_before_runtime_attachment`. No narrow/failed refresh masquerades as full attachment. |
| `UI/test_console_marker_projection_native.py` | `test_marker_read_keeps_native_custody_and_current_publication` cases `repeated_cancel`, `conversation`, `dispose`. Original held native DB read and current publication. |

The existing `test_agent_runtime_gate_refreshes_without_screen_teardown` manually invokes core sync and cannot qualify the new producer-to-action route. Lightweight poll-reason tests establish shapes only. `test_captured_attach_timer_overlap_rearms_real_sync_worker` is strict xfail; it is not passing evidence. If scheduling/ownership of roleplay persistence changes, add a held original SQLite repair-retirement control; its existing doubles are insufficient.

## Task 3: Integrated qualification and documentation

**Files:** focused tests and existing diagnostic/report files only as needed; this plan, ADR226, task record and optimization ledger.

**Interfaces:** consumes saved integrated sources and unchanged original Send/input/render probes. Produces complete, separately attributed original action/native/performance receipts.

- [ ] Integration owner runs targeted checks after all implementation owners are ready. Check real expected branches, skips/xfails, and original callback effects; no full sweep.
- [ ] Run causal counts and diagnostic request-origin attribution separately from quiet timing. Do not add nested durations or label whole-phase opens as unique files/dispatch-only work.
- [ ] Run sequential saved-head original cold/warm three-Send and real-provider flows. Preserve actual input/render and provider-entry boundaries, required durability, limits and source/physical-retirement checks. Measure cold first use as well as stable polling; do not hide regression by reducing polling frequency.
- [ ] Keep platform results separate. Complete traces alone do not accept timing budgets; process containment alone does not prove app-owned worker/native cleanup.
- [ ] Reconcile actual outcomes, unresolved cases and implemented/deferred candidates in OPT60. Accept ADR226 and close the implementation task only after its own review/verification obligations are satisfied; planning completion is not product completion.

## Bounded first implementation: sole manual Preparing turn (TASK-34563.39)

The integration owner selected this smaller route after source review with the Reduce Console Send overhead owner and the qualified Task34563.38 original tab controls. General polling narrowing remains Proposed. The first production branch covers exactly one unpromoted, stock manual received intent for the current active session, with no queue revision, other custody, in-flight/background run, wake delivery, pending review publication or console-run worker. It preserves every original helper in the existing order; only repeated live core and roleplay preparation admissions may be omitted.

A completed current FULL establishes a disposable source witness only when its entry/end configuration identity, registry object and mutation generation, actual app/app-config/runtime/store/controller/session/settings/bridge objects, ordered membership and attach visit still match. Failed, deferred, source-changed or late-FULL passes do not establish that witness. This witness only chooses presentation work; it grants no file, provider or store mutation authority.

The manual receipt must retain its exact claim, task, captured current session inputs, staged-evidence and view attachment. The existing controller selection must equal the detached receipt selection. The cached controller runtime gate must match the canonical live in-memory app gate; the exact bridge and stock callback identities must remain current. Missing settings, unsupported/custom sources, changed source/owner/inputs/membership/visit, incomplete attachment, ordered resume or pending FULL uses the original FULL route. Pending/active roleplay plans, persistence/drain/repair/identity work and a changed effective global name also use FULL. Recheck after every retained helper await and before success. Tabs in this branch inspect established state before ensure/create and return escalation outside the tab lock on drift.

Retain Speech/H3, retrieval/dictionary/world-book, Character avatar and ambient context, transcript acknowledgement, workspace/status, checked rail/controls, recovery/readiness rows, mode, rail visibility, preference pruning and manual-read scheduling. Existing explicit FULL callers and final timer tail remain FULL. A routine result cannot complete attachment or consume a pending FULL; the existing exclusion/trailing replay owner handles escalation.

The smaller branch avoids a demonstrated dependency: runtime `_submit_fleet_wake` uses `resolve_runtime_turn_configuration_snapshot`, whose runtime gate comes from the cached controller field. Viewless/compatibility and missing-settings provider fallbacks also consume cached state. These paths remain FULL. The manual received path passes its detached selection into checked capture, rechecks its source, and submits the completed request configuration; it does not depend on a routine poll pushing live selection. No guessed whole-selection dirty key, new cache framework, scheduler, native owner, changed polling interval or deadline is introduced.

Qualification: consume the original causal Preparing RED and existing current-action/tab/source controls, add original routing/escalation checks, then run targeted GREEN and saved-head quiet DeepSeek three-turn UAT. Report counts and whole-Send elapsed samples separately; no under-one-second claim follows from fewer preparation calls.

The completed-FULL source witness also retains the exact app chat DB reference. A stock provider-direct receipt may have bridge=None only with chat_database=None; later DB/bridge publication forces FULL. Original action affinity and execution gates remain unchanged. Measured first Enter also exposed unused optional imports before receipt; those are now conditional on present owners, with all exact-type/source checks retained.
