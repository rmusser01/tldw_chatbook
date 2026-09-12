# Console Decomposition Wave 6 Design

**Status:** approved after written spec and quality review 2026-08-13; post-image
baseline amendment approved 2026-08-14; final closeout sequence superseded by
`2026-08-23-console-decomposition-wave6-closeout-amendment.md`

## Problem

The one-way Console size ratchet remains genuinely red after TASK-3070.2 landed on
current `origin/dev`:

- `tldw_chatbook/UI/Screens/chat_screen.py`: **22,172 lines** versus a **17,727**-line ceiling
- `ChatScreen`: **712 methods** versus a **593**-method ceiling

TASK-3070.1 retains immutable implementation base
`bed39af6b004e4db86218fad01d2ea515b332135` for the original Wave 6 families. After
TASK-3070.2, the serial rebase at `8d806b71d9c5ae7ed333ccb42780f6b2ea68acd0`
added fleet/wake, first-chat, browser-unseen, and auto-speak ownership to `ChatScreen`.
That post-image baseline is independently locked at 22,172 lines and 712 direct
methods; it does not rewrite the original evidence.

The remaining overage is 4,445 lines and 119 methods. Raising either ceiling is forbidden by
`Tests/Architecture/test_screen_size_ratchet.py`; it would erase the protection that
the earlier decomposition waves established. This wave must earn a reduction by moving
coherent ownership out of the screen.

The first draft used name matching and four controllers. Review rejected that arithmetic:
the four honest clusters contained only 4,415 method-body lines before delegation
overhead, and their framework entry points would leave too little method-count margin.
The post-image amendment keeps every approved child boundary intact, extends the
Workspace browser inventory by its new unseen-marker helper, and adds three later
atomic ownership slices: a fleet lifecycle controller plus coherent extensions of the
existing Session and HandsFree controllers.

| Remaining controller work | Inspected candidate bodies | Screen residue budget | Projected net line reduction | Projected method reduction |
|---|---:|---:|---:|---:|
| `ConsoleVideoController` | 1,292 / 33 | at most 10 / 2 | at least 1,282 | 31 |
| `ConsoleWorkspaceController` browser extension | 959 / 22 | at most 5 lines / 1 definition | at least 954 | 21 |
| `ConsoleRetrievalController` | 992 / 34 | at most 10 / 2 | at least 982 | 32 |
| `ConsoleSkillController` plus dead-path removal | 339 / 16 | at most 15 / 3 | at least 324 | 13 |
| `ConsoleCharacterController` | 281 / 8 | 0 / 0 | 281 | 8 |
| `ConsoleFleetLifecycleController` | 401 / 16 | 0 / 0 | 401 | 16 |
| `ConsoleSessionController` first-chat extension | 328 / 8 | 0 / 0 | 328 | 8 |
| `ConsoleHandsFreeController` auto-speak extension | 48 / 5 | at most 15 / 3 | at least 33 | 2 |
| Compatibility descriptors | 0 / 0 | at most 64 lines / 0 methods | -64 | 0 |
| **Remaining total** | **4,640 / 142** | **at most 119 lines / 11 definitions** | **at least 4,521** | **131** |

Production screen consumers are rewired to the owning controller, while every moved
plain assignable screen attribute is preserved through one small reusable module-level
read/write descriptor class plus explicit class assignments (31 attributes, budgeted
as at most 64 physical lines and zero direct `ChatScreen` methods). The remaining
projection is deliberately conservative: it charges the complete 64-line descriptor
budget again even though the image-family descriptor support has already landed. The
projected final screen is at most 17,651 lines / 581 methods, leaving 76 lines and 12
methods of margin. If characterization discovers another baseline plain
attribute, it must add the descriptor assignment and revalidate both margins before
production extraction. If either margin no longer clears, implementation stops for
another design review; no cluster is widened and no budget is raised.

## Existing Architecture

This design is revision work under the already-approved screen decomposition contract:

- `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`
- `DESIGN.md` section 7
- `tldw_chatbook/UI/Console_Modules/wiring.py`

The existing rules remain binding:

1. A region widget owns pixels; a controller owns non-DOM state and behaviour.
2. `action_*`, `@on(...)`, `@work(...)`, and command-dispatch names stay defined on
   `ChatScreen` when Textual or the command registry resolves them there. Their bodies
   become short, mutation-tested delegations.
3. Controllers do not use `query_one`; screen/region code passes data or named
   operations through explicit callables.
4. Dependencies are named, keyword-only constructor arguments wired as late-binding
   callables in `UI/Console_Modules/wiring.py`.
5. Cross-controller traffic uses those named callables, never a controller reaching
   through the screen to a sibling controller.
6. Every state name that was a plain assignable screen attribute retains read/write
   proxy compatibility, whether or not the current caller scan observes an assignment.
7. Existing worker group names, cancellation ownership, persistence ordering, and
   remount/shutdown behaviour are preserved.

No new ADR is required. The approved decomposition spec and `DESIGN.md` already govern
this module boundary; this wave applies them without changing a runtime, storage,
security, or service contract.

Wave 6 implements the canonical baseline-attribute rule with a small module-level
descriptor class rather than two direct `ChatScreen` methods per field. The descriptor
stores only two explicit constructor names (owning controller and target state), so its
`__get__`/`__set__` implementation and 31 one-line assignments fit a conservative
64-line budget without generated methods or closure factories. Each assignment reads/writes
the named controller attribute and raises `RuntimeError` if accessed before that
controller exists; setters never create shadow screen state. All moved defaults are
initialized by their controller, and the corresponding `ChatScreen.__init__`
assignments are removed rather than routed through a controller that is not built yet.
This compatibility begins after `build_console_controllers` returns. A getter or setter
before its owning controller exists raises `RuntimeError`; focused `__new__` fixtures
must construct the owning controller before assigning an override.
The screen owner names are fixed as `_image`, `_video`, `_workspace`, `_retrieval`,
`_skill`, and `_character`; descriptor families may land independently with their
owning extraction, but every family lands all of its descriptors and defaults together.

## Options Considered

### 1. Strict controller/region extraction — chosen

Move each coherent non-DOM feature family into one focused controller, keep Textual and
DOM boundaries on the screen or existing region widget, and wire dependencies
explicitly. This removes actual responsibilities and gives new work an obvious owner.

### 2. Mixin relocation — rejected

Mixins reduce the measured file while leaving implicit inherited responsibilities and
hidden dependency access. They satisfy the counter but not the ownership goal.

### 3. Raise or reset the ratchet — rejected

The current ceiling is intentionally one-way. Updating it upward would convert a hard
architecture failure into accepted growth.

## Component Boundaries

### `ConsoleImageController`

Owns transcript/generation image projections, remote image retrieval, generation-card
state, H3 reference snapshots and registry lifecycle, completion reconciliation,
failure merging, current-screen settlement, and generate/regenerate/select/keep
orchestration. Ordinary image generation retains its existing behaviour: it has no
shared cancellation event today, and this refactor must not invent one.

H3 keeps the exact existing cancellation-event identity from screen action through
registry/task/worker/adapter. The screen keeps the image-view DOM constructor, picker
presentation, composer/image preparation, command-palette paste, and modal launch.
`ConsoleMessageController` receives late-bound callables directly to this controller
for regenerate/select/keep/toggle; those calls do not detour through screen methods.

### `ConsoleVideoController`

Owns in-flight/cancellation state, video-store resolution, card specs and storage IDs,
pending artifacts, publication gates, shielded execution/drain, external-copy
validation and publication, outcome persistence, and generate/play/save/regenerate/
stream orchestration.

The controller resolves and validates a presentation request. The screen retains the
actual Textual modal/picker presentation and OS-player launch behind named callbacks;
the controller does not call `app.push_screen`, `query_one`, or OS launch APIs. Video
keeps the exact cancellation event, worker group, storage identity,
commit-before-cleanup order, and remount/shutdown drains.
`ConsoleMessageController` receives direct late-bound controller callables for
play/save/regenerate; its existing missing video-regenerate dependency is added in this
wave rather than routing that action through `ChatScreen`.

### `ConsoleWorkspaceController` conversation-browser extension

Owns conversation row acquisition, identity/filter/star/merge rules, persisted-row
cache, query token/timer/results/error state, background search, and post-selection
refresh.

`ConsoleWorkspaceController` already owns grouped-browser collapse preferences,
configuration, workspace activation/resume, the two toggle operations, and the legacy
conversation-search state. Extending that controller gives the whole browser lifecycle
one owner instead of creating a second query/timer/token/results authority plus a
mirror seam. The newer `_console_conversation_browser_*` compatibility names and the
existing `_console_workspace_conversation_*` scalar compatibility names become aliases
over the same Workspace-owned canonical query/timer/token/total/error state;
persisted-row cache state also moves there. Rows have one canonical rich representation:
`tuple[ConsoleConversationBrowserInputRow, ...]`. The legacy
`_console_workspace_conversation_search_rows` getter projects rich rows into
`ConsoleWorkspaceConversationRow`; its compatibility setter converts legacy rows into
bounded rich rows with explicit default metadata. The legacy Workspace refresh and
after-selection implementations are retired in favor of the canonical rich-row
pipeline, so no second writer can change the stored runtime row type. No
controller-to-controller browser mirror exists. The existing search event handler
stays on the screen as a delegate, and existing Workspace tests gain isolated no-mount
coverage for the added browser transitions and both projection directions.

### `ConsoleRetrievalController`

Owns staged-RAG capture, effective-scope resolution/cache warming, scope read/write and
save policy, picker input/output validation, library-RAG request execution and outcome
policy, auto-retrieve-on-send decisions, both dictionary and world-book cached summary
projections/actions, and degraded/placeholder state.

The screen retains decorated `@work` entry points and direct widget synchronization.
Picker/modal presentation stays screen-owned; the controller accepts/returns plain
values and invokes narrowly named refresh/notification callbacks.

### `ConsoleSkillController`

Owns skill-context retrieval, trusted/blocked candidate projection, blocked-match
policy, refusal-row construction, and pending install/script state. The live
command-registry name and two Textual decision handlers stay as short screen delegates.

Source review found that the fallback resolver has no registered producer and the
picker chain is intentionally unreachable. Wave 6 deletes that dead surface instead of
giving it a new owner: the `KIND_FALLBACK` dispatch branch,
`_console_command_run_skill`, `_console_skill_search`,
`_run_resolved_console_skill`, `_open_console_skill_picker`, the unused picker widget,
its CSS selectors/cross-references, its stale mentions in the resolver/style-picker
source and style-picker tests, and their now-obsolete focused tests/imports. Live
`/skills`, `$name` substitution, blocked/refusal handling, candidate refresh, and
install/script decisions remain and move as described.

### `ConsoleCharacterController`

Owns picker-option projection, character handoff/session choice policy, active
conversation/character identity, card/avatar byte retrieval, and avatar refresh
decision/state. The screen and `ConsoleLeftRail` keep modal presentation and avatar DOM
rendering. `_refresh_active_character_avatar_if_scope_changed` is split: controller
computes whether/what to refresh, and a screen/region callback applies the pixels.

### `ConsoleFleetLifecycleController`

Owns the post-baseline fleet-completion handoff, durable unseen-marker cache and run
marker policy, mount-time wake claims, user-priority and in-view decisions, wake retry
and delivery-start transitions, teardown accounting, and survivor-tick lifecycle. It
receives named late-bound callbacks for the displayed composer, screen visibility,
transcript repaint scheduling, timer creation/stopping, and workspace/session
activation; it never queries the DOM or reaches through the screen to another
controller. The exact first-signal, durable-mark-before-view-clear, teardown ordering,
and idle-timer behavior remain unchanged.

### `ConsoleSessionController` first-chat extension

Owns the post-baseline first-chat default fence, pristine-session eligibility, exact
claim release/acknowledgement, rollback, and retry policy. Existing screen callbacks
perform the final mounted UI resynchronization and focus restoration only. Claim
identity, configuration-generation fencing, no-overwrite behavior, metadata-only
diagnostics, and retryable failure semantics remain byte-equivalent.

### `ConsoleHandsFreeController` auto-speak extension

Owns the post-baseline auto-speak destination and control-state decisions. The three
Textual `@on` entry points stay on `ChatScreen` as at-most-five-line delegates. No
speech queue, retry, resume, or presentation behavior changes.

## Source-Inspected Ownership Inventory

Legend: **M** removes the `ChatScreen` definition and rewires production callers
directly to the controller; **D** moves the body but keeps a framework-required screen
delegate whose complete AST definition span (`end_lineno - lineno + 1`, decorators
excluded) is at most five physical source lines; **S** stays screen/region-owned and is
included only in the residue budget, not claimed as an extracted method.

### Image/H3

**M (25):** `_build_console_image_specs`, `_extend_specs_with_remote_images`,
`_fetch_remote_transcript_image`, `_build_generation_card_specs`,
`_pending_console_generation_card_images`, `_console_imagegen_inflight_sessions`,
`_console_imagegen_inflight_message_ids`, `_console_generate_image_conversation_pairs`,
`_console_generate_image_llm_context_options`, `_h3_image_edit_registry`,
`_h3_reference_snapshot`, `_h3_reference_from_snapshot`,
`_filter_h3_attachment_from_app_stash`, `_h3_origin_screen_is_live`,
`_cleanup_h3_completion_in_store`, `_reconcile_h3_image_edit_completions`,
`_merge_h3_failure_notice_in_store`, `_settle_current_h3_outcome`,
`_schedule_current_h3_settlement`, `_append_h3_image_edit_error`,
`_run_h3_image_edit_command`, `_regenerate_console_generation_variant`,
`_select_console_generation_variant`, `_keep_console_generation_variant`, and
`_handle_console_toggle_image_view`.

**D (1):** `_console_command_generate_image`, retained because the command registry
resolves the handler on the screen.

**S (5):** `_ensure_console_image_view`, `_console_generation_browse`,
`_prep_console_images`, `_open_console_generate_image_modal`, and
`_paste_console_generate_image_command`.

### Video

**M (31):** `_console_videogen_inflight_sessions`,
`_console_videogen_cancel_events`, `_ensure_console_video_store`,
`_build_video_card_specs`, `_video_storage_message_id`,
`_pending_console_video_artifacts`, `_owns_pending_console_video`,
`_close_pending_console_video`, `_register_console_video_publication_gate`,
`_release_console_video_publication_gate`, `_begin_pending_console_video_operation`,
`_end_pending_console_video_operation`, `_await_shielded_console_video_task`,
`_run_pending_console_video_operation`, `_run_console_video_generation_operation`,
`_drain_pending_console_videos`, `_external_video_target_identity`,
`_external_video_stat_identity`, `_external_video_cleanup_identity`,
`_external_video_parent_identity`, `_require_external_video_pinned_capabilities`,
`_external_video_precommit_check`, `_copy_pending_video_external`,
`_retry_pending_console_video`, `_save_pending_console_video_external`,
`_normalize_pending_video_target`, `_resolve_generated_video_outcome`,
`_persist_generated_video_tuple`, `_play_console_video`,
`_save_console_video_copy`, and `_regenerate_console_video_message`.

**D (2):** `_console_command_generate_video` and `_console_command_stream_video`,
retained because the command registry resolves them on the screen.

**S:** `_open_video_with_os` and `_wait_for_console_screen_result`; they are presentation
seams and are not counted in the 1,292 candidate lines.

### Conversation browser

**M (21):** `_start_console_conversation_browser_search`,
`_console_browser_row_key`, `_console_browser_row_scope_copy`,
`_console_browser_row_matches_query`, `_filter_console_browser_rows_for_query`,
`_find_console_browser_row`, `_console_browser_display_identity`,
`_starred_console_conversation_ids`, `_apply_console_browser_star_state`,
`_native_console_browser_rows`, `_membership_console_browser_rows`,
`_persisted_console_browser_rows`, `_invalidate_console_persisted_rows_cache`,
`_sync_persisted_console_browser_rows`, `_compute_persisted_console_browser_rows`,
`_merge_console_browser_rows`, `_current_console_browser_rows`,
`_refresh_console_conversation_browser_search`,
`_refresh_console_conversation_browser_after_selection`,
`_with_console_conversation_browser_state`, and `_console_browser_unseen_marker`. The
screen's existing decorated input
handler arms the timer with a late-bound controller callback directly.

**D (1):** `on_console_workspace_conversation_search_changed` retains its `@on`
decorator, extracts the event's plain query/disabled state, and delegates the transition
to the controller within the five-line physical-span limit.

**S:** none. Collapse/config helpers and toggle operations already belong to Workspace
and remain there; they are not part of the 912-line screen-removal inventory.

### Retrieval/RAG

**M (32):** `_capture_console_staged_rag`, `_build_console_retrieval_scope_state`,
`_console_retrieval_scope_run_recipe_count`, `_resolve_console_effective_scope_state`,
`_refresh_console_effective_scope_and_sync`,
`_warm_console_effective_scope_cache_if_stale`, `_read_console_retrieval_scope`,
`_write_console_retrieval_scope`, `_console_scope_picker_listers`,
`_apply_console_retrieval_scope_save`, `_console_rag_source_status`,
`_active_console_dictionary_scope_ids`,
`_refresh_active_dictionaries_summary_if_scope_changed`,
`refresh_active_dictionaries_summary`, `_active_console_world_book_scope_ids`,
`refresh_active_world_books_summary`,
`_refresh_active_world_books_summary_if_scope_changed`,
`_console_dictionary_inspector_rows`, `_console_world_book_inspector_rows`,
`_console_dictionary_inspector_actions`, `_console_world_book_inspector_actions`,
`_console_library_rag_scope_label`, `_stage_console_library_rag_launch`,
`_maybe_auto_retrieve_for_send`, `_apply_console_rag_settings_choice`,
`_resolve_console_library_rag_scope`, `_apply_console_library_rag_search_outcome`,
`_rag_service_still_initializing`, `_notify_console_auto_rag_scope_empty`,
`_notify_auto_rag_degraded`, `_notify_console_auto_rag`, and
`_clear_console_auto_rag_placeholder`.

**D (2):** `_persist_console_rag_auto_retrieve_on_send` and
`_execute_console_library_rag_search` retain their `@work` decorators and worker groups.

**S:** picker/modal handlers and direct DOM sync methods, including
`_sync_console_retrieval_scope_row`, `_open_console_retrieval_scope_picker`,
`_set_console_library_rag_source_scope`, and the decorated button/input handlers.

### Skills

**M (9):** `_fetch_console_skill_context`,
`_console_skill_trusted_candidates_from_context`, `_console_skill_blocked_summaries`,
`_refresh_console_skill_candidates`, `_split_console_skill_name_args`,
`_console_skill_blocked_match_response`, `_append_skill_refuse_row`,
`_set_console_pending_skill_install`, and `_set_console_pending_skill_script`.

**D (3):** `_console_command_skills`,
`handle_console_skill_install_decided`, and `handle_console_skill_script_decided`.
The live registered command and decision paths call the controller directly; no screen
fallback method is retained.

**X — deleted as unreachable (4):** `_console_skill_search`,
`_console_command_run_skill`, `_run_resolved_console_skill`, and
`_open_console_skill_picker`, plus the dispatcher branch and picker module that only
supported them.

### Character

**M (8):** `_console_character_picker_options`,
`_current_console_rail_conversation_id`, `_current_console_rail_character_id`,
`_current_console_rail_character_name`, `_fetch_character_card_for_avatar`,
`_fetch_expression_image_bytes`, `_apply_console_character_choice_async`, and
`_refresh_active_character_avatar_if_scope_changed`. Existing screen callers and the
picker callback invoke the controller directly; the controller uses a named screen/
region callback only for the final pixel application.

### Fleet/wake post-baseline drift

**M (16):** `consume_pending_console_fleet_completion`,
`_claim_console_fleet_wake_marks`, `_console_wake_user_priority`,
`_console_wake_probe_composer`, `_console_screen_displayed`,
`_console_wake_conversation_in_view`, `_poke_console_wake_retry`,
`_on_console_wake_delivery_started`, `_console_wake_turn_active`,
`_record_console_fleet_teardown`, `_console_fleet_unseen_ids`,
`_console_run_marker_with_unseen`, `_console_fleet_survivors_live`,
`_maybe_start_console_fleet_survivor_tick`, `_stop_console_fleet_survivor_tick`,
and `_console_fleet_survivor_tick`.

Screen/DOM observations are supplied through named dependencies; none of these names
remain as direct `ChatScreen` definitions.

### First-chat post-baseline drift

**M (8):** `_first_chat_defaults_match`, `_current_first_chat_defaults`,
`eligible_console_first_chat_session_id`, `_release_first_chat_claim`,
`_log_first_chat_handoff_exception`, `_resync_console_after_first_chat_rollback`,
`_resync_mounted_console_after_first_chat_rollback`, and
`consume_pending_console_first_chat_intent`.

### Auto-speak post-baseline drift

**M (2):** `_resolve_console_auto_speak_destination` and
`_sync_console_auto_speak_controls`.

**D (3):** `on_console_auto_speak_changed`,
`on_console_auto_speak_resume_requested`, and
`on_console_auto_speak_retry_requested` retain their `@on` decorators and delegate
within the five-line physical-span limit.

## Baseline Attribute Compatibility Inventory

The following 31 names require assignable `ChatScreen` compatibility on the
implementation base. Thirty are discovered from plain `self.<name>` assignments (some
lazy); `_console_video_store` is the one additional externally written test-override
seam, read through `getattr` in production but assigned by focused video tests. Defaults
move into the owning controller and `ChatScreen` retains an explicit read/write
descriptor assignment for every name:

- Image (3): `_imagegen_inflight_sessions`, `_imagegen_inflight_message_ids`,
  `_console_h3_ui_generations`.
- Video (8): `_console_videogen_inflight`, `_console_videogen_cancels`,
  `_console_video_store`, `_pending_video_artifacts`,
  `_pending_video_artifacts_closed`, `_pending_video_operation_cancels`,
  `_pending_video_active_operations`, `_pending_video_deferred_closes`.
- Conversation browser (9): `_console_persisted_rows_cache`,
  `_console_persisted_rows_cache_key`, `_console_persisted_rows_cache_at`,
  `_console_conversation_browser_query`,
  `_console_conversation_browser_search_timer`,
  `_console_conversation_browser_search_token`, `_console_conversation_browser_rows`,
  `_console_conversation_browser_total`, `_console_conversation_browser_error`.
- Retrieval (6): `_console_retrieval_scope_cache`, `_console_effective_scope_cache`,
  `_active_dictionaries_summary`, `_last_console_dictionary_scope_ids`,
  `_active_world_books_summary`, `_last_console_world_book_scope_ids`.
- Skills (1): `_console_skill_candidates`.
- Character (4): `_active_character_avatar`, `_active_character_avatar_name`,
  `_last_console_avatar_scope`, `_console_expression_spec_cache`.

An AST baseline test compares the 30-name assignment subset to plain `self.<name>`
assignments on the recorded implementation base. A separate caller/source assertion
locks `_console_video_store` as the explicit external-write override seam. Both tests
prove every current class assignment is a read/write descriptor targeting the correct
controller/name pair. Characterization also proves all controller defaults exist before
the first post-construction read; both descriptor reads and writes raise `RuntimeError`
before the owning controller exists; `getattr(screen, name, default)` and
`hasattr(screen, name)` cannot hide that missing controller because it never raises
`AttributeError`; and the external `_console_video_store` override remains writable
after construction. Existing `ChatScreen.__new__` fixtures that assign controller state
must construct or directly target the owning controller first. Descriptor getter/setter
mutations must fail these tests; shadow attributes are forbidden.

## Data and Control Flow

1. A Textual handler or registered command remains on `ChatScreen` when framework name
   resolution requires it.
2. The handler reads event/DOM values and passes plain values to its controller.
3. The controller performs state transitions, persistence, and async orchestration
   through explicit dependencies.
4. The controller returns a value or invokes a narrowly named screen callback.
5. The screen or existing region widget applies DOM/presentation changes.

Construction for the six new controllers is added to `build_console_controllers`; the
existing Workspace construction receives the added browser dependencies and state.
Construction order is not semantically load-bearing: sibling dependencies are
late-bound callables.

## Error, Cancellation and Privacy Contracts

This is a behaviour-preserving refactor. Public/sanitized error copy and metadata-only
logging remain byte-equivalent unless a focused test requires an owner-name-only change.

The exact same cancellation event applies to H3 image edits and generated-video
operations only. Ordinary image generation has no such event and remains non-cancellable.
Late outcomes reconcile only onto the current matching screen/session/generation;
drains remain bounded and app shutdown keeps definitive ownership.

No attachment bytes, prompts, paths, message/session IDs, signed URLs, provider
payloads, or exception messages may be added to persistent diagnostics.

## Testing Strategy

Every controller family is characterized before extraction and moved in its own child
PR.

### Isolated controller tests

Each new controller gets a dedicated unit test module that constructs it with plain
fakes/call recorders and **does not mount a Textual app or screen**. The Workspace
browser extension adds the equivalent cases to the existing Workspace controller unit
module. These tests prove state transitions, dependency ordering, error containment,
and each controller's primary async/cancellation seam. Mounted tests remain for DOM
integration only.

### Cross-cutting architecture tests

- AST inspection rejects any call whose attribute name is `query_one` inside the six
  new controllers or the moved Workspace browser methods; this is structural, not a
  substring check.
- AST ownership inventory proves every **M** method is absent from `ChatScreen` and every
  **D** method has a complete definition span of at most five physical lines, excluding
  decorator lines, with a real framework/registry caller.
- screen entry points retain decorators/binding names and worker groups
- recorded baseline evidence determines the complete compatibility inventory: the
  30-name assignment AST plus the explicit `_console_video_store` caller/source
  assertion; later caller scans may add integration tests but never remove a proxy
- controller imports are repointed to defining modules; no screen re-export is added
- named dependency tests reject direct sibling-controller reach-through

### Product-boundary evidence

- Image/H3: generation actions/cards, mounted image flow, H3 cancellation,
  fresh-screen/remount, and attachment-stash tests.
- Video: generate/play/save/regenerate/stream, store/capacity/publication/cancellation,
  remount, container identity, and external-copy failure tests.
- Browser/retrieval: search/group/select/persistence, scope picker, library RAG,
  auto-retrieve, and existing real-SQLite scope/session cases.
- Skills/character: live `/skills` and `$name` routing, trust/block/install/script
  decisions, absence of the dead skill-picker surface, character picker/prompt
  seed/handoff, avatar refresh, and rail rendering.
- Fleet/first-chat/auto-speak: durable handoff and unseen markers, wake delivery and
  survivor ticks, pristine-session fencing/rollback, and speech resume/retry events.

Required mutation checks remove each screen delegate, cross-session/generation gate,
shared cancellation-event handoff, persistence-before-cleanup step, and unified
Workspace browser-state alias one at a time.

### Final gates

**Superseded for closeout by the approved 2026-08-23 amendment.** TASK-3070.14,
not TASK-3070.11, performs the final rebase and measures actual lines/methods. It
lowers the ratchet to the exact earned values and never raises it. The owner explicitly
prohibits a local full-suite run: run only related product tests plus the approved
architecture/static/privacy/diagnostic gates, then use required GitHub Actions as the
broad integration gate.

## Atomic Delivery

TASK-3070 is the coordinated wave. Its child tasks are atomic and independently
reviewable:

1. TASK-3070.1 characterizes and locks the ownership/ratchet inventory.
2. TASK-3070.2 extracts image/H3.
3. TASK-3070.3 extracts video.
4. TASK-3070.4 consolidates the conversation browser into Workspace.
5. TASK-3070.5 extracts retrieval/RAG.
6. TASK-3070.6 extracts skills.
7. TASK-3070.7 extracts character policy/avatar state.
8. TASK-3070.8 extracts fleet/wake lifecycle ownership.
9. TASK-3070.9 extends Session with first-chat handoff ownership.
10. TASK-3070.10 extends HandsFree with auto-speak ownership.
11. TASK-3070.11 freezes the invalidated closeout evidence and the approved amendment.
12. TASK-3070.12 extracts realtime orchestration.
13. TASK-3070.13 extracts review/selection workflow ownership.
14. TASK-3070.14 rebases, lowers the ratchet, updates canonical progress, and
    closes the parent.

Each child is delivered as one atomic PR and must return its focused gate to green
before it is reviewed and merged. TASK-3070.2 through TASK-3070.10 branch from the latest
`dev` only after their predecessor PR merges. TASK-3070.11 through TASK-3070.14 follow
the amended sequence in `2026-08-23-console-decomposition-wave6-closeout-amendment.md`.
Commits inside a child PR remain focused on that child. If a rebase changes the
inventory enough to invalidate the projection, stop and amend the binding design
before implementation.

## Success Criteria

- The size and method ratchets pass without increasing either budget.
- Relative to the final rebased starting measurement, `ChatScreen` loses at least the
  exact overage in both dimensions, with the ratchet lowered to the earned result.
- Image/H3, video, browser, retrieval, skill, character, fleet/wake, first-chat,
  auto-speak, realtime, and review/selection behaviour is unchanged, including
  ADR-068's screen-owned review-note workflow.
- Each family has an obvious non-DOM owner and isolated no-mount unit tests.
- All automated, static, privacy, cancellation, persistence, and lifecycle gates pass.
