# Dev-Gate Test Contract Repair Design

## Goal

Restore the mandatory `dev` pytest gate by reconciling stale or nondeterministic
tests with the production contracts that already exist, then review and refresh
the checked diagnostic inventory under ADR-029. It must not restore retired
Chat infrastructure, change audio-recording behavior, or silently admit unsafe
persistent diagnostics; a production edit is allowed only when the gate exposes
a real current-behavior defect and the repair is documented and test-first.

## Evidence

The failures reproduce on an exact `origin/dev` checkout:

- `Tests/Event_Handlers/test_worker_events_contract.py` imports `StreamDone`,
  although TASK-577 deliberately removed that event and made the retained
  adapter reject streaming.
- `Tests/Event_Handlers/test_worker_local_citation_capture.py` exclusively
  exercises the same retired streaming bridge, removed sentinel/error
  swallowing, and worker-owned citation-builder stripping. The sole retained
  caller is explicitly non-streaming and passes no citation builder; native
  Console now owns the live citation and privacy lifecycle.
- Latest `dev` independently removed the retired `TabState` fixture from
  `Tests/UI/test_chat_shell_bar.py` while adding current persona-label coverage.
  TASK-1333 preserves that upstream repair rather than carrying a competing
  edit.
- `Tests/Audio/test_audio_integration.py` starts a recorder thread and then
  calls the same recording loop repeatedly on the test thread. It also leaves
  VAD enabled, so the assertion depends on installed optional dependencies and
  whether synthetic bytes are classified as speech.
- `Tests/Audio/test_recording_service.py` repeats the same thread-and-direct-loop
  race with VAD enabled, so synthetic audio may never reach the callback that
  stops the loop.
- The adjacent SoundDevice flow fixture also leaves VAD enabled for a
  four-sample synthetic callback, so no audio reaches its queue assertion.
- Four `Tests/Chat/test_chat_functions.py` cases monkeypatch deleted
  module-level `settings` objects. The live Llama.cpp and DeepSeek adapters
  deliberately resolve `get_runtime_config_snapshot()` at each request
  boundary under ADR-029.
- Four `Tests/LLM/test_local_llm_provider_config.py` cases repeat the deleted
  `settings` pattern for the local-LLM adapter, which also resolves the
  immutable runtime snapshot at each request boundary.
- `Tests/LLM_Provider_Catalog/test_local_openai_compatible_provider_name.py`
  patches the snapshot seam but supplies local-LLM configuration at the retired
  top level instead of `api_settings.local-llm`.
- Three real-seam Notes fixtures pass a nonexistent `tmp_path/notes_base` to
  `NotesInteropService`. ADR-029's trusted-directory boundary correctly
  rejects missing directories instead of creating them implicitly.
- `Tests/DB/test_rag_indexing_db.py::test_large_batch_operations` combines
  useful 1,000-item persistence coverage with hard wall-clock limits. The
  unchanged `origin/dev` test took 24.98 seconds inside a loaded full-suite run
  but 1.25 seconds in isolation, proving the timing assertion depends on host
  contention rather than database behavior.
- `Tests/Event_Handlers/test_eval_db_operations_path.py` retargets
  `TLDW_CONFIG_PATH` beneath a nonexistent `profile-two` directory. The
  private-path boundary correctly rejects that missing trusted parent before
  config or database creation.
- The Library skill-name collision guard reports that the newer runtime tools
  `search_run_log`, `run_log_stats`, and `run_log_slice` are absent from
  `_SHADOWED_BUILTIN_NAMES`, allowing a local skill to collide with a real
  built-in runtime name.
- `Tests/Local_Ingestion/test_quick_ingest_db_path.py` expects the retired
  `tldw_cli_media_v2.db` fallback filename even though `quick_ingest()` now
  delegates to the canonical profile-aware `get_media_db_path()`, whose default
  filename is `tldw_chatbook_media_v2.db`.
- The RAG citation benchmark harness creates its isolated `config/` root but
  selects `config/tldw_cli/config.toml` without creating the intermediate
  trusted profile directory, so its host-secret isolation subprocess fails
  before the benchmark runs.
- TASK-1091 left-aligned Watchlists tree labels, invalidating TASK-997's
  documented assumption that a two-space source prefix plus Textual's centered
  minimum button width would place a child name past its parent. The existing
  compositor regression now renders `ArXiv` at column 4 and its parent at
  column 5 at the narrow visual-parity size.
- The Speech Lab-frame migration moved local dependency recovery out of the
  bare `STTSWindow`: exact visible taxonomy now belongs to
  `STTSScreen`'s inspector `#speech-capability-status`, while install guidance
  belongs to the rail `#speech-capability-summary` tooltip. The generic
  disabled-action suite still mounts the retired owner and therefore raises
  `NoMatches` before checking either distinct contract.
- The never-run Evals bench correctly renders its target as `Not yet checked`,
  but its screen-wide `.ds-recovery-callout` absence assertion also matches the
  valid `#evals-primary-action-reason` explaining why Run Bench is deferred.
  The target-readiness owner `#evals-inspector-bench` contains no recovery
  callout; the test conflates two unrelated uses of the shared callout style.
- The duplicate-target Evals regression similarly counts `.ds-status-badge`
  beneath broad `#evals-inspector-pane`. Its two intended target readiness rows
  now share that pane and class with the valid sibling
  `#evals-primary-action-status`, so the count is three even though both
  duplicate target rows compose correctly with distinct ids.
- The real Lab route mounts `LocalModelsWidget`, whose parent `on_mount()`
  immediately queries `#delete-confirm-dialog` before composed children are
  queryable and raises `NoMatches`. The dialog and delete flow remain live;
  their initial hidden state and later reactive visibility are coupled to an
  invalid parent/child lifecycle assumption.
- The selected and empty Library Collections regressions still expect
  `#library-use-in-console` to be disabled. TASK-716 deliberately made blocked
  handoff buttons pressable so their handler can emit the recovery warning,
  while `library-source-action-blocked` carries the blocked visual/state
  contract; dedicated tests already cover that press and tooltip behavior.
- Completing a Library ingest refreshes the local-source snapshot and
  intentionally recomposes the rail. The different-canvas isolation regression
  calls `query_one("#library-row-browse-media")` on every poll, so it raises
  during a legitimate teardown frame before it can observe the remounted
  `Media (1)` row while Notes remains selected.
- Four MCP import-file regressions replace
  `mcp_workbench_module.os.path.expanduser`, but that attribute is the shared
  process-wide `os.path.expanduser` function. During workbench mount the patch
  therefore also makes the isolated `TLDW_CONFIG_PATH` resolve to the temporary
  directory itself, which the private-file guard correctly rejects before the
  import panel can mount. The workbench already exposes `_mcp_import_home()` as
  the narrow containment-root seam those fixtures intend to control.
- The MCP audit-detail fake still emits retired `arguments`, `result_excerpt`,
  and free-form `error` fields, and three inspector tests still expect selected
  payload values to render. The live execution log and inspector now expose
  only ADR-029-safe metadata: registered argument names, unknown-argument
  count, result type/size, bounded categories, and exception type.
- Four current Media browsing-shell tests dispatch background searches through
  `Widget.run_worker()` but use only one `pilot.pause()` before reading rows,
  resetting the async service mock, or inspecting the next search call. The
  result-loading and item-selection nodes reproducibly reach those assertions
  before `_run_media_search()` publishes the mocked results.
- `Tests/Architecture/test_persistent_diagnostic_inventory.py` reports reviewed
  production-owner drift while the persistent sink topology remains unchanged.
  ADR-029 requires inspecting the changed calls before regenerating the checked
  inventory.

TASK-1333 owns these gate repairs and the reviewed generated-inventory refresh.

## Decision

Update the tests to describe current behavior:

1. Keep the non-streaming worker failure regression and delete its obsolete
   streaming-sentinel case. Delete the fully obsolete worker-local citation
   capture file rather than recreating retired streaming, sentinel, logging, or
   builder ownership. The existing
   `Tests/Event_Handlers/test_retained_worker_adapter.py` already pins the live
   delegation and streaming-rejection contracts, while native Console tests pin
   citation lifetime and privacy, so TASK-1333 adds no duplicates.
2. Preserve the latest `dev` chat-shell test unchanged. It already covers the
   live session and persona-label contract without `TabState`.
3. In the stream-error regression, run `_pyaudio_recording_loop()` exactly once
   with `is_recording = True` and
   VAD disabled, without calling `start_recording()`. Assert the exact two-chunk
   callback sequence, `stop_stream()`, `close()`, and final stopped state.
   Rename the test so it no longer claims automatic recovery that production
   does not implement.
4. Apply the same deterministic structure to the PyAudio happy-flow regression:
   disable VAD, set the callback and recording state directly, invoke one loop,
   and assert exactly three chunks plus cleanup. Do not start a background
   recorder.
5. Keep the SoundDevice flow through its public start/stop contract, but disable
   VAD for its tiny synthetic callback. Use a bounded event set by the mocked
   `InputStream` constructor before reading the captured callback, and stop the
   mocked recorder in `finally` before asserting that audio was queued.
6. Update the stale Llama.cpp, DeepSeek, and local-LLM request tests to patch
   `get_runtime_config_snapshot()` with a `RuntimeConfigSnapshot` containing
   their test configuration under the live `api_settings` shape. Do not restore
   or emulate mutable module-level settings.
7. Create each fixture-owned temporary Notes base directory with mode `0700`
   before constructing `NotesInteropService`, and close all per-user Notes
   connections before the template DB during fixture teardown. Do not weaken
   production path verification or make the service create a security-sensitive
   root implicitly.
8. Generate the candidate diagnostic inventory, inspect every changed owner and
   sink-topology entry against ADR-029's metadata-only boundary, and refresh the
   checked artifact only if the changes are safe. If review finds an unsafe log
   value, correct that violation under ADR-029 before regenerating; do not bless
   it as inventory drift.
9. Keep the large-batch indexing test's 1,000-item write and retrieval
   assertions, but remove its host-dependent elapsed-time measurements. This
   default functional gate must prove persistence correctness, not benchmark a
   contended workstation.
10. Create the retargeted profile fixture directory with mode `0700` before
    selecting its config file, and explicitly close the test-owned Evals
    database connection during teardown. Do not make production config loading
    create or trust a missing security-sensitive parent.
11. Add exactly the three registered run-log runtime tool names to the existing
    fixed Library skill shadow set. Keep the literal collision boundary; do not
    import the agent runtime registry into the pure display-state module.
12. Update the one stale Local Ingestion fallback assertion to the canonical
    media database filename. Preserve configured-path and traversal-rejection
    coverage; do not change production path resolution.
13. Create the benchmark harness's isolated `config/tldw_cli` directory as
    owner-only and idempotently before overriding `TLDW_CONFIG_PATH`. Do not
    relax private-path verification or expose host environment values.
14. In the production Console Stop regression, advance the Textual pilot after
    the Stop control becomes visible and before issuing the pointer click. The
    test must continue exercising the real visible action and proving provider
    cancellation plus preservation of the stopped partial response; do not
    bypass the UI through direct controller calls or weaken production
    cancellation behavior.
15. In both production Media lifecycle regressions, wait boundedly through the
    existing Textual pilot until the outgoing `MediaWindow` is both closed and
    detached. Keep the fresh replacement-instance assertion, stale-owner
    exclusion, and durable last-edit-wins behavior unchanged. Do not change
    production screen teardown or replace lifecycle checks with an arbitrary
    sleep.
16. In the production provider-selection ownership regression, wait for the
    recomposed provider and model controls before using them. Deliver the same
    `Changed` events to the live Settings handlers after programmatic test
    assignments, then wait boundedly for the staged mapping and saved app
    defaults. Preserve the distinction between global defaults, the existing
    user-owned Console session, and the later explicit handoff. Do not change
    production Settings or replace state predicates with arbitrary pauses.
17. Remove the RAG UI integration fixture and test that expect a recognized
    canonical media candidate to fall back to raw pipeline context when the
    app cannot establish current prompt authority. Do not restore that
    fail-open behavior or duplicate its replacement: the dedicated local
    citation-capture suite already proves authority failure returns no context,
    current-authority exclusion cannot revive legacy bytes without a builder,
    and unsupported external results retain the narrow legacy fallback. Update
    `get_rag_context_capture_for_chat`'s docstring to state that recognized
    candidates require completed current authority regardless of builder
    availability and that only unsupported results retain raw legacy context;
    do not change runtime behavior.
18. In the lazy RAG-admin app fixture, replace direct assignments to the
    retired runtime compatibility fields with
    `_publish_runtime_policy_projection(context.state)`, the same live owner
    seam used by current mounted-app fixtures. Keep the fake policy state and
    all lazy service construction, caching, fallback, and wiring assertions
    unchanged. Do not add a setter or compatibility shim to production.
19. Replace every retired unscoped `tmp_path/.local/share/tldw_cli/backups`
    expectation in the profile-backup integration module with the live
    profile-aware user-data root selected by production. This includes both
    legacy cleanup regressions, which must accept either a missing backup root
    or an existing empty root after cancellation or worker failure. Keep the
    stronger distinct-directory, manifest, partial-failure, no-artifact,
    no-success-notification, and cleared in-progress-state assertions. Do not
    change production cleanup or hard-code the current test profile path.
20. Observe real manifest staging by wrapping the imported
    `create_private_text` owner seam rather than `Path.open`, which the secure
    helper no longer uses on guarded platforms. To exercise cleanup precedence,
    let serialization and secure stage creation complete, then inject the
    ordinary or control-flow failure at the second worker-cancellation check so
    a stage actually exists before unlinking. Keep the exclusive-create,
    worker-thread, control-flow, value-free diagnostic, and cleanup contracts;
    narrow the replace-failure privacy assertion to the injected private
    manifest value rather than the entire test root, whose isolated config path
    is legitimately logged.
21. In both profile-backup setup helpers, install the temporary ChaChaNotes and
    Media database resolvers through the window's live `_DB_PATH_RESOLVERS`
    map, while retaining the existing module-level `get_prompts_db_path` patch
    because `_backup_worker` calls that symbol directly. Production
    deliberately ignores `db_config` in `_get_database_path`; do not restore
    that retired seam or pretend Prompts is routed through the map. Keep the
    real resolver dispatch, copying, manifest-entry, success, and partial
    profile-failure coverage.
22. Remove only `atomic_write_text` from the TTS preference read-purity test's
    monkeypatch list. That symbol was deleted from `config` when persistence
    moved to the private atomic owner; keep guards for all four live public
    settings mutation helpers plus the unchanged input and zero-call
    assertions. Do not restore the implementation symbol or patch private
    file-writing internals that the pure preference parser does not own.
23. In Parakeet MLX result normalization, synthesize an untimed segment only
    when the model returned non-empty text without sentence timestamps. Empty
    text plus no sentences must return an empty segment list even when audio
    duration is known (including `0.0`). Preserve real sentence timestamps,
    non-empty untimed fallback behavior, response metadata, and the existing
    empty-audio regression. Update the one contradictory very-short-audio
    assertion to expect no segment when its mocked model also returns empty
    text and no sentences. This is a direct correction to the established
    cross-provider empty-transcription contract, not a new service boundary.
24. Before loading a Parakeet MLX model, inspect available sound-file metadata.
    When a valid audio container reports zero frames or exactly zero duration,
    return the normal provider result with empty text and segments, resolved
    model/precision/attention metadata, chunk settings, sample rate, duration
    `0.0`, and a terminal progress update. Do not invoke the model loader or
    inference. Metadata-probe failures must continue into normal validation and
    decoding so invalid files still fail clearly; non-empty input behavior is
    unchanged. The mocked zero-frame test must prove the loader was not called,
    while the separate 10 ms empty-model test retains post-inference
    normalization coverage.
25. In the Parakeet MLX no-SoundFile regression, patch the imported `sf` runtime
    object to `None` as well as setting `SOUNDFILE_AVAILABLE` false. The runtime
    module object is the actual branch input; changing only the flag while
    SoundFile is installed lets the nonexistent `dummy.wav` case proceed to a
    real Hugging Face model load. Retain the local `TranscriptionError`
    assertion and do not alter production dependency probing.
26. In the command-palette provider unit regressions, supply a mounted Console
    owner whose current-provider method returns the expected value. For switch
    commands, assert that the provider intent is staged through
    `pending_handoffs` and consumed by that mounted Console owner. Do not assign
    or assert `chat_api_provider_value`; that retired app-root reactive is no
    longer provider authority. Keep production behavior unchanged because the
    production ownership suite already proves live-session, configured-default,
    and off-Console queued-handoff behavior.
27. In the Library ingest option-persistence integration regression, patch the
    live `save_settings_to_cli_config` batch seam. Capture its single
    section-to-values mapping and prove the submitted PDF engine plus generic
    chunk and chunk-size values are present in that same batch. Do not patch
    `save_setting_to_cli_config`, restore per-key persistence, or change
    production; submission deliberately batches these settings to avoid
    repeated config reads, writes, and cache invalidations.
28. In the structured config-mutation module, replace every reference to the
    deleted `config.atomic_write_text` seam with the live
    `config.atomic_private_write_text` owner. Preserve the existing assertions
    for one replacement, zero writes on overlap, contained pre-replacement
    failure, lock serialization, batch-save delegation, delete-wrapper
    delegation, resulting content, and owner-only permissions. In the Phase
    6.6 packaging/data-safety source-seam regression, replace the obsolete
    positive assertion for `atomic_write_text(DEFAULT_CONFIG_PATH` with
    positive assertions scoped to `_write_raw_cli_config_unlocked` for
    `atomic_private_write_text` and its `application_owned_directory` posture;
    retain the existing effective-path assertions. Whole-file substring checks
    are insufficient because unrelated snapshot and bootstrap paths contain
    the same tokens. These tests intentionally instrument or inspect the
    private writer because atomic replacement is their subject; do not restore
    the generic writer, hard-code the default path, or weaken the security
    assertions.
29. In the Console command-composer regression constants, append
    `/generate-image` and `/rewind` in the order returned by the live registry.
    Preserve exact hint-copy assertions and all first-Enter interception,
    second-Enter literal-send, edit-disarm, round-trip-edit, and system-message
    count coverage. Do not derive the expected string by calling the production
    hint formatter inside the test, and do not remove the two registered
    commands from production.
30. In the two Console literal-send regressions, assert that `submit_draft`
    receives both the exact draft and the active session id captured for
    dispatch. In every Console prompt-insert regression, stage text through
    `app.pending_handoffs` on `HandoffChannel.CONSOLE_PROMPT_INSERT` and assert
    terminal consumption with `has_pending`; retain mount/resume timing,
    stale-session draft protection, clean insert, append, collapsed paste,
    blocked notification, and no-op coverage. In the Library prompt editor
    regressions, claim the staged value to verify its exact text and settle the
    claim, while dirty and empty cases prove the channel remains empty. Do not
    restore `pending_console_prompt_insert`, omit the dispatch-time session id,
    or weaken the existing lifecycle and interaction assertions.
31. In all UI regressions that still assign or inspect
    `app.pending_console_launch`, stage the existing launch mapping through
    `app.pending_handoffs` on `HandoffChannel.CONSOLE_LIVE_WORK`. Helper tests
    must claim the staged value, assert the normalized
    `ConsoleLiveWorkLaunch`, and settle the claim. Mounted Console tests must
    assert the channel is no longer pending after consumption while retaining
    rendered status, source, artifact, inspector, navigation, primary-action,
    and staged-context assertions. The Home isolation case must retain its
    absence-of-controls coverage and explicitly prove the Console-owned
    channel remains pending.
    Direct assertions against ChatScreen's private
    `_pending_console_launch_context` remain valid where the screen's accepted
    context itself is under test. Do not restore the app-root launch field or
    change production.
32. In the Library prompt editor, treat a queued `Input.Changed` or
    `TextArea.Changed` event as a mount/recompose echo when the live fields
    still equal the canonical editor state rendered from the current prompt
    detail, or from the active conflict snapshot while conflict controls are
    shown. Follow the existing Skills editor equality-guard pattern rather
    than relying only on `call_after_refresh` ordering. Genuine field changes
    must still set and retain dirty state; successful save and create-conflict
    overwrite must clear dirty state and the Unsaved marker. Retain exact
    clean, empty, and dirty Library-to-Console behavior. Do not clear dirty
    state from tests or add arbitrary pauses that would mask the user-visible
    navigation veto.
33. In the nested Library UI prompt-import harness, route the unrun
    `app_instance` worker manager through the active `LibraryHarness` worker
    manager before pressing Import. Keep the existing bounded status wait,
    button wiring, exact outcome copy, and database assertions; retain the
    real-app tests' app-node/group ownership and survive-unmount assertions.
    Production continues to own import work on the real application so durable
    saves can finish after the initiating screen unmounts. Do not move the
    production worker back to the screen or merely increase the timeout.
34. In the bundled-CSS multi-row Console approval geometry regression, let
    `ChatApprovalCard.on_mount` finish its deferred initial batch-body hide
    before calling `set_batch`. This matches the already-green single-row
    geometry fixture and the live application, where approval batches arrive
    after the mounted Console has settled. Preserve the two-row batch and every
    nonzero-size, fixed-width, non-overlap, compact-height, container-height,
    and action-position assertion. Do not change production layout or weaken
    the CSS contract to accommodate a test-only mount-order inversion.
35. In Schedules and Workflows recent-work regressions, seed the current
    `screen_state_store` under the `RuntimeIdentity` projected from the active
    runtime policy instead of assigning retired `_screen_states`. In the
    Artifacts requested-target regression, stage
    `HandoffChannel.ARTIFACT_CHATBOOK_TARGET`, give the local Chatbook fake the
    existing exact `get_chatbook` service seam, and prove the target is
    consumed before latest-item fallback. In the Home flashcards regression,
    claim the staged `HandoffChannel.STUDY_INITIAL_SECTION`, verify
    `"flashcards"`, acknowledge it, and prove terminal settlement. Preserve
    positive recent-work, exact target/launch payload, requested-before-latest,
    row/button, one-hop navigation, and destination-isolation coverage. Do not
    restore any app-root compatibility field or change production.
36. In the MCP approval-cancellation execution-log regression, assert the
    current metadata-only record: denied decision, blocked/failed outcome, and
    `error_category == "approval_cancelled"`. Retain server/tool identity and
    durable-record coverage. Do not restore free-form error text to the
    persistent MCP audit log; ADR-029 requires bounded metadata rather than
    payload-derived diagnostic strings.
37. In the curated Console remote-default regression, replace only Anthropic's
    retired `claude-sonnet-4-20250514` expectation with
    `claude-sonnet-5`. Retain the independently curated expected mapping and
    catalog-membership assertion for Anthropic, Cohere, Google, and
    HuggingFace. Production configuration, the provider catalog, model
    capabilities, and their dedicated default-model coverage already agree on
    `claude-sonnet-5`; do not revert them or derive the expected mapping from
    the values under test.
38. In the Console runtime-discovery and UI selector merge-cap regressions,
    call `resolve_provider_model_options` with its current explicit
    `providers_models` mapping and catalog scope service positional inputs
    instead of passing an app-shaped object. Retain the same fake data and all
    runtime-discovery ordering/label/warning, merge-cap boundary,
    uncapped-picker, transient current-model, and catalog scope-call
    assertions. The provider-layer resolver suite already pins the explicit
    boundary; do not restore application introspection to production or add a
    compatibility overload.
39. In Console missing-key recovery surfaces, convert the canonical provider
    key to the shared `provider_display_name` only when composing the
    user-facing blocker and Settings tooltip. Preserve canonical lowercase
    provider storage, readiness lookup, routing, recovery target/field, and
    send-blocking behavior. Unknown provider keys continue to display
    unchanged through the shared helper's existing fallback. Do not add a new
    display map or recase endpoint recovery copy outside the observed failure.
40. In the Console empty-transcript “Choose model” action-routing regression,
    explicitly seed the existing OpenAI provider with an empty model before
    mounting the harness. Preserve the rendered live setup action, pointer
    click, and settings-destination assertion. Do not change the settings modal
    or retain a fixture whose blank provider conflicts with the test's stated
    missing-model state.
41. In the live-config Console journey's fake runtime-policy loader, publish
    the existing local `RuntimeSourceState` through
    `TldwCli._publish_runtime_policy_projection` after assigning the fake
    policy context. Remove the unused `current_runtime_source` assignment and
    the now-read-only `current_runtime_backend` assignment. Preserve the real
    config boot, Settings adapter writes, screen navigation, restored session,
    readiness, and no-restart unblocking assertions. Do not make the production
    projection writable or replace the journey with a lighter fake.
42. In the same live-config journey, resolve its owner-only temporary user-data
    directory and give `get_subscriptions_db_path` a file below that directory
    rather than including it in the `:memory:` getter loop. Preserve
    `:memory:` for the unrelated single-thread test databases and preserve the
    real scheduler worker. This lets `SubscriptionsDB` schema initialized on
    the construction thread remain visible when `PriorityQueue.load` queries
    it through `asyncio.to_thread`. Do not disable scheduling, change
    production connection ownership, or point the fixture at host data.
43. In the Console resolution-view regression, remove the retired app-root
    provider/model reactive setup and stage the simulated user provider choice
    on `ChatScreen._console_control_provider`, the current state written by the
    mounted compact-provider handler. Preserve the fresh disk-backed
    llama.cpp-default assertion followed by Anthropic Console-control
    precedence. Update the test name/comments to describe the current owner.
    Do not restore app-reactive coupling or add a compatibility path.
44. In the two skill regressions that wrap
    `ConsoleChatController.submit_draft`, capture the active Console session id
    before sending and include it in all three exact spy assertions. Preserve
    the raw `$code-review` draft text, controller-side skill execution,
    transcript retention, and picker argument/no-argument cases. Do not weaken
    to `ANY`, remove exact text checks, or drop the session-routing contract
    threaded through normal Console sends.
45. In the bare-slash skill-name regression, add `/generate-image` and
    `/rewind` to the independently curated expected unknown-command hint in
    the same registered order used by the already-aligned command-composer
    regressions. Preserve the non-execution, preserved draft, uncalled submit,
    and exact second-Enter arming assertions. Do not derive the expected copy
    from production or change command registration.
46. In `CONSOLE_PARITY_MATRIX["attachments_images"]`, replace the deleted
    `Tests/UI/test_chat_image_attachment.py` file reference with the current
    native Console integration node
    `Tests/UI/test_console_native_chat_flow.py::test_console_attachment_worker_stages_image_and_inlines_text`.
    Retain the existing chat-functions image/RAG reference and the matrix's
    exact file/test existence gate. Confirm no other matrix reference is
    missing. Do not recreate the retired Chat UI test or weaken validation.
47. In the active-conversation workspace marker regression, call
    `_sync_console_workspace_context()` without the obsolete
    `ChatSessionData` argument and remove that now-unused import. Preserve the
    preceding `restore_persisted_session` setup and the exact assertion that
    one selected row contains the restored conversation title. Do not add an
    ignored compatibility parameter or revive the retired legacy tab session
    path already described by the test's rationale.
48. Restore the Watchlists source label's textual indent from two to four
    spaces so a left-aligned child paints strictly to the right of its parent.
    Keep the relative rendered-column assertion unchanged, update only its
    explanatory comment and the source stylesheet explanation, and regenerate
    the bundled stylesheet rather than adding a second geometry rule.
49. Retarget the generic Speech dependency-recovery regression to
    `STTSScreen(_build_test_app())` through its existing screen host. Patch the
    probes at `lab_speech_status`, keep the shared dependency flags false,
    assert the complete independent taxonomy against the inspector detail,
    and assert the exact install tooltip against the rail summary. Do not add
    a compatibility widget to `STTSWindow` or duplicate another harness.
50. Scope the never-run Evals absence assertion to
    `#evals-inspector-bench` before querying `.ds-recovery-callout`. Preserve
    the positive `Not yet checked` and negative Ready/Blocked/Unavailable
    assertions, and leave the unrelated screen-level action-deferral callout
    intact.
51. Query duplicate-target readiness badges beneath
    `#evals-inspector-bench` rather than its broader pane. Preserve the exact
    two-row count, nonzero geometry checks, and all four index-derived editor
    and inspector ids; leave the sibling primary-action status unchanged.
52. Make the local-model dialog hidden by its existing component CSS, remove
    the eager child query from `on_mount()`, and defer reactive show/hide
    application until after refresh through a helper that tolerates a not-yet
    composed child. Exercise hidden-first, show, and hide state in the existing
    real-shell Lab route regression without adding a harness.
53. In both Library Collections branches, replace only the stale disabled
    assertion with an enabled/pressable assertion plus the existing blocked
    class assertion. Preserve selection, copy, item counts, empty guidance,
    geometry, and the dedicated blocked-press coverage without changing
    production or duplicating the handler test.
54. In the different-canvas ingest isolation regression, read the current Media
    label through a test-local nullable helper and use the existing bounded
    `_wait_for_condition` until it remounts with count 1. Preserve the final
    Notes selection and ingest-widget absence assertions; do not change
    production or add another wait abstraction.
55. In the four MCP import-file path regressions, patch
    `mcp_workbench_module._mcp_import_home` instead of the shared
    `os.path.expanduser` attribute. Preserve each temporary containment root and
    all picker-loading, unreadable-file, outside-home rejection, and size-cap
    assertions, and update the affected test rationale to name the seam actually
    patched; do not change production config or private-path handling.
56. Update the test-local MCP audit-record factory to the current metadata-only
    public schema. Seed and parse the rendered inspector detail through the
    existing async UI tests, preserving identity, decision, duration, and
    button coverage while asserting argument-name/count and result-type/size
    metadata. Inject legacy payload fields only in the two privacy regressions
    and assert their values/excerpts/text never render. Do not extract a new
    production helper or restore payload display.
57. In the four Media tests that activate a type, await the widget host app's
    existing worker manager before any action that depends on the initial
    search. In the search-button and pagination tests, also await the newly
    dispatched worker before inspecting its call. Retain the existing pilot
    pause for reactive presentation; do not add sleeps, a polling helper, or a
    production return-value seam.

The only planned production behavior changes outside an ADR-029 diagnostic
correction are the three-name synchronization of the existing Library
collision boundary, the canonical-state guard that prevents untouched prompt
fields from becoming dirty during mount/recompose, and the shared display-name
rendering for missing-key recovery copy, plus the four-space Watchlists child
label indent that restores the already-tested visible hierarchy. The RAG
capture edit is documentation-only and records already-live fail-closed
behavior. No compatibility shims. No broad deletion of live tests.

## Alternatives

- Restoring `StreamDone` or replacing the upstream `TabState` repair would
  contradict the accepted retirement architecture and revive dead ownership.
- Deleting live test files would make collection green but discard useful
  retained non-streaming, shell-label, and audio-error coverage. The one deleted
  file contains no live contract after worker ownership moved to native Console.
- Replacing retired models with local fixtures or duplicating the existing
  streaming-rejection test would keep dead or redundant coverage alive.
- Teaching the retained adapter to strip, log, or consume citation builders
  would revive ownership that moved to native Console.
- Blindly regenerating the diagnostic inventory would defeat its review gate.
- Raising the indexing test's timeout would retain a machine-dependent failure
  with a different arbitrary threshold.
- Deriving the Library shadow set dynamically from the agent registry would
  couple a pure display-state module to runtime ownership and replace the
  intentional fixed collision boundary.
- Replacing the visible Console click with a direct controller call would no
  longer prove that the user-facing Stop control is wired. Merely increasing
  the timeout would not render the control before hit testing.
- Asserting Media closure immediately after the incoming screen becomes active
  conflates two asynchronous Textual lifecycle milestones. Removing the
  closure checks would lose the stale-owner contract; a bounded predicate wait
  preserves it without assuming those milestones complete atomically.
- A single `pilot.pause()` after programmatically assigning Settings controls
  does not guarantee their queued `Changed` messages have staged draft state
  before save. Directly exercising the live handlers and waiting on their
  observable staged/persisted results keeps this ownership test deterministic
  without adding a driver-level form-interaction scenario it does not need.
- Changing the obsolete RAG assertion to expect `None` would duplicate the
  focused citation-capture suite while retaining a misleading DB-free fixture.
  Deleting that fixture and test preserves the authoritative security coverage
  in its actual owner without restoring raw recognized candidates to prompts.
- Restoring a setter for `current_runtime_backend` would reintroduce competing
  runtime authority. Updating the stale fixture to publish its state through
  the existing owner preserves the architectural boundary and the test's real
  lazy-wiring purpose.
- Requiring `backup_root.iterdir()` to succeed after failed-backup cleanup
  treats the presence of an empty implementation directory as product
  behavior. Accepting either absence or emptiness proves the actual
  no-artifact contract without weakening the database, temporary-file,
  manifest, notification, or worker-state assertions.
- Continuing to inspect the retired unscoped root would make no-artifact checks
  pass without observing production and make successful-backup tests fail for
  the wrong reason. Deriving the root from the live owner keeps the assertions
  profile-aware without duplicating its path algorithm.
- Restoring `Path.open` staging or creating a partial disk file during JSON
  serialization would bypass the secure exclusive-create helper. Wrapping that
  helper and injecting failures only after it succeeds preserves the intended
  concurrency and cleanup coverage under the current implementation.
- Putting temporary paths back into `config_data` cannot drive ChaChaNotes or
  Media backup resolution because canonical resolvers now own that boundary.
  Overriding those two entries on the fixture window's resolver map while
  retaining the direct Prompts patch uses the current seams and keeps host
  databases out of the test without changing production.
- Restoring `config.atomic_write_text` or patching the replacement private-file
  implementation would couple a pure parser regression to deleted or internal
  persistence details. The public config mutation guards already express the
  relevant no-write boundary.
- Treating a known duration as sufficient to create an empty Parakeet segment
  produces a meaningless addressable record and contradicts faster-whisper and
  MLX Whisper empty-result behavior. Keying the fallback on non-empty text
  preserves useful untimed transcripts without manufacturing empty content.
- Passing a valid zero-frame buffer into Parakeet MLX underflows its internal
  sequence length and can request an impossible Metal allocation. Recognizing
  the empty container before model load is deterministic, avoids needless
  downloads/resources, and returns the same empty transcription contract
  already expected by callers.
- Treating the availability flag alone as the dependency seam makes the
  no-SoundFile test host-dependent. Patching the imported module object too
  exercises the intended branch and prevents an unrelated network download.
- Restoring the app-root provider reactive or teaching the command provider to
  read it would recreate competing provider authority. Retargeting the unit
  regression to the mounted Console and pending-handoff seams matches the
  production ownership suite without adding compatibility state.
- Restoring per-key Library ingest persistence would undo the accepted
  single-write behavior just to satisfy a stale mock. Observing the batch seam
  directly preserves both the user-facing persistence contract and the current
  write-efficiency boundary.
- Removing config mutation write-count, failure, or lock assertions would lose
  the behavior those tests exist to prove. Wrapping the live private atomic
  writer retains that coverage while following the hardened config owner.
- Building the expected unknown-command hint with the same production method
  under test would hide copy or registry omissions. Updating the two curated
  expected constants keeps the assertion independent and preserves the
  Enter-again safety behavior.
- Accepting a draft-only `submit_draft` assertion would miss the session-switch
  routing guarantee, while restoring a mutable prompt field would bypass claim
  settlement and retry semantics. Following the live session argument and
  typed handoff store preserves both ownership contracts without production
  compatibility state.
- Making the Save/inspector tests assign a screen-private context directly
  would skip the cross-destination ownership boundary. Staging the typed
  launch channel exercises the real claim path and keeps Home from becoming a
  competing consumer.
- Clearing the Library prompt dirty flag in test helpers or adding more pauses
  would hide a real mount-event ordering defect that can veto user navigation.
  Comparing live fields with the canonical rendered state ignores only
  unchanged mount echoes and preserves genuine edits.
- Running prompt imports on the Library screen in production would break the
  accepted durable app-owned worker contract. The failing status assertions
  come from an inactive nested test app's screen stack, so sharing the active
  harness worker manager fixes the fixture without changing runtime ownership.
- Changing approval-card production CSS or removing geometry assertions would
  hide a fixture-ordering mistake: the test installs a batch before the card's
  deferred mount-time hide runs. Settling that callback first mirrors the live
  lifecycle and the existing single-row geometry fixture.
- Restoring `_screen_states` or destination-specific pending attributes would
  recreate competing application state owners. Seeding the current
  `screen_state_store` and typed channels exercises the accepted paths with
  less fixture-only state.
- Restoring the approval-cancellation error sentence to the persistent MCP log
  would violate the metadata-only audit boundary. The bounded
  `approval_cancelled` category preserves the actionable outcome without
  durable free-form text.
- Reverting the configured Anthropic default or deriving the expected Console
  value from production would either undo the reviewed provider refresh or make
  the regression tautological. Updating the one curated literal retains an
  independent drift check against both configuration and catalog membership.
- Restoring app-shaped input support in the provider-model resolver would
  reintroduce application-state coupling removed by the current explicit API.
  Passing the two existing fake values separately preserves every behavior
  assertion without compatibility code.
- Lowercasing the missing-key assertion would ratify a user-facing regression
  caused by canonical provider storage. The shared provider catalog already
  owns branded display names, so using it only at the two failing copy
  boundaries avoids a second map and leaves routing untouched.
- Making the provider Select accept an invalid empty value would broaden
  production behavior to accommodate a routing test that claims to exercise
  “Choose model.” Giving that test the same explicit provider-plus-empty-model
  state used by adjacent missing-model coverage exercises its named contract
  and leaves modal validation intact.
- Restoring writable runtime fields for one live-config fixture would recreate
  retired competing state. Publishing its already-constructed state through
  the same projection method used by the current app harnesses preserves the
  full journey while changing only fixture setup.
- A `:memory:` SQLite database belongs to one connection, while
  `SubscriptionsDB` uses thread-local connections and the production scheduler
  loads subscriptions off-thread. A private file-backed fixture is the
  smallest faithful seam: it shares the initialized schema across those two
  connections without mocking away scheduler behavior.
- Teaching `_effective_console_provider_model` to read app-root reactives again
  would reverse the accepted explicit resolver boundary. Staging the existing
  screen-owned control field keeps this integration check useful without
  duplicating event or compatibility machinery.
- Reverting `submit_draft` to text-only calls would reopen the cross-tab routing
  race fixed by the dispatch-time session contract. Capturing the already-live
  active id keeps the skill tests exact without adding a fixture abstraction.
- Leaving the skill test's copy at four commands makes it disagree with the
  same live registry already pinned by the composer suite. Updating its one
  independent literal preserves the regression without a shared tautological
  helper.
- Recreating a deleted Chat image test solely to satisfy parity metadata would
  revive retired UI ownership. The native Console flow already verifies image
  staging plus text inlining, so pointing the matrix at that exact node keeps
  the gate meaningful with a one-line repair.
- Allowing `_sync_console_workspace_context` to accept and ignore legacy
  `ChatSessionData` would contradict its current owned-state contract. The test
  already restores the native session that the sync reads, so dropping the
  dead argument and import completes its earlier migration.
- Weakening the Watchlists compositor assertion or pinning one absolute column
  would hide the visible parent-child ordering regression or overfit one
  viewport. Restoring four spaces at the existing label seam preserves the
  relative contract without another CSS layout mechanism.
- Restoring the retired Speech status widget inside `STTSWindow` would create a
  second recovery owner after the Lab-frame migration. Deleting the generic
  regression would lose its exact Why/Next/Owner and tooltip assertions, so
  following the existing inspector/rail split preserves distinct coverage with
  one test-only migration.
- Removing the Evals callout assertion or the valid Run Bench deferral would
  weaken different contracts. Querying the existing recovery class beneath its
  target-readiness owner proves the original intent without production changes
  or a broad query-idiom rewrite.
- Filtering duplicate-target badges by text or weakening the exact count would
  make the mount-collision regression less precise. Moving only the query root
  to the existing EvalsInspector owner keeps the shared class and all rendered
  row/id assertions meaningful.
- Removing the local-model dialog or suppressing Models construction would
  discard a live delete flow or hide the real route defect. CSS owning initial
  visibility and the watcher owning later visibility preserves behavior with
  one lifecycle-safe seam and no compatibility surface.
- Re-disabling the Library handoff button would make its recovery handler
  unreachable, while deleting the assertion would lose blocked-state coverage.
  Asserting pressable plus the established blocked class preserves both parts
  of the accepted interaction.
- Catching `NoMatches` broadly or increasing sleeps would obscure whether the
  Media row ever returns. Treating only temporary row absence as a false
  bounded predicate keeps the count and canvas-isolation contracts exact.
- Relaxing the private-file guard or special-casing a directory-valued config
  path would hide a fixture leak and weaken production safety. Patching the
  existing workbench-local import-root seam expresses the test's intended home
  boundary without changing process-wide path resolution.
- Restoring redacted-or-allowlisted payload values in the audit inspector would
  violate the accepted metadata-only persistence and display boundary. A new
  production payload-projection helper is unnecessary: the existing rendered
  UI tests can pin the public schema directly with current fake records.
- Adding more unbounded `pilot.pause()` calls would keep the Media tests
  scheduler-dependent, while changing `_perform_search()` solely to return a
  test handle would alter production for fixture convenience. Textual's worker
  manager already owns the exact completion boundary these tests need.
- The selected edits remove only obsolete assertions, make the audio contracts
  deterministic, retain large-batch correctness coverage, and preserve the
  existing privacy boundary.

## Verification

Run the repaired modules first, then their nearby affected suites, the
diagnostic-inventory guard, static checks on changed files, and the
repository-wide suite. TASK-1333 is complete only when these failures are
absent; unrelated or environment-dependent failures remain recorded rather
than hidden.

## Architecture Decision Record

ADR required: no

ADR path: backlog/decisions/029-local-private-data-boundary.md

Reason: This reconciles tests with already-accepted production boundaries and
applies ADR-029's existing metadata-only inventory review requirement. It does
not introduce a new runtime, storage, security, dependency, or cross-module
decision.
