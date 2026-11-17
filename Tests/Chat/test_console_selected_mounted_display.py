"""Local selected mechanism on the original mounted Console consumer.

The staged real-object fixture is not normal application-startup qualification.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace

import pytest
import pytest_asyncio

from Tests.UI.test_console_native_chat_flow import (
    _build_console_send_test_app,
    _configure_native_ready_console,
    _wait_for_selector,
)
from Tests.Chat.test_console_control_settings import _entered, _hold_full_writer

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

_SELECTED_EAGER_TASK_CONTEXT = pytest.StashKey[object]()
_EXPECTED_DISPLAY_RETIREMENT_ERROR = pytest.StashKey[BaseException]()


def _enter_selected_eager_tasks(request):
    """Retain the stock scheduler through the owning fixture's actual teardown."""
    from Tests.UI.test_console_send_acknowledgement import eager_tasks

    assert request.node.stash.get(_SELECTED_EAGER_TASK_CONTEXT, None) is None
    context = eager_tasks()
    context.__enter__()
    request.node.stash[_SELECTED_EAGER_TASK_CONTEXT] = context


@pytest_asyncio.fixture
async def selected_mounted_console(request, monkeypatch):
    from tldw_chatbook.Chat import console_runtime as runtime_source
    from tldw_chatbook.Chat.console_native_commit import (
        ConsoleNativeRetirementUncertain,
    )
    from tldw_chatbook.Chat.console_domain_services import (
        ConsoleDomainServices,
        ConsoleDomainSources,
    )
    from tldw_chatbook.Chat.console_presentation_port import ConsolePresentationPort
    from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    home_first = getattr(request, "param", None) == "home"
    app = _build_console_send_test_app(
        configured_default="home" if home_first else "chat"
    )
    _configure_native_ready_console(app)
    if getattr(request, "param", None) == "realtime":
        from tldw_chatbook import config

        monkeypatch.setenv("OPENAI_API_KEY", "sk-task61-fake-transport-never-network")
        for section, key, value in (
            ("dictation", "handsfree_engine", "realtime"),
            ("dictation", "acoustic_barge_in", False),
            ("realtime", "enabled", True),
        ):
            assert config.save_setting_to_cli_config(section, key, value)
            app.app_config.setdefault(section, {})[key] = value
    app._wire_chat_conversation_services()
    original = app.console_runtime
    assert original.view is None and original._chat_store is None
    await original.dispose()
    assert original._disposed
    # Capture the one genuine eventual runtime identity before constructing
    # its concrete port. No controller/store/mount/work observes it until init.
    runtime = runtime_source.ConsoleRuntime.__new__(runtime_source.ConsoleRuntime)
    app.console_runtime = runtime
    sources = ConsoleDomainSources(
        configuration=app.app_config,
        chat_database=app.chachanotes_db,
        prompt_database=getattr(app, "prompts_db", None),
        visual_repository=VisualIdentityRepository(app.chachanotes_db),
        scratch=ConsoleScratchSpaceManager(),
        conversation_scope_service=app.chat_conversation_scope_service,
        settings_durability_owner=app.console_settings_durability_owner,
        control_plane=app.unified_mcp_service,
        policy_enforcer=app.service_policy_enforcer,
    )
    port = ConsolePresentationPort(app)
    domain = ConsoleDomainServices(
        sources, runtime_source=runtime_source, presentation=port
    )
    runtime_source.ConsoleRuntime.__init__(runtime, app, domain_services=domain)
    port.require_association(port.association)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await app._initial_screen_setup_task
            if home_first:
                assert type(app.screen).__name__ == "HomeScreen"
                assert runtime.view is None and runtime._chat_store is None
                store = runtime.ensure_chat_store()
                controller = runtime.ensure_chat_controller(
                    store=store,
                    provider_gateway=runtime.ensure_provider_gateway(),
                    agent_runtime_enabled=False,
                    provider=app.chat_api_provider_value,
                    model=app.chat_api_model_value,
                )
                permission_store = await asyncio.to_thread(
                    lambda: sources.control_plane.permission_store
                )
                assert (
                    permission_store is not None
                    and sources.control_plane._permission_store is permission_store
                )
                controls = runtime._execution_service.open_local_controls(
                    profile_epoch="resume-none",
                    service_epoch="local",
                )
                assert (
                    controls._store is store and runtime.chat_controller is controller
                )
                assert (
                    runtime.view is None
                    and store.active_session_id is None
                    and not store.sessions()
                )
                yield app, runtime, None, pilot
                return
            screen = None
            for _ in range(200):
                screen = app._navigation_outgoing_screen()
                if type(screen) is ChatScreen:
                    break
                await pilot.pause(0.05)
            assert type(screen) is ChatScreen
            await _wait_for_selector(screen, pilot, "#console-native-composer")
            assert type(screen) is ChatScreen
            for _ in range(50):
                if (
                    screen._console_attach_reconciled
                    and not screen._console_attach_reconcile_running
                    and not screen._console_sync_in_progress
                    and not getattr(
                        screen, "_console_control_bar_replay_whole_sync", False
                    )
                ):
                    break
                await pilot.pause(0.1)
            assert (
                screen._console_attach_reconciled
                and not screen._console_attach_reconcile_running
            )
            assert not screen._console_sync_in_progress
            assert not getattr(screen, "_console_control_bar_replay_whole_sync", False)
            assert screen._console_runtime() is runtime and runtime.view is screen
            port.require_association(port.association)
            yield app, runtime, screen, pilot
    except ConsoleNativeRetirementUncertain as error:
        if error is not request.node.stash.get(
            _EXPECTED_DISPLAY_RETIREMENT_ERROR, None
        ):
            raise
        request.node.user_properties.append(
            (
                "app_shutdown_retirement",
                "exact retained ConsoleNativeRetirementUncertain",
            )
        )
    finally:
        try:
            try:
                await runtime.dispose()
            except ConsoleNativeRetirementUncertain as error:
                if error is not request.node.stash.get(
                    _EXPECTED_DISPLAY_RETIREMENT_ERROR, None
                ):
                    raise
                request.node.user_properties.append(
                    (
                        "runtime_dispose_retirement",
                        "exact retained ConsoleNativeRetirementUncertain",
                    )
                )
            else:
                assert (
                    request.node.stash.get(_EXPECTED_DISPLAY_RETIREMENT_ERROR, None)
                    is None
                )
        finally:
            context = request.node.stash.get(_SELECTED_EAGER_TASK_CONTEXT, None)
            if context is not None:
                del request.node.stash[_SELECTED_EAGER_TASK_CONTEXT]
                context.__exit__(None, None, None)


async def test_original_mounted_full_retains_nested_default_convergence(
    selected_mounted_console, request
):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        ReadProjection,
    )
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
    )
    from tldw_chatbook.Chat import console_parent_settings as parent_settings

    from Tests.UI.test_console_send_acknowledgement import eager_tasks

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    # Observe the actual finite owners in this Task, beyond fixture handoff.
    # No producer is warmed, cancelled, or disabled to manufacture quiet.
    for _ in range(50):
        readiness = getattr(screen, "_console_readiness_config_projection", None)
        context = getattr(screen, "_console_context_read_snapshot", None)
        settled = not (
            screen._console_sync_in_progress
            or getattr(screen, "_console_control_bar_replay_whole_sync", False)
            or screen._console_control_bar_sync_scheduled
            or getattr(readiness, "pending", False)
            or getattr(context, "pending_key", None) is not None
            or controls._parent_settings_captures
            or controls._session_capture_tails
            or any(
                worker.group
                in {
                    "console-sync",
                    "console-readiness-config",
                    "console-readiness-publication",
                }
                and not worker.is_finished
                for worker in app.workers
            )
        )
        if settled:
            break
        await pilot.pause(0.1)
    assert settled, "Original Console publication owners did not settle"
    session = store._sessions[store.active_session_id]
    assert session.persisted_conversation_id is None and not session.has_user_work
    settings = replace(
        blank_console_session_settings(app.app_config), model="older-pristine-model"
    )
    store.replace_session_settings(
        session.id, settings, mark_user_work=False, canonical_settings_baseline=settings
    )
    assert not session.has_user_work
    conversation_id = store.persistence.create_conversation(
        conversation_title="Nested default convergence"
    )
    store.publish_first_persisted_conversation(session.id, conversation_id)
    assert store.active_session_id == session.id
    identity = ControlIdentity(
        "mounted", "local", 1, "nested-display", session.id, session.incarnation_id
    )
    before = store.session_settings_revision(session.id)
    replay_release = asyncio.Event()
    replay_tasks = []
    replay_captures = []
    readiness_entries = []

    async def delayed_replay():
        await replay_release.wait()
        assert parent_settings.settings_inline_source(runtime) is None
        await parent_settings.prepare_parent_default_settings(
            screen, purpose="early-full-defaults"
        )

    from Tests.Chat.test_console_control_trace import _observe

    full_returns = []
    full_entries = []
    core_pushes = []
    warm_returns = []
    from tldw_chatbook.UI.Console_Modules import config_sync

    def observe_core_push(frame):
        if (
            frame.f_locals.get("self") is screen
            and frame.f_back.f_code is config_sync._push_console_core_state.__code__
        ):
            core_pushes.append(id(asyncio.current_task()))

    def observe_warm_return(frame, values):
        if frame.f_locals["self"].screen is screen:
            warm_returns.append((id(asyncio.current_task()), values[-1]))

    def full_owner(frame):
        task = asyncio.current_task()
        source = parent_settings.settings_inline_source(runtime)
        return dict(
            frame=id(frame),
            task=id(task),
            code_file=frame.f_code.co_filename,
            code_line=frame.f_code.co_firstlineno,
            command_task=controls._settings_command_tasks.get(task) is not None,
            source_is_command=source is not None and source.identity is identity,
        )

    def observe_original_full_entry(frame):
        if frame.f_locals.get("self") is screen:
            full_entries.append(full_owner(frame))

    def observe_original_parent(frame):
        if (
            not replay_tasks
            and parent_settings.settings_inline_source(runtime) is not None
        ):
            # Created by the actual command, but resumed only after retirement.
            replay_tasks.append(asyncio.create_task(delayed_replay()))

    def observe_original_capture(frame):
        capture = frame.f_locals["capture"]
        if capture.port._view is screen:
            replay_captures.append(capture)
            if capture.purpose == "readiness-defaults":
                dependency = next(
                    (
                        d
                        for d in controls._settings_readiness_captures.values()
                        if d.capture is capture
                    ),
                    None,
                )
                if dependency is not None:
                    readiness_entries.append(
                        (
                            capture,
                            dependency,
                            controls._settings_readiness_waits.get(dependency.producer),
                        )
                    )

    def observe_original_full_return(frame, values):
        if frame.f_locals.get("self") is screen:
            full_returns.append(
                dict(
                    **full_owner(frame),
                    line=frame.f_lineno,
                    value=values[-1],
                    prepared=frame.f_locals.get("prepared"),
                    held=frame.f_locals.get("held"),
                    maintenance=getattr(
                        screen, "_console_sync_maintenance_paused", False
                    ),
                    whole_replay=getattr(
                        screen, "_console_control_bar_replay_whole_sync", False
                    ),
                    in_progress=screen._console_sync_in_progress,
                )
            )

    try:
        with (
            eager_tasks(),
            _observe(
                type(screen)._sync_console_chat_core_state.__code__,
                sys.monitoring.events.PY_RETURN,
                observe_core_push,
            ),
            _observe(
                ConsoleReadinessConfigProjection.warm.__code__,
                sys.monitoring.events.PY_RETURN,
                observe_warm_return,
                values=True,
            ),
            _observe(
                type(screen)._sync_native_console_chat_ui.__code__,
                sys.monitoring.events.PY_START,
                observe_original_full_entry,
            ),
            _observe(
                parent_settings.prepare_parent_default_settings.__code__,
                sys.monitoring.events.PY_START,
                observe_original_parent,
            ),
            _observe(
                parent_settings.run_parent_settings_capture.__code__,
                sys.monitoring.events.PY_START,
                observe_original_capture,
            ),
            _observe(
                type(screen)._sync_native_console_chat_ui.__code__,
                sys.monitoring.events.PY_RETURN,
                observe_original_full_return,
                values=True,
            ),
        ):
            with _hold_full_writer(store.persistence) as (entered, release, calls):
                receipt = controls.dispatch(ReadProjection(identity))
                body_error = None
                try:
                    await _entered(entered)
                    pending = settings_pending(store, session.id)
                    assert pending is not None and pending.convergence is not None
                    assert not receipt.applied.done(), (
                        type(pending.source).__name__,
                        full_entries,
                        full_returns,
                    )
                    if getattr(pending.source, "identity", None) is identity:
                        route = "inline-command"
                    else:
                        assert (
                            type(pending.source)
                            is parent_settings.ParentSettingsCapture
                        )
                        source = pending.source
                        dependencies = [
                            d
                            for d in controls._settings_readiness_captures.values()
                            if d.capture is source
                        ]
                        assert len(dependencies) == 1, (
                            source.purpose,
                            full_entries,
                            full_returns,
                        )
                        dependency = dependencies[0]
                        observed = [
                            entry
                            for cap, dep, entry in readiness_entries
                            if cap is source and dep is dependency
                        ]
                        assert len(observed) == 1 and observed[0] is not None, (
                            source.purpose,
                            readiness_entries,
                        )
                        entry = observed[0]
                        original, owner, projection, handoff, key = entry
                        live_entry = controls._settings_readiness_waits.get(
                            dependency.producer
                        )
                        if live_entry is not entry:
                            # Eligibility is intentionally revoked before the original
                            # context exit joins its already-entered native producer.
                            coroutine = owner.get_coro()
                            exits = []
                            while coroutine is not None:
                                frame = getattr(coroutine, "cr_frame", None)
                                context = (
                                    frame.f_locals.get("self")
                                    if frame is not None
                                    else None
                                )
                                generator = getattr(context, "gen", None)
                                if (
                                    getattr(generator, "ag_code", None)
                                    is parent_settings.await_readiness_settings.__wrapped__.__code__
                                ):
                                    state = generator.ag_frame.f_locals
                                    if (
                                        state.get("entry") is entry
                                        and state.get("child") is dependency.producer
                                        and state.get("dependency") is dependency
                                        and state.get("entered") is True
                                    ):
                                        exits.append(generator)
                                        exception_type = frame.f_locals.get("typ")
                                        request.node.user_properties.append(
                                            (
                                                "task61_readiness_wait_exit",
                                                "normal-settled-event"
                                                if exception_type is None
                                                else exception_type.__name__,
                                            )
                                        )
                                coroutine = getattr(coroutine, "cr_await", None)
                            assert len(exits) == 1, (
                                live_entry,
                                owner.get_stack(),
                                full_returns,
                            )
                        assert (
                            source.purpose == "readiness-defaults"
                            and source.pending_operation is pending
                        )
                        assert (
                            original.identity is identity
                            and owner is receipt.work
                            and not owner.done()
                        )
                        assert controls._settings_command_tasks[owner] is original
                        assert (
                            projection is dependency.projection
                            and projection.screen is screen
                        )
                        assert handoff is dependency.handoff and key is dependency.key
                        assert (
                            handoff[2]._task is dependency.producer
                            and not dependency.producer.done()
                        )
                        assert source.sources is original.sources
                        assert key[5] == source.session_id == session.id
                        assert source.incarnation_id == session.incarnation_id
                        route = "command-awaits-original-W-registered-B"
                    request.node.user_properties.append(
                        ("task61_full_settings_route", route)
                    )
                    assert (
                        pending.retirement is not None
                        and not pending.retirement.confirmed
                    )
                    assert store.session_settings_revision(session.id) == before
                except BaseException as error:
                    body_error = error
                    raise
                finally:
                    release.set()
                    async with asyncio.timeout(10):
                        try:
                            await receipt.applied
                            if body_error is not None:
                                body_error.add_note(
                                    f"Applied succeeded; actual FULL entries: {full_entries!r}; returns: {full_returns!r}"
                                )
                        except BaseException as error:
                            error.add_note(
                                f"Actual FULL entries: {full_entries!r}; returns: {full_returns!r}"
                            )
                            replay_release.set()
                            if body_error is not None:
                                body_error.add_note(
                                    str(error) + " " + repr(error.__notes__)
                                )
                                raise body_error from error
                            raise
            assert pending.retirement.confirmed and pending.commit is not None
            request.node.user_properties.append(
                ("task61_command_core_push", id(receipt.work) in core_pushes)
            )
            request.node.user_properties.append(
                (
                    "task61_command_warm_returns",
                    [value for task, value in warm_returns if task == id(receipt.work)],
                )
            )
            assert any(
                entry["task"] == id(receipt.work) and entry["source_is_command"]
                for entry in full_entries
            )
            assert any(
                result["task"] == id(receipt.work) and result["value"] is True
                for result in full_returns
            )
            assert replay_tasks and not replay_tasks[0].done()
            assert receipt.work.done() and not controls._settings_command_tasks
            replay_release.set()
            async with asyncio.timeout(10):
                await replay_tasks[0]
            assert any(
                capture.purpose == "early-full-defaults" and capture.started
                for capture in replay_captures
            )
            assert not controls._parent_settings_captures
    finally:
        replay_release.set()
    projection = ConsoleReadinessConfigProjection.for_screen(screen)
    assert calls == ["local-model"]
    assert session.settings.model == "local-model"
    assert store.session_settings_revision(session.id) == before + 1
    assert projection.key == projection._key()
    assert screen._session._default_convergences[session.id] == (before, before + 1)
    assert not controls._effects


async def test_parent_capture_admission_cancel_refusal_and_fence(
    selected_mounted_console, monkeypatch
):
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity, FenceSource
    from tldw_chatbook.Chat.console_parent_settings import (
        issue_parent_settings_capture,
        require_parent_settings_capture,
    )

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="parent", service_epoch="local"
    )
    capture = issue_parent_settings_capture(controls, "settings-render-replay", None)
    assert capture.effect in controls._effects
    with pytest.raises(RuntimeError, match="owner changed"):
        require_parent_settings_capture(replace(capture))
    controls._queue_session_capture(capture, capture)
    cancelled_tail = controls._session_capture_tails[session.id]
    capture.work.cancel()
    async with asyncio.timeout(10):
        assert await capture.retired
    assert capture.applied.cancelled()
    assert (
        capture.effect not in controls._effects
        and capture not in controls._parent_settings_captures
    )
    assert capture.effect.completed.result() and capture.pending_operation is None
    assert (
        cancelled_tail.done()
        and controls._session_capture_tails.get(session.id) is not cancelled_tail
    )
    with pytest.raises(RuntimeError, match="owner changed"):
        require_parent_settings_capture(capture)

    original_create = controls._loop.create_task
    for refused_name in (
        "_retain_control_work.<locals>.execute",
        "_queue_session_capture.<locals>.capture",
    ):
        capture = issue_parent_settings_capture(
            controls, "settings-render-replay", None
        )

        def refuse(
            coroutine, *args, refused_name=refused_name, capture=capture, **kwargs
        ):
            locals_ = coroutine.cr_frame.f_locals
            if coroutine.cr_code.co_qualname.endswith(refused_name) and (
                locals_.get("receipt") is capture or locals_.get("command") is capture
            ):
                raise RuntimeError("injected parent task scheduling refusal")
            return original_create(coroutine, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(controls._loop, "create_task", refuse)
            with pytest.raises(
                RuntimeError, match="injected parent task scheduling refusal"
            ):
                controls._queue_session_capture(capture, capture)
                async with asyncio.timeout(10):
                    await capture.applied
        assert capture.applied.done() and capture.retired.result()
        assert (
            capture.effect not in controls._effects
            and capture not in controls._parent_settings_captures
        )
        assert capture.effect.completed.result() and capture.pending_operation is None
        assert capture.operation_task is None

    capture = issue_parent_settings_capture(controls, "settings-render-replay", None)
    fence = controls.dispatch(
        FenceSource(
            ControlIdentity(
                "parent", "local", 1, "parent-fence", session.id, session.incarnation_id
            ),
            "configuration",
        )
    )
    controls._queue_session_capture(capture, capture)
    async with asyncio.timeout(10):
        with pytest.raises(RuntimeError, match="fenced"):
            await capture.applied
        assert await capture.retired
        await fence.applied
    assert (
        capture.effect not in controls._effects
        and capture not in controls._parent_settings_captures
    )
    assert capture.effect.completed.result() and capture.pending_operation is None
    assert (
        not controls._effects
    )  # The completed source fence drains every admitted effect.
    with pytest.raises(RuntimeError, match="fenced"):
        issue_parent_settings_capture(controls, "settings-render-replay", None)


async def _entered_action(event, capture, controls):
    try:
        async with asyncio.timeout(10):
            while not event.is_set():
                if capture.applied.done():
                    capture.applied.result()
                await asyncio.sleep(0.01)
    except TimeoutError:
        pytest.fail(
            f"Original writer entry did not arrive: purpose={capture.purpose}, "
            f"started={capture.started}, work_done={capture.work.done()}, "
            f"operation_task_done={capture.operation_task.done() if capture.operation_task else None}, "
            f"effects={len(controls._effects)}, tails={len(controls._session_capture_tails)}"
        )


async def test_selected_compact_and_prompt_actions_keep_order_and_full_post_on_cancel(
    selected_mounted_console,
):
    from Tests.Backup_Recovery.test_console_main_display_batch import (
        _worker_connections,
    )
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    store.replace_session_settings(
        session.id, replace(session.settings, temperature=0.3)
    )
    conversation = store.persistence.create_conversation(
        conversation_title="Ordered parent actions"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="actions", service_epoch="local"
    )
    before = session.settings_revision
    posts = []

    def original_post(frame):
        if frame.f_locals["self"] is screen:
            posts.append((asyncio.current_task(), dict(frame.f_locals["updates"])))

    with _observe(
        ChatScreen._finish_compact_shell_controls.__code__,
        sys.monitoring.events.PY_START,
        original_post,
    ):
        with (
            _worker_connections(store.persistence.db) as connections,
            _hold_full_writer(store.persistence) as (entered, release, calls),
        ):
            screen._sync_compact_shell_controls(temperature="0.4")
            first = next(
                item
                for item in controls._parent_settings_captures
                if item.purpose == "compact-controls"
            )
            screen._sync_compact_shell_controls(model="later-local-model")
            second = next(
                item
                for item in controls._parent_settings_captures
                if item.purpose == "compact-controls" and item is not first
            )
            try:
                await _entered_action(entered, first, controls)
                pending = settings_pending(store, session.id)
                assert pending is not None and pending.source is first
                assert worker_leases(store.persistence.db) and connections
                assert (
                    session.settings_revision == before
                    and session.settings.temperature == 0.4
                )
                assert not second.applied.done() and not posts
                credential_gate = getattr(screen, "_console_credential_pre_gate", None)
                screen._poll_console_credential_readiness()
                screen._sync_console_workbench_actions_from_draft()
                screen._sync_console_transcript_guidance()
                screen._context_spend._refresh_settings_summary()
                screen._context_spend._refresh_cost_chip()
                assert (
                    getattr(screen, "_console_credential_pre_gate", None)
                    == credential_gate
                )
                screen._prompts._apply_console_system_prompt_editor_result(
                    "  original prompt spacing  "
                )
                prompt = next(
                    item
                    for item in controls._parent_settings_captures
                    if item.purpose == "prompt-editor-result"
                )
                first.work.cancel()
                await asyncio.sleep(0)
                first.work.cancel()
                await asyncio.sleep(0)
                assert (
                    not first.retired.done()
                    and not second.applied.done()
                    and not prompt.applied.done()
                )
                assert app._exception is None
            finally:
                release.set()
            async with asyncio.timeout(15):
                with pytest.raises(asyncio.CancelledError):
                    await first.applied
                await second.applied
                await prompt.applied
                assert (
                    await first.retired
                    and await second.retired
                    and await prompt.retired
                )
            assert calls == ["local-model", "later-local-model"]
            assert posts == [
                (first.operation_task, {"temperature": "0.4"}),
                (second.operation_task, {"model": "later-local-model"}),
            ]
            assert (
                session.settings.temperature == 0.4
                and session.settings.model == "later-local-model"
            )
            assert session.settings.system_prompt == "  original prompt spacing  "
            assert session.settings_revision == before + 3
            assert not settings_pending(store, session.id)
            assert (
                not store._fork_source_transitions
                and not store._voice_promotion_mutation_admissions
            )
            assert pending.commit is session and pending.retirement.confirmed
            import sqlite3

            for operation in (
                pending,
                second.pending_operation,
                prompt.pending_operation,
            ):
                connection = operation.retirement.connection
                assert operation.retirement.confirmed and connection in connections
                with pytest.raises(sqlite3.ProgrammingError):
                    sqlite3.Connection.in_transaction.__get__(connection)
                assert operation.retirement.lease not in worker_leases(
                    store.persistence.db
                )


async def test_parent_view_switch_keeps_exact_old_backend_completion(
    selected_mounted_console,
):
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
    from tldw_chatbook.Chat.console_control_settings import settings_pending

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    store.replace_session_settings(
        session.id, replace(session.settings, temperature=0.3)
    )
    conversation = store.persistence.create_conversation(
        conversation_title="Old parent action"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="switch", service_epoch="local"
    )
    before = session.settings_revision
    with _hold_full_writer(store.persistence) as (entered, release, calls):
        screen._sync_compact_shell_controls(temperature="0.4")
        capture = next(
            item
            for item in controls._parent_settings_captures
            if item.purpose == "compact-controls"
        )
        try:
            await _entered_action(entered, capture, controls)
            pending = settings_pending(store, session.id)
            assert pending is not None and worker_leases(store.persistence.db)
            waiting_intents = [
                asyncio.create_task(screen._consume_settled_vllm_console_intent()),
                asyncio.create_task(screen._consume_settled_llamacpp_console_intent()),
            ]
            await asyncio.sleep(0)
            assert all(
                task in controls._control_tasks and not task.done()
                for task in waiting_intents
            )
            other = store.create_session(settings=session.settings)
            assert store.active_session_id == other.id and other is not session
        finally:
            release.set()
        async with asyncio.timeout(15):
            with pytest.raises(RuntimeError, match="presentation changed"):
                await capture.applied
            assert await capture.retired
            assert await asyncio.gather(*waiting_intents) == [False, False]
        assert app._exception is None
        assert calls == ["local-model"]
        assert pending.commit is session and pending.retirement.confirmed
        assert session.settings_revision == before + 1
        assert not settings_pending(store, session.id)
        assert (
            not store._fork_source_transitions
            and not store._voice_promotion_mutation_admissions
        )
        assert pending.retirement.lease not in worker_leases(store.persistence.db)
        import sqlite3

        with pytest.raises(sqlite3.ProgrammingError):
            sqlite3.Connection.in_transaction.__get__(pending.retirement.connection)


async def test_cold_compact_action_retains_its_actual_readiness_producer(
    selected_mounted_console,
):
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_parent_settings as parent
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    settings = replace(
        blank_console_session_settings(app.app_config), model="older-pristine-model"
    )
    session = store.create_session(
        settings=settings, canonical_settings_baseline=settings
    )
    conversation = store.persistence.create_conversation(
        conversation_title="Cold ordered action"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="cold-action", service_epoch="local"
    )
    assert config.save_setting_to_cli_config(
        "console.rail_state", "active_rail", "sessions"
    )
    before = session.settings_revision
    with _hold_full_writer(store.persistence) as (entered, release, calls):
        screen._sync_compact_shell_controls(temperature="0.4")
        capture = next(
            item
            for item in controls._parent_settings_captures
            if item.purpose == "compact-controls"
        )
        try:
            await _entered_action(entered, capture, controls)
            pending = settings_pending(store, session.id)
            assert (
                pending is not None
                and pending.source is capture
                and pending.convergence is not None
            )
            assert pending.retirement is not None and worker_leases(
                store.persistence.db
            )
            entries = [
                (task, entry)
                for task, entry in controls._settings_readiness_waits.items()
                if entry[0] is capture
            ]
            assert len(entries) == 1
            child, (_, owner, projection, handoff, key) = entries[0]
            assert owner is capture.operation_task and child is handoff[2]._task
            assert projection._handoff_read is handoff and key[5] == session.id
            assert (
                child is not owner and not child.done() and not capture.applied.done()
            )
            assert (
                parent.settings_inline_parent(runtime) is None
            )  # This test Task has no authority.
            assert (
                not projection._reads
            )  # Config native read retired before settings replacement.
        finally:
            release.set()
        async with asyncio.timeout(15):
            await capture.applied
            assert await capture.retired
        assert calls == ["local-model", "local-model"]
        assert (
            session.settings.temperature == 0.4
            and session.settings.model == "local-model"
        )
        assert session.settings_revision == before + 2
        assert all(
            entry[0] is not capture
            for entry in controls._settings_readiness_waits.values()
        )
        assert not settings_pending(store, session.id) and app._exception is None


@pytest.mark.parametrize(
    "mode",
    [
        "complete",
        "cancel_waiter_after_entry",
        "cancel_action_before_entry",
        "stale_before_ensure",
        "expire_before_settlement",
    ],
)
async def test_existing_readiness_capture_keeps_intervening_user_order(
    selected_mounted_console, mode
):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_parent_settings as parent
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        FenceSource,
        UpdateInputs,
    )
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
    )

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    settings = replace(
        blank_console_session_settings(app.app_config), model="older-pristine-model"
    )
    session = store.create_session(
        settings=settings, canonical_settings_baseline=settings
    )
    conversation = store.persistence.create_conversation(
        conversation_title="Existing readiness dependency"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="dependency", service_epoch="local"
    )
    projection = ConsoleReadinessConfigProjection.for_screen(screen)
    assert config.save_setting_to_cli_config(
        "console.rail_state", "active_rail", "sessions"
    )
    staged = {}

    def identity(number):
        return ControlIdentity(
            "dependency",
            "local",
            number,
            f"dependency-{number}",
            session.id,
            session.incarnation_id,
        )

    def admit_original_actions(frame):
        if (
            staged
            or frame.f_locals.get("screen") is not screen
            or frame.f_locals.get("purpose") != "readiness-defaults"
        ):
            return
        worker = asyncio.current_task()
        handoff = projection._handoff_read
        assert handoff is not None and handoff[2]._task is worker
        assert parent.settings_inline_source(runtime) is None
        assert parent.settings_inline_parent(runtime) is None
        screen._sync_compact_shell_controls(temperature="0.4")
        action = next(
            item
            for item in controls._parent_settings_captures
            if item.purpose == "compact-controls"
        )
        first_tail = controls._session_capture_tails[session.id]
        middle = controls.dispatch(
            UpdateInputs(identity(1), "middle user input", (1, 1))
        )
        middle_tail = controls._session_capture_tails[session.id]
        staged.update(
            worker=worker,
            handoff=handoff,
            action=action,
            first_tail=first_tail,
            middle=middle,
            middle_tail=middle_tail,
        )
        if mode == "expire_before_settlement":
            from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
                ConsoleReadinessConfigRead,
            )

            caller = frame.f_back
            assert caller.f_code is type(projection)._refresh.__code__
            result = caller.f_locals["result"]
            assert (
                type(result) is ConsoleReadinessConfigRead
                and result.value is frame.f_locals["checked_config"]
            )
            staged["original_result"] = result
        if mode == "cancel_action_before_entry":

            def cancel_actual_inner():
                assert action.operation_task is not None and not action.started
                action.operation_task.cancel()

            controls._loop.call_soon(cancel_actual_inner)

    def observe_original_defaults(frame):
        capture = frame.f_locals["capture"]
        if (
            staged
            and capture.purpose == "readiness-defaults"
            and capture.port._view is screen
            and capture.session_id == session.id
        ):
            assert (
                controls._settings_readiness_captures[staged["worker"]].capture
                is capture
            )
            staged["defaults"] = capture

    prepare_results, due_refusals, ensure_consumptions, action_posts = [], [], [], []

    def observe_prepare_return(frame, values):
        if staged and asyncio.current_task() is staged["action"].operation_task:
            prepare_results.append(
                (
                    values[-1],
                    frame.f_locals.get("settings"),
                    frame.f_locals.get("next_settings"),
                )
            )

    def observe_action_post(frame):
        if staged and asyncio.current_task() is staged["action"].operation_task:
            action_posts.append(frame.f_locals["updates"])

    def observe_ensure_consumption(frame):
        task = asyncio.current_task()
        request = controls._due_settings_reads.get(task)
        if staged and request is not None and request.source is staged["action"]:
            ensure_consumptions.append((task, request, request.consumed))
            if mode == "stale_before_ensure":
                assert (
                    len(ensure_consumptions) == 1
                    and request.active
                    and not request.consumed
                )
                assert request.result is not None and not request.projection._reads
                # Actual local config publication after the exact read settled,
                # before original ensure consumes it. Never replace a reader.
                assert config.save_setting_to_cli_config(
                    "console.rail_state", "active_rail", "context"
                )

    def observe_due_refusal(frame, values):
        request = frame.f_locals.get("request")
        if staged and request is not None and request.source is staged["action"]:
            due_refusals.append((type(values[-1]).__name__, str(values[-1])))

    with (
        _observe(
            parent.prepare_parent_default_settings.__code__,
            sys.monitoring.events.PY_START,
            admit_original_actions,
        ),
        _observe(
            parent.run_parent_settings_capture.__code__,
            sys.monitoring.events.PY_START,
            observe_original_defaults,
        ),
        _observe(
            type(screen)._prepare_compact_shell_controls.__code__,
            sys.monitoring.events.PY_RETURN,
            observe_prepare_return,
            values=True,
        ),
        _observe(
            parent._due_settings_read_current.__code__,
            sys.monitoring.events.RAISE,
            observe_due_refusal,
            values=True,
        ),
        _observe(
            parent.consume_due_settings_result.__code__,
            sys.monitoring.events.PY_START,
            observe_ensure_consumption,
        ),
        _observe(
            type(screen)._finish_compact_shell_controls.__code__,
            sys.monitoring.events.PY_START,
            observe_action_post,
        ),
    ):
        with _hold_full_writer(store.persistence) as (entered, release, calls):
            projection.run(
                lambda: None
            )  # The original autonomous producer starts first.
            try:
                async with asyncio.timeout(10):
                    while not staged:
                        await asyncio.sleep(0.01)
                action = staged["action"]
                if mode == "cancel_action_before_entry":
                    async with asyncio.timeout(10):
                        with pytest.raises(asyncio.CancelledError):
                            await action.applied
                        await staged["middle"].applied
                        await staged["worker"]
                        original = staged["defaults"]
                        assert await action.retired and await original.retired
                        last = controls.dispatch(
                            UpdateInputs(identity(2), "last user input", (1, 2))
                        )
                        await last.applied
                        await controls.dispatch(
                            FenceSource(identity(3), "configuration")
                        ).applied
                    assert action.operation_task.cancelled() and not action.started
                    assert (
                        session.has_user_work
                    )  # The original middle user edit blocks default convergence.
                    assert original.started and original.pending_operation is None
                    assert (
                        session.settings is settings
                        and not entered.is_set()
                        and calls == []
                    )
                    assert staged["middle"].applied.result() < last.applied.result()
                    assert (
                        store.session_input_snapshot(session.id).draft
                        == "last user input"
                    )
                    assert all(
                        tail.done() and tail not in controls._capture_predecessors
                        for tail in (staged["first_tail"], staged["middle_tail"])
                    )
                    assert staged["worker"] not in controls._settings_readiness_captures
                    assert not controls._effects and app._exception is None
                    return
                async with asyncio.timeout(10):
                    while not entered.is_set():
                        if projection._settled.is_set():
                            pytest.fail(
                                f"Existing producer settled without its writer: action_done={action.applied.done()}"
                            )
                        await asyncio.sleep(0.01)
                dependency = controls._settings_readiness_captures[staged["worker"]]
                original = dependency.capture
                pending = settings_pending(store, session.id)
                assert (
                    pending is not None
                    and pending.source is original
                    and pending.convergence is not None
                )
                assert (
                    original is not action
                    and original.operation_task is not action.operation_task
                )
                assert (
                    dependency.handoff is staged["handoff"]
                    and dependency.producer is staged["worker"]
                )
                assert worker_leases(store.persistence.db) and not projection._reads
                last = controls.dispatch(
                    UpdateInputs(identity(2), "last user input", (1, 2))
                )
                last_tail = controls._session_capture_tails[session.id]
                assert not staged["middle"].applied.done()
                assert controls._settings_readiness_waits[staged["worker"]][0] is action
                if mode == "cancel_waiter_after_entry":
                    action.operation_task.cancel()
                    async with asyncio.timeout(5):
                        while staged["worker"] in controls._settings_readiness_waits:
                            await asyncio.sleep(0)
                    assert not action.operation_task.done()
                    assert (
                        not staged["first_tail"].done()
                        and not staged["middle"].applied.done()
                    )
                if mode == "expire_before_settlement":
                    import time

                    proof = staged["original_result"]._display_proof
                    async with asyncio.timeout(5):
                        while time.monotonic() - proof.at < projection.max_age:
                            await asyncio.sleep(0.01)
                    assert (
                        not action.applied.done()
                        and not staged["middle"].applied.done()
                    )
                assert not last.applied.done() and not original.retired.done()
            finally:
                release.set()
            async with asyncio.timeout(15):
                if mode in {"complete", "expire_before_settlement"}:
                    await action.applied
                    assert session.settings.temperature == 0.4, (
                        prepare_results,
                        due_refusals,
                        session.settings,
                    )
                    assert action.pending_operation is not None, (
                        prepare_results,
                        due_refusals,
                    )
                    assert len(ensure_consumptions) == 1
                    assert ensure_consumptions[0][0] is action.operation_task
                    assert (
                        ensure_consumptions[0][2] is False
                        and ensure_consumptions[0][1].consumed
                    )
                    assert action.operation_task not in controls._due_settings_reads
                    if mode == "expire_before_settlement":
                        assert (
                            ensure_consumptions[0][1].result
                            is not staged["original_result"]
                        )
                        assert (
                            ensure_consumptions[0][1].producer is not staged["worker"]
                        )
                        assert ensure_consumptions[0][1].source is action
                elif mode == "stale_before_ensure":
                    with pytest.raises(
                        parent.ParentSettingsDeferred, match="did not settle current"
                    ):
                        await action.applied
                    assert action.pending_operation is None and action_posts == []
                    assert session.settings.temperature == settings.temperature
                    assert (
                        session.settings.model == "local-model"
                    )  # B's real commit remains.
                    assert (
                        len(ensure_consumptions) == 1
                        and not ensure_consumptions[0][1].consumed
                    )
                    assert action.operation_task not in controls._due_settings_reads
                else:
                    with pytest.raises(asyncio.CancelledError):
                        await action.applied
                await staged["middle"].applied
                await last.applied
                assert await action.retired and await original.retired
                await staged["worker"]
                await controls.dispatch(
                    FenceSource(identity(3), "configuration")
                ).applied
            assert store.session_input_snapshot(session.id).draft == "last user input"
            assert staged["middle"].applied.result() < last.applied.result()
            assert calls == (
                ["local-model", "local-model"]
                if mode in {"complete", "expire_before_settlement"}
                else ["local-model"]
            )
            assert (
                pending.retirement.confirmed
                and pending.retirement.lease not in worker_leases(store.persistence.db)
            )
            import sqlite3

            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(pending.retirement.connection)
            assert all(
                tail.done() and tail not in controls._capture_predecessors
                for tail in (staged["first_tail"], staged["middle_tail"], last_tail)
            )
            assert staged["worker"] not in controls._settings_readiness_captures
            assert staged["worker"] not in controls._settings_readiness_waits
            assert not controls._effects and app._exception is None


async def test_due_core_read_refusal_keeps_the_original_episode(monkeypatch):
    """Policy regression only: a won probe is not an actual core push."""
    from Tests.UI.test_console_projection_reads_off_loop import (
        _CorePushScreen,
        _refusal_clock,
    )
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_parent_settings as parent
    from tldw_chatbook.UI.Console_Modules import config_sync

    now = _refusal_clock(monkeypatch, config, object)
    screen = _CorePushScreen()
    screen._console_core_refused_since = now[0]
    now[0] += config_sync.CORE_REFUSAL_EPISODE_SECONDS + 0.1
    calls = []

    async def refuse_due_read(source, *, retry_expired):
        calls.append(source)
        assert retry_expired is False
        assert screen._console_core_refused_since == 1000.0
        raise parent.ParentSettingsDeferred("Original checked read refused")

    from tldw_chatbook.Chat.console_control_settings import (
        ConsoleDefaultConvergenceRequired,
    )

    def due_original_push():
        raise ConsoleDefaultConvergenceRequired("Original ensure reached due defaults")

    monkeypatch.setattr(screen, "_sync_console_chat_core_state", due_original_push)
    monkeypatch.setattr(parent, "settle_parent_action_defaults", refuse_due_read)
    source = object()
    await config_sync.sync_selected_console_core_state(screen, source)
    assert calls == [source]
    assert screen.pushes == 0 and screen.replays == []
    assert screen._console_config_unreadable_installs == config._CONFIG_CACHE_INSTALLS


async def test_warm_selected_core_push_starts_no_due_read(monkeypatch):
    """Policy regression: the original warm coherent push keeps its fast path."""
    from Tests.UI.test_console_projection_reads_off_loop import (
        _CorePushScreen,
        _refusal_clock,
    )
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_parent_settings as parent
    from tldw_chatbook.UI.Console_Modules import config_sync

    _refusal_clock(monkeypatch, config, lambda: None)
    screen = _CorePushScreen()
    screen._console_core_refused_since = 999.0

    async def unexpected_read(_source, *, retry_expired):
        pytest.fail("Warm original push scheduled another config read")

    monkeypatch.setattr(parent, "settle_parent_action_defaults", unexpected_read)
    await config_sync.sync_selected_console_core_state(screen, object())
    assert screen.pushes == 1 and screen.replays == []
    assert screen._console_core_refused_since is None


@pytest.mark.parametrize(
    "phase,drift",
    [("activation", "system"), ("manual", "system"), ("snapshot", "session")],
)
async def test_prompt_resolution_completion_keeps_original_flow_target(
    selected_mounted_console, phase, drift
):
    from tldw_chatbook.Chat.console_parent_settings import apply_parent_system_prompt
    from tldw_chatbook.Widgets.Console.console_prompts_modal import ConsolePromptsModal

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    conversation = store.persistence.create_conversation(
        conversation_title="Resolution completion"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="resolution", service_epoch="local"
    )
    screen._prompts._open_console_prompts_modal()
    modal_capture = next(
        c for c in controls._parent_settings_captures if c.purpose == "prompt-modal"
    )
    async with asyncio.timeout(10):
        await modal_capture.applied
    await pilot.pause()
    modal = app.screen_stack[-1]
    assert isinstance(modal, ConsolePromptsModal)
    flow = modal._activate_improvement_context.__self__
    if phase == "snapshot":
        await flow.capture_manual_resolution()
    pin_before = flow._pinned_resolution
    gateway = flow._gateway
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldResolution:
        # Only the flow's provider result delivery is held; the actual domain
        # gateway, original resolver and selected settings authority stay intact.
        async def resolve_for_send(self, selection):
            result = await gateway.resolve_for_send(selection)
            entered.set()
            await release.wait()
            return result

    flow._gateway = HeldResolution()
    work = asyncio.create_task(
        flow.activate_improvement_context()
        if phase == "activation"
        else flow.capture_manual_resolution()
        if phase == "manual"
        else flow.build_improvement_snapshot(request_id="held-resolution", mode="auto")
    )
    try:
        async with asyncio.timeout(10):
            await entered.wait()
        if drift == "system":
            await apply_parent_system_prompt(
                screen, "Changed during original resolution."
            )
            assert (
                session.settings.system_prompt == "Changed during original resolution."
            )
        else:
            store.create_session(settings=session.settings)
            assert store.active_session_id != session.id
        release.set()
        async with asyncio.timeout(10):
            with pytest.raises(
                ValueError, match="System prompt changed|active Console session changed"
            ):
                await work
        assert flow._pinned_resolution is pin_before
        assert not any(
            c.purpose == "prompt-resolution" for c in controls._parent_settings_captures
        )
    finally:
        release.set()
        await asyncio.gather(work, return_exceptions=True)
        flow._gateway = gateway


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
async def test_parent_task_registration_precedes_actual_body(
    selected_mounted_console, monkeypatch, eager
):
    from tldw_chatbook.Chat.console_parent_settings import issue_parent_settings_capture
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat import console_parent_settings as parent_settings

    app, runtime, screen, pilot = selected_mounted_console
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="startup", service_epoch="local"
    )
    loop = asyncio.get_running_loop()
    factory = loop.get_task_factory()
    entered = []

    def observe_entry(frame):
        capture = frame.f_locals["capture"]
        if capture in targets:
            task = asyncio.current_task()
            assert capture.operation_task is task
            assert controls._parent_settings_captures[capture][-1] is task
            assert task in controls._control_tasks
            entered.append(capture)

    targets = []
    try:
        loop.set_task_factory(asyncio.eager_task_factory if eager else None)
        with _observe(
            parent_settings.run_parent_settings_capture.__code__,
            sys.monitoring.events.PY_START,
            observe_entry,
        ):
            capture = issue_parent_settings_capture(
                controls, "settings-render-replay", None
            )
            targets.append(capture)
            controls._queue_session_capture(capture, capture)
            async with asyncio.timeout(10):
                await capture.applied
                assert await capture.retired
            assert entered == [capture]
            assert capture.operation_task.done() and capture.started
            assert capture.effect.completed.result()

            for mode in ("cancel-inner", "refuse-inner"):
                capture = issue_parent_settings_capture(
                    controls, "settings-render-replay", None
                )
                targets.append(capture)
                original_create = loop.create_task
                tasks = []

                def schedule(
                    coroutine,
                    *args,
                    capture=capture,
                    mode=mode,
                    original_create=original_create,
                    tasks=tasks,
                    **kwargs,
                ):
                    frame = coroutine.cr_frame
                    exact = (
                        coroutine.cr_code.co_qualname.endswith(
                            "_queue_session_capture.<locals>.capture"
                        )
                        and frame is not None
                        and frame.f_locals.get("command") is capture
                    )
                    if exact and mode == "refuse-inner":
                        raise RuntimeError("injected exact parent startup refusal")
                    task = original_create(coroutine, *args, **kwargs)
                    if exact:
                        tasks.append(task)
                        assert not capture.started
                        task.cancel()
                    return task

                with monkeypatch.context() as patch:
                    patch.setattr(loop, "create_task", schedule)
                    controls._queue_session_capture(capture, capture)
                    async with asyncio.timeout(10):
                        with pytest.raises(
                            asyncio.CancelledError
                            if mode == "cancel-inner"
                            else RuntimeError
                        ):
                            await capture.applied
                        assert await capture.retired
                assert capture not in entered and not capture.started
                assert (
                    capture.effect.completed.result()
                    and capture.pending_operation is None
                )
                assert capture not in controls._parent_settings_captures
                assert all(task.cancelled() for task in tasks)
    finally:
        loop.set_task_factory(factory)


@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "eager"])
async def test_cancelled_parent_startup_keeps_actual_writer_predecessor(
    selected_mounted_console, monkeypatch, eager
):
    from Tests.Chat.test_console_settings_apply_store import _submission
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
        FenceSource,
        UpdateInputs,
    )
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.Chat.console_parent_settings import issue_parent_settings_capture

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    conversation = store.persistence.create_conversation(
        conversation_title="Cancelled parent predecessor"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="relay", service_epoch="local"
    )

    def identity(sequence):
        return ControlIdentity(
            "relay",
            "local",
            sequence,
            f"relay-{sequence}",
            session.id,
            session.incarnation_id,
        )

    submission = _submission(
        store, session.id, submission_id="startup-relay", model="gate-model"
    )
    loop = asyncio.get_running_loop()
    factory = loop.get_task_factory()
    try:
        loop.set_task_factory(asyncio.eager_task_factory if eager else None)
        with _hold_full_writer(store.persistence) as (entered, release, calls):
            first = controls.dispatch(ApplySettings(identity(1), submission))
            body_error = None
            try:
                await _entered(entered)
                operation = settings_pending(store, session.id)
                assert operation is not None and operation.retirement is not None
                assert not operation.retirement.confirmed and not first.applied.done()
                parent = issue_parent_settings_capture(
                    controls, "settings-render-replay", None
                )
                original_create = loop.create_task
                inner = []

                def cancel_exact(coroutine, *args, **kwargs):
                    exact = (
                        coroutine.cr_code.co_qualname.endswith(
                            "_queue_session_capture.<locals>.capture"
                        )
                        and coroutine.cr_frame.f_locals.get("command") is parent
                    )
                    task = original_create(coroutine, *args, **kwargs)
                    if exact:
                        inner.append(task)
                        task.cancel()
                    return task

                with monkeypatch.context() as patch:
                    patch.setattr(loop, "create_task", cancel_exact)
                    controls._queue_session_capture(parent, parent)
                    tail = controls._session_capture_tails[session.id]
                    later = controls.dispatch(
                        UpdateInputs(identity(2), "after held predecessor", (1, 1))
                    )
                    async with asyncio.timeout(10):
                        with pytest.raises(asyncio.CancelledError):
                            await parent.applied
                        assert await parent.retired
                assert inner and inner[0].cancelled() and not parent.started
                assert (
                    parent.effect.completed.result()
                    and parent not in controls._parent_settings_captures
                )
                assert not tail.done() and not later.applied.done()
                assert not first.applied.done() and not operation.retirement.confirmed
                assert calls == ["gate-model"]
            except BaseException as error:
                body_error = error
                raise
            finally:
                release.set()
                async with asyncio.timeout(10):
                    try:
                        await first.applied
                    except BaseException as error:
                        if body_error is not None:
                            body_error.add_note(
                                f"Original writer cleanup also failed: {error!r}"
                            )
                            raise body_error from error
                        raise
            async with asyncio.timeout(10):
                await later.applied
                assert await first.retired
                await controls.dispatch(
                    FenceSource(identity(3), "configuration")
                ).applied
            assert tail.done() and operation.retirement.confirmed
            assert (
                store.session_input_snapshot(session.id).draft
                == "after held predecessor"
            )
            assert operation.admission not in operation.durability._leases
            assert not operation.lifecycle.pending_live and not controls._effects
    finally:
        loop.set_task_factory(factory)


@pytest.mark.parametrize("selected_mounted_console", ["realtime"], indirect=True)
@pytest.mark.parametrize(
    "stale", [False, True], ids=["current-ready", "retired-attempt"]
)
async def test_realtime_ready_keeps_seed_and_audio_behind_actual_settings(
    selected_mounted_console,
    stale,
):
    from Tests.UI.test_console_realtime_wiring import _install_realtime_fakes, _wait_for
    from Tests.Chat.test_console_settings_apply_store import _submission
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Backup_Recovery.test_console_main_display_batch import (
        _worker_connections,
    )
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
    )
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.Chat.console_parent_settings import ParentSettingsOriginChanged
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )
    from tldw_chatbook.UI.Console_Modules.realtime import (
        ConsoleRealtimeController,
        ConsoleRealtimeSession,
    )
    from tldw_chatbook.Chat import console_parent_settings as parent
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    store.replace_session_settings(
        session.id,
        replace(
            blank_console_session_settings(app.app_config),
            system_prompt="Keep this actual saved System.",
        ),
    )
    conversation = store.persistence.create_conversation(
        conversation_title="Realtime settings ordering"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    rig = _install_realtime_fakes(app)
    entries = []

    def observe(frame):
        capture = frame.f_locals["capture"]
        if capture.controls is controls and capture.purpose in {
            "realtime-ready",
            "realtime-instructions",
        }:
            entries.append((capture, asyncio.current_task()))

    with _observe(
        parent.run_parent_settings_capture.__code__,
        sys.monitoring.events.PY_START,
        observe,
    ):
        screen._hands_free.action_toggle_console_hands_free()
        await _wait_for(lambda: bool(rig.sessions) and rig.session.connected, pilot)
        owner, voice, provider = screen._realtime, screen._realtime.session, rig.session
        assert (
            type(owner) is ConsoleRealtimeController
            and type(voice) is ConsoleRealtimeSession
        )
        assert provider.config.instructions == "Keep this actual saved System."
        instructions = [
            item for item, task in entries if item.purpose == "realtime-instructions"
        ]
        assert len(instructions) == 1 and instructions[0].operation_task.done()
        assert (
            not provider.seeds
            and not voice.ready
            and voice.controller.state == "connecting"
        )
        identity = ControlIdentity(
            "mounted", "local", 1, "ready-write", session.id, session.incarnation_id
        )
        submission = _submission(
            store, session.id, submission_id="ready-write", model="ready-local-model"
        )
        with (
            _worker_connections(store.persistence.db) as connections,
            _hold_full_writer(store.persistence) as (entered, release, calls),
        ):
            receipt = controls.dispatch(ApplySettings(identity, submission))
            ready = None
            first_error = None
            try:
                await _entered(entered)
                pending = settings_pending(store, session.id)
                assert pending is not None and pending.source.identity is identity
                assert worker_leases(store.persistence.db) and connections
                rig.recorder.push(b"aa")
                provider.fire_ready()
                async with asyncio.timeout(10):
                    while not any(
                        item.purpose == "realtime-ready"
                        for item in controls._parent_settings_captures
                    ):
                        await asyncio.sleep(0.01)
                ready = next(
                    item
                    for item in controls._parent_settings_captures
                    if item.purpose == "realtime-ready"
                )
                assert ready.checked_config == (voice, voice.connect_attempt)
                assert not ready.applied.done() and not voice.ready
                assert not provider.seeds and not provider.audio_frames
                if stale:
                    await owner.teardown()
                    assert owner.session is None and provider.closed
            except BaseException as error:
                first_error = error
                raise
            finally:
                release.set()
                try:
                    await receipt.applied
                    await receipt.retired
                except BaseException:
                    if first_error is None:
                        raise
                finally:
                    if ready is not None:
                        await asyncio.gather(
                            ready.applied, ready.retired, return_exceptions=True
                        )
            assert (
                pending.retirement.confirmed
                and pending.retirement.connection is not None
            )
            assert pending.commit is not None and calls == ["ready-local-model"]
            if stale:
                with pytest.raises(
                    ParentSettingsOriginChanged, match="attempt changed"
                ):
                    await ready.applied
                assert (
                    not provider.seeds and not provider.audio_frames and not voice.ready
                )
            else:
                await ready.applied
                assert provider.seeds == [([], "Keep this actual saved System.")]
                assert provider.audio_frames == [b"aa"]
                assert voice.ready and voice.controller.state == "live"
                await owner.teardown()
            assert await ready.retired
            count = len(entries)
            provider.fire_ready()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert owner.session is None and len(entries) == count
            assert ready not in controls._parent_settings_captures


@pytest.mark.parametrize("selected_mounted_console", ["realtime"], indirect=True)
async def test_realtime_ready_refuses_changed_original_settings_edges(
    selected_mounted_console,
):
    from types import CellType, FunctionType
    from Tests.UI.test_console_realtime_wiring import _install_realtime_fakes, _wait_for
    from tldw_chatbook.Chat.console_parent_settings import ParentSettingsSourceChanged
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    store.replace_session_settings(
        session.id,
        replace(
            blank_console_session_settings(app.app_config),
            system_prompt="Original source only.",
        ),
    )
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    rig = _install_realtime_fakes(app)
    screen._hands_free.action_toggle_console_hands_free()
    await _wait_for(lambda: bool(rig.sessions) and rig.session.connected, pilot)
    owner, voice, provider = screen._realtime, screen._realtime.session, rig.session
    calls = []
    try:
        for name, value in (
            ("_ensure_session_settings", session.settings),
            ("_runtime_accessor", runtime),
            ("_chat_store_accessor", store),
        ):
            original = getattr(owner, name)

            def substitute(value=value, name=name):
                calls.append(name)
                return value

            setattr(owner, name, substitute)
            try:
                with pytest.raises(
                    ParentSettingsSourceChanged, match="callback changed"
                ):
                    owner._on_console_realtime_ready(voice)
                assert not calls and not provider.seeds and not voice.ready
                assert not any(
                    item.purpose == "realtime-ready"
                    for item in controls._parent_settings_captures
                )
            finally:
                setattr(owner, name, original)
        original = owner._ensure_session_settings
        owner._ensure_session_settings = FunctionType(
            original.__code__,
            original.__globals__,
            closure=(CellType(object()),),
        )
        try:
            with pytest.raises(ParentSettingsSourceChanged, match="callback changed"):
                owner._on_console_realtime_ready(voice)
            assert not provider.seeds and not voice.ready
        finally:
            owner._ensure_session_settings = original
        provider.fire_ready()
        await _wait_for(lambda: voice.ready, pilot)
        assert provider.seeds == [([], "Original source only.")]
    finally:
        await owner.teardown()


@pytest.mark.parametrize(
    "route,stale",
    [
        ("file-image", False),
        ("file-text", False),
        ("file-text", True),
        ("clipboard", False),
        ("clipboard", True),
        ("paths", False),
        ("paths", True),
    ],
)
async def test_attachment_processing_keeps_original_completion_target(
    selected_mounted_console,
    tmp_path,
    monkeypatch,
    route,
    stale,
):
    from contextlib import contextmanager
    from io import BytesIO
    from PIL import Image
    from Tests.Chat.test_console_settings_apply_store import _submission
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Backup_Recovery.test_console_main_display_batch import (
        _worker_connections,
    )
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
    from tldw_chatbook.Chat import attachment_core
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
    )
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.Chat.console_paste_attach import ClipboardGrab
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )
    from tldw_chatbook.UI.Screens import chat_screen

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    store.replace_session_settings(
        session.id, blank_console_session_settings(app.app_config)
    )
    conversation = store.persistence.create_conversation(
        conversation_title="Attachment completion owner"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    composer = screen._console_composer_or_none()
    initial_draft = composer.draft_text()
    buffer = BytesIO()
    Image.new("RGB", (4, 4), (0, 100, 0)).save(buffer, format="PNG")
    image = tmp_path / "target.png"
    if route == "paths":
        from pathlib import Path
        from uuid import uuid4

        image = Path.home() / f"task61-attachment-{uuid4().hex}.png"
    image.write_bytes(buffer.getvalue())
    if route == "paths":
        assert chat_screen.looks_attachable(str(image))
    text = tmp_path / "target.md"
    text.write_text("task61 original file body", encoding="utf-8")
    load = attachment_core.load_processed_file

    async def rooted(file_path, *, allowed_root=None):
        return await load(
            file_path, allowed_root=(None if route == "paths" else str(tmp_path))
        )

    monkeypatch.setattr(attachment_core, "load_processed_file", rooted)
    if route in {"clipboard", "paths"}:
        grab = (
            ClipboardGrab(kind="image", png_bytes=buffer.getvalue())
            if route == "clipboard"
            else ClipboardGrab(kind="paths", paths=tuple(str(image) for _ in range(6)))
        )
        monkeypatch.setattr(chat_screen, "grab_clipboard_image", lambda: grab)
    function = (
        attachment_core.process_attachment_bytes
        if route == "clipboard"
        else attachment_core.process_attachment_path
    )
    processed, release_file = threading.Event(), threading.Event()
    results = []

    @contextmanager
    def hold_original_result():
        def observe(frame, values):
            result = values[-1]
            if type(result) is attachment_core.PendingAttachment:
                results.append(result)
                if len(results) == 1:
                    processed.set()
                    assert release_file.wait(
                        10
                    ), "Original attachment result was not released"

        with _observe(
            function.__code__, sys.monitoring.events.PY_RETURN, observe, values=True
        ):
            try:
                yield
            finally:
                release_file.set()

    notifications = []
    notify = app.notify

    def record_notice(frame):
        if frame.f_locals.get("self") is app:
            notifications.append(str(frame.f_locals["message"]))

    identity = ControlIdentity(
        "mounted", "local", 1, "attachment-write", session.id, session.incarnation_id
    )
    submission = _submission(
        store,
        session.id,
        submission_id="attachment-write",
        model="attachment-local-model",
    )
    work = None
    with (
        hold_original_result(),
        _observe(
            notify.__func__.__code__, sys.monitoring.events.PY_START, record_notice
        ),
    ):
        try:
            coroutine = (
                screen._paste_console_clipboard_image()
                if route in {"clipboard", "paths"}
                else screen._process_console_attachment(
                    str(text if route == "file-text" else image)
                )
            )
            work = asyncio.create_task(coroutine)
            try:
                async with asyncio.timeout(10):
                    while not processed.is_set():
                        if work.done():
                            work.result()
                            raise AssertionError(
                                f"Attachment returned before processing: notices={notifications!r}"
                            )
                        await asyncio.sleep(0.01)
            except TimeoutError as error:
                frames = [
                    (frame.f_code.co_name, frame.f_lineno) for frame in work.get_stack()
                ]
                captures = [
                    (item.purpose, item.started, item.work.done())
                    for item in controls._parent_settings_captures
                ]
                raise AssertionError(
                    f"Attachment processing stage timed out: frames={frames!r}, captures={captures!r}, notices={notifications!r}"
                ) from error
            with (
                _worker_connections(store.persistence.db) as connections,
                _hold_full_writer(store.persistence) as (entered, release, calls),
            ):
                receipt = controls.dispatch(ApplySettings(identity, submission))
                first_error = None
                try:
                    await _entered(entered)
                    pending = settings_pending(store, session.id)
                    assert (
                        pending is not None
                        and worker_leases(store.persistence.db)
                        and connections
                    )
                    release_file.set()
                    async with asyncio.timeout(10):
                        while not any(
                            item.purpose == "attachment-completion"
                            and item.checked_config[1] in {"file", "clipboard-image"}
                            for item in controls._parent_settings_captures
                        ):
                            if work.done():
                                work.result()
                            await asyncio.sleep(0.01)
                    completion = next(
                        item
                        for item in controls._parent_settings_captures
                        if item.purpose == "attachment-completion"
                        and item.checked_config[1] in {"file", "clipboard-image"}
                    )
                    assert (
                        completion.session is session and not completion.applied.done()
                    )
                    assert not store.pending_attachments(session.id)
                    assert composer.draft_text() == initial_draft and not work.done()
                    if stale:
                        replacement = store.create_session(settings=session.settings)
                        assert store.active_session_id == replacement.id
                except BaseException as error:
                    first_error = error
                    raise
                finally:
                    release.set()
                    try:
                        await receipt.applied
                        await receipt.retired
                    except BaseException:
                        if first_error is None:
                            raise
                assert pending.retirement.confirmed and pending.commit is not None
                assert calls == ["attachment-local-model"]
            await work
            if stale:
                assert not store.pending_attachments(session.id)
                assert not store.pending_attachments(replacement.id)
                assert (
                    "task61 original file body"
                    not in screen._console_composer_or_none().draft_text()
                )
                assert len(results) == 1
                assert not any(
                    "attached" in message.lower()
                    or "inserted as text" in message.lower()
                    for message in notifications
                )
            elif route == "file-text":
                assert composer.draft_text().count("task61 original file body") == 1
                assert not store.pending_attachments(session.id)
            else:
                expected = 5 if route == "paths" else 1
                assert len(store.pending_attachments(session.id)) == expected
                assert len(results) == expected
                if route == "paths":
                    assert "Attached first 5 of 6 dropped files." in notifications
            assert not any(
                item.purpose == "attachment-completion"
                for item in controls._parent_settings_captures
            )
        finally:
            release_file.set()
            if work is not None:
                if not work.done():
                    work.cancel()
                await asyncio.gather(work, return_exceptions=True)
            if route == "paths":
                image.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "mode",
    [
        "native-exception",
        "no-entry-exception",
        "writer-false",
        "display-error",
        "cancel-observer",
        "switch-exception",
        "draft-conflict",
        "composer-conflict",
        "cancel-write-before-entry",
        "cancel-parent-entered",
    ],
)
async def test_guarded_library_prompt_retains_original_transaction(
    selected_mounted_console, monkeypatch, mode
):
    import json
    from contextlib import ExitStack
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Backup_Recovery.test_console_main_display_batch import (
        _worker_connections,
    )
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
    from tldw_chatbook.Chat import console_guarded_prompt as guarded
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )
    from tldw_chatbook.UI.Console_Modules import wiring
    from tldw_chatbook.UI.Console_Modules.prompts import (
        PromptVariableApplication,
        fingerprint_system_text,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    store.replace_session_settings(
        session.id,
        replace(
            blank_console_session_settings(app.app_config),
            system_prompt="before System",
        ),
    )
    conversation = store.persistence.create_conversation(
        conversation_title="Guarded prompt owner", system_prompt="before System"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    original_record = store.persistence.db.get_conversation_by_id(conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    composer = screen._console_composer_or_none()
    composer.load_draft("before draft")
    store.set_session_draft(session.id, composer.draft_text())
    checkpoint = composer.capture_transaction_checkpoint()
    application = PromptVariableApplication(
        system_text="after System",
        user_text="after draft",
        apply_system=True,
        apply_user=True,
        destination="append_active",
        target_session_id=session.id,
        composer_fingerprint=None,
        system_fingerprint=fingerprint_system_text("before System"),
    )
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, application)
    entered, release = threading.Event(), threading.Event()
    captures, full_writes, notices, faults = [], [], [], []

    def capture_entry(frame):
        capture = frame.f_locals["capture"]
        if capture.controls is controls:
            captures.append(capture)

    def prompt_return(frame):
        if (
            frame.f_locals.get("self") is store
            and frame.f_locals.get("session") is session
        ):
            entered.set()
            assert release.wait(10), "Original prompt writer was not released"
            if mode in {
                "native-exception",
                "switch-exception",
                "draft-conflict",
                "composer-conflict",
            }:
                faults.append("after-native")
                raise RuntimeError("task61 original mutation return fault")

    def prepare_return(frame):
        if (
            mode == "no-entry-exception"
            and not faults
            and frame.f_locals.get("self") is store
        ):
            faults.append("before-native")
            raise RuntimeError("task61 original prepare return fault")

    def full_return(frame):
        if frame.f_locals.get("self") is store.persistence:
            full_writes.append(frame.f_locals["settings"])

    def surface_entry(frame):
        if (
            mode == "display-error"
            and not faults
            and frame.f_locals.get("screen") is screen
        ):
            faults.append("display")
            raise RuntimeError("task61 original surface fault")

    def notice_entry(frame):
        if frame.f_locals.get("self") is app:
            notices.append(str(frame.f_locals["message"]))

    if mode == "writer-false":
        with store.persistence.db.transaction() as cursor:
            cursor.execute(
                "CREATE TRIGGER task61_prompt_fault BEFORE UPDATE OF system_prompt ON conversations WHEN NEW.system_prompt = 'after System' BEGIN SELECT RAISE(FAIL, 'task61 original SQLite write fault'); END"
            )
    work = None
    first_error = None
    canceled_writes = []
    if mode == "cancel-write-before-entry":
        from tldw_chatbook.Chat.console_control_settings import PendingSettingsApply

        create_task = controls._loop.create_task

        def schedule(coroutine, *args, **kwargs):
            frame = getattr(coroutine, "cr_frame", None)
            operation = frame.f_locals.get("self") if frame is not None else None
            task = create_task(coroutine, *args, **kwargs)
            if (
                getattr(coroutine, "cr_code", None)
                is PendingSettingsApply.apply.__code__
                and type(operation) is PendingSettingsApply
                and operation.controls is controls
                and operation.mutation is not None
                and operation.mutation[0] == "system_prompt"
            ):
                assert not operation.apply_started and not operation.native_entered
                canceled_writes.append((task, operation))
                assert task.cancel()
            return task

        monkeypatch.setattr(controls._loop, "create_task", schedule)
    with ExitStack() as stack:
        connections = stack.enter_context(_worker_connections(store.persistence.db))
        stack.enter_context(
            _observe(
                guarded.run_guarded_prompt.__code__,
                sys.monitoring.events.PY_START,
                capture_entry,
            )
        )
        stack.enter_context(
            _observe(
                store._write_session_system_prompt.__func__.__code__,
                sys.monitoring.events.PY_RETURN,
                prompt_return,
            )
        )
        stack.enter_context(
            _observe(
                store._prepare_session_system_prompt.__func__.__code__,
                sys.monitoring.events.PY_RETURN,
                prepare_return,
            )
        )
        stack.enter_context(
            _observe(
                store.persistence.update_conversation_console_session_settings.__func__.__code__,
                sys.monitoring.events.PY_RETURN,
                full_return,
            )
        )
        stack.enter_context(
            _observe(
                wiring._selected_prompt_surfaces_code,
                sys.monitoring.events.PY_START,
                surface_entry,
            )
        )
        stack.enter_context(
            _observe(
                app.notify.__func__.__code__,
                sys.monitoring.events.PY_START,
                notice_entry,
            )
        )
        try:
            work = asyncio.create_task(
                screen._prompts._consume_pending_console_prompt_insert()
            )
            if mode not in {"no-entry-exception", "cancel-write-before-entry"}:
                async with asyncio.timeout(10):
                    while not entered.is_set():
                        if work.done():
                            work.result()
                            raise AssertionError(
                                f"Guarded prompt returned before native stage: notices={notices!r}"
                            )
                        await asyncio.sleep(0.01)
                capture = captures[0]
                first = capture.guarded_transaction.first_operation
                assert first.native_entered and first.actual_task is not None
                assert (
                    first.effect is capture.effect
                    and capture.effect in controls._effects
                )
                assert worker_leases(store.persistence.db) and connections
                assert not capture.applied.done() and not work.done()
                handoffs, claim = capture.checked_config[3]
                assert handoffs.is_current_claim(claim)
                if mode == "switch-exception":
                    replacement = store.create_session(settings=session.settings)
                    assert store.active_session_id == replacement.id
                elif mode == "composer-conflict":
                    composer.insert_text(" newer")
                    changed_draft = (
                        session.draft,
                        session.draft_revision,
                        session._draft_authored_token,
                    )
                    edited_composer = composer.capture_transaction_checkpoint()
                    assert (
                        screen._session._console_visible_draft_revision
                        == session.draft_revision
                    )
                elif mode == "draft-conflict":
                    token = session._draft_authored_token
                    store.set_session_draft(
                        session.id,
                        session.draft,
                        authored_token=(token[0], token[1] + 1),
                    )
                    changed_draft = (
                        session.draft,
                        session.draft_revision,
                        session._draft_authored_token,
                    )
                if mode in {"cancel-observer", "cancel-parent-entered"}:
                    target_task = (
                        work if mode == "cancel-observer" else capture.operation_task
                    )
                    target_task.cancel()
                    await asyncio.sleep(0)
                    target_task.cancel()
                    await asyncio.sleep(0)
                    assert not work.done() and handoffs.is_current_claim(claim)
                release.set()
            if mode in {
                "cancel-observer",
                "cancel-parent-entered",
                "cancel-write-before-entry",
            }:
                with pytest.raises(asyncio.CancelledError):
                    await work
            else:
                await work
            capture = captures[0]
            first = capture.guarded_transaction.first_operation
            assert await capture.retired
            assert capture not in controls._parent_settings_captures
            assert not app.pending_handoffs.has_pending(
                HandoffChannel.CONSOLE_PROMPT_INSERT
            )
            assert not app.pending_handoffs.is_current_claim(
                capture.checked_config[3][1]
            )
            assert not store._selected_settings_pending.get(session.id)
            if mode in {"draft-conflict", "composer-conflict"}:
                assert capture.applied.result() is False and not full_writes
                assert (
                    session.draft,
                    session.draft_revision,
                    session._draft_authored_token,
                ) == changed_draft
                assert session.settings.system_prompt == "after System"
                assert first.failed_settled and first.retirement.confirmed
                if mode == "composer-conflict":
                    current_composer = composer.capture_transaction_checkpoint()
                    assert current_composer.draft == edited_composer.draft
                    assert current_composer.undo_stack == edited_composer.undo_stack
            elif mode in {
                "native-exception",
                "no-entry-exception",
                "switch-exception",
                "cancel-write-before-entry",
            }:
                if mode == "cancel-write-before-entry":
                    assert len(canceled_writes) == 1 and canceled_writes[0][1] is first
                    assert (
                        first.actual_task is canceled_writes[0][0]
                        and first.actual_task.cancelled()
                    )
                    assert not first.apply_started and not first.native_entered
                    assert (
                        capture.guarded_transaction.result is False
                        and capture.guarded_transaction.cancelled
                    )
                    assert screen._prompts._PROMPT_APPLICATION_FAILED_COPY in notices
                    with pytest.raises(asyncio.CancelledError):
                        capture.applied.result()
                else:
                    assert capture.applied.result() is False
                assert (
                    session.settings.system_prompt == "before System"
                    and session.draft == "before draft"
                )
                if mode != "switch-exception":
                    restored = composer.capture_transaction_checkpoint()
                    assert restored.draft.segments == checkpoint.draft.segments
                    assert (
                        restored.undo_stack == checkpoint.undo_stack
                        and restored.redo_stack == checkpoint.redo_stack
                    )
                else:
                    assert store.active_session_id == replacement.id
                    assert (
                        replacement.settings.system_prompt == "after System"
                        and not replacement.draft
                    )
                assert full_writes == (
                    []
                    if mode == "cancel-write-before-entry"
                    else [capture.guarded_transaction.transaction.settings]
                )
                assert first.failed_settled and first.error is not None
                assert first.native_entered is (
                    mode not in {"no-entry-exception", "cancel-write-before-entry"}
                )
                assert (first.retirement is None) is (
                    mode in {"no-entry-exception", "cancel-write-before-entry"}
                )
                if first.retirement is not None:
                    assert first.retirement.confirmed
                compensation = capture.pending_operation
                assert compensation is not first and compensation.commit is session
                record = store.persistence.db.get_conversation_by_id(conversation)
                if mode == "cancel-write-before-entry":
                    # Original replacement writes only when settings changed;
                    # canceled-before-apply changed only the guarded draft.
                    assert (
                        compensation.apply_started and compensation.actual_task.done()
                    )
                    assert (
                        not compensation.native_entered
                        and compensation.retirement is None
                    )
                    assert record == original_record
                else:
                    assert compensation.retirement.confirmed
                    metadata = json.loads(record["metadata"])
                    assert (
                        metadata["console_session_settings"]["system_prompt"]
                        == "before System"
                    )
                # Original full replacement restores metadata, not the separate
                # top-level System column owned by the first component write.
                assert record["system_prompt"] == (
                    "before System"
                    if mode in {"no-entry-exception", "cancel-write-before-entry"}
                    else "after System"
                )
            else:
                if mode == "cancel-parent-entered":
                    assert (
                        capture.guarded_transaction.result is True
                        and capture.guarded_transaction.cancelled
                    )
                    with pytest.raises(asyncio.CancelledError):
                        capture.applied.result()
                else:
                    assert capture.applied.result() is True
                assert session.settings.system_prompt == "after System"
                assert session.draft == "before draft\nafter draft" and not full_writes
                assert first.retirement.confirmed and first.commit is not None
                if mode == "writer-false":
                    assert first.commit[1] is False
                    assert (
                        screen._prompts._PROMPT_SYSTEM_PERSISTENCE_FAILED_COPY
                        in notices
                    )
                if mode == "display-error":
                    assert screen._prompts._PROMPT_DISPLAY_SYNC_FAILED_COPY in notices
        except BaseException as error:
            first_error = error
            raise
        finally:
            release.set()
            if work is not None:
                outcomes = await asyncio.gather(work, return_exceptions=True)
                if (
                    first_error is None
                    and mode
                    not in {
                        "cancel-observer",
                        "cancel-parent-entered",
                        "cancel-write-before-entry",
                    }
                    and isinstance(outcomes[0], BaseException)
                ):
                    raise outcomes[0]
            if mode == "writer-false":
                with store.persistence.db.transaction() as cursor:
                    cursor.execute("DROP TRIGGER task61_prompt_fault")


@pytest.mark.parametrize("mode", ["cancel-before-entry", "scheduling-refusal"])
async def test_guarded_library_no_entry_releases_exact_claim(
    selected_mounted_console, monkeypatch, mode
):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat import console_guarded_prompt as guarded
    from tldw_chatbook.Chat.console_parent_settings import ParentSettingsCapture
    from tldw_chatbook.UI.Console_Modules.prompts import PromptVariableApplication
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    before = (
        session.draft,
        session.draft_revision,
        session._draft_authored_token,
        session.settings,
    )
    application = PromptVariableApplication(
        system_text=None,
        user_text="not entered",
        apply_system=False,
        apply_user=True,
        destination="append_active",
        target_session_id=session.id,
        composer_fingerprint=None,
        system_fingerprint=None,
    )
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, application)
    captures = []
    create_task = controls._loop.create_task

    def schedule(coroutine, *args, **kwargs):
        frame = getattr(coroutine, "cr_frame", None)
        receipt = frame.f_locals.get("receipt") if frame is not None else None
        if (
            mode == "scheduling-refusal"
            and type(receipt) is ParentSettingsCapture
            and receipt.controls is controls
            and receipt.purpose == "guarded-prompt"
        ):
            captures.append(receipt)
            raise RuntimeError("task61 exact guarded scheduling refusal")
        return create_task(coroutine, *args, **kwargs)

    def issued(frame, values):
        capture = values[-1]
        if (
            mode == "cancel-before-entry"
            and type(capture) is ParentSettingsCapture
            and capture.controls is controls
        ):
            captures.append(capture)
            assert capture.work.cancel()

    with monkeypatch.context() as patch:
        patch.setattr(controls._loop, "create_task", schedule)
        with _observe(
            guarded.queue_guarded_prompt.__code__,
            sys.monitoring.events.PY_RETURN,
            issued,
            values=True,
        ):
            with pytest.raises(
                asyncio.CancelledError
                if mode == "cancel-before-entry"
                else RuntimeError
            ):
                await screen._prompts._consume_pending_console_prompt_insert()
    assert len(captures) == 1
    capture = captures[0]
    if capture.work is not None:
        await asyncio.gather(capture.work, return_exceptions=True)
    assert capture.guarded_transaction is None and capture.pending_operation is None
    assert (
        capture.guarded_claim_settled
        and capture not in controls._parent_settings_captures
    )
    assert capture.effect.completed.done() and capture.effect.physically_retired
    assert (
        session.draft,
        session.draft_revision,
        session._draft_authored_token,
        session.settings,
    ) == before
    assert app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_PROMPT_INSERT)
    retry = app.pending_handoffs.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert retry is not None and retry.value == application
    assert retry.revision == capture.checked_config[3][1].revision
    assert retry is not capture.checked_config[3][1]
    assert app.pending_handoffs.acknowledge(retry)


async def test_guarded_draft_callback_cannot_become_compensation_state(
    selected_mounted_console,
):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat import console_guarded_prompt as guarded
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )
    from tldw_chatbook.UI.Console_Modules.prompts import (
        PromptVariableApplication,
        fingerprint_system_text,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    store.replace_session_settings(
        session.id,
        replace(blank_console_session_settings(app.app_config), system_prompt="before"),
    )
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    composer = screen._console_composer_or_none()
    composer.load_draft("before draft")
    store.set_session_draft(session.id, composer.draft_text())
    application = PromptVariableApplication(
        system_text="must not apply",
        user_text="owned append",
        apply_system=True,
        apply_user=True,
        destination="append_active",
        target_session_id=session.id,
        composer_fingerprint=None,
        system_fingerprint=fingerprint_system_text("before"),
    )
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, application)
    captures, changed, checked_records = [], [], []

    def issued_record(frame, values):
        record = values[-1]
        if (
            record is not None
            and record[0].controls is controls
            and "forged" not in checked_records
        ):
            from tldw_chatbook.Chat.console_parent_settings import (
                ParentSettingsSourceChanged,
            )

            checked_records.append("forged")
            with pytest.raises(
                ParentSettingsSourceChanged, match="not issued or was consumed"
            ):
                guarded.finish_guarded_draft_write(tuple(list(record)), record[1])
            assert record[0].guarded_transaction.draft_write is record

    def consumed_record(frame):
        record = frame.f_locals.get("record")
        if (
            record is not None
            and record[0].controls is controls
            and "replay" not in checked_records
        ):
            from tldw_chatbook.Chat.console_parent_settings import (
                ParentSettingsSourceChanged,
            )

            checked_records.append("replay")
            with pytest.raises(
                ParentSettingsSourceChanged, match="not issued or was consumed"
            ):
                guarded.finish_guarded_draft_write(record, record[1])

    def observe_capture(frame):
        capture = frame.f_locals["capture"]
        if capture.controls is controls:
            captures.append(capture)

    def callback_return(frame):
        if (
            changed
            or frame.f_locals.get("self") is not store
            or frame.f_locals.get("session") is not session
            or not captures
            or captures[0].guarded_transaction is None
        ):
            return
        # Observe the original setter's handoff boundary, without replacing it.
        assert not store._preparation_lock._is_owned()
        changed.append(None)
        token = session._draft_authored_token
        store.set_session_draft(
            session.id, "newer callback draft", authored_token=(token[0], token[1] + 1)
        )
        changed[0] = (
            session.draft,
            session.draft_revision,
            session._draft_authored_token,
        )

    with (
        _observe(
            guarded.run_guarded_prompt.__code__,
            sys.monitoring.events.PY_START,
            observe_capture,
        ),
        _observe(
            store._publish_session_draft_handoff.__func__.__code__,
            sys.monitoring.events.PY_RETURN,
            callback_return,
        ),
        _observe(
            guarded.begin_guarded_draft_write.__code__,
            sys.monitoring.events.PY_RETURN,
            issued_record,
            values=True,
        ),
        _observe(
            guarded.finish_guarded_draft_write.__code__,
            sys.monitoring.events.PY_RETURN,
            consumed_record,
        ),
    ):
        await screen._prompts._consume_pending_console_prompt_insert()
    capture = captures[0]
    assert (
        changed
        and (session.draft, session.draft_revision, session._draft_authored_token)
        == changed[0]
    )
    assert checked_records == ["forged", "replay"]
    assert session.settings.system_prompt == "before"
    assert capture.guarded_transaction.first_operation is None
    assert capture.guarded_transaction.transaction.witness_error is not None
    assert capture.applied.result() is False and await capture.retired
    assert capture.guarded_claim_settled and not app.pending_handoffs.has_pending(
        HandoffChannel.CONSOLE_PROMPT_INSERT
    )


async def test_original_prompt_dialog_owns_selected_target_and_transaction(
    selected_mounted_console,
):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat import console_guarded_prompt as guarded
    from tldw_chatbook.Chat.console_parent_settings import ParentSettingsCapture
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )
    from tldw_chatbook.Widgets.Console.prompt_variables_dialog import (
        PromptVariablesDialog,
        APPLY_BUTTON_ID,
        SYSTEM_CHECKBOX_ID,
    )

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    store.replace_session_settings(
        session.id,
        replace(
            blank_console_session_settings(app.app_config), system_prompt="before modal"
        ),
    )
    conversation = store.persistence.create_conversation(
        conversation_title="Original dialog owner", system_prompt="before modal"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    composer = screen._console_composer_or_none()
    composer.load_draft("replace this draft")
    store.set_session_draft(session.id, composer.draft_text())
    target = await guarded.capture_prompt_target(screen._prompts)
    assert (
        target.session_id == session.id
        and target.composer_snapshot == composer.capture_draft_snapshot()
    )
    captures = []

    def issued(frame, values):
        capture = values[-1]
        if type(capture) is ParentSettingsCapture and capture.controls is controls:
            captures.append(capture)

    with _observe(
        guarded.queue_guarded_prompt.__code__,
        sys.monitoring.events.PY_RETURN,
        issued,
        values=True,
    ):
        screen._prompts._launch_prompt_application(
            {
                "name": "Task61 actual dialog",
                "system_prompt": "modal System",
                "user_prompt": "modal draft",
            },
            target,
        )
        await pilot.pause()
        assert isinstance(app.screen, PromptVariablesDialog)
        await pilot.click(f"#{SYSTEM_CHECKBOX_ID}")
        await pilot.pause()
        await pilot.click(f"#{APPLY_BUTTON_ID}")
        async with asyncio.timeout(10):
            while not captures:
                await asyncio.sleep(0.01)
        capture = captures[0]
        assert await capture.applied is True and await capture.retired
    assert len(captures) == 1 and capture.checked_config[3] is None
    operation = capture.guarded_transaction.first_operation
    assert (
        operation.native_entered
        and operation.retirement.confirmed
        and operation.commit[1] is True
    )
    assert (
        session.settings.system_prompt == "modal System"
        and session.draft == "modal draft"
    )
    assert composer.draft_text() == "modal draft"
    assert (
        store.persistence.db.get_conversation_by_id(conversation)["system_prompt"]
        == "modal System"
    )


async def test_guarded_prompt_checks_original_edges_before_invoking_them(
    selected_mounted_console,
):
    from types import CellType, FunctionType
    from tldw_chatbook.Chat import console_guarded_prompt as guarded
    from tldw_chatbook.Chat.console_parent_settings import ParentSettingsSourceChanged
    from tldw_chatbook.UI.Console_Modules.prompts import PromptVariableApplication

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    owner = screen._prompts
    composer = screen._console_composer_or_none()
    application = PromptVariableApplication(
        system_text=None,
        user_text="untrusted edge must not run",
        apply_system=False,
        apply_user=True,
        destination="append_active",
        target_session_id=session.id,
        composer_fingerprint=None,
        system_fingerprint=None,
    )
    before = (session.draft, session.draft_revision, session.settings)
    calls = []
    for name, value in (
        ("_chat_store_accessor", store),
        ("_composer_accessor", composer),
        ("_ensure_active_console_session_settings_fn", session.settings),
    ):
        original = getattr(owner, name)

        def substitute(value=value, name=name):
            calls.append(name)
            return value

        setattr(owner, name, substitute)
        try:
            with pytest.raises(ParentSettingsSourceChanged, match="callback changed"):
                guarded.queue_guarded_prompt(owner, application, None)
            assert not calls
            assert not any(
                item.purpose == "guarded-prompt"
                for item in controls._parent_settings_captures
            )
        finally:
            setattr(owner, name, original)
    original = owner._composer_accessor
    owner._composer_accessor = FunctionType(
        original.__code__, original.__globals__, closure=(CellType(object()),)
    )
    try:
        with pytest.raises(ParentSettingsSourceChanged, match="callback changed"):
            guarded.queue_guarded_prompt(owner, application, None)
        assert not calls
    finally:
        owner._composer_accessor = original
    original = screen._ensure_console_chat_store
    screen._ensure_console_chat_store = lambda: calls.append("screen-store") or store
    try:
        capture = guarded.queue_guarded_prompt(owner, application, None)
        with pytest.raises(ParentSettingsSourceChanged, match="screen edge changed"):
            await capture.applied
        assert await capture.retired
        assert capture.guarded_transaction is None and capture.pending_operation is None
        assert not calls
    finally:
        del screen.__dict__["_ensure_console_chat_store"]
    assert (session.draft, session.draft_revision, session.settings) == before


async def test_prompt_target_keeps_original_none_creation_checkpoint(
    selected_mounted_console,
):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_guarded_prompt as guarded
    from tldw_chatbook.Chat import console_parent_settings as parent

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="none-target", service_epoch="local"
    )
    for _ in range(50):
        if (
            not controls._parent_settings_captures
            and not controls._session_capture_tails
            and not screen._console_sync_in_progress
        ):
            break
        await pilot.pause(0.1)
    assert (
        not controls._parent_settings_captures and not controls._session_capture_tails
    )
    created, captures = [], []

    def creation_due(frame, values):
        if frame.f_locals.get("owner") is screen._prompts:
            witness = frame.f_locals.get("witness")
            if witness is not None:
                created.append(witness)

    def capture_started(frame):
        capture = frame.f_locals["capture"]
        if capture.controls is controls and capture.purpose == "prompt-target":
            captures.append(capture)

    path = config._get_effective_config_path()
    good = path.read_bytes()
    installs = config._CONFIG_CACHE_INSTALLS
    try:
        path.write_bytes(good + b"\n[unterminated\n")
        await asyncio.to_thread(config.load_settings, True)
        assert config._CONFIG_CACHE_INSTALLS == installs
        assert config.get_config_load_failure() is not None
        with (
            _observe(
                guarded.capture_prompt_target.__code__,
                sys.monitoring.events.PY_RETURN,
                creation_due,
                values=True,
            ),
            _observe(
                parent.run_parent_settings_capture.__code__,
                sys.monitoring.events.PY_START,
                capture_started,
            ),
        ):
            for session in tuple(store.sessions()):
                store.close_session(session.id)
            assert store.active_session_id is None and not store.sessions()
            assert screen._console_composer_or_none() is not None
            assert screen._console_derivation_memo is None
            from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
                live_config_read_is_cold,
            )

            with config._settings_rebuild_lock(), config._config_file_lock():
                assert live_config_read_is_cold(screen)
            target = await guarded.capture_prompt_target(screen._prompts)
        assert len(created) == 1 and len(captures) == 1, (
            target,
            created,
            captures,
            store.active_session_id,
        )
        witness, capture = created[0], captures[0]
        session = witness[2]
        assert witness[0] is screen._session and witness[1] is store
        assert witness[-1] is asyncio.current_task()
        assert (
            store._sessions[session.id] is session and target.session_id == session.id
        )
        assert (
            capture.session_id == session.id
            and capture.operation_task is not asyncio.current_task()
        )
        assert await capture.retired and capture.effect.completed.result()
        assert capture not in controls._parent_settings_captures
        assert session.settings is not None and not session.has_user_work
        assert config._CONFIG_CACHE_INSTALLS == installs
        assert app._exception is None
    finally:
        path.write_bytes(good)
        await asyncio.to_thread(config.load_settings, True)
        assert config._CONFIG_CACHE_INSTALLS > installs


@pytest.mark.parametrize("selected_mounted_console", ["home"], indirect=True)
async def test_ordered_home_resume_target_keeps_original_none_checkpoint(
    selected_mounted_console, request
):
    from textual.widgets import Button
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.UI.test_home_console_resume import _seed_resume_target, _until
    from tldw_chatbook.Chat import console_guarded_prompt as guarded
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app, runtime, _, pilot = selected_mounted_console
    conversation_id = _seed_resume_target(app, "Selected cold resume target")
    service = app.local_chat_conversation_service
    entered, release = threading.Event(), threading.Event()
    observed, creation, target_tasks = [], [], []
    from inspect import unwrap
    from Tests.UI.test_console_send_acknowledgement import eager_tasks
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    refused_fulls, origin_rows, held_fulls, full_returns = [], [], [], []
    display_entered, display_release, display_expired = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def original_display_metadata(frame):
        owner = runtime._execution_service._display_owner
        full = owner._full
        if (
            display_entered.is_set()
            or frame.f_locals.get("self") is not owner.sources.chat_database
            or full is None
            or not full.phase_pending
            or full.session.persisted_conversation_id is not None
        ):
            return
        assert threading.current_thread() is not threading.main_thread()
        held_fulls.append(full)
        display_entered.set()
        if not display_release.wait(10):
            display_expired.set()

    def original_full_return(frame, values):
        if held_fulls and asyncio.current_task() is held_fulls[0].selection.task:
            full_returns.append(values[-1])

    def original_display_refusal(frame, values):
        import json
        from tldw_chatbook.Chat.console_display_phases import (
            ConsoleDisplaySessionChanged,
        )

        full, error = frame.f_locals["self"], values[-1]
        if (
            full.owner.runtime is not runtime
            or full in refused_fulls
            or type(error) is not ConsoleDisplaySessionChanged
        ):
            return
        refused_fulls.append(full)
        row = {
            "owner_full_same": full.owner._full is full,
            "controls_same": runtime._execution_service._local_controls
            is full.controls,
            "domain_same": full.controls._domain is full.owner.domain,
            "sources_same": full.owner.domain.sources is full.sources,
            "session_registered": full.store._sessions.get(full.session.id)
            is full.session,
            "incarnation_same": full.session.incarnation_id == full.incarnation,
            "active_same": full.store.active_session_id == full.session.id,
            "active_id": full.store.active_session_id,
            "captured_id": full.session.id,
            "prior_failure": repr(full._failed),
            "command": full.selection.command_source is not None,
            "full_entry_task": full.selection.parent_entry.task is full.selection.task,
            "phase_pending": full.phase_pending,
            "phases": [
                {
                    "native": x.native_attempted,
                    "joined": x.joined,
                    "confirmed": x.retirement.confirmed,
                }
                for x in full.retirements
            ],
        }
        origin_rows.append(row)
        request.node.user_properties.append(("display_origin_refusal", json.dumps(row)))

    def original_creation_entry(frame):
        store = frame.f_locals.get("self")
        screen = runtime.view
        if (
            creation
            or store is not runtime._chat_store
            or type(screen) is not ChatScreen
            or store.active_session_id is not None
            or not screen._console_ordered_resume_pending()
        ):
            return
        assert screen._console_composer_or_none() is not None
        controls = runtime._execution_service._local_controls
        assert controls is not None and controls._store is store
        stack, cursor = [], frame
        while cursor is not None:
            stack.append((cursor.f_code.co_qualname, cursor.f_lineno))
            cursor = cursor.f_back
        creation.append((store, screen, controls, stack))
        request.node.user_properties.append(
            ("task61_ordered_none_creator", repr(stack))
        )
        request.node.user_properties.append(
            (
                "task61_ordered_none_route",
                "preselected-graph; eager reentrant target branch",
            )
        )
        with eager_tasks():
            task = asyncio.create_task(guarded.capture_prompt_target(screen._prompts))
        target_tasks.append(task)
        assert task.done(), "Original ordered None target unexpectedly awaited work"
        assert task.result() is None
        assert store.active_session_id is None and not store.sessions()
        assert not any(
            capture.purpose == "prompt-target"
            for capture in controls._parent_settings_captures
        )

    def hold_original_tree_return(frame, values):
        if (
            observed
            or frame.f_locals.get("self") is not service
            or frame.f_locals.get("conversation_id") != conversation_id
        ):
            return
        assert threading.current_thread() is not threading.main_thread()
        observed.append((threading.get_ident(), values[-1]))
        entered.set()
        assert release.wait(10), "Original saved-tree return was not released"

    try:
        from tldw_chatbook.Chat.console_domain_display import ConsoleDisplayFull
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
        import pytest_timeout
        with (
            _observe(
                CharactersRAGDB.get_local_authority_id.__code__,
                sys.monitoring.events.PY_RETURN,
                original_display_metadata,
            ),
            _observe(
                ChatScreen._sync_native_console_chat_ui.__code__,
                sys.monitoring.events.PY_RETURN,
                original_full_return,
                values=True,
            ),
            _observe(
                ConsoleDisplayFull.require_current.__code__,
                sys.monitoring.events.RAISE,
                original_display_refusal,
                values=True,
            ),
            _observe(
                unwrap(ConsoleChatStore.create_session).__code__,
                sys.monitoring.events.PY_START,
                original_creation_entry,
            ),
            _observe(
                type(service).get_conversation_tree.__code__,
                sys.monitoring.events.PY_RETURN,
                hold_original_tree_return,
                values=True,
            ),
        ):
            assert (
                not pytest_timeout.is_debugging()
                and float(request.config.getini("timeout")) == 300
            )
            request.node.user_properties.append(
                ("effective_resume_watchdog", "300/debugger=False")
            )
            home = app.screen
            await home._refresh_home_content_snapshot().wait()
            await pilot.pause()
            resume = home.query_one("#home-resume-latest", Button)
            assert "Selected cold resume target" in str(resume.label)
            resume.press()
            await _entered(entered)
            await _entered(display_entered)
            assert len(held_fulls) == 1 and not held_fulls[0].physically_retired
            screen = app._navigation_outgoing_screen()
            assert type(screen) is ChatScreen and runtime.view is screen
            store = runtime.ensure_chat_store()
            assert screen._console_ordered_resume_pending()
            assert (
                len(creation) == 1
                and creation[0][0] is store
                and creation[0][1] is screen
            )
            assert target_tasks[0].result() is None
            release.set()
            await _until(
                lambda: store.active_session_id is not None
                and store._sessions[store.active_session_id].persisted_conversation_id
                == conversation_id
            )
            assert held_fulls[0].session.id != store.active_session_id
            display_release.set()
            await _until(lambda: not screen._resume_navigation_startup_in_progress)
            worker = screen._resume_navigation_startup_worker
            if worker is not None:
                async with asyncio.timeout(10):
                    await worker.wait()
            assert (
                store._session_or_raise(
                    store.active_session_id
                ).persisted_conversation_id
                == conversation_id
            )
            assert (
                sum(
                    session.persisted_conversation_id == conversation_id
                    for session in store.sessions()
                )
                == 1
            )
            request.node.user_properties.append(
                ("ordered_resume_total_sessions", str(len(store.sessions())))
            )
            assert not screen._console_ordered_resume_pending()
            async with asyncio.timeout(10):
                await asyncio.shield(
                    asyncio.gather(held_fulls[0].selection.task, return_exceptions=True)
                )
            assert refused_fulls == held_fulls and full_returns == [False]
            assert not display_expired.is_set()
            row = origin_rows[0]
            assert row["active_same"] is False and row["prior_failure"] == "None"
            assert row["command"] is False
            assert all(
                row[key]
                for key in (
                    "owner_full_same",
                    "controls_same",
                    "domain_same",
                    "sources_same",
                    "session_registered",
                    "incarnation_same",
                    "full_entry_task",
                )
            )
            full = held_fulls[0]
            assert full.physically_retired and full.phase_issuance_closed
            assert full.logical.retired.done() and not full.logical.callback_returned
            assert (
                full.effect.completed.result() is True
                and full.effect.physically_retired
            )
            assert all(task.done() for task in full.tasks.values())
            assert all(future.cancelled() for _, _, future in full.sections.values())
            assert len(observed) == 1 and app._exception is None
    finally:
        release.set()
        display_release.set()
        if target_tasks:
            await asyncio.gather(*target_tasks, return_exceptions=True)
        for full in refused_fulls:
            async with asyncio.timeout(10):
                await asyncio.shield(
                    asyncio.gather(full.selection.task, return_exceptions=True)
                )
            request.node.user_properties.append(
                (
                    "display_refusal_settlement",
                    repr(
                        {
                            "physical": full.physically_retired,
                            "logical_done": full.logical.retired.done(),
                            "effect_done": full.effect.completed.done(),
                            "effect_retired": full.effect.physically_retired,
                            "callback_returned": full.logical.callback_returned,
                        }
                    ),
                )
            )


async def _whole_core_until(pilot, predicate, *, timeout):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def _whole_core_settled(app, screen, controls, pilot):
    # Existing FULL node's owner predicate, after async fixture handoff.
    def settled():
        readiness = getattr(screen, "_console_readiness_config_projection", None)
        context = getattr(screen, "_console_context_read_snapshot", None)
        return not (
            screen._console_sync_in_progress
            or getattr(screen, "_console_control_bar_replay_whole_sync", False)
            or screen._console_control_bar_sync_scheduled
            or getattr(readiness, "pending", False)
            or getattr(context, "pending_key", None) is not None
            or controls._parent_settings_captures
            or controls._session_capture_tails
            or any(
                w.group
                in {
                    "console-sync",
                    "console-readiness-config",
                    "console-readiness-publication",
                }
                and not w.is_finished
                for w in app.workers
            )
        )

    await _whole_core_until(pilot, settled, timeout=5)


@contextmanager
def _whole_core_trace(screen, runtime, controls):
    from tldw_chatbook.Chat import console_parent_settings as parent
    from tldw_chatbook.UI.Console_Modules import (
        config_sync,
        console_spend_projection as spend,
    )
    from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusChips

    rows = dict(
        full=[],
        core=[],
        core_entry=[],
        refusal=[],
        cold=[],
        replay=[],
        notice=[],
        tokens=[],
        native=[],
        identity=[],
        identity_posts=[],
        render_admission=[],
        due_refusal=[],
        full_capture=[],
    )
    identity_body = type(screen)._sync_console_identity_surfaces.__wrapped__.__code__
    mounted_chips = tuple(screen.query(ConsoleStatusChips))

    def owner(frame):
        # Observation only: do not call authority getters/validators here.
        cursor = frame
        while cursor is not None:
            if cursor.f_code is config_sync.sync_selected_console_core_state.__code__:
                source = cursor.f_locals.get("source")
                task = asyncio.current_task()
                command = controls._settings_command_tasks.get(task)
                parent_owned = (
                    type(source) is parent.ParentSettingsCapture
                    and source in controls._parent_settings_captures
                    and source.operation_task is task
                    and source.started
                )
                return (
                    task,
                    source,
                    (source is not None and source is command) or parent_owned,
                )
            cursor = cursor.f_back
        return asyncio.current_task(), None, False

    def full(frame, values):
        if frame.f_locals.get("self") is screen:
            rows["full"].append(
                (
                    time.monotonic(),
                    asyncio.current_task(),
                    "start" if len(values) == 1 else "return",
                    None if len(values) == 1 else values[-1],
                    controls._settings_command_tasks.get(asyncio.current_task()),
                    getattr(screen, "_console_control_bar_replay_whole_sync", False),
                )
            )

    def identity_post(frame, values):
        if (
            frame.f_locals.get("self") is screen
            and frame.f_back.f_code is identity_body
        ):
            task = asyncio.current_task()
            capture = next(
                (
                    c
                    for c in controls._parent_settings_captures
                    if c.operation_task is task and c.purpose == "identity-surfaces"
                ),
                None,
            )
            if capture is not None:
                rows["identity_posts"].append(
                    (capture, frame.f_code.co_name, time.monotonic())
                )

    def identity(frame, values):
        if frame.f_locals.get("self") is screen:
            task = asyncio.current_task()
            capture = next(
                (
                    c
                    for c in controls._parent_settings_captures
                    if c.operation_task is task and c.purpose == "identity-surfaces"
                ),
                None,
            )
            rows["identity"].append(
                (
                    time.monotonic(),
                    task,
                    capture,
                    "start" if len(values) == 1 else "return",
                )
            )

    def core(frame, values):
        if len(values) > 1:
            identity_post(frame, values)
        if (
            frame.f_locals.get("self") is screen
            and frame.f_back.f_code is config_sync._push_console_core_state.__code__
        ):
            if len(values) == 1:
                rows["core_entry"].append((time.monotonic(), *owner(frame)))
                return
            selection = values[-1]
            rows["core"].append(
                (
                    time.monotonic(),
                    *owner(frame),
                    selection,
                    getattr(runtime.chat_controller, "provider", None),
                    getattr(runtime.chat_controller, "configured_model", None),
                )
            )

    def refusal(frame, values):
        if frame.f_locals.get("screen") is not screen:
            return
        request = getattr(screen, "_console_config_handoffs", {}).get("core")
        token = getattr(request, "token", None)
        if token is not None and token not in rows["tokens"]:
            rows["tokens"].append(token)
        rows["refusal"].append(
            (
                time.monotonic(),
                *owner(frame),
                "start" if len(values) == 1 else "return",
                getattr(screen, "_console_core_refused_since", None),
                getattr(screen, "_console_config_unreadable_installs", None),
                getattr(screen, "_console_control_bar_replay_whole_sync", False),
            )
        )

    def cold(frame, values):
        if (
            frame.f_locals.get("screen") is screen
            and frame.f_locals.get("owner") == "sync_console_core_state"
        ):
            rows["cold"].append((time.monotonic(), values[-1]))

    def replay(frame, values):
        if frame.f_locals.get("screen") is screen:
            caller = frame.f_back
            rows["replay"].append(
                (
                    time.monotonic(),
                    *owner(frame),
                    caller.f_code.co_filename,
                    caller.f_code.co_name,
                    caller.f_lineno,
                )
            )

    def full_capture(frame, values):
        if (
            frame.f_locals.get("screen") is screen
            and frame.f_locals.get("purpose") == "early-full-defaults"
        ):
            capture = frame.f_locals.get("capture")
            if capture is not None:
                rows["full_capture"].append(
                    (
                        asyncio.current_task(),
                        capture,
                        capture.sources,
                        capture in controls._parent_settings_captures,
                    )
                )

    def render_admission(frame, values):
        if frame.f_locals.get("screen") is not screen:
            return
        pending = frame.f_locals.get("pending")
        tail = frame.f_locals.get("tail")
        capture = frame.f_locals.get("capture")
        error = sys.exception()
        stack, cursor = [], frame.f_back
        while cursor is not None and len(stack) < 14:
            stack.append(
                (cursor.f_code.co_filename, cursor.f_code.co_name, cursor.f_lineno)
            )
            cursor = cursor.f_back
        rows["render_admission"].append(
            (
                time.monotonic(),
                id(asyncio.current_task()),
                values[-1],
                type(error).__name__,
                str(error),
                id(pending) if pending is not None else None,
                getattr(getattr(pending, "source", None), "purpose", None),
                id(tail) if tail is not None else None,
                tail.done() if tail is not None else None,
                id(capture) if capture is not None else None,
                getattr(capture, "purpose", None),
                id(getattr(capture, "operation_task", None)),
                getattr(screen, "_console_sync_in_progress", None),
                stack,
            )
        )

    # RAISE is global-only in Python 3.12. Observe only executed raise
    # statement lines in this original validator, without global tracing.
    import ast
    import inspect
    import textwrap

    due_lines, due_start = inspect.getsourcelines(parent._validate_due_settings_read)
    due_raises = {
        due_start + node.lineno - 1: ast.unparse(node.exc)
        for node in ast.walk(ast.parse(textwrap.dedent("".join(due_lines))))
        if isinstance(node, ast.Raise) and node.exc is not None
    }

    def due_refusal(frame, values):
        due = frame.f_locals.get("request")
        raised = due_raises.get(values[-1])
        if due is None or due.screen is not screen or raised is None:
            return
        proof = getattr(getattr(due, "result", None), "_display_proof", None)
        rows["due_refusal"].append(
            (
                time.monotonic(),
                id(asyncio.current_task()),
                raised,
                getattr(due.source, "purpose", None),
                time.monotonic() - proof.at if proof is not None else None,
                frame.f_lineno,
            )
        )

    def notice(frame, values):
        if frame.f_locals.get("self") in mounted_chips:
            shown = frame.f_locals["shown"]
            if not rows["notice"] or rows["notice"][-1][1] != shown:
                rows["notice"].append((time.monotonic(), shown))

    def token_begin(frame, values):
        request = frame.f_locals.get("self")
        token = values[-1]
        if request.screen is screen and request.owner == "core" and token is not None:
            rows["tokens"].append(token)

    projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)

    def native(frame, values):
        if frame.f_locals.get("projection") is projection:
            rows["native"].append(
                (
                    time.monotonic(),
                    threading.get_ident(),
                    frame.f_back.f_locals.get("read"),
                    projection.read_current,
                )
            )

    monitoring, events = sys.monitoring, sys.monitoring.events
    callbacks = {
        type(screen)._sync_native_console_chat_ui.__code__: (
            events.PY_START | events.PY_RETURN,
            full,
        ),
        type(screen)._sync_console_chat_core_state.__code__: (
            events.PY_START | events.PY_RETURN,
            core,
        ),
        config_sync._refuse_console_core_state.__code__: (
            events.PY_START | events.PY_RETURN,
            refusal,
        ),
        spend.defer_cold_live_read.__code__: (events.PY_RETURN, cold),
        config_sync.replay_whole_console_sync.__code__: (events.PY_START, replay),
        ConsoleStatusChips.sync_config_unreadable.__code__: (events.PY_RETURN, notice),
        config_sync._ConsoleConfigHandoff.begin.__code__: (
            events.PY_RETURN,
            token_begin,
        ),
        projection.read_current.__code__: (events.PY_START, native),
        parent.defer_parent_settings_render.__code__: (
            events.PY_RETURN,
            render_admission,
        ),
        parent._validate_due_settings_read.__code__: (events.LINE, due_refusal),
        parent.prepare_parent_default_settings.__code__: (
            events.PY_YIELD,
            full_capture,
        ),
        identity_body: (events.PY_START | events.PY_RETURN, identity),
        type(screen)._sync_console_settings_summary.__code__: (
            events.PY_RETURN,
            identity_post,
        ),
        type(screen)._sync_console_rail_system_line.__code__: (
            events.PY_RETURN,
            identity_post,
        ),
        type(screen)._sync_console_control_bar.__code__: (
            events.PY_RETURN,
            identity_post,
        ),
    }
    slot = next(
        value
        for value in range(6)
        if value != monitoring.DEBUGGER_ID and monitoring.get_tool(value) is None
    )
    monitoring.use_tool_id(slot, "task61-whole-core")

    def observe(code, *values):
        entry = callbacks.get(code)
        if entry is not None:
            entry[1](sys._getframe(1), values)

    try:
        monitoring.register_callback(slot, events.PY_START, observe)
        monitoring.register_callback(slot, events.PY_RETURN, observe)
        monitoring.register_callback(slot, events.LINE, observe)
        monitoring.register_callback(slot, events.PY_YIELD, observe)
        for code, (event, _) in callbacks.items():
            monitoring.set_local_events(slot, code, event)
        yield rows
    finally:
        for code in callbacks:
            monitoring.set_local_events(slot, code, 0)
        monitoring.register_callback(slot, events.PY_START, None)
        monitoring.register_callback(slot, events.PY_RETURN, None)
        monitoring.register_callback(slot, events.LINE, None)
        monitoring.register_callback(slot, events.PY_YIELD, None)
        monitoring.free_tool_id(slot)


@asynccontextmanager
async def _whole_core_cleanup(config, path, original, receipts, held, expected=None):
    # held is a test-owned list of exact (thread, release, expired) triples.
    # Keep the first assertion/cancellation and each cleanup error.
    errors = []
    try:
        yield
    except BaseException as error:
        # Preserve the body error even if actual App teardown cannot finish.
        import traceback

        traceback.print_exception(error, file=sys.__stderr__)
        errors.append(error)
    finally:
        for _thread, release, _expired in held:
            release.set()
        for thread, _release, expired in held:
            try:
                await asyncio.to_thread(thread.join, 5)
                assert not thread.is_alive(), "Exact config lock owner did not retire"
                assert not expired.is_set(), "Original finite config lock hold expired"
            except BaseException as error:
                errors.append(error)
        try:
            path.write_bytes(original)
            await asyncio.to_thread(config.load_settings, True)
        except BaseException as error:
            errors.append(error)
        for receipt in receipts:
            for future in (receipt.work, receipt.applied):
                try:
                    await asyncio.wait_for(asyncio.shield(future), 10)
                except BaseException as error:
                    if expected is None or expected.get(future) is not error:
                        errors.append(error)
    if errors:
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("Whole-core body and cleanup", errors)


async def _whole_core_hold_file_lock(config, held):
    # Same real other-thread file-lock seam/bounds as _hold_config_file_lock:
    # entered 5, finite release 10, joined 5; never wait on UI thread.
    entered, release, expired = threading.Event(), threading.Event(), threading.Event()

    def hold():
        with config._config_file_lock():
            entered.set()
            if not release.wait(10):
                expired.set()

    thread = threading.Thread(target=hold)
    held.append((thread, release, expired))
    thread.start()
    assert await asyncio.to_thread(entered.wait, 5)
    return release


def _whole_core_dispatch(controls, store, receipts, label):
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        ReadProjection,
    )

    session = store._sessions[store.active_session_id]
    identity = ControlIdentity(
        "mounted", "local", len(receipts) + 1, label, session.id, session.incarnation_id
    )
    from Tests.UI.test_console_send_acknowledgement import eager_tasks

    with eager_tasks():
        receipt = controls.dispatch(ReadProjection(identity))
    receipts.append(receipt)
    return receipt


def _whole_core_notice_visible(screen):
    from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusChips

    return any(
        segment.name == "config-unreadable" and segment.visible
        for chips in screen.query(ConsoleStatusChips)
        for segment in chips._build_segments()
    )


def _whole_core_watchdog(request):
    """Check the installed timeout plugin while both real observers are active."""
    import hashlib
    from pathlib import Path
    import pytest_timeout
    from Tests.Chat.test_console_control_trace import _observe

    with _observe(
        _whole_core_notice_visible.__code__,
        sys.monitoring.events.PY_START,
        lambda frame: None,
    ):
        assert not pytest_timeout.is_debugging()
        assert float(request.config.getini("timeout")) == 300
        active = tuple(
            (slot, sys.monitoring.get_tool(slot))
            for slot in range(6)
            if sys.monitoring.get_tool(slot) is not None
        )
        assert any(name == "task61-whole-core" for _, name in active)
        assert any(name == "task60-trace-custody" for _, name in active)
        request.node.user_properties.append(
            ("task61_effective_watchdog", repr((300, False, active)))
        )
        plugin = Path(pytest_timeout.__file__)
        request.node.user_properties.append(
            (
                "task61_timeout_plugin",
                f"{plugin}:{hashlib.sha256(plugin.read_bytes()).hexdigest()}",
            )
        )


async def test_selected_whole_core_invalid_toml_reaches_due_fallback(
    selected_mounted_console, request
):
    from tldw_chatbook import config

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    path = config._get_effective_config_path()
    original = path.read_bytes()
    receipts, held, expected = [], [], {}
    async with _whole_core_cleanup(config, path, original, receipts, held, expected):
        await _whole_core_settled(app, screen, controls, pilot)
        with _whole_core_trace(screen, runtime, controls) as rows:
            _whole_core_watchdog(request)
            await pilot.pause(2)
            idle_fulls = sum(row[2] == "start" for row in rows["full"])
            for observed in rows.values():
                observed.clear()
            installs = config._CONFIG_CACHE_INSTALLS
            path.write_bytes(original + b"\n[unterminated\n")
            await asyncio.to_thread(config.load_settings, True)
            assert config._settings_cache_hit(path) is None
            assert config.get_config_load_failure() is not None
            assert config._CONFIG_CACHE_INSTALLS == installs
            receipt = _whole_core_dispatch(controls, store, receipts, "invalid-core")
            screen._sync_console_identity_surfaces()
            await _whole_core_until(
                pilot, lambda: any(row[3] for row in rows["core"]), timeout=10
            )
            # Actual selected source and actual method return, not FULL True.
            assert any(
                row[1] is receipt.work and row[2] == "start" for row in rows["full"]
            )
            assert any(
                row[3]
                and row[4].provider == row[5]
                and row[4].configured_model == row[6]
                for row in rows["core"]
            )
            request.node.user_properties.append(
                (
                    "task61_core_routes",
                    repr(
                        [
                            (row[1] is receipt.work, type(row[2]).__name__, row[3])
                            for row in rows["core"]
                        ]
                    ),
                )
            )
            assert [value for _, value in rows["cold"]][:2] == [True, False], rows[
                "cold"
            ]
            await _whole_core_until(
                pilot,
                lambda: any(
                    row[2] is not None and row[3] == "return"
                    for row in rows["identity"]
                ),
                timeout=10,
            )
            completed = next(
                row[2]
                for row in rows["identity"]
                if row[2] is not None and row[3] == "return"
            )
            assert [
                name
                for capture, name, _ in rows["identity_posts"]
                if capture is completed
            ] == [
                "_sync_console_chat_core_state",
                "_sync_console_settings_summary",
                "_sync_console_rail_system_line",
                "_sync_console_control_bar",
            ]
            await _whole_core_until(pilot, lambda: completed.retired.done(), timeout=10)
            assert await completed.retired and completed.applied.result() is None
            assert completed not in controls._parent_settings_captures
            _, identity_request, identity_token = completed.checked_config
            assert identity_token is None or not identity_token.active
            try:
                await receipt.applied
            except RuntimeError as error:
                assert str(error) == "Original Console FULL publication was deferred"
                assert any(
                    row[1] is receipt.work and row[2] == "return" and row[3] is False
                    for row in rows["full"]
                )
                expected[receipt.applied] = error
                request.node.user_properties.append(
                    (
                        "task61_command_full",
                        "original coalescing refused; selected fallback source separately observed",
                    )
                )
            request.node.user_properties.append(
                (
                    "task61_identity_posts",
                    repr(
                        [
                            name
                            for capture, name, _ in rows["identity_posts"]
                            if capture is completed
                        ]
                    ),
                )
            )
            assert config._settings_cache_hit(path) is None
            assert config._CONFIG_CACHE_INSTALLS == installs
            await pilot.pause(0.5)
            count = sum(row[2] == "start" for row in rows["full"])
            await pilot.pause(2)
            request.node.user_properties.append(
                (
                    "task61_replay_callers",
                    repr([(type(row[2]).__name__, row[4:]) for row in rows["replay"]]),
                )
            )
            for label in ("render_admission", "due_refusal"):
                request.node.user_properties.append(
                    ("task61_" + label, repr(rows[label]))
                )
            assert (
                sum(row[2] == "start" for row in rows["full"]) - count <= idle_fulls + 1
            ), rows["replay"]
            assert app._exception is None


@pytest.mark.parametrize("unreadable", [False, True], ids=["parseable", "unreadable"])
async def test_selected_whole_core_holds_original_refusal_until_reader_retires(
    selected_mounted_console, unreadable, request
):
    from tldw_chatbook import config
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
    )

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    path, receipts, held = config._get_effective_config_path(), [], []
    original = path.read_bytes()
    async with _whole_core_cleanup(config, path, original, receipts, held):
        await _whole_core_settled(app, screen, controls, pilot)
        with _whole_core_trace(screen, runtime, controls) as rows:
            _whole_core_watchdog(request)
            if unreadable:
                installs = config._CONFIG_CACHE_INSTALLS
                path.write_bytes(original + b"\n[unterminated\n")
                await asyncio.to_thread(config.load_settings, True)
                assert config.get_config_load_failure() is not None
                assert (
                    config._settings_cache_hit(path) is None
                    and config._CONFIG_CACHE_INSTALLS == installs
                )
            else:
                assert config.get_config_load_failure() is None
            release = await _whole_core_hold_file_lock(config, held)
            held_at = time.monotonic()
            receipt = _whole_core_dispatch(controls, store, receipts, "held-whole-core")
            projection = ConsoleReadinessConfigProjection.for_screen(screen)
            await _whole_core_until(
                pilot,
                lambda: (
                    any(
                        row[0] >= held_at and row[3] and row[4] == "return"
                        for row in rows["refusal"]
                    )
                    and projection.pending
                    and bool(projection._reads)
                    and any(
                        at >= held_at
                        and read in projection._reads
                        and reader is projection.read_current
                        for at, _, read, reader in rows["native"]
                    )
                ),
                timeout=5,
            )
            reads = tuple(projection._reads)
            assert reads and all(
                read._producer is not None
                and not read._producer.done()
                and not read.retired.done()
                for read in reads
            )
            assert screen._console_sync_in_progress
            assert not [row for row in rows["core_entry"] if row[0] >= held_at]
            assert any(
                row[1] is receipt.work and row[2] == "start" for row in rows["full"]
            )
            assert rows["tokens"] and any(token.active for token in rows["tokens"])
            # A pending read cannot certify the independent coalesced core gate.
            core_handoff = screen._console_config_handoffs["core"]
            assert core_handoff.receipt is None
            full_tasks = {
                row[1]
                for row in rows["full"]
                if row[0] >= held_at and row[2] == "start" and not row[1].done()
            }
            assert len(full_tasks) == 1, rows["full"]
            held_full = full_tasks.pop()
            # Bind this actual pending FULL to its original warm/read owner.
            coroutine, warms = held_full.get_coro(), []
            while coroutine is not None:
                frame = getattr(coroutine, "cr_frame", None)
                if (
                    frame is not None
                    and frame.f_code is type(projection).warm.__code__
                    and frame.f_locals.get("self") is projection
                ):
                    warms.append(coroutine)
                coroutine = getattr(coroutine, "cr_await", None)
            assert len(warms) == 1
            if held_full is receipt.work:
                assert not receipt.applied.done()
            else:
                assert receipt.applied.done() and receipt.applied.exception() is None
                assert any(
                    task is held_full
                    and registered
                    and source is controls._domain.sources
                    and any(row[2] is capture and row[3] for row in rows["refusal"])
                    for task, capture, source, registered in rows["full_capture"]
                ), rows["full_capture"]
            request.node.user_properties.append(
                (
                    "task61_held_full",
                    "command"
                    if held_full is receipt.work
                    else "autonomous-selected; command returned original projection before owed replay",
                )
            )
            request.node.user_properties.append(
                (
                    "task61_held_refusal_routes",
                    repr(
                        [
                            (row[1] is receipt.work, type(row[2]).__name__, row[3])
                            for row in rows["refusal"]
                            if row[0] >= held_at
                        ]
                    ),
                )
            )
            request.node.user_properties.append(
                (
                    "task61_episode_limit",
                    "first refusal and retained whole FULL; no repeated three-second probe claim",
                )
            )
            release.set()
            await _whole_core_until(
                pilot, lambda: all(read.retired.done() for read in reads), timeout=10
            )
            assert all(
                read._producer.done() and read.callback_returned for read in reads
            )
            async with asyncio.timeout(10):
                await asyncio.shield(held_full)
                await receipt.applied
                await receipt.work
            assert any(
                row[1] is held_full and row[2] == "return" for row in rows["full"]
            )
            if unreadable:
                path.write_bytes(original)
                await asyncio.to_thread(config.load_settings, True)
                assert config._CONFIG_CACHE_INSTALLS > installs
            await _whole_core_settled(app, screen, controls, pilot)
            recovered_at = time.monotonic()
            recovered = _whole_core_dispatch(
                controls, store, receipts, "released-whole-core"
            )
            await _whole_core_until(
                pilot,
                lambda: any(row[0] >= recovered_at and row[3] for row in rows["core"]),
                timeout=10,
            )
            assert any(
                row[3]
                and row[4].provider == row[5]
                and row[4].configured_model == row[6]
                for row in rows["core"]
                if row[0] >= recovered_at
            )
            await _whole_core_until(
                pilot,
                lambda: all(not token.active for token in rows["tokens"]),
                timeout=10,
            )
            assert not _whole_core_notice_visible(screen) and app._exception is None
            request.node.user_properties.append(
                (
                    "task61_recovery_routes",
                    repr(
                        [
                            (row[1] is recovered.work, type(row[2]).__name__, row[3])
                            for row in rows["core"]
                            if row[0] >= recovered_at
                        ]
                    ),
                )
            )


@pytest.mark.parametrize("mode", ["complete", "newer-draft"])
async def test_guarded_prompt_retains_its_actual_post_display_due_convergence(
    selected_mounted_console,
    mode,
):
    from contextlib import ExitStack
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
    from tldw_chatbook import config
    from tldw_chatbook.Chat import (
        console_guarded_prompt as guarded,
        console_parent_settings as parent,
    )
    from tldw_chatbook.UI.Console_Modules.prompts import (
        PromptVariableApplication,
        fingerprint_system_text,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    assert (
        not session.has_user_work
        and not session.draft
        and session.settings.system_prompt is None
    )
    conversation = store.persistence.create_conversation(
        conversation_title="Pristine post-display convergence"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    application = PromptVariableApplication(
        system_text="",
        user_text=None,
        apply_system=True,
        apply_user=False,
        destination="append_active",
        target_session_id=session.id,
        composer_fingerprint=None,
        system_fingerprint=fingerprint_system_text(""),
    )
    app.pending_handoffs.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, application)
    entered, release = threading.Event(), threading.Event()
    captures, due_results, advances = [], [], []

    def capture_entry(frame):
        capture = frame.f_locals["capture"]
        if capture.controls is controls and capture.purpose == "guarded-prompt":
            captures.append(capture)

    def writer_return(frame):
        if (
            frame.f_locals.get("self") is store
            and frame.f_locals.get("session") is session
        ):
            assert threading.current_thread() is not threading.main_thread()
            entered.set()
            assert release.wait(10), "Original unchanged System writer was not released"

    def due_consumed(frame):
        request = controls._due_settings_reads.get(asyncio.current_task())
        if captures and request is not None and request.source is captures[0]:
            due_results.append(request)

    def post_advance(frame):
        capture = frame.f_locals["capture"]
        if not captures or capture is not captures[0]:
            return
        request = frame.f_locals["request"]
        advances.append((capture.guarded_transaction.settings_state, request))
        if mode == "newer-draft":
            screen._console_composer_or_none().insert_text("newer authored draft")

    path = config._get_effective_config_path()
    good = path.read_bytes()
    work, first_error, expected_error = None, None, False
    try:
        with ExitStack() as stack:
            stack.enter_context(
                _observe(
                    guarded.run_guarded_prompt.__code__,
                    sys.monitoring.events.PY_START,
                    capture_entry,
                )
            )
            stack.enter_context(
                _observe(
                    store._write_session_system_prompt.__func__.__code__,
                    sys.monitoring.events.PY_RETURN,
                    writer_return,
                )
            )
            stack.enter_context(
                _observe(
                    parent.consume_due_settings_result.__code__,
                    sys.monitoring.events.PY_RETURN,
                    due_consumed,
                )
            )
            stack.enter_context(
                _observe(
                    guarded._advance_guarded_post_settings.__code__,
                    sys.monitoring.events.PY_START,
                    post_advance,
                )
            )
            from Tests.UI.test_console_send_acknowledgement import eager_tasks

            with eager_tasks():
                work = asyncio.create_task(
                    screen._prompts._consume_pending_console_prompt_insert()
                )
            async with asyncio.timeout(10):
                while not entered.is_set():
                    if work.done():
                        work.result()
                        pytest.fail(
                            "Original guarded System writer did not run: "
                            f"captures={[(c.started, c.guarded_transaction, c.applied.done()) for c in captures]!r}; "
                            f"pending={app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_PROMPT_INSERT)}"
                        )
                    await asyncio.sleep(0.01)
            capture = captures[0]
            first = capture.guarded_transaction.first_operation
            assert first.native_entered and worker_leases(store.persistence.db)
            assert not session.has_user_work and not capture.applied.done()
            installs = config._CONFIG_CACHE_INSTALLS
            path.write_bytes(good + b"\n[unterminated\n")
            await asyncio.to_thread(config.load_settings, True)
            assert (
                config.get_config_load_failure() is not None
                and config._CONFIG_CACHE_INSTALLS == installs
            )
            release.set()
            async with asyncio.timeout(15):
                if mode == "newer-draft":
                    with pytest.raises(
                        parent.ParentSettingsOriginChanged,
                        match="resulting state changed",
                    ):
                        await work
                    expected_error = True
                else:
                    await work
                assert await capture.retired
            assert first.retirement.confirmed and first.commit is not None
            assert advances and advances[-1][1].convergences
            if mode == "complete":
                assert due_results and due_results[-1].consumed
            else:
                assert not due_results
                assert session.draft == "newer authored draft"
                assert capture.guarded_transaction.settings_state is advances[-1][0]
            convergence = capture.pending_operation
            assert convergence is not first and convergence.convergence is not None
            assert (
                convergence.source is capture
                and convergence.commit is not None
                and convergence.retired
            )
            assert (
                convergence.retirement is not None and convergence.retirement.confirmed
            )
            assert (
                capture.guarded_transaction.completed
                and capture.guarded_transaction.result
                is (True if mode == "complete" else None)
            )
            assert (
                capture.guarded_claim_settled
                and not app.pending_handoffs.has_pending(
                    HandoffChannel.CONSOLE_PROMPT_INSERT
                )
            )
            assert (
                store.persistence.db.get_conversation_by_id(conversation)[
                    "system_prompt"
                ]
                is None
            )
            assert app._exception is None
    except BaseException as error:
        first_error = error
        raise
    finally:
        release.set()
        path.write_bytes(good)
        await asyncio.to_thread(config.load_settings, True)
        if work is not None:
            outcome = await asyncio.gather(work, return_exceptions=True)
            if (
                first_error is None
                and not expected_error
                and isinstance(outcome[0], BaseException)
            ):
                raise outcome[0]


async def test_selected_editor_opens_after_original_cold_ensure(
    selected_mounted_console,
):
    from textual.widgets import TextArea
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_parent_settings as parent
    from tldw_chatbook.Widgets.Console.console_system_prompt_modal import (
        ConsoleSystemPromptModal,
        TEXT_AREA_ID,
    )

    app, runtime, screen, pilot = selected_mounted_console
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    await _whole_core_settled(app, screen, controls, pilot)
    path, captures, due = config._get_effective_config_path(), [], []
    original = path.read_bytes()

    def entered(frame):
        capture = frame.f_locals["capture"]
        if capture.controls is controls and capture.purpose == "prompt-editor-open":
            captures.append(capture)

    def read(frame, values):
        source = frame.f_locals.get("source")
        if getattr(source, "purpose", None) == "prompt-editor-open":
            due.append(values[-1])

    try:
        path.write_bytes(original + b"\n[unterminated\n")
        await asyncio.to_thread(config.load_settings, True)
        assert config._settings_cache_hit(path) is None
        with (
            _observe(
                parent.run_parent_settings_capture.__code__,
                sys.monitoring.events.PY_START,
                entered,
            ),
            _observe(
                parent.settle_parent_action_defaults.__code__,
                sys.monitoring.events.PY_RETURN,
                read,
                values=True,
            ),
        ):
            screen.action_open_console_system_prompt_editor()
            await _whole_core_until(
                pilot,
                lambda: isinstance(app.screen, ConsoleSystemPromptModal),
                timeout=10,
            )
            assert len(captures) == 1 and due
            capture = captures[0]
            await _whole_core_until(pilot, lambda: capture.retired.done(), timeout=10)
            assert capture.applied.result() is None and capture.retired.result()
            assert capture not in controls._parent_settings_captures
            assert app.screen.query_one(f"#{TEXT_AREA_ID}", TextArea).text == (
                capture.session.settings.system_prompt or ""
            )
            app.screen.dismiss(None)
            await pilot.pause()
            assert app._exception is None
    finally:
        path.write_bytes(original)
        await asyncio.to_thread(config.load_settings, True)
        for capture in captures:
            await asyncio.gather(capture.work, capture.applied, return_exceptions=True)


@pytest.mark.parametrize(
    "mode", ["complete", "newer-draft", "switch", "observer-cancel"]
)
async def test_selected_prefill_keeps_original_native_and_output_owners(
    selected_mounted_console, mode
):
    import json
    import sqlite3
    from textual.widgets import Button
    from textual.worker import get_current_worker, WorkerState
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
    from tldw_chatbook.Chat.console_control_settings import PendingSettingsApply
    from tldw_chatbook.Chat import console_parent_settings as parent

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    conversation = store.persistence.create_conversation(
        conversation_title="Saved prefill owner"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    await _whole_core_settled(app, screen, controls, pilot)
    composer = screen._console_composer_or_none()
    entered, release = threading.Event(), threading.Event()
    captures, operations, callers, workers = [], [], [], []
    rewarm_dependencies = []
    transcript_held = asyncio.Event()
    transcript_locks = []
    transcript_sources = []

    def transcript_entry(frame):
        if mode != "newer-draft" or transcript_locks or not captures:
            return
        if (
            frame.f_locals.get("self") is not screen
            or asyncio.current_task() is not captures[0].operation_task
        ):
            return
        lock = screen._change_review_projection._prepare_lock
        assert type(lock) is asyncio.Lock and not lock.locked()
        acquired = asyncio.Task(
            lock.acquire(), loop=asyncio.get_running_loop(), eager_start=True
        )
        assert acquired.done() and acquired.result()
        transcript_locks.append(lock)
        projection = screen._console_readiness_config_projection
        transcript_sources.append((projection, projection.key))
        transcript_held.set()

    def operation_entry(frame):
        operation = frame.f_locals["self"]
        if operation.controls is controls and operation.mutation == (
            "pinned_prefill",
            "Voice:",
        ):
            operations.append(operation)
            capture = operation.parent_capture
            assert (
                type(capture) is parent.ParentSettingsCapture
                and capture.purpose == "prefill-command"
            )
            if capture not in captures:
                captures.append(capture)

    def caller(frame):
        if (
            frame.f_locals.get("screen") is screen
            and frame.f_locals.get("purpose") == "prefill-command"
        ):
            callers.append(asyncio.current_task())
            workers.append(get_current_worker())

    def writer_return(frame):
        if (
            frame.f_locals.get("self") is store
            and frame.f_locals.get("session") is session
        ):
            assert threading.current_thread() is not threading.main_thread()
            entered.set()
            assert release.wait(10), "Original pinned writer return was not released"

    def readiness_return(frame):
        dependency = frame.f_locals.get("dependency")
        if (
            dependency is None
            or dependency.capture.controls is not controls
            or not captures
        ):
            return
        entry = controls._settings_readiness_waits.get(dependency.producer)
        if entry is None or entry[0] is not captures[0]:
            return
        task = captures[0].operation_task
        awaitable, names = task.get_coro(), []
        while awaitable is not None:
            code = getattr(awaitable, "cr_code", None)
            if code is not None:
                names.append(code.co_name)
            awaitable = getattr(awaitable, "cr_await", None)
        if "rewarm_console_presentation" not in names:
            return
        assert entry[1] is task and entry[2] is dependency.projection
        assert entry[3] is dependency.handoff and entry[4] is dependency.key
        assert not dependency.previous.done() and not dependency.capture.started
        assert dependency.producer is dependency.handoff[2]._task
        rewarm_dependencies.append(dependency)

    stage = "observers"

    def diagnostic(label, *, first_error=False):
        def task_state(task):
            if task is None:
                return None
            chain, seen = [], set()
            awaitable = task.get_coro() if isinstance(task, asyncio.Task) else None
            while awaitable is not None and id(awaitable) not in seen:
                seen.add(id(awaitable))
                frame = getattr(awaitable, "cr_frame", None)
                code = getattr(awaitable, "cr_code", None)
                if code is not None:
                    chain.append(
                        (
                            code.co_filename,
                            code.co_name,
                            frame.f_lineno if frame else None,
                        )
                    )
                awaitable = getattr(awaitable, "cr_await", None)
            return {
                "id": id(task),
                "done": task.done(),
                "cancelled": task.cancelled(),
                "await": chain,
            }

        import os
        import pytest_timeout

        payload = {
            "stage": stage,
            "mode": mode,
            "entered": entered.is_set(),
            "captures": [
                {
                    "purpose": item.purpose,
                    "started": item.started,
                    "work": task_state(item.work),
                    "operation": task_state(item.operation_task),
                    "applied_done": item.applied.done(),
                    "retired_done": item.retired.done(),
                }
                for item in captures
            ],
            "operations": [
                {
                    "apply_started": item.apply_started,
                    "native_entered": item.native_entered,
                    "retired": item.retired,
                    "physical": item.retirement.confirmed if item.retirement else None,
                    "commit": item.commit is not None,
                }
                for item in operations
            ],
            "callers": [task_state(item) for item in callers],
            "tail": task_state(controls._session_capture_tails.get(session.id)),
            "tasks": [
                task_state(item) for item in asyncio.all_tasks() if not item.done()
            ],
        }
        payload["debugger_active"] = pytest_timeout.is_debugging()
        payload["monitoring"] = [(i, sys.monitoring.get_tool(i)) for i in range(6)]
        print("TASK61 PREFILL", label, payload, file=sys.__stderr__, flush=True)
        if path := os.environ.get("TLDW_TASK61_DIAGNOSTIC_PATH"):
            with open(path, "a", encoding="utf-8") as stream:
                if first_error:
                    import traceback

                    traceback.print_exc(file=stream)
                print("TASK61 PREFILL", label, payload, file=stream, flush=True)

    try:
        with (
            _observe(
                type(screen)._sync_native_console_transcript.__code__,
                sys.monitoring.events.PY_START,
                transcript_entry,
            ),
            _observe(
                parent.run_parent_async_settings_action.__code__,
                sys.monitoring.events.PY_START,
                caller,
            ),
            _observe(
                PendingSettingsApply.apply.__code__,
                sys.monitoring.events.PY_START,
                operation_entry,
            ),
            _observe(
                parent.wait_readiness_predecessor.__code__,
                sys.monitoring.events.PY_RETURN,
                readiness_return,
            ),
            _observe(
                type(store)._write_session_pinned_prefill.__code__,
                sys.monitoring.events.PY_RETURN,
                writer_return,
            ),
        ):
            composer.load_draft("/prefill pin Voice:")
            screen.query_one("#console-send-message", Button).press()
            stage = "writer-entry"
            diagnostic("stage")
            await _entered(entered)
            assert len(captures) == len(operations) == 1
            capture, operation = captures[0], operations[0]
            assert capture.operation_task is not callers[0]
            assert session.settings.pinned_prefill == "Voice:"
            assert (
                operation.retirement is not None and not operation.retirement.confirmed
            )
            assert worker_leases(store.persistence.db)
            stage = "held-action"
            diagnostic("stage")
            if mode == "newer-draft":
                composer.move_cursor_end()
                composer.insert_text(" newer")
                newer = composer.draft_text()
            elif mode == "switch":
                successor = store.create_session(
                    title="Successor", settings=session.settings
                )
                assert store.active_session_id == successor.id
            elif mode == "observer-cancel":
                callers[0].cancel()
            stage = "post-retirement"
            diagnostic("stage")
            release.set()
            async with asyncio.timeout(10):
                if mode == "newer-draft":
                    await transcript_held.wait()
                    projection, original_key = transcript_sources[0]
                    assert projection is screen._console_readiness_config_projection
                    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
                        _same_readiness_key,
                    )

                    assert _same_readiness_key(original_key, projection._key())
                    while time.monotonic() - projection.at < projection.max_age:
                        await asyncio.sleep(0.01)
                    # Original autonomous stock scheduling, after genuine expiry;
                    # A still awaits its original transcript preparation lock.
                    projection.run(lambda: None, life=projection.max_age)
                    while not any(
                        item.projection is projection
                        for item in controls._settings_readiness_captures.values()
                    ):
                        await asyncio.sleep(0.01)
                    transcript_locks.pop().release()
                while not capture.retired.done():
                    await asyncio.sleep(0.01)
            assert capture.retired.result() and operation.retirement.confirmed
            assert operation.commit == (session, True)
            assert operation.retirement.lease not in worker_leases(store.persistence.db)
            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(
                    operation.retirement.connection
                )
            assert (
                json.loads(
                    store.persistence.db.get_conversation_by_id(conversation)[
                        "metadata"
                    ]
                )["pinned_response_prefill"]
                == "Voice:"
            )
            if mode == "switch":
                assert isinstance(
                    capture.applied.exception(), parent.ParentSettingsOriginChanged
                )
                assert not any(
                    "Prefill pinned" in str(message.content)
                    for message in store.messages_for_session(successor.id)
                )
            else:
                assert capture.applied.result() is None
                assert any(
                    "Prefill pinned" in str(message.content)
                    for message in store.messages_for_session(session.id)
                )
                assert composer.draft_text() == (newer if mode == "newer-draft" else "")
            if mode == "newer-draft":
                assert (
                    rewarm_dependencies
                ), "Original rewarm must exercise its actual A/W/B dependency"
                for dependency in rewarm_dependencies:
                    assert dependency.producer.done()
                    assert (
                        dependency.capture.retired.done()
                        and dependency.capture.retired.result()
                    )
                    assert dependency.capture not in controls._parent_settings_captures
            if mode == "observer-cancel":
                await asyncio.gather(callers[0], return_exceptions=True)
                assert workers[0]._task is callers[0]
                assert workers[0].state is WorkerState.CANCELLED
                assert isinstance(workers[0]._error, asyncio.CancelledError)
            assert app._exception is None
    except BaseException:
        import traceback

        traceback.print_exc(file=sys.__stderr__)
        diagnostic("first-body-error", first_error=True)
        raise
    finally:
        release.set()
        for lock in transcript_locks:
            lock.release()
        pending = {
            item for capture in captures for item in (capture.work, capture.applied)
        }
        pending.update(callers)
        if pending:
            done, unfinished = await asyncio.wait(pending, timeout=10)
            for item in done:
                if not item.cancelled():
                    item.exception()
            if unfinished:
                diagnostic("cleanup-still-pending")


async def test_prefill_refuses_copied_message_edges_before_mutation(
    selected_mounted_console, monkeypatch
):
    from copy import copy
    from types import FunctionType
    from tldw_chatbook.Chat.console_command_grammar import default_console_registry
    from tldw_chatbook.Chat import console_parent_settings as parent

    app, runtime, screen, pilot = selected_mounted_console
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    await _whole_core_settled(app, screen, controls, pilot)
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    original_settings, revision = session.settings, session.settings_revision
    owner = screen._message
    parse = default_console_registry().parse("/prefill pin refused")
    for edge in ("_chat_store_accessor", "_sync_native_console_chat_ui_fn", "receiver"):
        with monkeypatch.context() as patch:
            if edge == "receiver":
                patch.setattr(screen, "_message", copy(owner))
            else:
                original = getattr(owner, edge)
                cloned = FunctionType(
                    original.__code__,
                    original.__globals__,
                    closure=original.__closure__,
                )
                patch.setattr(owner, edge, cloned)
            with pytest.raises(
                parent.ParentSettingsSourceChanged, match="Prefill message"
            ):
                await screen._console_command_prefill(parse)
        assert (
            session.settings is original_settings
            and session.settings_revision == revision
        )
        assert not any(
            c.purpose == "prefill-command" for c in controls._parent_settings_captures
        )
    assert app._exception is None


async def test_original_identity_no_entry_cancel_balances_unissued_capture(
    selected_mounted_console,
):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_parent_settings as parent

    app, runtime, screen, pilot = selected_mounted_console
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    await _whole_core_settled(app, screen, controls, pilot)
    path, captures = config._get_effective_config_path(), []
    original = path.read_bytes()

    def cancel_before_entry(frame, values):
        capture = frame.f_locals.get("capture")
        if captures or capture is None or capture.controls is not controls:
            return
        assert capture.purpose == "identity-surfaces" and not capture.started
        assert capture.operation_task is None
        captures.append(capture)
        capture.work.cancel()

    try:
        path.write_bytes(original + b"\n[unterminated\n")
        await asyncio.to_thread(config.load_settings, True)
        with _observe(
            parent.defer_parent_identity.__code__,
            sys.monitoring.events.PY_RETURN,
            cancel_before_entry,
            values=True,
        ):
            screen._sync_console_identity_surfaces()
            await _whole_core_until(
                pilot, lambda: captures and captures[0].retired.done(), timeout=10
            )
        capture = captures[0]
        assert capture.applied.cancelled() and capture.retired.result()
        assert (
            not capture.started
            and capture.operation_task is None
            and capture.pending_operation is None
        )
        assert (
            capture.effect.completed.result()
            and capture.effect not in controls._effects
        )
        assert capture not in controls._parent_settings_captures
        _, request, token = capture.checked_config
        # This original cold-body attempt issued no presentation token.
        # No-entry capture/effect retirement does not imply token retirement.
        assert request is not None and token is None
        assert app._exception is None
    finally:
        path.write_bytes(original)
        await asyncio.to_thread(config.load_settings, True)
        for capture in captures:
            await asyncio.gather(capture.work, capture.applied, return_exceptions=True)


@pytest.mark.parametrize("action", ["help", "recovery"])
async def test_final_fixed_action_owns_original_cold_read(
    selected_mounted_console, action, request
):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_parent_settings as parent
    import pytest_timeout

    app, runtime, screen, pilot = selected_mounted_console
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    await _whole_core_settled(app, screen, controls, pilot)
    path, captures, due = config._get_effective_config_path(), [], []
    original = path.read_bytes()
    purpose = "workbench-help" if action == "help" else "recovery-prepare"
    work = None
    first_error = None
    refresh_observations = []
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
        _same_readiness_key,
    )

    def refresh_event(frame, values):
        projection = frame.f_locals.get("self")
        if projection.screen is not screen:
            return
        task = asyncio.current_task()
        refresh_observations.append(
            {
                "line": frame.f_lineno,
                "exception": repr(values[-1])
                if isinstance(values[-1], BaseException)
                else None,
                "task": id(task),
                "installed": frame.f_locals.get("installed"),
                "retry": frame.f_locals.get("retry"),
                "result": type(frame.f_locals.get("result")).__name__,
                "key_current": _same_readiness_key(
                    frame.f_locals.get("key"), projection._key()
                ),
                "waiter": repr(controls._settings_readiness_waits.get(task))[:500],
                "pending": projection.pending,
                "retry_delay": projection._retry_delay,
            }
        )

    def entered(frame):
        capture = frame.f_locals["capture"]
        if capture.controls is controls and capture.purpose == purpose:
            captures.append(capture)

    def settled(frame, values):
        source = frame.f_locals.get("source")
        if getattr(source, "purpose", None) == purpose:
            due.append(values[-1])

    try:
        path.write_bytes(original + b"\n[unterminated\n")
        installs = config._CONFIG_CACHE_INSTALLS
        await asyncio.to_thread(config.load_settings, True)
        assert config._settings_cache_hit(path) is None
        assert config._CONFIG_CACHE_INSTALLS == installs
        with (
            _observe(
                parent.run_parent_settings_capture.__code__,
                sys.monitoring.events.PY_START,
                entered,
            ),
            _observe(
                parent.settle_parent_action_defaults.__code__,
                sys.monitoring.events.PY_RETURN,
                settled,
                values=True,
            ),
            _observe(
                ConsoleReadinessConfigProjection._refresh.__code__,
                sys.monitoring.events.RAISE,
                refresh_event,
                values=True,
            ),
            _observe(
                ConsoleReadinessConfigProjection._refresh.__code__,
                sys.monitoring.events.PY_RETURN,
                refresh_event,
                values=True,
            ),
        ):
            assert not pytest_timeout.is_debugging()
            assert float(request.config.getini("timeout")) == 300
            request.node.user_properties.append(
                ("effective_observer_watchdog", "300/debugger=False")
            )
            work = asyncio.create_task(
                screen.action_show_workbench_help()
                if action == "help"
                else screen._open_console_provider_recovery()
            )
            async with asyncio.timeout(15):
                await asyncio.shield(work)
            assert len(captures) == 1 and due
            capture = captures[0]
            await _whole_core_until(pilot, lambda: capture.retired.done(), timeout=10)
            assert (
                capture.retired.result()
                and capture not in controls._parent_settings_captures
            )
            assert capture.effect.completed.result()
            if action == "help":
                assert app.screen.query_one("#workbench-help-panel")
                assert "Settings" in str(
                    app.screen.query_one("#workbench-help-body").renderable
                )
            else:
                assert (
                    app.screen is not screen
                ), "Original recovery did not publish its route"
                request.node.user_properties.append(
                    ("recovery_surface", type(app.screen).__name__)
                )
            assert app._exception is None
    except BaseException as error:
        first_error = error
        raise
    finally:
        import json

        request.node.user_properties.append(
            ("cold_refresh_phases", json.dumps(refresh_observations))
        )
        path.write_bytes(original)
        await asyncio.to_thread(config.load_settings, True)
        if work is not None:
            try:
                async with asyncio.timeout(10):
                    await asyncio.gather(work, return_exceptions=True)
            except BaseException:
                if first_error is None:
                    raise
        while app.screen is not screen and len(app.screen_stack) > 1:
            await app.pop_screen()


async def test_public_pressed_command_waits_for_settings_and_keeps_newer_draft(
    selected_mounted_console, request
):
    from textual.widgets import Button
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_settings_apply_store import _submission
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
    from tldw_chatbook.Chat.console_control_dispatch import (
        ApplySettings,
        ControlIdentity,
    )
    from tldw_chatbook.Chat.console_control_settings import settings_pending
    from tldw_chatbook.UI.Console_Modules import send_acknowledgement as sending
    import pytest_timeout

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    conversation = store.persistence.create_conversation(
        conversation_title="Held public press"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    await _whole_core_settled(app, screen, controls, pilot)
    composer = screen._console_composer_or_none()
    pressed = "/task61-unknown-command"
    composer.load_draft(pressed)
    await pilot.pause()
    submission = _submission(
        store, session.id, submission_id="public-press", model=session.settings.model
    )
    submission = replace(
        submission,
        draft=replace(
            submission.draft, settings=replace(session.settings, temperature=0.35)
        ),
    )
    entered, release = threading.Event(), threading.Event()
    scheduled, first_error = [], None

    def writer_return(frame):
        if (
            frame.f_locals.get("self") is store
            and frame.f_locals.get("session_id") == session.id
        ):
            assert threading.current_thread() is not threading.main_thread()
            entered.set()
            assert release.wait(10), "Original saved writer was not released"

    def schedule_return(frame):
        if frame.f_locals.get("screen") is screen:
            scheduled.append(frame.f_locals["request"])

    receipt = None
    try:
        with (
            _observe(
                store._write_session_settings_snapshot.__func__.__code__,
                sys.monitoring.events.PY_RETURN,
                writer_return,
            ),
            _observe(
                sending._schedule_visible_send_capture.__code__,
                sys.monitoring.events.PY_RETURN,
                schedule_return,
            ),
        ):
            assert not pytest_timeout.is_debugging()
            assert float(request.config.getini("timeout")) == 300
            request.node.user_properties.append(
                ("effective_observer_watchdog", "300/debugger=False")
            )
            receipt = controls.dispatch(
                ApplySettings(
                    ControlIdentity(
                        "mounted",
                        "local",
                        1,
                        "held-public-press",
                        session.id,
                        session.incarnation_id,
                    ),
                    submission,
                )
            )
            await _entered(entered)
            operation = settings_pending(store, session.id)
            assert operation is not None and operation.native_entered
            assert (
                worker_leases(store.persistence.db)
                and not operation.retirement.confirmed
            )
            screen.query_one("#console-send-message", Button).press()
            await _whole_core_until(
                pilot,
                lambda: any(
                    c.purpose == "visible-send"
                    for c in controls._parent_settings_captures
                ),
                timeout=10,
            )
            capture = next(
                c
                for c in controls._parent_settings_captures
                if c.purpose == "visible-send"
            )
            assert (
                capture.checked_config.stash.text == pressed
                and not capture.applied.done()
            )
            assert not scheduled and not receipt.applied.done()
            composer.move_cursor_end()
            composer.insert_text(" newer")
            newer = composer.draft_text()
            assert newer == pressed + " newer"
            authored = session.draft_revision
            release.set()
            async with asyncio.timeout(15):
                await receipt.applied
                await capture.applied
            await _whole_core_until(
                pilot, lambda: screen._console_unknown_send_armed == pressed, timeout=10
            )
            await _whole_core_until(
                pilot,
                lambda: not sending._flight(screen).tasks
                and screen._console_pending_send is None,
                timeout=10,
            )
            assert scheduled and scheduled[0].stash is capture.checked_config.stash
            assert scheduled[0].received_inputs is None
            assert (
                composer.draft_text() == newer
                and store.session_draft(session.id) == newer
            )
            assert session.draft_revision == authored
            assert capture.retired.done() and capture.retired.result()
            assert operation.retired and operation.retirement.confirmed
            assert not any(
                lease is operation.retirement.lease
                for lease in worker_leases(store.persistence.db)
            )
            assert app._exception is None
            request.node.user_properties.append(
                (
                    "press_route",
                    "captured unknown-command; newer draft; no fabricated resident input snapshot",
                )
            )
    except BaseException as error:
        first_error = error
        raise
    finally:
        release.set()
        if receipt is not None:
            try:
                async with asyncio.timeout(10):
                    await asyncio.gather(
                        receipt.work, receipt.applied, return_exceptions=True
                    )
                    await asyncio.gather(
                        *tuple(sending._flight(screen).tasks), return_exceptions=True
                    )
            except BaseException:
                if first_error is None:
                    raise


@pytest.mark.parametrize("mode", ["system", "improvement", "improvement-newer-draft"])
async def test_same_value_prompt_finishes_cold_posts_without_replaying_write(
    selected_mounted_console, mode, request
):
    from contextlib import ExitStack
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
    from tldw_chatbook import config
    from tldw_chatbook.Chat import console_parent_settings as parent
    from tldw_chatbook.Chat.console_control_settings import PendingSettingsApply
    from tldw_chatbook.UI.Console_Modules.session import follows_saved_defaults
    from tldw_chatbook.Widgets.Console.console_prompts_modal import (
        ConsolePromptsModal,
        ConsolePromptsResult,
    )
    import pytest_timeout

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    assert session.settings.system_prompt is None and not session.draft
    assert follows_saved_defaults(
        store, session, screen._session._console_new_chat_default_generation()
    )
    conversation = store.persistence.create_conversation(
        conversation_title="Same-value fixed prompt posts"
    )
    store.publish_first_persisted_conversation(session.id, conversation)
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    await _whole_core_settled(app, screen, controls, pilot)
    modal = None
    composer = screen._console_composer_or_none()
    if mode != "system":
        screen._prompts._open_console_prompts_modal()
        opening = next(
            c for c in controls._parent_settings_captures if c.purpose == "prompt-modal"
        )
        async with asyncio.timeout(10):
            await opening.applied
        await pilot.pause()
        modal = app.screen_stack[-1]
        assert isinstance(modal, ConsolePromptsModal)
        result = ConsolePromptsResult(
            kind="apply",
            composer_snapshot=composer.capture_draft_snapshot(),
            user_text=None,
            system_text="",
            apply_user=False,
            apply_system=True,
            captured_system_fingerprint=None,
        )
    entered, release = threading.Event(), threading.Event()
    operations, writes, posts, advances, preparations = [], [], [], [], []
    path = config._get_effective_config_path()
    good = path.read_bytes()
    work, first_error = None, None

    def operation_entry(frame):
        operation = frame.f_locals["self"]
        if operation.controls is controls and operation.mutation == (
            "system_prompt",
            "",
        ):
            operations.append(operation)

    def writer_return(frame):
        if (
            frame.f_locals.get("self") is store
            and frame.f_locals.get("session") is session
        ):
            writes.append(frame.f_locals["normalized"])
            assert threading.current_thread() is not threading.main_thread()
            if len(writes) == 1:
                entered.set()
                assert release.wait(
                    10
                ), "Original same-value System writer was not released"

    def post_entry(frame):
        if frame.f_locals["capture"].controls is controls:
            posts.append(frame.f_locals["operation"])

    def advanced(frame):
        capture = frame.f_locals["capture"]
        if capture.controls is controls:
            advances.append((capture, frame.f_locals["request"]))
            if mode == "improvement-newer-draft" and len(advances) == 1:
                composer.insert_text("newer post-write draft")

    def prepared(frame):
        capture = frame.f_locals["capture"]
        if capture.controls is controls and capture.purpose == "prompt-resolution":
            preparations.append(capture.checked_config[2])

    try:
        with ExitStack() as stack:
            for code, event, callback in (
                (
                    PendingSettingsApply.apply.__code__,
                    sys.monitoring.events.PY_START,
                    operation_entry,
                ),
                (
                    store._write_session_system_prompt.__func__.__code__,
                    sys.monitoring.events.PY_RETURN,
                    writer_return,
                ),
                (
                    parent._finish_parent_prompt_posts.__code__,
                    sys.monitoring.events.PY_START,
                    post_entry,
                ),
                (
                    parent._advance_parent_post_settings.__code__,
                    sys.monitoring.events.PY_RETURN,
                    advanced,
                ),
                (
                    parent._finish_parent_prompt_resolution.__code__,
                    sys.monitoring.events.PY_START,
                    prepared,
                ),
            ):
                stack.enter_context(_observe(code, event, callback))
            assert (
                not pytest_timeout.is_debugging()
                and float(request.config.getini("timeout")) == 300
            )
            request.node.user_properties.append(
                ("effective_observer_watchdog", "300/debugger=False")
            )
            work = asyncio.create_task(
                parent.apply_parent_system_prompt(screen, "")
                if mode == "system"
                else modal._apply_improvement_result(result, None)
            )
            async with asyncio.timeout(10):
                while not entered.is_set():
                    if work.done():
                        work.result()
                        pytest.fail("Original same-value System writer was not reached")
                    await asyncio.sleep(0.01)
            assert len(operations) == 1 and len(writes) == 1
            first = operations[0]
            capture = first.source
            assert first.native_entered and not first.retirement.confirmed
            assert worker_leases(store.persistence.db) and not capture.applied.done()
            assert follows_saved_defaults(
                store, session, screen._session._console_new_chat_default_generation()
            )
            installs = config._CONFIG_CACHE_INSTALLS
            path.write_bytes(good + b"\n[unterminated\n")
            await asyncio.to_thread(config.load_settings, True)
            assert (
                config.get_config_load_failure() is not None
                and config._CONFIG_CACHE_INSTALLS == installs
            )
            release.set()
            async with asyncio.timeout(15):
                outcome = await asyncio.shield(work)
                assert await capture.retired
            assert (
                first.retirement.confirmed
                and first.commit is not None
                and first.retired
            )
            assert first in posts and advances and advances[0][0] is capture
            assert advances[0][1].convergences
            assert len(writes) == 1 and len(operations) == 1
            assert (
                store.persistence.db.get_conversation_by_id(conversation)[
                    "system_prompt"
                ]
                is None
            )
            if mode == "system":
                assert outcome is True and not session.draft
            elif mode == "improvement-newer-draft":
                assert (
                    outcome.kind == "stale"
                    and "draft changed" in outcome.user_message.lower()
                )
                assert (
                    session.draft == composer.draft_text() == "newer post-write draft"
                )
            else:
                assert outcome.kind == "applied" and "prepare-apply" in preparations
                # Original persistence retry uses the live System, not the old
                # modal fingerprint. The invalid file is still genuinely cold.
                assert config._settings_cache_hit(path) is None
                async with asyncio.timeout(15):
                    retried = await modal._retry_improvement_persistence(result)
                assert retried.kind == "applied" and "prepare-retry" in preparations
                assert len(operations) == len(writes) == 2
                assert all(
                    op.retired and op.retirement.confirmed and op.commit is not None
                    for op in operations
                )
            assert (
                capture.effect.completed.result()
                and capture not in controls._parent_settings_captures
            )
            assert app._exception is None
            request.node.user_properties.append(
                (
                    "prompt_posts",
                    f"{mode}; original System writes={len(writes)}; due advances={len(advances)}",
                )
            )
    except BaseException as error:
        first_error = error
        raise
    finally:
        release.set()
        path.write_bytes(good)
        await asyncio.to_thread(config.load_settings, True)
        if work is not None:
            try:
                async with asyncio.timeout(10):
                    await asyncio.gather(work, return_exceptions=True)
            except BaseException:
                if first_error is None:
                    raise
        if modal is not None and app.screen is modal:
            await app.pop_screen()


async def test_selected_settings_open_keeps_original_context_facts_across_phases(
    selected_mounted_console, request
):
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Chat import console_parent_settings as parent
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )
    import pytest_timeout

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    session = store._sessions[store.active_session_id]
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    await _whole_core_settled(app, screen, controls, pilot)
    captures, current_facts, final_facts = [], [], []

    def phase_entry(frame):
        capture = frame.f_locals["capture"]
        if capture.controls is not controls or capture.purpose not in {
            "settings-open-prepare",
            "settings-context-current",
            "settings-display-prepare",
            "settings-modal-prepare",
        }:
            return
        captures.append(capture)
        args = capture.checked_config[0]
        if capture.purpose == "settings-context-current":
            current_facts.append(args)
        elif capture.purpose == "settings-modal-prepare":
            final_facts.append(args[1])

    work, first_error = None, None
    try:
        with _observe(
            parent.run_parent_settings_capture.__code__,
            sys.monitoring.events.PY_START,
            phase_entry,
        ):
            assert (
                not pytest_timeout.is_debugging()
                and float(request.config.getini("timeout")) == 300
            )
            request.node.user_properties.append(
                ("effective_observer_watchdog", "300/debugger=False")
            )
            work = asyncio.create_task(screen._open_console_settings())
            async with asyncio.timeout(15):
                opened = await asyncio.shield(work)
            assert opened is True and isinstance(app.screen, ConsoleSettingsModal)
            assert [c.purpose for c in captures].count("settings-open-prepare") == 1
            assert [c.purpose for c in captures].count("settings-display-prepare") == 1
            assert [c.purpose for c in captures].count("settings-modal-prepare") == 1
            assert len(current_facts) == 3 and len(final_facts) == 1
            facts = current_facts[0]
            assert (
                all(item is facts for item in current_facts) and final_facts[0] is facts
            )
            assert (
                facts[0] is controls._controller
                and facts[1] is store
                and facts[2] == session.id
            )
            initial = next(
                c for c in captures if c.purpose == "settings-open-prepare"
            ).applied.result()
            assert facts[3] is initial[4] and facts[5] == facts[4]._key(
                controls._controller, session.id
            )
            assert all(
                c.retired.done() and c.retired.result() and c.effect.completed.result()
                for c in captures
            )
            assert all(c not in controls._parent_settings_captures for c in captures)
            assert app._exception is None
            request.node.user_properties.append(
                (
                    "settings_phases",
                    "prepare; original current x3; display; same-facts modal preparation; actual modal push",
                )
            )
    except BaseException as error:
        first_error = error
        raise
    finally:
        if work is not None:
            try:
                async with asyncio.timeout(10):
                    await asyncio.gather(work, return_exceptions=True)
            except BaseException:
                if first_error is None:
                    raise
        if isinstance(app.screen, ConsoleSettingsModal):
            await app.pop_screen()


async def test_settings_final_guard_cannot_publish_after_queued_session_switch(
    selected_mounted_console, request
):
    from textual.app import App
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )
    import pytest_timeout

    app, runtime, screen, pilot = selected_mounted_console
    store = runtime.ensure_chat_store()
    original = store._sessions[store.active_session_id]
    successor = store.create_session(
        title="Settings publication successor",
        settings=original.settings,
        activate=False,
    )
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    await _whole_core_settled(app, screen, controls, pilot)
    events, captures = [], []
    switched = asyncio.Event()
    work, first_error = None, None

    def switch_after_guard():
        store.switch_session(successor.id)
        events.append(("switch", store.active_session_id))
        switched.set()

    def preparation_return(frame):
        if frame.f_locals.get("self") is not screen:
            return
        capture = next(
            c
            for c in controls._parent_settings_captures
            if c.purpose == "settings-modal-prepare"
            and c.operation_task is asyncio.current_task()
        )
        captures.append(capture)
        assert store.active_session_id == original.id
        events.append(("guard-return", original.id))
        asyncio.get_running_loop().call_soon(switch_after_guard)

    def push_entry(frame):
        modal = frame.f_locals.get("screen")
        if frame.f_locals.get("self") is app and isinstance(
            modal, ConsoleSettingsModal
        ):
            events.append(("push", store.active_session_id))

    try:
        with (
            _observe(
                screen._prepare_console_settings_modal.__func__.__code__,
                sys.monitoring.events.PY_RETURN,
                preparation_return,
            ),
            _observe(
                App.push_screen.__code__, sys.monitoring.events.PY_START, push_entry
            ),
        ):
            assert not pytest_timeout.is_debugging()
            assert float(request.config.getini("timeout")) == 300
            work = asyncio.create_task(screen._open_console_settings())
            async with asyncio.timeout(15):
                result = await asyncio.shield(work)
                await switched.wait()
            request.node.user_properties.append(("publication_events", repr(events)))
            request.node.user_properties.append(
                ("effective_observer_watchdog", "300/debugger=False")
            )
            assert captures and captures[0].retired.done()
            assert captures[0].effect.completed.result()
            assert result is True and [event for event, _ in events] == [
                "push",
                "guard-return",
                "switch",
            ], events
            assert all(
                target == original.id for event, target in events if event == "push"
            ), events
            if result:
                assert isinstance(app.screen, ConsoleSettingsModal)
                assert [event for event, _ in events].index("push") < [
                    event for event, _ in events
                ].index("switch")
            assert app._exception is None
    except BaseException as error:
        first_error = error
        raise
    finally:
        if work is not None:
            try:
                async with asyncio.timeout(10):
                    await asyncio.gather(work, return_exceptions=True)
            except BaseException:
                if first_error is None:
                    raise
        if isinstance(app.screen, ConsoleSettingsModal):
            await app.pop_screen()


async def test_settings_cancellation_before_push_result_unwinds_exact_modal(
    selected_mounted_console, request
):
    from textual.widget import AwaitMount
    from Tests.Chat.test_console_control_trace import _observe
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )
    import pytest_timeout

    app, runtime, screen, pilot = selected_mounted_console
    controls = runtime._execution_service.open_local_controls(
        profile_epoch="mounted", service_epoch="local"
    )
    await _whole_core_settled(app, screen, controls, pilot)
    records, captures, unwind, mount_joins = [], [], [], []
    transfer_commits = []
    work, first_error = None, None

    def phase_return(frame):
        if frame.f_locals.get("self") is not screen:
            return
        publication = frame.f_locals["publication"]
        assert isinstance(publication.modal, ConsoleSettingsModal)
        assert publication.modal in app.screen_stack
        capture = next(
            c
            for c in controls._parent_settings_captures
            if c.purpose == "settings-modal-prepare"
            and c.operation_task is asyncio.current_task()
        )
        assert not capture.applied.done() and not work.done()
        records.append(publication)
        captures.append(capture)
        work.cancel()

    def unwind_entry(frame):
        if frame.f_locals.get("self") is screen:
            assert frame.f_locals["modal"] is records[0].modal
            assert captures[0].retired.done() and captures[0].retired.result()
            assert captures[0].effect.completed.result()
            assert len(mount_joins) == 1 and mount_joins[0].done()
            assert not mount_joins[0].cancelled() and mount_joins[0].exception() is None
            assert records[0].modal._mounted_event.is_set()
            unwind.append(frame.f_locals["modal"])

    def mount_entry(frame):
        if not records or frame.f_locals.get("self") is not records[0].mount:
            return
        mount_work = work.get_coro().cr_frame.f_locals.get("mount_work")
        if mount_work is asyncio.current_task():
            assert captures[0].retired.done() and captures[0].retired.result()
            mount_joins.append(mount_work)
            work.cancel()  # a repeated observer cancellation cannot release the join

    def transfer_committed():
        transfer_commits.append(True)
        return True

    try:
        with (
            _observe(
                screen._prepare_console_settings_modal.__func__.__code__,
                sys.monitoring.events.PY_RETURN,
                phase_return,
            ),
            _observe(
                screen._unwind_failed_console_settings_modal.__func__.__code__,
                sys.monitoring.events.PY_START,
                unwind_entry,
            ),
            _observe(
                AwaitMount.__await__.__code__,
                sys.monitoring.events.PY_START,
                mount_entry,
            ),
        ):
            assert not pytest_timeout.is_debugging()
            assert float(request.config.getini("timeout")) == 300
            work = asyncio.create_task(
                screen._open_console_settings(_on_transfer_committed=transfer_committed)
            )
            async with asyncio.timeout(15):
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.shield(work)
            assert len(records) == 1 and unwind == [records[0].modal]
            assert records[0].modal not in app.screen_stack
            assert not transfer_commits
            assert captures[0] not in controls._parent_settings_captures
            assert app._exception is None
            request.node.user_properties.append(
                (
                    "publication_cancellation",
                    "pushed; result undelivered; capture retired; exact modal unwound; no transfer commit",
                )
            )
            request.node.user_properties.append(
                ("effective_observer_watchdog", "300/debugger=False")
            )
    except BaseException as error:
        first_error = error
        raise
    finally:
        if work is not None:
            try:
                async with asyncio.timeout(10):
                    await asyncio.gather(work, return_exceptions=True)
            except BaseException:
                if first_error is None:
                    raise
        if records and app.screen is records[0].modal:
            await app.pop_screen()
