"""FULL waits for its existing checked config owner before rail presentation."""

import asyncio
import sys
import threading
from contextlib import ExitStack, contextmanager, nullcontext
from types import MethodType, SimpleNamespace

import pytest

from Tests.UI.test_console_checked_display_scope import _actual_calls, _screen, _warm
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.config_participants import checked_config_identity
from tldw_chatbook.UI.Console_Modules.attach_visit import ConsoleAttachVisit
from tldw_chatbook.UI.Screens import chat_screen

pytestmark = pytest.mark.bootstrap_profile


def _full_screen(screen, controller, monkeypatch):
    """Keep original FULL and rail/config entry; omit unrelated UI projections."""
    effects = []

    def no_op(*_args, **_kwargs):
        return None

    async def async_no_op(*_args, **_kwargs):
        return None

    runtime = SimpleNamespace(view=screen, _attached_generation=1)
    screen._console_runtime = lambda: runtime
    screen._console_chat_controller = controller
    screen._current_console_run_status_value = MethodType(
        chat_screen.ChatScreen._current_console_run_status_value, screen
    )
    screen._console_attach_visit_generation = 1
    screen._current_console_attach_visit = lambda: ConsoleAttachVisit(
        runtime, screen, 1, screen._console_attach_visit_generation, True
    )
    screen._console_sync_in_progress = False
    screen._console_sync_requested = False
    screen._console_terminal_return_reading = None
    screen.app_instance._ui_ready = False
    screen._console_attach_sync_complete = False
    screen._console_attach_reconciled = False
    screen._console_attach_reconcile_running = False
    screen._console_control_bar_replay_whole_sync = False
    screen._console_terminal_return_reading = None  # The class default.
    screen._message = SimpleNamespace(reconcile_console_speech_context=no_op)
    screen._image = SimpleNamespace(
        _h3_image_edit_registry=lambda: SimpleNamespace(drop_session=no_op),
        _reconcile_h3_image_edit_completions=no_op,
    )
    screen._session = SimpleNamespace(
        _sync_console_session_draft=no_op,
        _maybe_refresh_stale_default_console_settings=no_op,
        schedule_manual_read_acknowledgement=no_op,
    )
    screen._retrieval = SimpleNamespace(
        _warm_console_effective_scope_cache_if_stale=async_no_op,
        _refresh_active_dictionaries_summary_if_scope_changed=async_no_op,
        _refresh_active_world_books_summary_if_scope_changed=async_no_op,
    )
    screen._character = SimpleNamespace(
        _refresh_active_character_avatar_if_scope_changed=async_no_op,
    )
    screen._character_context = SimpleNamespace(refresh_if_scope_changed=async_no_op)
    screen._workspace = SimpleNamespace(tick_workspace_build_scope=nullcontext)
    for name in (
        "_record_ui_worker_started",
        "_record_ui_worker_finished",
        "_sync_console_chat_core_state",
        "_sync_console_temporary_chip",
        "_current_console_rail_state",
        "_sync_console_settings_summary",
        "_sync_console_settings_recovery_surfaces",
        "_sync_console_live_work_readiness_rows",
        "_sync_console_mode_bar",
        "_dispatch_active_console_roleplay_refresh",
        "_sync_console_workspace_context",
        "_sync_console_rail_visibility_if_changed",
        "_dispatch_console_rail_preference_prune",
        "_start_console_attach_reconciliation",
        "_sync_console_temporary_chip",  # TASK-37840; TASK-38973 drift
    ):
        setattr(screen, name, no_op)
    screen._sync_console_native_session_tabs = async_no_op
    screen._sync_native_console_transcript = async_no_op
    # Since ab43a0303b a started app's FULL also syncs archive recovery.
    archive = SimpleNamespace(sync=async_no_op)
    screen._archive_recovery_view = lambda: archive
    screen._sync_console_control_bar_under_config = lambda _rail: effects.append("rail")
    screen._request_console_control_bar_sync = lambda **_: effects.append("retry")
    screen.call_after_refresh = lambda _: effects.append("attach")
    screen._sync_console_rail_and_controls = MethodType(
        chat_screen.ChatScreen._sync_console_rail_and_controls, screen
    )
    monkeypatch.setattr(
        chat_screen.project_instruction_ui,
        "sync_project_instruction_status_for_screen",
        no_op,
    )
    return effects


@pytest.mark.asyncio
async def test_expired_original_full_awaits_config_without_main_native_entry(
    tmp_path, monkeypatch, record_property
):
    database, _store, controller, screen, tasks = _screen(tmp_path)
    try:
        _full_screen(screen, controller, monkeypatch)
        projection = await _warm(screen, tasks)
        # Let the real checked display expire; changing its proof clock would
        # select a different refusal route from ordinary resume.
        await asyncio.sleep(projection.max_age + 0.02)
        with _actual_calls() as calls:
            completed = await chat_screen.ChatScreen._sync_native_console_chat_ui(
                screen
            )
        record_property("main_config_entries", calls["main_scopes"])
        record_property("main_native_opens", calls["main_opens"])
        assert completed is True
        assert screen._console_attach_sync_complete
        assert calls["main_scopes"] == 0, calls
        assert calls["main_opens"] == 0, calls
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@contextmanager
def _held_original_config_read(*, projection=None, snapshot=None):
    """Hold the real read after its second source check, retaining actual leases."""
    entered, release = threading.Event(), threading.Event()
    observed = []
    returns = 0
    main = threading.get_ident()
    old_main, old_thread = sys.getprofile(), threading.getprofile()

    def observe(frame, event, _argument):
        nonlocal returns
        if (
            threading.get_ident() != main
            and event == "return"
            and frame.f_code is checked_config_identity.__code__
        ):
            if projection is None:
                returns += 1
                if returns != 2:
                    return
            else:
                caller = frame.f_back
                request = caller.f_locals.get("request")
                if (
                    entered.is_set()
                    or _argument is None  # A profiler return also reports unwind.
                    or caller.f_code is not projection.read_current.__code__
                    or caller.f_locals.get("projection") is not projection
                    or request is None
                    or request[2][-1] is not snapshot
                    or "before" not in caller.f_locals
                ):
                    return
            observed.append(raw._states[frame.f_locals["active"]])
            entered.set()
            assert release.wait(10), "test did not release original config read"

    threading.setprofile_all_threads(observe)
    try:
        yield entered, release, observed
    finally:
        release.set()
        threading.setprofile_all_threads(old_thread)
        sys.setprofile(old_main)


async def _wait_for_read(entered, full):
    """Observe producer errors instead of waiting only for a never-set event."""
    waiter = asyncio.create_task(asyncio.to_thread(entered.wait, 5))
    try:
        done, _pending = await asyncio.wait(
            (waiter, full), timeout=6, return_when=asyncio.FIRST_COMPLETED
        )
        if full in done:
            await full
            pytest.fail("FULL completed before its checked config read retired")
        assert waiter in done and await waiter, "original config read did not start"
    finally:
        await waiter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["none", "session", "source", "visit", "maintenance"]
)
async def test_full_waits_responsively_and_rechecks_held_original_config(
    tmp_path, monkeypatch, change
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    full = None
    try:
        effects = _full_screen(screen, controller, monkeypatch)
        store.create_session(session_id="other")
        store.active_session_id = "session-1"
        projection = await _warm(screen, tasks)
        await asyncio.sleep(projection.max_age + 0.02)
        with _held_original_config_read() as (entered, release, observed):
            full = asyncio.create_task(
                chat_screen.ChatScreen._sync_native_console_chat_ui(screen)
            )
            try:
                await _wait_for_read(entered, full)
                state = observed[0]
                assert state.active and projection.pending and projection._reads
                # These turns happen before release: the original native call
                # is blocked, while UI callbacks still get scheduled.
                heartbeat = []
                asyncio.get_running_loop().call_soon(heartbeat.append, True)
                await asyncio.sleep(0)
                assert heartbeat == [True]
                assert not full.done() and "rail" not in effects
                assert not screen._console_attach_sync_complete
                if change == "session":
                    store.active_session_id = "other"
                elif change == "source":
                    identity = config.current_config_identity()
                    monkeypatch.setattr(
                        config,
                        "current_config_identity",
                        lambda: (identity[0] + 1, identity[1]),
                    )
                elif change == "visit":
                    screen._console_attach_visit_generation += 1
                elif change == "maintenance":
                    screen._console_sync_maintenance_paused = True
                release.set()
                completed = await full
                if change == "none":
                    assert completed is True and "rail" in effects
                    assert screen._console_attach_sync_complete
                else:
                    assert completed is False and "rail" not in effects
                    assert not screen._console_attach_sync_complete
                assert not state.active
                with storage._lock:
                    assert all(
                        lease not in storage._live_leases for lease in state.leases
                    )
            finally:
                release.set()
                await asyncio.gather(full, *tasks, return_exceptions=True)
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_cancelled_full_keeps_original_config_read_owned_until_retirement(
    tmp_path, monkeypatch
):
    database, _store, controller, screen, tasks = _screen(tmp_path)
    try:
        effects = _full_screen(screen, controller, monkeypatch)
        projection = await _warm(screen, tasks)
        await asyncio.sleep(projection.max_age + 0.02)
        with _held_original_config_read() as (entered, release, observed):
            full = asyncio.create_task(
                chat_screen.ChatScreen._sync_native_console_chat_ui(screen)
            )
            try:
                await _wait_for_read(entered, full)
                state = observed[0]
                full.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await full
                assert projection.pending and projection._reads and state.active
                assert not screen._console_attach_sync_complete
                assert "rail" not in effects
                with storage._lock:
                    assert all(lease in storage._live_leases for lease in state.leases)
                release.set()
                await asyncio.gather(*tasks)
                assert not projection.pending and not projection._reads
                assert not state.active
                with storage._lock:
                    assert all(
                        lease not in storage._live_leases for lease in state.leases
                    )
            finally:
                release.set()
                await asyncio.gather(full, *tasks, return_exceptions=True)
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_post_attach_full_keeps_screen_messages_moving_during_native_read(
    monkeypatch, record_property
):
    from Tests.UI.test_console_attach_completion import _mounted_console
    from Tests.UI.test_console_session_tab_close import ProductionConsoleHarness
    from Tests.UI.test_console_tab_publication_maintenance import _awaiting, _loop_wait
    from Tests.UI.test_destination_shells import _build_test_app
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
    )

    app = _build_test_app()
    app.app_config = config.load_settings()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    host = ProductionConsoleHarness(app)
    original = ConsoleReadinessConfigProjection.warm
    gate = None
    held = False
    held_reads = ExitStack()

    async def cold_post_attach(projection):
        nonlocal gate, held
        screen = projection.screen
        if screen._console_attach_view_started and not held:
            held = True
            # A replacement App config snapshot needs its own checked display read.
            app.app_config = dict(app.app_config)
            # The first warm may defer before entry; keep observing its replay.
            gate = held_reads.enter_context(
                _held_original_config_read(
                    projection=projection, snapshot=app.app_config
                )
            )
        return await original(projection)

    monkeypatch.setattr(ConsoleReadinessConfigProjection, "warm", cold_post_attach)
    with held_reads:
        async with _mounted_console(host) as pilot:
            try:
                assert await _loop_wait(
                    lambda: gate is not None and gate[0].is_set()
                ), {
                    "held": held,
                    "gate": gate,
                    "attached": getattr(
                        host.screen, "_console_attach_reconciled", None
                    ),
                    "sections": tuple(app.app_config),
                }
                screen = host.screen
                state = gate[2][0]
                assert state.active, (
                    "the original checked read must still own its leases"
                )
                record_property("held_screen_await_chain", _awaiting(screen._task))
                delivered = []
                screen.call_later(delivered.append, True)
                assert await _loop_wait(lambda: delivered == [True], timeout=1), (
                    "post-attach FULL parked the screen message pump on a native read",
                    _awaiting(screen._task),
                )
                assert state.active, "screen responsiveness was measured after release"
            finally:
                if gate is not None:
                    gate[1].set()
                    assert await _loop_wait(
                        lambda: (
                            all(not state.active for state in gate[2])
                            and not host.screen._console_sync_in_progress
                        )
                    )
                    await pilot.pause()


@pytest.mark.asyncio
async def test_full_preparation_defers_for_busy_file_lock(tmp_path):
    """A refused stock read releases REBUILD and can prepare after FILE retires."""
    from tldw_chatbook.UI.Console_Modules.config_sync import (
        prepare_console_presentation_sync,
    )

    database, _store, _controller, screen, tasks = _screen(tmp_path)
    held, release = threading.Event(), threading.Event()
    holder_expired = threading.Event()
    preparation = None

    def hold_file():
        with config._config_file_lock():
            held.set()
            if not release.wait(5):
                holder_expired.set()

    holder = threading.Thread(target=hold_file)
    try:
        projection = await _warm(screen, tasks)
        proof = projection._display_proof
        assert proof is not None
        await asyncio.sleep(projection.max_age + 0.02)
        holder.start()
        try:
            assert await asyncio.to_thread(held.wait, 3)
            preparation = asyncio.create_task(prepare_console_presentation_sync(screen))
            done, _ = await asyncio.wait((preparation,), timeout=2)
            assert preparation in done, "display preparation waited for FILE"
            assert await preparation is False
            assert not holder_expired.is_set()
            assert projection._display_proof is proof
            assert not projection.pending and not projection._reads
            rebuild = config._settings_rebuild_lock()
            assert rebuild.acquire(blocking=False), "refused FILE leaked REBUILD"
            rebuild.release()
        finally:
            release.set()
            await asyncio.to_thread(holder.join, 3)
        assert not holder.is_alive()
        assert await prepare_console_presentation_sync(screen)
        assert projection._display_proof is not proof
    finally:
        if preparation is not None:
            await asyncio.gather(preparation, return_exceptions=True)
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["open", "move", "cancel"])
async def test_explicit_settings_keeps_busy_config_custody(
    tmp_path, monkeypatch, outcome
):
    from Tests.UI.test_console_shared_context_policy import _screen as settings_screen
    from tldw_chatbook.Backup_Recovery.config_participants import operation
    from tldw_chatbook.UI.Console_Modules.config_sync import (
        prepare_console_presentation_sync,
        rewarm_console_presentation,
    )
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
        prepare_console_settings_context,
    )

    database, store, controller, screen, tasks = settings_screen(tmp_path)
    screen._ensure_console_chat_controller = lambda: controller
    screen._ensure_console_chat_store = lambda: store
    store.create_session(session_id="other")
    store.active_session_id = "session-1"
    projection = ConsoleReadinessConfigProjection.for_screen(screen)
    held, release, waiting, expired = (threading.Event() for _ in range(4))
    admitted, retire = threading.Event(), threading.Event()
    active_states = []
    native_thread = None
    rebuild = config._settings_rebuild_lock()

    def hold():
        with rebuild:
            held.set()
            if not release.wait(10):
                expired.set()

    holder = threading.Thread(target=hold)
    warm = projection.warm

    async def warm_with_holder(**kwargs):
        if holder.ident is None:
            # The explicit route has already checked the live thinking policy.
            holder.start()
            assert await asyncio.to_thread(held.wait, 3)
        return await warm(**kwargs)

    monkeypatch.setattr(projection, "warm", warm_with_holder)
    native_code = operation.__wrapped__.__code__
    reader_code = projection.read_current.__code__
    old_main, old_thread = sys.getprofile(), threading.getprofile()

    def observe(frame, event, result):
        nonlocal native_thread
        if not held.is_set():
            return
        if event == "call" and frame.f_code is native_code and not waiting.is_set():
            caller = frame.f_back
            while caller is not None:
                if (
                    caller.f_code is reader_code
                    and caller.f_locals.get("projection") is projection
                ):
                    native_thread = threading.get_ident()
                    waiting.set()
                    return
                caller = caller.f_back
        elif (
            outcome == "open"
            and event == "return"
            and frame.f_code is checked_config_identity.__code__
            and threading.get_ident() == native_thread
            and result is not None
            and release.is_set()
            and not admitted.is_set()
        ):
            caller = frame.f_back
            if caller.f_code is reader_code and "before" in caller.f_locals:
                active_states.append(raw._states[frame.f_locals["active"]])
                admitted.set()
                assert retire.wait(10)

    async def background_full():
        ready = await prepare_console_presentation_sync(screen)
        await rewarm_console_presentation(screen)
        return ready

    opener = waiter = background = moved = admitted_full = None
    threading.setprofile_all_threads(observe)
    try:
        origin = store.capture_console_settings_origin("session-1")
        opener = asyncio.create_task(
            prepare_console_settings_context(
                screen, controller, store, "session-1", origin, max_age=1
            )
        )
        waiter = asyncio.create_task(asyncio.to_thread(waiting.wait, 5))
        done, _ = await asyncio.wait(
            (opener, waiter), timeout=6, return_when=asyncio.FIRST_COMPLETED
        )
        assert opener not in done, (
            "explicit Settings lost its open on a busy config lock"
        )
        assert waiter in done and await waiter, (
            "the original native operation never tried to enter"
        )
        assert projection.pending and projection._reads
        background = asyncio.create_task(background_full())
        done, _ = await asyncio.wait((background,), timeout=2)
        assert background in done, "passive FULL waited for an unadmitted Settings read"
        assert await background is False
        assert not expired.is_set() and not opener.done()
        if outcome == "move":
            store.active_session_id = "other"
            new_origin = store.capture_console_settings_origin("other")

            def new_owner_is_current():
                return (
                    store.active_session_id == "other"
                    and store.capture_console_settings_origin("other") == new_origin
                )

            moved = asyncio.create_task(
                warm(retry_busy_if_current=new_owner_is_current)
            )
            done, _ = await asyncio.wait((moved,), timeout=2)
            assert moved in done, "new Settings owner joined the old unadmitted read"
            assert await moved is False
        elif outcome == "cancel":
            opener.cancel()
            with pytest.raises(asyncio.CancelledError):
                await opener
            assert projection.pending and projection._reads
        release.set()
        if outcome == "open":
            assert await asyncio.to_thread(admitted.wait, 5)
            state = active_states[0]
            assert state.active and projection.pending and projection._reads
            admitted_full = asyncio.create_task(
                prepare_console_presentation_sync(screen)
            )
            done, _ = await asyncio.wait((admitted_full,), timeout=0.05)
            assert admitted_full not in done, "FULL skipped the original admitted read"
            assert state.active
            retire.set()
            prepared = await opener
            assert prepared is not None and prepared[1]()
            assert await admitted_full is True
            assert not state.active
        elif outcome == "move":
            assert await opener is None
        await asyncio.gather(*tasks)
        assert not projection.pending and not projection._reads
    finally:
        release.set()
        retire.set()
        await asyncio.gather(
            *(
                task
                for task in (opener, waiter, background, moved, admitted_full, *tasks)
                if task is not None
            ),
            return_exceptions=True,
        )
        if holder.ident is not None:
            await asyncio.to_thread(holder.join, 3)
        threading.setprofile_all_threads(old_thread)
        sys.setprofile(old_main)
        database.close()
    assert not holder.is_alive()



@pytest.mark.asyncio
async def test_explicit_settings_refreshes_after_a_retired_old_config_read(tmp_path):
    database, store, _controller, screen, tasks = _screen(tmp_path)
    opener = None
    try:
        projection = await _warm(screen, tasks)
        old_mapping = screen.app_instance.app_config
        with _held_original_config_read(
            projection=projection, snapshot=old_mapping
        ) as (entered, release, observed):
            projection.run(lambda: None, life=0)
            assert await asyncio.to_thread(entered.wait, 5)
            old = projection._admission
            state = observed[0]
            assert old is not None and state.active and projection._reads
            session = store.active_session_id
            origin = store.capture_console_settings_origin(session)
            current_mapping = screen.app_instance.app_config = dict(old_mapping)
            assert projection._key()[-1] is not old.key[-1]

            def owns_settings():
                return (
                    screen.app_instance.app_config is current_mapping
                    and store.capture_console_settings_origin(session) == origin
                )

            opener = asyncio.create_task(
                projection.warm(retry_busy_if_current=owns_settings)
            )
            await asyncio.sleep(0)
            assert not opener.done() and projection._admission is old
            release.set()
            assert await opener
            assert projection.key[-1] is current_mapping
            assert not state.active and not projection.pending and not projection._reads
            with storage._lock:
                assert all(lease not in storage._live_leases for lease in state.leases)
    finally:
        if opener is not None:
            await asyncio.gather(opener, return_exceptions=True)
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()
