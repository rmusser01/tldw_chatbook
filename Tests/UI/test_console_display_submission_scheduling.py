"""Stock display readers queue before submission while foreground work proceeds."""

import asyncio
import sys
import threading
from types import CodeType

import pytest

from Tests.Backup_Recovery.test_console_main_display_batch import _configured_display
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root  # noqa: PLC0414
from Tests.Chat.test_console_hook_preparation_reads import _until
from Tests.UI.test_console_workspace_display_batch import (
    _rename_workspace,
    _request_registry,
    workspace_display as workspace_display,
)  # noqa: PLC0414
from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
from tldw_chatbook.Chat.console_runtime import CONSOLE_RUNTIME_ATTR, ConsoleRuntime
from tldw_chatbook.UI.Console_Modules import config_sync
from tldw_chatbook.UI.Console_Modules.character_context import _CharacterRefreshReaders
from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
    ConsoleReadinessConfigRead,
    _handoff_projection_is_stock,
)
from tldw_chatbook.UI.Console_Modules.workspace import (
    ConsoleWorkspaceController,
    _workspace_availability_readers_current,
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.usefixtures("local_root"),
]


def _callback_code(method, name):
    return next(
        code
        for code in method.__code__.co_consts
        if isinstance(code, CodeType) and code.co_name == name
    )


async def test_stock_display_callbacks_wait_before_submission_without_blocking_foreground(
    tmp_path,
    workspace_display,  # noqa: F811 - imported pytest fixture
    record_property,
):
    database, screen, tasks = _configured_display(tmp_path)
    case = workspace_display
    screen.app_instance.workspace_registry_service = case.registry
    case.controller.app_instance = screen.app_instance
    # Both real display owners resolve the same existing app-owned runtime.
    runtime = ConsoleRuntime(screen.app_instance, canvas_enabled_reader=lambda: False)
    setattr(screen.app_instance, CONSOLE_RUNTIME_ATTR, runtime)
    projection = screen._console_readiness_config_projection
    assert _handoff_projection_is_stock(projection)
    assert _workspace_availability_readers_current(case.controller, case.registry)
    codes = {
        projection.read_current.__code__: "readiness",
        _callback_code(config_sync._MainDbDisplayRead.run, "read_sections"): "main",
        _callback_code(
            ConsoleWorkspaceController._read_owned_workspace_availability,
            "read_captured_availability",
        ): "workspace",
    }
    entered, release = threading.Event(), threading.Event()
    physical = []
    midpoints = []
    midpoint_code = _callback_code(_CharacterRefreshReaders.ambient_on_loop, "validate")
    previous, previous_main = threading.getprofile(), sys.getprofile()

    def observe(frame, event, _result):
        if event == "call" and frame.f_code is midpoint_code:
            slot = getattr(runtime, "_display_submission_slot", None)
            midpoints.append(
                (
                    threading.current_thread(),
                    slot.locked() if slot is not None else False,
                )
            )
        if event == "call" and frame.f_code in codes:
            physical.append(codes[frame.f_code])
            entered.set()
            assert release.wait(30), "display callback was not released"

    main_task = None
    threading.setprofile_all_threads(observe)
    try:
        assert not projection.run(lambda: None)
        await _until(entered.is_set)
        main_task = asyncio.create_task(config_sync.prepare_main_db_display(screen))
        case.controller._request_workspace_files_availability_refresh(("alpha",))
        await _until(lambda: len(runtime._preparation_reads) == 3)
        assert len(projection._reads) == len(case.controller._preparation_reads) == 1
        assert (
            await run_preparation_read(
                lambda: "foreground completed",
                creator=runtime,
                session_id=None,
                reads=runtime._preparation_reads,
                require_current=lambda: None,
            )
            == "foreground completed"
        )
        submitted = [
            read.callback.__name__
            for read in runtime._preparation_reads
            if read._producer is not None
        ]
        held_physical = list(physical)
        record_property("held_display_submissions", sorted(submitted))
        record_property("held_display_callbacks", held_physical)
        release.set()
        await asyncio.gather(
            main_task, *tasks, *(worker for worker, _ in case.screen.workers)
        )
        assert set(physical) == {"readiness", "main", "workspace"}
        assert projection.value is not None
        assert screen._character_context.state.scope_fingerprint is not None
        assert case.controller._workspace_files_availability_by_id == {"alpha": True}
        assert not runtime._preparation_reads
        assert submitted == [
            "read_current"
        ], "competing stock display reads reached the executor"
        assert held_physical == ["readiness"]
        assert midpoints and all(
            thread is threading.current_thread() and held for thread, held in midpoints
        )
    finally:
        release.set()
        await asyncio.gather(
            *([main_task] if main_task is not None else []),
            *tasks,
            *(worker for worker, _ in case.screen.workers),
            return_exceptions=True,
        )
        threading.setprofile_all_threads(previous)
        sys.setprofile(previous_main)
        database.close()


async def test_custom_readiness_and_registry_display_keep_independent_submission(
    tmp_path,
    workspace_display,  # noqa: F811 - imported pytest fixture
):
    database, screen, tasks = _configured_display(tmp_path)
    case = workspace_display
    runtime = ConsoleRuntime(screen.app_instance, canvas_enabled_reader=lambda: False)
    setattr(screen.app_instance, CONSOLE_RUNTIME_ATTR, runtime)
    setattr(case.controller.app_instance, CONSOLE_RUNTIME_ATTR, runtime)
    projection = screen._console_readiness_config_projection
    key = projection._key()
    projection.read_current = lambda: ConsoleReadinessConfigRead(
        key[0], {"custom": True}, key[0]
    )
    original = case.controller._read_workspace_files_availability
    case.controller._read_workspace_files_availability = (
        lambda *args, **kwargs: original(*args, **kwargs)
    )
    assert not _handoff_projection_is_stock(projection)
    assert not _workspace_availability_readers_current(case.controller, case.registry)
    _rename_workspace(case)
    await runtime._display_submission_slot.acquire()
    try:
        assert not projection.run(lambda: None)
        _request_registry(case)
        async with asyncio.timeout(15):
            await asyncio.gather(*tasks, *(worker for worker, _ in case.screen.workers))
        assert projection.value == {"custom": True}
        assert "Renamed" in {row.name for row in case.view.list_workspaces()}
        assert (
            not runtime._preparation_reads and runtime._display_submission_slot.locked()
        )
    finally:
        runtime._display_submission_slot.release()
        await asyncio.gather(
            *tasks,
            *(worker for worker, _ in case.screen.workers),
            return_exceptions=True,
        )
        database.close()


async def test_custom_runtime_view_properties_are_not_read_by_readiness(tmp_path):
    database, screen, tasks = _configured_display(tmp_path)
    touched = []

    class CustomRuntime:
        _disposed = False

        @property
        def view(self):
            touched.append("view")
            raise AssertionError("custom runtime view was inspected")

        @property
        def _attached_generation(self):
            touched.append("generation")
            raise AssertionError("custom runtime generation was inspected")

    setattr(screen.app_instance, CONSOLE_RUNTIME_ATTR, CustomRuntime())
    projection = screen._console_readiness_config_projection
    try:
        assert not projection.run(lambda: None)
        await asyncio.gather(*tasks)
        assert (
            not touched
        ), "custom runtime properties were inspected before the original read"
        assert projection.value is not None
        assert not projection._reads
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.parametrize(
    "change",
    ["readiness_reader", "runtime_disposed", "main_closing", "workspace_lifecycle"],
)
async def test_queued_stock_display_refuses_changed_owner_before_executor_submission(
    tmp_path,
    workspace_display,  # noqa: F811 - imported pytest fixture
    change,
):
    database, screen, tasks = _configured_display(tmp_path)
    case = workspace_display
    runtime = ConsoleRuntime(screen.app_instance, canvas_enabled_reader=lambda: False)
    setattr(screen.app_instance, CONSOLE_RUNTIME_ATTR, runtime)
    setattr(case.controller.app_instance, CONSOLE_RUNTIME_ATTR, runtime)
    projection = screen._console_readiness_config_projection
    main_task = None
    slot = runtime._display_submission_slot
    await slot.acquire()
    try:
        if change == "workspace_lifecycle":
            case.controller._request_workspace_files_availability_refresh(("alpha",))
        elif change == "main_closing":
            main_task = asyncio.create_task(config_sync.prepare_main_db_display(screen))
        else:
            assert not projection.run(lambda: None)
        await _until(lambda: bool(runtime._preparation_reads))
        (read,) = runtime._preparation_reads
        assert read._producer is None
        if change == "readiness_reader":
            projection.read_current = lambda: pytest.fail(
                "queued replacement was invoked"
            )
        elif change == "runtime_disposed":
            runtime._disposed = True
        elif change == "main_closing":
            screen._closing = True
        else:
            case.controller._screen_lifecycle_token_accessor = lambda: object()
        slot.release()
        await asyncio.gather(
            *([main_task] if main_task is not None else []),
            *tasks,
            *(worker for worker, _ in case.screen.workers),
            return_exceptions=True,
        )
        assert read._producer is None and read.retired.done()
        assert not runtime._preparation_reads and not slot.locked()
        assert projection.value is None
        assert case.controller._workspace_files_availability_by_id == {}
    finally:
        if slot.locked():
            slot.release()
        await asyncio.gather(
            *([main_task] if main_task is not None else []),
            *tasks,
            *(worker for worker, _ in case.screen.workers),
            return_exceptions=True,
        )
        database.close()
