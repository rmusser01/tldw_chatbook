"""Original mounted lifecycle joins the image owner's real payload producer."""

import asyncio
import threading
import time
from io import BytesIO

import pytest

from Tests.private_profile import private_profile_test

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.timeout(300)]


@pytest.mark.parametrize("pending_read", [False, True])
def test_empty_image_selection_needs_no_reader_and_preserves_owned_work(
    tmp_path, pending_read
):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules.image import ConsoleImageController

    def unexpected_worker(coro, **kwargs):
        coro.close()
        pytest.fail("An empty transcript cannot need a recovered-media read")

    retained = object()
    tasks = {retained} if pending_read else set()
    rows, selection = {"prior": object()}, object()
    owner = SimpleNamespace(
        app_instance=SimpleNamespace(
            recovered_media_root=tmp_path, recovered_media_profile="empty"
        ),
        _screen=SimpleNamespace(run_worker=unexpected_worker),
        _recovered_image_tasks=tasks,
        _recovered_image_paused=False,
        _recovered_image_rows=rows,
        _recovered_image_lookup=selection,
    )
    assert ConsoleImageController._recovered_console_image_specs(
        owner, [], object(), object()
    ) == ({}, set())
    assert owner._recovered_image_tasks == ({retained} if pending_read else set())
    assert owner._recovered_image_rows is rows
    assert owner._recovered_image_lookup is selection


@pytest.mark.asyncio
@pytest.mark.parametrize("detached", [False, True], ids=["current", "detached"])
@private_profile_test
async def test_app_shutdown_joins_recovered_image_native_read(
    request, tmp_path, monkeypatch, detached
):
    from PIL import Image
    from textual.screen import Screen

    from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs
    from Tests.UI.background_signals import await_background_task
    from Tests.UI.test_console_button_routing import _mounted_console
    from Tests.UI.test_original_character_teardown_lifetime import _stock_private_app
    from tldw_chatbook.Backup_Recovery import recovered_media
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )

    app = _stock_private_app()
    media = recovered_media.RecoveredMedia(tmp_path / "recovered")
    source = tmp_path / "source.png"
    data = BytesIO()
    Image.new("RGB", (2, 2), "red").save(data, format="PNG")
    source.write_bytes(data.getvalue())
    asset = media.retain(
        source, profile="p", message="m", slug="shutdown", media_type="image/png"
    )
    payload_path = media.resolve(asset)[1]
    app.recovered_media_root = media.root
    app.recovered_media_profile = "p"
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original_read = recovered_media._read
    reads = []
    image_task = shutdown = second_waiter = switch = None
    facts = {}

    def held_read(path):
        if path != payload_path:
            return original_read(path)
        assert threading.current_thread() is not threading.main_thread()
        reads.append(path)
        entered.set()
        assert release.wait(10), "Image native producer was never released"
        payload = original_read(path)
        assert payload == data.getvalue()
        returned.set()  # Original private stream's context has retired.
        return payload

    try:
        async with app.run_test(size=(160, 44)) as pilot:
            await await_background_task(
                app._initial_screen_setup_task, what="original initial Console setup"
            )
            console = await _mounted_console(app, pilot, "#console-native-composer")
            runtime = app.console_runtime
            assert runtime.view is console and console.app_instance is app
            image = console._image
            # Retire only prior image selection work; no all-worker idle oracle.
            image._recovered_images_close_admission()
            assert await image._recovered_images_drain(time.monotonic() + 10)
            monkeypatch.setattr(recovered_media, "_read", held_read)
            message = ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="Recovered image",
                status="complete",
            )
            message.persisted_message_id = "m"
            image._recovered_images_resume()
            image._build_console_image_specs([message])
            assert len(image._recovered_image_tasks) == 1
            image_task = next(iter(image._recovered_image_tasks))
            try:
                async with asyncio.timeout(10):
                    while not entered.is_set():
                        await asyncio.sleep(0.01)
                assert not returned.is_set() and not image_task.done()
                if detached:
                    # Actual Textual removal invokes original on_unmount. The
                    # old screen leaves the stack synchronously; uninstalling
                    # it then permits the normal replacement to remove it.
                    switch = app.switch_screen(Screen())
                    if app.is_screen_installed(console):
                        app.uninstall_screen(console)
                    async with asyncio.timeout(10):
                        while runtime.view is console:
                            await asyncio.sleep(0.01)
                    assert runtime.view is None
                    facts["unmount_waited"] = not switch.is_done
                shutdown = asyncio.create_task(app._shutdown_console_runtime())
                async with asyncio.timeout(10):
                    while app._console_runtime_shutdown_task is None:
                        await asyncio.sleep(0)
                terminal = app._console_runtime_shutdown_task
                await asyncio.sleep(0.03)
                facts["sealed"] = image._recovered_image_paused
                facts["terminal_waited"] = not terminal.done()
                # Concurrent callers join the same original app-owned task.
                second_waiter = asyncio.create_task(app._shutdown_console_runtime())
                shutdown.cancel()
                await asyncio.sleep(0)
                shutdown.cancel()
                image_task.cancel()
                await asyncio.sleep(0)
                image_task.cancel()
                await asyncio.sleep(0.03)
                facts["native_waited"] = not image_task.done() and not returned.is_set()
                facts["cancelled_caller_waited"] = not shutdown.done()
                facts["same_terminal"] = app._console_runtime_shutdown_task is terminal
            finally:
                release.set()
                if image_task is not None:
                    await asyncio.gather(image_task, return_exceptions=True)
                if shutdown is not None:
                    await asyncio.gather(shutdown, return_exceptions=True)
                if second_waiter is not None:
                    await second_waiter
                if switch is not None:
                    await switch
            assert returned.is_set() and reads == [payload_path]
            assert image_task.done() and not image._recovered_image_tasks
            assert terminal.done() and terminal.result() is None
            await app._shutdown_console_runtime()
            assert app._console_runtime_shutdown_task is terminal
        assert app._exception is None
    finally:
        release.set()
        drain_active_service_patches()
        drain_created_dirs()
    # Original-source RED occurs only after the real read and owner cleanup.
    assert all(facts.values()), facts


@pytest.mark.asyncio
@private_profile_test
async def test_control_loss_keeps_execution_lease_through_native_app_drains(
    request, tmp_path, monkeypatch
):
    """Control loss fences sends, but only native app retirement releases ownership."""
    import json
    from pathlib import Path

    from PIL import Image

    from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs
    from Tests.UI.background_signals import await_background_task
    from Tests.UI.test_console_button_routing import _mounted_console
    from Tests.UI.test_original_character_teardown_lifetime import _stock_private_app
    from tldw_chatbook import app_served
    from tldw_chatbook.Backup_Recovery import recovered_media
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
        ConsoleSubmissionOrigin,
    )
    from tldw_chatbook.Chat.console_execution_owner import (
        ConsoleExecutionBusy,
        ConsoleExecutionOwner,
    )
    from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
    from tldw_chatbook.Workspaces.project_index import runtime as index_runtime
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy

    async def until(predicate):
        async with asyncio.timeout(10):
            while not predicate():
                await asyncio.sleep(0.01)

    def competitor_refused(run_path):
        try:
            competitor = ConsoleExecutionOwner.acquire(run_path)
        except ConsoleExecutionBusy:
            return True
        competitor.close()
        return False

    app = _stock_private_app()
    media = recovered_media.RecoveredMedia(tmp_path / "recovered")
    source = tmp_path / "source.png"
    data = BytesIO()
    Image.new("RGB", (2, 2), "red").save(data, format="PNG")
    source.write_bytes(data.getvalue())
    asset = media.retain(
        source, profile="p", message="m", slug="control-loss", media_type="image/png"
    )
    payload_path = media.resolve(asset)[1]
    app.recovered_media_root, app.recovered_media_profile = media.root, "p"
    image_entered, image_release, image_returned = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    view_entered, view_release, view_returned = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    project_entered, project_release = threading.Event(), threading.Event()
    reads = set()
    facts = {}
    image_task = worker = loss = None
    original_read = recovered_media._read

    def held_image_read(path):
        if path != payload_path:
            return original_read(path)
        image_entered.set()
        assert image_release.wait(10), "original image read was not released"
        payload = original_read(path)
        assert payload == data.getvalue()
        image_returned.set()
        return payload

    project_root = tmp_path / "project"
    project_root.mkdir()
    view_source = project_root / "physical.txt"
    view_source.write_text("original physical payload")

    def held_view_read():
        view_entered.set()
        assert view_release.wait(10), "original view read was not released"
        payload = view_source.read_text()
        assert payload == "original physical payload"
        view_returned.set()
        return payload

    observation = None
    try:
        async with app.run_test(size=(160, 44)) as pilot:
            try:
                await await_background_task(
                    app._initial_screen_setup_task,
                    what="original initial Console setup",
                )
                console = await _mounted_console(app, pilot, "#console-native-composer")
                runtime = app.console_runtime
                controller = runtime.chat_controller
                assert controller is runtime._chat_controller
                await asyncio.to_thread(runtime.ensure_activity_receipt_service)
                run_path = Path(app.chachanotes_db.db_path).parent / "agent_runs.db"
                owner = runtime._execution_owner
                owner.require_current(run_path)
                assert owner in ConsoleExecutionOwner._live
                assert competitor_refused(run_path)

                registry = app.workspace_registry_service
                registry.create_workspace(
                    workspace_id="control-loss", name="Control loss"
                )
                binding = registry.add_folder_binding("control-loss", project_root)
                service = await asyncio.to_thread(
                    index_runtime.get_project_index_service, app
                )
                key = await asyncio.to_thread(
                    index_runtime.project_index_key,
                    app,
                    "control-loss",
                    binding.binding_id,
                    project_root,
                )
                original_scan = service._scanner.scan

                def held_scan(authority, scope, job, frontier, emit, stop, **kwargs):
                    def publish(unit):
                        accepted = emit(unit)
                        if accepted and unit.captures:
                            project_entered.set()
                            assert project_release.wait(30)
                        return accepted

                    return original_scan(
                        authority, scope, job, frontier, publish, stop, **kwargs
                    )

                monkeypatch.setattr(service._scanner, "scan", held_scan)
                off = await asyncio.to_thread(service.status, key)
                enabled = await asyncio.to_thread(
                    service.enable,
                    key,
                    ProjectIndexPolicy((), ()),
                    expected_scope=off.scope,
                )
                job = await asyncio.to_thread(
                    service.refresh,
                    key,
                    expected_scope=enabled.scope,
                    expected_job_revision=enabled.job_revision,
                )
                project_run = service._runs[job]
                await until(project_entered.is_set)

                image = console._image
                image._recovered_images_close_admission()
                assert await image._recovered_images_drain(time.monotonic() + 10)
                monkeypatch.setattr(recovered_media, "_read", held_image_read)
                message = ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="Recovered image",
                    status="complete",
                )
                message.persisted_message_id = "m"
                image._recovered_images_resume()
                image._build_console_image_specs([message])
                (image_task,) = image._recovered_image_tasks
                worker = app.run_worker(
                    run_preparation_read(
                        held_view_read,
                        creator=app,
                        session_id=None,
                        reads=reads,
                        require_current=lambda: None,
                    ),
                    group="console-agent-history",
                    exit_on_error=False,
                )
                await until(lambda: image_entered.is_set() and view_entered.is_set())
                loss = asyncio.create_task(app_served._retire_after_control_loss(app))
                await until(lambda: controller._disposed)
                await until(lambda: app._exit)

                async def observe_retirement():
                    try:
                        await until(
                            lambda: getattr(app, "_project_index_closed", False)
                        )
                        for origin in (
                            ConsoleSubmissionOrigin.MANUAL,
                            ConsoleSubmissionOrigin.AGENT_WAKE,
                            ConsoleSubmissionOrigin.AGENT_CHAT_START,
                        ):
                            result = await controller.submit_draft(
                                "must never dispatch", origin=origin
                            )
                            facts[origin.value + "_shutdown_refusal"] = (
                                not result.accepted
                                and result.visible_copy == "Console is shutting down."
                            )
                        facts.update(
                            same_original_view=runtime.view is console,
                            native_lease_retained=competitor_refused(run_path),
                            project_still_physical=not project_run.native_retired,
                            image_still_physical=not image_returned.is_set()
                            and not image_task.done(),
                            view_still_physical=not view_returned.is_set()
                            and bool(reads),
                        )
                        assert all(facts.values()), facts
                        project_release.set()
                        await until(
                            lambda: (
                                project_run.native_retired
                                and app._console_runtime_shutdown_task is not None
                            )
                        )
                        assert image._recovered_image_paused
                        assert competitor_refused(run_path)
                        image_task.cancel()
                        await asyncio.sleep(0)
                        image_task.cancel()
                        worker.cancel()
                        worker.cancel()
                        await asyncio.sleep(0.03)
                        assert not image_task.done() and not image_returned.is_set()
                        assert not view_returned.is_set() and competitor_refused(
                            run_path
                        )
                        view_release.set()
                        await until(view_returned.is_set)
                        assert not image_returned.is_set() and competitor_refused(
                            run_path
                        )
                        image_release.set()
                        await until(image_returned.is_set)
                        await loss
                    finally:
                        project_release.set()
                        view_release.set()
                        image_release.set()

                observation = asyncio.create_task(observe_retirement())
            finally:
                if observation is None:
                    project_release.set()
                    view_release.set()
                    image_release.set()
        assert observation is not None
        await observation
        assert app._exception is None
        assert project_run.native_retired and service._scanner.retired
        assert not reads and not image._recovered_image_tasks
        assert owner not in ConsoleExecutionOwner._live
        replacement = ConsoleExecutionOwner.acquire(run_path)
        replacement.close()
        facts["native_retirement_then_reacquire"] = True
    finally:
        project_release.set()
        view_release.set()
        image_release.set()
        if observation is not None:
            await asyncio.gather(observation, return_exceptions=True)
        if loss is not None:
            await asyncio.gather(loss, return_exceptions=True)
        (tmp_path / "control-loss-retirement.json").write_text(
            json.dumps(facts, indent=2)
        )
        drain_active_service_patches()
        drain_created_dirs()


@pytest.mark.asyncio
@private_profile_test
async def test_control_loss_still_exits_after_native_controller_fence_error(
    request, monkeypatch
):
    from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs
    from Tests.UI.background_signals import await_background_task
    from Tests.UI.test_console_button_routing import _mounted_console
    from Tests.UI.test_original_character_teardown_lifetime import _stock_private_app
    from tldw_chatbook import app_served

    app = _stock_private_app()
    try:
        async with app.run_test(size=(160, 44)) as pilot:
            await await_background_task(
                app._initial_screen_setup_task, what="original initial Console setup"
            )
            console = await _mounted_console(app, pilot, "#console-native-composer")
            controller = app.console_runtime.chat_controller
            original = controller.begin_shutdown

            def fail_after_native_fence():
                original()
                raise RuntimeError("injected after native admission fence")

            monkeypatch.setattr(controller, "begin_shutdown", fail_after_native_fence)
            try:
                with pytest.raises(
                    RuntimeError, match="injected after native admission fence"
                ):
                    await app_served._retire_after_control_loss(app)
                assert controller._disposed and app._exit and app._shutting_down
                assert console._console_chat_tearing_down
            finally:
                monkeypatch.setattr(controller, "begin_shutdown", original)
        assert app._exception is None
    finally:
        drain_active_service_patches()
        drain_created_dirs()
