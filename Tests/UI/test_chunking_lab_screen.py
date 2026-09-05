"""Library tool navigation and real durable authoring through the mounted screen."""

import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from textual import events
from textual.widgets import TextArea

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route


async def settle_lab(app, screen, pilot):
    """Await only the Lab-owned workers; the full app also owns long-lived workers."""
    from textual.worker import WorkerCancelled

    for _ in range(3):
        workers = [
            worker
            for worker in app.workers
            if worker.node is screen or screen in worker.node.ancestors
        ]
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                *(worker.wait() for worker in workers), return_exceptions=True
            ),
            10,
        )
        assert all(
            not isinstance(outcome, BaseException)
            or isinstance(outcome, WorkerCancelled)
            for outcome in outcomes
        ), outcomes
        await pilot.pause()


@contextmanager
def hold_final_edit_render(screen):
    """Pause only the consumer's first final render, not transition computation."""
    entered, release = asyncio.Event(), asyncio.Event()
    refresh = screen._refresh_session

    async def paused(*, edit_complete=False):
        if edit_complete and not entered.is_set():
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
        await refresh(edit_complete=edit_complete)

    with patch.object(screen, "_refresh_session", paused):
        try:
            yield entered, release
        finally:
            release.set()


@pytest.mark.asyncio
async def test_final_render_edit_is_consumed_without_another_keystroke(lab_app):
    from tldw_chatbook.Chunking import lab_state

    async with lab_app.run_test(size=(80, 24)) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        await settle_lab(lab_app, screen, pilot)
        with hold_final_edit_render(screen) as (entered, release):
            screen.queue_edit(
                lambda session: lab_state.replace_sample(
                    session, "first", {"kind": "paste"}
                )
            )
            await asyncio.wait_for(entered.wait(), 3)
            session = screen.coordinator.session
            assert session.samples[session.view["sample_hash"]]["text"] == "first"
            screen.queue_edit(
                lambda session: lab_state.replace_sample(
                    session, "last", {"kind": "paste"}
                )
            )
            release.set()
            # No drain or new keystroke rescues the consumer's final-render tail.
            await asyncio.wait_for(asyncio.shield(screen._edit_task), 5)
            session = screen.coordinator.session
            assert session.samples[session.view["sample_hash"]]["text"] == "last"
            await screen.drain_edits()
            assert not screen._edits
            assert screen._edit_task.done()


@pytest.mark.asyncio
async def test_real_navigation_routes_lab_and_checkpoints_return_to_opener(lab_app):
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    async with lab_app.run_test(size=(80, 24)):
        await lab_app.push_screen(
            resolve_screen_route("home").load_screen_class()(lab_app)
        )
        # This harness mounts its initial content explicitly, so also establish
        # the real router's post-startup admission boundary (not splash timing).
        lab_app._initial_screen_pushed = True
        await lab_app.handle_screen_navigation(
            NavigateToScreen("chunking_lab", {"return_route": "home"})
        )
        screen = lab_app.screen
        assert screen.screen_name == "chunking_lab"
        await screen.wait_until_ready()
        assert screen.return_route == "home"
        owner = screen.coordinator
        await lab_app.handle_screen_navigation(NavigateToScreen(screen.return_route))
        assert lab_app.screen.screen_name == "home"
        assert await lab_app.get_chunking_lab_coordinator() is owner


@pytest.mark.asyncio
async def test_invalid_structural_json_remains_editable_and_visible(lab_app):
    from tldw_chatbook.Chunking import lab_state

    async with lab_app.run_test(size=(80, 24)):
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        screen.queue_edit(
            lambda session: lab_state.edit_json(
                session, screen._b_id(session), '{"chunking":{"method":[]}}'
            )
        )
        await screen.drain_edits()
        await screen._refresh_session()
        assert (
            screen.query_one("#lab-json", TextArea).text == '{"chunking":{"method":[]}}'
        )
        assert "Ready for local preview" not in str(
            screen.query_one("#lab-validation").content
        )


@pytest.mark.asyncio
async def test_narrow_results_paging_and_inspector_are_keyboard_reachable(lab_app):
    from textual.widgets import DataTable

    from tldw_chatbook.Chunking import lab_state

    async with lab_app.run_test(size=(80, 24)) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        await settle_lab(lab_app, screen, pilot)
        screen.queue_edit(
            lambda session: lab_state.replace_sample(
                session,
                " ".join(f"word{index}" for index in range(150)),
                {"kind": "paste"},
            )
        )
        screen.queue_edit(
            lambda session: lab_state.edit_control(
                session, screen._b_id(session), "chunking.config.max_size", "1"
            )
        )
        screen.queue_edit(
            lambda session: lab_state.edit_control(
                session, screen._b_id(session), "chunking.config.overlap", "0"
            )
        )
        await screen.run_candidates()
        await settle_lab(lab_app, screen, pilot)
        assert await pilot.click("#lab-show-results")
        await screen.drain_edits()
        await settle_lab(lab_app, screen, pilot)
        screen.query_one("#last-b").focus()
        await pilot.pause()
        await asyncio.wait_for(pilot.wait_for_scheduled_animations(), 3)
        assert screen.query_one("#last-b").region in screen.content_region
        await pilot.press("enter")
        await settle_lab(lab_app, screen, pilot)
        table = screen.query_one("#chunks-b", DataTable)
        table.focus()
        await pilot.press("ctrl+end", "enter")
        await settle_lab(lab_app, screen, pilot)
        inspector = screen.query_one("#chunk-inspector", TextArea)
        inspector.focus()
        await pilot.pause()
        await asyncio.wait_for(pilot.wait_for_scheduled_animations(), 3)
        assert lab_app.focused is inspector
        assert inspector.region in screen.content_region, [
            (
                identity,
                screen.query_one(identity).region,
                screen.query_one(identity).virtual_size,
                screen.query_one(identity).scroll_offset,
            )
            for identity in ("#lab-work", "#lab-results-scroll", "#lab-results")
        ]
        assert screen.query_one("#mapping-status").region in screen.content_region
        assert inspector.text == "word149"


@pytest.mark.asyncio
async def test_template_export_retains_tags_and_recovery_transfer(lab_app, tmp_path):
    route = resolve_screen_route("chunking_lab")
    assert route is not None
    async with lab_app.run_test(size=(120, 40)) as pilot:
        await lab_app.push_screen(route.load_screen_class()(lab_app))
        screen = lab_app.screen
        await screen.wait_until_ready()
        assert hasattr(screen, "file_operation")
        from tldw_chatbook.Chunking import lab_state

        screen.queue_edit(
            lambda session: lab_state.edit_record_fields(
                session,
                screen._b_id(session),
                {"name": "Exact", "tags": ["one", "two"]},
            )
        )
        await screen.drain_edits()
        selected = str(tmp_path / "template.json")
        await screen.file_operation(
            "export-template", {"path": selected, "overwrite": False}
        )
        import json

        payload = json.loads((tmp_path / "template.json").read_text())
        assert payload["tags"] == ["one", "two"]
        with pytest.raises(ValueError, match="already exists"):
            await screen.file_operation("export-template", {"path": selected})
        await screen.file_operation(
            "export-template", {"path": selected, "overwrite": True}
        )
        (tmp_path / "template.json").chmod(0o644)
        original_mode = (tmp_path / "template.json").stat().st_mode
        await screen.file_operation("import-template", {"path": selected})
        assert (tmp_path / "template.json").stat().st_mode == original_mode
        draft = screen.coordinator.session.candidates[screen._b_id()]["draft"]
        assert draft["record_fields"]["tags"] == ["one", "two"]
        assert draft["expected_record"] is None
        recovery = str(tmp_path / "recovery.json")
        await screen.file_operation(
            "export-recovery", {"path": recovery, "overwrite": False}
        )
        clearing = asyncio.create_task(screen._menu_action("clear"))
        await pilot.pause()
        assert await pilot.click("#dialog-accept")
        await clearing
        from Tests.UI.test_chunking_lab_recovery_flow import wait_for_dialog

        restoring = asyncio.create_task(
            screen.file_operation("restore", {"path": recovery})
        )
        await wait_for_dialog(lab_app, pilot)
        assert await pilot.click("#dialog-accept")
        await restoring
        assert (
            screen.coordinator.session.candidates[screen._b_id()]["draft"][
                "record_fields"
            ]["name"]
            == "Exact"
        )
        await screen._menu_action("undo-restore")
        assert (
            screen.coordinator.session.candidates[screen._b_id()]["draft"][
                "record_fields"
            ]["name"]
            == ""
        )


def test_file_excerpt_is_explicit_utf8_and_never_silently_truncated(tmp_path):
    from tldw_chatbook.UI.Chunking_Lab_Modules import sample_region

    assert hasattr(sample_region, "read_sample_excerpt")
    selected = tmp_path / "large.txt"
    selected.write_text("é" * (1024 * 1024 + 5), encoding="utf-8")
    with pytest.raises(ValueError, match="excerpt"):
        sample_region.read_sample_file(str(selected))
    text, source = sample_region.read_sample_excerpt(str(selected), 2, 7)
    assert text == "é" * 5
    assert source["start"] == 2 and source["end"] == 7
    selected.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        sample_region.read_sample_file(str(selected))


@pytest.fixture
def lab_app(tmp_path, monkeypatch):
    monkeypatch.setattr("tldw_chatbook.config.get_user_data_dir", lambda: tmp_path)
    app = _build_test_app()
    # This fixture mounts its own initial Lab screen during each test.
    app._initial_screen_pushed = True
    return app


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Requires POSIX FIFO support")
@pytest.mark.asyncio
async def test_template_import_refuses_fifo_without_changing_session(
    lab_app, tmp_path, monkeypatch
):
    selected = tmp_path / "template.json"
    os.mkfifo(selected)
    original_open = Path.open

    def guard_blocking_open(path, *args, **kwargs):
        # Fail before the old blocking open: a regression must not hang teardown.
        if path == selected:
            pytest.fail("Template import attempted a blocking FIFO open")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_blocking_open)
    async with lab_app.run_test(size=(80, 24)) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        await settle_lab(lab_app, screen, pilot)
        await screen.drain_edits()
        before = screen.coordinator.session
        with pytest.raises(ValueError, match="regular"):
            await screen.file_operation("import-template", {"path": str(selected)})
        assert screen.coordinator.session == before


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Requires POSIX FIFO support")
@pytest.mark.asyncio
async def test_template_import_refuses_file_replaced_by_fifo_at_open(
    lab_app, tmp_path, monkeypatch
):
    selected = tmp_path / "template.json"
    selected.write_text('{"chunking":{"method":"words"}}')
    original_open = os.open
    replaced = False

    def replace_at_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if Path(path) == selected:
            selected.unlink()
            os.mkfifo(selected)
            replaced = True
            # Bound the failure even if the nonblocking guard regresses.
            assert flags & os.O_NONBLOCK, "Template FIFO open could block"
        return original_open(path, flags, *args, **kwargs)

    async with lab_app.run_test(size=(80, 24)) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        await settle_lab(lab_app, screen, pilot)
        await screen.drain_edits()
        before = screen.coordinator.session
        monkeypatch.setattr(os, "open", replace_at_open)
        with pytest.raises(ValueError, match="regular"):
            await screen.file_operation("import-template", {"path": str(selected)})
        assert replaced
        assert screen.coordinator.session == before


@pytest.mark.asyncio
async def test_manually_mounted_lab_owns_deferred_initial_screen(lab_app):
    async with lab_app.run_test():
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        await lab_app._push_initial_screen()
        assert lab_app.screen is screen
        assert lab_app._initial_screen_pushed is True


@pytest.mark.asyncio
async def test_slow_tags_keep_unfinished_separator_after_render_reopen_and_failed_save(
    lab_app, tmp_path
):
    from textual.widgets import Input

    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    lab_app.media_db = MediaDatabase(str(tmp_path / "tags.sqlite3"), client_id="tags")
    async with lab_app.run_test(size=(80, 24)) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        await settle_lab(lab_app, screen, pilot)
        assert await pilot.click("#lab-show-configure")
        tags = screen.query_one("#lab-tags", Input)
        tags.focus()
        await pilot.press(*"alpha,")
        await pilot.pause()
        await screen.drain_edits()
        await settle_lab(lab_app, screen, pilot)
        assert tags.value == "alpha,"
        assert await screen.confirm_navigation()
        owner = screen.coordinator
        await lab_app.pop_screen()
        await owner.close()
        lab_app._chunking_lab_coordinator = None
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        await settle_lab(lab_app, screen, pilot)
        tags = screen.query_one("#lab-tags", Input)
        assert tags.value == "alpha,"
        tags.focus()
        await pilot.press("end", *"beta")
        await pilot.pause()
        await screen.drain_edits()
        with pytest.raises(Exception, match="name cannot be empty"):
            await screen.save_candidate("B")
        await settle_lab(lab_app, screen, pilot)
        assert tags.value == "alpha,beta"
        screen._queue_editor_edit("record", "name", "Separated")
        screen._queue_editor_edit("record", "description", "Two tags")
        record = await screen.save_candidate("B")
        assert record["tags"] == ["alpha", "beta"]
        await settle_lab(lab_app, screen, pilot)
        assert tags.value == "alpha,beta"


@pytest.mark.asyncio
async def test_builtin_dialog_defaults_to_copy_and_preserves_builtin(lab_app, tmp_path):
    from textual.widgets import Checkbox, Input

    from Tests.UI.test_chunking_lab_recovery_flow import wait_for_dialog
    from tldw_chatbook.Chunking.chunking_interop_library import ChunkingInteropService
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    lab_app.media_db = MediaDatabase(
        str(tmp_path / "builtins.sqlite3"), client_id="builtin"
    )
    service = ChunkingInteropService(lab_app.media_db)
    identity = service.create_template(
        "Protected",
        "Builtin description",
        {"chunking": {"method": "words"}},
        is_builtin=True,
    )
    original = service.get_template_by_id(identity)
    async with lab_app.run_test(size=(80, 24)) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()

        async def select_builtin(_):
            return original

        with patch.object(screen, "_dialog", select_builtin):
            await screen._menu_action("catalog")
        saving = asyncio.create_task(screen._save("B"))
        try:
            dialog = await wait_for_dialog(lab_app, pilot)
            assert dialog.query_one("#dialog-new", Checkbox).value
            dialog.query_one("#dialog-name", Input).value = "Custom copy"
            accept = dialog.query_one("#dialog-accept")
            accept.focus()
            await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            assert accept.region in dialog.content_region
            await pilot.press("enter")
            await asyncio.wait_for(saving, 5)
        finally:
            if not saving.done():
                saving.cancel()
            await asyncio.gather(saving, return_exceptions=True)
        assert service.get_template_by_id(identity) == original
        copy = service.get_template_by_name("Custom copy")
        assert copy and not copy["is_builtin"]


@pytest.mark.asyncio
async def test_unsupported_authored_template_exports_without_run_or_save_admission(
    lab_app, tmp_path
):
    from tldw_chatbook.Chunking import lab_state

    body = {"legacy_operation": {"removed": True}, "metadata": {"author": "preserve"}}
    async with lab_app.run_test(size=(80, 24)):
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        screen.queue_edit(
            lambda session: lab_state.replace_template(
                session,
                screen._b_id(session),
                body,
                record_fields={
                    "name": "Unsupported",
                    "description": "Exportable",
                    "tags": ["kept"],
                },
                expected_record=None,
            )
        )
        await screen.drain_edits()
        target = tmp_path / "preserved.json"
        await screen.file_operation("export-template", {"path": str(target)})
        exported = json.loads(target.read_text())
        assert exported["template_json"] == body
        assert exported["tags"] == ["kept"]
        with pytest.raises(ValueError):
            await screen.save_candidate("B")
        with pytest.raises(ValueError):
            await screen.run_candidates()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,labels",
    [
        ({"kind": "file", "name": "chosen.txt"}, ("chosen.txt", "Full text")),
        (
            {"kind": "file_excerpt", "name": "excerpt.txt", "start": 12, "end": 30},
            ("excerpt.txt", "Excerpt", "12", "30"),
        ),
        (
            {
                "kind": "local_media_excerpt",
                "local_media_id": 42,
                "start": 2,
                "end": 15,
            },
            ("42", "Excerpt", "2", "15"),
        ),
    ],
)
async def test_sample_source_and_extent_are_visible_after_reopen(
    lab_app, source, labels
):
    from tldw_chatbook.Chunking import lab_state

    async with lab_app.run_test(size=(80, 24)) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        screen.queue_edit(
            lambda session: lab_state.replace_sample(session, "copied text", source)
        )
        await screen.drain_edits()
        assert await screen.confirm_navigation()
        owner = screen.coordinator
        await lab_app.pop_screen()
        await owner.close()
        lab_app._chunking_lab_coordinator = None
        reopened = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(reopened)
        await reopened.wait_until_ready()
        await settle_lab(lab_app, reopened, pilot)
        label = str(reopened.query_one("#lab-sample-source").content)
        assert all(value in label for value in labels)


@pytest.mark.asyncio
async def test_initial_load_survives_yielding_mount_handler(lab_app, monkeypatch):
    """Mount-time scheduling must not silently abandon recovery initialization."""
    from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen

    async def yielding_mount(self):
        # A real Mount handler may yield before Textual sets is_mounted. This
        # reproduces the live PTY ordering without modifying framework state.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    monkeypatch.setattr(BaseAppScreen, "on_mount", yielding_mount)
    async with lab_app.run_test(size=(120, 40)) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await asyncio.wait_for(screen.wait_until_ready(), 3)
        await settle_lab(lab_app, screen, pilot)
        assert screen.coordinator is not None
        assert not screen.query_one("#lab-sample").disabled


@pytest.mark.asyncio
async def test_lazy_screen_workers_abandon_work_after_teardown(lab_app, monkeypatch):
    import inspect

    async with lab_app.run_test(size=(80, 24)) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        pending = []
        monkeypatch.setattr(
            screen, "run_worker", lambda work, **kwargs: pending.append(work)
        )
        try:
            await lab_app.push_screen(screen)
            await pilot.pause()
            screen._coordinator_changed(None)
            assert len(pending) == 2
            assert all(inspect.iscoroutinefunction(work) for work in pending)
            await lab_app.pop_screen()
            for work in tuple(pending):
                await work()
            assert screen.coordinator is None
            assert getattr(lab_app, "_chunking_lab_coordinator", None) is None
        finally:
            for work in pending:
                if inspect.iscoroutine(work):
                    work.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,outcome",
    [((80, 24), "failed"), ((120, 40), "canceled"), ((160, 50), "pending")],
)
async def test_previous_choice_survives_reopen_without_completing_failed_batch(
    lab_app, size, outcome
):
    from textual.widgets import Button

    from Tests.Chunking.test_lab_state import _completed
    from tldw_chatbook.Chunking import lab_state
    from tldw_chatbook.Chunking.lab_models import RunResult
    from tldw_chatbook.UI.Chunking_Lab_Modules.results_region import ResultsRegion

    async with lab_app.run_test(size=size) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        initial = screen.coordinator.session
        request = lab_state.capture_batch(initial, (screen._b_id(),))[0]
        successful = lab_state.accept_result(
            lab_state.install_batch(initial, (request,)),
            _completed(request, "retained successful output"),
        )
        pinned = lab_state.pin_baseline(successful)
        reruns = lab_state.capture_batch(pinned, tuple(pinned.candidates))
        current = lab_state.install_batch(pinned, reruns)
        current = lab_state.accept_result(
            current, _completed(reruns[0], "current A output")
        )
        if outcome != "pending":
            current = lab_state.accept_result(
                current,
                RunResult(
                    request=reruns[1],
                    status=outcome,
                    report=None,
                    started_at="",
                    finished_at="",
                    elapsed_ms=0,
                    error={"message": "current B failure"},
                ),
            )
        screen.coordinator.set_session(current)
        await screen._refresh_session()
        await settle_lab(lab_app, screen, pilot)
        assert await pilot.click("#lab-show-results")
        await screen.drain_edits()
        region = screen.query_one(ResultsRegion)
        assert "retained successful output" not in region.query_one(TextArea).text
        previous = screen.query_one("#lab-previous-b", Button)
        assert not previous.disabled
        previous.focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert previous.region in screen.content_region
        await pilot.press("enter")
        await pilot.pause()
        await screen.drain_edits()
        await settle_lab(lab_app, screen, pilot)
        assert region.query_one(TextArea).text == "retained successful output"
        assert "Previous" in str(region.query_one("#status-b").content)
        assert (
            "current batch"
            in str(region.query_one("#comparison-status").content).lower()
        )
        assert (
            "B minus A common counts" not in region._prepared["documents"]["statistics"]
        )
        assert screen.coordinator.session.batch["outcomes"].get(reruns[1].run_id) == (
            None if outcome == "pending" else outcome
        )
        assert await screen.confirm_navigation()
        owner = screen.coordinator
        await lab_app.pop_screen()
        await owner.close()
        lab_app._chunking_lab_coordinator = None
        reopened = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(reopened)
        await reopened.wait_until_ready()
        await settle_lab(lab_app, reopened, pilot)
        region = reopened.query_one(ResultsRegion)
        assert region.query_one(TextArea).text == "retained successful output"
        assert "Previous" in str(region.query_one("#status-b").content)
        current_button = reopened.query_one("#lab-current-b", Button)
        current_button.focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert current_button.region in reopened.content_region
        await pilot.press("enter")
        await pilot.pause()
        await reopened.drain_edits()
        await settle_lab(lab_app, reopened, pilot)
        assert "Previous" not in str(region.query_one("#status-b").content)
        assert "retained successful output" not in region.query_one(TextArea).text
        assert set(reopened.coordinator.session.candidates) == set(current.candidates)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 50)])
async def test_complete_workflow_viewports(lab_app, tmp_path, size):
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestFormState,
        build_library_ingest_state,
    )
    from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import LibraryIngestCanvas

    database = MediaDatabase(str(tmp_path / "media.sqlite3"), client_id="lab-viewports")
    lab_app.media_db = database

    class Catalog:
        async def list_templates(self, *, mode):
            assert mode == "local"
            return await asyncio.to_thread(
                LocalRAGAdminService(database).list_templates
            )

    lab_app._rag_admin_scope_service = Catalog()
    from tldw_chatbook.Chunking import lab_state

    async with lab_app.run_test(size=size) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(lab_app)
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        await settle_lab(lab_app, screen, pilot)
        sample = "\n".join(
            f"Section {index}: exact local sample words for comparison."
            for index in range(1, 19)
        )
        assert await pilot.click("#lab-sample-text")
        await pilot.pause()
        assert lab_app.focused is screen.query_one("#lab-sample-text")
        lab_app.post_message(events.Paste(sample))
        await pilot.pause()
        await screen.drain_edits()
        assert (
            screen.coordinator.session.samples[
                screen.coordinator.session.view["sample_hash"]
            ]["text"]
            == sample
        )
        assert await pilot.click("#lab-run")
        await settle_lab(lab_app, screen, pilot)
        await pilot.pause()
        assert len(screen.coordinator.session.results) == 1
        result = next(iter(screen.coordinator.session.results.values()))
        assert result["status"] == "completed", result.get("error")
        assert (
            result["request"]["sample"]["sample_hash"]
            == screen.coordinator.session.view["sample_hash"]
        )
        from tldw_chatbook.Chunking import lab_state

        assert not lab_state.is_result_stale(
            screen.coordinator.session, screen._b_id()
        ), screen.coordinator.session.candidates[screen._b_id()]["draft"]
        assert await pilot.click("#lab-pin")
        await settle_lab(lab_app, screen, pilot)
        await pilot.pause()
        assert len(screen.coordinator.session.candidates) == 2, str(
            screen.query_one("#lab-message").content
        )
        assert await pilot.click("#lab-show-configure")
        await pilot.pause()
        screen.query_one("#lab-name").value = "Local comparison recipe"
        screen.query_one(
            "#lab-description"
        ).value = "Locally verified comparison recipe"
        screen.query_one("#lab-max-size").value = "120"
        screen.query_one("#lab-overlap").value = "20"
        screen.query_one("#lab-method").value = "fixed_size"
        await pilot.pause()
        await screen.drain_edits()
        assert await pilot.click("#lab-both")
        await settle_lab(lab_app, screen, pilot)
        await pilot.pause()
        assert len(screen.coordinator.session.batch["outcomes"]) == 2
        assert await pilot.click("#lab-save-b")
        await pilot.pause()
        assert lab_app.screen is not screen
        assert await pilot.click("#dialog-accept")
        await pilot.pause(0.5)
        assert lab_app.screen is screen, getattr(
            lab_app.screen, "explanation", "dialog remained"
        )
        await settle_lab(lab_app, screen, pilot)
        await pilot.pause()
        assert lab_app.screen is screen
        saved = LocalRAGAdminService(database).list_templates()
        assert any(record["name"] == "Local comparison recipe" for record in saved)
        if size[0] != 120:
            assert await pilot.click("#lab-show-results")
        await pilot.pause()
        assert await pilot.click("#view-compare")
        await pilot.pause()
        await screen.drain_edits()
        await screen.coordinator.cancel()
        await settle_lab(lab_app, screen, pilot)
        await pilot.pause()
        await settle_lab(lab_app, screen, pilot)
        await pilot.pause()
        if size[0] == 80:
            inspector = screen.query_one("#chunk-inspector", TextArea)
            inspector.focus()
            await pilot.pause()
            await asyncio.wait_for(pilot.wait_for_scheduled_animations(), 3)
            assert lab_app.focused is inspector
            assert inspector.region in screen.content_region
            assert screen.query_one("#mapping-status").region in screen.content_region
        elif size[0] == 160:
            screen.query_one("#detail-kind").value = "effective"
            await settle_lab(lab_app, screen, pilot)
            assert (
                "Captured snapshots"
                in screen.query_one("#chunk-inspector", TextArea).text
            )
        capture = os.environ.get("TLDW_LAB_CAPTURE_DIR")
        if capture:
            directory = Path(capture)
            directory.mkdir(parents=True, exist_ok=True)
            lab_app.save_screenshot(f"lab-{size[0]}x{size[1]}.svg", path=directory)
            (directory / f"profile-{size[0]}x{size[1]}.json").write_text(
                json.dumps(
                    {
                        "lab_profile": screen.coordinator.session.profile_key,
                        "config": os.environ["TLDW_CONFIG_PATH"],
                        "home": os.environ["HOME"],
                        "sample_hash": screen.coordinator.session.view["sample_hash"],
                        "revision": screen.coordinator.session.revision,
                    },
                    indent=2,
                )
            )
        # The actual ingest consumer sees the saved canonical name on next display.
        await lab_app.pop_screen()
        canvas = LibraryIngestCanvas(
            build_library_ingest_state((), form=LibraryIngestFormState()),
            id="verification-ingest",
        )
        await lab_app.screen.mount(canvas)
        await canvas._fetch_chunk_templates()
        assert "Local comparison recipe" in canvas._chunk_template_names
        canvas.query_one(
            "#opt-generic-chunk_template"
        ).value = "Local comparison recipe"
        await pilot.pause()
        assert (
            canvas.query_one("#opt-generic-chunk_template").value
            == "Local comparison recipe"
        )


def test_lab_is_library_owned_and_lazy():
    route = resolve_screen_route("chunking_lab")
    assert route is not None
    assert route.canonical_tab == "library"
    assert route.module_path == "tldw_chatbook.UI.Screens.chunking_lab_screen"


def test_library_handoff_uses_local_id_and_refuses_server(lab_app):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    assert hasattr(LibraryScreen, "open_chunking_lab")
    screen = LibraryScreen(lab_app)
    messages = []
    lab_app.post_message = lambda message: messages.append(message)
    screen._library_media_reader_session = SimpleNamespace(
        external_detail=False,
        loaded_backing_id=42,
        loaded_id="local:media:42",
        pending_request=None,
    )
    screen.open_chunking_lab(use_selected=True)
    assert messages[-1].screen_name == "chunking_lab"
    assert messages[-1].screen_context == {
        "return_route": "library",
        "local_media_id": 42,
    }
    screen._library_media_reader_session.external_detail = True
    screen.open_chunking_lab(use_selected=True)
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    assert (
        len([message for message in messages if isinstance(message, NavigateToScreen)])
        == 1
    )


@pytest.mark.asyncio
async def test_profile_owner_is_single_flight_and_recovers(lab_app):
    assert hasattr(lab_app, "get_chunking_lab_coordinator")
    first, second = await asyncio.gather(
        lab_app.get_chunking_lab_coordinator(), lab_app.get_chunking_lab_coordinator()
    )
    assert first is second
    from tldw_chatbook.Chunking.lab_state import replace_sample

    first.set_session(
        await asyncio.to_thread(
            replace_sample, first.session, "exact\ntext", {"kind": "paste"}
        )
    )
    await first.close()
    lab_app._chunking_lab_coordinator = None
    recovered = await lab_app.get_chunking_lab_coordinator()
    assert (
        recovered.session.samples[recovered.session.view["sample_hash"]]["text"]
        == "exact\ntext"
    )
    assert not recovered.busy
    await recovered.close()


@pytest.mark.asyncio
async def test_local_shortcuts_do_not_steal_editor_text(lab_app):
    route = resolve_screen_route("chunking_lab")
    assert route is not None
    async with lab_app.run_test(size=(80, 24)) as pilot:
        await lab_app.push_screen(route.load_screen_class()(lab_app))
        await pilot.pause()
        await lab_app.screen.wait_until_ready()
        editor = lab_app.screen.query_one("#lab-sample-text", TextArea)
        editor.focus()
        await pilot.press("r", "p", "s")
        await lab_app.screen.drain_edits()
        assert editor.text.endswith("rps")
        coordinator = await lab_app.get_chunking_lab_coordinator()
        assert coordinator.session.samples[coordinator.session.view["sample_hash"]][
            "text"
        ].endswith("rps")
        await pilot.press("f6")
        await pilot.pause()
        assert lab_app.screen._active_region == "configure", (
            lab_app.active_bindings.get("f6")
        )
        assert lab_app.screen.focused.id == "lab-name"
        await pilot.press("f6")
        await pilot.pause()
        assert lab_app.screen._active_region == "results"


@pytest.mark.asyncio
async def test_real_library_text_save_a_and_conflict_preserve_state(lab_app, tmp_path):
    import json

    from tldw_chatbook.Chunking import lab_state
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.RAG_Admin.chunking_lab_service import TemplateSaveConflict

    database = MediaDatabase(str(tmp_path / "media.sqlite3"), client_id="lab-test")
    lab_app.media_db = database
    text = "line\r\n" + "local full source " * 1800 + "exact tail"
    media_id, _, _ = database.add_media_with_keywords(
        title="Full source", media_type="document", content=text
    )
    route = resolve_screen_route("chunking_lab")
    async with lab_app.run_test(size=(120, 40)):
        screen = route.load_screen_class()(lab_app)
        screen.apply_navigation_context(
            {"return_route": "library", "local_media_id": media_id}
        )
        await lab_app.push_screen(screen)
        await screen.wait_until_ready()
        assert (
            screen.coordinator.session.samples[
                screen.coordinator.session.view["sample_hash"]
            ]["text"]
            == text
        )
        authored = {
            "chunking": {"method": "words", "config": {"max_size": 80}},
            "metadata": {"custom": {"kept": True}},
            "classifier": {"media_types": ["document"]},
        }
        screen.queue_edit(
            lambda session: lab_state.replace_template(
                session,
                screen._b_id(session),
                authored,
                record_fields={
                    "name": "Captured name",
                    "description": "captured",
                    "tags": ["pin"],
                },
                expected_record=None,
            )
        )
        await screen.drain_edits()
        await screen.run_candidates()
        await screen._pin()
        screen.queue_edit(
            lambda session: lab_state.edit_record_fields(
                session,
                screen._b_id(session),
                {"name": "Changed B", "tags": ["different"]},
            )
        )
        await screen.drain_edits()
        record_a = await screen.save_candidate("A")
        assert record_a["name"] == "Captured name" and record_a["tags"] == ["pin"]
        body = json.loads(record_a["template_json"])
        assert body["metadata"] == authored["metadata"]
        assert body["classifier"] == authored["classifier"]
        assert body["chunking"]["config"]["overlap"] == 50
        record_b = await screen.save_candidate("B")
        service = screen._catalog_service()
        await asyncio.to_thread(
            service.update_template, record_b["id"], description="external update"
        )
        before = screen.coordinator.session.candidates[screen._b_id()]["draft"]
        with pytest.raises(TemplateSaveConflict):
            await screen.save_candidate("B")
        assert screen.coordinator.session.candidates[screen._b_id()]["draft"] == before
        assert (await asyncio.to_thread(database.get_media_by_id, media_id))[
            "content"
        ] == text
