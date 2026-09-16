"""Actual Run/setup/review controls with production CSS and real local owners."""

import asyncio
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from textual.screen import Screen
from textual.widgets import Button, Input, OptionList, Static, TextArea

from Tests.UI.test_workflows_editor import (
    WorkflowEditorHarness,
    assert_hit,
    painted_text,
    svg_text_contrast,
)
from Tests.Workflows import test_session as session_tests
from Tests.Workflows.test_session import until
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import save_setting_to_cli_config
from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
    LocalLLMProviderCatalogService,
)
from tldw_chatbook.UI.Screens.workflows_screen import WorkflowsScreen

harness = session_tests.harness


class WorkflowRunHarness(WorkflowEditorHarness):
    ensure_workflow_session = TldwCli.ensure_workflow_session
    action_quit = TldwCli.action_quit
    _confirm_and_quit = TldwCli._confirm_and_quit
    _confirm_console_runtime_quit = TldwCli._confirm_console_runtime_quit
    _confirm_workflow_session_quit = TldwCli._confirm_workflow_session_quit

    def __init__(self, tmp_path, h):
        super().__init__(tmp_path)
        self._workflow_session = None
        self._workflow_database_path = tmp_path / "workflows.sqlite3"
        self.notes_scope_service = h.scope
        self.notes_user_id = "reader"
        self.unified_mcp_service = SimpleNamespace(permission_store=h.store)
        self.local_llm_provider_catalog_service = LocalLLMProviderCatalogService(
            provider_catalog_loader=lambda: {"llama_cpp": ["actual-model"]}
        )
        self.notes_fixture = h
        self.library_app = None
        self._quit_in_progress = False
        self._shutting_down = False
        self.cleanup_calls = 0

    async def _run_approved_quit_cleanup(self):
        self.cleanup_calls += 1

    def _close_boot_worker_gate(self, reason):
        pass

    def on_navigate_to_screen(self, message):
        # Match the production navigation worker: mounting Library must not
        # block the app message pump that its readiness messages need.
        self.run_worker(self._navigate(message), group="screen-navigation")

    async def _navigate(self, message):
        from Tests.UI.app_factory import _build_test_app
        from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
        from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

        assert message.screen_name == "library"
        self.library_app = _build_test_app()
        self.library_app.notes_scope_service = self.notes_scope_service
        self.library_app.notes_service = self.notes_fixture.owner
        self.library_app.notes_user_id = self.notes_user_id
        self.library_app.chachanotes_db = self.notes_fixture.template
        self.notes_scope_service.folder_repository = LocalNoteFolderRepository(
            self.notes_fixture.template
        )

        # The shared factory deliberately omits Media/Conversation DBs. Supply
        # empty unrelated lists; Note save/detail/list still use the real owner.
        async def empty_sources(**kwargs):
            return []

        self.library_app.media_reading_scope_service = SimpleNamespace(
            list_media_items=empty_sources
        )
        self.library_app.chat_conversation_scope_service = SimpleNamespace(
            list_conversations=empty_sources
        )
        screen = LibraryScreen(self.library_app)
        screen.apply_navigation_context(message.screen_context)
        await self.switch_screen(screen)

    async def on_unmount(self):
        if self._workflow_session is not None:
            await self._workflow_session.close()
        await super().on_unmount()


async def configure_model():
    save_setting_to_cli_config(
        "api_settings",
        "llama_cpp",
        {
            "api_url": "http://127.0.0.1:9099",
            "credential_source": "none",
            "timeout": 120,
        },
    )


async def test_real_run_press_opens_reviewed_setup(tmp_path, harness):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        run = app.screen.query_one("#workflow-run", Button)
        assert not run.disabled
        assert_hit(app.screen, run)
        await pilot.click("#workflow-run")
        await pilot.pause()
        assert app.screen.query("#workflow-source")
        assert "Local Note" in painted_text(app.screen)
        await pilot.click("#workflow-setup-cancel")
        await pilot.pause()


async def open_setup(app, pilot, source, capture_size=None):
    await pilot.pause()
    await pilot.click("#workflow-run")
    await pilot.pause()
    app.screen.query_one("#workflow-source", Input).value = str(source)
    await pilot.pause()
    if capture_size:
        capture(app, capture_size, "setup-inputs")
    await pilot.click("#workflow-setup-review")
    async with asyncio.timeout(10):
        while not app.screen.query("#workflow-start"):
            await pilot.pause()
    assert "Keyless execution" in "\n".join(
        str(widget.renderable) for widget in app.screen.query(Static)
    )


async def start_review(app, pilot, source):
    await open_setup(app, pilot, source)
    await pilot.click("#workflow-start")
    await until(app._workflow_session, lambda v: v.state == "review")
    await pilot.pause()
    return app._workflow_session.view()


def capture(app, size, state, *, svg_only=False):
    if not os.environ.get("WORKFLOW_UI_CAPTURES"):
        return
    folder = (
        Path(__file__).parents[2]
        / ".superpowers/sdd/2026-09-16-workflows-first-run/ui-captures"
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"task-5-{size[0]}x{size[1]}-{state}.svg"
    svg = app.export_screenshot(simplify=True)
    path.write_text(svg, encoding="utf-8")
    if not svg_only:
        import cairosvg

        cairosvg.svg2png(
            bytestring=svg.encode(), write_to=str(path.with_suffix(".png"))
        )


def control_contrast(app, label):
    # Rich merges adjacent same-style labels into one painted text span.
    from xml.etree import ElementTree

    svg = app.export_screenshot(simplify=True)
    tree = ElementTree.fromstring(svg)
    span = next(
        " ".join((node.text or "").split())
        for node in tree.iter("{http://www.w3.org/2000/svg}text")
        if label in " ".join((node.text or "").split())
    )
    return svg_text_contrast(svg, span)[2]


@pytest.mark.parametrize("size", [(160, 48), (110, 36), (60, 20)])
async def test_pressed_flow_edits_note_navigates_and_paints(
    tmp_path, harness, size, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=size) as pilot:
        await open_setup(app, pilot, harness.setup.source, size)
        start = app.screen.query_one("#workflow-start", Button)
        start.focus()
        await pilot.pause()
        assert_hit(app.screen, start)
        capture(app, size, "setup")
        from textual.containers import VerticalScroll

        app.screen.query_one(VerticalScroll).scroll_end(animate=False)
        await pilot.pause()
        assert "Saved Notes remain." in painted_text(app.screen)
        capture(app, size, "setup-destination")
        await pilot.press("enter")
        await until(app._workflow_session, lambda v: v.state == "review")
        await pilot.pause()
        review = app.screen.query_one("#workflow-review-text", TextArea)
        review.load_text("Human-edited summary")
        review.focus()
        review.scroll_visible(animate=False)
        await pilot.pause()
        assert_hit(app.screen, review)
        assert "Human-edited summary" in painted_text(app.screen)
        assert app.focused is review
        assert review.content_size.height > 0
        capture(app, size, "review")
        accept = app.screen.query_one("#workflow-review-accept", Button)
        assert_hit(app.screen, accept)
        await pilot.hover("#workflow-session-status")
        await pilot.pause()
        rest = accept.styles.background
        assert control_contrast(app, "Accept") >= 4.5
        await pilot.hover("#workflow-review-accept")
        await pilot.pause()
        assert accept.styles.background != rest
        assert control_contrast(app, "Accept") >= 4.5
        capture(app, size, "review-hover")
        await pilot.click("#workflow-review-accept")
        await pilot.pause()
        final = await until(app._workflow_session, lambda v: v.state == "completed")
        await pilot.pause()
        assert harness.rows()[0]["content"] == "Human-edited summary"
        assert harness.rows()[0]["id"] == final.note_id
        button = app.screen.query_one("#workflow-open-note", Button)
        button.focus()
        await pilot.pause()
        assert_hit(app.screen, button)
        assert app.focused is button
        assert app.screen.query_one("#workflow-cancel", Button).disabled
        assert control_contrast(app, "Open Note") >= 4.5
        assert control_contrast(app, "Cancel run") >= 3
        capture(app, size, "result")
        await pilot.click("#workflow-open-note")
        await pilot.pause()
        assert app.library_app is not None, painted_text(app.screen)
        async with asyncio.timeout(15):
            while not app.screen.query("#library-note-body"):
                await pilot.pause()
            while (
                app.screen.query_one("#library-note-body", TextArea).text
                != "Human-edited summary"
            ):
                await pilot.pause()
        assert app.screen._notes_state.selected_note_id == final.note_id
        assert app.screen._notes_state.load_state == "loaded"
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.library_app._library_source_snapshot_cache[3] is None
        assert "Library source services unavailable" not in painted_text(app.screen)
        assert "Couldn't load" not in painted_text(app.screen)
        assert "Couldn’t load" not in painted_text(app.screen)
        capture(app, size, "library")


async def test_navigation_retains_edits_and_original_revision(tmp_path, harness):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        view = await start_review(app, pilot, harness.setup.source)
        app.screen.query_one("#workflow-review-text", TextArea).load_text(
            "Retained edit"
        )
        await pilot.pause()
        saved_state = app.screen.save_state()
        assert "Retained edit" not in repr(saved_state)
        await app.switch_screen(Screen())
        await pilot.pause()
        assert app._workflow_session.view().review_text == "Retained edit"
        doc = json.loads(
            app.workflow_documents.get_revision(
                view.workflow_id, view.revision_id
            ).raw_json
        )
        doc["name"] = "A different workflow"
        doc["metadata"]["tldw_workflow"]["workflow_id"] = str(uuid4())
        doc["metadata"]["tldw_workflow"]["revision_id"] = str(uuid4())
        other = app.workflow_documents.create(json.dumps(doc))
        await app.switch_screen(WorkflowsScreen(app))
        await pilot.pause()
        await app.screen._select_workflow(other.workflow_id, other.revision_id)
        await pilot.pause()
        assert app.screen.controller.head.revision_id == other.revision_id
        assert app._workflow_session.view().revision_id == view.revision_id
        assert (
            app.screen.query_one("#workflow-review-text", TextArea).text
            == "Retained edit"
        )
        await pilot.press("f6", "tab")
        assert app.focused is not None
        await pilot.click("#workflow-review-reject")
        await until(app._workflow_session, lambda v: v.state == "rejected")
        assert harness.rows() == []


@pytest.mark.parametrize(
    "provider,credential", [("ollama", None), ("llama_cpp", "secret")]
)
async def test_unsupported_or_credentialed_provider_blocks_without_effects(
    tmp_path, harness, provider, credential
):
    await configure_model()
    if credential:
        save_setting_to_cli_config(
            "api_settings",
            "llama_cpp",
            {
                "api_url": "http://127.0.0.1:9099",
                "api_key": credential,
            },
        )
    app = WorkflowRunHarness(tmp_path, harness)
    app.local_llm_provider_catalog_service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {provider: ["actual-model"]}
    )
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        await pilot.click("#workflow-run")
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert "keyless llama.cpp" in painted_text(app.screen)
        assert app._workflow_session is None
        assert harness.requests == [] and harness.rows() == []


async def test_snapshot_is_current_and_final_bindings_survive_settings_change(
    tmp_path, harness
):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    app.app_config = {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:1"}}}
    async with app.run_test(size=(110, 36)) as pilot:
        await open_setup(app, pilot, harness.setup.source)
        save_setting_to_cli_config(
            "api_settings",
            "llama_cpp",
            {
                "api_url": "http://127.0.0.1:2",
                "api_key": "new secret",
            },
        )
        await pilot.click("#workflow-start")
        await until(app._workflow_session, lambda v: v.state == "review")
        assert len(harness.requests) == 1
        view = app._workflow_session.view()
        assert (
            app._workflow_session.run_bindings(view.run_id).model.selected_url
            == "http://127.0.0.1:9099"
        )


async def test_invalid_review_does_not_accept_old_value_and_reject_writes_nothing(
    tmp_path, harness
):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        await start_review(app, pilot, harness.setup.source)
        text = app.screen.query_one("#workflow-review-text", TextArea)
        text.load_text("x" * (1024 * 1024 + 1))
        await pilot.pause()
        assert app.screen.query_one("#workflow-review-accept", Button).disabled
        assert app._workflow_session.view().review_text is None
        assert app._workflow_session.view().message_code == "review_invalid"
        await pilot.click("#workflow-review-reject")
        await until(app._workflow_session, lambda v: v.state == "rejected")
        assert harness.rows() == []


async def test_exact_effect_ask_double_delivery_and_literal_review_instructions(
    tmp_path, harness
):
    await configure_model()
    harness.set_permission("workflow_read_file", "ask")
    app = WorkflowRunHarness(tmp_path, harness)
    head = app.workflow_documents.list_workflows()[0]
    document = json.loads(head.raw_json)
    document["steps"][3]["config"]["instructions"] = (
        "Review {{ inputs.note_title }} [bold]literal[/bold]"
    )
    app.workflow_documents.put_draft(
        head.workflow_id, head.revision_id, json.dumps(document), 1
    )
    app.workflow_documents.save_revision(head.workflow_id, head.revision_id, 1)
    async with app.run_test(size=(110, 36)) as pilot:
        await open_setup(app, pilot, harness.setup.source)
        await pilot.click("#workflow-start")
        await until(app._workflow_session, lambda v: v.state == "approval")
        await pilot.pause()
        assert str(harness.setup.source) in str(
            app.screen.query_one("#workflow-effect-text", Static).renderable
        )
        button = app.screen.query_one("#workflow-effect-approve", Button)
        button.press()
        button.press()
        await until(app._workflow_session, lambda v: v.state == "review")
        await pilot.pause()
        instructions = app.screen.query_one("#workflow-review-instructions", Static)
        assert (
            str(instructions.renderable)
            == "Review Reviewed file summary [bold]literal[/bold]"
        )
        instructions.scroll_visible(animate=False)
        await pilot.pause()
        assert "[bold]literal[/bold]" in painted_text(app.screen)
        button = app.screen.query_one("#workflow-review-accept", Button)
        button.press()
        button.press()
        await until(app._workflow_session, lambda v: v.state == "completed")
        assert len(harness.rows()) == len(harness.requests) == 1


async def test_changed_destination_refuses_open_but_later_note_edits_are_legal(
    tmp_path, harness
):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        await start_review(app, pilot, harness.setup.source)
        await pilot.click("#workflow-review-accept")
        view = await until(app._workflow_session, lambda v: v.state == "completed")
        await pilot.pause()
        # A legitimate edit must not trigger workflow readback reconciliation or recreation.
        with harness.template.transaction() as cursor:
            cursor.execute(
                "UPDATE notes SET content = ? WHERE id = ?",
                ("Later edit", view.note_id),
            )
        app.notes_user_id = "other"
        await pilot.click("#workflow-open-note")
        await pilot.pause()
        assert isinstance(app.screen, WorkflowsScreen)
        assert "destination changed" in str(
            app.screen.query_one("#workflow-draft-status", Static).renderable
        )
        app.notes_user_id = "reader"
        await pilot.pause(0.3)  # Let the real Button's active-click debounce expire.
        assert_hit(app.screen, app.screen.query_one("#workflow-open-note"))
        assert await pilot.click("#workflow-open-note")
        await pilot.pause()
        assert app.library_app is not None, painted_text(app.screen)
        async with asyncio.timeout(15):
            while not app.screen.query("#library-note-body"):
                await pilot.pause()
            while (
                app.screen.query_one("#library-note-body", TextArea).text
                != "Later edit"
            ):
                await pilot.pause()
        assert len(harness.rows()) == 1


@pytest.mark.parametrize("cancel_second", [True, False])
async def test_open_note_during_second_note_keeps_loop_and_stop_responsive(
    tmp_path, harness, monkeypatch, cancel_second
):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    head = app.workflow_documents.list_workflows()[0]
    document = json.loads(head.raw_json)
    second = json.loads(json.dumps(document["steps"][-1]))
    second["id"] = "save_again"
    document["steps"].append(second)
    app.workflow_documents.put_draft(
        head.workflow_id, head.revision_id, json.dumps(document), 1
    )
    app.workflow_documents.save_revision(head.workflow_id, head.revision_id, 1)
    entered, release = threading.Event(), threading.Event()
    real_lock = harness.owner._db_lock
    loop_thread = threading.get_ident()
    blocking_loop_requests = []
    worker_acquires = []

    class GuardedLock:
        """Keep the real mutex; reject the deadlocking acquire before it waits."""

        def acquire(self, blocking=True):
            on_loop = threading.get_ident() == loop_thread
            if on_loop and blocking and real_lock.locked():
                blocking_loop_requests.append(True)
                raise RuntimeError("contended blocking acquire on app loop")
            acquired = real_lock.acquire(blocking=blocking)
            if acquired and not on_loop:
                worker_acquires.append(True)
                if len(worker_acquires) == 2:
                    # Second real Note owns the mutex, before its inner authority
                    # callback can synchronously wait for the application loop.
                    entered.set()
                    if not release.wait(30):
                        real_lock.release()
                        raise RuntimeError("test barrier watchdog expired")
            return acquired

        def release(self):
            real_lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *exc):
            self.release()

    async with app.run_test(size=(110, 36)) as pilot:
        await start_review(app, pilot, harness.setup.source)
        app.screen.query_one("#workflow-review-text", TextArea).load_text(
            "Exact accepted text é\nSecond line."
        )
        await pilot.pause()
        monkeypatch.setattr(harness.owner, "_db_lock", GuardedLock())
        app.install_screen(app.screen, "retained-workflows")
        try:
            await pilot.click("#workflow-review-accept")
            assert await asyncio.to_thread(entered.wait, 10)
            await pilot.pause()
            session = app._workflow_session
            first = session.view()
            assert first.step_id == "save_again" and first.note_id
            assert harness.rows() == [
                {
                    "id": first.note_id,
                    "title": "Reviewed file summary",
                    "content": "Exact accepted text é\nSecond line.",
                    "deleted": 0,
                }
            ]
            button = app.screen.query_one("#workflow-open-note", Button)
            assert not button.disabled
            # Queue a real captured event; disabling its button cannot guard it.
            queued = Button.Pressed(button)
            queued.workflow_view = first
            app.screen.query_one("#workflow-session").post_message(queued)
            await pilot.pause()
            assert not blocking_loop_requests, (
                "Open Note requested a contended blocking acquire on the app loop"
            )
            assert isinstance(app.screen, WorkflowsScreen)
            status = app.screen.query_one("#workflow-draft-status", Static)
            status.scroll_visible(animate=False)
            await pilot.pause()
            assert "busy" in painted_text(app.screen).lower()
            assert "try" in painted_text(app.screen).lower()
            heartbeat = asyncio.Event()
            app.call_later(heartbeat.set)
            await asyncio.wait_for(heartbeat.wait(), 2)
            # Quit/Stay remains interactive even with the worker physically held.
            app.action_quit()
            await pilot.pause()
            assert app.screen.query_one("#cancel-button", Button).label.plain == "Stay"
            await pilot.click("#cancel-button")
            await pilot.pause()
            assert not app._quit_in_progress and app.cleanup_calls == 0
            assert not session._run_task.done()
            if cancel_second:
                assert await pilot.click("#workflow-cancel")
                await pilot.pause()
                assert session.view().state == "stopping"
                assert not session._run_task.done()
                assert len(harness.rows()) == 1
        finally:
            release.set()
        final = await until(
            session,
            lambda v: v.state == ("cancelled" if cancel_second else "completed"),
        )
        await asyncio.wait_for(asyncio.shield(session._run_task), 5)
        assert not real_lock.locked()
        assert len(worker_acquires) == 2
        rows = harness.rows()
        assert len(rows) == (1 if cancel_second else 2)
        assert len({row["id"] for row in rows}) == len(rows)
        assert all(
            row["content"] == "Exact accepted text é\nSecond line." for row in rows
        )
        assert (final.note_id == first.note_id) is cancel_second
        await pilot.pause()
        if not cancel_second:
            # A queued first-result event must not open the newly displayed Note.
            stale = Button.Pressed(button)
            stale.workflow_view = first
            app.screen.query_one("#workflow-session").post_message(stale)
            await pilot.pause()
            assert isinstance(app.screen, WorkflowsScreen)
            assert app.library_app is None
        # Explicit fresh retry opens exactly the currently confirmed result.
        app.screen.query_one("#workflow-open-note", Button).press()
        async with asyncio.timeout(15):
            while not app.screen.query("#library-note-body"):
                await pilot.pause()
            while app.screen.query_one("#library-note-body", TextArea).text != (
                "Exact accepted text é\nSecond line."
            ):
                await pilot.pause()
        assert app.screen._notes_state.selected_note_id == final.note_id
        assert len(harness.rows()) == len(rows)
        await app.switch_screen("retained-workflows")
        await pilot.pause()
        status = app.screen.query_one("#workflow-draft-status", Static)
        status.scroll_visible(animate=False)
        await pilot.pause()
        assert "busy" not in painted_text(app.screen).lower()


@pytest.mark.parametrize(
    "change", ["scope", "user", "owner", "path", "client", "cache", "missing"]
)
async def test_queued_open_note_rechecks_destination(
    tmp_path, harness, monkeypatch, change
):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        await start_review(app, pilot, harness.setup.source)
        await pilot.click("#workflow-review-accept")
        view = await until(app._workflow_session, lambda v: v.state == "completed")
        await pilot.pause()
        destination = app._workflow_session.run_bindings(view.run_id).notes
        queued = Button.Pressed(app.screen.query_one("#workflow-open-note", Button))
        queued.workflow_view = view
        with monkeypatch.context() as changed:
            if change == "scope":
                changed.setattr(app, "notes_scope_service", None)
            elif change == "user":
                changed.setattr(app, "notes_user_id", "other")
            elif change == "owner":
                changed.setattr(harness.scope, "local_notes_service", None)
            elif change == "path":
                changed.setattr(destination.db, "db_path_str", "other")
            elif change == "client":
                changed.setattr(destination.db, "client_id", "other")
            elif change == "cache":
                changed.setitem(harness.owner._db_instances, "reader", harness.template)
            else:
                changed.delitem(harness.owner._db_instances, "reader")
            app.screen.query_one("#workflow-session").post_message(queued)
            await pilot.pause()
            assert isinstance(app.screen, WorkflowsScreen)
            assert app.library_app is None
            assert "destination changed" in str(
                app.screen.query_one("#workflow-draft-status", Static).renderable
            )
        assert len(harness.rows()) == 1


async def test_delayed_accept_message_cannot_accept_a_replacement_run(
    tmp_path, harness, monkeypatch
):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        first = await start_review(app, pilot, harness.setup.source)
        button = app.screen.query_one("#workflow-review-accept", Button)
        delayed = []
        original_post = button.post_message

        def retain(message):
            if isinstance(message, Button.Pressed):
                delayed.append(message)
                return True
            return original_post(message)

        monkeypatch.setattr(button, "post_message", retain)
        button.press()
        app._workflow_session.answer_review(first.run_id, first.step_id, accept=False)
        await until(app._workflow_session, lambda v: v.state == "rejected")
        await pilot.pause()
        second = await start_review(app, pilot, harness.setup.source)
        assert first.run_id != second.run_id
        monkeypatch.setattr(button, "post_message", original_post)
        app.screen.query_one("#workflow-session").post_message(delayed[0])
        await pilot.pause()
        assert app._workflow_session.view().state == "review"
        assert harness.rows() == []


@pytest.mark.parametrize("choice", ["saved", "save", "history"])
async def test_dirty_draft_requires_explicit_revision_choice(tmp_path, harness, choice):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        original = app.screen.controller.head
        changed = json.loads(app.workflow_drafts.current.raw_text)
        changed["inputs"]["note_title"] = "Changed draft title"
        app.workflow_drafts.update(json.dumps(changed))
        await pilot.pause()
        if choice == "history":
            await pilot.click("#workflow-save-revision")
            await pilot.pause()
            await app.workers.wait_for_complete()
            await app.screen._inspect(original.revision_id)
        await pilot.click("#workflow-run")
        await pilot.pause()
        if choice != "history":
            options = app.screen.query_one("#workflow-dialog-choices", OptionList)
            options.highlighted = options.get_option_index(choice)
            options.focus()
            await pilot.press("enter")
            await pilot.pause()
        assert app.screen.query("#workflow-source"), painted_text(app.screen)
        displayed = app.screen.revision
        assert (displayed.revision_id == original.revision_id) == (choice != "save")
        title = app.screen.query_one("#workflow-note-title", Input).value
        assert title == (
            "Changed draft title" if choice == "save" else "Reviewed file summary"
        )
        await pilot.click("#workflow-setup-cancel")
        await pilot.pause()
        assert harness.rows() == [] and harness.requests == []


async def test_review_expiry_while_unmounted_disables_accept_on_return(
    tmp_path, harness
):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    head = app.workflow_documents.list_workflows()[0]
    document = json.loads(head.raw_json)
    document["steps"][3]["config"]["timeout_seconds"] = 1
    app.workflow_documents.put_draft(
        head.workflow_id, head.revision_id, json.dumps(document), 1
    )
    app.workflow_documents.save_revision(head.workflow_id, head.revision_id, 1)
    async with app.run_test(size=(110, 36)) as pilot:
        await start_review(app, pilot, harness.setup.source)
        await app.switch_screen(Screen())
        view = await until(app._workflow_session, lambda v: v.state == "failed")
        assert view.message_code == "review_expired"
        await app.switch_screen(WorkflowsScreen(app))
        await pilot.pause()
        assert not app.screen.query_one("#workflow-review-accept").display
        assert harness.rows() == []


async def test_setup_picker_is_the_existing_txt_picker_and_cancel_has_no_effect(
    tmp_path, harness
):
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        await pilot.click("#workflow-run")
        await pilot.pause()
        await pilot.click("#workflow-source-choose")
        await pilot.pause()
        assert isinstance(app.screen, EnhancedFileOpen)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.click("#workflow-setup-cancel")
        await pilot.pause()
        assert app._workflow_session.view() is None
        assert harness.rows() == [] and harness.requests == []


async def test_real_stay_button_restores_painted_accept(tmp_path, harness):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        await start_review(app, pilot, harness.setup.source)
        assert "No active workflow run" not in painted_text(app.screen)
        assert "No existing Console handoff" in painted_text(app.screen)
        app.action_quit()
        await pilot.pause()
        assert app.screen.query_one("#cancel-button", Button).label.plain == "Stay"
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert app.cleanup_calls == 0
        assert not app._quit_in_progress
        assert not app.screen.query_one("#workflow-review-accept", Button).disabled
        await pilot.click("#workflow-review-reject")
        await until(app._workflow_session, lambda v: v.state == "rejected")


async def test_pressed_cancel_retains_held_note_commit(tmp_path, harness, monkeypatch):
    await configure_model()
    entered, release = threading.Event(), threading.Event()
    save = harness.scope.save_note

    async def held(**kwargs):
        result = await save(**kwargs)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(harness.scope, "save_note", held)
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        await start_review(app, pilot, harness.setup.source)
        await pilot.click("#workflow-review-accept")
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            await pilot.pause()
            assert await pilot.click("#workflow-cancel")
            await pilot.pause()
            assert app._workflow_session.view().state == "stopping"
            assert app.screen.query_one("#workflow-run", Button).disabled
            assert app.screen.query_one("#workflow-open-note", Button).disabled
            assert len(harness.rows()) == 1
        finally:
            release.set()
        final = await until(app._workflow_session, lambda v: v.state == "cancelled")
        assert final.message_code == "saved_after_cancel"
        assert final.note_id == harness.rows()[0]["id"]
        await pilot.pause()
        assert not app.screen.query_one("#workflow-open-note", Button).disabled


async def test_delayed_accept_does_not_accept_changed_review(
    tmp_path, harness, monkeypatch
):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=(110, 36)) as pilot:
        await start_review(app, pilot, harness.setup.source)
        button = app.screen.query_one("#workflow-review-accept", Button)
        delayed = []
        original_post = button.post_message

        def retain(message):
            if isinstance(message, Button.Pressed):
                delayed.append(message)
                return True
            return original_post(message)

        monkeypatch.setattr(button, "post_message", retain)
        button.press()
        app.screen.query_one("#workflow-review-text", TextArea).load_text("New review")
        await pilot.pause()
        monkeypatch.setattr(button, "post_message", original_post)
        app.screen.query_one("#workflow-session").post_message(delayed[0])
        await pilot.pause()
        assert app._workflow_session.view().state == "review"
        assert app._workflow_session.view().review_text == "New review"
        assert harness.rows() == []


@pytest.mark.parametrize("size", [(160, 48), (110, 36), (60, 20)])
@pytest.mark.parametrize(
    ("kind", "permission", "action"),
    [
        ("file", "workflow_read_file", "Read this local UTF-8 file"),
        ("model", "workflow_local_model", "Send input to local model"),
        ("note", "create_note", "Create a Local Note"),
    ],
)
async def test_capture_effect_ask_with_focused_approval(
    tmp_path, harness, size, monkeypatch, kind, permission, action
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    await configure_model()
    harness.set_permission(permission, "ask")
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=size) as pilot:
        await open_setup(app, pilot, harness.setup.source)
        await pilot.click("#workflow-start")
        if kind == "note":
            await until(app._workflow_session, lambda v: v.state == "review")
            await pilot.pause()
            await pilot.click("#workflow-review-accept")
        view = await until(app._workflow_session, lambda v: v.state == "approval")
        await pilot.pause()
        approve = app.screen.query_one("#workflow-effect-approve", Button)
        approve.focus()
        effect = app.screen.query_one("#workflow-effect-text", Static)
        summary = app.screen.query_one("#workflow-effect-summary", Static)
        summary.scroll_visible(animate=False, top=True)
        await pilot.pause()
        assert app.focused is approve
        assert_hit(app.screen, approve)
        assert_hit(app.screen, app.screen.query_one("#workflow-effect-reject"))
        assert str(effect.renderable) == view.pending_effect.payload_json
        assert view.pending_effect.kind == kind
        assert action in painted_text(app.screen)
        destination = (
            str(harness.setup.source)
            if kind == "file"
            else "http://127.0.0.1:9099/v1/chat/completions"
            if kind == "model"
            else app._workflow_session.run_bindings(view.run_id).notes.db_path
        )
        region = summary.content_region.intersection(
            app.screen.query_one("#workflow-session-content").content_region
        )
        painted_summary = "".join(
            row[region.x : region.right].strip()
            for row in painted_text(app.screen).splitlines()[region.y : region.bottom]
        )
        assert str(destination) in painted_summary
        if kind == "model":
            assert "llama_cpp / actual-model" in str(summary.renderable)
        elif kind == "note":
            assert "Reviewed file summary" in str(summary.renderable)
            assert "reader" in str(summary.renderable)
        assert "Approve once" in painted_text(app.screen)
        if kind == "file":
            capture(app, size, "ask-file-focused", svg_only=True)
        effect.scroll_visible(animate=False, top=True)
        await pilot.pause()
        assert '"config"' in painted_text(app.screen)
        assert app._workflow_session.view().pending_effect == view.pending_effect
        assert not app._workflow_session.answer_effect(
            view.run_id, view.step_id, str(summary.renderable), approve=True
        )
        assert app._workflow_session.view().pending_effect == view.pending_effect
        await pilot.click("#workflow-effect-reject")
        await until(app._workflow_session, lambda v: v.state == "rejected")
        assert len(harness.requests) == (1 if kind == "note" else 0)
        assert harness.rows() == []


@pytest.mark.parametrize("size", [(160, 48), (110, 36), (60, 20)])
async def test_capture_recoverable_setup_error(tmp_path, harness, size, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.click("#workflow-run")
        await pilot.pause()
        app.screen.query_one("#workflow-source", Input).value = str(
            harness.setup.source
        )
        inputs = app.screen.query_one("#workflow-inputs", TextArea)
        inputs.load_text("{invalid}")
        await pilot.click("#workflow-setup-review")
        await pilot.pause()
        inputs.focus()
        app.screen.query_one("#workflow-setup-error").scroll_visible(animate=False)
        await pilot.pause()
        assert inputs.text == "{invalid}"
        assert app.screen.query_one("#workflow-source", Input).value == str(
            harness.setup.source
        )
        assert "{invalid}" in painted_text(app.screen)
        assert "Inputs JSON is invalid" in painted_text(app.screen)
        assert "Use valid JSON" in painted_text(app.screen)
        assert app.focused is inputs
        assert_hit(app.screen, inputs)
        capture(app, size, "error-setup-focused", svg_only=True)
        inputs.load_text("{}")
        recovery = app.screen.query_one("#workflow-setup-review", Button)
        recovery.focus()
        # Textual debounces both mouse and keyboard activation while active.
        async with asyncio.timeout(2):
            while recovery.has_class("-active"):
                await pilot.pause()
        await pilot.press("enter")
        async with asyncio.timeout(10):
            while not app.screen.query("#workflow-start"):
                await pilot.pause()
        await pilot.click("#workflow-start-cancel")
        assert harness.requests == [] and harness.rows() == []


@pytest.mark.parametrize("size", [(160, 48), (110, 36), (60, 20)])
@pytest.mark.parametrize(
    ("field", "value", "problem", "correction"),
    [
        ("workflow-inputs", "{invalid}", "Inputs JSON is invalid", "Use valid JSON"),
        ("workflow-inputs", "[]", "Inputs must be a JSON object", "not a list"),
        ("workflow-source", "relative.txt", "Source must be an absolute", "Choose"),
        ("workflow-model", " ", "Model ID is required", "Enter the actual model ID"),
        (
            "workflow-note-title",
            " ",
            "Note title is required",
            "Enter a non-empty title",
        ),
    ],
)
async def test_setup_error_identifies_field_and_preserves_inputs(
    tmp_path, harness, size, field, value, problem, correction
):
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.click("#workflow-run")
        await pilot.pause()
        values = {
            "workflow-source": str(harness.setup.source),
            "workflow-model": "actual-model",
            "workflow-note-title": "Retained title [literal]",
            "workflow-inputs": '{"keep": "retained input"}',
        }
        values[field] = value
        for identifier, text in values.items():
            widget = app.screen.query_one("#" + identifier)
            if isinstance(widget, TextArea):
                widget.load_text(text)
            else:
                widget.value = text
        await pilot.click("#workflow-setup-review")
        await pilot.pause()
        error = app.screen.query_one("#workflow-setup-error", Static)
        error.scroll_visible(animate=False)
        await pilot.pause()
        assert problem in painted_text(app.screen)
        assert correction in painted_text(app.screen)
        for identifier, text in values.items():
            widget = app.screen.query_one("#" + identifier)
            assert (
                widget.text if isinstance(widget, TextArea) else widget.value
            ) == text
        assert not app.screen.query("#workflow-start")
        assert harness.requests == [] and harness.rows() == []


async def test_capture_narrow_setup_and_review_context(tmp_path, harness, monkeypatch):
    from textual.containers import VerticalScroll
    from textual.widgets import Select

    monkeypatch.delenv("NO_COLOR", raising=False)
    await configure_model()
    app = WorkflowRunHarness(tmp_path, harness)
    size = (60, 20)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.click("#workflow-run")
        await pilot.pause()
        app.screen.query_one("#workflow-source", Input).value = str(
            harness.setup.source
        )
        choice = app.screen.query_one("#workflow-model-choice", Select)
        choice.focus()
        choice.scroll_visible(animate=False, top=True)
        await pilot.pause()
        for identifier in (
            "workflow-model-choice",
            "workflow-model",
            "workflow-note-title",
        ):
            assert_hit(app.screen, app.screen.query_one("#" + identifier))
        assert "llama_cpp" in painted_text(app.screen)
        assert "Actual model ID" in painted_text(app.screen)
        assert "Reviewed file summary" in painted_text(app.screen)
        capture(app, size, "setup-inputs-scrolled", svg_only=True)
        await pilot.click("#workflow-setup-review")
        async with asyncio.timeout(10):
            while not app.screen.query("#workflow-start"):
                await pilot.pause()
        await pilot.click("#workflow-start")
        view = await until(app._workflow_session, lambda v: v.state == "review")
        await pilot.pause()
        app.screen.query_one("#workflow-review-accept", Button).focus()
        app.screen.query_one("#workflow-session-content", VerticalScroll).scroll_home(
            animate=False
        )
        await pilot.pause()
        painted = painted_text(app.screen)
        assert view.run_id in painted and view.revision_id in painted
        assert view.review_instructions in painted
        assert_hit(app.screen, app.screen.query_one("#workflow-review-accept"))
        capture(app, size, "review-context-scrolled", svg_only=True)
        await pilot.click("#workflow-review-reject")
        await until(app._workflow_session, lambda v: v.state == "rejected")
