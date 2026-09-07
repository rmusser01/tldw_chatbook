"""Product maturity Phase 3.4 source-selected Study generation contract."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import (
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _build_test_app,
    _wait_for_library_snapshot,
)
from Tests.UI.test_study_dashboard import (
    DashboardQuizScopeService,
    DashboardStudyScopeService,
)
import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Screens.study_screen import StudyScreen
from tldw_chatbook.UI.Screens.study_scope_models import (
    MATERIAL_SOURCE_LIBRARY,
    MATERIAL_TITLE_LIBRARY_SOURCES,
    StudyScopeContext,
    StudySourceItem,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
PHASE_3_4_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-phase-3-4-source-study-generation.md"
)
TASK_10 = Path(
    "backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md"
)
TASK_10_4 = Path(
    "backlog/tasks/task-10.4 - Product-Maturity-Phase-3.4-Source-Selected-Study-Generation.md"
)


class RecordingSourceStudyService(DashboardStudyScopeService):
    """Dashboard service stub that records source-selected study pack jobs."""

    async def create_study_pack_job(
        self,
        *,
        mode: str | None = None,
        title: str,
        source_items: list[dict[str, object]],
        workspace_id: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "create_study_pack_job",
                mode,
                title,
                workspace_id,
                source_items,
            )
        )
        return {"job": {"id": 42, "status": "queued"}}

    async def get_study_pack_job_status(
        self,
        *,
        mode: str | None = None,
        job_id: int,
    ) -> dict[str, object]:
        assert isinstance(job_id, int)
        self.calls.append(("get_study_pack_job_status", mode, job_id))
        statuses = getattr(self, "study_pack_job_statuses", [])
        if statuses:
            return statuses.pop(0)
        return {"job": {"id": job_id, "status": "queued"}}


async def _wait_for_source_generation_call(
    service: RecordingSourceStudyService,
    pilot,
    *,
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(call[0] == "create_study_pack_job" for call in service.calls):
            return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Timed out waiting for study pack generation call: {service.calls}"
    )


async def _wait_for_service_call(
    service: RecordingSourceStudyService,
    pilot,
    call_name: str,
    *,
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(call[0] == call_name for call in service.calls):
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {call_name}: {service.calls}")


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _static_text(widget: Static) -> str:
    return str(widget.render())


async def _close_production_app(app: TldwCli) -> None:
    """Release full-app resources even when an assertion fails."""
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


def _splash_disabled_setting(real_get_cli_setting):
    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    return get_cli_setting_without_splash


def _set_runtime_backend(app: TldwCli, source: str) -> None:
    state = RuntimeSourceState(
        active_source=source,
        server_configured=source == "server",
    )
    app.runtime_policy.state = state
    app._publish_runtime_policy_projection(state)


@asynccontextmanager
async def _run_library_app(app: TldwCli):
    """Run the full production application on its Library route."""
    app.app_config["_first_run"] = False
    app._initial_tab_value = "library"
    real_get_cli_setting = app_module.get_cli_setting
    try:
        with patch(
            "tldw_chatbook.app.get_cli_setting",
            side_effect=_splash_disabled_setting(real_get_cli_setting),
        ):
            async with app.run_test(size=(180, 50)) as pilot:
                for _ in range(300):
                    if app.current_tab == "library" and isinstance(
                        app.screen, LibraryScreen
                    ):
                        break
                    await pilot.pause(0.01)
                else:
                    raise AssertionError(
                        "full TldwCli did not finish routing to Library."
                    )
                yield app.screen, pilot
    finally:
        await _close_production_app(app)


@asynccontextmanager
async def _run_study_app(app: TldwCli, scope_context: StudyScopeContext):
    """Run a Library scope handoff through the full production application."""
    app.app_config["_first_run"] = False
    app._initial_tab_value = "home"
    real_get_cli_setting = app_module.get_cli_setting
    try:
        with patch(
            "tldw_chatbook.app.get_cli_setting",
            side_effect=_splash_disabled_setting(real_get_cli_setting),
        ):
            async with app.run_test() as pilot:
                for _ in range(300):
                    if app.current_tab == "home":
                        break
                    await pilot.pause(0.01)
                else:
                    raise AssertionError("full TldwCli did not reach Home.")

                app.open_study_screen(scope_context)
                for _ in range(300):
                    if (
                        app.current_tab == "study"
                        and isinstance(app.screen, StudyScreen)
                        and app.screen.scope_state.source_items
                        == scope_context.source_items
                    ):
                        break
                    await pilot.pause(0.01)
                else:
                    raise AssertionError(
                        "full TldwCli did not consume the Study scope handoff."
                    )
                yield app.screen, pilot
    finally:
        await _close_production_app(app)


@pytest.mark.asyncio
async def test_library_source_context_carries_study_pack_source_items() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "id": "chat-1"}]
    )
    app.open_study_screen = Mock()

    async with _run_library_app(app) as (screen, pilot):
        await _wait_for_library_snapshot(screen, pilot)
        # The retired hub rendered #library-open-study globally; the rail +
        # canvas shell only mounts it inside the create-study mode canvas, so
        # the Study row must be selected first to reach the same button.
        await pilot.click("#library-row-create-study")
        await pilot.pause(0.1)
        await pilot.click("#library-open-study")
        await pilot.pause(0.1)

    scope_context = app.open_study_screen.call_args.args[0]

    assert scope_context.source_items == (
        StudySourceItem(source_type="note", source_id="note-1", label="Research Note"),
        StudySourceItem(source_type="media", source_id="media-1", label="Transcript A"),
    )
    assert scope_context.material_titles == (
        "Research Note",
        "Transcript A",
        "Planning Chat",
    )


@pytest.mark.asyncio
async def test_server_study_dashboard_launches_source_selected_study_pack_job() -> None:
    service = RecordingSourceStudyService()
    app = _build_test_app()
    app.study_scope_service = service
    app.study_quiz_scope_service = DashboardQuizScopeService()
    _set_runtime_backend(app, "server")
    notifications = Mock(wraps=app.notify)
    app.notify = notifications
    scope_context = StudyScopeContext(
        material_source=MATERIAL_SOURCE_LIBRARY,
        material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
        material_summary="Local Library source snapshot staged for Study.",
        material_titles=("Research Note", "Transcript A"),
        source_items=(
            StudySourceItem(
                source_type="note", source_id="note-1", label="Research Note"
            ),
            StudySourceItem(
                source_type="media", source_id="media-1", label="Transcript A"
            ),
        ),
    )

    async with _run_study_app(app, scope_context) as (screen, pilot):
        generate_button = screen.query_one("#study-generate-source-pack", Button)

        assert generate_button.disabled is False
        assert "selected Library sources" in str(generate_button.tooltip)

        generate_button.press()
        await _wait_for_source_generation_call(service, pilot)

        status = screen.query_one("#study-source-generation-status", Static)

        assert "queued" in _static_text(status).lower()
        assert "42" in _static_text(status)

    assert (
        "create_study_pack_job",
        "server",
        "Local Library Sources",
        None,
        [
            {
                "source_type": "note",
                "source_id": "note-1",
                "label": "Research Note",
                "locator": {},
            },
            {
                "source_type": "media",
                "source_id": "media-1",
                "label": "Transcript A",
                "locator": {},
            },
        ],
    ) in service.calls
    notifications.assert_called_with(
        "Study pack generation queued.", severity="information"
    )


@pytest.mark.asyncio
async def test_server_study_dashboard_observes_completed_source_pack_for_reuse() -> (
    None
):
    service = RecordingSourceStudyService()
    service.study_pack_job_statuses = [
        {
            "job": {"id": 42, "status": "completed"},
            "study_pack": {
                "id": 9,
                "title": "Research Note Study Pack",
                "deck_id": 7,
                "status": "active",
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
            },
        }
    ]
    app = _build_test_app()
    app.study_scope_service = service
    app.study_quiz_scope_service = DashboardQuizScopeService()
    _set_runtime_backend(app, "server")
    notifications = Mock(wraps=app.notify)
    app.notify = notifications
    scope_context = StudyScopeContext(
        material_source=MATERIAL_SOURCE_LIBRARY,
        material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
        material_titles=("Research Note",),
        source_items=(
            StudySourceItem(
                source_type="note", source_id="note-1", label="Research Note"
            ),
        ),
    )

    async with _run_study_app(app, scope_context) as (screen, pilot):
        screen.query_one("#study-generate-source-pack", Button).press()
        await _wait_for_service_call(service, pilot, "get_study_pack_job_status")

        status = screen.query_one("#study-source-generation-status", Static)
        recent_decks = screen.query_one("#study-recent-decks", Static)
        resume_button = screen.query_one("#study-resume-last", Button)

        assert "ready" in _static_text(status).lower()
        assert "Research Note Study Pack" in _static_text(status)
        assert "deck 7" in _static_text(status)
        assert "Research Note Study Pack" in _static_text(recent_decks)
        assert resume_button.disabled is False
        assert "flashcards" in str(resume_button.label).lower()

        resume_button.press()
        await pilot.pause(0.2)

        assert screen.current_section == "flashcards"

    assert ("get_study_pack_job_status", "server", 42) in service.calls
    notifications.assert_any_call("Study pack ready.", severity="information")


@pytest.mark.asyncio
async def test_server_study_dashboard_keeps_failed_source_pack_generation_recoverable() -> (
    None
):
    service = RecordingSourceStudyService()
    service.study_pack_job_statuses = [
        {
            "job": {"id": 42, "status": "failed"},
            "error": "<b>Embedding service unavailable</b> javascript: onerror=retry",
        }
    ]
    app = _build_test_app()
    app.study_scope_service = service
    app.study_quiz_scope_service = DashboardQuizScopeService()
    _set_runtime_backend(app, "server")
    notifications = Mock(wraps=app.notify)
    app.notify = notifications
    scope_context = StudyScopeContext(
        material_source=MATERIAL_SOURCE_LIBRARY,
        material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
        material_titles=("Research Note",),
        source_items=(
            StudySourceItem(
                source_type="note", source_id="note-1", label="Research Note"
            ),
        ),
    )

    async with _run_study_app(app, scope_context) as (screen, pilot):
        screen.query_one("#study-generate-source-pack", Button).press()
        await _wait_for_service_call(service, pilot, "get_study_pack_job_status")

        status = screen.query_one("#study-source-generation-status", Static)
        generate_button = screen.query_one("#study-generate-source-pack", Button)

        assert "failed" in _static_text(status).lower()
        assert "Embedding service unavailable" in _static_text(status)
        assert "<b>" not in _static_text(status)
        assert "javascript:" not in _static_text(status)
        assert "onerror=" not in _static_text(status)
        assert generate_button.disabled is False
        assert "Retry source study-pack generation" in str(generate_button.tooltip)

    error_notifications = [
        call.args[0]
        for call in notifications.call_args_list
        if call.kwargs.get("severity") == "error"
    ]
    assert any(
        "Embedding service unavailable" in message for message in error_notifications
    )
    assert all("<b>" not in message for message in error_notifications)
    assert all("javascript:" not in message for message in error_notifications)
    assert all("onerror=" not in message for message in error_notifications)


@pytest.mark.asyncio
async def test_local_study_dashboard_explains_source_generation_server_requirement() -> (
    None
):
    service = RecordingSourceStudyService()
    app = _build_test_app()
    app.study_scope_service = service
    app.study_quiz_scope_service = DashboardQuizScopeService()
    _set_runtime_backend(app, "local")
    scope_context = StudyScopeContext(
        material_source=MATERIAL_SOURCE_LIBRARY,
        material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
        material_summary="Local Library source snapshot staged for Study.",
        material_titles=("Research Note",),
        source_items=(
            StudySourceItem(
                source_type="note", source_id="note-1", label="Research Note"
            ),
        ),
    )

    async with _run_study_app(app, scope_context) as (screen, _pilot):
        generate_button = screen.query_one("#study-generate-source-pack", Button)
        status = screen.query_one("#study-source-generation-status", Static)

        assert generate_button.disabled is True
        assert "server mode" in str(generate_button.tooltip).lower()
        assert "server mode" in _static_text(status).lower()

    assert not any(call[0] == "create_study_pack_job" for call in service.calls)
