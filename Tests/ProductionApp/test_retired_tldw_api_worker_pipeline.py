from __future__ import annotations

import pytest
from textual.css.query import NoMatches
from textual.widgets import Input

import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_INGEST,
    TAB_CHAT,
    TAB_LIBRARY,
)
from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


class RecordingServerIngestService:
    """Record cancellation and settle the real production poller.

    Args:
        remote_job_id: Server job identifier returned by status polling.
    """

    def __init__(self, *, remote_job_id: str) -> None:
        self.remote_job_id = remote_job_id
        self.cancelled_batch_ids: list[str] = []
        self.listed_batch_ids: list[str] = []

    async def cancel_media_ingest_jobs_batch(self, *, batch_id: str) -> None:
        """Record a request made through the public batch-cancellation seam.

        Args:
            batch_id: Batch identifier submitted by the production Library screen.

        Returns:
            None.
        """
        self.cancelled_batch_ids.append(batch_id)

    async def list_media_ingest_jobs(
        self,
        batch_id: str,
        *,
        offset: int = 0,
    ) -> dict[str, object]:
        """Return a cancelled server job so the production poller can settle.

        Args:
            batch_id: Batch identifier requested by the production poller.
            offset: Pagination offset requested by the production poller.

        Returns:
            A server response containing the cancelled job and pagination state.
        """
        self.listed_batch_ids.append(batch_id)
        return {
            "jobs": [{"id": self.remote_job_id, "status": "cancelled"}],
            "has_more": False,
            "next_offset": offset,
        }


def _production_app(monkeypatch: pytest.MonkeyPatch) -> TldwCli:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)
    app = TldwCli()
    app.app_config["_first_run"] = False
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    return app


async def _wait_for_library_screen(
    app: TldwCli,
    pilot,
    *,
    previous_screen=None,
) -> LibraryScreen:
    for _ in range(400):
        if (
            type(app.screen) is LibraryScreen
            and app.current_tab == TAB_LIBRARY
            and app.screen is not previous_screen
        ):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError("production TldwCli did not route to the exact LibraryScreen")


async def _wait_for_selector(screen: LibraryScreen, pilot, selector: str):
    for _ in range(400):
        try:
            widget = screen.query_one(selector)
        except NoMatches:
            pass
        else:
            if widget.region.width > 0 and widget.region.height > 0:
                return widget
        await pilot.pause(0.01)
    raise AssertionError(f"production LibraryScreen did not render {selector}")


@pytest.mark.asyncio
async def test_production_ingest_alias_uses_library_owner_and_public_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert resolve_screen_target("ingest") == (
        "library",
        TAB_LIBRARY,
        LibraryScreen,
    )

    app = _production_app(monkeypatch)
    remote_job_id = "remote-task-905"
    batch_id = "batch-task-905"
    server_service = RecordingServerIngestService(remote_job_id=remote_job_id)
    app.server_media_reading_service = server_service
    app.REMOTE_INGEST_POLL_SECONDS = 0.01

    async with app.run_test(size=(180, 55)) as pilot:
        app.post_message(
            NavigateToScreen(
                "ingest",
                {LIBRARY_NAV_CONTEXT_INGEST: True},
            )
        )
        library = await _wait_for_library_screen(app, pilot)
        assert library._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA

        title_input = await _wait_for_selector(
            library,
            pilot,
            "#library-ingest-title",
        )
        assert type(title_input) is Input
        title_input.value = "TASK-905 transient form value"
        await pilot.pause()
        assert library._library_ingest_form.title == "TASK-905 transient form value"

        app.post_message(NavigateToScreen("chat"))
        for _ in range(400):
            if app.current_tab == TAB_CHAT and app.screen is not library:
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("production TldwCli did not leave Library")

        app.post_message(
            NavigateToScreen(
                "ingest",
                {LIBRARY_NAV_CONTEXT_INGEST: True},
            )
        )
        returned_library = await _wait_for_library_screen(
            app,
            pilot,
            previous_screen=library,
        )
        assert returned_library._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
        returned_title = await _wait_for_selector(
            returned_library,
            pilot,
            "#library-ingest-title",
        )
        assert type(returned_title) is Input
        assert returned_title.value == ""
        assert returned_library._library_ingest_form.title == ""

        job = app.library_ingest_jobs.submit(
            source_path="https://example.invalid/task-905.pdf",
            detected_type="pdf",
            origin="server",
        )
        attached = app.library_ingest_jobs.attach_remote(
            job.job_id,
            remote_job_id=remote_job_id,
            batch_id=batch_id,
        )
        assert attached is not None

        cancel_selector = f"#library-ingest-cancel-{job.job_id}"
        await _wait_for_selector(returned_library, pilot, cancel_selector)
        assert await pilot.click(cancel_selector)

        for _ in range(400):
            current = app.library_ingest_jobs.get_job(job.job_id)
            if (
                server_service.cancelled_batch_ids == [batch_id]
                and current is not None
                and current.state is IngestJobState.CANCELLED
            ):
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError(
                "production cancellation did not reach the server seam and settle "
                f"the real registry: calls={server_service.cancelled_batch_ids!r}, "
                f"job={app.library_ingest_jobs.get_job(job.job_id)!r}"
            )

        assert server_service.listed_batch_ids
        assert set(server_service.listed_batch_ids) == {batch_id}
