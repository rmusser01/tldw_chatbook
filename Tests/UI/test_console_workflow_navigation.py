"""Workflow history belongs to the selected Chat and never launches on selection."""

import pytest

from Tests.UI.test_console_workflow_command import run_workflow_ui

pytestmark = pytest.mark.bootstrap_profile


def test_narrow_history_reopens_same_run_after_detach(tmp_path):
    run_workflow_ui(tmp_path, "navigation")


def test_detach_preserves_planning_and_background_link_identifies_owner(tmp_path):
    run_workflow_ui(tmp_path, "detach")


def test_detach_waits_for_original_finite_workspace_read(tmp_path):
    run_workflow_ui(tmp_path, "read-detach")


def test_saved_history_paginates_without_starting_a_workflow(tmp_path):
    run_workflow_ui(tmp_path, "history-page")


def test_passive_default_setup_preserves_existing_stale_binding(tmp_path):
    run_workflow_ui(tmp_path, "default")


@pytest.mark.asyncio
async def test_history_sqlite_error_is_retryable_with_escape_and_original_diagnostic(
    tmp_path,
):
    import asyncio
    import sqlite3
    from contextlib import closing
    from types import SimpleNamespace

    from loguru import logger
    from textual.app import App

    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowHistory,
    )

    database = tmp_path / "history.sqlite"

    class HistoryService:
        def list_history(self, _chat, *, offset):
            with closing(sqlite3.connect(database)) as conn:
                conn.row_factory = sqlite3.Row
                return conn.execute(
                    "SELECT id,status FROM workflow_runs LIMIT 50 OFFSET ?", (offset,)
                ).fetchall()

        async def call(self, operation, *args, **kwargs):
            return await asyncio.to_thread(operation, *args, **kwargs)

    host = App()
    history = ConsoleWorkflowHistory(
        SimpleNamespace(service=HistoryService()), "chat", None, None
    )
    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        async with host.run_test() as pilot:
            await host.push_screen(history)
            await pilot.pause()
            assert host.screen is history and "History unavailable" in str(
                history.query_one("#workflow-page").content
            )
            assert any(
                "OperationalError" in str(m) and "no such table" in str(m)
                for m in messages
            )
            assert history.query_one("#workflow-open").disabled
            with closing(sqlite3.connect(database)) as conn:
                conn.execute("CREATE TABLE workflow_runs(id TEXT, status TEXT)")
                conn.execute("INSERT INTO workflow_runs VALUES('saved','completed')")
                conn.commit()
            assert await pilot.click("#workflow-retry")
            await pilot.pause()
            assert history.query_one("#workflow-saved").value == "saved"
            assert (
                not history.query_one("#workflow-open").disabled
                and not history.query_one("#workflow-retry").display
            )
            await pilot.press("escape")
            assert history not in host.screen_stack
    finally:
        logger.remove(sink)


def test_registry_storage_failure_leaves_console_usable_and_setup_can_retry(tmp_path):
    run_workflow_ui(tmp_path, "registry-error")
