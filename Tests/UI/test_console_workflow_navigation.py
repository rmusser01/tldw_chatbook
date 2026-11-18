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
                "OperationalError" in str(m)
                and "sqlite_code=1" in str(m)
                and "no such table" not in str(m)
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


def test_malformed_saved_objective_stays_out_of_console_diagnostics(tmp_path):
    run_workflow_ui(tmp_path, "saved-privacy")


async def check_private_saved_record(
    rig, host, console, composer, pilot, click, wait_until, workflow_id
):
    """Read a separate synthetic stopped record through the actual history/detail views."""
    import hashlib
    import json
    import time

    from loguru import logger
    from textual.widgets import Select

    marker = "orange-orchard-private-sample"
    archived = "malformed-stopped-record"
    with rig.service.db.transaction() as conn:
        original = conn.execute(
            "SELECT request_json FROM workflow_runs WHERE id=?", (workflow_id,)
        ).fetchone()[0]
        malformed = json.loads(original)
        malformed["objective"] = [marker]
        malformed[marker] = marker
        data = json.dumps(malformed)
        chat = rig.workflow.request.conversation_id
        chain = rig.service.db.automatic_work._create_chain(
            conn,
            chat,
            root_submission_id="workflow:" + archived,
            limits=rig.workflow.request.policy,
        )
        conn.execute(
            "INSERT INTO workflow_runs(id,launch_id,request_json,payload_hash,conversation_id,chain_id,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                archived,
                "archived-launch",
                data,
                hashlib.sha256(data.encode()).hexdigest(),
                chat,
                chain,
                "stopped",
                time.time(),
                time.time(),
            ),
        )
    warnings = []
    sink = logger.add(
        warnings.append,
        level="WARNING",
        format="{message}",
        filter="tldw_chatbook.UI.Console_Modules.workflows",
    )
    try:
        await pilot.press("escape")
        await wait_until(pilot, lambda: host.screen is console)
        composer.load_draft("/workflow")
        await click(pilot, "#console-send-message")
        await wait_until(
            pilot,
            lambda: (
                host.screen.query("#workflow-saved")
                and host.screen.query_one("#workflow-saved", Select).value == archived
            ),
        )
        await click(pilot, "#workflow-open")

        def unavailable():
            return host.screen.query(
                "#workflow-notice"
            ) and "Review unavailable" in str(
                host.screen.query_one("#workflow-notice").content
            )

        await wait_until(pilot, lambda: warnings and unavailable())
        count = len(warnings)
        await click(pilot, "#workflow-refresh")
        await wait_until(pilot, lambda: len(warnings) > count and unavailable())
        copy = str(host.screen.query_one("#workflow-notice").content)
        assert "ValidationError" in copy and "objective" in copy and marker not in copy
        assert any(
            "ValidationError" in str(m)
            and "objective" in str(m)
            and "string_type" in str(m)
            for m in warnings
        )
        assert all(
            marker not in str(m)
            and marker not in repr(m.record)
            and m.record["exception"] is None
            for m in warnings
        )
        assert all(
            host.screen.query_one("#workflow-" + action).disabled
            for action in ("approve", "accept", "pause", "stop", "resume", "revise")
        )
        await pilot.press("escape")
        await wait_until(pilot, lambda: host.screen is console)
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        with rig.service.db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT request_json FROM workflow_runs WHERE id=?", (workflow_id,)
                ).fetchone()[0]
                == original
            )
        print("WORKFLOW_UI_PRIVATE_SAVED_ROW_METADATA_ONLY", len(warnings), flush=True)
    finally:
        logger.remove(sink)


def test_workflow_diagnostic_omits_dynamic_input_location_context_and_chained_message():
    from loguru import logger
    from pydantic import BaseModel, ValidationError, field_validator
    from pydantic_core import PydanticCustomError

    from tldw_chatbook.UI.Console_Modules.workflows import workflow_view_error
    from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError

    marker = "orchard-private-note"

    class InvalidObjective(BaseModel):
        objective: str

        @field_validator("objective")
        @classmethod
        def invalid(cls, value):
            raise ValueError(value)

    with pytest.raises(ValidationError) as invalid:
        InvalidObjective(objective=marker)
    dynamic = ValidationError.from_exception_data(
        marker,
        [
            {
                "type": PydanticCustomError(marker, marker, {"reason": marker}),
                "loc": (marker,),
                "input": marker,
            }
        ],
    )
    outer = WorkspaceRegistryServiceError(marker)
    outer.__cause__ = invalid.value
    changed = "Displayed planning provider or model changed. Reopen setup."
    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        for context, error in (("Review", outer), (marker, dynamic)):
            assert marker not in workflow_view_error(context, error)
        assert changed in workflow_view_error("Prepare plan", PermissionError(changed))
        suppressed = workflow_view_error(
            "Prepare plan", PermissionError(f"{changed} {marker}")
        )
        assert changed not in suppressed and marker not in suppressed
    finally:
        logger.remove(sink)
    assert len(messages) == 4
    assert "WorkspaceRegistryServiceError" in str(
        messages[0]
    ) and "ValidationError" in str(messages[0])
    assert "objective" in str(messages[0]) and "value_error" in str(messages[0])
    assert all(
        marker not in str(m)
        and marker not in repr(m.record)
        and m.record["exception"] is None
        for m in messages
    )
