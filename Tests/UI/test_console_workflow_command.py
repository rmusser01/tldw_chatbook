"""Workflow commands traverse the real Console composer without launching from history."""

import pytest

from tldw_chatbook.Chat.console_command_grammar import (
    CommandParse,
    default_console_registry,
)

pytestmark = pytest.mark.bootstrap_profile


def test_workflow_commands_are_not_dispatched_as_ordinary_prompts():
    assert default_console_registry().parse("/workflow new") == CommandParse(
        "command", "workflow", "new"
    )
    assert default_console_registry().parse("/workflow") == CommandParse(
        "command", "workflow", ""
    )


from Tests.Backup_Recovery.test_home_citation_retirement import _run


def run_workflow_ui(tmp_path, mode):
    script = """
import asyncio, os, sys
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
os.environ.update(TLDW_AGENTS_WORKFLOW_RUNS_ENABLED="true", TLDW_AGENTS_AUTOWAKE_ENABLED="true", OPENAI_API_KEY="private-native-workflow-test")
from Tests.UI.test_console_workflow_command import mounted_workflow
with user_fixture_default_owner():
    asyncio.run(mounted_workflow(Path(os.environ["XDG_DATA_HOME"]).parent, sys.argv[2]))
"""
    if mode in {"metadata", "unknown-task"}:
        from Tests.Chat.test_console_workflow_scheduling import (
            _expect_retained_planning_child,
        )

        _expect_retained_planning_child(
            (tmp_path, "workflow-ui", mode),
            {"script": script, "timeout": 60},
            f"WORKFLOW_UI_{mode.upper()}_REFUSED_ACTIONS",
            f"WORKFLOW_UI_{mode.upper()}_RETAINED_NO_CREATORS",
        )
    else:
        _run(tmp_path, "workflow-ui", mode, script=script, timeout=60)


def test_workflow_command_opens_setup_and_preserves_composer(tmp_path):
    run_workflow_ui(tmp_path, "command")


async def mounted_workflow(tmp_path, mode):
    import asyncio
    import json
    import os
    import sys
    import threading
    from dataclasses import replace
    from pathlib import Path

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

    async def wait_until(pilot, predicate, timeout=8):
        async with asyncio.timeout(timeout):
            while not predicate():
                await pilot.pause(0.03)

    from textual.widgets import Button, Input, Select, TextArea

    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Agents.agent_models import AgentDefinition
    from tldw_chatbook.Chat.attachment_core import PendingAttachment
    from tldw_chatbook.UI.console_command_provider import ConsoleCommandProvider
    from tldw_chatbook.Widgets.Console.console_workflow_setup import (
        ConsoleWorkflowSetup,
    )
    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowStatus,
    )

    held = mode in {"pause", "detach", "metadata"}

    class CaptureProvider(WorkflowProvider):
        def __call__(self, **kwargs):
            response = super().__call__(**kwargs)
            if self.calls[-1][0] == "task-1":
                # This is an external fixture edit during the actual phase window.
                (tmp_path / "project" / "observed.txt").write_text(
                    "external fixture edit\n"
                )
            return response

    failed_task = {
        "mixed-failure": "task-2",
        "clean-failure": "task-1",
        "unknown-task": "task-1",
    }.get(mode)

    class FailedTaskProvider(WorkflowProvider):
        def __call__(self, **kwargs):
            from tldw_chatbook.LLM_Calls.provider_outcomes import (
                PreEffectPermanentError,
            )

            response = super().__call__(**kwargs)
            if self.calls[-1][0] == failed_task:
                if mode == "unknown-task":
                    raise RuntimeError("Unknown fixture provider outcome")
                raise PreEffectPermanentError(
                    "Private provider refused before dispatch"
                )
            return response

    provider = (
        FailedTaskProvider()
        if failed_task
        else CaptureProvider()
        if mode in {"capture", "remove-cancel"}
        else WorkflowProvider(hold=("planning",) if held else ())
    )
    rig = await build_workflow_rig(
        tmp_path,
        None,
        provider,
        change_review=mode in {"capture", "remove-cancel"},
        target_models=("worker-a", "worker-b") if mode == "preset" else None,
    )
    if mode == "preset":
        rig.service.db.create_agent_definition(
            AgentDefinition(
                name="review-team",
                instructions="Only the target list is used.",
                provider="openai",
                model="worker-a",
                fallback_models=(("openai", "worker-b"),),
            )
        )
    if mode == "mixed-failure":
        from copy import deepcopy

        sibling = deepcopy(provider.plan["phases"][0]["tasks"][0])
        sibling.update(
            id="task-2", instructions="Execute task-2", dependencies=["task-1"]
        )
        provider.plan["phases"][0]["tasks"].append(sibling)
    if mode == "two-phase":
        from copy import deepcopy

        second = deepcopy(provider.plan["phases"][0])
        second["id"] = "phase-2"
        second["tasks"][0].update(
            id="task-2", instructions="Execute task-2", dependencies=["task-1"]
        )
        provider.plan["phases"].append(second)
    if mode == "history-page":
        for i in range(51):
            saved = rig.service.db.workflow_runs.create(
                rig.workflow.request, launch_id=f"saved-{i}"
            )
            rig.service.db.workflow_runs.set_status(
                saved.id,
                status="stopped",
                expected_control_revision=saved.control_revision,
            )
        old_row = await rig.service.call(
            rig.service.list_history, rig.workflow.request.conversation_id, offset=50
        )
        assert len(old_row) == 1
    if mode == "stale":
        (tmp_path / "project" / "input.txt").write_text("original reviewed input")
        provider.plan["phases"][0]["tasks"][0].update(
            inputs=["input.txt"], footprint=[{"path": "input.txt", "access": "read"}]
        )

    capacity = rig.runtime.execution_capacity
    # Match the mounted workspace reconciler's actual selection to the fixture
    # chat; selecting only its tab leaves the registry on Default and races a
    # native reconciliation that opens a different blank chat.
    rig.controller.app.workspace_registry_service.set_active_workspace("w")
    host = ConsoleHarness(rig.controller.app)

    async def click(pilot, widget_id):
        button = host.screen.query_one(widget_id, Button)
        button.scroll_visible(animate=False)

        def hit():
            if (
                not button.is_on_screen
                or button.disabled
                or not button.visible
                or not button.region.area
            ):
                return False
            x, y = button.region.center
            if not host.screen.region.contains(int(x), int(y)):
                return False
            target, _ = host.screen.get_widget_at(int(x), int(y))
            return target is button or button in target.ancestors

        await wait_until(pilot, hit)
        rect = button.region
        assert await pilot.click(widget_id, offset=(rect.width // 2, rect.height // 2))
        print("WORKFLOW_UI_CLICK_TRUE", widget_id, rect, flush=True)
        return button

    def capture(name):
        target = os.environ.get("TLDW_WORKFLOW_UI_CAPTURE_DIR")
        removal = mode in {"remove-cancel", "remove-small-cancel"}
        if removal and not target:
            target = tmp_path / "removal-ui"
        if target and (
            removal
            or mode in {"command", "controls", "navigation"}
            or name == "phase-handoff"
        ):
            folder = Path(target)
            folder.mkdir(exist_ok=True, parents=True)
            key = f"{name}-{host.size.width}x{host.size.height}"
            host.save_screenshot(filename=key + ".svg", path=str(folder))
            data = {
                b.id: {
                    "region": list(b.region),
                    "visible": b.visible and b.display,
                    "disabled": b.disabled,
                }
                for b in host.screen.query(Button)
                if b.id
            }
            (folder / (key + ".json")).write_text(json.dumps(data, indent=2))

    try:
        if mode == "default":
            from Tests.Workspaces.test_workspace_registry_service import (
                _insert_stale_default_binding,
            )
            from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID

            registry = rig.service.persistence.workspace_registry
            registry.ensure_default_workspace()
            _insert_stale_default_binding(registry.db, "stale-default")

            def stale_rows():
                with registry.db.connection() as conn:
                    return [
                        tuple(row)
                        for row in conn.execute(
                            "SELECT * FROM workspace_runtime_bindings WHERE binding_id='stale-default'"
                        )
                    ]

            before = stale_rows()
            assert len(before) == 1
            for workspace_id in (DEFAULT_WORKSPACE_ID, f" {DEFAULT_WORKSPACE_ID} "):
                assert await rig.service.call(
                    rig.service.setup_choices, workspace_id
                ) == ((), (), None)
                assert stale_rows() == before and not provider.calls
            print("WORKFLOW_DEFAULT_STALE_ROW_RETAINED", flush=True)
            return
        async with host.run_test(
            size=(80, 24)
            if mode == "navigation" or mode.startswith("remove-small-")
            else (140, 42)
        ) as pilot:
            console = host.screen
            await pilot.pause()
            # Current Console mount may schedule a coalesced control-bar replay.
            # Select the fixture chat only after that original warmup returns.
            await wait_until(pilot, lambda: not console._console_sync_in_progress and not getattr(console, "_console_control_bar_replay_whole_sync", False))
            assert console._console_runtime() is rig.runtime
            owner = next(s for s in rig.store.sessions() if s.workspace_id == "w")
            await wait_until(
                pilot, lambda: bool(console.query("#console-session-tab-" + owner.id))
            )
            if rig.store.active_session_id != owner.id:
                await click(pilot, "#console-session-tab-" + owner.id)
                await wait_until(pilot, lambda: rig.store.active_session_id == owner.id)
            await wait_until(pilot, lambda: console._console_visible_draft_session_id == owner.id and not console._console_sync_in_progress and not getattr(console, "_console_control_bar_replay_whole_sync", False))
            if mode == "detach":
                other = rig.store.create_session(
                    title="Other workflow chat",
                    workspace_id="w",
                    settings=owner.settings,
                    activate=False,
                    project_instruction_state=owner.project_instruction_state,
                )
                rig.store.persist_session_if_needed(other.id)
                await console._sync_native_console_chat_ui()
                await wait_until(
                    pilot,
                    lambda: bool(console.query("#console-session-tab-" + other.id)),
                )
                assert rig.store.active_session_id == owner.id
            composer = console.query_one("#console-native-composer")
            if mode == "remove-producer":
                from tldw_chatbook.Chat.console_provider_gateway import (
                    _automatic_generation,
                )

                entered, release = threading.Event(), threading.Event()
                original = _automatic_generation.__wrapped__.__code__

                monitor = sys.monitoring
                tool = next(i for i in range(6) if monitor.get_tool(i) is None)

                def hold_settled_planning_usage(code, offset, value):
                    frame = sys._getframe(1)
                    if (
                        frame.f_locals.get("completed")
                        and frame.f_locals.get("actual") == 10
                    ):
                        entered.set()
                        assert release.wait(10)

                monitor.use_tool_id(tool, "original-planning-usage")
                monitor.register_callback(
                    tool, monitor.events.PY_RETURN, hold_settled_planning_usage
                )
                monitor.set_local_events(tool, original, monitor.events.PY_RETURN)
                try:
                    started = rig.workflow = await rig.coordinator.start(
                        rig.workflow.request, launch_id=rig.workflow.launch_id
                    )
                    await wait_until(pilot, entered.is_set)
                    run = rig.coordinator._runs[started.id]
                    stopped = await rig.coordinator.stop(
                        started.id, expected_control_revision=started.control_revision
                    )
                    assert (
                        stopped.status == "stopping"
                        and rig.coordinator._runs[started.id] is run
                    )
                    revision = stopped.control_revision
                    chat = started.request.conversation_id

                    async def remove(key, identity=started.id):
                        return await rig.coordinator.remove_payloads(
                            identity,
                            conversation_id=chat,
                            expected_control_revision=revision,
                            command_id=key,
                        )

                    with pytest.raises(
                        PermissionError, match="original producer unavailable"
                    ):
                        await remove("held-consent")
                finally:
                    release.set()
                    monitor.set_local_events(tool, original, 0)
                    monitor.register_callback(tool, monitor.events.PY_RETURN, None)
                    monitor.free_tool_id(tool)
                await asyncio.shield(run.driver)
                assert (
                    started.id not in rig.coordinator._runs
                    and not rig.runtime.execution_capacity.snapshot().executions
                )
                settled = await rig.coordinator.get(started.id)
                assert (settled.status, settled.control_revision) == (
                    "stopped",
                    revision,
                )
                with pytest.raises(
                    PermissionError, match="original producer unavailable"
                ):
                    await remove("held-consent")
                with pytest.raises(ValueError, match="workflow_command_conflict"):
                    await remove("held-consent", "another-row")
                assert (
                    await rig.coordinator.get(started.id)
                ).control_revision == revision
                assert (await remove("fresh-consent")).status == "removed"
                return
            if mode.startswith("remove-"):
                from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

                rig.workflow = await rig.coordinator.start(
                    rig.workflow.request, launch_id=rig.workflow.launch_id
                )
                await rig.wait_for_status("awaiting_phase_review")
                for prepare, target in (
                    (rig.service.prepare_decision, "awaiting_result_review"),
                    (rig.service.prepare_quality_review, "completed"),
                ):
                    decision = await rig.service.call(
                        prepare, rig.workflow.id, phase_id="phase-1"
                    )
                    await rig.coordinator.approve(decision, command_id=target)
                    await rig.wait_for_status(target, timeout=20)

                def rows(table):
                    with rig.service.db.connection() as conn:
                        return tuple(
                            tuple(row) for row in conn.execute(f"SELECT * FROM {table}")
                        )

                async def history():
                    if host.screen is not console:
                        await pilot.press("escape")
                        await wait_until(pilot, lambda: host.screen is console)
                    composer.load_draft("/workflow")
                    await wait_until(
                        pilot, lambda: rig.store.session_draft(owner.id) == "/workflow"
                    )
                    await click(pilot, "#console-send-message")
                    await wait_until(
                        pilot,
                        lambda: (
                            bool(host.screen.query("#workflow-history"))
                            and host.screen.query_one("#workflow-saved", Select).value
                            == workflow_id
                        ),
                    )
                    await click(pilot, "#workflow-open")
                    await wait_until(
                        pilot,
                        lambda: (
                            isinstance(host.screen, ConsoleWorkflowStatus)
                            and host.screen.snapshot is not None
                        ),
                    )
                    return host.screen

                workflow_id = rig.workflow.id
                detail = await history()
                capture("saved-workflow-removal")
                calls = list(provider.calls)
                evidence = (
                    "workflow_results",
                    "workflow_checkpoints",
                    "workflow_payload_reservations",
                    "workflow_runs",
                    "workflow_change_windows",
                    "workflow_change_window_refs",
                )
                retained = (
                    "agent_runs",
                    "agent_run_steps",
                    "workflow_plan_revisions",
                    "workflow_decisions",
                    "workflow_task_attempts",
                    "automatic_work_reservations",
                    "change_snapshots",
                )
                before = {table: rows(table) for table in evidence + retained}
                assert before["workflow_results"] and before["workflow_checkpoints"]
                if mode == "remove-cancel":
                    assert (
                        before["change_snapshots"]
                        and before["workflow_change_window_refs"]
                    )
                scenario = mode.split("-")[-1]
                guard = scenario in {
                    "crosschat",
                    "dispose",
                    "superseded",
                    "covered",
                    "nonboolean",
                }

                async def cover_final_removal_read():
                    from textual.screen import ModalScreen

                    from tldw_chatbook.Agents.workflow_run_service import (
                        WorkflowRunService,
                    )

                    cover = ModalScreen()
                    entered, release = threading.Event(), threading.Event()
                    read = WorkflowRunService.get.__wrapped__.__code__
                    confirm_read = ConsoleWorkflowStatus._confirm_removal.__code__
                    original_reads = detail.reads.read.__func__.__code__
                    final_read_task = []

                    def observe(frame, event, value):
                        if (
                            event == "return"
                            and frame.f_code is original_reads
                            and frame.f_back.f_code is confirm_read
                        ):
                            task = frame.f_locals.get("task")
                            if task is not None:
                                final_read_task.append(task)
                        if (
                            event == "call"
                            and frame.f_code is read
                            and frame.f_locals.get("workflow_id") == workflow_id
                        ):
                            cell = frame.f_locals.get("_accounting")
                            if cell is not None and cell.issuing in final_read_task:
                                entered.set()
                                assert release.wait(8)

                    prior_commands = tuple(rig.coordinator._commands)
                    threading.setprofile_all_threads(observe)
                    try:
                        await click(pilot, "#confirm-button")
                        await wait_until(pilot, entered.is_set)
                        assert host.screen is detail and detail._sending
                        await host.push_screen(cover)
                    finally:
                        release.set()
                        threading.setprofile_all_threads(None)
                    await wait_until(pilot, lambda: not detail._sending)
                    assert host.screen is cover
                    assert tuple(rig.coordinator._commands) == prior_commands
                    await host.pop_screen()

                for answer in (
                    (scenario,)
                    if guard or scenario == "held"
                    else (scenario, "confirm")
                ):
                    assert detail.query("#workflow-remove"), (
                        "selected terminal workflow lacks saved-result removal"
                    )
                    remove = detail.query_one("#workflow-remove", Button)
                    remove.focus()
                    await pilot.pause()
                    await click(pilot, "#workflow-remove")
                    await wait_until(
                        pilot, lambda: isinstance(host.screen, ConfirmationDialog)
                    )
                    assert host.screen.focused.id == "cancel-button"
                    assert workflow_id in host.screen.message
                    confirm_button = host.screen.query_one("#confirm-button", Button)
                    cancel_button = host.screen.query_one("#cancel-button", Button)
                    assert not confirm_button.region.overlaps(cancel_button.region)
                    capture("saved-workflow-consent")
                    if guard:
                        dialog = host.screen
                        if answer == "crosschat":
                            other = rig.store.create_session(
                                title="Another saved chat",
                                workspace_id="w",
                                settings=owner.settings,
                            )
                            rig.store.persist_session_if_needed(other.id)
                            rig.store.switch_session(other.id)
                        elif answer == "dispose":
                            await rig.runtime.dispose()
                        elif answer == "superseded":
                            # Simulate a later durable writer between consent and read.
                            with rig.service.db.transaction() as conn:
                                conn.execute(
                                    "UPDATE workflow_runs SET control_revision=control_revision+1 WHERE id=?",
                                    (workflow_id,),
                                )
                            before["workflow_runs"] = rows("workflow_runs")
                        elif answer == "covered":
                            await cover_final_removal_read()
                        if answer == "nonboolean":
                            dialog.dismiss(1)
                        elif answer != "covered":
                            await click(pilot, "#confirm-button")
                        await wait_until(pilot, lambda: not detail._sending)
                        assert {
                            table: rows(table) for table in evidence + retained
                        } == before
                        assert provider.calls == calls
                        return

                    if answer == "held":
                        from tldw_chatbook.DB.workflow_runs import WorkflowRunsStore

                        entered, release = threading.Event(), threading.Event()
                        monitor = sys.monitoring
                        marker = WorkflowRunsStore._required_results.__code__
                        original = WorkflowRunsStore.remove_payloads.__code__

                        def hold(
                            code,
                            offset,
                            entered=entered,
                            release=release,
                            original=original,
                        ):
                            caller = sys._getframe(1).f_back
                            if (
                                caller
                                and caller.f_code is original
                                and caller.f_locals["workflow_id"] == workflow_id
                            ):
                                entered.set()
                                assert release.wait(8)

                        tool = next(i for i in range(6) if monitor.get_tool(i) is None)
                        monitor.use_tool_id(tool, "workflow-removal-owner")
                        monitor.register_callback(tool, monitor.events.PY_START, hold)
                        monitor.set_local_events(tool, marker, monitor.events.PY_START)
                        try:
                            await click(pilot, "#confirm-button")
                            await wait_until(pilot, entered.is_set, timeout=4)
                            responsive = asyncio.Event()
                            host.call_later(responsive.set)
                            await asyncio.wait_for(responsive.wait(), 1)
                            assert (
                                rows("workflow_results") == before["workflow_results"]
                            )
                        finally:
                            release.set()
                            monitor.set_local_events(tool, marker, 0)
                            monitor.register_callback(
                                tool, monitor.events.PY_START, None
                            )
                            monitor.free_tool_id(tool)
                        answer = "confirmed-held"
                    elif answer == "retry":
                        with rig.service.db.transaction() as conn:
                            conn.execute(
                                "UPDATE automatic_work_chains SET status='review_required' WHERE id=?",
                                (detail.snapshot.chain_id,),
                            )
                        try:
                            await click(pilot, "#confirm-button")
                            await wait_until(pilot, lambda: not detail._sending)
                            assert (
                                rows("workflow_results") == before["workflow_results"]
                            )
                            (prior_key,) = (
                                key
                                for key in rig.coordinator._commands
                                if key[0] == "remove"
                            )
                        finally:
                            with rig.service.db.transaction() as conn:
                                conn.execute(
                                    "UPDATE automatic_work_chains SET status='active' WHERE id=?",
                                    (detail.snapshot.chain_id,),
                                )
                        with pytest.raises(
                            ValueError, match="workflow_has_active_evidence"
                        ):
                            await rig.coordinator.remove_payloads(
                                workflow_id,
                                conversation_id=detail.conversation_id,
                                expected_control_revision=detail.snapshot.control_revision,
                                command_id=prior_key[1],
                            )
                        detail.request_refresh()
                        await wait_until(pilot, lambda: detail._removable)
                        continue
                    elif answer in {"cancel", "confirm"}:
                        await click(pilot, "#" + answer + "-button")
                    else:
                        await pilot.press(answer)
                    await wait_until(
                        pilot, lambda: host.screen is detail and not detail._sending
                    )
                    if answer not in {"confirm", "confirmed-held"}:
                        assert {
                            table: rows(table) for table in evidence + retained
                        } == before
                await wait_until(pilot, lambda: detail.snapshot.status == "removed")
                assert not rows("workflow_results") and not rows("workflow_checkpoints")
                assert rows("workflow_payload_reservations")[0][-1] == "removed"
                assert not rows("workflow_change_window_refs")
                with rig.service.db.connection() as conn:
                    assert not conn.execute(
                        "SELECT 1 FROM workflow_change_windows WHERE state!='removed' OR roots_json!='[]' OR snapshot_count!=0 OR reserved_bytes!=0"
                    ).fetchone()
                assert {table: rows(table) for table in retained} == {
                    table: before[table] for table in retained
                }
                reopened = await history()
                assert reopened.snapshot.status == "removed"
                remove = reopened.query_one("#workflow-remove", Button)
                assert remove.disabled
                await reopened.on_button_pressed(Button.Pressed(remove))
                assert host.screen is reopened and not reopened._sending
                if scenario == "retry":
                    entries = [
                        (key, cell)
                        for key, cell in rig.coordinator._commands.items()
                        if key[0] == "remove"
                    ]
                    assert len(entries) == 2 and entries[0][0] != entries[1][0]
                    key, (identity, done) = entries[1]
                    assert (
                        await rig.coordinator.remove_payloads(
                            workflow_id,
                            conversation_id=identity[1],
                            expected_control_revision=identity[2],
                            command_id=key[1],
                        )
                        is done.result()
                    )
                    other = rig.store.create_session(
                        title="Different saved chat",
                        workspace_id="w",
                        settings=owner.settings,
                    )
                    rig.store.persist_session_if_needed(other.id)
                    rig.store.switch_session(other.id)
                    with pytest.raises(PermissionError, match="another chat"):
                        await rig.coordinator.remove_payloads(
                            workflow_id,
                            conversation_id=identity[1],
                            expected_control_revision=identity[2],
                            command_id=key[1],
                        )
                    assert (
                        len([k for k in rig.coordinator._commands if k[0] == "remove"])
                        == 2
                    )
                assert provider.calls == calls
                return
            if mode == "registry-error":
                from loguru import logger

                registry = rig.service.persistence.workspace_registry
                diagnostics = []
                sink = logger.add(
                    diagnostics.append, level="WARNING", format="{message}"
                )
                with registry.db.transaction() as conn:
                    conn.execute(
                        "ALTER TABLE workspace_runtime_bindings RENAME TO fixture_hidden_bindings"
                    )
                try:
                    console._workflows.command("new")
                    await wait_until(
                        pilot,
                        lambda: any(
                            "Setup unavailable" in n.message
                            and "/workflow new" in n.message
                            for n in host._notifications
                        ),
                    )
                    assert host.screen is console and not provider.calls
                    assert any(
                        "WorkspaceRegistryServiceError" in str(m)
                        and "OperationalError" in str(m)
                        and "sqlite_code=1" in str(m)
                        and "no such table" not in str(m)
                        for m in diagnostics
                    )
                finally:
                    logger.remove(sink)
                    with registry.db.transaction() as conn:
                        conn.execute(
                            "ALTER TABLE fixture_hidden_bindings RENAME TO workspace_runtime_bindings"
                        )
                console._workflows.command("new")
                await wait_until(
                    pilot, lambda: isinstance(host.screen, ConsoleWorkflowSetup)
                )
                await pilot.press("escape")
                await wait_until(pilot, lambda: host.screen is console)
                assert not provider.calls
                print("WORKFLOW_UI_REGISTRY_REFUSAL_RECOVERED", flush=True)
                return
            if mode == "history-page":
                palette = ConsoleCommandProvider(console)
                hits = [h async for h in palette.discover()]
                next(h for h in hits if "Workflow runs" in h.text).command()
                await wait_until(
                    pilot,
                    lambda: (
                        bool(host.screen.query("#workflow-next"))
                        and not host.screen.query_one("#workflow-next", Button).disabled
                    ),
                )
                await click(pilot, "#workflow-next")
                await wait_until(
                    pilot,
                    lambda: (
                        host.screen.query_one("#workflow-saved", Select).value
                        == old_row[0]["id"]
                    ),
                )
                await click(pilot, "#workflow-open")
                await wait_until(
                    pilot,
                    lambda: (
                        isinstance(host.screen, ConsoleWorkflowStatus)
                        and host.screen.snapshot
                    ),
                )
                assert (
                    host.screen.snapshot.id == old_row[0]["id"]
                    and host.screen.snapshot.status == "stopped"
                )
                assert (
                    not provider.calls
                    and host.screen.snapshot.accounting.used["generation"] == 0
                )
                return
            if mode == "command":
                attachment = PendingAttachment(
                    str(tmp_path / "brief.txt"),
                    "brief.txt",
                    "text",
                    "attachment",
                    text_content="keep draft",
                )
                assert rig.store.add_pending_attachment(owner.id, attachment)
                composer.load_draft("Unrelated draft")
                palette = ConsoleCommandProvider(console)
                hits = [h async for h in palette.discover()]
                next(h for h in hits if "Workflow runs" in h.text).command()
                await wait_until(
                    pilot, lambda: bool(host.screen.query("#workflow-history"))
                )
                assert composer.draft_text() == "Unrelated draft"
                assert rig.store.pending_attachments(owner.id) == [attachment]
                await pilot.press("escape")
            if mode == "command":
                from tldw_chatbook.Widgets.Console.console_command_popup import (
                    ConsoleCommandPopup,
                )

                composer.load_draft("")
                console._focus_console_composer_if_needed(force=True)
                await pilot.press("/", "w", "o", "r", "k", "f")
                popup = console.query_one(ConsoleCommandPopup)
                await wait_until(
                    pilot,
                    lambda: (
                        popup.is_open
                        and [s.label for s in popup._suggestions] == ["/workflow"]
                    ),
                )
                await pilot.press("tab")
                await wait_until(pilot, lambda: composer.draft_text() == "/workflow ")
                await pilot.press("n", "e", "w", "enter")
            else:
                composer.load_draft("/workflow new")
                await wait_until(pilot, lambda: rig.store.session_draft(owner.id) == "/workflow new")
                await click(pilot, "#console-send-message")
            await pilot.pause(0.25)
            try:
                await wait_until(
                    pilot,
                    lambda: (
                        isinstance(host.screen, ConsoleWorkflowSetup)
                        and bool(host.screen.query("#workflow-objective"))
                    ),
                )
            except TimeoutError:
                print("WORKFLOW_SETUP_TIMEOUT", type(host.screen).__name__, repr(composer.draft_text()), rig.store.active_session_id, owner.id, console._console_visible_draft_session_id, repr(rig.controller.app._notifications), [(m.role, m.content) for m in rig.store.messages_for_session(owner.id)], flush=True)
                raise
            assert composer.draft_text() == "/workflow new"
            assert not rig.provider.calls
            capture("setup")
            if mode == "command":
                assert rig.store.pending_attachments(owner.id) == [attachment]
                modal = host.screen
                modal.query_one("#workflow-objective", TextArea).load_text(
                    "Check settings without launching"
                )
                field = modal.query_one("#workflow-limit-generations", Input)
                field.value = str(modal.policy.generations + 1)
                await wait_until(
                    pilot,
                    lambda: (
                        modal.query_one("#workflow-plan", Button).disabled
                        and "between"
                        in str(modal.query_one("#workflow-notice").content)
                    ),
                )
                field.value = "1"
                await wait_until(
                    pilot,
                    lambda: not modal.query_one("#workflow-plan", Button).disabled,
                )
                from loguru import logger

                marker = "long-orchard-objective"
                warnings = []
                sink = logger.add(
                    warnings.append,
                    level="WARNING",
                    format="{message}",
                    filter="tldw_chatbook.UI.Console_Modules.workflows",
                )
                try:
                    modal.query_one("#workflow-objective", TextArea).load_text(
                        marker + "x" * 8192
                    )
                    await click(pilot, "#workflow-plan")
                    await wait_until(
                        pilot,
                        lambda: (
                            warnings
                            and "objective/value_error"
                            in str(modal.query_one("#workflow-notice").content)
                        ),
                    )
                    assert marker not in str(
                        modal.query_one("#workflow-notice").content
                    )
                    assert all(
                        marker not in str(m) and marker not in repr(m.record)
                        for m in warnings
                    )
                    assert (
                        not provider.calls
                        and not rig.service.db.workflow_runs.list_for_conversation(
                            rig.workflow.request.conversation_id
                        )
                    )
                    modal.query_one("#workflow-objective", TextArea).load_text(
                        "Review the same form"
                    )
                    await pilot.pause()
                    assert not modal.query_one("#workflow-plan", Button).disabled
                    print("WORKFLOW_UI_PRIVATE_SETUP_FIELDS_ONLY", flush=True)
                finally:
                    logger.remove(sink)
                await pilot.press("tab")
                assert host.screen.focused is not host.screen.query_one(
                    "#workflow-plan"
                )
                await pilot.press("escape")
                await wait_until(pilot, lambda: host.screen is console)
                return
            host.screen.query_one("#workflow-objective", TextArea).load_text(
                rig.workflow.request.objective
            )
            if mode == "preset":
                host.screen.query_one("#workflow-target", Select).value = "review-team"
                await pilot.pause()
                assert "worker-b" in str(
                    host.screen.query_one("#workflow-targets").content
                )
            if mode == "malformed":
                from tldw_chatbook.Agents.workflow_models import WorkflowSetup

                co = rig.coordinator
                request = rig.workflow.request
                setup_value = WorkflowSetup(
                    conversation_id=request.conversation_id,
                    objective=request.objective,
                    binding=request.binding,
                    policy=request.policy,
                    planner=("openai", "test-model"),
                )
                forged = setup_value.model_copy(
                    update={"targets": tuple(("openai", str(i)) for i in range(17))}
                )
                with pytest.raises(ValueError):
                    await co.start(forged, launch_id="oversize-setup")
                forged_request = request.model_copy(
                    update={"targets": (("openai", "test-model"),)}
                )
                with pytest.raises(
                    PermissionError, match="original frozen workflow target required"
                ):
                    await co.start(forged_request, launch_id="malformed-frozen")
                assert not rig.service.db.workflow_runs.list_for_conversation(
                    request.conversation_id
                )
                assert not provider.calls and not co._pending_planning and not co._runs
                return
            if mode == "selection":
                rig.store.replace_session_settings(
                    owner.id, replace(owner.settings, model="changed-model")
                )
            setup = host.screen
            await click(pilot, "#workflow-plan")
            if mode == "selection":
                await wait_until(
                    pilot,
                    lambda: (
                        "changed" in str(setup.query_one("#workflow-notice").content)
                    ),
                )
                assert host.screen is setup and not provider.calls
                assert not rig.service.db.workflow_runs.list_for_conversation(
                    rig.workflow.request.conversation_id
                )
                return
            await wait_until(
                pilot,
                lambda: (
                    isinstance(host.screen, ConsoleWorkflowStatus)
                    and host.screen.snapshot is not None
                ),
                timeout=15,
            )
            status = host.screen
            if held:
                await rig.wait_for_provider_entry()
                assert status.snapshot.status == "planning"
                disabled_remove = status.query_one("#workflow-remove", Button)
                assert disabled_remove.disabled
                await status.on_button_pressed(Button.Pressed(disabled_remove))
                assert host.screen is status and not status._sending
                workflow_id = status.workflow_id
                rig.workflow = await rig.coordinator.get(workflow_id)
                if mode == "metadata":
                    # Intentional exception in the native terminal publication, before any subscriber is called.
                    code = type(rig.coordinator)._publish.__code__
                    fault = RuntimeError("fixture failed terminal publication")
                    failed = []

                    def inject(frame, event, value):
                        if frame.f_code is code and event == "call" and not failed:
                            run = frame.f_locals["_terminal_run"]
                            if run is not None and run.workflow_id == workflow_id:
                                assert run.prepared._planning_release_return is not None
                                failed.append(run)
                                raise fault

                    assert sys.getprofile() is None
                    sys.setprofile(inject)
                    try:
                        provider.released.set()
                        await wait_until(
                            pilot, lambda: failed and failed[0].driver.done()
                        )
                    finally:
                        sys.setprofile(None)
                    assert failed[0].status == "recovery_required"
                    await click(pilot, "#workflow-refresh")
                    await wait_until(
                        pilot, lambda: status.snapshot.status == "recovery_required"
                    )
                    assert status.decision is None and all(
                        status.query_one("#workflow-" + name, Button).disabled
                        for name in (
                            "approve",
                            "accept",
                            "resume",
                            "revise",
                            "pause",
                            "stop",
                        )
                    )
                    assert not status.snapshot.results
                    disabled_remove = status.query_one("#workflow-remove", Button)
                    assert disabled_remove.disabled
                    await status.on_button_pressed(Button.Pressed(disabled_remove))
                    assert host.screen is status and not status._sending
                    print("WORKFLOW_UI_METADATA_REFUSED_ACTIONS", flush=True)
                    return
                if mode == "pause":
                    await click(pilot, "#workflow-pause")
                    await wait_until(pilot, lambda: status.snapshot.status == "pausing")
                    provider.released.set()
                    await wait_until(pilot, lambda: status.snapshot.status == "paused")
                    assert not status.query_one("#workflow-resume", Button).disabled
                    assert status.query_one("#workflow-remove", Button).disabled
                    await status.on_button_pressed(
                        Button.Pressed(status.query_one("#workflow-remove", Button))
                    )
                    assert host.screen is status and not status._sending
                    await click(pilot, "#workflow-resume")
                    await wait_until(
                        pilot,
                        lambda: (
                            status.snapshot.status == "awaiting_phase_review"
                            and status.decision
                        ),
                    )
                    assert status.snapshot.accounting.used["child_launch"] == 0
                    await click(pilot, "#workflow-stop")
                    await wait_until(pilot, lambda: status.snapshot.status == "stopped")
                    return
                if mode == "detach":
                    await pilot.press("escape")
                    await wait_until(pilot, lambda: host.screen is console)
                    assert status._unsubscribe is None
                    await click(pilot, "#console-session-tab-" + other.id)
                    await wait_until(
                        pilot, lambda: rig.store.active_session_id == other.id
                    )
                    composer.load_draft("Keep the other chat draft")
                    provider.released.set()
                    await rig.wait_for_status("awaiting_phase_review", timeout=10)
                    line = console.query_one("#console-workflow-line")
                    await wait_until(
                        pilot,
                        lambda: (
                            line.target
                            == (rig.workflow.request.conversation_id, workflow_id)
                        ),
                    )
                    await click(pilot, "#console-workflow-line")
                    await wait_until(
                        pilot,
                        lambda: (
                            isinstance(host.screen, ConsoleWorkflowStatus)
                            and host.screen.snapshot
                        ),
                    )
                    assert (
                        host.screen.workflow_id == workflow_id
                        and rig.store.active_session_id == other.id
                    )
                    assert composer.draft_text() == "Keep the other chat draft"
                    await click(pilot, "#workflow-stop")
                    await wait_until(
                        pilot, lambda: host.screen.snapshot.status == "stopped"
                    )
                    await pilot.press("escape")
                    await wait_until(
                        pilot, lambda: host.screen is console and not line.display
                    )
                    await click(pilot, "#console-session-tab-" + owner.id)
                    await click(pilot, "#console-session-tab-" + other.id)
                    await pilot.pause()
                    assert (
                        not line.display
                        and composer.draft_text() == "Keep the other chat draft"
                    )
                    return
            await wait_until(pilot, lambda: status.decision is not None, timeout=15)
            assert status.focused is not status.query_one("#workflow-approve")
            workflow_id = status.workflow_id
            assert (
                len(
                    rig.service.db.workflow_runs.list_for_conversation(
                        rig.workflow.request.conversation_id
                    )
                )
                == 1
            )
            assert status.snapshot.accounting.used["child_launch"] == 0
            capture("phase-review")
            if mode == "preset":
                assert status.snapshot.request.targets == rig.workflow.request.targets
                assert (
                    await rig.coordinator.start(
                        setup._submitted, launch_id=setup.launch_id
                    )
                ).id == workflow_id
                assert [key for key, _ in provider.calls] == ["planning"]
                return
            if mode == "saved-privacy":
                from Tests.UI.test_console_workflow_navigation import (
                    check_private_saved_record,
                )

                await check_private_saved_record(
                    rig, host, console, composer, pilot, click, wait_until, workflow_id
                )
                return
            if mode == "stale":
                stale_decision = status.decision
                (tmp_path / "project" / "input.txt").write_text("changed after display")
                await click(pilot, "#workflow-approve")
                await wait_until(
                    pilot, lambda: bool(status._action_notice) and not status._sending
                )
                now = await rig.coordinator.get(workflow_id)
                assert (
                    now.status == "awaiting_phase_review"
                    and now.accounting.used["child_launch"] == 0
                )
                assert now.control_revision == status.snapshot.control_revision and [
                    key for key, _ in provider.calls
                ] == ["planning"]
                await wait_until(
                    pilot,
                    lambda: (
                        status.decision
                        and status.decision != stale_decision
                        and status.decision.evidence.ready
                        and not status._reading
                    ),
                )
                fresh = status.decision
                assert fresh.control_revision == stale_decision.control_revision
                assert (
                    fresh.evidence.input_manifest
                    != stale_decision.evidence.input_manifest
                )
                await click(pilot, "#workflow-approve")
                await wait_until(
                    pilot,
                    lambda: status.snapshot.status == "awaiting_result_review",
                    timeout=20,
                )
                full = await rig.coordinator.get(workflow_id)
                with rig.service.db.connection() as conn:
                    payloads = [
                        row[0]
                        for row in conn.execute(
                            "SELECT payload_json FROM workflow_decisions WHERE workflow_id=?",
                            (workflow_id,),
                        )
                    ]
                assert payloads == [fresh.canonical_json()]
                assert (
                    full.accounting.used["child_launch"] == 1
                    and full.accounting.used["generation"] == 2
                )
                assert [key for key, _ in provider.calls] == ["planning", "task-1"]
                receipts = {
                    identity: key[1]
                    for key, (identity, _) in rig.coordinator._commands.items()
                    if key[0] == "approve"
                }
                assert set(receipts) == {
                    stale_decision.canonical_json(),
                    fresh.canonical_json(),
                }
                assert (
                    receipts[fresh.canonical_json()]
                    != receipts[stale_decision.canonical_json()]
                )
                await rig.coordinator.approve(
                    fresh, command_id=receipts[fresh.canonical_json()]
                )
                assert (
                    await rig.coordinator.get(workflow_id)
                ).accounting.used == full.accounting.used
                print(
                    "WORKFLOW_FRESH_DECISION_ONE_APPROVAL",
                    fresh.id,
                    fresh.control_revision,
                    full.accounting.used,
                    flush=True,
                )
                return
            if mode == "revise":
                status.query_one(
                    "#workflow-instruction", Input
                ).value = "Revise the remaining work"
                await click(pilot, "#workflow-revise")
                await wait_until(
                    pilot, lambda: len(status.snapshot.plans) == 2, timeout=15
                )
                assert status.snapshot.accounting.used["generation"] == 2
                assert status.snapshot.accounting.used["child_launch"] == 0
                return
            if mode == "read-detach":
                from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
                from tldw_chatbook.Backup_Recovery import storage_admission as storage

                entered, released, operations = threading.Event(), threading.Event(), []
                monitor = sys.monitoring
                code = storage._repository_operation.__wrapped__.__code__
                review_code = WorkflowRunService.prepare_decision.__wrapped__.__code__

                def hold_original(code_, offset, value):
                    frame = sys._getframe(1)
                    op = frame.f_locals.get("operation")
                    if (
                        operations
                        or op is None
                        or op.participant.owner_id != "db.workspaces"
                    ):
                        return
                    parent = frame.f_back
                    while parent and parent.f_code is not review_code:
                        parent = parent.f_back
                    if parent is not None and parent.f_locals["self"] is rig.service:
                        operations.append(op)
                        entered.set()
                        assert released.wait(8), "original finite read was not released"

                index = next(i for i in range(6) if monitor.get_tool(i) is None)
                monitor.use_tool_id(index, "workflow-view-read-test")
                monitor.register_callback(index, monitor.events.PY_YIELD, hold_original)
                monitor.set_local_events(index, code, monitor.events.PY_YIELD)
                try:
                    await click(pilot, "#workflow-refresh")
                    await wait_until(pilot, entered.is_set, timeout=4)
                    closing = asyncio.create_task(pilot.press("escape"))
                    async with asyncio.timeout(4):
                        while not status.reads.closed:
                            await asyncio.sleep(0.01)
                    with storage._lock:
                        assert operations[0] in storage._operations
                    assert status.reads.tasks and status._unsubscribe is None
                    released.set()
                    await closing
                    await wait_until(
                        pilot, lambda: host.screen is console and not status.reads.tasks
                    )
                    with storage._lock:
                        assert operations[0] not in storage._operations
                    print("WORKFLOW_UI_ORIGINAL_REPOSITORY_READ_DRAINED", flush=True)
                finally:
                    released.set()
                    monitor.free_tool_id(index)
                return
            if mode == "detail-storage-error":
                with rig.service.db.transaction() as conn:
                    conn.execute(
                        "ALTER TABLE workflow_change_windows RENAME TO fixture_hidden_windows"
                    )
                try:
                    await click(pilot, "#workflow-refresh")
                    await wait_until(
                        pilot,
                        lambda: (
                            "OperationalError"
                            in str(status.query_one("#workflow-notice").content)
                            and "sqlite_code=1"
                            in str(status.query_one("#workflow-notice").content)
                        ),
                    )
                    assert host.screen is status and status.decision is None
                    assert all(
                        status.query_one("#workflow-" + name, Button).disabled
                        for name in (
                            "approve",
                            "accept",
                            "revise",
                            "resume",
                            "pause",
                            "stop",
                        )
                    )
                finally:
                    with rig.service.db.transaction() as conn:
                        conn.execute(
                            "ALTER TABLE fixture_hidden_windows RENAME TO workflow_change_windows"
                        )
                await click(pilot, "#workflow-refresh")
                await wait_until(
                    pilot,
                    lambda: (
                        status.decision is not None
                        and not status.query_one("#workflow-approve").disabled
                    ),
                )
                assert status.snapshot.accounting.used["child_launch"] == 0
                await pilot.press("escape")
                await wait_until(pilot, lambda: host.screen is console)
                print("WORKFLOW_UI_DETAIL_SQLITE_RETRY", flush=True)
                return
            if mode == "pending":
                rig.controller.add_pending_round(owner.id, "workflow-test-pending")
                try:
                    await click(pilot, "#workflow-refresh")
                    await wait_until(
                        pilot,
                        lambda: status.query_one("#workflow-pending", Button).display,
                    )
                    await click(pilot, "#workflow-pending")
                    await wait_until(pilot, lambda: host.screen is console)
                    assert rig.controller.pending_round_count(owner.id) == 1
                    assert (await rig.coordinator.get(workflow_id)).accounting.used[
                        "child_launch"
                    ] == 0
                finally:
                    rig.controller.discard_pending_round(
                        owner.id, "workflow-test-pending"
                    )
                return
            if mode == "navigation":
                await pilot.press("escape")
                await wait_until(pilot, lambda: host.screen is console)
                assert (
                    await rig.coordinator.get(workflow_id)
                ).status == "awaiting_phase_review"
                composer.load_draft("/workflow")
                await click(pilot, "#console-send-message")
                await wait_until(
                    pilot, lambda: bool(host.screen.query("#workflow-history"))
                )
                capture("history")
                select = host.screen.query_one("#workflow-saved", Select)
                assert select.value == workflow_id
                host.screen.query_one("#workflow-open", Button).focus()
                await pilot.press("enter")
                await wait_until(
                    pilot,
                    lambda: (
                        isinstance(host.screen, ConsoleWorkflowStatus)
                        and host.screen.snapshot is not None
                    ),
                )
                assert host.screen.workflow_id == workflow_id
                return
            if mode == "unknown-task":
                earlier_full = status.snapshot
                earlier_view = await rig.service.call(
                    rig.service.runtime_snapshot, workflow_id
                )
            first_rect = status.query_one("#workflow-approve", Button).region
            point = (
                first_rect.x + first_rect.width // 2,
                first_rect.y + first_rect.height // 2,
            )
            await click(pilot, "#workflow-approve")
            if failed_task:
                await wait_until(
                    pilot,
                    lambda: status.snapshot.status in {"paused", "recovery_required"},
                    timeout=20,
                )
                previous = status.snapshot
                if mode == "unknown-task":
                    assert previous.status == "recovery_required"
                    assert all(
                        status.query_one("#workflow-" + name, Button).disabled
                        for name in (
                            "approve",
                            "accept",
                            "revise",
                            "resume",
                            "pause",
                            "stop",
                        )
                    )
                    # The earlier coherent review must also refuse a now-uncertain native decision.
                    assert (
                        earlier_full.status == "awaiting_phase_review"
                        and not earlier_full.accounting.uncertain
                    )
                    with pytest.raises(
                        ValueError, match="Recovery required · uncertain_execution"
                    ):
                        await status._review(earlier_full, earlier_view)
                    print(
                        "WORKFLOW_UI_EARLIER_VIEW_REFUSED_ACTUAL_TYPED_RECOVERY",
                        flush=True,
                    )
                    print("WORKFLOW_UI_UNKNOWN-TASK_REFUSED_ACTIONS", flush=True)
                    return
                assert previous.status == "paused"
                assert [r.status for r in previous.results] == (
                    ["completed", "failed"] if mode == "mixed-failure" else ["failed"]
                )
                assert previous.accounting.used["child_launch"] == len(previous.results)
                assert not previous.accounting.uncertain
                assert (
                    status.query_one("#workflow-instruction").display
                    and not status.query_one("#workflow-revise", Button).disabled
                )
                if mode == "mixed-failure":
                    await click(pilot, "#workflow-resume")
                    await wait_until(
                        pilot,
                        lambda: (
                            status.snapshot.status == "awaiting_phase_review"
                            and not status._sending
                        ),
                    )
                    assert (
                        status.snapshot.control_revision
                        == previous.control_revision + 1
                    )
                assert status.snapshot.results == previous.results
                assert (
                    status.query_one("#workflow-approve", Button).disabled
                    and not status.query_one("#workflow-stop", Button).disabled
                )
                if mode == "mixed-failure":
                    provider.plan["phases"][0]["tasks"][-1]["instructions"] = (
                        "Inspect and repair the failed task"
                    )
                    status.query_one(
                        "#workflow-instruction", Input
                    ).value = "Preserve successful work and revise the failed task"
                    await click(pilot, "#workflow-revise")
                    await wait_until(
                        pilot,
                        lambda: (
                            len(status.snapshot.plans) == 2
                            and status.snapshot.status == "awaiting_phase_review"
                        ),
                        timeout=15,
                    )
                    revised = status.snapshot
                    assert (
                        previous.plans == revised.plans[:1]
                        and revised.results == previous.results
                    )
                    assert (
                        revised.accounting.used["generation"]
                        == previous.accounting.used["generation"] + 1
                    )
                    assert (
                        revised.accounting.used["child_launch"]
                        == previous.accounting.used["child_launch"]
                    )
                    assert [key for key, _ in provider.calls] == [
                        "planning",
                        "task-1",
                        "task-2",
                        "planning",
                    ]
                await click(pilot, "#workflow-stop")
                await wait_until(pilot, lambda: status.snapshot.status == "stopped")
                assert (
                    await rig.coordinator.get(workflow_id)
                ).results == previous.results
                print(
                    "WORKFLOW_UI_DRAINED_FAILURE",
                    mode,
                    status.snapshot.accounting.used,
                    flush=True,
                )
                return
            # Repeat the physical point, even if the first click has already changed the visible controls.
            assert await pilot.click(offset=point)
            await wait_until(
                pilot,
                lambda: host.screen.snapshot.status == "awaiting_result_review",
                timeout=20,
            )
            await wait_until(pilot, lambda: host.screen.checkpoint is not None)
            capture("result-review")
            if mode == "capture":
                assert status.windows and all(
                    w["baseline"] == "completed" for w in status.windows
                )
                snapshots = [
                    (run, ref) for w in status.windows for run, ref in w["snapshots"]
                ]
                assert snapshots, status.windows
                run_id, ref = snapshots[0]
                status.query_one("#workflow-windows", Select).value = ref
                await click(pilot, "#workflow-changes")
                from tldw_chatbook.UI.Screens.change_review_screen import (
                    ChangeReviewScreen,
                )

                await wait_until(
                    pilot,
                    lambda: (
                        isinstance(host.screen, ChangeReviewScreen)
                        and host.screen._active_turn
                    ),
                )
                assert host.screen._active_turn.run_id == run_id
                await wait_until(pilot, lambda: host.screen._focused_leaf >= 0)
                row, change = host.screen._leaves[host.screen._focused_leaf]
                assert row["id"] == ref and change.path == "observed.txt"
                print("WORKFLOW_UI_EXACT_NATIVE_SNAPSHOT", run_id, ref, flush=True)
                await pilot.press("escape")
                await wait_until(pilot, lambda: host.screen is status)
            await click(pilot, "#workflow-accept")
            if mode == "two-phase":
                await wait_until(
                    pilot,
                    lambda: (
                        status.decision
                        and status.decision.phase_id == "phase-2"
                        and status.snapshot.status == "awaiting_phase_review"
                    ),
                )
                assert (
                    str(status.query_one("#workflow-approve", Button).label)
                    == "Approve next phase"
                )
                heading = str(status.query_one("#workflow-state").content)
                assert (
                    f"phase {status.decision.phase_id} · plan {status.decision.plan_revision} · control revision {status.decision.control_revision}"
                    in heading
                )
                assert status.decision.phase_id == "phase-2"
                capture("phase-handoff")
                print("WORKFLOW_HANDOFF_HEADING", heading, flush=True)
                await click(pilot, "#workflow-approve")
                await wait_until(
                    pilot,
                    lambda: (
                        status.snapshot.status == "awaiting_result_review"
                        and len(status.snapshot.results) == 2
                    ),
                    timeout=20,
                )
                await click(pilot, "#workflow-accept")
            await wait_until(
                pilot, lambda: host.screen.snapshot.status == "completed", timeout=15
            )
            full = await rig.coordinator.get(workflow_id)
            count = 2 if mode == "two-phase" else 1
            assert (
                full.status == "completed"
                and full.accounting.used["child_launch"] == count
                and len(full.results) == count
            )
            with rig.service.db.connection() as conn:
                actions = [
                    json.loads(row[0])["action"]
                    for row in conn.execute(
                        "SELECT payload_json FROM workflow_decisions WHERE workflow_id=? ORDER BY control_revision",
                        (workflow_id,),
                    )
                ]
            assert actions == ["approve", "complete"] * count
            if mode == "controls":
                selected = status.query_one("#workflow-task", Select).value
                assert selected == full.results[0].run_id
                await click(pilot, "#workflow-fleet")
                await wait_until(
                    pilot,
                    lambda: (
                        host.screen is console
                        and console._agent._console_agent_drilldown_run_id == selected
                    ),
                )
                assert (
                    console._agent._current_console_rail_conversation_id()
                    == full.request.conversation_id
                )
    finally:
        try:
            await rig.close()
            assert not capacity.snapshot().executions
        except RuntimeError as error:
            if (
                mode not in {"metadata", "unknown-task"}
                or str(error) != "workflow_original_shutdown_unresolved"
            ):
                raise
            assert rig.coordinator._runs[workflow_id].status == "recovery_required"
            print(f"WORKFLOW_UI_{mode.upper()}_RETAINED_NO_CREATORS", flush=True)
