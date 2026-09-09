"""Immutable launch validation and double Start through real service provisioning."""

import importlib

import pytest
from textual.app import App
from textual.widgets import TextArea

from Tests.Chat.test_goal_conversation_provisioning import service
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture

stores = _stores_fixture


def setup_view():
    spec = importlib.util.find_spec(
        "tldw_chatbook.Widgets.Console.console_goal_setup_modal"
    )
    assert spec is not None, "goal launch form missing"
    return importlib.import_module(spec.name).ConsoleGoalSetupModal


@pytest.mark.asyncio
async def test_double_start_retains_launch_identity_and_provisions_one_conversation(
    stores,
):
    import asyncio

    owner = service(stores)
    launches = []
    release = asyncio.Event()

    async def start(request, launch_id):
        goal = owner.create(request, launch_id=launch_id)
        launches.append(goal)
        await release.wait()
        return goal

    app = App()
    async with app.run_test(size=(80, 24)) as pilot:
        modal = setup_view()(stores[3], start=start)
        app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#goal-objective", TextArea).load_text("Repair fixture")
        modal.query_one("#goal-criteria", TextArea).load_text("Validation exits zero")
        await pilot.click("#goal-start")
        await pilot.click("#goal-start")
        assert len(launches) == 1
        release.set()
        await pilot.pause()
        # Retrying the same immutable modal launch after a lost response is safe.
        goal = owner.create(launches[0].request, launch_id=modal.launch_id)
        assert len(owner.list_goals()) == 1
        assert goal.id == launches[0].id
        assert goal.request.human_review_required is True
        with stores[1].db.transaction() as conn:
            assert conn.execute("SELECT count(*) FROM conversations").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_verifier_factory_uses_trust_owner_and_rejects_editable_verifier(
    tmp_path,
):
    from Tests.Chat.test_goal_cli_verification import trusted_skill

    scope, path, trust = trusted_skill(tmp_path, "print('verified')\n")
    assert hasattr(scope, "goal_verifier_reference"), (
        "trusted verifier reference factory missing"
    )
    spec = await scope.goal_verifier_reference(
        "verifier",
        "scripts/check.py",
        arguments=("/tmp/project",),
        input_paths=("fixture.txt",),
    )
    import hashlib

    assert spec.verifier_path == str(path)
    assert spec.verifier_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert spec.skill_trust_ref == trust.current_fingerprint_digest("verifier")
    path.write_text("print('changed')\n")
    from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError

    with pytest.raises(SkillTrustBlockedError):
        await scope.goal_verifier_reference(
            "verifier", "scripts/check.py", arguments=(), input_paths=("fixture.txt",)
        )


@pytest.mark.asyncio
async def test_runtime_owned_launch_survives_view_wait_cancel_and_shutdown_drains(
    stores, monkeypatch
):
    import asyncio
    import threading
    from types import SimpleNamespace

    from Tests.Chat.test_console_goal_dispatch import build_goal_rig

    goal, _store, _, _, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    entered, release = threading.Event(), threading.Event()
    create = coordinator.service.create

    def delayed(*args, **kwargs):
        entered.set()
        release.wait(3)
        return create(*args, **kwargs)

    monkeypatch.setattr(coordinator.service, "create", delayed)
    try:
        assert hasattr(coordinator, "launch"), "setup lacks a runtime owner"
        app = SimpleNamespace(chachanotes_db=stores[1].db, app_config={})
        task = coordinator.launch(goal.request, goal.launch_id, app=app)
        assert coordinator.launch(goal.request, goal.launch_id, app=app) is task
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()

        async def view_wait():
            return await asyncio.shield(task)

        view = asyncio.create_task(view_wait())
        view.cancel()
        await asyncio.gather(view, return_exceptions=True)
        drain = asyncio.create_task(coordinator.shutdown())
        await asyncio.sleep(0.03)
        assert not drain.done() and not task.done()
        release.set()
        await drain
        assert calls == []
        assert len(coordinator.service.list_goals()) == 1
    finally:
        release.set()
        await coordinator.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted", [False, True])
async def test_mounted_start_and_interrupted_setup_retry_dispatch_real_runtime(
    stores, monkeypatch, interrupted
):
    import asyncio
    from types import SimpleNamespace

    from Tests.Chat.test_console_goal_dispatch import build_goal_rig
    from tldw_chatbook.UI.Console_Modules.goals import ConsoleGoalsController
    from tldw_chatbook.Widgets.Console.console_goal_status import ConsoleGoalStatus

    goal, store, _, controller, co, gateway, calls = build_goal_rig(stores, monkeypatch)
    ordinary = store.create_session()
    ordinary.draft = "ordinary untouched"
    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    backend = SimpleNamespace(
        chachanotes_db=stores[1].db,
        app_config={},
        chat_conversation_scope_service=ChatConversationScopeService(
            local_service=ChatConversationService(stores[1].db), server_service=None
        ),
    )
    provision = co.service.persistence.provision_goal_conversation
    failed = False

    def flaky(intent):
        nonlocal failed
        if interrupted and not failed:
            failed = True
            raise RuntimeError("lost setup reply")
        return provision(intent)

    monkeypatch.setattr(co.service.persistence, "provision_goal_conversation", flaky)
    app = App()
    ui = ConsoleGoalsController(
        app_instance=backend,
        get_controller=lambda: controller,
        get_coordinator=lambda: co,
        push_screen=lambda *args: None,
        run_worker=asyncio.create_task,
        open_changes=lambda *args, **kw: None,
    )
    received = []

    def launched(saved):
        received.append(saved)
        app.push_screen(ConsoleGoalStatus(co, saved.id, app_instance=backend))

    try:
        async with app.run_test(size=(80, 24)) as pilot:
            modal = setup_view()(goal.request, start=ui.start)
            app.push_screen(modal, callback=launched)
            await pilot.pause()
            await pilot.click("#goal-start")
            await pilot.pause()
            assert len(received) == 1
            launched_id = received[0].id
            assert received[0].launch_id == modal.launch_id
            if interrupted:
                assert co.service.get(launched_id).status == "starting"
                assert not calls
                await pilot.click("#goal-retry-setup")
                await pilot.pause()
            for _ in range(100):
                if calls and co.active_goal_id is None:
                    break
                await asyncio.sleep(0.01)
            assert calls
            assert co.service.get(launched_id).iteration_count > 0
            assert (
                len(co.service.list_goals()) == 2
            )  # existing fixture goal plus one new launch
            assert (
                store.active_session_id == ordinary.id
                and ordinary.draft == "ordinary untouched"
            )
            assert (
                co.service.get(launched_id).conversation_id
                == received[0].conversation_id
            )
    finally:
        await co.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
async def test_actual_setup_derives_immutable_authority_and_reviews_before_launch(
    stores, monkeypatch
):
    import asyncio
    from types import SimpleNamespace

    from Tests.Chat.test_console_goal_dispatch import build_goal_rig
    from tldw_chatbook.UI.Console_Modules.goals import ConsoleGoalsController

    goal, _store, _session, controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    notices = []
    app = App()
    ui = ConsoleGoalsController(
        app_instance=SimpleNamespace(notify=lambda *args, **kw: notices.append(args)),
        get_controller=lambda: controller,
        get_coordinator=lambda: co,
        push_screen=app.push_screen,
        run_worker=asyncio.create_task,
        open_changes=lambda *args, **kw: None,
    )
    try:
        async with app.run_test(size=(100, 35)) as pilot:
            await ui._open_setup()
            await pilot.pause()
            assert not notices, notices
            modal = app.screen
            assert modal.request.provider == goal.request.provider
            assert modal.request.binding == goal.request.binding
            assert modal.request.human_review_required
            await pilot.click("#goal-start")
            await pilot.pause()
            assert modal._submitted is not None
            assert modal.query_one("#goal-start").label.plain == "Start"
            assert not calls
            assert len(co.service.list_goals()) == 1
    finally:
        await co.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
async def test_ro_first_rw_second_binding_refresh_launches_authorized_edit(
    stores, monkeypatch, tmp_path
):
    import asyncio
    import json
    from types import SimpleNamespace

    from textual.widgets import Select, SelectionList

    import tldw_chatbook.Chat.console_chat_controller as controller_module
    from Tests.Agents.test_goal_iteration_report import report
    from Tests.Chat.test_console_goal_dispatch import build_goal_rig
    from Tests.Chat.test_console_local_review_hook import ALLOW, _FakeService

    def tool(name, arguments):
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": "edit",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }

    from tldw_chatbook.Chat.chat_conversation_scope_service import (
        ChatConversationScopeService,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.UI.Console_Modules.goals import ConsoleGoalsController
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

    writable = tmp_path / "writable"
    writable.mkdir()
    artifact = writable / "fixture.txt"
    artifact.write_text("invalid\n")
    registry = stores[2]
    registry.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="workspace",
            binding_id="binding",
            binding_kind="local-filesystem",
            label="Read only",
            locator=stores[3].binding.locator,
            status="ready",
            metadata={"access": "ro"},
        )
    )
    registry.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="workspace",
            binding_id="writable",
            binding_kind="local-filesystem",
            label="Writable",
            locator=str(writable),
            status="ready",
            metadata={"access": "rw"},
        )
    )
    n = 0

    def provider(**kwargs):
        nonlocal n
        n += 1
        message = (
            tool(
                "fs_edit",
                {"path": "fixture.txt", "old_string": "invalid", "new_string": "valid"},
            )
            if n == 1
            else {"content": report(summary="edited")}
        )
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    _, store, _, controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch, provider
    )
    controller.app = SimpleNamespace(unified_mcp_service=_FakeService(state=ALLOW))
    setting = controller_module.get_cli_setting
    monkeypatch.setattr(
        controller_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            True
            if (section, key) == ("console", "local_tools_enabled")
            else setting(section, key, default)
        ),
    )
    backend = SimpleNamespace(
        chachanotes_db=stores[1].db,
        app_config={},
        chat_conversation_scope_service=ChatConversationScopeService(
            local_service=ChatConversationService(stores[1].db), server_service=None
        ),
        notify=lambda *args, **kw: pytest.fail(str(args)),
    )
    ordinary = store.create_session(workspace_id="workspace")
    ordinary.draft = "ordinary draft"
    app = App()
    ui = ConsoleGoalsController(
        app_instance=backend,
        get_controller=lambda: controller,
        get_coordinator=lambda: co,
        push_screen=app.push_screen,
        run_worker=asyncio.create_task,
        open_changes=lambda *args, **kw: None,
    )
    try:
        async with app.run_test(size=(100, 40)) as pilot:
            await ui._open_setup()
            await pilot.pause()
            modal = app.screen
            binding = modal.query_one("#goal-binding", Select)
            choices = modal.query_one("#goal-tools", SelectionList)
            assert binding.value == "binding"
            assert "local:fs_edit" not in modal.tool_ids
            assert "local:fs_read" in choices.selected
            binding.value = "writable"
            await pilot.pause()
            assert "local:fs_edit" in modal.tool_ids
            choices.select("local:fs_edit")
            binding.value = "binding"
            await pilot.pause()
            assert "local:fs_edit" not in modal.tool_ids
            assert "local:fs_edit" not in choices.selected
            assert "local:fs_read" in choices.selected
            binding.value = "writable"
            await pilot.pause()
            choices.select("local:fs_edit")
            choices.deselect("local:fs_read")
            await pilot.click("#goal-start")
            await pilot.pause()
            assert modal._submitted.binding.binding_id == "writable"
            summary = str(modal.query_one("#goal-selected-tools").render())
            assert "local:fs_edit" in summary and "local:fs_read" not in summary
            await pilot.click("#goal-start")
            for _ in range(200):
                if calls and co.active_goal_id is None:
                    break
                await pilot.pause(0.01)
            assert artifact.read_text() == "valid\n"
            assert (
                ordinary.draft == "ordinary draft"
                and store.active_session_id == ordinary.id
            )
            new = next(
                g
                for g in co.service.list_goals()
                if co.service.get(g.id).launch_id == modal.launch_id
            )
            assert co.service.get(new.id).request.tool_scope.catalog_tools == (
                "local:fs_edit",
            )
    finally:
        await co.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
async def test_binding_discovery_ignores_late_cancelled_refresh(stores):
    import asyncio

    from textual.widgets import Select, SelectionList

    req = stores[3]
    readonly = req.binding.model_copy(update={"binding_id": "readonly", "access": "ro"})
    entered, release = asyncio.Event(), asyncio.Event()

    async def discover(binding):
        if binding.binding_id == "readonly":
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return ("local:fs_read",)
        return ("local:fs_read", "local:fs_edit")

    async def no_start(*args):
        pytest.fail("Discovery cannot dispatch")

    app = App()
    async with app.run_test() as pilot:
        modal = setup_view()(
            req,
            start=no_start,
            bindings=(req.binding, readonly),
            tool_ids=("local:fs_read", "local:fs_edit"),
            discover_tools=discover,
            configure=lambda *args: None,
        )
        app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#goal-binding", Select).value = "readonly"
        await asyncio.wait_for(entered.wait(), 1)
        assert modal.query_one("#goal-start").disabled
        modal.query_one("#goal-binding", Select).value = req.binding.binding_id
        await pilot.pause()
        release.set()
        await pilot.pause()
        assert modal.tool_ids == ("local:fs_read", "local:fs_edit")
        assert "local:fs_edit" in modal.query_one("#goal-tools", SelectionList).selected
        assert not modal.query_one("#goal-start").disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("reject", [False, True])
async def test_setup_freezes_all_fields_before_validation_and_restores_on_failure(
    stores, tmp_path, reject
):
    import asyncio

    from textual.widgets import Checkbox, Input, SelectionList

    from Tests.Chat.test_goal_cli_verification import trusted_skill
    from tldw_chatbook.Agents.goal_models import GoalRequest
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding

    req = stores[3]
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = req.binding.model_copy(
        update={"binding_id": "source", "locator": str(source_root), "access": "ro"}
    )
    stores[2].save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="workspace",
            binding_id="source",
            binding_kind="local-filesystem",
            label="Source",
            locator=str(source_root),
            status="ready",
            metadata={"access": "ro"},
        )
    )
    skills, _path, _trust = trusted_skill(tmp_path, "print('valid')\n")
    entered, release = asyncio.Event(), asyncio.Event()
    captured = []
    owner = service(stores)
    launched = []

    async def configure(values, *args):
        captured.append(values)
        entered.set()
        await release.wait()
        if reject:
            raise ValueError("validation rejected")
        skill, script, arguments, inputs = args
        values["verifiers"] = (
            (
                await skills.goal_verifier_reference(
                    skill, script, arguments=arguments, input_paths=inputs
                )
            ).model_dump(),
        )
        return GoalRequest.model_validate(values)

    async def start(request, launch_id):
        goal = owner.create(request, launch_id=launch_id)
        launched.append(goal)
        return goal

    app = App()
    async with app.run_test(size=(100, 40)) as pilot:
        modal = setup_view()(
            req,
            start=start,
            bindings=(req.binding, source),
            tool_ids=req.tool_scope.catalog_tools,
            configure=configure,
        )
        app.push_screen(modal)
        await pilot.pause()
        modal.query_one("#goal-objective", TextArea).load_text("Frozen objective")
        modal.query_one("#goal-criteria", TextArea).load_text("Frozen criteria")
        modal.query_one("#goal-sources", SelectionList).select("source")
        modal.query_one("#goal-skill", Input).value = "verifier"
        modal.query_one("#goal-script", Input).value = "scripts/check.py"
        modal.query_one("#goal-arguments", Input).value = '["exact argument"]'
        modal.query_one("#goal-inputs", Input).value = '["checked.txt"]'
        modal.query_one("#goal-human-review", Checkbox).value = False
        await pilot.click("#goal-start")
        await asyncio.wait_for(entered.wait(), 1)
        fields = list(modal.query("Input, TextArea, Select, SelectionList, Checkbox"))
        assert all(field.disabled for field in fields)
        release.set()
        await pilot.pause()
        if reject:
            assert all(not field.disabled for field in fields)
            assert not modal.query_one("#goal-start").disabled
            assert modal._submitted is None
        else:
            assert modal._submitted.objective == "Frozen objective"
            assert modal._submitted.criteria == "Frozen criteria"
            assert modal._submitted.source_bindings == (source,)
            assert modal._submitted.tool_scope == req.tool_scope
            assert modal._submitted.verifiers[0].arguments == ("exact argument",)
            assert modal._submitted.verifiers[0].input_paths == ("checked.txt",)
            assert not modal._submitted.human_review_required
            assert all(field.disabled for field in fields)
            await pilot.click("#goal-start")
            await pilot.pause()
            assert launched[0].request == GoalRequest.model_validate(captured[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("reject", [False, True])
async def test_rejected_validation_restores_completed_discovery_without_stale_choices(
    stores,
    reject,
):
    import asyncio

    from textual.widgets import Select, SelectionList

    req = stores[3]
    source = req.binding.model_copy(update={"binding_id": "source", "access": "ro"})
    entered, release = asyncio.Event(), asyncio.Event()

    async def configure(*args):
        entered.set()
        await release.wait()
        if reject:
            raise ValueError("rejected validation")
        from tldw_chatbook.Agents.goal_models import GoalRequest

        return GoalRequest.model_validate(args[0])

    async def discover(binding):
        return (
            ("local:fs_read",)
            if binding.access == "ro"
            else ("local:fs_read", "local:fs_edit")
        )

    async def no_start(*args):
        pytest.fail("Rejected setup must not launch")

    app = App()
    async with app.run_test() as pilot:
        modal = setup_view()(
            req,
            start=no_start,
            bindings=(req.binding, source),
            tool_ids=("local:fs_read", "local:fs_edit"),
            configure=configure,
            discover_tools=discover,
        )
        app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#goal-start")
        await asyncio.wait_for(entered.wait(), 1)
        # An already queued binding update can finish discovery during validation.
        modal.query_one("#goal-binding", Select).value = "source"
        await pilot.pause()
        assert not modal._refreshing_tools
        release.set()
        await pilot.pause()
        choices = modal.query_one("#goal-tools", SelectionList)
        assert not choices.disabled and not modal.query_one("#goal-start").disabled
        assert "local:fs_edit" not in choices.selected
        assert modal._submitted is None
