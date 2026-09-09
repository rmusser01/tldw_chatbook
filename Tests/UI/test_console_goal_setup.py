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
