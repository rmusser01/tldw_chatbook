"""Reviewed revision transitions use the real service's exact work custody."""

import asyncio

import pytest

from tldw_chatbook.Agents.activation import worker_guard

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.Plugins.test_revocation_persistence import (
    revocation_case as _revocation_case,
)

revocation_case = _revocation_case


@pytest.mark.asyncio
async def test_revision_drain_closes_new_admission_but_finishes_owned_work(
    revocation_case,
    native_package,
):
    case = revocation_case
    service = case.service
    old = await service._call(service._coordinator.published_snapshot)
    original = next(
        row for row in old["revisions"] if row["installation_id"] == case.installation
    )
    maximum = service.capture_maximum("a")
    assert maximum["available_skills"]
    await service.check_entries(case.entries["a"])
    await service.check_entries(case.entries["b"])
    import json

    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )

    pin = await asyncio.to_thread(
        service.capture_resume_pin,
        case.entries["a"],
        "root-a",
        "conversation",
        "message",
    )
    sealed = await asyncio.to_thread(
        service.seal_resume_checkpoint,
        pin,
        parse_provider_continuation_json(json.dumps(_checkpoint())),
        "conversation",
        "message",
    )
    assert (await service.resume_maximum(maximum, sealed, "conversation", "message"))[
        "available_skills"
    ]
    root = native_package()
    skill = root / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nREVIEWED NEW REVISION\n")
    review = await service.review_revision(case.installation, root)
    # Merely reviewing does not fence the running revision.
    admitted = await service.admit(maximum, "review-still-open")
    assert admitted["available_skills"]
    apply = asyncio.create_task(service.apply_revision(review, review.operation_id))
    try:
        for _ in range(100):
            if service.revision_drain.blockers(review.drain_token):
                break
            await asyncio.sleep(0.01)
        blockers = service.revision_drain.blockers(review.drain_token)
        assert {row["run_id"] for row in blockers if row["kind"] == "active_work"} == {
            "root-a",
            "root-b",
        }
        assert not apply.done()
        assert not case.cancelled_b.is_set()
        assert case.child.poll() is None
        await service.check_entries(case.entries["a"])
        await service.check_entries(case.entries["b"])
        with pytest.raises(PermissionError, match="drain"):
            await service.admit(maximum, "late-admission")
        with pytest.raises(PermissionError, match="drain"):
            await asyncio.to_thread(
                service.bind_run,
                case.entries["a"],
                "late-child",
                lambda: None,
                "handle",
                parent_run_id="root-a",
            )
        apply.cancel()
        with pytest.raises(asyncio.CancelledError):
            await apply
        assert service.revision_drain.blockers(review.drain_token)
        case.child.terminate()
        await asyncio.to_thread(case.child.wait, 2)
        await asyncio.to_thread(service.complete_run, "root-a")
        await asyncio.to_thread(service.complete_run, "root-b")
        # The original caller has left. Publication must complete on the retained
        # operation itself, without a second Apply or an unlock refreshing catalog.
        ticket = service.revision_drain.ticket(review.drain_token)
        for _ in range(200):
            if ticket.task.done():
                break
            await asyncio.sleep(0.01)
        assert ticket.task.done() and ticket.task.result().committed
        new = await service._call(service._coordinator.published_snapshot)
        installed = next(
            row
            for row in new["installations"]
            if row["installation_id"] == case.installation
        )
        assert installed["revision_digest"] == review.inspection.effective_digest
        retained = next(
            row
            for row in new["revisions"]
            if row["revision_digest"] == original["revision_digest"]
        )
        assert retained == original
        assert (
            installed["alias"]
            == next(
                row
                for row in old["installations"]
                if row["installation_id"] == case.installation
            )["alias"]
        )
        fresh = await service.admit(service.capture_maximum("a"), "after-update")
        assert fresh["available_skills"]
        await service.check_entries(fresh["available_skills"])
        with pytest.raises(PermissionError):
            await service.resume_maximum(
                service.capture_maximum("a"), sealed, "conversation", "message"
            )
        with pytest.raises(PermissionError):
            await service.check_entries(case.entries["a"])
    finally:
        if not apply.done():
            service.revision_drain.cancel(review.drain_token)
            await asyncio.gather(apply, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_proposal_does_not_cancel_work_or_another_drain(
    revocation_case, native_package
):
    case = revocation_case
    service = case.service
    root = native_package()
    skill = root / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nREPLACEMENT\n")
    review = await service.review_revision(case.installation, root)
    maximum = service.capture_maximum("a")
    other = service.revision_drain.begin(case.installation, review.previous_revision)
    task = asyncio.create_task(service.apply_revision(review, review.operation_id))
    try:
        for _ in range(100):
            if service.revision_drain.blockers(review.drain_token):
                break
            await asyncio.sleep(0.01)
        service.revision_drain.cancel(review.drain_token)
        with pytest.raises(ValueError, match="cancelled"):
            await task
        assert case.child.poll() is None and not case.cancelled_b.is_set()
        await service.check_entries(case.entries["a"])
        with pytest.raises(PermissionError, match="drain"):
            await service.admit(maximum, "still-blocked")
        service.revision_drain.cancel(other)
        assert (await service.admit(maximum, "reopened"))["available_skills"]
        with pytest.raises(ValueError, match="cancelled"):
            await service.apply_revision(review, review.operation_id)
    finally:
        if not task.done():
            service.revision_drain.cancel(review.drain_token)
            await asyncio.gather(task, return_exceptions=True)
        service.revision_drain.cancel(other)


@pytest.mark.asyncio
async def test_explicit_cancel_work_retains_survivors_until_confirmed(
    revocation_case, native_package
):
    case = revocation_case
    case.keep_alive()
    root = native_package()
    skill = root / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nREPLACEMENT\n")
    review = await case.service.review_revision(case.installation, root)
    task = asyncio.create_task(case.service.apply_revision(review, review.operation_id))
    try:
        for _ in range(100):
            if case.service.revision_drain.blockers(review.drain_token):
                break
            await asyncio.sleep(0.01)
        case.service.revision_drain.cancel_work(review.drain_token)
        await asyncio.wait_for(case.cleanup_started.wait(), 2)
        assert case.cancelled_b.is_set() and case.child.poll() is None
        assert not task.done()
        case.child.terminate()
        await asyncio.to_thread(case.child.wait, 2)
        await asyncio.to_thread(case.service.complete_run, "root-a")
        await asyncio.to_thread(case.service.complete_run, "root-b")
        assert (await task).committed
    finally:
        if not task.done():
            case.service.revision_drain.cancel(review.drain_token)
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_rollback_retains_current_disables_and_new_selection_exclusions(
    revocation_case, native_package
):
    from pathlib import Path

    case = revocation_case
    service = case.service
    root = native_package()
    skill = root / "skills/review/SKILL.md"
    skill.write_text(skill.read_text() + "\nREPLACEMENT\n")
    # A newly discovered supported skill must stay outside the old selection.
    import json

    manifest = json.loads((root / "plugin.json").read_text())
    (root / "skills/new").mkdir()
    (root / "skills/new/SKILL.md").write_text(
        "---\nname: new\ndescription: Newly added capability.\n---\nNew instructions.\n"
    )
    (root / "plugin.json").write_text(json.dumps(manifest))
    review = await service.review_revision(case.installation, root)
    assert review.selection == ("skill:review",)
    case.child.terminate()
    await asyncio.to_thread(case.child.wait, 2)
    await asyncio.to_thread(service.complete_run, "root-a")
    await asyncio.to_thread(service.complete_run, "root-b")
    assert (await service.apply_revision(review, review.operation_id)).committed
    disabled = await service.review_activation(
        case.installation, workspace_id="b", intent="disabled"
    )
    assert (await service.commit(disabled, disabled.operation_id)).committed
    rollback = await service.review_rollback(
        case.installation, review.previous_revision
    )
    assert rollback.rollback and rollback.data_compatibility == "unknown"
    assert (await service.apply_revision(rollback, rollback.operation_id)).committed
    assert not service.capture_maximum("b")["available_skills"]
    assert service.capture_maximum("a")["available_skills"]
    snapshot = await service._call(service._coordinator.published_snapshot)
    assert all(
        Path(row["materialized_identity"]).is_dir() for row in snapshot["revisions"]
    )
    assert snapshot["installations"][0]["revision_digest"] == review.previous_revision


@pytest.mark.asyncio
@pytest.mark.parametrize("ownership", ["pending", "unresolved", "idle"])
async def test_public_standalone_drain_waits_for_runtime_ownership(
    plugin_stack, native_package, ownership
):
    import subprocess
    import sys

    from Tests.Plugins.test_coordinator import reviewed

    stack = plugin_stack
    review = reviewed(stack, native_package())
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    launch = stack.call(
        lambda: stack.owner.reserve_launch(
            "drain-owned-launch",
            review.installation_id,
            None,
            review.inspection.effective_digest,
        )
    )
    child = None
    if ownership == "unresolved":
        stack.call(lambda: stack.owner.settle_process(launch, False))
    if ownership == "idle":
        child = await asyncio.to_thread(
            subprocess.Popen,
            [
                sys.executable,
                "-I",
                "-c",
                'import sys; print("ready",flush=True); sys.stdin.read()',
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert await asyncio.to_thread(child.stdout.readline) == "ready\n"
        stack.call(
            lambda: stack.owner.publish_process(
                launch, {"pid": child.pid, "owner": "test-owned-idle-child"}
            )
        )
        stack.call(lambda: stack.owner.set_process_kind(launch, "idle_connection"))
        assert (
            stack.call(
                lambda: stack.owner.active_revision_leases(
                    review.installation_id, review.inspection.effective_digest
                )
            )
            == 0
        )
    drain = stack.coordinator.revision_drain
    ticket = drain.begin(review.installation_id, review.inspection.effective_digest)
    waiter = None
    try:
        assert drain.blockers(ticket), (
            "standalone drain falsely reports no unresolved ownership"
        )
        waiter = asyncio.create_task(
            asyncio.to_thread(stack.call, lambda: drain.wait(ticket))
        )
        for _ in range(100):
            if any(row.get("lease_token") == launch for row in drain.blockers(ticket)):
                break
            await asyncio.sleep(0.01)
        assert any(row.get("lease_token") == launch for row in drain.blockers(ticket))
        assert not waiter.done()
        if child is not None:
            child.terminate()
            await asyncio.to_thread(child.wait, 2)
        await asyncio.to_thread(
            stack.call, lambda: stack.owner.settle_process(launch, True)
        )
        await asyncio.wait_for(waiter, 2)
        assert not drain.blockers(ticket)
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                await asyncio.to_thread(child.wait, 2)
            child.stdin.close()
            child.stdout.close()
        drain.cancel(ticket)
        if waiter is not None:
            await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_work", [False, True])
async def test_real_console_pending_permission_approval_blocks_revision_publication(
    native_console, native_package, tmp_path, cancel_work
):
    import functools
    import threading
    from types import SimpleNamespace

    from Tests.Chat.test_console_agent_bridge import _fence, _FleetChunkGateway, _run
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.MCP.permission_store import (
        BUILTIN_TOOL_SERVER_KEY,
        MCPPermissionStore,
    )

    rig = native_console
    installed = await rig.install()
    permissions = MCPPermissionStore(tmp_path / "approval-permissions.json")
    permissions.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "calculator", "ask")
    gate = BuiltinToolGate(SimpleNamespace(permission_store=permissions))
    stopped, shown = threading.Event(), threading.Event()
    cards = []
    rig.controller.app.call_from_thread = lambda fn, *args, **kwargs: fn(
        *args, **kwargs
    )

    def show(payload):
        if payload is not None:
            cards.append(payload)
            shown.set()

    rig.controller.set_pending_approval = show
    rig.controller.mcp_approval_timeout_seconds = lambda: 15.0
    hook = build_tool_review_hook(
        gate,
        BuiltinToolProvider(gate=gate),
        None,
        functools.partial(
            rig.controller.request_mcp_approvals, session_id=rig.session.id
        ),
    )
    gateway = _FleetChunkGateway(
        [[_fence("calculator", {"expression": "6*7"})], ["approved result"]], []
    )
    runs = AgentRunsDB(
        tmp_path / "approval-runs.sqlite", client_id="plugin-drain-approval"
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs,
        store=rig.store,
        provider_gateway=gateway,
        skills_service=rig.skills,
    )
    owner = rig.store.append_message(
        rig.session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    maximum = rig.service.capture_maximum("workspace-a")
    maximum["plugin_run_id"] = "pending:approval-drain"
    run = asyncio.create_task(
        asyncio.to_thread(
            worker_guard(bridge)(_run),
            bridge,
            rig.store,
            rig.session,
            owner.id,
            conversation_id="drain-approval",
            workspace_id="workspace-a",
            skills_context=maximum,
            plugin_cancel_root=stopped.set,
            should_cancel=stopped.is_set,
            revoke_approvals=rig.controller.revoke_approval_rounds_for_run,
            builtin_gate=gate,
            review_tool_calls=hook,
        )
    )
    apply = None
    try:
        assert await asyncio.to_thread(shown.wait, 4), (
            "real permission approval never surfaced"
        )
        card = cards[-1]
        state = rig.controller._pending_approval_rounds[card["round_id"]]
        assert not state["event"].is_set() and state["decisions"] == {}
        assert card["calls"][0]["llm_name"] == "calculator"
        root = native_package()
        skill = root / "skills/review/SKILL.md"
        skill.write_text(skill.read_text() + "\nRevision after permission approval\n")
        review = await rig.service.review_revision(installed.installation_id, root)
        apply = asyncio.create_task(
            rig.service.apply_revision(review, review.operation_id)
        )
        for _ in range(100):
            blockers = rig.service.revision_drain.blockers(review.drain_token)
            if any(row.get("run_id") == state["run_id"] for row in blockers):
                break
            await asyncio.sleep(0.01)
        assert any(row.get("run_id") == state["run_id"] for row in blockers)
        assert not apply.done() and not run.done()
        before = await rig.service._call(rig.service._coordinator.published_snapshot)
        assert before["installations"][0]["revision_digest"] == review.previous_revision
        with pytest.raises(PermissionError, match="drain"):
            await rig.service.admit(maximum, "approval-new-run")
        assert not state["event"].is_set() and state["decisions"] == {}
        if cancel_work:
            rig.service.revision_drain.cancel_work(review.drain_token)
        else:
            key = card["calls"][0].get("call_id") or "calculator"
            rig.controller.resolve_pending_approval(
                {key: "approve_once"}, round_id=card["round_id"]
            )
        outcome = await asyncio.wait_for(run, 5)
        assert outcome.status == ("cancelled" if cancel_work else "done"), outcome
        assert (
            any(
                step.tool_name == "calculator" and step.tool_outcome == "success"
                for step in outcome.steps
            )
            != cancel_work
        )
        assert (await asyncio.wait_for(apply, 5)).committed
        assert (
            await rig.service.admit(
                rig.service.capture_maximum("workspace-a"), "after-approval-drain"
            )
        )["available_skills"]
    finally:
        stopped.set()
        for round_id in tuple(rig.controller._pending_approval_rounds):
            rig.controller.resolve_pending_approval(
                {"calculator": "deny"}, round_id=round_id
            )
        await asyncio.gather(run, return_exceptions=True)
        if apply is not None and not apply.done():
            rig.service.revision_drain.cancel(review.drain_token)
            await asyncio.gather(apply, return_exceptions=True)
        runs.close()
