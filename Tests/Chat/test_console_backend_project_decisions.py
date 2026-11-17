"""Actual original runtime notice, parent visibility, and decision resolution."""

import asyncio

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
async def test_actual_notice_clock_requires_matching_parent_and_original_resolution(request):
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_chat_controller import ProjectInstructionDispatchNotice
    from tldw_chatbook.Chat.console_control_dispatch import ControlIdentity
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import open_console_backend_profile
    from tldw_chatbook.Chat.console_execution_protocol import ConsoleWireMessage, WireIdentity

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(expected_config_path=path, expected_data_directory=directory)
    backend = ConsoleExecutionBackend(profile, profile_epoch="profile", service_epoch="project-decisions")
    pending = None
    try:
        backend.start()
        session = backend.store.create_session(title="Actual project notice")
        lifecycle = backend.presentation.view_lifecycle
        await lifecycle.prepare(lifecycle.admit("actual-client", 1, "mount"))
        owner = backend.presentation.project_decisions
        identity = ControlIdentity("profile", "project-decisions", 1, "capture", session.id, session.incarnation_id, 0)
        notice = ProjectInstructionDispatchNotice(session.id, "Local endpoint", "AGENTS.md", ".", 42, ("active",), ())
        pending = asyncio.create_task(asyncio.to_thread(backend.runtime._confirm_project_instruction_dispatch, notice))

        async def wait_head():
            while (head := owner.capture(identity)) is None:
                await asyncio.sleep(0.001)
            return head

        head = await asyncio.wait_for(wait_head(), 10)
        original = backend.runtime._project_dispatch_decisions[head.decision_id]
        assert head.notice is notice
        assert original.answerable_since is None and not original.ever_answerable

        async def command(operation, payload):
            sequence = backend._sequence + 1
            receipt = backend.dispatch(ConsoleWireMessage(
                WireIdentity("profile", "project-decisions", sequence, f"command-{sequence}", session.id, session.incarnation_id, 0),
                "command", operation, payload,
            ))
            return await receipt.applied

        checkpoint = {"attachment_id": "actual-client", "visit": 2, "decision_id": head.decision_id, "action": "mounted"}
        assert not await command("ProjectDecisionPresentation", checkpoint)
        assert original.answerable_since is None
        checkpoint["visit"] = 1
        assert await command("ProjectDecisionPresentation", checkpoint)
        started = original.answerable_since
        assert started is not None and original.ever_answerable
        assert not backend.controller._interrupt_host.view_visible
        assert await command("ProjectDecisionPresentation", checkpoint)
        assert original.answerable_since == started
        assert await command("ProjectDecisionPresentation", {**checkpoint, "action": "hidden"})
        assert original.answerable_since is None and original.ever_answerable
        assert not pending.done()
        assert await command("ProjectDecisionPresentation", checkpoint)
        assert original.answerable_since >= started
        assert await command("ResolveDecision", {"decision": {
            "kind": "ProjectDispatchDecision", "decision_id": head.decision_id, "action": "proceed",
        }})
        assert await asyncio.wait_for(pending, 10) == "proceed"
        assert owner.capture(identity) is None
    finally:
        if pending is not None and not pending.done():
            for decision_id in tuple(backend.runtime._project_dispatch_decisions):
                backend.runtime.resolve_project_instruction_dispatch(decision_id, "cancel")
            await asyncio.shield(pending)
        await backend.aclose()
