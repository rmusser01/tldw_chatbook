"""Scoped revocation must precede every worker/storage await."""

import asyncio

import pytest

from Tests.Plugins import test_revocation_persistence

revocation_case = test_revocation_persistence.revocation_case

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.mark.asyncio
async def test_disable_starts_cleanup_before_blocked_persistence(revocation_case):
    case = revocation_case
    case.block_persistence()
    operation = case.start_disable_here()
    try:
        await asyncio.wait_for(case.cleanup_started.wait(), 2)
        assert not case.admission_allows_a()
        assert case.admission_allows_b()
        assert not case.cancelled_b.is_set()
        await asyncio.to_thread(case.child.wait, 2)
        assert not operation.done()
    finally:
        case.release_persistence()
    receipt = await operation
    assert receipt.committed


@pytest.mark.parametrize(
    "workspace,everywhere,default",
    [
        (None, False, False),
        ("a", True, False),
        (None, True, True),
        ("global", False, False),
        ("", False, False),
    ],
)
def test_target_requires_unambiguous_scope(workspace, everywhere, default):
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    with pytest.raises(ValueError):
        RevocationTarget("installation", workspace, everywhere, default)


@pytest.mark.asyncio
async def test_surviving_child_and_cancelled_waiter_keep_terminal_custody(
    revocation_case,
):
    case = revocation_case
    case.keep_alive()
    case.block_persistence()
    operation = case.start_disable_here()
    try:
        await asyncio.wait_for(case.cleanup_started.wait(), 2)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        status = case.service.revocation_status(case.requests["disable-a"])
        assert (
            not status.committed
            and not status.runtime_stopped
            and status.cleanup_pending
        )
        assert case.child.poll() is None
        assert {run.run_id for run in case.service.live_runs()} == {"root-a", "root-b"}
    finally:
        case.release_persistence()
    receipt = await case.start_disable_here()
    assert receipt.committed and not receipt.runtime_stopped and receipt.cleanup_pending
    case.child.terminate()
    await asyncio.to_thread(case.child.wait, 2)
    await asyncio.to_thread(case.service.complete_run, "root-a")
    status = case.service.revocation_status(case.requests["disable-a"])
    assert status.runtime_stopped and not status.cleanup_pending
    assert not case.cancelled_b.is_set()


@pytest.mark.asyncio
async def test_fresh_enable_never_revives_old_callback_or_pending_ceiling(
    revocation_case,
):
    case = revocation_case
    maximum = case.service.capture_maximum("a")
    assert (await case.start_disable_here()).committed
    assert (await case.enable("a", "fresh-enable")).committed
    fresh = await case.service.admit(case.service.capture_maximum("a"), "fresh-pending")
    await case.service.check_entries(fresh["available_skills"])
    with pytest.raises(PermissionError):
        await case.service.check_entries(case.entries["a"])
    with pytest.raises(PermissionError):
        await case.service.admit(maximum, "late-pending")
    await case.service.check_entries(case.entries["b"])


@pytest.mark.asyncio
async def test_global_default_cancels_only_captured_inheritors(revocation_case):
    import threading

    from tldw_chatbook.Plugins.revocation import RevocationTarget

    case = revocation_case
    await case.enable(None, "default-on")
    admitted = await case.service.admit(case.service.capture_maximum("c"), "pending-c")
    entries = admitted["available_skills"]
    cancelled = threading.Event()
    await asyncio.to_thread(case.service.bind_run, entries, "root-c", cancelled.set)
    case.block_persistence()
    operation = asyncio.create_task(
        case.disable(
            RevocationTarget(case.installation, None, False, global_default=True),
            "default-off",
        )
    )
    try:
        await asyncio.sleep(0)
        assert cancelled.is_set()
        case.service.check_entries_live(case.entries["a"])
        case.service.check_entries_live(case.entries["b"])
        with pytest.raises(PermissionError):
            case.service.check_entries_live(entries)
    finally:
        case.release_persistence()
        await operation
        await asyncio.to_thread(case.service.complete_run, "root-c")
    await case.service.check_entries(case.entries["a"])
    await case.service.check_entries(case.entries["b"])
    assert case.service.revocation_status(case.requests["default-off"]).runtime_stopped


@pytest.mark.asyncio
async def test_everywhere_cancels_all_and_fresh_enable_is_scoped(revocation_case):
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    case = revocation_case
    receipt = await case.disable(
        RevocationTarget(case.installation, None, True), "everywhere"
    )
    assert receipt.committed and case.cancelled_b.is_set()
    await case.enable("a", "a-resume")
    fresh = await case.service.admit(case.service.capture_maximum("a"), "fresh")
    await case.service.check_entries(fresh["available_skills"])
    assert not case.service.capture_maximum("b")["available_skills"]
    with pytest.raises(PermissionError):
        await case.service.check_entries(case.entries["a"])


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", ["disabled", "inherit"])
async def test_reviewed_disabled_commit_also_seals_before_worker(
    revocation_case, intent
):
    case = revocation_case
    review = await case.service.review_activation(
        case.installation, workspace_id="a", intent=intent
    )
    case.block_persistence()
    operation = asyncio.create_task(case.service.commit(review, review.operation_id))
    try:
        await asyncio.wait_for(case.cleanup_started.wait(), 2)
        assert not case.admission_allows_a() and case.admission_allows_b()
    finally:
        case.release_persistence()
        await operation


@pytest.mark.asyncio
async def test_direct_console_revocation_retains_real_provider_until_terminal(
    native_console, monkeypatch
):
    import threading

    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderSelection,
    )
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    rig = native_console
    installed = await rig.install()
    name = rig.service.capture_maximum("workspace-a")["available_skills"][0]["name"]
    entered, release, terminal = threading.Event(), threading.Event(), threading.Event()

    def transport(**kwargs):
        entered.set()
        try:
            assert release.wait(10)
            return {"choices": [{"message": {"content": "LATE_REVOKED_REPLY"}}]}
        finally:
            terminal.set()

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=transport,
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
    )
    original = gateway.resolve_for_send

    async def resolve(selection):
        return await original(
            ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
        )

    monkeypatch.setattr(gateway, "resolve_for_send", resolve)
    monkeypatch.setattr(rig.controller, "provider_gateway", gateway)
    submit = asyncio.create_task(
        rig.controller.submit_draft(f"${name} inspect", session_id=rig.session.id)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        records = rig.service.live_runs()
        assert len(records) == 1
        request = rig.service.begin_disable(
            RevocationTarget(installed.installation_id, "workspace-a", False)
        )
        receipt = await rig.service.finish_revocation(request)
        assert (
            receipt.committed
            and receipt.cleanup_pending
            and not receipt.runtime_stopped
        )
        assert not terminal.is_set() and not records[0].completed.is_set()
        release.set()
        await asyncio.gather(submit, return_exceptions=True)
        assert await asyncio.to_thread(terminal.wait, 5)
        for _ in range(200):
            if not rig.service.live_runs():
                break
            await asyncio.sleep(0.01)
        assert not rig.service.live_runs()
        assert records[0].completed.is_set()
        assert rig.service.revocation_status(request).runtime_stopped
        assert all(
            "LATE_REVOKED_REPLY" not in str(message.content)
            for message in rig.store.messages_for_session(rig.session.id)
        )
    finally:
        release.set()
        await asyncio.gather(submit, return_exceptions=True)
        await asyncio.to_thread(terminal.wait, 5)
        await gateway.aclose()


@pytest.mark.asyncio
async def test_late_direct_cancel_preserves_callers_next_work_and_reports_all_closers(
    native_console, monkeypatch
):
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    rig = native_console
    installed = await rig.install()
    name = rig.service.capture_maximum("workspace-a")["available_skills"][0]["name"]
    completed = asyncio.get_running_loop().create_future()
    unrelated = asyncio.Event()
    consumer_finished = asyncio.Event()
    closes = []

    def first_close():
        closes.append("first")
        raise RuntimeError("private close diagnostic")

    def second_close():
        closes.append("second")

    async def stream(resolution, messages, **kwargs):
        signals = kwargs["signals"]
        assert signals.register_provider_work(completed, first_close)
        assert signals.register_provider_work(completed, second_close)
        yield "valid original reply"

    monkeypatch.setattr(rig.gateway, "stream_chat", stream)

    async def caller():
        await rig.controller.submit_draft(f"${name} inspect", session_id=rig.session.id)
        consumer_finished.set()
        await unrelated.wait()

    task = asyncio.create_task(caller())
    try:
        await asyncio.wait_for(consumer_finished.wait(), 3)
        assert rig.service.live_runs()
        request = rig.service.begin_disable(
            RevocationTarget(installed.installation_id, "workspace-a", False)
        )
        await rig.service.finish_revocation(request)
        await asyncio.sleep(0.02)
        assert closes == ["first", "second"]
        assert not task.done() and task.cancelling() == 0
        receipt = rig.service.revocation_status(request)
        assert receipt.cleanup_errors and not receipt.runtime_stopped
    finally:
        completed.set_result(None)
        unrelated.set()
        await asyncio.gather(task, return_exceptions=True)
        for _ in range(200):
            if not rig.service.live_runs():
                break
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_exact_child_cancel_and_late_registration_are_scoped(revocation_case):
    import threading

    case = revocation_case
    cancelled_child = threading.Event()
    await asyncio.to_thread(
        case.service.bind_run,
        case.entries["a"],
        "child-a",
        cancelled_child.set,
        "child-handle",
        parent_run_id="root-a",
    )
    try:
        await case.start_disable_here()
        assert cancelled_child.is_set() and not case.cancelled_b.is_set()
        with pytest.raises(PermissionError):
            await asyncio.to_thread(
                case.service.bind_run,
                case.entries["a"],
                "late-child",
                lambda: None,
                "late",
                parent_run_id="root-a",
            )
        assert {record.run_id for record in case.service.live_runs()} == {
            "root-a",
            "child-a",
            "root-b",
        }
    finally:
        await asyncio.to_thread(case.service.complete_run, "child-a")


def test_direct_coordinator_uses_the_admissions_live_owner(
    plugin_stack, native_package
):
    from Tests.Plugins.test_admission import activate, admission, install, trust
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    stack = plugin_stack
    installation = install(stack, native_package())
    trust(stack, installation)
    activate(stack, installation, "workspace-a", "enabled")
    gate = admission(stack)
    snapshot = stack.call(lambda: gate.capture(installation, "workspace-a", "pending"))
    assert gate.fences is stack.coordinator.fences
    assert stack.call(
        lambda: stack.coordinator.disable(
            RevocationTarget(installation, "workspace-a", False)
        )
    ).committed
    with pytest.raises(PermissionError):
        gate.fences.check_snapshot(snapshot)


@pytest.mark.asyncio
async def test_explicit_inherit_resumes_fresh_work_under_true_default(revocation_case):
    case = revocation_case
    await case.enable(None, "default-true")
    await case.start_disable_here()
    review = await case.service.review_activation(
        case.installation, workspace_id="a", intent="inherit"
    )
    assert (await case.service.commit(review, review.operation_id)).committed
    fresh = await case.service.admit(case.service.capture_maximum("a"), "fresh-inherit")
    assert fresh["available_skills"]
    await case.service.check_entries(fresh["available_skills"])
    with pytest.raises(PermissionError):
        await case.service.check_entries(case.entries["a"])


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", ["enabled", "inherit"])
@pytest.mark.parametrize("disable_everywhere", [False, True])
@pytest.mark.parametrize("unrelated_publication", [False, True])
async def test_recovered_resume_reopens_only_its_current_authority(
    revocation_case, intent, disable_everywhere, unrelated_publication
):
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    case = revocation_case
    if intent == "inherit":
        await case.enable(None, "recovery-default-on")
    await case.start_disable_here()
    review = await case.service.review_activation(
        case.installation, workspace_id="a", intent=intent
    )
    fault = OSError("injected certificate milestone failure")

    def fail(phase):
        if phase == "certified":
            raise fault

    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", fail)
    )
    with pytest.raises(OSError) as caught:
        await case.service.commit(review, review.operation_id)
    assert caught.value is fault
    assert not case.admission_allows_a()
    await case.service._call(
        lambda: setattr(case.service._coordinator, "progress", None)
    )
    if unrelated_publication:
        # An unrelated B marker change after recovery must not prevent A's resume.
        await case.service.unlock("test passphrase")
        await case.enable("b", "unrelated-b-after-recovery")
    recovered = await case.service.commit(review, review.operation_id)
    assert recovered.committed and recovered.phase == "complete"
    fresh = await case.service.admit(
        case.service.capture_maximum("a"), "recovered-fresh"
    )
    assert fresh["available_skills"], (
        "recovered explicit resume must open fresh admission"
    )
    await case.service.check_entries(fresh["available_skills"])
    with pytest.raises(PermissionError):
        await case.service.check_entries(case.entries["a"])
    newer = RevocationTarget(
        case.installation, None if disable_everywhere else "a", disable_everywhere
    )
    assert (await case.disable(newer, "newer-disable")).committed
    # Replaying history is not a new enable and cannot clear this later fence.
    historical = await case.service.commit(review, review.operation_id)
    assert historical.committed and historical.phase == "complete"
    assert not case.service.capture_maximum("a")["available_skills"]
    with pytest.raises(PermissionError):
        await case.service.check_entries(fresh["available_skills"])
