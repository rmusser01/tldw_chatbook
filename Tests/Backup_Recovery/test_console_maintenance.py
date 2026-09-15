"""Real Console controller intake and queue maintenance boundaries."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio, sys, time
from Tests.Chat.test_console_prompt_queue_coordinator import SequencedGateway, _arm_controller, _queue
from tldw_chatbook.Chat.console_prompt_queue import PromptQueueMode

case = sys.argv[1]
class Gateway(SequencedGateway):
    def __init__(self):
        super().__init__()
        self.preflight = asyncio.Event()
        self.resolve_release = asyncio.Event()
    async def resolve_for_send(self, selection):
        self.preflight.set()
        await self.resolve_release.wait()
        return await super().resolve_for_send(selection)

async def main():
    gateway = Gateway()
    controller, store, session = _arm_controller(gateway)
    store.set_session_draft(session, "keep this draft")
    if case == "refusal":
        message = store.append_message(session, role="assistant", content="existing")
        controller.maintenance_close_admission()
        before = list(store.messages_for_session(session))
        for call in (
            lambda:controller.run_prompt_chain("new", session_id=session),
            lambda:controller.submit_draft("new", session_id=session),
            lambda:controller.retry_message("missing"),
            lambda:controller.continue_from_message("missing"),
            lambda:controller.regenerate_message("missing"),
            lambda:controller.summarize_up_to("missing"),
            lambda:controller.edit_and_resend_message("missing", "new"),
        ):
            result = await call()
            assert not result.accepted and not result.should_clear_draft
        snapshot = controller.prompt_queue_registry.snapshot(session)
        for call in (
            lambda:controller.resume_prompt_queue(session),
            lambda:controller.skip_and_resume_prompt_queue(session),
            lambda:controller.use_current_context_and_resume_prompt_queue(session, expected_revision=snapshot.revision, reviewed_context_epoch=0),
        ):
            assert not (await call()).applied
        for retry in (controller.retry_failed_queue_turn, controller.retry_stopped_queue_turn):
            for keyword in (False, True):
                result = await retry(message_id=message.id) if keyword else await retry(message.id)
                assert not result.applied and result.snapshot == snapshot
                try:
                    if keyword: await retry(message_id="missing")
                    else: await retry("missing")
                except KeyError as exc:
                    assert "Unknown Console message" in str(exc)
                else:
                    raise AssertionError("missing-message behavior changed")
        assert controller.prompt_queue_registry.snapshot(session) == snapshot
        assert not await controller.recover_provider_continuation("resume", "missing", 1)
        assert (await controller.compact_context_now(session))[0] is False
        assert not (await controller.impersonate_user_reply(session)).text
        assert store.messages_for_session(session) == before
        assert store.session_draft(session) == "keep this draft"
        assert not gateway.preflight.is_set()
        called = []
        async def initial():
            called.append(True)
            raise AssertionError("coordinator crossed closed admission")
        assert not (await controller.prompt_queue_coordinator.run_prompt_chain(session, initial)).accepted
        assert not called
        import tldw_chatbook.Chat.console_fleet_wake as wakes
        original_enabled = wakes.autowake_enabled
        def forbidden_config():
            raise AssertionError("wake read config after intake closed")
        wakes.autowake_enabled = forbidden_config
        try:
            controller._fleet_wake._attempt("pending")
        finally:
            wakes.autowake_enabled = original_enabled
        assert await controller.maintenance_drain(time.monotonic()+1)
        controller.maintenance_resume()
    else:
        running = asyncio.create_task(controller.run_prompt_chain("first", session_id=session))
        await gateway.preflight.wait()
        if case == "preflight":
            controller.maintenance_close_admission()
            assert not await controller.maintenance_drain(time.monotonic())
            waiter = asyncio.create_task(controller.maintenance_drain(time.monotonic()+10))
            await asyncio.sleep(0)
            waiter.cancel()
            try: await waiter
            except asyncio.CancelledError: pass
            assert not running.done()
        gateway.resolve_release.set()
        await gateway.started[0].wait()
        if case != "preflight":
            _queue(controller, session, "second")
            if case == "user_pause":
                snapshot = controller.prompt_queue_registry.snapshot(session)
                controller.pause_prompt_queue_after_turn(session, expected_revision=snapshot.revision)
            controller.maintenance_close_admission()
            snapshot = controller.prompt_queue_registry.snapshot(session)
            rejected = controller.queue_prompt(session, text="third", expected_revision=snapshot.revision)
            assert not rejected.applied
            assert not await controller.maintenance_drain(time.monotonic())
        gateway.release[0].set()
        assert (await running).accepted
        assert await controller.maintenance_drain(time.monotonic()+1)
        assert gateway.user_turns == ["first"]
        if case != "preflight":
            snapshot = controller.prompt_queue_registry.snapshot(session)
            assert snapshot.waiting_count == 1
            controller.maintenance_resume()
            if case == "repeat_pause":
                controller.maintenance_close_admission()
                assert await controller.maintenance_drain(time.monotonic()+1)
                assert gateway.user_turns == ["first"]
                controller.maintenance_resume()
            if case == "user_pause":
                await asyncio.sleep(.05)
                assert gateway.user_turns == ["first"]
                assert controller.prompt_queue_registry.snapshot(session).mode is PromptQueueMode.PAUSED
            else:
                await asyncio.wait_for(gateway.started[1].wait(), 2)
                controller.maintenance_close_admission()
                gateway.release[1].set()
                assert await controller.maintenance_drain(time.monotonic()+2)
                assert gateway.user_turns == ["first", "second"]
        else:
            controller.maintenance_resume()
    print("retired and reopened")
asyncio.run(main())
"""


_AGENT_SCRIPT = r"""
import asyncio, threading, time, sys
from types import SimpleNamespace
from Tests.Chat.test_console_prompt_queue_coordinator import SequencedGateway, _arm_controller
from tldw_chatbook.Agents.agent_models import RunOutcome, RUN_DONE

async def main():
    controller, store, session = _arm_controller(SequencedGateway())
    from Tests.Chat.test_console_chat_controller import _arm_session
    _arm_session(store)
    entered, release = threading.Event(), threading.Event()
    published = []
    def run_reply(**kwargs):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test release timed out")
        published.append("native publication finished")
        if sys.argv[1] == "native_error":
            raise ValueError("native failure after cancellation")
        return "run-test", RunOutcome(status=RUN_DONE, steps=[], final_text="published")
    controller._agent_bridge = SimpleNamespace(run_reply=run_reply)
    running = asyncio.create_task(controller.submit_draft("first", session_id=session))
    try:
        assert await asyncio.to_thread(entered.wait, 8), (running.result() if running.done() else controller.run_state)
        controller.maintenance_close_admission()
        running.cancel()
        try: await running
        except asyncio.CancelledError: pass
        assert not published
        assert not await controller.maintenance_drain(time.monotonic()), "native publication escaped drain"
        release.set()
        assert await controller.maintenance_drain(time.monotonic()+2)
        assert published == ["native publication finished"]
    finally:
        release.set()
        await asyncio.gather(running, return_exceptions=True)
    print("retired and reopened")
asyncio.run(main())
"""


@pytest.mark.parametrize("case", ["native_success", "native_error"])
def test_console_drain_keeps_cancelled_native_agent_publication(tmp_path, case):
    _run(tmp_path, case, "success", script=_AGENT_SCRIPT)


@pytest.mark.parametrize(
    "case", ["refusal", "preflight", "queue", "user_pause", "repeat_pause"]
)
def test_console_maintenance_preserves_admitted_turns_and_queued_prompts(
    tmp_path, case
):
    _run(tmp_path, case, "success", script=_SCRIPT)
