"""Original cold Send must receive input before its first configuration SQL read."""

import asyncio
import threading
import time

import pytest

from Tests.UI.test_console_hook_review_send_freeze import _until
from Tests.UI.test_console_received_intent_feedback import (
    _NaturalPreparingFrame,
    _received_console_case,
    _send,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("route", ["enter", "send-button"])
async def test_cold_stock_send_receives_before_original_configuration_sql(
    route, monkeypatch, record_property
):
    """Observe cold fallback without forcing an unused trust constructor."""
    async with _received_console_case(
        monkeypatch, f"cold-trust-{route}", durable=True
    ) as case:
        app = case.console.app_instance
        local = vars(app)["_local_skills_service"]
        assert vars(app)["_local_skill_trust_service"] is None
        assert local is not None and vars(local)["_trust_service"] is None
        factory = vars(local)["_trust_service_factory"]
        assert callable(factory)
        assert not case.probe.entered.is_set()
        loop = asyncio.get_running_loop()
        input_thread = threading.current_thread()
        stop = threading.Event()
        progressed = threading.Event()
        finished = threading.Event()
        facts = {}
        preparing = _NaturalPreparingFrame(case)

        def note_loop_progress():
            facts["loop_progress_while_held"] = not case.probe.release.is_set()
            progressed.set()

        def release_independently():
            try:
                while not case.probe.entered.wait(0.01):
                    if stop.is_set():
                        return
                # The actual SQL callback is held here, before the producer can
                # return or any acceptance path can consume this received input.
                claim = case.store.received_turn_for_session(case.session.id)
                record = (
                    case.runtime._turn_custody.get(claim.request_id)
                    if claim is not None
                    else None
                )
                intent = getattr(record, "received_intent", None)
                facts.update(
                    received_before_native=(
                        claim is not None
                        and record is not None
                        and record.received_claim is claim
                        and record.request is None
                        and intent is not None
                        and intent.session_id == case.session.id
                        and intent.inputs.draft == case.draft
                    ),
                    original_draft_kept=(
                        case.store.session_draft(case.session.id) == case.draft
                    ),
                    same_active_session=case.store.active_session_id == case.session.id,
                    configuration_on_input_thread=case.probe.thread is input_thread,
                    native_lease_live=case.probe.live_at_entry,
                    trust_factory_unchanged=(
                        vars(local)["_trust_service_factory"] is factory
                    ),
                )
                loop.call_soon_threadsafe(note_loop_progress)
                deadline = time.perf_counter() + 0.5
                progressed.wait(0.5)
                preparing.painted.wait(max(0, deadline - time.perf_counter()))
            except BaseException as error:
                facts["observer_error"] = type(error).__name__
            finally:
                case.probe.release.set()
                finished.set()

        releaser = threading.Thread(target=release_independently)
        with preparing.installed():
            releaser.start()
            try:
                preparing.action_at = time.perf_counter()
                _send(case, route)
                assert await _until(
                    case.probe.entered.is_set, 10
                ), "Cold Send did not reach the original configuration SQL reader"
                assert await _until(finished.is_set, 2)
                assert await _until(progressed.is_set, 2)
                record_property("cold_send_at_original_configuration_hold", facts)
                record_property("held_reader_release_budget_seconds", 0.5)
                record_property(
                    "send_to_preparing_frame_seconds",
                    None
                    if preparing.frame_at is None
                    else preparing.frame_at - preparing.action_at,
                )
                record_property("preparing_frame_while_held", preparing.while_held)
                record_property("headless_supplied_frame_only", True)
                assert "observer_error" not in facts, facts
                assert facts["native_lease_live"] and not case.probe.release_timed_out
                assert facts["trust_factory_unchanged"]
                assert facts["same_active_session"] and facts["original_draft_kept"]
                # Original-API causal oracle, independent of proposed helper names.
                assert facts[
                    "received_before_native"
                ], "Cold stock Send entered original configuration SQL before receipt"
                assert not facts["configuration_on_input_thread"], facts
                assert facts["loop_progress_while_held"], facts
            finally:
                stop.set()
                case.probe.release.set()
                releaser.join(1)
                assert not releaser.is_alive(), "Cold SQL observer did not retire"
