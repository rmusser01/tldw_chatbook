"""Real saved-session preparation on a reused worker, without a provider call."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pytest

from Tests.Chat.test_console_domain_boundary import domain_case as _domain_case

domain_case = _domain_case
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@asynccontextmanager
async def _routing(monkeypatch, loop, callback):
    with monkeypatch.context() as routing:
        routing.setattr(loop, "run_in_executor", callback)
        yield


@pytest.mark.parametrize("domain_case", ["control_saved"], indirect=True)
@pytest.mark.parametrize("borrowed", [False, True])
async def test_saved_preparations_retire_on_reused_original_worker(
    domain_case, monkeypatch, record_property, borrowed
):
    from Tests.Chat.test_console_domain_boundary import _selection
    from Tests.Chat.test_console_first_send_atomicity import _until
    from tldw_chatbook.Chat.console_control_dispatch import (
        CancelTurn,
        ControlIdentity,
        ReceiveTurn,
        UpdateInputs,
        _command_source,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case = domain_case
    case.controller.prompt_history = PromptHistory(
        case.database.db_path.parent / "history.jsonl"
    )
    case.sources.configuration["api_settings"] = {
        "openai": {"api_key": "task62-local-readiness-only"}
    }
    case.store.replace_session_settings(
        case.session.id,
        ConsoleSessionSettings(
            provider="openai", model="gpt-4o-mini", base_url="http://127.0.0.1:1/v1"
        ),
    )
    case.store.replace_session_trace_privacy_override(
        case.session.id,
        capture_enabled=False,
        pii_redaction_enabled=None,
        expected_policy_revision=case.store.capture_policy_state(
            case.session.id
        ).policy_revision,
    )
    if not borrowed:
        conversation_id = case.store.persistence.create_conversation(
            conversation_title="Reused preparation worker"
        )
        case.store.publish_first_persisted_conversation(
            case.session.id, conversation_id
        )
    await (
        case.runtime._execution_service.ensure_permission_operations().set_kill_switch(
            False
        )
    )
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="retirement-profile", service_epoch="retirement-service"
    )

    def identity(sequence):
        return ControlIdentity(
            "retirement-profile",
            "retirement-service",
            sequence,
            f"reused-command-{sequence}",
            case.session.id,
            case.session.incarnation_id,
        )

    loop = asyncio.get_running_loop()
    original_submit = loop.run_in_executor
    retained = []

    @asynccontextmanager
    async def independent_connection(executor):
        connection = None
        if borrowed:
            connection = await original_submit(executor, case.database.get_connection)
        try:
            yield connection
        finally:
            if connection is not None:
                # Its real original caller retains ownership; preparation must
                # not close this thread's already-open handle on its behalf.
                def release_original():
                    assert case.database._local.conn is connection
                    assert connection.execute("SELECT 1").fetchone()[0] == 1
                    case.database.close_connection()

                await original_submit(executor, release_original)

    with ThreadPoolExecutor(max_workers=1) as executor:

        def route_original_read(selected, callback, *args):
            source = _command_source.get()
            if selected is None and source is not None and source.controls is controls:
                selected = executor
            return original_submit(selected, callback, *args)

        async with (
            independent_connection(executor),
            _routing(monkeypatch, loop, route_original_read),
        ):
            for attempt in range(2):
                sequence = attempt * 3 + 1
                turn_id = f"reused-turn-{attempt}"
                controls.dispatch(
                    UpdateInputs(identity(sequence), "original draft", (sequence, 1))
                )
                receipt = controls.dispatch(
                    ReceiveTurn(
                        identity(sequence + 1),
                        turn_id,
                        case.session.draft_revision,
                        _selection(case),
                    )
                )
                invocation = receipt.turn_retirement
                try:
                    assert await _until(
                        lambda turn_id=turn_id, receipt=receipt: any(
                            plan.turn_id == turn_id for plan in controls.commit_plans
                        )
                        or receipt.work.done()
                    )
                    if receipt.work.done():
                        receipt.work.result()
                    reads = tuple(invocation.reads)
                    retained.append(reads)
                    record_property(f"attempt_{attempt}_read_count", len(reads))
                    assert reads
                    assert all(read.read.task is receipt.work for read in reads)
                    # Re-evaluate the original proof after all subsequent reads,
                    # including original commit reservation, have completed.
                    assert all(
                        read.confirmed() for cohort in retained for read in cohort
                    )
                    cancelled = controls.dispatch(
                        CancelTurn(identity(sequence + 2), turn_id)
                    )
                    assert cancelled.retired is receipt.retired
                    outcomes = await asyncio.gather(
                        receipt.work, return_exceptions=True
                    )
                    assert isinstance(outcomes[0], asyncio.CancelledError)
                    assert await asyncio.shield(receipt.retired) is True
                    assert invocation.logical_closed and invocation.source_closed
                    assert turn_id not in controls._turn_retirements
                finally:
                    if not receipt.work.done():
                        receipt.work.cancel()
                    async with asyncio.timeout(10):
                        await asyncio.gather(receipt.work, return_exceptions=True)
                        if invocation.settlement is not None:
                            await asyncio.gather(
                                invocation.settlement, return_exceptions=True
                            )
