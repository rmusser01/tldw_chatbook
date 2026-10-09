"""Driver Send control for the existing original stock trust lifetime fixture.

The integration owner invokes this while that fixture holds its actual startup
builder. This module owns neither App construction nor the builder's native
lifetime; the original fixture must verify both after this control returns.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_configuration_worker_lifetime import (
    _OriginalConfigurationWorkspaceRead,
)
from Tests.UI.test_console_hook_review_send_freeze import _until
from Tests.UI.test_console_received_intent_feedback import (
    _NaturalPreparingFrame,
    _send,
)


def prepare_cold_send_workspace(app, case):
    """Select a real workspace before original mount/reconciliation."""
    assert case in {"send_enter", "send_button"}
    registry = app.workspace_registry_service
    workspace_id = f"stock-cold-{case}"
    registry.create_workspace(workspace_id=workspace_id, name=workspace_id)
    registry.set_active_workspace(workspace_id)
    return workspace_id


async def observe_cold_send(app, screen, case, workspace_id, release_trust, facts):
    """Record receipt at actual SQL entry; release holds before causal assertions.

    Caller supplies the existing startup callback's release event and keeps its
    unchanged ten-second control deadline. Causal assertions belong after the
    original fixture has verified App/native/creator retirement.
    """
    assert case in {"send_enter", "send_button"}
    route = "enter" if case == "send_enter" else "send-button"
    loop = asyncio.get_running_loop()
    assert loop.get_task_factory() is None
    runtime = app.console_runtime
    assert screen is runtime.view and screen is app.screen
    assert screen.app is app and screen.app_instance is app
    local = vars(app)["_local_skills_service"]
    factory = vars(local)["_trust_service_factory"]
    controller = screen._ensure_console_chat_controller()
    store = controller.store
    probe = _OriginalConfigurationWorkspaceRead(app.workspace_registry_service)
    stop, progressed, finished = threading.Event(), threading.Event(), threading.Event()
    input_thread = threading.current_thread()
    releaser = None
    observed = facts["cold_send"] = {"route": route, "sql_release_budget_s": 0.5}
    calls = []

    def reply(**kwargs):
        calls.append(kwargs)
        return "original cold Send reply"

    try:
        assert await _until(
            lambda: screen._console_attach_reconciled
            and not screen._console_attach_reconcile_running,
            2,
        ), "Original attachment did not settle during the held startup builder"
        initial = app._initial_screen_setup_task
        assert (
            initial.done() and not initial.cancelled() and initial.exception() is None
        )
        session = store.ensure_session()
        assert session.workspace_id == workspace_id
        assert store.active_session_id == session.id
        assert screen._console_visible_draft_session_id == session.id
        composer = screen._console_composer_or_none()
        draft = "An original cold Send draft"
        composer.load_draft(draft)
        composer.focus()
        assert await _until(lambda: app.focused is composer, 1)
        assert composer.draft_text() == store.session_draft(session.id) == draft
        assert vars(app)["_local_skill_trust_service"] is None
        assert vars(local)["_trust_service"] is None
        assert not release_trust.is_set()
        context = SimpleNamespace(
            host=app,
            console=screen,
            probe=probe,
            composer=composer,
        )
        preparing = _NaturalPreparingFrame(context)

        def note_progress():
            observed["loop_progress_while_sql_held"] = not probe.release.is_set()
            progressed.set()

        def release_sql_independently():
            try:
                while not probe.entered.wait(0.01):
                    if stop.is_set():
                        return
                claim = store.received_turn_for_session(session.id)
                record = (
                    runtime._turn_custody.get(claim.request_id)
                    if claim is not None
                    else None
                )
                intent = getattr(record, "received_intent", None)
                observed.update(
                    received_before_native=(
                        claim is not None
                        and record is not None
                        and record.received_claim is claim
                        and record.request is None
                        and intent is not None
                        and intent.session_id == session.id
                        and intent.inputs.draft == draft
                    ),
                    original_draft_kept=store.session_draft(session.id) == draft,
                    same_active_session=store.active_session_id == session.id,
                    configuration_on_input_thread=probe.thread is input_thread,
                    native_lease_live=probe.live_at_entry,
                    trust_still_cold=(
                        vars(app)["_local_skill_trust_service"] is None
                        and vars(local)["_trust_service"] is None
                        and not release_trust.is_set()
                    ),
                    trust_factory_unchanged=(
                        vars(local)["_trust_service_factory"] is factory
                    ),
                    provider_calls_at_sql_entry=len(calls),
                )
                loop.call_soon_threadsafe(note_progress)
                deadline = time.perf_counter() + 0.5
                progressed.wait(0.5)
                preparing.painted.wait(max(0, deadline - time.perf_counter()))
            except BaseException as error:
                observed["observer_error"] = type(error).__name__
            finally:
                probe.release.set()
                finished.set()

        releaser = threading.Thread(
            target=release_sql_independently, name="original-cold-Send-SQL-observer"
        )
        with pytest.MonkeyPatch.context() as patch, probe.installed(), preparing.installed():
            # This is the only replaced product boundary: the remote response.
            patch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", reply)
            releaser.start()
            try:
                preparing.action_at = time.perf_counter()
                _send(context, route)
                assert await _until(
                    probe.entered.is_set, 3
                ), "Cold Send did not reach original configuration SQL"
                assert await _until(finished.is_set, 1)
                assert await _until(progressed.is_set, 1)
                observed.update(
                    sql_release_timed_out=probe.release_timed_out,
                    preparing_frame_while_held=preparing.while_held,
                    send_to_preparing_frame_seconds=(
                        None
                        if preparing.frame_at is None
                        else preparing.frame_at - preparing.action_at
                    ),
                    headless_supplied_frame_only=True,
                )
            finally:
                # Never make original native/App retirement depend on a RED.
                stop.set()
                probe.release.set()
                release_trust.set()
                controller.stop_active_run()
                app.workers.cancel_group(screen, "console-hook-send-review")
                tasks = tuple(
                    row.task
                    for row in runtime._turn_custody.values()
                    if row.session_id == session.id and row.task is not None
                )
                for task in tasks:
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), 2
                    )
                assert await _until(lambda: not screen._hooks._busy, 2)
                releaser.join(1)
                assert not releaser.is_alive()
                observed["independent_sql_observer_retired"] = True
    finally:
        stop.set()
        probe.release.set()
        release_trust.set()
        if releaser is not None and releaser.ident is not None:
            releaser.join(1)
            assert not releaser.is_alive()


def assert_cold_send_receipt(facts):
    """Call only after the original App fixture's physical retirement checks."""
    observed = facts["cold_send"]
    assert "observer_error" not in observed, observed
    assert observed["native_lease_live"] and not observed["sql_release_timed_out"]
    assert observed["trust_still_cold"] and observed["trust_factory_unchanged"]
    assert observed["same_active_session"] and observed["original_draft_kept"]
    assert observed["provider_calls_at_sql_entry"] == 0
    assert observed["independent_sql_observer_retired"]
    assert observed[
        "received_before_native"
    ], "Cold stock Send entered original configuration SQL before receipt"
    assert not observed["configuration_on_input_thread"], observed
    assert observed["loop_progress_while_sql_held"], observed
