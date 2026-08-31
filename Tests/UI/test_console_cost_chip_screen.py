"""Screen-level tests for the Console cost chip (PR3 task-5).

Covers ``ChatScreen._build_console_cost_state``/``_sync_console_cost_chip``,
the 10s WARM->EXPIRED TTL repaint timer, and the ``ConsoleCostChipPressed``
breakdown-modal handler -- the integration task that wires together the
tracker math (Tests/Chat/test_console_cost_tracker.py), the controller's
fingerprint/cache-TTL ground truth (task-3), and the chip widget itself
(Tests/Chat/test_console_status_chips_cost.py).

Harness: mirrors ``Tests/UI/test_console_native_chat_flow.py`` -- a real
mounted ``ChatScreen`` behind ``ConsoleHarness``, a stub provider gateway
injected via ``app.console_provider_gateway_factory`` (never a real network
call), and ``_wait_for_visible_text``/``_wait_for_selector`` from
``test_destination_shells`` to settle on real UI state instead of sleeping.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import attach_chachanotes_db
from Tests.UI.test_destination_shells import (
    _visible_text,
    _wait_for_selector,
    _wait_for_visible_text,
)
from Tests.UI.test_console_native_chat_flow import (
    _build_console_send_test_app as _build_test_app,
)
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

from tldw_chatbook.Chat.citation_evidence_models import (
    EvidenceBundle,
    EvidenceReference,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCacheState
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleCostChip
from tldw_chatbook.config import load_settings, save_settings_to_cli_config
from Tests.console_provider_doubles import provider_resolution

_ASYNC_SETTLE_TIMEOUT = 10.0

# Anthropic-native usage-payload shapes (see ProviderUsage.from_provider_payload):
# a plain priced call with no cache activity, and one that reports a cache
# WRITE so the controller's `_attach_stream_usage` stamps `_cache_warm_until`.
PRICED_USAGE = {
    "input_tokens": 1000,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "output_tokens": 200,
}
WARM_USAGE = {
    "input_tokens": 200,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 800,
    "output_tokens": 120,
}


def _configure_anthropic_ready_console(app, model: str = "claude-sonnet-4-6") -> None:
    """Configure a send-ready Console on a priced Anthropic model.

    Mirrors ``test_console_native_chat_flow._configure_native_ready_console``
    for llama.cpp: sets both ``chat_defaults`` (the new session's provider/
    model at mount) and ``api_settings`` (so
    ``build_console_settings_readiness`` sees a configured key and doesn't
    gate the send behind the first-run setup modal).
    """
    assert save_settings_to_cli_config(
        {
            "chat_defaults": {"provider": "anthropic", "model": model},
            "api_settings.anthropic": {"api_key": "test-anthropic-key"},
        }
    )
    app.app_config = load_settings()
    app.chat_api_provider_value = "anthropic"
    app.chat_api_model_value = model


class _AnthropicCostGateway:
    """Stub gateway: ready Anthropic resolution (prompt caching on) that
    records a fixed usage payload through ``signals`` on every call -- the
    minimum shape needed to drive the cost chip end to end without a real
    network call (see ``ConsoleProviderStreamSignals.record_usage_payload``,
    read by the controller's ``_attach_stream_usage``)."""

    def __init__(self, usage_payload: dict, *, reply: str) -> None:
        self._usage_payload = usage_payload
        self._reply = reply
        self.sent_messages: list[list[dict]] = []

    async def resolve_for_send(self, selection):
        return provider_resolution(
            provider="anthropic",
            base_url=selection.base_url or "",
            model=selection.explicit_model
            or selection.configured_model
            or "claude-sonnet-4-6",
            ready=True,
            visible_copy="",
            prompt_caching=True,
        )

    async def stream_chat(self, resolution, messages, **kwargs):
        self.sent_messages.append(list(messages))
        signals = kwargs.get("signals")
        if signals is not None:
            signals.record_usage_payload(self._usage_payload)
        yield self._reply


class _AnthropicWaitingGateway:
    """Stub gateway that stalls mid-stream until released -- the STREAMING
    window used to prove the fingerprint recompute is skipped while active
    (mirrors ``test_console_native_chat_flow.WaitingGateway``)."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def resolve_for_send(self, selection):
        return provider_resolution(
            provider="anthropic",
            base_url=selection.base_url or "",
            model=selection.explicit_model
            or selection.configured_model
            or "claude-sonnet-4-6",
            ready=True,
            visible_copy="",
            prompt_caching=True,
        )

    async def stream_chat(self, resolution, messages, **kwargs):
        yield "partial"
        self.started.set()
        await self.release.wait()
        yield " done"


async def _send_and_settle(console, pilot, draft: str, expect_text: str) -> None:
    """Load a composer draft, press Send, and wait for the reply to land."""
    await _wait_for_selector(console, pilot, "#console-native-composer")
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.load_draft(draft)
    console.query_one("#console-send-message", Button).press()
    await _wait_for_visible_text(console, pilot, expect_text)
    # The 0.2s tick's own post-completion sync races test assertions; force
    # one more deterministic sync (this is exactly what that tick calls).
    await console._sync_native_console_chat_ui()
    await pilot.pause()


# --- (a) priced send shows a real dollar figure -----------------------------


@pytest.mark.asyncio
async def test_cost_chip_shows_dollar_figure_after_priced_send():
    gateway = _AnthropicCostGateway(PRICED_USAGE, reply="the priced answer")
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _send_and_settle(console, pilot, "hello", "the priced answer")

        chip = console.query_one("#console-cost-chip")
        assert chip.display is True
        rendered = str(chip.render())
        assert "$" in rendered

        state = console._last_console_cost_state
        assert state is not None
        assert state.alert is False


# --- (b)/(c) editing earlier history alerts a WARM cache, reverting clears --


async def _mount_and_send_warm_reply(console, pilot):
    """Send one turn whose usage warms the Anthropic prompt cache.

    Returns the store and the active session id once the post-send sync has
    settled, so callers can go straight to editing history.
    """
    await _send_and_settle(console, pilot, "hello", "warm reply")
    store = console._ensure_console_chat_store()
    session_id = store.active_session_id
    controller = console._ensure_console_chat_controller()
    warm_until, had_activity = controller.cache_ttl_snapshot(session_id)
    assert had_activity is True and warm_until is not None, (
        "test setup: the stub usage must actually warm the cache"
    )
    return store, session_id


@pytest.mark.asyncio
async def test_editing_earlier_history_alerts_the_warm_cache_chip():
    gateway = _AnthropicCostGateway(WARM_USAGE, reply="warm reply")
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        store, session_id = await _mount_and_send_warm_reply(console, pilot)

        # A completed send with no edits afterward is not a break -- the
        # baseline's history is a strict prefix of the post-reply history.
        baseline_state = console._last_console_cost_state
        assert baseline_state is not None
        assert baseline_state.alert is False

        user_message = next(
            message
            for message in store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.USER
        )
        store.update_message_content(user_message.id, "EDITED EARLIER HISTORY")

        await console._sync_native_console_chat_ui()
        await pilot.pause()

        state = console._last_console_cost_state
        assert state is not None
        assert state.alert is True
        assert "earlier history changed" in state.tooltip
        # Finding 3 (Qodo round): the alert path is the ONLY path that ever
        # renders `projected_delta_usd` -- confirm narrowing its computation
        # to the break-reason case didn't also drop the rendered figure.
        assert "~+$" in state.tooltip

        chip = console.query_one("#console-cost-chip")
        assert chip.has_class("console-chip-alert")


@pytest.mark.asyncio
async def test_projected_delta_estimator_skipped_when_warm_without_break_reason():
    """Finding 3 (Qodo round, PR3): don't burn a whole-transcript token
    estimate on every sync tick when there's nothing to show it for.

    ``_estimate_tokens_locally`` feeds ``projected_delta_usd``, which
    ``console_cost_tracker.build_cost_state``/``_cache_state_line`` only
    ever read inside their own ``break_reason``-gated branches (the label's
    ``~+$`` suffix and the tooltip's cache-state line) -- with the cache
    WARM but no break reason, the value is computed and then silently
    discarded on every call. This pins that the estimator is skipped in
    that case, and still runs (with the delta rendered) once a break
    reason shows up -- mirrors
    ``test_editing_earlier_history_alerts_the_warm_cache_chip``'s edit
    flow, but spies on the estimator instead of just reading chip state.
    """
    gateway = _AnthropicCostGateway(WARM_USAGE, reply="warm reply")
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        store, session_id = await _mount_and_send_warm_reply(console, pilot)

        # A completed send with no edits afterward is WARM with no break
        # reason -- exactly the case that must skip the estimator.
        baseline_state = console._last_console_cost_state
        assert baseline_state is not None
        assert baseline_state.alert is False
        assert console._console_cost_cache_state == ConsoleCacheState.WARM

        spy = Mock(wraps=chat_screen_module._estimate_tokens_locally)
        original = chat_screen_module._estimate_tokens_locally
        chat_screen_module._estimate_tokens_locally = spy
        try:
            state = console._build_console_cost_state()
            assert spy.call_count == 0, (
                "estimator ran on a WARM cache with no break reason -- "
                "projected_delta_usd is unused without one"
            )
            assert state is not None
            assert state.alert is False

            # Now introduce a break reason (same edit-earlier-history flow
            # as test_editing_earlier_history_alerts_the_warm_cache_chip)
            # and confirm the estimator DOES run, and the alert path still
            # shows the projected delta.
            user_message = next(
                message
                for message in store.messages_for_session(session_id)
                if message.role is ConsoleMessageRole.USER
            )
            store.update_message_content(user_message.id, "EDITED EARLIER HISTORY")

            alert_state = console._build_console_cost_state()

            assert spy.call_count == 1, (
                "estimator must run once a break reason is present"
            )
            assert alert_state is not None
            assert alert_state.alert is True
            assert "~+$" in alert_state.tooltip
        finally:
            chat_screen_module._estimate_tokens_locally = original


@pytest.mark.asyncio
async def test_reverting_the_edit_clears_the_alert():
    gateway = _AnthropicCostGateway(WARM_USAGE, reply="warm reply")
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        store, session_id = await _mount_and_send_warm_reply(console, pilot)

        user_message = next(
            message
            for message in store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.USER
        )
        original_content = user_message.content
        store.update_message_content(user_message.id, "EDITED EARLIER HISTORY")
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        assert console._last_console_cost_state.alert is True

        store.update_message_content(user_message.id, original_content)
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        state = console._last_console_cost_state
        assert state is not None
        assert state.alert is False
        # task-2115: the alert clearing was never in question -- what the
        # live-provider verification flagged is whether the cache-state
        # READOUT underneath it also comes back to plain warm, rather than
        # landing on cold/expired despite real TTL time remaining (no clock
        # manipulation happened between the edit and the revert here, so
        # the deadline recorded by the original send is still comfortably
        # in the future).
        assert state.cold is False
        assert console._console_cost_cache_state == ConsoleCacheState.WARM

        chip = console.query_one("#console-cost-chip")
        assert not chip.has_class("console-chip-alert")
        assert not chip.has_class("console-chip-cold")


# --- task-2115: revert-with-TTL-remaining vs. genuine TTL lapse -------------


async def _mount_and_send_warm_system_prompt(console, pilot):
    """Send one turn that warms the cache, matching
    ``_mount_and_send_warm_reply`` but returning the store/session id the
    same way for the system-prompt-edit tests below."""
    await _send_and_settle(console, pilot, "hello", "warm reply")
    store = console._ensure_console_chat_store()
    session_id = store.active_session_id
    controller = console._ensure_console_chat_controller()
    warm_until, had_activity = controller.cache_ttl_snapshot(session_id)
    assert had_activity is True and warm_until is not None, (
        "test setup: the stub usage must actually warm the cache"
    )
    return store, session_id, controller


@pytest.mark.asyncio
async def test_reverting_system_prompt_edit_with_ttl_remaining_returns_to_warm():
    """Confirm reverting a system-prompt edit with TTL time remaining returns the chip to warm, not cold.

    No clock manipulation happens here, so the 300s TTL recorded by the
    original send is still comfortably in the future throughout -- the
    scripted reproduction that settled task-2115.
    """
    gateway = _AnthropicCostGateway(WARM_USAGE, reply="warm reply")
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        store, session_id, controller = await _mount_and_send_warm_system_prompt(
            console, pilot
        )
        warm_until_before, _ = controller.cache_ttl_snapshot(session_id)

        console._session._apply_console_session_system_prompt("You are a pirate.")
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        edited_state = console._last_console_cost_state
        assert edited_state is not None
        assert edited_state.alert is True
        assert "system prompt changed" in edited_state.tooltip

        # Revert to no override -- same session, no new send, no elapsed
        # clock beyond real test-execution time (well under the 300s TTL).
        console._session._apply_console_session_system_prompt(None)
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        reverted_state = console._last_console_cost_state
        assert reverted_state is not None
        assert reverted_state.alert is False
        assert reverted_state.cold is False
        assert "Cache: warm" in reverted_state.tooltip
        assert "expired" not in reverted_state.tooltip
        assert console._console_cost_cache_state == ConsoleCacheState.WARM

        # The deadline itself is untouched by the edit/revert round trip --
        # not just "still warm" but the SAME deadline the original send set.
        warm_until_after, had_activity_after = controller.cache_ttl_snapshot(session_id)
        assert had_activity_after is True
        assert warm_until_after == warm_until_before

        chip = console.query_one("#console-cost-chip")
        assert not chip.has_class("console-chip-alert")
        assert not chip.has_class("console-chip-cold")


@pytest.mark.asyncio
async def test_system_prompt_revert_after_genuine_ttl_lapse_still_reports_expired():
    """Confirm a break-then-revert sequence still reports expired once the TTL deadline has genuinely passed.

    The companion case to the test above: the deadline lapse is simulated
    by pushing the controller's own ``_cache_warm_until`` entry into the
    past (never by freezing or jumping the real monotonic clock a running
    app also depends on).
    """
    gateway = _AnthropicCostGateway(WARM_USAGE, reply="warm reply")
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        store, session_id, controller = await _mount_and_send_warm_system_prompt(
            console, pilot
        )

        console._session._apply_console_session_system_prompt("You are a pirate.")
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        assert console._last_console_cost_state.alert is True

        console._session._apply_console_session_system_prompt(None)
        # The deadline has genuinely lapsed by the time the revert is
        # observed -- e.g. a slow manual round trip through the editor
        # modal, exactly the real-world scenario the live verification hit.
        controller._cache_warm_until[session_id] = time.monotonic() - 1.0
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        state = console._last_console_cost_state
        assert state is not None
        assert state.alert is False
        assert state.cold is True
        assert "Cache: expired" in state.tooltip
        assert console._console_cost_cache_state == ConsoleCacheState.EXPIRED

        chip = console.query_one("#console-cost-chip")
        assert chip.has_class("console-chip-cold")


# --- (d) no active native session -> None / hidden --------------------------


def test_build_console_cost_state_returns_none_without_native_session():
    """Mirrors the unmounted-``ChatScreen`` unit-test idiom already used for
    other builder methods (e.g. ``test_console_store_uses_app_citation_
    repository_for_matching_database`` in test_console_native_chat_flow.py)
    -- no store has been created yet, so there is no active native session."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    screen = ChatScreen(app)
    assert screen._console_chat_store is None

    state = screen._build_console_cost_state()

    assert state is None
    assert screen._console_cost_cache_state == ConsoleCacheState.NONE


@pytest.mark.asyncio
async def test_build_console_cost_state_counts_staged_evidence_before_send():
    """task-6: reproduces the critique's exact symptom -- "0 tok" with five
    sources staged (one a 942 KB corpus), none of it sent yet. Staged,
    unsent evidence must price into the chip as an ESTIMATED row so the
    total moves off zero and the `~` honesty marker (an unsent estimate,
    not a real spend) appears -- even though the session has zero
    messages."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")

        empty_state = console._build_console_cost_state()
        assert empty_state is not None
        assert "Tokens: 0" in empty_state.tooltip

        big_reference = EvidenceReference(
            evidence_id="S1",
            source_id="media-1",
            source_type="media",
            title="Big corpus",
            snippet="corpus text " * 80_000,  # ~960 KB before staging's own cap
            authority_label="local",
            status="available",
            source_owner="local",
        )
        bundle = EvidenceBundle(
            bundle_id="bundle-big",
            query="question",
            source="Library Search/RAG",
            references=(big_reference,),
        )
        launch = ConsoleLiveWorkLaunch.from_values(
            source="Library Search/RAG",
            title="Library Search/RAG retrieval",
            payload={"query": "question", "evidence_bundle": bundle.to_payload()},
            status="staged",
        )
        console._retrieval._stage_console_library_rag_launch(launch)
        await pilot.pause()

        state = console._build_console_cost_state()

        assert state is not None
        assert "Tokens: 0" not in state.tooltip
        assert "includes estimated" in state.tooltip.lower()
        assert state.label.startswith("~")


@pytest.mark.asyncio
async def test_build_console_cost_state_includes_fleet_token_spend():
    """PR2b Task 5 (cost rollup): the active conversation's LIVE sub-agent
    fleet spend reaches the cost chip's tooltip -- the aggregate this task
    wires from `ConsoleAgentController._console_agent_fleet_token_total`
    into `build_cost_snapshot`'s `fleet_tokens` keyword. Overrides the
    controller method directly (rather than wiring a fake bridge through
    the rail's own conversation-id plumbing) so this test pins the ONE
    thing new here -- that `_build_console_cost_state` actually reads and
    forwards that seam -- without re-deriving the fleet-token-total math
    itself (already covered in `Tests/UI/test_console_agent_rail.py`).
    """
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")

        baseline = console._build_console_cost_state()
        assert baseline is not None
        assert "Sub-agents:" not in baseline.tooltip

        console._agent._console_agent_fleet_token_total = lambda: 4200

        state = console._build_console_cost_state()
        assert state is not None
        assert "Sub-agents: 4.2k tok (not priced)" in state.tooltip


@pytest.mark.asyncio
async def test_sync_cost_chip_hides_the_chip_when_state_is_none():
    app = _build_test_app()
    attach_chachanotes_db(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")
        # A freshly mounted Console session has no usage/cost yet, so the
        # snapshot is empty and the chip renders hidden either way -- force
        # the "no session at all" branch directly, matching how
        # `_sync_console_cost_chip` is actually invoked off the sync path.
        console._console_chat_store = None
        console._sync_console_cost_chip()
        await pilot.pause()

        chip = console.query_one("#console-cost-chip")
        assert chip.display is False


# --- (e) TTL: past warm_until flips EXPIRED/cold and stops the timer --------


@pytest.mark.asyncio
async def test_ttl_timer_expires_the_chip_and_stops_itself():
    gateway = _AnthropicCostGateway(WARM_USAGE, reply="warm reply")
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        store, session_id = await _mount_and_send_warm_reply(console, pilot)

        assert console._console_cost_cache_state == ConsoleCacheState.WARM
        assert console._console_cost_ttl_timer is not None

        # Push the recorded warm-until deadline into the past directly on
        # the controller's own ground-truth map, rather than monkeypatching
        # the global `time.monotonic` -- Textual's own event loop/animation
        # scheduling depends on real monotonic time too, and freezing/
        # jumping it out from under a *running* app hangs the test.
        controller = console._ensure_console_chat_controller()
        controller._cache_warm_until[session_id] = time.monotonic() - 1.0

        # Simulate one TTL-timer tick without waiting 10 real seconds --
        # the timer's callback IS `_sync_console_cost_chip` itself.
        console._sync_console_cost_chip()
        await pilot.pause()

        assert console._console_cost_cache_state == ConsoleCacheState.EXPIRED
        state = console._last_console_cost_state
        assert state is not None
        assert state.cold is True
        assert console._console_cost_ttl_timer is None

        chip = console.query_one("#console-cost-chip")
        assert chip.has_class("console-chip-cold")


# --- (f) fingerprint recompute is skipped while STREAMING -------------------


@pytest.mark.asyncio
async def test_fingerprint_recompute_is_skipped_while_streaming():
    gateway = _AnthropicWaitingGateway()
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello")
        console.query_one("#console-send-message", Button).press()
        await asyncio.wait_for(gateway.started.wait(), timeout=_ASYNC_SETTLE_TIMEOUT)
        await _wait_for_visible_text(console, pilot, "partial")

        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        controller = console._ensure_console_chat_controller()
        assert controller.run_state_for(session_id).status is ConsoleRunStatus.STREAMING

        before_memo = console._console_cost_fp_revisions.get(session_id)

        settings = store.session_settings(session_id)
        assert settings is not None
        store.replace_session_settings(
            session_id,
            replace(settings, system_prompt="bumped-while-streaming"),
        )
        bumped_revision = store.payload_revision(session_id)

        console._build_console_cost_state()

        assert console._console_cost_fp_revisions.get(session_id) == before_memo
        assert console._console_cost_fp_revisions.get(session_id) != bumped_revision

        gateway.release.set()
        await _wait_for_visible_text(console, pilot, "partial done")


# --- Breakdown modal ----------------------------------------------------


@pytest.mark.asyncio
async def test_cost_chip_press_opens_the_breakdown_modal():
    """task-8: the cost chip now opens the shared Conversation Inspector
    (not the retired standalone cost modal, deleted in task-10), starting
    on its Costs tab -- copies this file's existing chip-press driver
    verbatim, only the pushed-modal assertion changes."""
    gateway = _AnthropicCostGateway(PRICED_USAGE, reply="the priced answer")
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _send_and_settle(console, pilot, "hello", "the priced answer")

        # The chip strip is a hard-pinned single row after 9 other chips
        # (dossier §1c/§1f) -- at a modest terminal width the cost chip can
        # sit past the visible edge, so drive activation the same way
        # Tests/Chat/test_console_status_chips_cost.py's keyboard-activation
        # test does (`action_open_cost_breakdown`) rather than a real
        # `pilot.click`, which would need an unrealistically wide screen to
        # guarantee the chip is on-screen.
        chip = console.query_one("#console-cost-chip", ConsoleCostChip)
        chip.action_open_cost_breakdown()
        await pilot.pause()

        from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
            TAB_COSTS,
            ConsoleConversationInspector,
        )

        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleConversationInspector)
        assert modal.query_one("#console-inspector-tabs").active == TAB_COSTS
        modal_text = _visible_text(modal)
        assert "Conversation Inspector" in modal_text
        assert "Total" in modal_text


# --- Per-session state isolation across session tabs -------------------


@pytest.mark.asyncio
async def test_cost_chip_state_isolated_across_session_tabs():
    """Design spec's PR3 testing checklist item: "per-session state
    isolation across session tabs."

    Two native Console sessions with different cost profiles -- A gets a
    priced send (real dollar total), B is never sent to (no usage at all,
    so it renders the tokens-only fallback) -- switched between through
    ``_activate_native_console_session``, the same shared tab-click/Ctrl+K/
    Alt+1..9 activation entry point already exercised by
    ``test_switch_between_resumed_sessions_refreshes_stale_workspace_scope``
    (Tests/UI/test_console_scope_row.py) and
    ``test_activate_native_console_session_clears_stale_drilldown``
    (Tests/UI/test_console_agent_rail.py) -- never a raw store mutation.

    ``store.create_session`` activates the session it creates, so B is
    already the active session immediately after creation; calling the
    activation helper with B's own id right then would no-op (its sync
    branch only runs when the requested id differs from the CURRENT active
    session -- see its docstring). The two transitions this test actually
    needs to prove -- "switching TO B shows B's state" and "switching BACK
    to A restores A's state, not a ghost of B's" -- both require a real
    active-session change, so the sequence below hops B -> A -> B -> A,
    each hop through the real entry point.
    """
    gateway = _AnthropicCostGateway(PRICED_USAGE, reply="priced on A")
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        store = console._ensure_console_chat_store()

        session_a = console._session._active_native_console_session()
        assert session_a is not None
        await _send_and_settle(console, pilot, "hello", "priced on A")
        state_a = console._last_console_cost_state
        assert state_a is not None and "$" in state_a.label
        revision_a = console._console_cost_fp_revisions.get(session_a.id)
        assert revision_a is not None

        session_b = store.create_session(title="Session B")
        assert store.active_session_id == session_b.id

        # Hop 1: B (active from creation) -> back to A, through the real
        # switch entry point -- proves restoring an EXISTING session's
        # state on the very first real activation isn't a ghost of
        # whatever the chip last happened to show.
        await console._session._activate_native_console_session(session_a.id)
        await pilot.pause()
        assert console._session._active_native_console_session().id == session_a.id
        assert console._last_console_cost_state == state_a

        # Hop 2: A -> B -- the "switching TO B shows B's state" case.
        await console._session._activate_native_console_session(session_b.id)
        await pilot.pause()
        assert console._session._active_native_console_session().id == session_b.id
        state_b = console._last_console_cost_state
        assert state_b is not None
        assert state_b != state_a
        # B was never sent to -- no priced total, so no dollar glyph at all
        # (build_cost_state's tokens-only fallback for an all-zero snapshot).
        assert "$" not in state_b.label

        # Hop 3: B -> A again -- A's state must come back EXACTLY, not a
        # ghost of B's (e.g. a stale single shared memo instead of a
        # per-session one).
        await console._session._activate_native_console_session(session_a.id)
        await pilot.pause()
        assert console._session._active_native_console_session().id == session_a.id
        assert console._last_console_cost_state == state_a

        # Fingerprint/revision memos are keyed per session and must not
        # cross-contaminate: both ids present as distinct dict keys, and
        # A's own entry is exactly what it was right after the original
        # send -- visiting B in between never touched it.
        assert session_a.id in console._console_cost_fp_revisions
        assert session_b.id in console._console_cost_fp_revisions
        assert session_a.id != session_b.id
        assert console._console_cost_fp_revisions[session_a.id] == revision_a


@pytest.mark.asyncio
async def test_build_console_cost_state_includes_a_survivors_post_turn_spend():
    """PR3a-1 Task 6b (audit F3): the money a child billed AFTER its turn's
    usage was attached reaches the chip.

    It is not on the message row and will not be until PR 3a-2's
    last-child-done re-attach, so the chip's existing unpriced sub-agent
    line is where it is named instead of vanishing. Overrides the
    controller seam directly, exactly as the sibling fleet-token test does,
    so this pins the ONE new thing: that `_build_console_cost_state` reads
    and forwards it.
    """
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")

        baseline = console._build_console_cost_state()
        assert baseline is not None
        assert "Sub-agents:" not in baseline.tooltip

        controller = console._ensure_console_chat_controller()
        assert controller is not None
        controller.unattributed_fleet_tokens = lambda session_id: 1300

        state = console._build_console_cost_state()
        assert state is not None
        assert "Sub-agents: 1.3k tok (not priced)" in state.tooltip


# --- task-25836: first-send context/tool/draft rows -------------------------
#
# The chip's running total ignored everything a FIRST send actually ships
# (session system prompt, tool schemas, the draft itself), so a brand-new
# conversation with a typed draft still read "0 tok" until the first reply
# landed. The fold-in is first-send-only: once a reply (or any usage row)
# exists, the chip reverts to the running-total semantics.


_DEMO_TOOL_SCHEMA = {
    "name": "demo_tool",
    "description": "does demo things",
    "parameters": {"type": "object", "properties": {}},
}


def _first_send_tool_bridge():
    return SimpleNamespace(
        native_tool_schemas=lambda: [_DEMO_TOOL_SCHEMA],
    )


def _row_roles(rows) -> list[str]:
    return [str(getattr(row.role, "value", row.role)) for row in rows]


@pytest.mark.asyncio
async def test_first_send_pseudo_rows_absent_while_draft_is_blank():
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        assert console._console_first_send_pseudo_rows() == []
        state = console._build_console_cost_state()
        assert state is not None
        assert "Tokens: 0" in state.tooltip


@pytest.mark.asyncio
async def test_first_send_pseudo_rows_carry_system_tools_and_draft():
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        console._session._apply_console_session_system_prompt(
            "You are a thorough assistant. " * 5
        )
        console._ensure_console_agent_bridge = _first_send_tool_bridge
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("hello, this is my first message")

        rows = console._console_first_send_pseudo_rows()
        roles = _row_roles(rows)
        contents = [row.content for row in rows]
        assert all(getattr(row, "usage", "missing") is None for row in rows)
        system_contents = [
            content for role, content in zip(roles, contents) if role == "system"
        ]
        assert any("thorough assistant" in content for content in system_contents)
        assert any('"demo_tool"' in content for content in system_contents)
        assert any(
            role == "user" and "first message" in content
            for role, content in zip(roles, contents)
        )

        state = console._build_console_cost_state()
        assert state is not None
        assert "Tokens: 0" not in state.tooltip
        assert state.label.startswith("~")


@pytest.mark.asyncio
async def test_first_send_pseudo_rows_draft_only_without_prompt_or_tools():
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("just a draft")

        rows = console._console_first_send_pseudo_rows()
        assert _row_roles(rows) == ["user"]
        assert "just a draft" in rows[0].content


@pytest.mark.asyncio
async def test_first_send_pseudo_rows_stop_once_a_reply_exists():
    gateway = _AnthropicCostGateway(PRICED_USAGE, reply="the priced answer")
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _send_and_settle(console, pilot, "hello", "the priced answer")

        before = console._build_console_cost_state()
        assert before is not None

        # Everything the fold-in would count, queued behind a finished reply:
        # none of it may leak into the conversation's running total.
        console._session._apply_console_session_system_prompt(
            "You are a thorough assistant. " * 20
        )
        console._ensure_console_agent_bridge = _first_send_tool_bridge
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("a queued follow-up message " * 100)

        assert console._console_first_send_pseudo_rows() == []

        after = console._build_console_cost_state()
        assert after is not None

        def _tokens_line(state):
            return next(
                line
                for line in state.tooltip.splitlines()
                if line.startswith("Tokens:")
            )

        assert _tokens_line(after) == _tokens_line(before)


@pytest.mark.asyncio
async def test_console_next_send_token_estimate_counts_context_not_just_draft():
    """task-25836: the estimate wired into the inspector's Next Send header
    must count the assembled payload (system prompt + draft turn), not the
    draft text alone -- the first-message case where the gap is largest."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
    from tldw_chatbook.Utils.token_counter import estimate_tokens

    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session_id = store.active_session_id or store.ensure_session(
            title="First message"
        ).id
        console._session._apply_console_session_system_prompt(
            "You are a thorough assistant. " * 5
        )
        await pilot.pause()

        snapshot = await controller.build_context_snapshot(
            draft="hello there", session_id=session_id
        )
        estimate = console._console_next_send_token_estimate(snapshot)

        draft_only = estimate_tokens("hello there", "", "")
        assert estimate is not None
        assert estimate > draft_only

        degraded = ConsoleContextSnapshot(
            current_messages=[], next_send_payload={"error": "boom"}
        )
        assert console._console_next_send_token_estimate(degraded) is None
