"""Screen-level tests: the cost chip must not re-tokenize a frozen transcript.

task-15451. ``_sync_console_cost_chip`` runs on the 0.2 s tick for the whole
duration of any active run (plus every control-bar sync pass and the 10 s TTL
timer), and its equality guard gates only the REPAINT -- the state build itself
ran unconditionally, re-running ``_estimate_tokens_locally`` over every
usage-less row every single time. With tiktoken absent from base deps
(task-2526) that estimator is a per-character Python loop, so this was
O(transcript chars) on the event loop, five times a second.

These tests count estimator calls rather than timing anything: a second
IDENTICAL build must tokenize nothing, and a build after a one-row edit must
tokenize exactly that one row. The chip's SEMANTICS (mid-stream freeze, staged
evidence, ``~`` prefix, TTL states, fingerprint gating) are pinned unchanged by
Tests/UI/test_console_cost_chip_screen.py, which this file deliberately does
not modify.
"""

from __future__ import annotations

from tldw_chatbook.UI.Console_Modules import context_spend as context_spend_module

import time
from unittest.mock import Mock

import pytest
from Tests.private_profile import private_profile_test

from Tests.UI.test_console_cost_chip_screen import (
    WARM_USAGE,
    _AnthropicCostGateway,
    _configure_anthropic_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

from tldw_chatbook.Chat import console_cost_tracker as cost_tracker_module
from tldw_chatbook.Chat import console_session_settings as settings_module
from tldw_chatbook.Chat.citation_evidence_models import (
    EvidenceBundle,
    EvidenceReference,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module

_TRANSCRIPT_ROWS = 12
_ROW_TEXT = "the quick brown fox jumps over the lazy dog. " * 40


def _seed_usageless_transcript(console) -> tuple[object, str]:
    """Append rows with real text and no ``ProviderUsage`` -- the estimated
    rows that ``build_cost_snapshot`` prices with the local estimator."""
    store = console._ensure_console_chat_store()
    session_id = store.active_session_id
    for index in range(_TRANSCRIPT_ROWS):
        store.append_message(
            session_id,
            role=(
                ConsoleMessageRole.USER
                if index % 2 == 0
                else ConsoleMessageRole.ASSISTANT
            ),
            content=f"row {index}: {_ROW_TEXT}",
            persist=False,
        )
    return store, session_id


def _spy_on_estimator(monkeypatch) -> Mock:
    spy = Mock(wraps=cost_tracker_module._estimate_tokens_locally)
    monkeypatch.setattr(cost_tracker_module, "_estimate_tokens_locally", spy)
    return spy


@pytest.mark.asyncio
@private_profile_test
async def test_second_identical_tick_does_not_retokenize_the_transcript(
    monkeypatch, request
):
    """The headline defect: two identical ticks, two full re-tokenizations."""
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")
        _seed_usageless_transcript(console)

        spy = _spy_on_estimator(monkeypatch)

        first = console._context_spend._build_console_cost_state()
        first_calls = spy.call_count
        assert (
            first_calls == _TRANSCRIPT_ROWS
        ), "test setup: every seeded row must be an estimated row"

        second = console._context_spend._build_console_cost_state()

        assert spy.call_count == first_calls, (
            "the cost chip re-tokenized an unchanged transcript on the next "
            f"tick ({spy.call_count - first_calls} extra estimator calls)"
        )
        assert second == first


@pytest.mark.asyncio
@private_profile_test
async def test_editing_one_row_retokenizes_only_that_row(monkeypatch, request):
    """O(changed), not O(transcript): one edited row costs one estimate."""
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")
        store, session_id = _seed_usageless_transcript(console)

        console._context_spend._build_console_cost_state()
        spy = _spy_on_estimator(monkeypatch)

        target = store.messages_for_session(session_id)[-1]
        store.update_message_content(target.id, "a different, much shorter row")
        state = console._context_spend._build_console_cost_state()

        assert spy.call_count == 1, (
            "editing one row re-tokenized "
            f"{spy.call_count} rows -- the other rows are unchanged"
        )
        assert state is not None


@pytest.mark.asyncio
@private_profile_test
async def test_late_terminal_usage_replaces_settled_cost_without_payload_edit(
    monkeypatch, request
):
    """A stopped answer's delayed provider usage must reprice Current."""
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        store.append_message(
            session_id, role=ConsoleMessageRole.USER, content="question"
        )
        answer = store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content="answer"
        )
        before = console._context_spend._build_console_cost_state()
        payload_revision = store.payload_revision(session_id)
        store.set_message_usage(
            answer.id,
            ProviderUsage(
                uncached_input=100_000,
                output=20_000,
                provider="anthropic",
                model="claude-sonnet-4-6",
            ),
        )
        after = console._context_spend._build_console_cost_state()

        assert store.payload_revision(session_id) == payload_revision
        assert before is not None and after is not None
        assert before != after
        assert "Current $0.60" in after.label


@pytest.mark.asyncio
@private_profile_test
async def test_edited_row_is_repriced_not_served_stale(monkeypatch, request):
    """The other half of the guarantee: a cached row must never outlive its
    content. Shrinking one row's text has to move the reported total."""
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")
        store, session_id = _seed_usageless_transcript(console)

        before = console._context_spend._build_console_cost_state()
        target = store.messages_for_session(session_id)[-1]
        original_content = target.content
        store.update_message_content(target.id, "tiny")
        after = console._context_spend._build_console_cost_state()

        assert before is not None and after is not None
        assert after.tooltip != before.tooltip
        # And restoring the original text restores the original reading.
        store.update_message_content(target.id, original_content)
        restored = console._context_spend._build_console_cost_state()
        assert restored is not None
        assert restored.tooltip == before.tooltip


@pytest.mark.asyncio
@private_profile_test
async def test_staged_evidence_row_is_not_retokenized_every_tick(monkeypatch, request):
    """The staged-evidence context text is estimated once while unchanged."""
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")
        before_used = console._context_spend._active_console_settings_context_estimate().used_tokens

        reference = EvidenceReference(
            evidence_id="S1",
            source_id="media-1",
            source_type="media",
            title="Big corpus",
            snippet="corpus text " * 20_000,
            authority_label="local",
            status="available",
            source_owner="local",
        )
        bundle = EvidenceBundle(
            bundle_id="bundle-big",
            query="question",
            source="Library Search/RAG",
            references=(reference,),
        )
        # The staged text contributes to the context/next-send estimate,
        # which calls the session-settings estimator rather than the
        # settled-spend row estimator.
        spy = Mock(wraps=settings_module._estimate_tokens_locally)
        monkeypatch.setattr(settings_module, "_estimate_tokens_locally", spy)
        console._retrieval._stage_console_library_rag_launch(
            ConsoleLiveWorkLaunch.from_values(
                source="Library Search/RAG",
                title="Library Search/RAG retrieval",
                payload={"query": "question", "evidence_bundle": bundle.to_payload()},
                status="staged",
            )
        )
        await pilot.pause()

        first = console._context_spend._build_console_cost_state()
        settled_calls = spy.call_count
        assert settled_calls >= 1, "test setup: staged text must be estimated"
        assert first is not None
        assert (
            console._context_spend._active_console_settings_context_estimate().used_tokens
            > before_used
        )

        second = console._context_spend._build_console_cost_state()

        assert (
            spy.call_count == settled_calls
        ), "the staged-evidence context was re-tokenized on the next tick"
        assert second == first


@pytest.mark.asyncio
@private_profile_test
async def test_projected_delta_estimate_is_not_recomputed_every_tick(request):
    """A mounted warm-cache alert estimates its transcript only once."""
    gateway = _AnthropicCostGateway(WARM_USAGE, reply="warm reply")
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        user_message = store.append_message(
            session_id,
            role=ConsoleMessageRole.USER,
            content="ORIGINAL EARLIER HISTORY",
            persist=False,
        )
        controller = console._ensure_console_chat_controller()
        controller._payload_fingerprint_baselines[session_id] = (
            controller.compute_current_fingerprint(session_id)
        )
        controller._cache_last_activity[session_id] = True
        controller._cache_warm_until[session_id] = time.monotonic() + 300.0
        store.update_message_content(user_message.id, "EDITED EARLIER HISTORY")

        spy = Mock(wraps=context_spend_module._estimate_tokens_locally)
        original = context_spend_module._estimate_tokens_locally
        context_spend_module._estimate_tokens_locally = spy
        try:
            alert_state = console._context_spend._build_console_cost_state()
            assert alert_state is not None and alert_state.alert is True
            assert spy.call_count == 1, "test setup: the projection must run once"

            repeat_state = console._context_spend._build_console_cost_state()

            assert spy.call_count == 1, (
                "the projected cache-break delta re-tokenized the whole "
                "transcript on an unchanged tick"
            )
            assert repeat_state is not None
            assert repeat_state.label == alert_state.label
            assert repeat_state.alert is True
        finally:
            context_spend_module._estimate_tokens_locally = original
