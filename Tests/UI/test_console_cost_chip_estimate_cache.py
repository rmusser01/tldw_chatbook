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

from unittest.mock import Mock

import pytest

from Tests.UI.test_console_cost_chip_screen import (
    WARM_USAGE,
    _AnthropicCostGateway,
    _configure_anthropic_ready_console,
    _mount_and_send_warm_reply,
    _next_send_dollars,
)
from Tests.UI.app_factory import attach_chachanotes_db
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)

from tldw_chatbook.Chat import console_cost_tracker as cost_tracker_module
from tldw_chatbook.Chat.citation_evidence_models import (
    EvidenceBundle,
    EvidenceReference,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Console_Modules import context_cost as chat_screen_module
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

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
async def test_second_identical_tick_does_not_retokenize_the_transcript(monkeypatch):
    """The headline defect: two identical ticks, two full re-tokenizations."""
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")
        _seed_usageless_transcript(console)

        spy = _spy_on_estimator(monkeypatch)

        first = console._build_console_cost_state()
        first_calls = spy.call_count
        assert first_calls == _TRANSCRIPT_ROWS, (
            "test setup: every seeded row must be an estimated row"
        )

        second = console._build_console_cost_state()

        assert spy.call_count == first_calls, (
            "the cost chip re-tokenized an unchanged transcript on the next "
            f"tick ({spy.call_count - first_calls} extra estimator calls)"
        )
        assert second == first


@pytest.mark.asyncio
async def test_editing_one_row_retokenizes_only_that_row(monkeypatch):
    """O(changed), not O(transcript): one edited row costs one estimate."""
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")
        store, session_id = _seed_usageless_transcript(console)

        console._build_console_cost_state()
        spy = _spy_on_estimator(monkeypatch)

        target = store.messages_for_session(session_id)[-1]
        store.update_message_content(target.id, "a different, much shorter row")
        state = console._build_console_cost_state()

        assert spy.call_count == 1, (
            "editing one row re-tokenized "
            f"{spy.call_count} rows -- the other rows are unchanged"
        )
        assert state is not None


@pytest.mark.asyncio
async def test_edited_row_is_repriced_not_served_stale(monkeypatch):
    """The other half of the guarantee: a cached row must never outlive its
    content. Shrinking one row's text has to move the reported total."""
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")
        store, session_id = _seed_usageless_transcript(console)

        before = console._build_console_cost_state()
        target = store.messages_for_session(session_id)[-1]
        original_content = target.content
        store.update_message_content(target.id, "tiny")
        after = console._build_console_cost_state()

        assert before is not None and after is not None
        assert after.tooltip != before.tooltip
        # And restoring the original text restores the original reading.
        store.update_message_content(target.id, original_content)
        restored = console._build_console_cost_state()
        assert restored is not None
        assert restored.tooltip == before.tooltip


@pytest.mark.asyncio
async def test_staged_evidence_row_is_not_retokenized_every_tick(monkeypatch):
    """Staged evidence belongs to Next Send, never current-spend tokenization."""
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(200, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-cost-chip")

        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("price this request")
        console._sync_console_settings_summary()
        before = console._build_console_cost_state()
        assert before is not None
        assert "Current $0.00" in before.label

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
        # Staging itself drives a sync pass; it must not count unsent evidence
        # as spend there or on any subsequent unchanged tick.
        spy = _spy_on_estimator(monkeypatch)
        console._retrieval._stage_console_library_rag_launch(
            ConsoleLiveWorkLaunch.from_values(
                source="Library Search/RAG",
                title="Library Search/RAG retrieval",
                payload={"query": "question", "evidence_bundle": bundle.to_payload()},
                status="staged",
            )
        )
        await pilot.pause()

        first = console._build_console_cost_state()
        settled_calls = spy.call_count
        assert settled_calls == 0, "unsent evidence must not tokenize as current spend"
        assert first is not None and "Current $0.00" in first.label
        assert _next_send_dollars(first) > _next_send_dollars(before)

        second = console._build_console_cost_state()

        assert spy.call_count == settled_calls, (
            "unsent evidence was tokenized as current spend on the next tick"
        )
        assert second == first


@pytest.mark.asyncio
async def test_projected_delta_estimate_is_not_recomputed_every_tick():
    """The WARM+break_reason projection estimates the WHOLE transcript in one
    call, and (unlike the fingerprint) is recomputed on every build -- so an
    alerting session paid it 5x/s for the life of the alert."""
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
        store.update_message_content(user_message.id, "EDITED EARLIER HISTORY")

        spy = Mock(wraps=chat_screen_module._estimate_tokens_locally)
        original = chat_screen_module._estimate_tokens_locally
        chat_screen_module._estimate_tokens_locally = spy
        try:
            alert_state = console._build_console_cost_state()
            assert alert_state is not None and alert_state.alert is True
            assert spy.call_count == 1, "test setup: the projection must run once"

            repeat_state = console._build_console_cost_state()

            assert spy.call_count == 1, (
                "the projected cache-break delta re-tokenized the whole "
                "transcript on an unchanged tick"
            )
            assert repeat_state == alert_state
        finally:
            chat_screen_module._estimate_tokens_locally = original
