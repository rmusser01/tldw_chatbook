"""Tests for the pure Console next-send pricing tooltip builder."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.citation_evidence_models import (
    EvidenceBundle,
    EvidenceReference,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleNextSendHistoryProjection
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.LLM_Calls.pricing_catalog import ModelPricing
from tldw_chatbook.UI.Console_Modules.send_price import (
    ConsoleNextSendPrice,
    ConsoleSendPriceController,
    build_next_send_price,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import (
    BASE_ACTIONS_WIDTH,
    ConsoleComposerBar,
)


KNOWN_PRICING = ModelPricing(
    input_per_mtok=3.0,
    output_per_mtok=20.4,
    cache_read_per_mtok=None,
    cache_write_per_mtok=None,
    as_of="2026-08-01",
)
ZERO_PRICING = ModelPricing(
    input_per_mtok=0.0,
    output_per_mtok=0.0,
    cache_read_per_mtok=None,
    cache_write_per_mtok=None,
    as_of="2026-08-01",
)


class _Catalog:
    def __init__(self, pricing=KNOWN_PRICING):
        self.pricing = pricing

    def get_pricing(self, provider, model):
        return self.pricing


class _PriceComposerApp(App[None]):
    def __init__(self, tooltip_provider=None):
        super().__init__()
        self.tooltip_provider = tooltip_provider

    def compose(self) -> ComposeResult:
        yield ConsoleComposerBar(
            id="console-native-composer",
            send_price_tooltip_provider=self.tooltip_provider,
        )


def _controller_fixture(*, rows=(("system", "system"), ("user", "history"))):
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    state = {
        "settings": ConsoleSessionSettings(
            provider="anthropic", model="model-a", max_tokens=64
        ),
        "projection": ConsoleNextSendHistoryProjection(rows=tuple(rows)),
        "launch": None,
    }
    counter_calls = []

    def counter(messages, model, provider):
        counter_calls.append((messages, model, provider))
        return 123

    controller = ConsoleSendPriceController(
        settings_accessor=lambda: state["settings"],
        chat_store_accessor=lambda: store,
        provider_history_accessor=lambda _session_id: state["projection"],
        pending_launch_accessor=lambda: state["launch"],
        pricing_catalog_accessor=lambda: _Catalog(),
        token_counter=counter,
    )
    return controller, store, session, state, counter_calls


def _staged_launch(text="staged evidence"):
    bundle = EvidenceBundle(
        bundle_id="bundle",
        query="question",
        references=(
            EvidenceReference(
                evidence_id="S1",
                source_id="source",
                source_type="note",
                title="Source",
                snippet=text,
                authority_label="local",
                source_owner="local",
            ),
        ),
    )
    return ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Evidence",
        payload={"evidence_bundle": bundle.to_payload()},
    )


@pytest.mark.asyncio
async def test_composer_price_affordance_tracks_send_queue_attachment_and_width():
    calls = []

    def tooltip_provider(draft):
        calls.append(draft)
        return f"Next request: up to ~$0.01\nDraft: {draft or '(attachment only)'}"

    app = _PriceComposerApp(tooltip_provider)
    async with app.run_test(size=(100, 24)) as pilot:
        composer = app.query_one(ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        actions = composer.query_one("#console-composer-actions")

        assert send_button.label.plain == "Send"
        assert send_button.styles.width.value == 6
        assert send_button.tooltip == "Type a message to send."
        assert calls == []

        composer.load_draft("priced draft")
        assert send_button.label.plain == "Send | $"
        assert send_button.tooltip == (
            "Next request: up to ~$0.01\nDraft: priced draft"
        )
        assert send_button.styles.width.value == cell_len("Send | $") + 2
        assert actions.styles.width.value == BASE_ACTIONS_WIDTH + 4

        composer._sync_current_action_state()
        composer._sync_current_action_state()
        assert send_button.label.plain == "Send | $"
        assert composer._send_label == "Send"

        composer.sync_action_state(
            has_draft=True,
            run_active=True,
            can_save_chatbook=False,
            send_blocked=False,
            send_label="Queue",
        )
        assert send_button.label.plain == "Queue | $"
        assert send_button.styles.width.value == cell_len("Queue | $") + 2
        assert composer._send_label == "Queue"

        composer.clear_draft()
        composer.sync_action_state(
            has_draft=False,
            run_active=False,
            can_save_chatbook=False,
            send_blocked=False,
            send_label="Send",
        )
        composer.set_pending_attachment_label("photo.png · 5 B")
        composer._sync_current_action_state()
        await pilot.pause()
        assert send_button.label.plain == "Send | $"
        assert send_button.tooltip == (
            "Next request: up to ~$0.01\nDraft: (attachment only)"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup_reason", "wake_turn_active", "expected_tooltip"),
    [
        (
            "Choose a model in Console Settings before sending.",
            False,
            "Choose a model in Console Settings before sending.",
        ),
        (
            "",
            True,
            "A background sub-agent result is being delivered. Wait for it to finish.",
        ),
        ("", False, "Wait for the active Console run to finish before sending."),
    ],
)
async def test_composer_blocker_tooltip_and_unsuffixed_label_win_over_price(
    setup_reason,
    wake_turn_active,
    expected_tooltip,
):
    calls = []

    def tooltip_provider(draft):
        calls.append(draft)
        return "Next request: up to ~$0.01"

    app = _PriceComposerApp(tooltip_provider)
    async with app.run_test(size=(100, 24)):
        composer = app.query_one(ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)
        composer.load_draft("sendable")
        calls_before_block = len(calls)

        composer.sync_action_state(
            has_draft=True,
            run_active=True,
            can_save_chatbook=False,
            send_blocked=True,
            setup_blocked_reason=setup_reason,
            send_label="Queue",
            wake_turn_active=wake_turn_active,
        )

        assert send_button.disabled is True
        assert send_button.label.plain == "Queue"
        assert send_button.styles.width.value == 7
        assert send_button.tooltip == expected_tooltip
        assert calls_before_block == len(calls)


@pytest.mark.asyncio
async def test_composer_without_price_provider_keeps_existing_send_behavior():
    app = _PriceComposerApp()
    async with app.run_test(size=(100, 24)):
        composer = app.query_one(ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)

        composer.load_draft("ordinary draft")

        assert send_button.label.plain == "Send"
        assert send_button.styles.width.value == 6
        assert send_button.tooltip == "Send the active Console session draft."


@pytest.mark.asyncio
async def test_composer_price_provider_failure_never_blocks_send():
    def broken_provider(_draft):
        raise RuntimeError("pricing unavailable")

    app = _PriceComposerApp(broken_provider)
    async with app.run_test(size=(100, 24)):
        composer = app.query_one(ConsoleComposerBar)
        send_button = composer.query_one("#console-send-message", Button)

        composer.load_draft("still sendable")

        assert send_button.disabled is False
        assert send_button.label.plain == "Send | $"
        assert send_button.tooltip == "Next request: cost unavailable"


def test_console_send_price_controller_counts_canonical_draft_and_staged_context_once():
    controller, _store, _session, state, calls = _controller_fixture()
    state["launch"] = _staged_launch()

    result = controller.presentation_for_draft("  live draft  ")

    assert result is not None
    assert calls == [
        (
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "history"},
                {"role": "user", "content": "  live draft  "},
                {"role": "user", "content": "staged evidence"},
            ],
            "model-a",
            "anthropic",
        )
    ]
    assert controller.tooltip_for_draft("  live draft  ") == result.tooltip
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change", ["draft", "provider", "model", "system", "history", "staged"]
)
def test_console_send_price_controller_cache_reuses_and_recomputes_full_signature(
    change,
):
    controller, _store, _session, state, calls = _controller_fixture()
    controller.presentation_for_draft("draft")
    controller.presentation_for_draft("draft")
    assert len(calls) == 1

    draft = "draft"
    if change == "draft":
        draft = "changed draft"
    elif change == "provider":
        state["settings"] = replace(state["settings"], provider="openai")
    elif change == "model":
        state["settings"] = replace(state["settings"], model="model-b")
    elif change == "system":
        state["projection"] = ConsoleNextSendHistoryProjection(
            rows=(("system", "changed system"), ("user", "history"))
        )
    elif change == "history":
        state["projection"] = ConsoleNextSendHistoryProjection(
            rows=(("system", "system"), ("user", "changed history"))
        )
    else:
        state["launch"] = _staged_launch("changed staged")

    controller.presentation_for_draft(draft)
    assert len(calls) == 2


def test_console_send_price_controller_attachment_refresh_does_not_retokenize():
    controller, store, session, _state, calls = _controller_fixture()
    first = controller.presentation_for_draft("draft")
    store.add_pending_attachment(
        session.id,
        PendingAttachment(
            file_path="/tmp/a.png",
            display_name="a.png",
            file_type="image",
            insert_mode="attachment",
            data=b"image",
            mime_type="image/png",
            original_size=5,
            processed_size=5,
        ),
    )
    second = controller.presentation_for_draft("draft")

    assert len(calls) == 1
    assert first is not None and "Attachments:" not in first.tooltip
    assert second is not None and "Attachments: 1" in second.tooltip


def test_console_send_price_controller_reply_limit_refreshes_without_retokenizing():
    controller, _store, _session, state, calls = _controller_fixture()
    first = controller.presentation_for_draft("draft")
    state["settings"] = replace(state["settings"], max_tokens=128)

    second = controller.presentation_for_draft("draft")

    assert len(calls) == 1
    assert first is not None and "Reply: up to 64 tokens" in first.tooltip
    assert second is not None and "Reply: up to 128 tokens" in second.tooltip


def test_console_send_price_controller_session_change_recomputes_cache_slot():
    controller, store, _session, _state, calls = _controller_fixture()
    controller.presentation_for_draft("draft")
    second_session = store.create_session(ephemeral=True)

    controller.presentation_for_draft("draft")

    assert store.active_session_id == second_session.id
    assert len(calls) == 2


def test_console_send_price_controller_attachment_only_and_blank_empty_behavior():
    controller, store, session, _state, calls = _controller_fixture(rows=())
    assert controller.presentation_for_draft("   ") is None
    store.add_pending_attachment(
        session.id,
        PendingAttachment(
            file_path="/tmp/a.png",
            display_name="a.png",
            file_type="image",
            insert_mode="attachment",
            data=b"image",
            mime_type="image/png",
            original_size=5,
            processed_size=5,
        ),
    )

    result = controller.presentation_for_draft("   ")

    assert result is not None and "Attachments: 1" in result.tooltip
    assert calls[-1][0] == []


def test_console_send_price_controller_historical_media_stays_out_of_token_rows():
    controller, _store, _session, state, calls = _controller_fixture()
    state["projection"] = ConsoleNextSendHistoryProjection(
        rows=(("user", "image question"),), historical_media_count=2
    )

    result = controller.presentation_for_draft("draft")
    state["projection"] = replace(state["projection"], historical_media_count=3)
    refreshed = controller.presentation_for_draft("draft")

    assert result is not None and "Media context: 2 items" in result.tooltip
    assert refreshed is not None and "Media context: 3 items" in refreshed.tooltip
    assert len(calls) == 1
    assert all("media" not in str(row).lower() for row in calls[-1][0])


def test_console_send_price_controller_counter_failure_keeps_detailed_unavailable_copy():
    controller, store, _session, state, _calls = _controller_fixture()
    controller = ConsoleSendPriceController(
        settings_accessor=lambda: state["settings"],
        chat_store_accessor=lambda: store,
        provider_history_accessor=lambda _session_id: state["projection"],
        pending_launch_accessor=lambda: None,
        pricing_catalog_accessor=lambda: _Catalog(),
        token_counter=lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = controller.presentation_for_draft("draft")

    assert result is not None
    assert "Input: token estimate unavailable" in result.tooltip
    assert "Reply: up to 64 tokens" in result.tooltip
    assert "anthropic · model-a · rates as of" in result.tooltip


def test_console_send_price_controller_missing_store_and_broader_failures_degrade_safely():
    settings = ConsoleSessionSettings(provider="anthropic", model="model-a")
    missing = ConsoleSendPriceController(
        settings_accessor=lambda: settings,
        chat_store_accessor=lambda: None,
        provider_history_accessor=lambda _session_id: (
            ConsoleNextSendHistoryProjection()
        ),
        pending_launch_accessor=lambda: None,
        pricing_catalog_accessor=lambda: _Catalog(),
        token_counter=lambda *_args: 1,
    )
    broken = ConsoleSendPriceController(
        settings_accessor=lambda: settings,
        chat_store_accessor=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        provider_history_accessor=lambda _session_id: (
            ConsoleNextSendHistoryProjection()
        ),
        pending_launch_accessor=lambda: None,
        pricing_catalog_accessor=lambda: _Catalog(),
        token_counter=lambda *_args: 1,
    )
    catalog_broken = ConsoleSendPriceController(
        settings_accessor=lambda: settings,
        chat_store_accessor=lambda: None,
        provider_history_accessor=lambda _session_id: (
            ConsoleNextSendHistoryProjection()
        ),
        pending_launch_accessor=lambda: None,
        pricing_catalog_accessor=lambda: (_ for _ in ()).throw(
            RuntimeError("catalog boom")
        ),
        token_counter=lambda *_args: 1,
    )
    closed_store = ConsoleChatStore()
    closed_session = closed_store.create_session(ephemeral=True)
    closed_store.close_session(closed_session.id)
    closed = ConsoleSendPriceController(
        settings_accessor=lambda: settings,
        chat_store_accessor=lambda: closed_store,
        provider_history_accessor=lambda _session_id: (
            (_ for _ in ()).throw(AssertionError("closed session was projected"))
        ),
        pending_launch_accessor=lambda: None,
        pricing_catalog_accessor=lambda: _Catalog(),
        token_counter=lambda *_args: 1,
    )

    expected = ConsoleNextSendPrice("Next request: cost unavailable")
    assert missing.presentation_for_draft("draft") == expected
    assert closed.presentation_for_draft("draft") == expected
    assert broken.presentation_for_draft("draft") == expected
    assert catalog_broken.presentation_for_draft("draft") == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"input_tokens": 1284, "max_reply_tokens": 4096, "pricing": KNOWN_PRICING},
            "Next request: up to ~$0.0874\n"
            "Input: ~1,284 tokens · ~$0.0039\n"
            "Reply: up to 4,096 tokens · ~$0.0836\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {"input_tokens": 1284, "max_reply_tokens": 4096, "pricing": None},
            "Next request: cost unavailable\n"
            "Input: ~1,284 tokens\n"
            "Reply: up to 4,096 tokens\n"
            "anthropic · claude-sonnet-4-6 · pricing not configured",
        ),
        (
            {"input_tokens": 1284, "max_reply_tokens": 4096, "pricing": ZERO_PRICING},
            "Next request: up to ~$0.00\n"
            "Input: ~1,284 tokens · ~$0.00\n"
            "Reply: up to 4,096 tokens · ~$0.00\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {"input_tokens": None, "max_reply_tokens": 4096, "pricing": KNOWN_PRICING},
            "Next request: cost unavailable\n"
            "Input: token estimate unavailable\n"
            "Reply: up to 4,096 tokens · ~$0.0836\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {"input_tokens": 1284, "max_reply_tokens": None, "pricing": KNOWN_PRICING},
            "Next request: cost unavailable\n"
            "Input: ~1,284 tokens · ~$0.0039\n"
            "Reply: limit not configured\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {
                "input_tokens": 1284,
                "max_reply_tokens": 4096,
                "pricing": KNOWN_PRICING,
                "attachment_count": 1,
            },
            "Next request: cost unavailable\n"
            "Input text: ~1,284 tokens · ~$0.0039\n"
            "Reply text: up to 4,096 tokens · ~$0.0836\n"
            "Attachments: 1 · media cost not estimated\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {
                "input_tokens": 1284,
                "max_reply_tokens": 4096,
                "pricing": KNOWN_PRICING,
                "historical_media_count": 1,
            },
            "Next request: cost unavailable\n"
            "Input text: ~1,284 tokens · ~$0.0039\n"
            "Reply text: up to 4,096 tokens · ~$0.0836\n"
            "Media context: 1 item · media cost not estimated\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {
                "input_tokens": 1284,
                "max_reply_tokens": 4096,
                "pricing": KNOWN_PRICING,
                "historical_media_count": 2,
            },
            "Next request: cost unavailable\n"
            "Input text: ~1,284 tokens · ~$0.0039\n"
            "Reply text: up to 4,096 tokens · ~$0.0836\n"
            "Media context: 2 items · media cost not estimated\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {
                "input_tokens": 1284,
                "max_reply_tokens": 4096,
                "pricing": KNOWN_PRICING,
                "attachment_count": 2,
                "historical_media_count": 3,
            },
            "Next request: cost unavailable\n"
            "Input text: ~1,284 tokens · ~$0.0039\n"
            "Reply text: up to 4,096 tokens · ~$0.0836\n"
            "Attachments: 2 · media cost not estimated\n"
            "Media context: 3 items · media cost not estimated\n"
            "anthropic · claude-sonnet-4-6 · rates as of 2026-08-01",
        ),
        (
            {
                "input_tokens": 1,
                "max_reply_tokens": 1,
                "pricing": None,
                "provider": "",
                "model": "",
            },
            "Next request: cost unavailable\n"
            "Input: ~1 token\n"
            "Reply: up to 1 token\n"
            "pricing not configured",
        ),
    ],
)
def test_build_next_send_price_formats_requested_estimate(
    kwargs: dict[str, object], expected: str
) -> None:
    """Each pricing state yields an explicit, honest request preview."""
    result = build_next_send_price(
        **{"provider": "anthropic", "model": "claude-sonnet-4-6", **kwargs}
    )

    assert result.tooltip == expected


@pytest.mark.parametrize(
    ("input_tokens", "max_reply_tokens", "expected_input", "expected_reply"),
    [
        (1, 2, "Input: ~1 token", "Reply: up to 2 tokens"),
        (2, 1, "Input: ~2 tokens", "Reply: up to 1 token"),
    ],
)
def test_build_next_send_price_uses_singular_token_grammar(
    input_tokens: int,
    max_reply_tokens: int,
    expected_input: str,
    expected_reply: str,
) -> None:
    """A one-token estimate uses singular grammar in each line."""
    result = build_next_send_price(
        input_tokens=input_tokens,
        max_reply_tokens=max_reply_tokens,
        pricing=None,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )

    assert result.tooltip.splitlines()[1:3] == [expected_input, expected_reply]


@pytest.mark.parametrize(
    ("provider", "model", "expected_provenance"),
    [
        (
            "  anthropic  ",
            "  claude-sonnet-4-6  ",
            "anthropic · claude-sonnet-4-6 · pricing not configured",
        ),
        ("   ", "  claude-sonnet-4-6  ", "claude-sonnet-4-6 · pricing not configured"),
    ],
)
def test_build_next_send_price_normalizes_provenance_identifiers(
    provider: str, model: str, expected_provenance: str
) -> None:
    """Provenance omits blank identifiers and trims the identifiers it shows."""
    result = build_next_send_price(
        input_tokens=1,
        max_reply_tokens=1,
        pricing=None,
        provider=provider,
        model=model,
    )

    assert result.tooltip.splitlines()[-1] == expected_provenance
