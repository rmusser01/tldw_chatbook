"""Recorded provider usage remains visible while optional estimates are cold."""

import time
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat import console_cost_tracker as tracker
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.LLM_Calls.pricing_catalog import PricingCatalog
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.UI.Console_Modules.context_spend import ConsoleContextSpendController

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("unmeasured_row", [False, True])
def test_current_spend_survives_unavailable_tokenizer(monkeypatch, unmeasured_row):
    def no_estimation(*_args, **_kwargs):
        pytest.fail("unprepared display entered token estimation or request projection")

    monkeypatch.setattr(spend, "console_tokenizer_display_ready", lambda *_: False)
    monkeypatch.setattr(tracker, "_estimate_tokens_locally", no_estimation)
    monkeypatch.setattr(spend, "console_rate_limit_line", lambda *_: None)
    usage = ProviderUsage(
        uncached_input=1_000_000,
        output=1_000_000,
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    rows = [
        SimpleNamespace(
            id="question",
            role=ConsoleMessageRole.USER,
            content="question",
            usage=None,
            status="complete",
            attachments=(),
        ),
        SimpleNamespace(
            id="answer",
            role=ConsoleMessageRole.ASSISTANT,
            content="answer",
            usage=None if unmeasured_row else usage,
            status="complete",
            attachments=(),
        ),
    ]
    history = SimpleNamespace(current_ids={row.id for row in rows}, request_ids=set())
    controller = SimpleNamespace(
        run_state_for=lambda _: SimpleNamespace(status=ConsoleRunStatus.IDLE),
        unattributed_fleet_tokens=lambda _: 5,
        cache_ttl_snapshot=lambda _: (time.monotonic() + 60, True),
        payload_fingerprint_baseline=no_estimation,
    )
    store = SimpleNamespace(
        session_settings_revision=lambda _: 1,
        payload_revision=lambda _: 1,
        pending_attachments=lambda _: (),
    )
    catalog = PricingCatalog(config={})
    cache = tracker.TokenEstimateCache()
    receiver = SimpleNamespace(
        _screen=object(),
        _active_console_provider_model_display=lambda: (
            "anthropic",
            "claude-sonnet-4-6",
            None,
        ),
        _ensure_console_chat_controller=lambda: controller,
        _current_chat_controller=lambda: controller,
        _console_display_history=lambda *_: (("history",), rows, history),
        _display_pricing_catalog=lambda: catalog,
        _response_rule_usage=lambda _: (),
        _fleet_token_total=lambda: 7,
        _console_cost_estimate_cache_or_new=lambda: cache,
        _build_cost_snapshot=tracker.build_cost_snapshot,
        _console_cost_break_reasons={"session": "changed"},
        _console_cost_fp_revisions={},
        _console_composer_or_none=lambda: None,
        # This belongs to an older prepared model; it must not supply Next Send.
        _last_console_context_control_state=SimpleNamespace(request_tokens=123),
        _active_console_context_control_state=no_estimation,
        _estimate_tokens_locally=no_estimation,
    )
    build = ConsoleContextSpendController._build_console_cost_state_under_config
    state = build(receiver, SimpleNamespace(id="session"), store)
    if unmeasured_row:
        assert state is None  # Never report a partial total as the whole Current cost.
        return
    assert state is not None
    assert state.label.startswith("$18.00")
    assert "On next send" not in state.label
    assert "~+$" not in state.label
    assert receiver._console_settled_spend_cache[2].total_tokens == 2_000_000
    assert "12" in state.tooltip  # Live and settled-but-unattributed fleet tokens.
    assert build(receiver, SimpleNamespace(id="session"), store) == state
