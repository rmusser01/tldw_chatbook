"""TASK-34100.5 AC#3 (entry-exit-handoff-03): a pre-dispatch capacity refusal.

The live first send after setup read: 'This request cannot fit the selected
model. Response reservation and safety margin leave no model input capacity.
Summarizing older turns cannot make enough room. Repair the model limit,
reduce mandatory context or the response maximum, or allow older turns to be
omitted.' -- internal terms, no model named, no way to fix it, and every
Retry appended a second copy of the same row. The refusal now names the model
and the real limiting reason in plain words, says the context size isn't known
only when it isn't, names Switch model and Set context size, and a Retry that
is refused again appends nothing new.

Real controller, real store and real ChaChaNotes persistence; only the
network is faked (the harness of ``test_console_compaction_live_session``).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from Tests.Chat.test_console_compaction_live_session import (
    _LiveProviderGateway,
    _close_live_databases,  # noqa: F401 - autouse fixture
    _live_controller,
    _system_rows,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides

pytestmark = pytest.mark.bootstrap_profile

_UNKNOWN_MODEL = "brand-new-model-x"


class _UnknownWindowGateway(_LiveProviderGateway):
    """The selected model has no catalog window: the provider fallback (4,096
    tokens, unverified) applies, and a 4,096-token reply reservation leaves
    no input capacity -- the shipped OpenAI-default failure, reproduced."""

    async def resolve_for_send(self, selection):
        resolution = await super().resolve_for_send(selection)
        return replace(resolution, model=_UNKNOWN_MODEL, max_tokens=4096)

    def prepare_chat_request(self, resolution, messages, **kwargs):
        return self._real.prepare_chat_request(resolution, messages, **kwargs)


@pytest.mark.asyncio
async def test_unknown_window_refusal_names_the_model_and_the_fix(
    tmp_path, monkeypatch
) -> None:
    from tldw_chatbook.Utils import token_counter

    monkeypatch.setitem(token_counter.PROVIDER_CONTEXT_WINDOWS, "openai", 4096)
    _db, store, controller, gateway = _live_controller(
        tmp_path,
        gateway=_UnknownWindowGateway(),
        overrides=ConsoleContextPolicyOverrides(),
    )

    await controller.submit_draft("hello there", session_id="session-1")

    assert gateway.stream_calls == 0, "the refusal must happen before dispatch"
    rows = [row for row in _system_rows(store) if _UNKNOWN_MODEL in row]
    assert len(rows) == 1, _system_rows(store)
    copy = rows[0]
    assert "context size isn't known" in copy
    assert "Switch model" in copy and "Alt+M" in copy
    assert "Set context size" in copy
    for jargon in (
        "Response reservation",
        "safety margin",
        "mandatory context",
        "Repair the model limit",
    ):
        assert jargon not in copy

    # A Retry that is refused again for the same reason adds no second row.
    await controller.retry_dispatch_recovery("session-1")
    assert [row for row in _system_rows(store) if _UNKNOWN_MODEL in row] == [copy]


@pytest.mark.asyncio
async def test_a_known_window_refusal_never_claims_the_size_is_unknown(
    tmp_path,
) -> None:
    """A verified window that the request outgrows is a different reason."""

    class _TinyKnownWindow(_LiveProviderGateway):
        async def resolve_for_send(self, selection):
            resolution = await super().resolve_for_send(selection)
            return replace(resolution, max_tokens=900)

    _db, store, controller, gateway = _live_controller(
        tmp_path,
        gateway=_TinyKnownWindow(context_window=1000),
        overrides=ConsoleContextPolicyOverrides(),
    )

    await controller.submit_draft("hello " * 400, session_id="session-1")

    assert gateway.stream_calls == 0
    rows = [row for row in _system_rows(store) if "gpt-test-live" in row]
    assert len(rows) == 1, _system_rows(store)
    assert "isn't known" not in rows[0]
    assert "Switch model" in rows[0]


@pytest.mark.asyncio
async def test_a_refused_send_reads_not_sent_and_retry_waits_for_a_setting(
    tmp_path, monkeypatch
) -> None:
    """The live first send showed 'Response accepted; waiting for dispatch.'
    with an enabled Retry under a refusal that no Retry could change. The
    recovery surface now says the message was not sent, and Retry stays
    disabled -- with the hint -- until a setting that could fix it changes."""
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleDispatchRecoveryActionId,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Utils import token_counter

    monkeypatch.setitem(token_counter.PROVIDER_CONTEXT_WINDOWS, "openai", 4096)
    _db, store, controller, gateway = _live_controller(
        tmp_path,
        gateway=_UnknownWindowGateway(),
        overrides=ConsoleContextPolicyOverrides(),
    )

    await controller.submit_draft("hello there", session_id="session-1")

    shown = store.dispatch_recovery_for_presentation("session-1")
    assert shown is not None, "the turn still needs Retry or Discard"
    assert "accepted" not in shown.visible_copy.lower(), shown.visible_copy
    assert "waiting for dispatch" not in shown.visible_copy
    assert "Not sent" in shown.visible_copy
    assert shown.warning == ""
    actions = {action.action_id: action for action in shown.actions}
    retry = actions[ConsoleDispatchRecoveryActionId.RETRY_RESPONSE]
    assert retry.enabled is False
    assert retry.disabled_reason == "Change a setting above, then retry"
    assert actions[ConsoleDispatchRecoveryActionId.DISCARD].enabled is True

    # Switching the model is a change that could fix it: Retry comes back.
    store.replace_session_settings(
        "session-1",
        ConsoleSessionSettings(provider="openai", model="gpt-4.1-mini"),
    )
    shown = store.dispatch_recovery_for_presentation("session-1")
    assert shown is not None
    retry = next(
        action
        for action in shown.actions
        if action.action_id is ConsoleDispatchRecoveryActionId.RETRY_RESPONSE
    )
    assert retry.enabled is True
    assert gateway.stream_calls == 0
