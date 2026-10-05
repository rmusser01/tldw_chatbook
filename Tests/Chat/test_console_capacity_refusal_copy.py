"""TASK-34100.5 AC#3 (entry-exit-handoff-03): a pre-dispatch capacity refusal.

The live first send after setup read: 'This request cannot fit the selected
model. Response reservation and safety margin leave no model input capacity.
Summarizing older turns cannot make enough room. Repair the model limit,
reduce mandatory context or the response maximum, or allow older turns to be
omitted.' -- internal terms, no model named, no way to fix it, under a
'Response accepted; waiting for dispatch' panel whose Retry could only be
refused again.

TASK-34350 (merged to dev first, owner ruling 2026-10-03) now owns the copy
and refuses a composer send before it is committed: the alert names the
model, what fills the window and the setting that changes it, and says when
the window is an estimate and where to set the real one. These tests pin that
outcome for AC#3, and pin AC#3's own half for a turn the context preflight
refuses AFTER it was accepted (a send that skips the pre-commit check): the
card reads 'Not sent', and Retry is refused for every caller.

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
#: The context policy's internal reasons, which must never reach the user.
_POLICY_JARGON = (
    "Response reservation and safety margin leave no model input capacity",
    "Mandatory request material",
    "Repair the model limit",
    "cannot fit the selected model",
)


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

    result = await controller.submit_draft("hello there", session_id="session-1")

    assert gateway.stream_calls == 0, "the refusal must happen before dispatch"
    assert result.accepted is False and result.should_clear_draft is False
    rows = [row for row in _system_rows(store) if _UNKNOWN_MODEL in row]
    assert len(rows) == 1, _system_rows(store)
    copy = rows[0]
    assert copy.startswith("Your message was not sent:")
    # The cause and its setting, and the window flagged as a guess with
    # where to set the real one -- only because it is one.
    assert "Max tokens" in copy
    assert "(an estimate)" in copy
    assert "F4 Settings > Providers & Models" in copy
    for jargon in _POLICY_JARGON:
        assert jargon not in copy
    # Refused before commit: no 'Response accepted' recovery panel, and no
    # Retry that could append a second copy of the row.
    assert store.dispatch_recovery_for_session("session-1") is None
    retry = await controller.retry_dispatch_recovery("session-1")
    assert retry.accepted is False
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
    assert "estimate" not in rows[0]
    assert "isn't known" not in rows[0]
    assert "1,000-token context window" in rows[0]
    for jargon in _POLICY_JARGON:
        assert jargon not in rows[0]


@pytest.mark.asyncio
async def test_a_refused_send_reads_not_sent_and_names_the_working_path(
    tmp_path, monkeypatch
) -> None:
    """The live first send showed 'Response accepted; waiting for dispatch.'
    with an enabled Retry under a refusal that no Retry could change. Retry
    replays the accepted turn with its frozen model and limit (live: Retry
    after Alt+M failed again under the old model), so the card says the
    message was not sent, keeps Retry disabled even after a setting changes,
    and names the path that works: Discard, then Resend."""
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

    # A send that skips TASK-34350's pre-commit check (here one that keeps the
    # composer, as a buddy-conversation send does) is refused by the context
    # preflight after it was accepted, so it has a recovery owner.
    await controller.submit_draft(
        "hello there", session_id="session-1", preserve_composer=True
    )

    def shown_actions():
        # The card's own projection: what the user sees and can press.
        from tldw_chatbook.UI.Console_Modules.dispatch_recovery import (
            derive_dispatch_recovery_presentation,
        )

        owner = store.dispatch_recovery_for_presentation("session-1")
        assert owner is not None, "the turn still needs Discard"
        shown = derive_dispatch_recovery_presentation(owner)
        return shown, {
            ConsoleDispatchRecoveryActionId(action.action_id): action
            for action in shown.actions
        }

    shown, actions = shown_actions()
    assert "accepted" not in shown.visible_copy.lower(), shown.visible_copy
    assert "waiting for dispatch" not in shown.visible_copy
    assert "Not sent" in shown.visible_copy
    assert "Discard and Resend" in shown.visible_copy
    assert shown.warning == ""
    retry = actions[ConsoleDispatchRecoveryActionId.RETRY_RESPONSE]
    assert retry.enabled is False
    assert "Discard, then Resend" in retry.disabled_reason
    assert actions[ConsoleDispatchRecoveryActionId.DISCARD].enabled is True

    # A changed model cannot reach this frozen turn: Retry stays disabled.
    store.replace_session_settings(
        "session-1",
        ConsoleSessionSettings(provider="openai", model="gpt-4.1-mini"),
    )
    _shown, actions = shown_actions()
    assert actions[ConsoleDispatchRecoveryActionId.RETRY_RESPONSE].enabled is False
    assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_the_controller_itself_refuses_retry_for_a_refused_send(
    tmp_path, monkeypatch
) -> None:
    """Review round 1 (F13): Retry was disabled only on the presented card;
    ``controller.retry_dispatch_recovery`` still claimed RETRY_RESPONSE from
    the raw owner and replayed the frozen turn. Any caller now gets the
    card's reason and nothing is claimed or sent."""
    from tldw_chatbook.Chat.console_predispatch_block import RETRY_DISABLED_REASON
    from tldw_chatbook.Utils import token_counter

    monkeypatch.setitem(token_counter.PROVIDER_CONTEXT_WINDOWS, "openai", 4096)
    _db, store, controller, gateway = _live_controller(
        tmp_path,
        gateway=_UnknownWindowGateway(),
        overrides=ConsoleContextPolicyOverrides(),
    )
    await controller.submit_draft(
        "hello there", session_id="session-1", preserve_composer=True
    )
    rows_before = list(_system_rows(store))

    result = await controller.retry_dispatch_recovery("session-1")

    assert result.accepted is False
    assert result.visible_copy == RETRY_DISABLED_REASON
    owner = store.dispatch_recovery_for_session("session-1")
    assert owner is not None and not owner.in_flight, "nothing was claimed"
    assert list(_system_rows(store)) == rows_before
    assert gateway.stream_calls == 0


def test_the_refusal_record_is_bounded() -> None:
    """Review round 1 (F13): the record of refused turns cannot grow with a
    long session; the oldest is forgotten first."""
    from tldw_chatbook.Chat import console_predispatch_block as blocks

    first = "assistant-bounded-0"
    blocks.note_predispatch_block(first)
    for index in range(1, blocks.MAX_NOTED_BLOCKS + 1):
        blocks.note_predispatch_block(f"assistant-bounded-{index}")

    assert not blocks.is_predispatch_block(first)
    assert blocks.is_predispatch_block(f"assistant-bounded-{blocks.MAX_NOTED_BLOCKS}")
    assert len(blocks._NOTED) == blocks.MAX_NOTED_BLOCKS


def test_the_store_carries_no_predispatch_side_table() -> None:
    """Review round 1 (F8): the store is over its size budget; the record
    lives beside the presentation, not in the store."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    assert not hasattr(store, "_predispatch_blocks")
    assert not hasattr(ConsoleChatStore, "note_predispatch_block")


def test_a_repeated_preflight_refusal_row_is_recognised() -> None:
    """A retried turn the preflight refuses again for the same reason must
    not append a second copy of the row the session already ends with."""
    from tldw_chatbook.Chat.console_predispatch_block import is_repeat_of_last_row
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    store = ConsoleChatStore()
    store.create_session(session_id="s", title="Chat 1")
    store.append_message("s", role=ConsoleMessageRole.SYSTEM, content="Refused.")

    assert is_repeat_of_last_row(store, "s", "Refused.")
    assert not is_repeat_of_last_row(store, "s", "Refused for another reason.")
    store.append_message("s", role=ConsoleMessageRole.USER, content="again")
    assert not is_repeat_of_last_row(store, "s", "Refused.")
    assert not is_repeat_of_last_row(store, "no-such-session", "Refused.")


def test_only_an_overflow_alert_marks_a_turn_not_sent() -> None:
    """Other preflight refusals (a compaction threshold, a changed character)
    keep the ordinary recovery card; only the copy the overflow alert
    produced marks the turn refused before dispatch."""
    from tldw_chatbook.Chat import console_predispatch_block as blocks

    alert = "Your message was not sent: it does not fit (overflow-unit-test)."
    assert blocks.remember_overflow_alert(alert) == alert
    assert blocks.remember_overflow_alert(None) is None

    blocks.note_if_overflow_alert("assistant-other-refusal", "Character changed.")
    blocks.note_if_overflow_alert("assistant-overflow-refusal", alert)

    assert not blocks.is_predispatch_block("assistant-other-refusal")
    assert blocks.is_predispatch_block("assistant-overflow-refusal")
