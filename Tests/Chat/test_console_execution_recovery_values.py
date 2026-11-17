"""Finite recovery displays retain original copy, never executable checkpoints."""

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleDispatchRecoveryAction,
    ConsoleDispatchRecoveryActionId,
    ConsoleDispatchRecoveryKind,
    ConsoleDispatchRecoveryState,
)
from tldw_chatbook.Chat.console_chat_store import (
    ConsoleChatSession,
    ConsoleSettingsComponent,
    ConsoleSettingsPolicyFailureLabel,
)
from tldw_chatbook.Chat.console_execution_composition_values import (
    decode_session_composition,
)
from tldw_chatbook.Chat.console_execution_projection_values import (
    decode_session_projection,
    encode_session_projection,
)
from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError
from tldw_chatbook.Chat.console_execution_recovery_values import (
    ConsoleRecoveryDisplay,
    ConsoleSettingsFailureDisplay,
    recovery_display,
)
from tldw_chatbook.Chat.console_trace_call_recovery import (
    TraceCallProviderContact,
    TraceProviderRejection,
)
from tldw_chatbook.Chat.console_turn_preparation import ContextCompactionHold
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    ProviderContinuationRecoveryState,
    ProviderContinuationTranscriptRegion,
    TraceCallRecoveryState,
)


def _recovery(session_id):
    return ConsoleRecoveryDisplay(
        ConsoleDispatchRecoveryState(
            ConsoleDispatchRecoveryKind.ACCEPTED,
            "assistant",
            "conversation",
            "Not sent.",
            (
                ConsoleDispatchRecoveryAction(
                    ConsoleDispatchRecoveryActionId.DISCARD,
                    "Discard",
                    True,
                ),
            ),
            checkpoint=object(),
            queue_entry_id="private-queue",
            preparation_id="private-owner",
        ),
        ProviderContinuationRecoveryState("assistant", 4, "local", "visible"),
        TraceCallRecoveryState(
            "preparation",
            context_hold=ContextCompactionHold(session_id, 100, 90, 80, True),
            provider_contact=TraceCallProviderContact(
                "yes",
                TraceProviderRejection("provider", 429, "rate limited"),
                True,
            ),
        ),
        (
            ConsoleSettingsFailureDisplay(
                ConsoleSettingsComponent.GENERATION_SETTINGS, 2, None
            ),
            ConsoleSettingsFailureDisplay(
                ConsoleSettingsComponent.CONTEXT_POLICY,
                3,
                ConsoleSettingsPolicyFailureLabel.COMPACTION,
            ),
            ConsoleSettingsFailureDisplay(
                ConsoleSettingsComponent.PROMPT_MIXINS, 4, None
            ),
        ),
    )


def test_original_recovery_display_detaches_and_excludes_execution_handles():
    original = _recovery("session")
    encoded = recovery_display(original, False)
    result = recovery_display(encoded, True)
    assert result.dispatch.checkpoint is None
    assert result.dispatch.queue_entry_id is None
    assert result.dispatch.preparation_id is None
    assert result.dispatch.actions == original.dispatch.actions
    assert result.dispatch.visible_copy == original.dispatch.visible_copy
    assert result.continuation == original.continuation
    assert result.trace == original.trace
    assert result.settings_failures == original.settings_failures
    assert result.trace is not original.trace
    assert "private-" not in str(encoded)
    assert not any(
        name in str(encoded) for name in ("generation_snapshot", "mixin_preferences")
    )
    for malformed in (
        dict(encoded, checkpoint=object()),
        dict(encoded, dispatch=dict(encoded["dispatch"], checkpoint={})),
        dict(encoded, settings_failures=[encoded["settings_failures"][0]] * 2),
        dict(
            encoded,
            settings_failures=[
                dict(encoded["settings_failures"][0], policy_failure_label="compaction")
            ],
        ),
    ):
        with pytest.raises(ConsoleProtocolError):
            recovery_display(malformed, True)


def test_composition_recovery_is_bound_to_current_session_and_clears():
    from Tests.Chat.test_console_execution_composition_values import _encode

    session = ConsoleChatSession()
    row = decode_session_projection(encode_session_projection(session))
    absent = _encode(session)
    assert decode_session_composition(absent, row).recovery == ConsoleRecoveryDisplay(
        None, None, None, ()
    )
    payload = dict(absent, recovery=recovery_display(_recovery(session.id), False))
    recovered = decode_session_composition(payload, row)
    assert len(recovered.recovery.settings_failures) == 3
    assert recovered.recovery.trace.context_hold.session_id == row.id
    assert recovered.runtime_backend == session.runtime_backend
    with pytest.raises(ConsoleProtocolError, match="session_changed"):
        decode_session_composition(payload, replace(row, id="other"))
    payload["recovery"] = recovery_display(_recovery("other"), False)
    with pytest.raises(ConsoleProtocolError, match="recovery_session_changed"):
        decode_session_composition(payload, row)
    assert decode_session_composition(absent, row).recovery.dispatch is None


def test_original_recovery_widget_consumes_published_state_and_clears_without_message():
    current = [_recovery("session").continuation]

    def forbidden_message():
        raise AssertionError(
            "Display must not ask the parent for a recovery checkpoint"
        )

    region = ProviderContinuationTranscriptRegion(
        session_surface_builder=lambda: None,
        recovery_message_builder=forbidden_message,
        recovery_state_builder=lambda: current[0],
        on_recovery_action=lambda *_args: False,
    )
    assert region._recovery_state() is current[0]
    current[0] = None
    assert region._recovery_state() is None
