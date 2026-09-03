from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from typing import Any

import pytest

from tldw_chatbook.ACP_Interop import runtime_session
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Prompt_Management.prompt_variables import (
    PromptVariableApplication,
)
from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
    AudioCppModelLibraryRequest,
    AudioCppModelLibraryResult,
)
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleFirstChatIntent,
    ConsoleProviderIntent,
    HandoffChannel,
    HandoffClaim,
    HandoffValueError,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Navigation.conversation_settings_navigation import (
    ConversationSettingsReturnIntent,
)
from tldw_chatbook.UI.Screens.study_scope_models import (
    STUDY_INITIAL_SECTIONS,
    StudyScopeContext,
    StudySourceItem,
)


def _chat_payload(title: str = "handoff") -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source="tests",
        item_type="document",
        title=title,
        body="context",
        metadata={"nested": {"items": ["original"]}},
    )


def _console_launch(title: str = "launch") -> ConsoleLiveWorkLaunch:
    return ConsoleLiveWorkLaunch.from_values(
        source="tests",
        title=title,
        payload={"nested": {"items": ["original"]}},
    )


def _study_scope() -> StudyScopeContext:
    return StudyScopeContext(
        material_title="Private study material",
        source_items=(
            StudySourceItem(
                source_type="media",
                source_id="media-1",
                locator={"nested": {"items": ["original"]}},
            ),
        ),
    )


def _prompt_application(
    user_text: str = "private rendered prompt",
    *,
    created_monotonic: float = 10.0,
) -> PromptVariableApplication:
    return PromptVariableApplication(
        system_text=None,
        user_text=user_text,
        apply_system=False,
        apply_user=True,
        destination="append_active",
        target_session_id="session-1",
        composer_fingerprint=None,
        system_fingerprint=None,
        created_monotonic=created_monotonic,
    )


def _claim_title(store: PendingHandoffStore, channel: HandoffChannel) -> str:
    claim = store.claim(channel)
    assert claim is not None
    title = claim.value.title
    assert store.acknowledge(claim) is True
    return title


def test_stage_replaces_unclaimed_value_with_channel_local_revision() -> None:
    store = PendingHandoffStore()

    assert store.stage(HandoffChannel.CHAT, _chat_payload("first")) == 1
    assert store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, _prompt_application()) == 1
    assert store.stage(HandoffChannel.CHAT, _chat_payload("second")) == 2

    assert _claim_title(store, HandoffChannel.CHAT) == "second"


def test_conversation_settings_return_handoff_replaces_and_detaches() -> None:
    store = PendingHandoffStore()
    first = ConversationSettingsReturnIntent("session-1", 4, "model", "console-settings-model-picker")
    second = ConversationSettingsReturnIntent("session-2", 5, "context", None)
    assert store.stage(HandoffChannel.CONVERSATION_SETTINGS_RETURN, first) == 1
    assert store.stage(HandoffChannel.CONVERSATION_SETTINGS_RETURN, second) == 2
    claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert claim is not None
    assert claim.revision == 2
    assert claim.value == second
    assert claim.value is not second
    assert store.acknowledge(claim) is True


def test_conversation_settings_return_handoff_requires_exact_ack_and_supports_release() -> None:
    store = PendingHandoffStore()
    intent = ConversationSettingsReturnIntent("session-1", 4, "model", None)
    store.stage(HandoffChannel.CONVERSATION_SETTINGS_RETURN, intent)
    claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert claim is not None
    assert store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN) is None
    assert store.acknowledge(replace(claim)) is False
    assert store.release(replace(claim)) is False
    assert store.release(claim) is True
    retry = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert retry is not None
    assert retry.revision == claim.revision
    assert store.acknowledge(retry) is True


def test_settle_transferred_claim_atomically_settles_exact_in_flight_return() -> None:
    """A modal-owned return is terminal without a screen-owned cleanup step."""

    store = PendingHandoffStore()
    intent = ConversationSettingsReturnIntent("session-1", 4, "model", None)
    revision = store.stage(HandoffChannel.CONVERSATION_SETTINGS_RETURN, intent)
    claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)

    assert claim is not None
    assert store.settle_transferred_claim(claim) is True
    assert (
        store.exact_revision_status(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN, revision
        )
        == "settled"
    )
    assert store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN) is None


def test_settle_transferred_claim_atomically_removes_exact_requeued_return() -> None:
    """A partial prior release cannot leave a snapshot-less pending replay."""

    store = PendingHandoffStore()
    intent = ConversationSettingsReturnIntent("session-1", 4, "model", None)
    revision = store.stage(HandoffChannel.CONVERSATION_SETTINGS_RETURN, intent)
    claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)

    assert claim is not None
    assert store.release(claim) is True
    assert store.settle_transferred_claim(claim) is True
    assert (
        store.exact_revision_status(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN, revision
        )
        == "settled"
    )


def test_settle_transferred_claim_preserves_pending_replacement() -> None:
    """Settling in-flight A leaves pending B's exact revision and value intact."""

    store = PendingHandoffStore()
    first = ConversationSettingsReturnIntent("session-1", 4, "model", None)
    replacement = ConversationSettingsReturnIntent(
        "session-2", 5, "context", "console-settings-provider"
    )
    first_revision = store.stage(HandoffChannel.CONVERSATION_SETTINGS_RETURN, first)
    first_claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    replacement_revision = store.stage(
        HandoffChannel.CONVERSATION_SETTINGS_RETURN, replacement
    )

    assert first_claim is not None
    assert store.settle_transferred_claim(first_claim) is True
    assert (
        store.exact_revision_status(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN, first_revision
        )
        == "superseded"
    )
    replacement_claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert replacement_claim is not None
    assert replacement_claim.revision == replacement_revision
    assert replacement_claim.value == replacement
    assert store.settle_transferred_claim(first_claim) is True
    assert store.is_current_claim(replacement_claim) is True


def test_settle_transferred_claim_recognizes_terminal_return_without_mutation() -> None:
    """Repeated or superseded settlement is terminal and cannot consume B."""

    store = PendingHandoffStore()
    first = ConversationSettingsReturnIntent("session-1", 4, "model", None)
    replacement = ConversationSettingsReturnIntent("session-2", 5, "context", None)
    store.stage(HandoffChannel.CONVERSATION_SETTINGS_RETURN, first)
    first_claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert first_claim is not None
    assert store.acknowledge(first_claim) is True
    assert store.settle_transferred_claim(first_claim) is True

    replacement_revision = store.stage(
        HandoffChannel.CONVERSATION_SETTINGS_RETURN, replacement
    )
    assert store.settle_transferred_claim(first_claim) is True
    replacement_claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert replacement_claim is not None
    assert replacement_claim.revision == replacement_revision
    assert replacement_claim.value == replacement


def test_settle_transferred_claim_rejects_other_current_claim_identity() -> None:
    """A stale claim object cannot settle a newly claimed retry of its revision."""

    store = PendingHandoffStore()
    intent = ConversationSettingsReturnIntent("session-1", 4, "model", None)
    store.stage(HandoffChannel.CONVERSATION_SETTINGS_RETURN, intent)
    first_claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert first_claim is not None
    assert store.release(first_claim) is True
    retry_claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)

    assert retry_claim is not None
    assert retry_claim is not first_claim
    assert store.settle_transferred_claim(first_claim) is False
    assert store.is_current_claim(retry_claim) is True
    assert store.settle_transferred_claim(retry_claim) is True


def test_settle_transferred_claim_validates_type_channel_and_owner_thread() -> None:
    """The atomic transfer boundary keeps the store's affine typed contract."""

    store = PendingHandoffStore()
    with pytest.raises(TypeError, match="HandoffClaim"):
        store.settle_transferred_claim(object())  # type: ignore[arg-type]

    store.stage(HandoffChannel.CHAT, _chat_payload())
    chat_claim = store.claim(HandoffChannel.CHAT)
    assert chat_claim is not None
    with pytest.raises(ValueError, match="Conversation settings"):
        store.settle_transferred_claim(chat_claim)

    store.stage(
        HandoffChannel.CONVERSATION_SETTINGS_RETURN,
        ConversationSettingsReturnIntent("session-1", 4, "model", None),
    )
    return_claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert return_claim is not None
    with pytest.raises(ValueError, match="positive exact integer"):
        store.settle_transferred_claim(replace(return_claim, revision=True))
    with ThreadPoolExecutor(max_workers=1) as executor:
        failure = executor.submit(
            store.settle_transferred_claim, return_claim
        ).exception()
    assert isinstance(failure, RuntimeError)
    assert "owner thread" in str(failure)


def test_exact_revision_status_distinguishes_pending_in_flight_and_terminal() -> None:
    """Consumers can distinguish ownership without reading a handoff value."""

    store = PendingHandoffStore()
    intent = ConversationSettingsReturnIntent("session-1", 4, "model", None)
    revision = store.stage(HandoffChannel.CONVERSATION_SETTINGS_RETURN, intent)

    assert (
        store.exact_revision_status(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN, revision
        )
        == "pending"
    )
    claim = store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN)
    assert claim is not None
    assert (
        store.exact_revision_status(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN, revision
        )
        == "in_flight"
    )
    assert store.acknowledge(claim)
    assert (
        store.exact_revision_status(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN, revision
        )
        == "settled"
    )

    newer_revision = store.stage(
        HandoffChannel.CONVERSATION_SETTINGS_RETURN,
        ConversationSettingsReturnIntent("session-2", 0, "context", None),
    )
    assert newer_revision > revision
    assert (
        store.exact_revision_status(
            HandoffChannel.CONVERSATION_SETTINGS_RETURN, revision
        )
        == "superseded"
    )


def test_conversation_settings_return_handoff_explicit_clear() -> None:
    store = PendingHandoffStore()
    store.stage(
        HandoffChannel.CONVERSATION_SETTINGS_RETURN,
        ConversationSettingsReturnIntent("session-1", 4, "model", None),
    )
    assert store.clear_pending(HandoffChannel.CONVERSATION_SETTINGS_RETURN) == 2
    assert store.claim(HandoffChannel.CONVERSATION_SETTINGS_RETURN) is None


def test_claim_is_exclusive_until_exact_claim_settles() -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, _chat_payload())

    claim = store.claim(HandoffChannel.CHAT)

    assert claim is not None
    assert store.claim(HandoffChannel.CHAT) is None
    assert store.acknowledge(replace(claim)) is False
    assert store.acknowledge(claim) is True
    assert store.claim(HandoffChannel.CHAT) is None


def test_has_pending_reports_only_unclaimed_channel_state() -> None:
    store = PendingHandoffStore()

    assert store.has_pending(HandoffChannel.CHAT) is False
    store.stage(HandoffChannel.CHAT, _chat_payload())
    assert store.has_pending(HandoffChannel.CHAT) is True

    claim = store.claim(HandoffChannel.CHAT)

    assert claim is not None
    assert store.has_pending(HandoffChannel.CHAT) is False
    assert store.release(claim) is True
    assert store.has_pending(HandoffChannel.CHAT) is True


def test_release_restores_same_revision_for_a_fresh_claim() -> None:
    store = PendingHandoffStore()
    revision = store.stage(HandoffChannel.CHAT, _chat_payload())
    first_claim = store.claim(HandoffChannel.CHAT)
    assert first_claim is not None

    assert store.release(first_claim) is True
    retry_claim = store.claim(HandoffChannel.CHAT)

    assert retry_claim is not None
    assert retry_claim is not first_claim
    assert retry_claim.revision == revision
    assert store.acknowledge(first_claim) is False
    assert store.acknowledge(retry_claim) is True


def test_release_does_not_overwrite_newer_replacement() -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, _chat_payload("first"))
    claim = store.claim(HandoffChannel.CHAT)
    assert claim is not None

    store.stage(HandoffChannel.CHAT, _chat_payload("second"))

    assert store.release(claim) is True
    assert _claim_title(store, HandoffChannel.CHAT) == "second"
    assert store.acknowledge(claim) is False


def test_only_latest_replacement_survives_while_claim_is_in_flight() -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, _chat_payload("first"))
    claim = store.claim(HandoffChannel.CHAT)
    assert claim is not None

    store.stage(HandoffChannel.CHAT, _chat_payload("second"))
    store.stage(HandoffChannel.CHAT, _chat_payload("third"))

    assert store.acknowledge(claim) is True
    assert _claim_title(store, HandoffChannel.CHAT) == "third"


def test_clear_pending_supersedes_an_in_flight_claim() -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, _chat_payload())
    claim = store.claim(HandoffChannel.CHAT)
    assert claim is not None

    clear_revision = store.clear_pending(HandoffChannel.CHAT)

    assert clear_revision == claim.revision + 1
    assert store.release(claim) is True
    assert store.claim(HandoffChannel.CHAT) is None
    assert store.acknowledge(claim) is False


def test_acknowledging_old_claim_preserves_newer_replacement() -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, _chat_payload("first"))
    claim = store.claim(HandoffChannel.CHAT)
    assert claim is not None
    store.stage(HandoffChannel.CHAT, _chat_payload("second"))

    assert store.acknowledge(claim) is True
    assert _claim_title(store, HandoffChannel.CHAT) == "second"


def test_acknowledge_current_rejects_replaced_claim_and_preserves_replacement() -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, _chat_payload("first"))
    claim = store.claim(HandoffChannel.CHAT)
    assert claim is not None
    store.stage(HandoffChannel.CHAT, _chat_payload("replacement"))

    assert store.acknowledge_current(claim) is False
    assert store.release(claim) is True
    replacement = store.claim(HandoffChannel.CHAT)

    assert replacement is not None
    assert replacement.value.title == "replacement"
    assert store.acknowledge_current(replacement) is True


def test_acknowledge_current_is_idempotent_for_exact_claim() -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, _chat_payload())
    claim = store.claim(HandoffChannel.CHAT)
    assert claim is not None

    assert store.acknowledge_current(claim) is True
    assert store.acknowledge_current(claim) is False


def test_chat_stage_claim_and_release_values_are_structurally_detached() -> None:
    source = _chat_payload()
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, source)
    source.metadata["nested"]["items"].append("producer-change")

    claim = store.claim(HandoffChannel.CHAT)
    assert claim is not None
    claim.value.metadata["nested"]["items"].append("consumer-change")
    assert store.release(claim) is True
    retry = store.claim(HandoffChannel.CHAT)

    assert retry is not None
    assert retry.value.metadata["nested"]["items"] == ["original"]


def test_chat_mapping_is_normalized_through_payload_contract() -> None:
    store = PendingHandoffStore()

    store.stage(
        HandoffChannel.CHAT,
        {
            "source": "tests",
            "item_type": "document",
            "title": "mapping",
            "body": "context",
            "metadata": {"nested": {"items": ["original"]}},
        },
    )
    claim = store.claim(HandoffChannel.CHAT)

    assert claim is not None
    assert isinstance(claim.value, ChatHandoffPayload)
    assert claim.value.title == "mapping"


def test_console_stage_claim_and_pending_payload_are_structurally_detached() -> None:
    source_payload = {"nested": {"items": ["original"]}}
    launch = ConsoleLiveWorkLaunch.from_values(
        source="tests",
        title="launch",
        payload=source_payload,
    )
    source_payload["nested"]["items"].append("producer-before-stage")
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CONSOLE_LIVE_WORK, launch)
    launch.payload["nested"]["items"].append("producer-after-stage")

    claim = store.claim(HandoffChannel.CONSOLE_LIVE_WORK)
    assert claim is not None
    claim.value.payload["nested"]["items"].append("consumer-change")
    assert store.release(claim) is True
    retry = store.claim(HandoffChannel.CONSOLE_LIVE_WORK)
    pending_payload = retry.value.to_pending_payload()
    pending_payload["payload"]["nested"]["items"].append("pending-change")

    assert retry.value.payload["nested"]["items"] == ["original"]


def test_console_from_pending_detaches_an_existing_launch() -> None:
    launch = _console_launch()

    reconstructed = ConsoleLiveWorkLaunch.from_pending(launch)
    assert reconstructed is not None
    reconstructed.payload["nested"]["items"].append("reconstructed-change")

    assert launch.payload["nested"]["items"] == ["original"]


@pytest.mark.parametrize("prompt", ["", "prompt", object(), {"user_text": "prompt"}])
def test_prompt_channel_rejects_untyped_values_without_mutating_pending(
    prompt: object,
) -> None:
    store = PendingHandoffStore()
    existing = _prompt_application("existing prompt")
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, existing)

    with pytest.raises(HandoffValueError, match="normalized"):
        store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, prompt)

    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert claim is not None
    assert claim.value == existing
    assert claim.value is not existing
    assert claim.revision == 1


def test_prompt_stage_claim_and_release_values_are_structurally_detached() -> None:
    source = _prompt_application("private original")
    store = PendingHandoffStore(monotonic_clock=lambda: 20.0)

    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, source)
    object.__setattr__(source, "user_text", "producer mutation")

    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert claim is not None
    assert claim.value.user_text == "private original"
    assert claim.value is not source
    object.__setattr__(claim.value, "user_text", "consumer mutation")

    assert store.release(claim) is True
    retry = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert retry is not None
    assert retry.value.user_text == "private original"
    assert retry.value is not claim.value


def test_prompt_latest_pending_application_wins() -> None:
    store = PendingHandoffStore(monotonic_clock=lambda: 20.0)

    first_revision = store.stage(
        HandoffChannel.CONSOLE_PROMPT_INSERT,
        _prompt_application("first"),
    )
    second = _prompt_application("second")
    second_revision = store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, second)
    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert second_revision == first_revision + 1
    assert claim is not None
    assert claim.revision == second_revision
    assert claim.value == second
    assert claim.status == "ready"


def test_prompt_claim_is_one_shot_and_exclusive_until_exact_settlement() -> None:
    store = PendingHandoffStore(monotonic_clock=lambda: 20.0)
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, _prompt_application())

    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert claim is not None
    assert store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None
    assert store.acknowledge(replace(claim)) is False
    assert store.release(replace(claim)) is False
    assert store.acknowledge(claim) is True
    assert store.acknowledge(claim) is False
    assert store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None


def test_prompt_claim_status_changes_at_exact_120_second_boundary() -> None:
    now = [129.999]
    store = PendingHandoffStore(monotonic_clock=lambda: now[0])
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, _prompt_application())

    ready = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert ready is not None
    assert ready.status == "ready"
    assert store.release(ready) is True

    now[0] = 130.0
    expired = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert expired is not None
    assert expired.status == "expired"


def test_prompt_expiring_between_claim_and_release_is_not_requeued() -> None:
    now = [129.0]
    store = PendingHandoffStore(monotonic_clock=lambda: now[0])
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, _prompt_application())
    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert claim is not None
    assert claim.status == "ready"

    now[0] = 130.0

    assert store.release_prompt_claim(claim) == "expired"
    assert store.has_pending(HandoffChannel.CONSOLE_PROMPT_INSERT) is False
    assert store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None


def test_ready_prompt_release_reports_ready_and_requeues_exact_claim() -> None:
    store = PendingHandoffStore(monotonic_clock=lambda: 129.9)
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, _prompt_application())
    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert claim is not None
    assert store.release_prompt_claim(claim) == "ready"
    retry = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert retry is not None
    assert retry.status == "ready"


def test_prompt_release_rejects_non_claim_before_dereferencing() -> None:
    store = PendingHandoffStore()

    with pytest.raises(TypeError, match="HandoffClaim"):
        store.release_prompt_claim(object())  # type: ignore[arg-type]


def test_expired_prompt_claim_is_visible_once_and_never_requeued() -> None:
    store = PendingHandoffStore(monotonic_clock=lambda: 130.0)
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, _prompt_application())

    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert claim is not None
    assert claim.status == "expired"
    assert store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None
    assert store.release(claim) is True
    assert store.release(claim) is False
    assert store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT) is None


def test_expired_prompt_claim_can_be_acknowledged_exactly_once() -> None:
    store = PendingHandoffStore(monotonic_clock=lambda: 130.0)
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, _prompt_application())
    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert claim is not None
    assert claim.status == "expired"
    assert store.acknowledge(claim) is True
    assert store.acknowledge(claim) is False


def test_expired_old_prompt_release_preserves_newer_pending_revision() -> None:
    now = [20.0]
    store = PendingHandoffStore(monotonic_clock=lambda: now[0])
    store.stage(
        HandoffChannel.CONSOLE_PROMPT_INSERT,
        _prompt_application("old", created_monotonic=10.0),
    )
    old_claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert old_claim is not None
    store.stage(
        HandoffChannel.CONSOLE_PROMPT_INSERT,
        _prompt_application("new", created_monotonic=20.0),
    )

    now[0] = 130.0

    assert store.release(old_claim) is True
    new_claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert new_claim is not None
    assert new_claim.value.user_text == "new"
    assert new_claim.status == "ready"


def test_non_prompt_claims_are_always_ready() -> None:
    store = PendingHandoffStore(monotonic_clock=lambda: 10_000.0)
    store.stage(HandoffChannel.CHAT, _chat_payload())

    claim = store.claim(HandoffChannel.CHAT)

    assert claim is not None
    assert claim.status == "ready"


def test_claim_status_rejects_values_outside_the_bounded_contract() -> None:
    with pytest.raises(ValueError, match="status"):
        HandoffClaim(
            channel=HandoffChannel.CHAT,
            revision=1,
            value=_chat_payload(),
            status="waiting",  # type: ignore[arg-type]
        )


def test_store_rejects_a_non_callable_clock() -> None:
    with pytest.raises(TypeError, match="clock"):
        PendingHandoffStore(monotonic_clock=10.0)  # type: ignore[arg-type]


def test_prompt_claim_rejects_non_finite_clock_without_moving_pending() -> None:
    store = PendingHandoffStore(monotonic_clock=lambda: float("nan"))
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, _prompt_application())

    with pytest.raises(HandoffValueError, match="clock"):
        store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert store.has_pending(HandoffChannel.CONSOLE_PROMPT_INSERT) is True


def test_prompt_release_with_invalid_clock_settles_and_fails_closed() -> None:
    now = [20.0]
    store = PendingHandoffStore(monotonic_clock=lambda: now[0])
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, _prompt_application())
    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert claim is not None

    now[0] = float("inf")

    assert store.release(claim) is True
    assert store.has_pending(HandoffChannel.CONSOLE_PROMPT_INSERT) is False


def test_clock_failure_does_not_expose_exception_or_prompt_values() -> None:
    secret = "PRIVATE-CLOCK-AND-PROMPT-SENTINEL"

    def fail_clock() -> float:
        raise ValueError(secret)

    store = PendingHandoffStore(monotonic_clock=fail_clock)
    store.stage(
        HandoffChannel.CONSOLE_PROMPT_INSERT,
        _prompt_application(secret),
    )

    with pytest.raises(HandoffValueError) as caught:
        store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)


def test_clock_numeric_coercion_failure_is_bounded_and_keeps_pending(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "PRIVATE-CLOCK-COERCION-SENTINEL"

    class SecretFloat(float):
        def __float__(self) -> float:
            raise ValueError(secret)

        def __repr__(self) -> str:
            return secret

        def __str__(self) -> str:
            return secret

    caplog.set_level(logging.DEBUG)
    store = PendingHandoffStore(monotonic_clock=lambda: SecretFloat(20.0))
    store.stage(
        HandoffChannel.CONSOLE_PROMPT_INSERT,
        _prompt_application(secret),
    )

    with pytest.raises(HandoffValueError) as caught:
        store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert store.has_pending(HandoffChannel.CONSOLE_PROMPT_INSERT) is True
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
    assert secret not in caplog.text


@pytest.mark.parametrize(
    ("channel", "value"),
    [
        (HandoffChannel.CHAT, object()),
        (HandoffChannel.CONSOLE_LIVE_WORK, object()),
        (HandoffChannel.CONSOLE_PROMPT_INSERT, object()),
    ],
)
def test_invalid_value_leaves_no_partial_slot(
    channel: HandoffChannel,
    value: Any,
) -> None:
    store = PendingHandoffStore()

    with pytest.raises((TypeError, ValueError)):
        store.stage(channel, value)

    assert store.claim(channel) is None


def test_claim_repr_never_contains_payload_content() -> None:
    sentinel = "TASK-645-PRIVATE-SENTINEL"
    store = PendingHandoffStore()
    store.stage(
        HandoffChannel.CONSOLE_PROMPT_INSERT,
        _prompt_application(sentinel),
    )

    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert claim is not None
    assert sentinel not in repr(claim)


def test_normalization_failures_do_not_expose_input_values() -> None:
    sentinel = "TASK-645-NORMALIZATION-PRIVATE-SENTINEL"

    class SecretCopyFailure:
        def __deepcopy__(self, _memo):
            raise ValueError(sentinel)

        def __repr__(self) -> str:
            return sentinel

        def __str__(self) -> str:
            return sentinel

    console_launch = _console_launch()
    console_launch.payload["nested"]["private"] = SecretCopyFailure()
    values = (
        (HandoffChannel.CHAT, SecretCopyFailure()),
        (HandoffChannel.CONSOLE_LIVE_WORK, console_launch),
        (HandoffChannel.CONSOLE_PROMPT_INSERT, SecretCopyFailure()),
    )

    for channel, value in values:
        store = PendingHandoffStore()
        with pytest.raises(HandoffValueError) as caught:
            store.stage(channel, value)

        assert sentinel not in str(caught.value)
        assert sentinel not in repr(caught.value)
        assert store.claim(channel) is None


def test_store_has_no_persistence_or_backing_map_api() -> None:
    store = PendingHandoffStore()

    assert not hasattr(store, "to_dict")
    assert not hasattr(store, "serialize")
    assert not hasattr(store, "slots")


def test_all_mutations_reject_off_owner_thread() -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, _chat_payload())
    claim = store.claim(HandoffChannel.CHAT)
    assert claim is not None
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, _prompt_application())
    prompt_claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert prompt_claim is not None

    operations = (
        lambda: store.stage(HandoffChannel.CHAT, _chat_payload("worker")),
        lambda: store.clear_pending(HandoffChannel.CHAT),
        lambda: store.claim(HandoffChannel.CHAT),
        lambda: store.has_pending(HandoffChannel.CHAT),
        lambda: store.acknowledge(claim),
        lambda: store.release(claim),
        lambda: store.release_prompt_claim(prompt_claim),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        failures = [executor.submit(operation).exception() for operation in operations]

    assert all(
        isinstance(error, RuntimeError) and "owner thread" in str(error)
        for error in failures
    )


def test_study_scope_stage_claim_and_release_are_deeply_detached() -> None:
    source = _study_scope()
    store = PendingHandoffStore()

    store.stage(HandoffChannel.STUDY_SCOPE, source)
    source.source_items[0].locator["nested"]["items"].append("producer-change")
    claim = store.claim(HandoffChannel.STUDY_SCOPE)

    assert claim is not None
    assert isinstance(claim.value, StudyScopeContext)
    claim.value.source_items[0].locator["nested"]["items"].append("consumer-change")
    assert store.release(claim) is True
    retry = store.claim(HandoffChannel.STUDY_SCOPE)
    assert retry is not None
    assert retry.value.source_items[0].locator == {"nested": {"items": ["original"]}}


def test_study_section_uses_shared_valid_section_contract() -> None:
    store = PendingHandoffStore()

    for section in STUDY_INITIAL_SECTIONS:
        store.stage(HandoffChannel.STUDY_INITIAL_SECTION, section)
        claim = store.claim(HandoffChannel.STUDY_INITIAL_SECTION)
        assert claim is not None
        assert claim.value == section
        assert store.acknowledge(claim) is True


@pytest.mark.parametrize("section", ["", " ", "unsupported", object()])
def test_invalid_study_section_does_not_replace_pending(section: Any) -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.STUDY_INITIAL_SECTION, "dashboard")

    with pytest.raises(HandoffValueError, match="normalized"):
        store.stage(HandoffChannel.STUDY_INITIAL_SECTION, section)

    claim = store.claim(HandoffChannel.STUDY_INITIAL_SECTION)
    assert claim is not None
    assert claim.value == "dashboard"
    assert claim.revision == 1


@pytest.mark.parametrize(
    ("channel", "raw", "canonical"),
    [
        (
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            " local:chatbook:chatbook-77 ",
            "local:chatbook:chatbook-77",
        ),
        (
            HandoffChannel.ACP_SESSION_TARGET,
            " local:acp_session:session-1 ",
            "local:acp_session:session-1",
        ),
    ],
)
def test_record_target_channels_return_complete_canonical_ids(
    channel: HandoffChannel,
    raw: str,
    canonical: str,
) -> None:
    store = PendingHandoffStore()

    store.stage(channel, raw)
    claim = store.claim(channel)

    assert claim is not None
    assert claim.value == canonical


@pytest.mark.parametrize(
    ("channel", "valid", "invalid"),
    [
        (
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
            "remote:chatbook:77",
        ),
        (
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
            "local:chatbook:   ",
        ),
        (
            HandoffChannel.ACP_SESSION_TARGET,
            "local:acp_session:session-1",
            "remote:acp_session:session-1",
        ),
        (
            HandoffChannel.ACP_SESSION_TARGET,
            "local:acp_session:session-1",
            object(),
        ),
    ],
)
def test_invalid_record_target_does_not_replace_pending(
    channel: HandoffChannel,
    valid: str,
    invalid: Any,
) -> None:
    store = PendingHandoffStore()
    store.stage(channel, valid)

    with pytest.raises(HandoffValueError, match="normalized"):
        store.stage(channel, invalid)

    claim = store.claim(channel)
    assert claim is not None
    assert claim.value == valid
    assert claim.revision == 1


def test_remaining_channels_have_independent_revisions_and_claims() -> None:
    store = PendingHandoffStore()

    assert store.stage(HandoffChannel.STUDY_SCOPE, _study_scope()) == 1
    assert store.stage(HandoffChannel.STUDY_INITIAL_SECTION, "dashboard") == 1
    assert (
        store.stage(
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
        )
        == 1
    )
    assert (
        store.stage(
            HandoffChannel.ACP_SESSION_TARGET,
            "local:acp_session:session-1",
        )
        == 1
    )

    claims = {
        channel: store.claim(channel)
        for channel in (
            HandoffChannel.STUDY_SCOPE,
            HandoffChannel.STUDY_INITIAL_SECTION,
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            HandoffChannel.ACP_SESSION_TARGET,
        )
    }
    assert all(claim is not None and claim.revision == 1 for claim in claims.values())


def test_audio_cpp_channels_detach_and_keep_foreign_claims_independent() -> None:
    store = PendingHandoffStore()
    request = AudioCppModelLibraryRequest("request-token", 3)
    result = AudioCppModelLibraryResult(
        "request-token",
        3,
        "audio-cpp-model",
        "a" * 40,
        "f16",
        "/managed/audio-cpp-model",
    )
    store.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)
    store.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT, result)

    request_claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    result_claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)

    assert request_claim is not None
    assert result_claim is not None
    assert request_claim.value == request
    assert request_claim.value is not request
    assert result_claim.value == result
    assert result_claim.value is not result
    assert store.acknowledge(request_claim) is True
    assert store.acknowledge(request_claim) is False
    assert store.release(result_claim) is True
    replay = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)
    assert replay is not None
    assert replay.value == result
    assert store.acknowledge(replay) is True


@pytest.mark.parametrize(
    ("channel", "first", "replacement"),
    [
        (
            HandoffChannel.ARTIFACT_CHATBOOK_TARGET,
            "local:chatbook:77",
            "local:chatbook:78",
        ),
        (
            HandoffChannel.ACP_SESSION_TARGET,
            "local:acp_session:session-1",
            "local:acp_session:session-2",
        ),
    ],
)
def test_remaining_target_replacement_survives_older_claim_settlement(
    channel: HandoffChannel,
    first: str,
    replacement: str,
) -> None:
    store = PendingHandoffStore()
    store.stage(channel, first)
    old_claim = store.claim(channel)
    assert old_claim is not None

    store.stage(channel, replacement)

    assert store.release(old_claim) is True
    replacement_claim = store.claim(channel)
    assert replacement_claim is not None
    assert replacement_claim.value == replacement


@pytest.mark.parametrize(
    ("channel", "value"),
    [
        (HandoffChannel.STUDY_SCOPE, _study_scope()),
        (HandoffChannel.STUDY_INITIAL_SECTION, "dashboard"),
    ],
)
def test_clearing_optional_study_channel_during_claim_prevents_resurrection(
    channel: HandoffChannel,
    value: Any,
) -> None:
    store = PendingHandoffStore()
    store.stage(channel, value)
    claim = store.claim(channel)
    assert claim is not None

    store.clear_pending(channel)

    assert store.release(claim) is True
    assert store.claim(channel) is None


def test_acp_session_record_id_normalizes_bare_session_ids() -> None:
    assert runtime_session.acp_session_record_id(None) is None
    assert runtime_session.acp_session_record_id(" \n ") is None
    assert (
        runtime_session.acp_session_record_id(" session-1 ")
        == "local:acp_session:session-1"
    )


@pytest.mark.parametrize(
    ("target_id", "session_id", "expected"),
    [
        ("local:acp_session:session-1", "session-1", True),
        ("local:acp_session:session-2", "session-1", False),
        (None, "session-1", False),
        (object(), "session-1", False),
        ("local:acp_session:session-1", None, False),
    ],
)
def test_current_acp_session_record_matcher_rejects_malformed_direct_inputs(
    target_id: Any,
    session_id: Any,
    expected: bool,
) -> None:
    assert (
        runtime_session.is_current_acp_session_record(target_id, session_id) is expected
    )


def test_acp_console_launch_uses_canonical_session_record_id() -> None:
    state = runtime_session.ACPRuntimeSessionState(
        runtime_id="runtime-1",
        session_id=" session-1 ",
        session_payload={"status": "ready"},
    )

    launch = state.to_console_live_work_launch()

    assert launch is not None
    assert launch.payload["target_id"] == "local:acp_session:session-1"


def test_provider_intent_is_normalized_and_contains_only_provider_identity() -> None:
    intent = ConsoleProviderIntent(provider="  Custom-OpenAI API  ")

    assert intent.provider == "custom_openai_api"
    assert [field.name for field in fields(intent)] == ["provider"]
    assert repr(intent) == "ConsoleProviderIntent(provider='custom_openai_api')"


@pytest.mark.parametrize("provider", ["", " \t "])
def test_provider_intent_rejects_blank_identity(provider: str) -> None:
    with pytest.raises(ValueError, match="provider"):
        ConsoleProviderIntent(provider=provider)


@pytest.mark.parametrize("provider", ["../private", "provider!", "éxample", "a" * 129])
def test_provider_intent_rejects_invalid_identity(provider: str) -> None:
    with pytest.raises(ValueError, match="provider"):
        ConsoleProviderIntent(provider=provider)


@pytest.mark.parametrize("provider", [None, 42])
def test_provider_intent_rejects_non_text_identity(provider: object) -> None:
    with pytest.raises(TypeError, match="provider"):
        ConsoleProviderIntent(provider=provider)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        {"provider": "openai", "api_key": "PRIVATE_API_KEY"},
        "openai",
    ],
)
def test_provider_channel_rejects_untyped_values(value: object) -> None:
    store = PendingHandoffStore()

    with pytest.raises(HandoffValueError):
        store.stage(HandoffChannel.CONSOLE_PROVIDER, value)


def test_provider_channel_replaces_pending_intent_while_claim_is_in_flight() -> None:
    store = PendingHandoffStore()
    first_revision = store.stage(
        HandoffChannel.CONSOLE_PROVIDER,
        ConsoleProviderIntent(provider="OpenAI"),
    )
    first_claim = store.claim(HandoffChannel.CONSOLE_PROVIDER)
    second_revision = store.stage(
        HandoffChannel.CONSOLE_PROVIDER,
        ConsoleProviderIntent(provider="Anthropic"),
    )

    assert first_claim is not None
    assert first_claim.revision == first_revision
    assert first_claim.value == ConsoleProviderIntent(provider="openai")
    assert second_revision > first_revision
    assert store.acknowledge(first_claim) is True
    assert store.acknowledge(first_claim) is False

    second_claim = store.claim(HandoffChannel.CONSOLE_PROVIDER)
    assert second_claim is not None
    assert second_claim.revision == second_revision
    assert second_claim.value == ConsoleProviderIntent(provider="anthropic")


def test_provider_channel_release_retries_the_exact_claim() -> None:
    store = PendingHandoffStore()
    store.stage(
        HandoffChannel.CONSOLE_PROVIDER,
        ConsoleProviderIntent(provider="OpenRouter"),
    )
    claim = store.claim(HandoffChannel.CONSOLE_PROVIDER)

    assert claim is not None
    assert store.release(claim) is True
    assert store.release(claim) is False

    retry = store.claim(HandoffChannel.CONSOLE_PROVIDER)
    assert retry is not None
    assert retry.revision == claim.revision
    assert retry is not claim
    assert retry.value == ConsoleProviderIntent(provider="openrouter")


def test_provider_intent_repr_cannot_contain_private_payload_fields() -> None:
    private_sentinels = {
        "credential": "PRIVATE_API_KEY",
        "endpoint": "https://private.example/v1",
        "prompt": "PRIVATE_SYSTEM_PROMPT",
        "response": "PRIVATE_RESPONSE_BODY",
        "catalog": "PRIVATE_CATALOG_PAYLOAD",
    }
    store = PendingHandoffStore()
    intent = ConsoleProviderIntent(provider="OpenAI")
    store.stage(HandoffChannel.CONSOLE_PROVIDER, intent)
    claim = store.claim(HandoffChannel.CONSOLE_PROVIDER)

    assert claim is not None
    rendered = repr(intent) + repr(claim)
    assert "openai" in rendered
    for sentinel in private_sentinels.values():
        assert sentinel not in rendered


def test_first_chat_intent_has_only_secret_free_target_fields() -> None:
    intent = ConsoleFirstChatIntent(
        session_id="session-1",
        provider="Custom-OpenAI API",
        model="private-model-name",
        config_revision=17,
    )

    assert tuple(item.name for item in fields(intent)) == (
        "session_id",
        "provider",
        "model",
        "config_revision",
    )
    assert intent.provider == "custom_openai_api"
    assert "endpoint" not in repr(intent).casefold()
    assert "credential" not in repr(intent).casefold()


def test_first_chat_channel_replacement_and_release_preserve_latest_intent() -> None:
    store = PendingHandoffStore()
    first = ConsoleFirstChatIntent("session-1", "openai", "model-a", 17)
    second = ConsoleFirstChatIntent("session-2", "llama_cpp", "model-b", 18)
    first_revision = store.stage(HandoffChannel.CONSOLE_FIRST_CHAT, first)
    first_claim = store.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    second_revision = store.stage(HandoffChannel.CONSOLE_FIRST_CHAT, second)

    assert first_claim is not None
    assert first_claim.revision == first_revision
    assert store.release(first_claim) is True

    replacement = store.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert replacement is not None
    assert replacement.revision == second_revision
    assert replacement.value == second
    assert replacement.value is not second
    assert store.acknowledge(replacement) is True


def test_reserved_first_chat_target_metadata_follows_exact_claim_revision() -> None:
    store = PendingHandoffStore()
    reserved = ConsoleFirstChatIntent("reserved-1", "openai", "model-a", 17)
    replacement = ConsoleFirstChatIntent("existing-2", "openai", "model-b", 18)
    store.stage_reserved_console_first_chat(reserved)
    reserved_claim = store.claim(HandoffChannel.CONSOLE_FIRST_CHAT)

    assert reserved_claim is not None
    assert store.claim_reserves_new_console_session(reserved_claim) is True
    assert store.claim_reserves_new_console_session(replace(reserved_claim)) is False

    store.stage(HandoffChannel.CONSOLE_FIRST_CHAT, replacement)
    assert store.release(reserved_claim) is True
    next_claim = store.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert next_claim is not None
    assert next_claim.value == replacement
    assert store.claim_reserves_new_console_session(next_claim) is False


def test_reserved_first_chat_claim_release_retains_reservation_for_retry() -> None:
    store = PendingHandoffStore()
    intent = ConsoleFirstChatIntent("reserved-1", "llama_cpp", "model-a", 17)
    store.stage_reserved_console_first_chat(intent)
    first_claim = store.claim(HandoffChannel.CONSOLE_FIRST_CHAT)

    assert first_claim is not None
    assert store.release(first_claim) is True
    retry = store.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert retry is not None
    assert retry.revision == first_claim.revision
    assert store.claim_reserves_new_console_session(retry) is True


def test_first_chat_claim_is_current_only_until_replaced() -> None:
    store = PendingHandoffStore()
    first = ConsoleFirstChatIntent("session-1", "openai", "model-a", 17)
    replacement = ConsoleFirstChatIntent("session-2", "openai", "model-b", 18)
    store.stage(HandoffChannel.CONSOLE_FIRST_CHAT, first)
    claim = store.claim(HandoffChannel.CONSOLE_FIRST_CHAT)

    assert claim is not None
    assert store.is_current_claim(claim) is True

    store.stage(HandoffChannel.CONSOLE_FIRST_CHAT, replacement)

    assert store.is_current_claim(claim) is False
    assert store.release(claim) is True
    next_claim = store.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert next_claim is not None
    assert next_claim.value == replacement
