from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import pytest

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
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


def _claim_title(store: PendingHandoffStore, channel: HandoffChannel) -> str:
    claim = store.claim(channel)
    assert claim is not None
    title = claim.value.title
    assert store.acknowledge(claim) is True
    return title


def test_stage_replaces_unclaimed_value_with_channel_local_revision() -> None:
    store = PendingHandoffStore()

    assert store.stage(HandoffChannel.CHAT, _chat_payload("first")) == 1
    assert store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, "prompt") == 1
    assert store.stage(HandoffChannel.CHAT, _chat_payload("second")) == 2

    assert _claim_title(store, HandoffChannel.CHAT) == "second"


def test_claim_is_exclusive_until_exact_claim_settles() -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, _chat_payload())

    claim = store.claim(HandoffChannel.CHAT)

    assert claim is not None
    assert store.claim(HandoffChannel.CHAT) is None
    assert store.acknowledge(replace(claim)) is False
    assert store.acknowledge(claim) is True
    assert store.claim(HandoffChannel.CHAT) is None


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


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_prompt_rejects_empty_text_without_mutating_existing_pending(
    prompt: str,
) -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, "existing prompt")

    with pytest.raises(ValueError, match="normalized"):
        store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, prompt)

    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert claim is not None
    assert claim.value == "existing prompt"
    assert claim.revision == 1


def test_prompt_preserves_user_text_exactly() -> None:
    store = PendingHandoffStore()
    prompt = "  keep surrounding whitespace\n"

    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, prompt)

    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)
    assert claim is not None
    assert claim.value == prompt


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
    store.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, sentinel)

    claim = store.claim(HandoffChannel.CONSOLE_PROMPT_INSERT)

    assert claim is not None
    assert sentinel not in repr(claim)


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

    operations = (
        lambda: store.stage(HandoffChannel.CHAT, _chat_payload("worker")),
        lambda: store.clear_pending(HandoffChannel.CHAT),
        lambda: store.claim(HandoffChannel.CHAT),
        lambda: store.acknowledge(claim),
        lambda: store.release(claim),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        failures = [executor.submit(operation).exception() for operation in operations]

    assert all(
        isinstance(error, RuntimeError) and "owner thread" in str(error)
        for error in failures
    )
