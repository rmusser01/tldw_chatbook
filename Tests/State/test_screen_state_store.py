from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields
import logging
from types import SimpleNamespace

import pytest

import tldw_chatbook.UI.Navigation.screen_state_store as screen_state_store
from tldw_chatbook.UI.Navigation.screen_state_store import (
    ConsolePromptTargetProjection,
    RuntimeIdentity,
    ScreenStateStore,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleSettingsDraftSnapshot,
)


_SYSTEM_FINGERPRINT = "sha256:" + "c" * 64


def _local_identity() -> RuntimeIdentity:
    return RuntimeIdentity("local")


def _projection(
    *,
    target_session_id: str = "session-1",
    system_fingerprint: str = _SYSTEM_FINGERPRINT,
) -> ConsolePromptTargetProjection:
    return ConsolePromptTargetProjection(
        target_session_id=target_session_id,
        system_fingerprint=system_fingerprint,
    )


def test_suspended_conversation_draft_snapshot_rejects_malformed_nested_state() -> None:
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-temperature": "0.7"},
        provider_model_drafts={"openai": "gpt-5"},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    malformed = snapshot.to_mapping()
    malformed["raw_values"] = {"not-a-modal-control": "value"}

    assert ConsoleSettingsDraftSnapshot.from_mapping(malformed) is None


def test_native_console_state_keeps_suspended_settings_draft_process_local() -> None:
    """The screen snapshot, not a handoff or route, owns raw draft content."""
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            base_url="http://127.0.0.1:9099",
            system_prompt="private system text",
            pinned_prefill="private prefill text",
        ),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-base-url": "http://127.0.0.1:9099"},
        provider_model_drafts={"llama_cpp": "model-a"},
        provider_base_url_drafts={"llama_cpp": "http://127.0.0.1:9099"},
        active_view="model",
        scroll_anchor=4,
        focus_control_id="console-settings-model-picker",
        disclosure_state={"advanced_generation": False, "connection_details": True},
    )

    def bare_screen(store: ConsoleChatStore) -> ChatScreen:
        screen = ChatScreen.__new__(ChatScreen)
        image_state = SimpleNamespace(
            prune=lambda _ids: None,
            serialize=lambda: {},
            restore=lambda _value: None,
        )
        screen._console_runtime_ref = SimpleNamespace(
            chat_store=store,
            set_chat_store=lambda value: setattr(
                screen._console_runtime_ref, "chat_store", value
            ),
        )
        screen._session = SimpleNamespace(_console_visible_draft_session_id=None)
        screen._stash_console_pending_attachments = lambda _store: None
        screen._console_visible_draft_session_id = None
        screen._console_composer_or_none = lambda: None
        screen._ensure_console_image_view = lambda: (
            image_state,
            SimpleNamespace(clear=lambda: None),
        )
        screen._task_resume_state = TaskResumeState()
        screen._console_library_rag_source_types = ("media", "notes", "conversations")
        screen._pending_console_launch_context = None
        screen._console_evidence_sent_notice = None
        screen._message = SimpleNamespace(
            invalidate_console_speech_context=lambda: None,
        )
        screen._ensure_console_chat_store = lambda: store
        screen._adopt_console_pending_attachments = lambda _store: None
        return screen

    store = ConsoleChatStore()
    store.create_session(settings=snapshot.settings)
    screen = bare_screen(store)
    screen._suspended_conversation_settings = snapshot

    payload = screen._serialize_native_console_state()

    assert payload is not None
    retained = payload["suspended_conversation_settings"]
    assert retained is not None
    assert retained["settings"]["system_prompt"] == "private system text"
    assert retained["settings"]["pinned_prefill"] == "private prefill text"
    assert retained["raw_values"]["console-settings-base-url"] == "http://127.0.0.1:9099"

    restored = bare_screen(ConsoleChatStore())
    restored._restore_native_console_state(payload)

    assert restored._suspended_conversation_settings == snapshot


def test_console_prompt_target_projection_is_minimal_frozen_and_safe() -> None:
    projection = _projection()

    assert tuple(item.name for item in fields(projection)) == (
        "target_session_id",
        "system_fingerprint",
    )
    assert not hasattr(projection, "__dict__")
    assert not hasattr(projection, "system_text")
    assert not hasattr(projection, "composer_text")
    assert _SYSTEM_FINGERPRINT not in repr(projection)
    for forbidden in ("to_dict", "serialize", "persist", "save"):
        assert not hasattr(projection, forbidden)
    with pytest.raises(FrozenInstanceError):
        projection.target_session_id = "session-2"  # type: ignore[misc]


def test_console_target_has_no_parallel_backing_store() -> None:
    store = ScreenStateStore()

    assert tuple(
        item.name for item in fields(screen_state_store._SnapshotEnvelope)
    ) == (
        "canonical_route",
        "snapshot",
        "runtime_identity",
        "console_prompt_target",
    )
    assert not hasattr(screen_state_store, "_ConsolePromptTargetEnvelope")
    assert not hasattr(store, "_console_prompt_targets")


@pytest.mark.parametrize(
    "target_session_id",
    ["", "   ", " session-1", "session-1 ", None, 1],
)
def test_console_prompt_target_projection_requires_a_target_session(
    target_session_id: object,
) -> None:
    with pytest.raises(ValueError, match="target session"):
        ConsolePromptTargetProjection(
            target_session_id=target_session_id,  # type: ignore[arg-type]
            system_fingerprint=_SYSTEM_FINGERPRINT,
        )


@pytest.mark.parametrize(
    "system_fingerprint",
    ["", "c" * 64, "sha256:" + "c" * 63, "sha256:" + "C" * 64, None, 1],
)
def test_console_prompt_target_projection_requires_system_fingerprint_shape(
    system_fingerprint: object,
) -> None:
    with pytest.raises(ValueError, match="System fingerprint"):
        ConsolePromptTargetProjection(
            target_session_id="session-1",
            system_fingerprint=system_fingerprint,  # type: ignore[arg-type]
        )


def test_existing_snapshot_outer_copy_behavior_is_unchanged() -> None:
    nested = {"history": ["large", "payload"]}
    original = {"selected": "row-1", "nested": nested}
    store = ScreenStateStore()
    identity = _local_identity()

    store.save("chat", original, identity)
    original["selected"] = "changed-after-save"
    restored = store.restore("chat", identity)

    assert restored == {"selected": "row-1", "nested": nested}
    assert restored is not original
    assert restored["nested"] is nested
    restored["selected"] = "consumer-change"
    assert store.restore("chat", identity)["selected"] == "row-1"


def test_library_screen_state_runtime_mismatch_rejects_continue_before_restore() -> None:
    store = ScreenStateStore()
    server_a = RuntimeIdentity("server", "server-a")
    server_b = RuntimeIdentity("server", "server-b")
    receipt = {
        "version": 1,
        "row_id": "browse-media",
        "scope": {"query": "", "media_type": None, "sort_by": "title_asc", "page": 2},
        "source_list_adjusted": False,
    }
    store.save(
        "library",
        {"library_selected_row_id": "", "library_continue_receipt": receipt},
        server_a,
    )

    assert store.restore("library", server_b) is None
    assert store.restore("library", server_a) is None


def test_publish_and_restore_return_detached_console_target_projections() -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    original = _projection()
    store.save("chat", {"conversation_id": "session-1"}, identity)

    store.publish_console_prompt_target("chat", original, identity)
    first = store.restore_console_prompt_target("chat", identity)
    second = store.restore_console_prompt_target("chat", identity)

    assert first == original
    assert second == original
    assert first is not original
    assert second is not original
    assert second is not first


def test_replacing_snapshot_invalidates_the_existing_console_target() -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("chat", {"conversation_id": "session-1"}, identity)
    store.publish_console_prompt_target("chat", _projection(), identity)

    store.save("chat", {"conversation_id": "session-2"}, identity)

    assert store.restore_console_prompt_target("chat", identity) is None
    assert store.restore("chat", identity) == {"conversation_id": "session-2"}


def test_console_target_is_route_scoped_and_absent_before_publication() -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("chat", {}, identity)

    assert store.restore_console_prompt_target("chat", identity) is None
    store.publish_console_prompt_target("chat", _projection(), identity)
    assert store.restore_console_prompt_target("library", identity) is None


def test_console_target_publication_requires_a_route_snapshot() -> None:
    store = ScreenStateStore()
    identity = _local_identity()

    with pytest.raises(ValueError, match="compatible screen snapshot"):
        store.publish_console_prompt_target("chat", _projection(), identity)

    assert store.restore_console_prompt_target("chat", identity) is None
    assert store.has_snapshots(identity) is False


def test_console_target_publication_rejects_an_incompatible_route_snapshot() -> None:
    store = ScreenStateStore()
    server_a = RuntimeIdentity("server", "server-a")
    server_b = RuntimeIdentity("server", "server-b")
    store.save("chat", {"conversation_id": "session-1"}, server_a)

    with pytest.raises(ValueError, match="compatible screen snapshot"):
        store.publish_console_prompt_target("chat", _projection(), server_b)

    assert store.restore_console_prompt_target("chat", server_b) is None
    assert store.restore("chat", server_a) is None


def test_runtime_incompatible_target_restore_discards_target_and_snapshot() -> None:
    store = ScreenStateStore()
    server_a = RuntimeIdentity("server", "server-a")
    server_b = RuntimeIdentity("server", "server-b")
    store.save("chat", {"conversation_id": "session-1"}, server_a)
    store.publish_console_prompt_target("chat", _projection(), server_a)

    assert store.restore_console_prompt_target("chat", server_b) is None
    assert store.restore("chat", server_a) is None
    assert store.restore_console_prompt_target("chat", server_a) is None


def test_incompatible_snapshot_restore_discards_corresponding_target() -> None:
    store = ScreenStateStore()
    server_a = RuntimeIdentity("server", "server-a")
    server_b = RuntimeIdentity("server", "server-b")
    store.save("chat", {"conversation_id": "session-1"}, server_a)
    store.publish_console_prompt_target("chat", _projection(), server_a)

    assert store.restore("chat", server_b) is None
    assert store.restore_console_prompt_target("chat", server_a) is None


def test_discard_removes_snapshot_and_corresponding_target() -> None:
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("chat", {"conversation_id": "session-1"}, identity)
    store.publish_console_prompt_target("chat", _projection(), identity)

    store.discard("chat")
    store.discard("chat")

    assert store.restore("chat", identity) is None
    assert store.restore_console_prompt_target("chat", identity) is None


def test_has_snapshots_runtime_cleanup_also_discards_corresponding_target() -> None:
    store = ScreenStateStore()
    server_a = RuntimeIdentity("server", "server-a")
    server_b = RuntimeIdentity("server", "server-b")
    store.save("chat", {"conversation_id": "session-1"}, server_a)
    store.publish_console_prompt_target("chat", _projection(), server_a)

    assert store.has_snapshots(server_b) is False
    assert store.restore_console_prompt_target("chat", server_a) is None


@pytest.mark.parametrize(
    "operation",
    [
        lambda store, identity: store.save("chat", {}, identity),
        lambda store, identity: store.restore("chat", identity),
        lambda store, _identity: store.discard("chat"),
        lambda store, identity: store.has_snapshots(identity),
        lambda store, identity: store.publish_console_prompt_target(
            "chat", _projection(), identity
        ),
        lambda store, identity: store.restore_console_prompt_target("chat", identity),
    ],
)
def test_all_snapshot_and_target_operations_reject_off_owner_thread(
    operation: Callable[[ScreenStateStore, RuntimeIdentity], object],
) -> None:
    store = ScreenStateStore()
    identity = _local_identity()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operation, store, identity)
        with pytest.raises(RuntimeError, match="owner thread"):
            future.result()


def test_target_validation_refusal_does_not_mutate_or_log_sensitive_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "CONSOLE-SYSTEM-TEXT-SECRET-b103"
    store = ScreenStateStore()
    identity = _local_identity()
    store.save("chat", {"private": secret}, identity)
    caplog.set_level(logging.DEBUG)

    with pytest.raises(TypeError, match="ConsolePromptTargetProjection") as caught:
        store.publish_console_prompt_target("chat", secret, identity)  # type: ignore[arg-type]

    assert store.restore_console_prompt_target("chat", identity) is None
    assert secret not in str(caught.value)
    assert secret not in caplog.text
