"""An untouched open Console chat follows newly saved defaults (TASK-33001.5, D1).

ADR-095's 2026-09-26 amendment: a pristine chat -- no messages, no user work,
settings still equal to the canonical baseline it was created with --
converges to the saved defaults whatever its readiness. The task-177 refresh
used to converge only a *blocked* chat and only onto *send-capable* defaults,
so a keyless llama.cpp chat (which reads Ready even with its server down) kept
its old provider, model and sampling after a Settings save.

These drive the real ``ConsoleSessionController`` (built by a real
``ChatScreen``) over a real ``ConsoleChatStore`` and the real
``blank_console_session_settings``; only the app object is a mock, holding an
in-memory config mapping. Replacing that mapping stands in for a published
config write, which is what a Settings save does to the fresh-config seam.
"""

from __future__ import annotations

import builtins
import io
import os
import socket
import sqlite3
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

import tldw_chatbook.UI.Console_Modules.session as session_module
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    blank_console_session_settings,
    build_console_settings_readiness,
)
from tldw_chatbook.Chat.provider_readiness import provider_config_key

LLAMA_URL = "http://127.0.0.1:9099"


@pytest.fixture(autouse=True)
def _no_cloud_keys(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _config(provider: str, model: str, **chat_defaults) -> dict:
    return {
        "chat_defaults": {"provider": provider, "model": model, **chat_defaults},
        "api_settings": {
            "llama_cpp": {"api_url": LLAMA_URL},
            "openai": {"api_key": ""},
            "anthropic": {"api_key": ""},
        },
    }


def _console(config: dict, *, generation: int = 0):
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    app = MagicMock(app_config=config, console_new_chat_default_generation=generation)
    console = ChatScreen(app)
    store = ConsoleChatStore()
    console._console_chat_store = store
    return app, console, store


def _pristine(store: ConsoleChatStore, settings: ConsoleSessionSettings, **kwargs):
    return store.create_session(
        settings=settings, canonical_settings_baseline=settings, **kwargs
    )


def _ensure(console) -> ConsoleSessionSettings:
    return console._session._ensure_active_console_session_settings()


def _label(settings: ConsoleSessionSettings, config: dict) -> str:
    return build_console_settings_readiness(
        settings, app_config=config, environ={}
    ).label


# The chat's settings before the save, and the readiness label that proves it.
PREVIOUS = {
    "ready-keyless-llama": (
        ConsoleSessionSettings(
            provider="llama_cpp", model="old-model", temperature=0.9
        ),
        "Ready",
    ),
    "missing-key-openai": (
        ConsoleSessionSettings(provider="openai", model="gpt-4o"),
        "Missing key",
    ),
    "unknown-provider": (
        ConsoleSessionSettings(provider="mystery-provider", model="m-1"),
        "Unknown",
    ),
}
# The saved defaults after the save: one that can send, one that cannot yet.
SAVED = {
    "sendable-llama": ("llama_cpp", "new-model"),
    "keyless-anthropic": ("anthropic", "claude-test"),
}


@pytest.mark.parametrize("saved", sorted(SAVED))
@pytest.mark.parametrize("previous", sorted(PREVIOUS))
def test_pristine_chat_converges_to_saved_defaults_whatever_its_readiness(
    previous, saved
):
    """AC#1/#2: Ready, blocked and unknown chats all take the saved defaults."""
    before, expected_label = PREVIOUS[previous]
    provider, model = SAVED[saved]
    config = _config(provider, model, temperature=0.3)
    _app, console, store = _console(config)
    session = _pristine(store, before)
    assert _label(before, config).startswith(expected_label)

    shown = _ensure(console)

    blank = blank_console_session_settings(config)
    assert shown == blank, "a converged chat must equal a blank chat made now"
    assert (provider_config_key(shown.provider), shown.model) == (provider, model)
    assert shown.temperature == pytest.approx(0.3)
    assert store.session_settings(session.id) == blank
    # Still pristine: the next save converges it again.
    assert session.canonical_settings_baseline == blank
    assert session.has_user_work is False
    if saved == "keyless-anthropic":
        assert not build_console_settings_readiness(
            shown, app_config=config, environ={}
        ).native_send_supported


def _add_message(store, session):
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")


def _mark_draft(store, session):
    store.set_session_draft(session.id, "typing")


def _edit_setting(store, session):
    store.replace_session_settings(
        session.id, replace(session.settings, temperature=0.11)
    )


def _apply_system_prompt(store, session):
    store.set_session_system_prompt(session.id, "Be concise.")


@pytest.mark.parametrize(
    "add_work",
    [_add_message, _mark_draft, _edit_setting, _apply_system_prompt],
    ids=["message", "user-work-marker", "edited-setting", "system-prompt"],
)
def test_chat_with_work_keeps_its_settings(add_work):
    """AC#5: any message, marker, edited field or /system prompt blocks it."""
    config = _config("llama_cpp", "new-model")
    _app, console, store = _console(config)
    session = _pristine(
        store, ConsoleSessionSettings(provider="openai", model="gpt-4o")
    )
    add_work(store, session)
    kept = store.session_settings(session.id)

    assert _ensure(console) == kept
    assert store.session_settings(session.id) == kept


def test_source_owned_chat_never_converges():
    """AC#6: Duplicate/Branch/Continue/handoff chats carry no baseline."""
    config = _config("llama_cpp", "new-model")
    _app, console, store = _console(config)
    source = ConsoleSessionSettings(provider="openai", model="source-model")
    session = store.create_session(settings=source)
    assert session.canonical_settings_baseline is None

    assert _ensure(console) == source


def test_pristine_chat_created_before_make_default_is_skipped():
    """AC#7: the creation-time default-generation check is unchanged."""
    config = _config("llama_cpp", "new-model")
    app, console, store = _console(config)
    before = ConsoleSessionSettings(provider="llama_cpp", model="old-model")
    session = _pristine(store, before)
    assert session.new_chat_default_generation == 0
    app.console_new_chat_default_generation = 1

    assert _ensure(console) == before


def test_make_default_write_before_its_publication_cannot_move_the_chat_it_came_from():
    """AC#7 in the Make-default write-then-publish window.

    ``apply_console_default_intent`` rewrites the file and reloads the config
    cache on a worker; ``console_new_chat_default_generation`` is bumped later,
    on the UI thread, when ChatScreen accepts the publication. A render in
    between reads the new default with the old generation. The chat that
    Make default was pressed on is outside that window by construction: both
    surfaces live-commit their draft first (``commit_console_settings_live``
    stamps ``source="user"``), so it is no longer pristine. An inactive chat is
    never re-derived; only a switch to it inside the window could be.
    """
    from tldw_chatbook.Chat.console_context_policy import (
        ConsoleContextPolicyOverrides,
    )
    from tldw_chatbook.Chat.console_settings_apply import (
        QUICK_MODEL_DEFAULT_FIELDS,
        ConsoleSettingsAction,
        ConsoleSettingsDraftState,
        ConsoleSettingsFieldDraft,
        ConsoleSettingsFieldProvenance,
        ConsoleSettingsSubmission,
        ConsoleSettingsSurface,
    )

    config = _config("llama_cpp", "old-default")
    app, console, store = _console(config)
    background = _pristine(store, blank_console_session_settings(config))
    origin = _pristine(store, blank_console_session_settings(config))
    assert store.active_session_id == origin.id
    made_default = replace(origin.settings, model="made-default")
    committed = store.commit_console_settings_live(
        ConsoleSettingsSubmission(
            submission_id="make-default-1",
            action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            surface=ConsoleSettingsSurface.QUICK_POPOVER,
            origin=store.capture_console_settings_origin(origin.id),
            draft=ConsoleSettingsDraftState(
                settings=made_default,
                context_policy_overrides=ConsoleContextPolicyOverrides(),
                # A quick submission carries one draft per quick-mask field
                # (TASK-33004.1 added max_tokens to that mask).
                field_drafts=tuple(
                    ConsoleSettingsFieldDraft(
                        name=name,
                        effective_value=getattr(made_default, name),
                        profile_override=getattr(made_default, name),
                        provenance=ConsoleSettingsFieldProvenance.INHERITED,
                        dirty=False,
                    )
                    for name in sorted(QUICK_MODEL_DEFAULT_FIELDS)
                ),
                model_drafts=(),
                endpoint_draft=None,
            ),
            user_display_name_override=None,
            default_field_mask=QUICK_MODEL_DEFAULT_FIELDS,
        )
    ).settings
    assert origin.has_user_work is True
    # The worker has written and reloaded the config; the UI thread has not
    # accepted the publication, so the generation is still the old one.
    app.app_config = _config("llama_cpp", "made-default", temperature=0.4)
    assert app.console_new_chat_default_generation == 0

    assert _ensure(console) == committed
    assert store.session_settings(origin.id) == committed
    assert store.session_settings(background.id) == blank_console_session_settings(
        config
    ), "an inactive chat is not re-derived"

    app.console_new_chat_default_generation = 1  # the publication lands
    store.switch_session(background.id)
    assert _ensure(console).model == "old-default"


def test_persona_chat_keeps_its_persona_when_it_converges():
    """AC#8: provider defaults change; Persona identity and prompt do not."""
    config = _config("llama_cpp", "new-model")
    _app, console, store = _console(config)
    persona = ConsoleSessionSettings(
        provider="llama_cpp",
        model="old-model",
        system_prompt="You are a literary companion.",
        character_label="Lit Agent",
        persona_memory_mode="read_write",
    )
    session = _pristine(
        store,
        persona,
        assistant_kind="persona",
        assistant_id="persona-1",
        persona_memory_mode="read_write",
    )

    shown = _ensure(console)

    assert shown.model == "new-model"
    assert shown.system_prompt == "You are a literary companion."
    assert shown.character_label == "Lit Agent"
    assert shown.persona_memory_mode == session.persona_memory_mode == "read_write"
    assert (session.assistant_kind, session.assistant_id) == ("persona", "persona-1")


def test_provider_change_posts_the_swap_notice_and_a_model_change_does_not():
    """AC#9: the task-16475 notice fires on a provider change only."""
    app, console, store = _console(_config("llama_cpp", "new-model"))
    _pristine(store, ConsoleSessionSettings(provider="llama_cpp", model="old-model"))
    assert _ensure(console).model == "new-model"
    app.notify.assert_not_called()

    app.app_config = _config("anthropic", "claude-test")
    assert provider_config_key(_ensure(console).provider) == "anthropic"
    app.notify.assert_called_once()
    message = app.notify.call_args.args[0]
    # TASK-33002.5 rewrote the notice to display names; no raw key remains.
    assert "llama.cpp" in message and "Anthropic" in message
    assert "llama_cpp" not in message and "anthropic" not in message
    assert app.notify.call_args.kwargs == {"severity": "warning"}


def test_swap_notice_names_a_cleared_provider_as_not_selected():
    """Final review (Task 5 minor): defaults without a provider read as the
    chip reads them, not "changed llama.cpp -> : ..."."""
    app, console, store = _console(_config("llama_cpp", "new-model"))
    _pristine(store, ConsoleSessionSettings(provider="llama_cpp", model="old-model"))
    _ensure(console)

    app.app_config = {"chat_defaults": {}, "api_settings": {}}
    _ensure(console)

    message = app.notify.call_args.args[0]
    assert message.startswith("Console provider changed llama.cpp -> not selected:")


def test_rebuilds_rederive_a_pristine_chat_at_most_once_per_saved_config(
    monkeypatch,
):
    """AC#10: between two saves, rebuilds derive the defaults at most once."""
    derivations: list[object] = []
    real = session_module.blank_console_session_settings

    def counting(app_config):
        derivations.append(app_config)
        return real(app_config)

    monkeypatch.setattr(session_module, "blank_console_session_settings", counting)
    app, console, store = _console(_config("llama_cpp", "new-model"))
    _pristine(store, ConsoleSessionSettings(provider="llama_cpp", model="old-model"))

    for _ in range(5):
        assert _ensure(console).model == "new-model"
    assert len(derivations) == 1, derivations

    app.app_config = _config("llama_cpp", "newer-model")  # a published save
    for _ in range(5):
        assert _ensure(console).model == "newer-model"
    assert len(derivations) == 2, derivations


def test_convergence_check_adds_no_disk_keyring_or_network_access(monkeypatch):
    """AC#11: the check is pure over the already-resolved config mapping."""
    import keyring

    # chat_screen imports load_settings by name, so the sentinel must replace
    # that binding; patching tldw_chatbook.config.load_settings intercepts
    # nothing (final-review M2).
    import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module

    app, console, store = _console(_config("llama_cpp", "new-model"))
    _pristine(store, ConsoleSessionSettings(provider="llama_cpp", model="old-model"))
    assert _ensure(console).model == "new-model"  # warm every lazy import
    app.app_config = _config("anthropic", "claude-test")

    attempts: list[str] = []

    def forbid(name):
        def _blocked(*_args, **_kwargs):
            attempts.append(name)
            raise AssertionError(f"{name} reached from the Console display path")

        return _blocked

    for target, name in (
        (builtins, "open"),
        (io, "open"),
        (os, "open"),
        (sqlite3, "connect"),
        (socket.socket, "connect"),
        (socket, "create_connection"),
        (keyring, "get_password"),
        (keyring, "get_credential"),
        (chat_screen_module, "load_settings"),
    ):
        monkeypatch.setattr(target, name, forbid(f"{target.__name__}.{name}"))

    shown = _ensure(console)
    monkeypatch.undo()

    assert attempts == []
    assert provider_config_key(shown.provider) == "anthropic"


def test_first_chat_handoff_target_is_not_rederived_until_the_config_changes(
    monkeypatch,
):
    """The Start chatting handoff owns its target for the config it applied.

    Its builder keeps llama.cpp's configured endpoint, which a blank chat
    drops, so re-deriving right away would move the target while the
    handoff's rollback fences still expect its own settings. A later save
    converges it like any untouched chat.
    """
    from tldw_chatbook.config import RuntimeConfigSnapshot
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        ConsoleFirstChatIntent,
        PendingHandoffStore,
    )

    config = _config("llama_cpp", "setup-model")
    app, console, store = _console(config)
    app.pending_handoffs = PendingHandoffStore()
    monkeypatch.setattr(
        session_module,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshot(7, config),
    )
    monkeypatch.setattr(
        session_module,
        "run_if_runtime_config_generation_current",
        lambda _generation, acknowledge: acknowledge(),
    )
    intent = ConsoleFirstChatIntent("first-chat-target", "llama_cpp", "setup-model", 7)
    app.pending_handoffs.stage_reserved_console_first_chat(intent)

    assert console._session.consume_pending_console_first_chat_intent(
        defer_presentation=True
    )
    handed = store.session_settings(intent.session_id)
    assert handed.base_url == LLAMA_URL
    assert _ensure(console) == handed

    app.app_config = _config("llama_cpp", "setup-model")  # a later save
    converged = _ensure(console)
    assert converged == blank_console_session_settings(app.app_config)
    assert converged.base_url is None
