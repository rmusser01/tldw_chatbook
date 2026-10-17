"""Settings lines that name a provider/model pair beside its readiness word.

Each line reads the one readiness vocabulary (spec 2026-09-26 §5) for the
connection it names, so Settings never shows a pair in words of its own.

``settings_screen`` imports this inside the functions that need it (ADR-097).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen


def readiness_label(screen: SettingsScreen, word: str | None = None) -> str:
    """The Key block's identity row: the form's pair, then its readiness word.

    TASK-33005.7 AC#1: "Provider readiness: <provider> / <model>" named the
    pair with no word. It now ends with the Key check row's word for that
    same connection, which is the Test result's Readiness row while that
    result describes the form.

    Args:
        screen: The Settings screen holding the Providers & Models form.
        word: The Key check row's verdict when the caller has built it, so the
            two rows say one verdict; built here when omitted.

    Returns:
        "Provider readiness: <provider> / <model> — <word>".
    """
    from .providers_models_card import _shown_model, key_check_verdict

    provider = screen._provider_widget_value()
    if not provider:
        return "Provider readiness: needs provider and model"
    name = screen._provider_display_label(provider)
    model = _shown_model(screen) or "not selected"
    word = word or key_check_verdict(screen)
    return f"Provider readiness: {name} / {model} — {word}"


def overview_configuration(screen: SettingsScreen) -> str:
    """The Overview's Configuration row: the saved pair a new chat opens on.

    TASK-33005.15: the pair and its Status are a new Console chat's, from
    saved config only: ``chat_defaults.model`` for its own provider, else
    that provider's table model (TASK-33001.8), read against the shared
    connection evidence (AC#1). An unsaved Providers & Models draft is named
    after them, never shown as the Configuration (AC#4).

    Args:
        screen: The Settings screen.

    Returns:
        "<provider> / <model>; Status: <word>", plus "; Unsaved draft: <provider>
        / <model>" while the draft holds edits.
    """
    from ...Chat.console_session_settings import (
        blank_console_session_settings,
        build_console_settings_readiness,
        readiness_words,
    )
    from ...Chat.provider_test_evidence import shared_connection_evidence

    app_config = screen._provider_readiness_app_config()
    settings = blank_console_session_settings(app_config)
    readiness = build_console_settings_readiness(
        settings,
        app_config=app_config,
        connection_evidence=shared_connection_evidence(lambda: screen.app),
    )
    line = (
        f"{_pair(screen, settings.provider, settings.model)}; "
        f"Status: {readiness_words(readiness)}"
    )
    draft = screen._provider_draft()
    if draft is not None and draft.is_dirty:
        staged = screen._resolve_provider_model_for_settings()
        line += f"; Unsaved draft: {_pair(screen, staged.provider, staged.model)}"
    return line


def _pair(screen: SettingsScreen, provider: object, model: object) -> str:
    name = screen._provider_display_label(str(provider or "")) or "Not selected"
    return f"{name} / {str(model or '').strip() or 'not selected'}"
