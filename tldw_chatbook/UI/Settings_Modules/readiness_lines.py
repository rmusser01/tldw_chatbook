"""Settings lines that name a provider/model pair beside its readiness word.

Each line reads the one readiness vocabulary (spec 2026-09-26 §5) for the
connection it names, so Settings never shows a pair in words of its own.

``settings_screen`` imports this inside the functions that need it (ADR-097).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen


def readiness_label(screen: SettingsScreen) -> str:
    """The Key block's identity row: the form's pair, then its readiness word.

    TASK-33005.7 AC#1: "Provider readiness: <provider> / <model>" named the
    pair with no word. It now ends with the Key check row's word for that
    same connection, which is the Test result's Readiness row while that
    result describes the form.

    Args:
        screen: The Settings screen holding the Providers & Models form.

    Returns:
        "Provider readiness: <provider> / <model> — <word>".
    """
    from .providers_models_card import _shown_model, key_check_verdict

    provider = screen._provider_widget_value()
    if not provider:
        return "Provider readiness: needs provider and model"
    name = screen._provider_display_label(provider)
    model = _shown_model(screen) or "not selected"
    return f"Provider readiness: {name} / {model} — {key_check_verdict(screen)}"
