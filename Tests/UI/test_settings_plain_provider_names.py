"""Settings ▸ Providers & Models: a provider name printed by a literal
widget is not escaped first (review of #3057, R1 #13 and #14).

The navigation-conflict summary, the Inspector's provider catalog row and
the "Unavailable for ..." guide row are plain Statics (markup=False), but
their text was built from the markup-escaped name, so a name such as
"[local] box" was shown with a backslash. A Button label is a markup sink
and keeps the escaped name.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Utils.input_validation import escape_markup

pytestmark = pytest.mark.bootstrap_profile

_NAME = "[local] GPU [/] box"
_ID = "custom-ep:gpu-box"


def _screen() -> SettingsScreen:
    screen = SettingsScreen.__new__(SettingsScreen)
    screen.app_instance = SimpleNamespace(
        app_config={
            "custom_endpoints": {
                "gpu-box": {
                    "display_name": _NAME,
                    "family": "llama_cpp",
                    "base_url": "http://192.168.1.5:8080",
                    "models": ["model-a"],
                }
            }
        }
    )
    screen._settings_drafts = {}
    screen._provider_navigation_conflict_target = SimpleNamespace(provider=_ID)
    return screen


def test_the_conflict_summary_prints_the_name_and_the_button_escapes_it():
    screen = _screen()
    assert escape_markup(_NAME) != _NAME

    assert f"configure {_NAME}, or return." in (
        screen._provider_navigation_conflict_copy()
    )
    assert screen._provider_conflict_discard_label() == (
        f"Discard changes and configure {escape_markup(_NAME)}"
    )


def test_the_unavailable_guide_row_prints_the_name():
    assert _screen()._unsupported_model_profile_placeholder(_ID) == (
        f"Unavailable for {_NAME}"
    )


def test_the_catalog_row_prints_names_as_they_are(monkeypatch):
    monkeypatch.setattr(
        SettingsScreen,
        "_provider_display_label",
        lambda _self, provider: f"[{provider}]",
    )

    summary = _screen()._provider_catalog_summary()

    assert "[openai]" in summary
    assert "\\" not in summary
