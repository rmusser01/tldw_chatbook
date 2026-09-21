"""The appearance picker must accept a pre-built emoji list (task-32804.12).

Building the emoji catalog costs ~180 ms on first process use. The Console
appearance control now builds it off the event loop and injects the result
into ``ConsoleAppearancePickerModal`` so the click never freezes the UI. These
tests pin the ``__init__`` contract that makes that possible: an injected list
is used verbatim and the synchronous ``_default_emoji_sequence`` build is *not*
invoked; omitting it falls back to that build so a direct construction (tests,
any future caller) still self-serves.

Pure ``__init__`` only -- no mount, no app -- so it runs without the ADR-126
storage admission binding that mounting would require.
"""

from __future__ import annotations

import tldw_chatbook.Widgets.Console.console_appearance_picker_modal as modal_mod
from tldw_chatbook.Widgets.Console.console_appearance_picker_modal import (
    ConsoleAppearancePickerModal,
)

_FAKE = [
    {"char": "😀", "name": "grinning", "category": "Smileys", "aliases": ["grinning"]},
    {"char": "🐱", "name": "cat", "category": "Animals", "aliases": ["cat"]},
]


def test_injected_emojis_bypass_catalog_build(monkeypatch):
    def _boom():
        raise AssertionError(
            "_default_emoji_sequence must not run when emojis are injected"
        )

    monkeypatch.setattr(modal_mod, "_default_emoji_sequence", _boom)

    modal = ConsoleAppearancePickerModal(
        conversation_id="c1", conversation_title="t", emojis=_FAKE
    )
    assert modal._all_emojis is _FAKE


def test_missing_emojis_falls_back_to_catalog(monkeypatch):
    calls = {"n": 0}
    sentinel = [{"char": "x", "name": "x", "category": "c", "aliases": []}]

    def _seq():
        calls["n"] += 1
        return sentinel

    monkeypatch.setattr(modal_mod, "_default_emoji_sequence", _seq)

    modal = ConsoleAppearancePickerModal(conversation_id="c2")
    assert calls["n"] == 1
    assert modal._all_emojis is sentinel
