"""The Playground's keyboard shortcuts must survive its rebuild.

`TTSPlaygroundWidget` bound five of them. Deleting that class took its
BINDINGS with it while the `action_*` methods moved into
`SpeechPlaybackMixin` -- so every method still existed, was still callable,
and no test failed, while Ctrl+G through Ctrl+S quietly stopped doing
anything. The screen went on advertising them in its shortcut line.

An action with no binding and a binding with no action fail in opposite
directions and neither raises, so both halves are asserted.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane

#: (key, action) exactly as the legacy widget bound them.
LEGACY_SHORTCUTS = {
    "ctrl+g": "generate_tts",
    "ctrl+r": "random_text",
    "ctrl+l": "clear_text",
    "ctrl+p": "play_audio",
    "ctrl+s": "stop_audio",
}


def _bindings() -> dict[str, str]:
    return {
        b.key: b.action
        for b in getattr(SpeechPlaygroundPane, "BINDINGS", [])
        if hasattr(b, "key")
    }


@pytest.mark.unit
@pytest.mark.parametrize("key,action", sorted(LEGACY_SHORTCUTS.items()))
def test_each_legacy_shortcut_is_still_bound(key, action):
    bound = _bindings()
    assert key in bound, f"{key} ({action}) is no longer bound"
    assert bound[key] == action, f"{key} now runs {bound[key]!r}, not {action!r}"


@pytest.mark.unit
@pytest.mark.parametrize("action", sorted(set(LEGACY_SHORTCUTS.values())))
def test_every_bound_action_exists(action):
    """A binding naming a missing method fails only when the key is pressed."""
    assert callable(getattr(SpeechPlaygroundPane, f"action_{action}", None)), (
        f"action_{action} does not exist, so its binding is dead"
    )
