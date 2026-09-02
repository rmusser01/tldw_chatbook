"""TASK-25835: don't advertise Enter-to-send with no composer on screen.

Collapsing the composer leaves the footer still offering "Enter send / queue"
while there is nothing to type into and Enter sends nothing. The screen already
solved this exact class for the setup modal --
CONSOLE_WORKBENCH_SHORTCUTS_SETUP_BLOCKED exists because "while the first-run
setup modal locks the composer, advertising 'Enter send' is a lie" -- but the
collapsed case had no equivalent.
"""

from __future__ import annotations


def _labels(shortcuts):
    return {key: label for key, label in shortcuts}


def test_collapsed_variant_drops_the_send_hint() -> None:
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_WORKBENCH_SHORTCUTS,
        CONSOLE_WORKBENCH_SHORTCUTS_COMPOSER_COLLAPSED,
    )

    assert _labels(CONSOLE_WORKBENCH_SHORTCUTS)["Enter"] == "send / queue"
    collapsed = _labels(CONSOLE_WORKBENCH_SHORTCUTS_COMPOSER_COLLAPSED)
    assert collapsed.get("Enter") != "send / queue"


def test_collapsed_variant_names_the_way_back() -> None:
    """The one action that matters while collapsed is restoring the composer."""
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_WORKBENCH_SHORTCUTS_COMPOSER_COLLAPSED,
    )

    labels = _labels(CONSOLE_WORKBENCH_SHORTCUTS_COMPOSER_COLLAPSED)
    assert "Esc" in labels
    assert "composer" in labels["Esc"].lower()


def test_collapsed_variant_keeps_the_rest_of_the_vocabulary() -> None:
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_WORKBENCH_SHORTCUTS,
        CONSOLE_WORKBENCH_SHORTCUTS_COMPOSER_COLLAPSED,
    )

    base = _labels(CONSOLE_WORKBENCH_SHORTCUTS)
    collapsed = _labels(CONSOLE_WORKBENCH_SHORTCUTS_COMPOSER_COLLAPSED)
    for key in ("F6", "F1", "Ctrl+K", "Ctrl+T", "Alt+I", "Ctrl+P"):
        assert collapsed[key] == base[key]
