"""TASK-25732: the chip and the surface it opens must use one word.

The Console status strip renders "Library · Auto off · Agent blocked"; clicking
it opened a modal whose matching control was labelled "Assistant Library
access". Same permission, two nouns -- and "Assistant" already means something
else in that same status bar ("Assistant: General", the persona), so the reader
has to work out that the agent's permission and the assistant's identity are
unrelated.
"""

from __future__ import annotations


def test_console_library_modal_uses_the_chips_noun() -> None:
    from tldw_chatbook.Widgets.Console import console_library_access_modal as modal

    source = open(modal.__file__).read()
    assert "Agent Library access" in source
    assert "Assistant Library access" not in source


def test_settings_matches_the_console_vocabulary() -> None:
    from tldw_chatbook.UI.Screens import settings_screen

    source = open(settings_screen.__file__).read()
    assert "Agent Library access" in source


def test_old_label_survives_as_a_search_alias() -> None:
    """Renaming must not cost anyone who searches the previous wording."""
    from tldw_chatbook.UI.Screens import settings_search_index

    source = open(settings_search_index.__file__).read()
    assert "Agent Library access" in source
    assert "Assistant Library access" in source
