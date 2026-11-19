"""The old UI import path of the display-text sanitizer still works (TASK-36944.6 AC#3)."""

from tldw_chatbook.UI import character_display_text as old
from tldw_chatbook.Utils import character_display_text as new


def test_old_ui_import_path_resolves_to_the_utils_functions() -> None:
    assert old.__all__ == new.__all__
    for name in new.__all__:
        assert getattr(old, name) is getattr(new, name)
