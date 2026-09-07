"""TASK-378: a full file path entered in the picker path bar must land ON the file.

Previously ``DirectoryNavigation`` navigated to the typed file's parent directory
and left the highlight on ``..`` (index 0), so a keyboard user pasting a known
path never selected the file. ``show_and_highlight`` now reveals and highlights it.
"""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Third_Party.textual_fspicker.parts.directory_navigation import (
    DirectoryEntry,
    DirectoryNavigation,
)


async def _highlighted_entry(nav: DirectoryNavigation, pilot, *, want: str):
    """Pump the event loop until the highlighted option is ``want`` (or give up)."""
    for _ in range(40):
        await pilot.pause(0.05)
        index = nav.highlighted
        if index is None:
            continue
        option = nav.get_option_at_index(index)
        if isinstance(option, DirectoryEntry) and option.location.name == want:
            return option
    return None if nav.highlighted is None else nav.get_option_at_index(nav.highlighted)


@pytest.mark.asyncio
async def test_show_and_highlight_lands_on_the_file_in_another_directory(tmp_path):
    """A file path in a different directory reloads that dir and highlights it."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "unrelated.txt").write_text("x")
    (tmp_path / "aaa.txt").write_text("a")
    target = tmp_path / "zzz-target.png"
    target.write_text("z")

    class _App(App):
        def compose(self) -> ComposeResult:
            # Start in the subdirectory; the target lives one level up, so a typed
            # absolute path must reload the parent dir AND highlight the file.
            yield DirectoryNavigation(str(sub))

    app = _App()
    async with app.run_test() as pilot:
        nav = app.query_one(DirectoryNavigation)
        await pilot.pause()
        await pilot.pause()

        nav.show_and_highlight(target)

        option = await _highlighted_entry(nav, pilot, want="zzz-target.png")
        assert option is not None, "nothing highlighted"
        assert isinstance(option, DirectoryEntry)
        assert option.location.name == "zzz-target.png"
        # And the navigation followed the file into its parent directory.
        assert nav.location == tmp_path


@pytest.mark.asyncio
async def test_show_and_highlight_survives_an_in_flight_load(tmp_path):
    """A highlight requested while the directory is still loading is not dropped.

    Qodo #816: the same-directory fast path can run before the async load
    populates the list, so a miss must KEEP the pending target for the post-load
    repopulate rather than clearing it. Reproduced deterministically by emptying
    the option list (the in-flight state) before the same-directory request.
    """
    (tmp_path / "aaa.txt").write_text("a")
    target = tmp_path / "zzz-target.png"
    target.write_text("z")

    class _App(App):
        def compose(self) -> ComposeResult:
            yield DirectoryNavigation(str(tmp_path))

    app = _App()
    async with app.run_test() as pilot:
        nav = app.query_one(DirectoryNavigation)
        await pilot.pause()
        await pilot.pause()

        # Simulate the list being empty mid-load, then request the highlight for
        # a file in the (already-current) directory -> hits the same-dir fast path
        # with no options to match yet.
        nav.clear_options()
        nav.show_and_highlight(target)

        # The request must survive the empty-list miss (the bug dropped it here).
        assert nav._pending_highlight is not None

        # When the load completes and the list repopulates, the file is revealed.
        nav._repopulate_display()
        assert nav.highlighted is not None
        option = nav.get_option_at_index(nav.highlighted)
        assert isinstance(option, DirectoryEntry)
        assert option.location.name == "zzz-target.png"
