"""Regressions for usable partial listings and bounded filesystem/UI work."""

import asyncio
import os
import threading
from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Select

from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, Filters
from tldw_chatbook.Third_Party.textual_fspicker.parts.directory_navigation import (
    DirectoryNavigation,
)
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen


async def wait_until(predicate, seconds=4):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.01)


def names(nav):
    return [
        option.location.name for option in nav.options if option.location.name != ".."
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("picker", [FileOpen, EnhancedFileOpen])
async def test_partial_listing_usable_before_enumeration_finishes(
    tmp_path, monkeypatch, picker
):
    """Publishing only after EOF loses the first entry behind a stalled disk."""
    tmp_path = tmp_path / "listing"
    tmp_path.mkdir()
    for index in range(80):
        (tmp_path / f"item-{index:03}.txt").touch()
    real_scandir, real_iterdir = os.scandir, Path.iterdir
    release = threading.Event()
    blocked = threading.Event()

    def gated(values):
        for index, entry in enumerate(values):
            if index == 65:
                blocked.set()
                release.wait(5)
            yield entry

    class Scan:
        def __enter__(self):
            self.scan = real_scandir(tmp_path)
            return gated(self.scan)

        def __exit__(self, *args):
            self.scan.close()

    def scandir(path):
        return Scan() if str(path) == str(tmp_path) else real_scandir(path)

    def iterdir(path):
        values = real_iterdir(path)
        return gated(values) if path == tmp_path else values

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    host = App()
    try:
        async with host.run_test(size=(100, 36)):
            await host.push_screen(picker(location=tmp_path))
            nav = host.screen.query_one(DirectoryNavigation)
            await wait_until(blocked.is_set)
            await asyncio.sleep(0.15)
            assert names(nav), (
                "directory is stalled but its discovered files must be usable"
            )
            assert "Scanning" in str(
                host.screen.query_one("#listing-progress").render()
            )
            nav.highlighted = 1
            assert nav.highlighted_option.location.name.startswith("item-")
            selected_path = nav.highlighted_option.location
            host.screen.query_one("#listing-sort", Select).value = "name"
            await wait_until(
                lambda: nav.sort_key == "name" and not nav._projection_running
            )
            assert names(nav) == sorted(names(nav))
            assert nav.highlighted_option.location == selected_path
            release.set()
            await wait_until(lambda: len(names(nav)) == 80)
            await wait_until(
                lambda: (
                    "Loaded" in str(host.screen.query_one("#listing-progress").render())
                )
            )
            assert names(nav) == [f"item-{index:03}.txt" for index in range(80)]
    finally:
        release.set()


@pytest.mark.asyncio
async def test_metadata_is_off_loop_and_only_near_viewport(tmp_path, monkeypatch):
    """Eager row construction/stat on the UI thread defeats lazy loading."""
    tmp_path = tmp_path / "listing"
    tmp_path.mkdir()
    for index in range(600):
        (tmp_path / f"entry-{index:04}.txt").touch()
    real_stat = Path.stat
    accesses = []
    main_thread = threading.get_ident()

    def stat(path, *args, **kwargs):
        if path.parent == tmp_path and path.name.startswith("entry-"):
            accesses.append((path, threading.get_ident()))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    host = App()
    async with host.run_test(size=(100, 36)):
        await host.push_screen(EnhancedFileOpen(location=tmp_path))
        nav = host.screen.query_one(DirectoryNavigation)
        await wait_until(lambda: len(names(nav)) == 600)
        await asyncio.sleep(0.2)
        assert accesses
        assert not any(thread == main_thread for _, thread in accesses)
        assert len({path for path, _ in accesses}) < 100
        nav.action_last()
        await asyncio.sleep(0.2)
        assert len({path for path, _ in accesses}) < 200


@pytest.mark.asyncio
@pytest.mark.parametrize("picker", [FileOpen, EnhancedFileOpen])
async def test_sort_controls_preserve_highlight_and_file_filter(tmp_path, picker):
    """Sorting must reorder rows without selecting another file or clearing filters."""
    tmp_path = tmp_path / "listing"
    tmp_path.mkdir()
    for index, name in enumerate(("z.txt", "a.txt", "m.txt")):
        path = tmp_path / name
        path.write_bytes(b"x" * (index + 1))
        os.utime(path, (100 + index, 300 - index))
    (tmp_path / "excluded.bin").touch()
    filters = Filters(("Text files", lambda path: path.suffix == ".txt"))
    host = App()
    async with host.run_test(size=(100, 36)):
        await host.push_screen(picker(location=tmp_path, filters=filters))
        nav = host.screen.query_one(DirectoryNavigation)
        await wait_until(lambda: len(names(nav)) == 3)
        nav.highlighted = next(
            i for i, option in enumerate(nav.options) if option.location.name == "m.txt"
        )
        controls = host.screen.query("#listing-sort")
        assert controls, "the user must be able to choose a sorting mode"
        sort = host.screen.query_one("#listing-sort", Select)
        sort.value = "name"
        await wait_until(lambda: names(nav) == ["a.txt", "m.txt", "z.txt"])
        assert nav.highlighted_option.location.name == "m.txt"
        sort.value = "accessed"
        await wait_until(lambda: names(nav) == ["z.txt", "a.txt", "m.txt"])
        host.screen.query_one("#listing-direction", Select).value = "descending"
        await wait_until(lambda: names(nav) == ["m.txt", "a.txt", "z.txt"])
        assert nav.highlighted_option.location.name == "m.txt"
        sort.value = "modified"
        await wait_until(lambda: names(nav) == ["z.txt", "a.txt", "m.txt"])
        sort.value = "size"
        await wait_until(lambda: names(nav) == ["m.txt", "a.txt", "z.txt"])
        assert "excluded.bin" not in names(nav)


@pytest.mark.asyncio
async def test_navigation_rejects_late_scan_and_metadata_results(tmp_path, monkeypatch):
    """An old metadata read must not repaint a replacement directory."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "old.txt").touch()
    (second / "new.txt").touch()
    started, release = threading.Event(), threading.Event()
    real_stat = Path.stat

    def slow_stat(path, *args, **kwargs):
        if (
            path == first / "old.txt"
            and threading.current_thread() is not threading.main_thread()
        ):
            started.set()
            release.wait(3)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", slow_stat)
    host = App()
    try:
        async with host.run_test(size=(100, 36)):
            await host.push_screen(EnhancedFileOpen(location=first))
            nav = host.screen.query_one(DirectoryNavigation)
            await wait_until(started.is_set)
            nav.location = second
            await wait_until(lambda: names(nav) == ["new.txt"])
            release.set()
            await asyncio.sleep(0.2)
            assert names(nav) == ["new.txt"]
            assert nav.location == second
    finally:
        release.set()


def test_timestamp_sort_missing_birth_time_is_last_both_directions(tmp_path):
    """POSIX ctime is never substituted for unavailable creation time."""
    from types import SimpleNamespace

    from tldw_chatbook.Third_Party.textual_fspicker.parts.progressive_directory_navigation import (
        FileRecord,
        project_records,
    )

    records = [
        FileRecord(
            tmp_path / "unknown",
            False,
            metadata=SimpleNamespace(st_ctime=999),
            metadata_loaded=True,
        ),
        FileRecord(
            tmp_path / "new",
            False,
            metadata=SimpleNamespace(st_birthtime=200),
            metadata_loaded=True,
        ),
        FileRecord(
            tmp_path / "old",
            False,
            metadata=SimpleNamespace(st_birthtime=100),
            metadata_loaded=True,
        ),
    ]
    for descending, expected in [
        (False, ["old", "new", "unknown"]),
        (True, ["new", "old", "unknown"]),
    ]:
        result, _ = project_records(
            records,
            show_hidden=True,
            query="",
            file_filter=None,
            sort_key="created",
            descending=descending,
            cancelled=threading.Event(),
        )
        assert [record.location.name for record in result] == expected


@pytest.mark.asyncio
async def test_navigation_does_not_wait_for_old_metadata_sort(tmp_path, monkeypatch):
    """Cancelled async projection must release the UI before its stat returns."""
    from tldw_chatbook.Third_Party.textual_fspicker.parts import (
        progressive_directory_navigation as listing,
    )

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for index in range(200):
        (first / f"old-{index}.txt").touch()
    (second / "new.txt").touch()
    started, release = threading.Event(), threading.Event()
    real_read = listing.read_metadata

    def slow_read(record):
        if record.location.parent == first and not record.metadata_loaded:
            started.set()
            release.wait(3)
        return real_read(record)

    host = App()
    try:
        async with host.run_test(size=(100, 36)):
            await host.push_screen(EnhancedFileOpen(location=first))
            nav = host.screen.query_one(DirectoryNavigation)
            await wait_until(lambda: len(names(nav)) == 200)
            await asyncio.sleep(0.1)
            monkeypatch.setattr(listing, "read_metadata", slow_read)
            nav.sort_key = "size"
            await wait_until(started.is_set)
            nav.location = second
            await wait_until(lambda: names(nav) == ["new.txt"], seconds=0.6)
            release.set()
            await asyncio.sleep(0.1)
            assert names(nav) == ["new.txt"]
    finally:
        release.set()


@pytest.mark.asyncio
async def test_click_from_old_painted_row_cannot_select_replacement(tmp_path):
    """A valid numeric index is insufficient after the list has changed."""
    from textual import events

    folder = tmp_path / "listing"
    folder.mkdir()
    (folder / "a.txt").touch()
    (folder / "z.txt").touch()
    host = App()
    async with host.run_test(size=(100, 36)):
        await host.push_screen(EnhancedFileOpen(location=folder))
        nav = host.screen.query_one(DirectoryNavigation)
        await wait_until(lambda: len(names(nav)) == 2)
        nav.sort_key = "name"
        await wait_until(lambda: names(nav) == ["a.txt", "z.txt"])
        nav.highlighted = 1
        strips = nav._get_option_render(
            nav.options[1], nav.get_visual_style("option-list--option")
        )
        old_style = next(
            segment.style
            for strip in strips
            for segment in strip
            if segment.style and "option" in segment.style.meta
        )
        nav.sort_descending = True
        await wait_until(lambda: names(nav) == ["z.txt", "a.txt"])
        assert nav.highlighted_option.location.name == "a.txt"
        event = events.Click(
            nav, 0, 0, 0, 0, 1, False, False, False, style=old_style, chain=1
        )
        nav._on_click(event)
        assert nav.highlighted_option.location.name == "a.txt"


@pytest.mark.asyncio
async def test_queued_selection_cannot_undo_newer_navigation(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    child = first / "child"
    child.mkdir()
    host = App()
    async with host.run_test(size=(100, 36)):
        await host.push_screen(FileOpen(location=first))
        nav = host.screen.query_one(DirectoryNavigation)
        await wait_until(lambda: "child" in names(nav))
        nav.highlighted = next(
            i for i, option in enumerate(nav.options) if option.location == child
        )
        nav.action_select()
        nav.location = second
        await asyncio.sleep(0.2)
        assert nav.location == second


@pytest.mark.asyncio
async def test_first_show_files_read_cannot_share_two_scan_queues(
    tmp_path, monkeypatch
):
    """A lazy reactive watcher can reenter _load while arguments are evaluated."""
    host = App()
    async with host.run_test(size=(100, 36)):
        await host.push_screen(EnhancedFileOpen(location=tmp_path))
        nav = host.screen.query_one(DirectoryNavigation)
        await wait_until(lambda: nav._scan_finished and not nav._projection_running)
        scans = []
        monkeypatch.setattr(
            nav,
            "_scan",
            lambda location, show_files, queue, cancel: scans.append((queue, cancel)),
        )
        # Recreate the descriptor's first-read state, which native mount can
        # reach before show_files has initialized (run_test usually reads it).
        del nav.__dict__["_reactive_show_files"]
        nav._load()
        assert len(scans) == 2, "the first reactive read must exercise reentry"
        assert scans[0][0] is not scans[1][0], (
            "superseded scans must not share publication queues"
        )
        assert scans[0][1].is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("picker", [FileOpen, EnhancedFileOpen])
@pytest.mark.parametrize("key", [None, "enter", "end"])
async def test_sort_publication_preserves_highlight(tmp_path, picker, key):
    """Sort changes and activation cannot select a temporary fallback row."""
    tmp_path = tmp_path / "listing"
    tmp_path.mkdir()
    for index in range(2000):
        (tmp_path / f"file-{index:04}.txt").touch()
    host = App()
    async with host.run_test(size=(100, 36)) as pilot:
        await host.push_screen(picker(location=tmp_path))
        nav = host.screen.query_one(DirectoryNavigation)
        await wait_until(lambda: nav._scan_finished and not nav._projection_running)
        nav.sort_key = "name"
        await wait_until(lambda: not nav._projection_running)
        nav.highlighted = 1700
        original = nav.highlighted_option.location
        nav.focus()
        await pilot.pause()
        nav.sort_descending = True
        # Catch a real yielded publication batch, before the target returns.
        async with asyncio.timeout(4):
            while not 0 < nav.option_count < 1000:
                await asyncio.sleep(0)
        assert nav.highlighted_option is None
        if key:
            await pilot.press(key)
            await pilot.pause()
            if key == "end":
                assert nav.highlighted_option.location != original
                original = nav.highlighted_option.location
        else:
            nav.action_select()
        assert nav.location == tmp_path
        nav.sort_key = "discovery"
        await wait_until(lambda: not nav._projection_running)
        assert nav.option_count == 2001
        assert nav.highlighted_option.location == original


@pytest.mark.asyncio
@pytest.mark.parametrize("picker", [FileOpen, EnhancedFileOpen])
async def test_enter_during_parent_only_scan_does_not_leave_folder(
    tmp_path, monkeypatch, picker
):
    folder = tmp_path / "listing"
    folder.mkdir()
    (folder / "first.txt").touch()
    release, blocked = threading.Event(), threading.Event()
    real_scandir = os.scandir

    def scan(path):
        if Path(path) == folder:
            blocked.set()
            release.wait(5)
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scan)
    host = App()
    try:
        async with host.run_test() as pilot:
            await host.push_screen(picker(location=folder))
            nav = host.screen.query_one(DirectoryNavigation)
            await wait_until(blocked.is_set)
            nav.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert nav.location == folder
            assert nav.highlighted is None
            release.set()
            await wait_until(lambda: nav._scan_finished and not nav._projection_running)
            assert nav.highlighted_option.location == folder / "first.txt"
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("picker", [FileOpen, EnhancedFileOpen])
async def test_scan_permission_error_reaches_dialog_alert(
    tmp_path, monkeypatch, picker
):
    folder = tmp_path / "denied"
    folder.mkdir()
    real_scandir = os.scandir

    def scan(path):
        if Path(path) == folder:
            raise PermissionError("denied")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scan)
    dialog = picker(location=folder)
    errors = []
    original = dialog._set_error

    def capture(message=""):
        errors.append(message)
        original(message)

    monkeypatch.setattr(dialog, "_set_error", capture)
    host = App()
    async with host.run_test() as pilot:
        await host.push_screen(dialog)
        nav = dialog.query_one(DirectoryNavigation)
        await wait_until(lambda: nav._scan_finished)
        await pilot.pause()
        assert "Permission error" in errors
        assert "Stopped" in nav.listing_status


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_probe", ["is_dir", "is_file", "is_symlink"])
async def test_permission_denied_entry_probe_keeps_file(
    tmp_path, monkeypatch, failed_probe
):
    folder = tmp_path / "listing"
    folder.mkdir()
    (folder / "restricted.txt").touch()
    real_scandir = os.scandir

    class Entry:
        name = "restricted.txt"

        def __getattr__(self, method):
            def probe():
                if method == failed_probe:
                    raise PermissionError("denied")
                return method == "is_file"

            return probe

    class Scan:
        def __enter__(self):
            return iter([Entry()])

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        os,
        "scandir",
        lambda path: Scan() if Path(path) == folder else real_scandir(path),
    )
    host = App()
    async with host.run_test():
        await host.push_screen(FileOpen(location=folder))
        nav = host.screen.query_one(DirectoryNavigation)
        await wait_until(lambda: nav._scan_finished and not nav._projection_running)
        assert names(nav) == ["restricted.txt"]


@pytest.mark.asyncio
async def test_order_preserving_metadata_sort_hydrates_existing_options(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Third_Party.textual_fspicker.parts import (
        DirectoryNavigation as Progressive,
    )

    folder = tmp_path / "listing"
    folder.mkdir()
    for index in range(80):
        (folder / f"file-{index:03}.txt").write_text("x")
    # Isolate metadata produced by sorting from the independent viewport worker.
    monkeypatch.setattr(Progressive, "_hydrate_visible", lambda *args: None)
    host = App()
    async with host.run_test():
        await host.push_screen(FileOpen(location=folder))
        nav = host.screen.query_one(DirectoryNavigation)
        await wait_until(lambda: nav._scan_finished and not nav._projection_running)
        nav.sort_key = "name"
        await wait_until(lambda: not nav._projection_running)
        before = names(nav)
        nav.sort_key = "size"
        await wait_until(lambda: not nav._projection_running)
        assert names(nav) == before
        assert all(option.record.metadata_loaded for option in nav.options[1:])


@pytest.mark.parametrize(
    "suffix",
    [
        "folder",
        "../elsewhere",
        "semi;colon",
        "pipe|name",
        "$(name)",
        "`name`",
        "\udcff",
    ],
)
def test_browsing_path_validation_preserves_legal_spelling(tmp_path, suffix):
    from tldw_chatbook.Utils.path_validation import validate_browsing_path

    candidate = tmp_path / suffix
    assert validate_browsing_path(candidate) == candidate
    assert validate_browsing_path(str(candidate)) == candidate


@pytest.mark.parametrize("value", [None, 123, [], "relative", "/bad\x00name"])
def test_browsing_path_validation_rejects_malformed_values(value):
    from tldw_chatbook.Utils.path_validation import validate_browsing_path

    with pytest.raises(ValueError):
        validate_browsing_path(value)


@pytest.mark.asyncio
async def test_invalid_sort_events_leave_controls_and_navigation_unchanged(tmp_path):
    host = App()
    async with host.run_test():
        dialog = FileOpen(location=tmp_path)
        await host.push_screen(dialog)
        nav = dialog.query_one(DirectoryNavigation)
        sort = dialog.query_one("#listing-sort", Select)
        direction = dialog.query_one("#listing-direction", Select)
        for invalid in [None, 123, [], Select.NULL, "unknown"]:
            for control in (sort, direction):
                dialog._change_listing_sort(Select.Changed(control, invalid))
                assert nav.sort_key == "discovery"
                assert nav.sort_descending is False
                assert direction.disabled is True


@pytest.mark.parametrize(
    "key", ["discovery", "name", "modified", "accessed", "created", "size"]
)
def test_sort_key_validator_accepts_supported_options(key):
    from tldw_chatbook.Utils.input_validation import validate_file_picker_sort_key

    assert validate_file_picker_sort_key(key) == key


@pytest.mark.parametrize("direction", ["ascending", "descending"])
def test_sort_direction_validator_accepts_supported_options(direction):
    from tldw_chatbook.Utils.input_validation import validate_file_picker_sort_direction

    assert validate_file_picker_sort_direction(direction) == direction


@pytest.mark.asyncio
async def test_invalid_scan_path_is_rejected_before_filesystem_access(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Third_Party.textual_fspicker.parts import (
        DirectoryNavigation as Progressive,
    )

    bad_path = tmp_path / "bad\x00name"
    real_scandir = os.scandir
    seen = []

    def scan(path):
        seen.append(path)
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scan)

    class Host(App):
        def compose(self):
            yield Progressive(bad_path)

    host = Host()
    async with host.run_test():
        nav = host.query_one(DirectoryNavigation)
        await wait_until(lambda: nav._scan_finished)
        assert bad_path not in seen
        assert "ValueError" in nav.listing_status
