from tldw_chatbook.Chat.console_display_state import TurnFileEntry, turn_file_entries
from tldw_chatbook.Workspaces.change_tracking import ChangedFile


def _row(root="/ws", kind="turn", tracking_error=None):
    return {"root": root, "kind": kind, "tracking_error": tracking_error,
            "files_changed": 1, "adds": 3, "dels": 1}


def test_single_root_entries_use_bare_relpaths():
    rows = [_row()]
    changed = {"/ws": [ChangedFile(path="a/b.py", status="M", adds=3, dels=1)]}
    entries = turn_file_entries(rows, changed)
    assert entries == [TurnFileEntry(
        label="a/b.py", path="a/b.py", root="/ws",
        status="M", adds=3, dels=1)]


def test_multi_root_entries_prefix_the_root_name():
    rows = [_row(root="/ws/one"), _row(root="/ws/two")]
    changed = {
        "/ws/one": [ChangedFile(path="x.md", status="A", adds=5, dels=0)],
        "/ws/two": [ChangedFile(path="y.md", status="D", adds=0, dels=7)],
    }
    labels = [e.label for e in turn_file_entries(rows, changed)]
    assert labels == ["one/x.md", "two/y.md"]


def test_tracking_error_rows_yield_no_entries():
    rows = [_row(tracking_error="git failed")]
    assert turn_file_entries(rows, {"/ws": []}) == []
