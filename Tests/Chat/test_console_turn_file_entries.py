from tldw_chatbook.Chat.console_display_state import TurnFileEntry, turn_file_entries
from tldw_chatbook.Workspaces.change_tracking import ChangedFile


def _row(root="/ws", kind="turn", tracking_error=None):
    return {"root": root, "kind": kind, "tracking_error": tracking_error,
            "files_changed": 1, "adds": 3, "dels": 1}


def test_single_root_entries_use_bare_relpaths():
    row = _row()
    files = [ChangedFile(path="a/b.py", status="M", adds=3, dels=1)]
    paired = turn_file_entries([(row, files)])
    assert paired == [(TurnFileEntry(
        label="a/b.py", path="a/b.py", root="/ws",
        status="M", adds=3, dels=1), row)]


def test_multi_root_entries_prefix_the_root_name():
    row_one = _row(root="/ws/one")
    row_two = _row(root="/ws/two")
    row_files = [
        (row_one, [ChangedFile(path="x.md", status="A", adds=5, dels=0)]),
        (row_two, [ChangedFile(path="y.md", status="D", adds=0, dels=7)]),
    ]
    labels = [entry.label for entry, _row in turn_file_entries(row_files)]
    assert labels == ["one/x.md", "two/y.md"]


def test_tracking_error_rows_yield_no_entries():
    row = _row(tracking_error="git failed")
    assert turn_file_entries([(row, [])]) == []


def test_two_windows_on_same_root_pair_entries_to_their_own_row():
    """PR3a-1 Task 6c: a turn window and its post-turn (sub-agent survivor)
    window can both cover the SAME root under the SAME run_id. The old
    root-keyed dict approach collapsed both windows' file lists into one
    dict slot -- the later window silently overwrote the earlier one, so
    every file appeared twice (all attributed to the later window) and the
    earlier window's files vanished entirely. Pairing by row position
    instead of by root keeps both windows' files distinct and correctly
    attributed, even though `root` is identical for both rows.
    """
    turn_row = _row(root="/ws", kind="turn")
    post_turn_row = _row(root="/ws", kind="subagent_post_turn")
    row_files = [
        (turn_row, [ChangedFile(path="turn_file.txt", status="M", adds=1, dels=1)]),
        (post_turn_row, [ChangedFile(path="post_turn_file.txt", status="A", adds=2, dels=0)]),
    ]
    paired = turn_file_entries(row_files)
    assert [entry.label for entry, _row in paired] == [
        "turn_file.txt",
        "post_turn_file.txt",
    ]
    # No root prefix: both rows share ONE root, so this is not "multi-root".
    entry_turn, row_for_turn = paired[0]
    entry_post_turn, row_for_post_turn = paired[1]
    assert row_for_turn is turn_row
    assert row_for_post_turn is post_turn_row
