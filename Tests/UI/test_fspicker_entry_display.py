"""task-2859 item 6: the vendored file/folder pickers used to show raw
byte counts ("30624") and second-precision timestamps
("2026-08-06 14:32:07") -- both a "TODO: format well for a file browser"
left unfinished in the upstream fork. Covers the bounded fix: human-
readable sizes and minute-precision timestamps in
``DirectoryEntry._size``/``_mtime``.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from tldw_chatbook.Third_Party.textual_fspicker.parts.directory_navigation import (
    DirectoryEntry,
    _human_readable_size,
)


# --- _human_readable_size -----------------------------------------------------


def test_human_readable_size_bytes():
    assert _human_readable_size(512) == "512 B"


def test_human_readable_size_kilobytes():
    assert _human_readable_size(30624) == "29.9 KB"


def test_human_readable_size_megabytes():
    assert _human_readable_size(4 * 1024 * 1024) == "4.0 MB"


def test_human_readable_size_gigabytes():
    assert _human_readable_size(2 * 1024 * 1024 * 1024) == "2.0 GB"


# --- DirectoryEntry._size / ._mtime -------------------------------------------


def test_directory_entry_size_is_human_readable_not_raw_bytes(tmp_path: Path):
    target = tmp_path / "report.pdf"
    target.write_bytes(b"x" * 30624)

    size_text = DirectoryEntry._size(target)

    assert size_text == "29.9 KB"
    # The old raw-byte-count behavior must be gone.
    assert size_text != "30624"


def test_directory_entry_mtime_is_minute_precision_not_second_precision(
    tmp_path: Path,
):
    target = tmp_path / "report.pdf"
    target.write_bytes(b"x")
    # Pin the mtime to a known moment with non-zero seconds, so a
    # second-precision regression would be visible in the assertion.
    stamp = datetime(2026, 8, 6, 14, 32, 7).timestamp()
    os.utime(target, (stamp, stamp))

    mtime_text = DirectoryEntry._mtime(target)

    assert mtime_text == "2026-08-06 14:32"
    # No seconds component, and no ISO "T" separator (also dropped).
    assert ":07" not in mtime_text
    assert "T" not in mtime_text
