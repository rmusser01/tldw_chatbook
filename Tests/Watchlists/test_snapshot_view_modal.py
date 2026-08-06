"""Unit tests for `snapshot_view_modal.py`'s header formatting.

Batch-4 review, I2: task-2308's own scope ("every Watchlists table
timestamp") missed this modal's `Captured <created_at>` line -- it opens
from the Full page/Previous snapshot buttons in the same `#content-actions`
row this batch's reader byline already humanizes.
"""

import pytest

pytestmark = pytest.mark.unit

from tldw_chatbook.UI.Watchlists_Modules.humane_time import humane_timestamp
from tldw_chatbook.UI.Watchlists_Modules.snapshot_view_modal import _snapshot_header


def test_snapshot_header_humanizes_created_at():
    header = _snapshot_header("https://example.test/page", "2026-08-04T18:15:22.123456+00:00")
    plain = header.plain
    assert "2026-08-04T18:15:22.123456+00:00" not in plain, (
        "the raw ISO timestamp must not reach the header verbatim"
    )
    assert humane_timestamp("2026-08-04T18:15:22.123456+00:00") in plain


def test_snapshot_header_degrades_honestly_with_no_created_at():
    header = _snapshot_header("https://example.test/page", None)
    assert "Captured at an unknown time" in header.plain


def test_snapshot_header_still_shows_the_url():
    header = _snapshot_header("https://example.test/page", "2026-08-04T18:15:22+00:00")
    assert "https://example.test/page" in header.plain
