"""Critique #10 media-row fixes: labelled age, applied-scope line, copy polish.

Covers tasks 32347 (the row age says it is an age), 32350 (a scope line
states the APPLIED filter, never the box's draft) and 32364 (a status word
never prefixes a title; one separator glyph; the import footer names what
Enter does at each step).
"""

from __future__ import annotations

from datetime import datetime, timezone

from tldw_chatbook.Library.library_media_state import media_added_age_copy


def test_the_age_label_says_what_the_age_is():
    now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
    assert media_added_age_copy("2026-09-11T11:50:00+00:00", now=now) == "added 10m ago"
    assert media_added_age_copy("2026-09-11T11:59:40+00:00", now=now) == "added just now"
    assert media_added_age_copy("", now=now) == ""
