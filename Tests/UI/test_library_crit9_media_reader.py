"""Media reader fixes from the critique-9 wave (tasks 32237, 32234, 32222, 32224).

Four independent reader defects the 2026-09-10 dual-agent live review found
at dev 02374bf66a, each pinned here against the surface the user actually
sees (painted text, the footer chip's own string, the stored row):

- task-32237: the More strip's danger action clipped "Move to trash" to
  "Move to" because ``.library-media-action-danger``'s 2-cell separation is
  taken out of the button's own 16-cell grid column.
- task-32234: ``_is_markdown_media`` ran a media-type allowlist BEFORE the
  content sniff, so a ``document`` starting ``# Roadmap sync`` painted its
  hashes literally under a note claiming there was no Markdown to render.
- task-32222: the Escape footer chip and two guide sentences named three
  different targets; the chip is pinned and the guide was not, so the guide
  now quotes the chip and this file asserts the quote.
- task-32224: Undo of a bulk delete does re-date the restored item -- the
  guide claimed otherwise. Pinned here so the corrected guide sentence and
  the code cannot drift apart again.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_media_viewer_state import _is_markdown_media
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_media_viewer import RENDERED_VIEW_NOTE

from Tests.UI.test_library_media_side_by_side import (
    _build_media_test_app,
    _open_media_list,
)
from Tests.UI.test_library_media_render_fixes import (
    _four_action_host,
    _open_first_reader_row,
    _open_reader_more,
    _painted,
)
from Tests.UI.test_library_media_reader_flow import _escape_fake
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _seed_conversations,
    _two_conversations,
)


#: The guide page these tests keep honest.
_GUIDE = (
    Path(__file__).resolve().parents[2]
    / "Docs"
    / "User_Guide"
    / "library"
    / "media-and-conversations.md"
)

_MORE_ACTION_LABELS = (
    "Edit metadata",
    "Open original",
    "Open manager",
    "Move to trash",
)


@pytest.mark.asyncio
async def test_the_more_strip_paints_every_label_at_the_narrowest_stage():
    """task-32237 AC#1: the 60x24 stage, where the strip wraps to two rows.

    The two existing pins cover 235x52 and the narrow reader width; this is
    the third stage the AC names, and the one where the grid reflows -- a
    column that is one cell short clips the wrapped row just as readily as
    the single-row one.
    """
    host = _four_action_host()
    async with host.run_test(size=(60, 24)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        actions = await _open_reader_more(screen, pilot)
        painted = _painted(host, actions.region)
        for label in _MORE_ACTION_LABELS:
            assert label in painted, painted
