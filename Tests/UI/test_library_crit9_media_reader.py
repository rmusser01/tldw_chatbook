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
from types import SimpleNamespace

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
    _MORE_ACTION_LABELS,
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


def _markdown_document_host() -> LibraryProductionCSSHarness:
    """Two `document`-typed items; the newest one's text is real Markdown.

    ``document`` is exactly the type the old allowlist excluded, and the
    content is exactly what the sniff recognises -- the pair the critique
    reproduced live. The second item keeps the list two rows deep (the
    shared ``_open_media_list`` waits for ``#library-media-row-1``) and is
    plain prose, so the same type still resolves both ways by content.
    """
    app = _build_media_test_app()
    items = [
        {
            "id": "media-1",
            "title": "Roadmap Sync Notes",
            "type": "document",
            "last_modified": "2026-07-06T10:00:00Z",
            "author": "Jordan Lee",
            "keywords": ["roadmap"],
            "content": "# Roadmap sync\n\nBody.\n",
            "version": 1,
        },
        {
            "id": "media-2",
            "title": "Plain Handover Doc",
            "type": "document",
            "last_modified": "2026-07-06T08:00:00Z",
            "author": "Morgan Lee",
            "keywords": ["handover"],
            "content": "Plain prose with no markers.\n",
            "version": 1,
        },
    ]
    _seed_conversations(app, _two_conversations(), media=items)
    return LibraryProductionCSSHarness(app)


def test_the_rendered_view_rule_reads_content_not_the_media_type():
    """task-32234: the sniff is the whole rule, for every type."""
    assert _is_markdown_media("document", "# Roadmap sync\n\nBody.\n") is True
    assert _is_markdown_media("document", "Plain prose with no markers.\n") is False
    # The types the old allowlist named keep behaving the same way -- the
    # gate that changed was in front of the sniff, not inside it.
    assert _is_markdown_media("plaintext", "# Heading\n") is True
    assert _is_markdown_media("plaintext", "Just a line.\n") is False


@pytest.mark.asyncio
async def test_a_document_with_real_markdown_renders_and_drops_the_false_note():
    """task-32234 AC#1/#2: a `document` with Markdown renders it.

    The note the Reader used to paint over this item -- "No Markdown
    formatting to render — showing the stored text" -- was a sentence the
    user could disprove by reading the hashes underneath it.
    """
    host = _markdown_document_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        assert screen.query("#library-media-viewer-content-markdown"), list(
            screen.query_one("#library-media-viewer-content").children
        )
        painted = _painted(host, screen.query_one("#library-media-viewer-content").region)
        assert "Roadmap sync" in painted and "# Roadmap" not in painted, painted
        assert RENDERED_VIEW_NOTE not in _painted(
            host, screen.query_one("#library-media-viewer").region
        )


def test_the_escape_chip_and_the_guide_quote_the_same_words():
    """task-32222 AC#1: one chip text, one guide sentence, one real target.

    The guide claimed two different Escape targets in two places while the
    chip named a third. The chip is pinned (test_library_media_reader_flow)
    and the guide was not, so the guide now QUOTES the chip -- and this
    asserts the quotes are the strings the code actually produces, so the
    two cannot drift apart again.
    """
    guide = _GUIDE.read_text(encoding="utf-8")

    # The plain viewer: focus in the Reader, nothing transient open.
    reader_fake, _calls, shell, _find = _escape_fake(region="reader")
    assert LibraryScreen._library_media_escape_label(reader_fake) == "focus Items"
    assert "`esc focus Items`" in guide

    # One step out: focus in the Items pane, Library still open.
    reader_fake.focused = SimpleNamespace(ancestors=(shell.items,))
    assert LibraryScreen._library_media_escape_label(reader_fake) == "focus Library"
    assert "`esc focus Library`" in guide

    # The fourth reading: the next pane out is collapsed, not on screen.
    shell.effective_layout.library_open = False
    assert LibraryScreen._library_media_escape_label(reader_fake) == "back"
    assert "`esc back`" in guide

    # Every transient sub-state shares the one word.
    more_fake, _calls, _shell, _find = _escape_fake(region="reader", more_open=True)
    assert LibraryScreen._library_media_escape_label(more_fake) == "close"
    assert "`esc close`" in guide

    # The contradicting claims are gone.
    assert "Escape never leaves the Reader at all" not in guide
    assert "in narrower layouts Escape shows the list again" not in guide


def test_a_delete_and_its_undo_both_stamp_the_item_modified_now(tmp_path):
    """task-32224: the re-dating starts at the DELETE, not at the Undo.

    The guide claimed "restore never rewrites the item". It does, and so
    does the delete that precedes it: both ``mark_as_trash`` and
    ``restore_from_trash`` bump ``version`` and stamp ``last_modified`` with
    the current time (they are optimistic-locking writes that log a sync
    event, so the stamp is that row's sync clock, not an edit marker).

    Dropping the stamp from the restore alone would therefore fix nothing
    -- the item would keep the DELETE's "now" and still sort to the top of
    Newest. Preserving the pre-delete time would mean capturing it before
    the delete and writing it back through the restore, i.e. a new
    cross-cutting write path in the media DB's shared sync contract for a
    P3 ordering nicety. AC#1's second clause is taken instead: the guide
    now states this behaviour, and this test is what keeps it honest.
    """
    db = MediaDatabase(str(tmp_path / "media.db"), client_id="crit9-media-reader")
    media_id, _uuid, _msg = db.add_media_with_keywords(
        title="Roadmap Sync Notes",
        media_type="document",
        content="# Roadmap sync\n\nBody.\n",
        keywords=["roadmap"],
        url="crit9://roadmap",
    )

    def stored_last_modified() -> str:
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT last_modified FROM Media WHERE id = ?", (media_id,)
            ).fetchone()
        return str(row[0])

    seeded = stored_last_modified()
    # ``_get_current_utc_timestamp_str`` formats %f truncated to
    # milliseconds, so 10 ms is already 10x the resolution.
    time.sleep(0.01)
    assert db.mark_as_trash(media_id) is True
    after_delete = stored_last_modified()
    time.sleep(0.01)
    assert db.restore_from_trash(media_id) is True
    after_undo = stored_last_modified()

    assert after_delete != seeded, (seeded, after_delete)
    assert after_undo != after_delete, (after_delete, after_undo)

    # The guide says exactly this, in the words the user reads (compared
    # with the page's own line wrapping collapsed, so a reflow is not a
    # test failure).
    guide = _GUIDE.read_text(encoding="utf-8")
    prose = " ".join(guide.split())
    assert (
        "Restore brings the item back and marks it changed now, so it "
        "returns at the top of a Newest sort" in prose
    ), "The Trash/Restore paragraph no longer states the real behaviour."
    assert "restore never rewrites the item" not in prose
