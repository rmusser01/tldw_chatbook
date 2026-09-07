"""Unit tests for `snapshot_view_modal.py`'s header and body rendering.

Batch-4 review, I2: task-2308's own scope ("every Watchlists table
timestamp") missed this modal's `Captured <created_at>` line -- it opens
from the Full page/Previous snapshot buttons in the same `#content-actions`
row this batch's reader byline already humanizes.

Batch-4 review round 2, N2: `_snapshot_body` -- the Full page/Previous
snapshot viewer's actual scraped page content, the highest-risk string in
this feature -- was still unstripped in this same module. A raw ESC byte
survived to render.
"""

import pytest
from tldw_chatbook.UI.Watchlists_Modules.humane_time import humane_timestamp
from tldw_chatbook.UI.Watchlists_Modules.snapshot_view_modal import (
    _snapshot_body,
    _snapshot_header,
)

pytestmark = pytest.mark.unit


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


# --- N2: the snapshot BODY is remote page content and must be inert too --


def test_snapshot_body_strips_a_raw_esc_byte():
    body = _snapshot_body("before \x1b[31mred\x1b[0m after")
    assert "\x1b" not in body.plain
    assert "before" in body.plain and "red" in body.plain and "after" in body.plain


def test_snapshot_body_strips_a_c1_control_byte():
    """The single-byte CSI introducer (0x80-0x9F) -- `html_text.strip_
    control_characters`'s own docstring calls it "just as capable" as the
    7-bit `ESC [` form, and unlike a raw ESC it is valid inside content a
    lenient HTML/text pipeline would not reject upstream.
    """
    body = _snapshot_body("Evil\x9b31mPage")
    assert "\x9b" not in body.plain
    assert "Evil" in body.plain and "31mPage" in body.plain


def test_snapshot_body_converts_raw_html_to_readable_prose():
    """`extracted_content` is not guaranteed to already be plain text --
    `URLMonitor._fetch_url_content` only strips HTML when the source's
    `extraction_method` is "full"/"auto"; a raw-extraction source stores
    literal HTML in this column.
    """
    body = _snapshot_body("<p>Hello <a href=\"https://example.test/x\">world</a></p>")
    assert "<p>" not in body.plain and "<a href=" not in body.plain
    assert "Hello" in body.plain and "world" in body.plain
    assert "https://example.test/x" in body.plain


def test_snapshot_body_keeps_markup_shaped_text_literal():
    """Unchanged from before N2 -- pinning the property the module docstring
    states, now routed through `readable_body_text` instead of a bare
    `str()`."""
    body = _snapshot_body("before [bold red]INJECTED[/] and [link=evil]click[/link] after")
    assert "[bold red]INJECTED[/]" in body.plain
    assert "[link=evil]click[/link]" in body.plain


def test_snapshot_body_with_no_content_says_so():
    from tldw_chatbook.UI.Watchlists_Modules.snapshot_view_modal import _NO_CONTENT

    assert _snapshot_body(None).plain == _NO_CONTENT
    assert _snapshot_body("").plain == _NO_CONTENT


def test_snapshot_body_hostile_payload_is_inert_through_a_real_console():
    """The property that actually matters, proven through a real Rich
    `Console` render rather than just checking which characters survive --
    the same discipline `test_watchlists_content_pane.py`'s
    `_render_to_console` helper applies to the reader.
    """
    import io

    from rich.console import Console

    console = Console(
        width=120, record=True, color_system="standard", force_terminal=True,
        file=io.StringIO(),
    )
    console.print(_snapshot_body("before \x1b]8;;http://evil.test\x07label\x1b]8;;\x07 after"))
    plain = console.export_text(clear=False)
    ansi = console.export_text(styles=True)
    assert "\x1b]8;;" not in ansi, "no OSC-8 hyperlink may be manufactured from raw bytes"
    assert "before" in plain and "label" in plain and "after" in plain
