"""TASK-1343: the producers of `content_kind` / `content_format` / `diff_summary`.

Phase D built two reader renderers and dispatches between them on
`content_kind` (`UI/Watchlists_Modules/content_pane.py`, `render_for`), but
nothing in the repo wrote the field. So every item -- including every site
change -- fell through to the article renderer, `render_change` was unreachable
in production, and `diff_summary` had no producer at all.

These tests drive the REAL producers (`URLMonitor.check_url`,
`FeedMonitor.check_feed`, `LocalWatchlistsService._normalize_api_item`) through
the real persistence path and then through the real renderer. Hand-built item
dicts would pass whether or not any producer writes anything, which is exactly
how this shipped: `Tests/UI/test_watchlists_content_pane.py` covers the
renderer thoroughly and every one of its fixtures sets `content_kind` itself.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import LocalWatchlistsService
from tldw_chatbook.Subscriptions.item_persist import (
    _VALID_PAIRINGS,
    _validate_content_pairing,
)

pytestmark = pytest.mark.unit


# --- harness ---------------------------------------------------------------


def _response(text: str, *, content_type: str = "text/html") -> SimpleNamespace:
    """A stand-in for the `httpx.Response` `guarded_fetch_httpx_async` returns.

    `_fetch_url_content` reads `.text`, `.headers` (as a dict) and
    `.status_code`; `_fetch_and_parse_feed` also reads
    `.headers.get("content-type")` and calls `.raise_for_status()`.
    """
    return SimpleNamespace(
        status_code=200,
        headers={"content-type": content_type},
        text=text,
        final_url="https://example.com/page",
        raise_for_status=lambda: None,
    )


def _serve(monkeypatch, pages: list[str], *, content_type: str = "text/html") -> None:
    """Serve `pages` in order from the monitoring engine's guarded fetch.

    The last page is repeated once the list is exhausted, so a test can do any
    number of extra checks without the fetch running out.
    """
    remaining = list(pages)

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        page = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return _response(page, content_type=content_type)

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )


async def _check(service: LocalWatchlistsService, source_id: int) -> dict:
    """Launch and execute one real run for `source_id`."""
    launched = await service.launch_run(source_id=source_id)
    return await service.execute_run(launched["run_id"])


def _stored_items(db: SubscriptionsDB, source_id: int) -> list[dict]:
    rows = db.conn.execute(
        "SELECT * FROM subscription_items WHERE subscription_id = ? ORDER BY id ASC",
        (source_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _rendered(item_row: dict) -> tuple[str, str]:
    """Normalize a stored row and render it exactly as the reader does.

    Goes through `normalize_watchlist_item` (the dict the screen actually hands
    `ContentPane`) and `render_for` (the dispatch under test), then through a
    real `rich.console.Console` so the returned pair is (painted characters,
    characters plus style codes).
    """
    from rich.console import Console

    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_for

    item = normalize_watchlist_item("local", item_row)
    console = Console(width=80, record=True, color_system="standard", force_terminal=True)
    console.print(render_for(item))
    return console.export_text(clear=False), console.export_text(styles=True)


_PAGE_BEFORE = """<html><body>
<h1>Anthropic status</h1>
<p>All systems operational.</p>
<p>Latest release: Opus 4.1 is available.</p>
</body></html>"""

_PAGE_AFTER = """<html><body>
<h1>Anthropic status</h1>
<p>All systems operational.</p>
<p>Latest release: Opus 4.5 is available.</p>
<p>Scheduled maintenance on Friday at 02:00 UTC.</p>
</body></html>"""

_RSS = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
  <title>Anthropic News</title>
  <item>
    <title>Opus 4.5 is now available</title>
    <link>https://example.com/news/opus-45</link>
    <description>The model is available in the API today.</description>
    <pubDate>Tue, 28 Jul 2026 09:00:00 +0000</pubDate>
  </item>
</channel></rss>"""


async def _site_source(tmp_path, monkeypatch, pages: list[str]):
    """A real `url` source, its DB and service, with `pages` served in order."""
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    _serve(monkeypatch, pages)
    source = await service.create_source(
        {
            "name": "Anthropic status",
            "url": "https://example.com/page",
            "source_type": "site",
        }
    )
    return db, service, int(source["source_id"])


# --- AC#1 / AC#3: the change path reaches `render_change` ------------------


@pytest.mark.asyncio
async def test_a_real_site_change_is_dispatched_to_the_change_renderer(
    tmp_path, monkeypatch
):
    """AC#3, driven end to end: fetch, detect, persist, normalize, dispatch.

    The first check only stores a baseline (`check_url` returns `None` with no
    previous snapshot), so it must produce no item at all; the second must
    produce one whose `content_kind` sends `render_for` to `render_change`.

    The discriminator is the one `Tests/UI/test_watchlists_content_pane.py`
    had to adopt for the same reason: `render_article`'s meta line emits
    `"<n> words"` and `render_change` emits it under no input, so asserting on
    it pins the dispatch in both directions. Deleting the `content_kind`
    assignment in `check_url` reddens this test.
    """
    db, service, source_id = await _site_source(
        tmp_path, monkeypatch, [_PAGE_BEFORE, _PAGE_AFTER]
    )

    first = await _check(service, source_id)
    assert first["status"] == "completed"
    assert _stored_items(db, source_id) == [], (
        "the first check only stores a baseline snapshot -- the precondition"
    )

    second = await _check(service, source_id)
    assert second["status"] == "completed"
    items = _stored_items(db, source_id)
    assert len(items) == 1

    item = items[0]
    assert item["content_kind"] == "change"
    assert item["content_format"] == "diff"

    plain, _ansi = _rendered(item)
    assert "words" not in plain, (
        "a site change must NOT render through render_article -- 'words' comes "
        "from its meta line and render_change never emits it"
    )
    assert "% changed" in plain, "the change headline must be present"


@pytest.mark.asyncio
async def test_the_stored_change_content_is_a_diff_not_the_whole_new_page(
    tmp_path, monkeypatch
):
    """`content` used to be `current_content["text"]` -- the entire new page.

    So the reader could see what the page says now but never what changed,
    while the full page was ALSO already stored in `url_snapshots`. Asserts
    the real before/after shows up as `-`/`+` lines, that the unchanged
    sentence appears once (as context) rather than the whole page being dumped
    twice, and that `render_change` actually colours those lines.
    """
    db, service, source_id = await _site_source(
        tmp_path, monkeypatch, [_PAGE_BEFORE, _PAGE_AFTER]
    )
    await _check(service, source_id)
    await _check(service, source_id)

    item = _stored_items(db, source_id)[0]
    content = item["content"]

    assert "-Latest release: Opus 4.1 is available." in content, (
        "the removed line must be present, prefixed for the renderer to colour"
    )
    assert "+Latest release: Opus 4.5 is available." in content
    assert "+Scheduled maintenance on Friday at 02:00 UTC." in content
    # The unchanged sentence survives once, as diff context -- not as part of a
    # second full copy of the page.
    assert content.count("All systems operational.") == 1

    # The full page is still recoverable, from the snapshot table that already
    # held it -- nothing was lost by storing the diff instead. Ordered by `id`,
    # not `created_at`: `url_snapshots.created_at` has one-second resolution, so
    # two checks in the same second are indistinguishable by it (see the report
    # -- `check_url`'s own "previous snapshot" query has the same weakness).
    snapshot = db.conn.execute(
        "SELECT extracted_content FROM url_snapshots WHERE subscription_id = ?"
        " ORDER BY id DESC LIMIT 1",
        (source_id,),
    ).fetchone()
    assert "Scheduled maintenance on Friday" in snapshot["extracted_content"]
    assert "Latest release: Opus 4.5" in snapshot["extracted_content"]

    plain, ansi = _rendered(item)
    assert "Opus 4.1" in plain and "Opus 4.5" in plain
    assert "\x1b[32m" in ansi, "an added (`+`) diff line must be painted green"
    assert "\x1b[31m" in ansi, "a removed (`-`) diff line must be painted red"


@pytest.mark.asyncio
async def test_the_change_headline_carries_a_diff_summary_and_a_real_percentage(
    tmp_path, monkeypatch
):
    """AC#4, plus the scale of `change_percentage`.

    `diff_summary` is rendered by `render_change` and had no producer at all.
    `change_percentage` is now stored on the 0-100 scale its column name, the
    renderer's `f"{float(pct):.0f}% changed"` and every renderer fixture read
    it on; `calculate_change_percentage` returns a 0.0-1.0 ratio, so an
    unscaled 25% change would have printed "0% changed".
    """
    db, service, source_id = await _site_source(
        tmp_path, monkeypatch, [_PAGE_BEFORE, _PAGE_AFTER]
    )
    await _check(service, source_id)
    await _check(service, source_id)

    item = _stored_items(db, source_id)[0]

    assert item["diff_summary"], "diff_summary must have a producer now"
    assert "\n" not in item["diff_summary"], "it is a one-line headline field"
    assert "added" in item["diff_summary"] and "removed" in item["diff_summary"]

    pct = item["change_percentage"]
    assert isinstance(pct, float)
    assert 1.0 < pct <= 100.0, (
        f"change_percentage must be a percentage, not a 0-1 ratio (got {pct!r})"
    )

    plain, _ansi = _rendered(item)
    assert f"{pct:.0f}% changed" in plain
    assert "0% changed" not in plain, (
        "a real change must never print as 0% -- the ratio/percent mismatch"
    )
    assert item["diff_summary"] in plain


@pytest.mark.asyncio
async def test_change_type_is_derived_from_the_snapshots_not_hardcoded(
    tmp_path, monkeypatch
):
    """`change_type` was the literal `"content"` for every change ever.

    Only the distinctions the two snapshots actually support are claimed:
    text appearing where there was none is `"new"`, text disappearing entirely
    is `"removed"`, anything else is `"content"`. A page whose text goes away
    used to be reported to the user as an ordinary content edit.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import classify_change_type

    assert classify_change_type("", "hello") == "new"
    assert classify_change_type("hello", "") == "removed"
    assert classify_change_type("   ", "hello") == "new"
    assert classify_change_type("hello", "goodbye") == "content"

    # And it reaches the DB from a real check: this page loses all of its text.
    db, service, source_id = await _site_source(
        tmp_path,
        monkeypatch,
        [_PAGE_BEFORE, "<html><body><script>x=1</script></body></html>"],
    )
    await _check(service, source_id)
    await _check(service, source_id)

    item = _stored_items(db, source_id)[0]
    assert item["change_type"] == "removed", (
        "a page that lost all of its text is not a 'content' edit"
    )
    plain, _ansi = _rendered(item)
    assert "removed" in plain


# --- AC#2: the feed and API paths ------------------------------------------


@pytest.mark.asyncio
async def test_a_feed_item_is_stored_as_an_article_with_a_legal_format(
    tmp_path, monkeypatch
):
    """AC#2. The RSS path wrote neither field, so it relied on `render_for`'s
    fallback -- which is what made the dispatch untestable in the first place.

    `"text"` is the honest format: `description` arrives as the publisher's
    plain text or HTML and nothing on this path converts it to markdown.
    Claiming `"markdown"` would hand publisher HTML to a CommonMark parser.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    _serve(monkeypatch, [_RSS], content_type="application/rss+xml")

    source = await service.create_source(
        {
            "name": "Anthropic News",
            "url": "https://example.com/feed.xml",
            "source_type": "rss",
        }
    )
    source_id = int(source["source_id"])
    completed = await _check(service, source_id)
    assert completed["status"] == "completed"

    items = _stored_items(db, source_id)
    assert len(items) == 1
    item = items[0]
    assert item["content_kind"] == "article"
    assert item["content_format"] == "text"
    assert item["content"] == "The model is available in the API today."

    plain, _ansi = _rendered(item)
    assert "words" in plain, "an article must render through render_article"
    assert "% changed" not in plain


@pytest.mark.asyncio
async def test_an_api_item_is_stored_as_an_article_with_a_legal_format(
    tmp_path, monkeypatch
):
    """The API path is normalized in the service, not the monitoring engine,
    and had the same gap. Its body is whatever JSON field the source's
    `field_map` points at, in whatever format the API chose and unconverted,
    so `"text"` is what was captured.
    """
    payload = {
        "items": [
            {
                "title": "Alpha update",
                "url": "https://api.example.com/a",
                "summary": "First item body.",
            }
        ]
    }

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            final_url=url,
            raise_for_status=lambda: None,
            json=lambda: payload,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.local_watchlists_service."
        "guarded_fetch_httpx_async",
        fake_guarded,
    )

    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source = await service.create_source(
        {
            "name": "API changelog",
            "url": "https://api.example.com/changes",
            "source_type": "api",
        }
    )
    source_id = int(source["source_id"])
    await _check(service, source_id)

    item = _stored_items(db, source_id)[0]
    assert item["content_kind"] == "article"
    assert item["content_format"] == "text"

    plain, _ansi = _rendered(item)
    assert "words" in plain


# --- the size bound --------------------------------------------------------


@pytest.mark.asyncio
async def test_an_oversized_change_is_truncated_and_says_so(tmp_path, monkeypatch):
    """A large page can produce an enormous diff, and this goes into a TEXT
    column and a pane about nine rows tall.

    The bound must hold, and the body must SAY it was cut -- a silently
    partial diff is a change presented as complete. The summary counts the
    whole diff, not the retained slice, so the headline stays true.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import (
        _MAX_DIFF_CHARS,
        _MAX_DIFF_LINES,
        build_change_diff,
    )

    before = " ".join(f"Paragraph {i} says the old thing." for i in range(900))
    after = " ".join(f"Paragraph {i} says the new thing instead." for i in range(900))

    body, summary = build_change_diff(before, after)
    lines = body.splitlines()

    assert "diff truncated" in body, "truncation must be stated in the content"
    assert "partial view" in body
    # The notice is appended after the cap, so it is the one line allowed past
    # it -- and it must not itself be coloured as a change.
    assert len(lines) <= _MAX_DIFF_LINES + 1
    assert not lines[-1].startswith(("+", "-"))
    assert len(body) <= _MAX_DIFF_CHARS + len(lines[-1]) + 1

    # 900 removals and 900 additions really were produced; the summary must
    # report those, not the ~400 lines that fit inside the cap.
    assert summary == "900 line(s) added, 900 removed", summary
    assert f"{len(lines)} line" not in summary

    # And the same bound holds on the live path, not just in this helper.
    db, service, source_id = await _site_source(
        tmp_path,
        monkeypatch,
        [f"<html><body><p>{before}</p></body></html>",
         f"<html><body><p>{after}</p></body></html>"],
    )
    await _check(service, source_id)
    await _check(service, source_id)
    stored = _stored_items(db, source_id)[0]["content"]
    assert "diff truncated" in stored
    assert len(stored.splitlines()) <= _MAX_DIFF_LINES + 1


@pytest.mark.asyncio
async def test_the_whole_stored_diff_body_is_exactly_this(tmp_path, monkeypatch):
    """Pin the body a real check produces, in full.

    The individual `in content` assertions above cannot show that the result is
    *readable*: four short lines with a hunk header, not a wall of text. This
    is what a reader sees in a pane about nine rows tall, and any change to the
    diff's shape has to come through here and be looked at.
    """
    db, service, source_id = await _site_source(
        tmp_path, monkeypatch, [_PAGE_BEFORE, _PAGE_AFTER]
    )
    await _check(service, source_id)
    await _check(service, source_id)

    assert _stored_items(db, source_id)[0]["content"] == (
        "@@ -1,2 +1,3 @@\n"
        " Anthropic status All systems operational.\n"
        "-Latest release: Opus 4.1 is available.\n"
        "+Latest release: Opus 4.5 is available.\n"
        "+Scheduled maintenance on Friday at 02:00 UTC."
    )


def test_a_change_with_no_textual_difference_says_so_rather_than_nothing():
    """The content hash is taken over the raw extracted text while
    segmentation trims whitespace, so a whitespace-only change hashes
    differently and diffs to nothing.

    An empty body would make `render_change` print "no body captured for this
    item -- re-check this source to fetch it": a claim that nothing was
    captured, when in fact it was captured and it matched.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import build_change_diff

    body, summary = build_change_diff("One sentence.  Two.", "One sentence. Two.")

    assert body, "never return an empty body -- the renderer would misreport it"
    assert "identical" in body
    assert not body.startswith(("+", "-")), "the notice must not read as a change"
    assert summary == "no textual change after normalization"


def test_the_diff_is_readable_in_a_narrow_pane():
    """`extract_text_from_html` joins every chunk of a page with a single
    space, so extracted page text is ONE line with no newlines in it.

    A line-based diff of two such snapshots is therefore always exactly
    `-<the entire old page>` / `+<the entire new page>` -- the full text twice.
    Both sides are re-segmented before diffing, and no emitted line may be
    wider than the reader pane can show.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import (
        _MAX_DIFF_SEGMENT_CHARS,
        build_change_diff,
    )

    before = "Alpha holds. " + "Beta " * 200 + "stays. Gamma was here."
    after = "Alpha holds. " + "Beta " * 200 + "stays. Delta is here now."

    body, _summary = build_change_diff(before, after)
    lines = body.splitlines()

    assert len(lines) > 2, "a single-line page must still produce a real diff"
    for line in lines:
        assert len(line) <= _MAX_DIFF_SEGMENT_CHARS + 2, (
            f"a diff line wider than the pane: {line[:60]!r}... ({len(line)} chars)"
        )
    assert any(line.startswith("-") and "Gamma was here." in line for line in lines)
    assert any(line.startswith("+") and "Delta is here now." in line for line in lines)
    # The unified-diff file headers would be painted red and green by
    # `render_change` as though the header were the change.
    assert not any(line.startswith(("---", "+++")) for line in lines)


# --- no path may emit an invalid pairing -----------------------------------


@pytest.mark.asyncio
async def test_no_producer_emits_a_pairing_persistence_would_reject(
    tmp_path, monkeypatch
):
    """`persist_subscription_item` RAISES on an invalid pairing, and that raise
    lands inside a scheduled fetch, where `execute_run` converts it into a
    failed run and drops every item the run collected.

    So every producer's output is collected here and checked against
    `_VALID_PAIRINGS` itself -- not against a copy of the list.
    """
    from tldw_chatbook.Subscriptions.local_watchlists_service import (
        LocalWatchlistsService as Service,
    )
    from tldw_chatbook.Subscriptions.monitoring_engine import FeedMonitor

    produced: list[tuple] = []

    monitor = FeedMonitor()
    produced.extend(
        (item.get("content_kind"), item.get("content_format"))
        for item in monitor._parse_xml_feed(_RSS, "rss")
    )
    produced.extend(
        (item.get("content_kind"), item.get("content_format"))
        for item in monitor._parse_xml_feed(_ATOM, "atom")
    )
    produced.extend(
        (item.get("content_kind"), item.get("content_format"))
        for item in monitor._parse_json_feed(_JSON_FEED)
    )
    api_item = Service._normalize_api_item(
        {"title": "t", "url": "https://example.com/x", "content": "body"},
        {},
        "https://example.com/api",
    )
    produced.append((api_item.get("content_kind"), api_item.get("content_format")))

    db, service, source_id = await _site_source(
        tmp_path, monkeypatch, [_PAGE_BEFORE, _PAGE_AFTER]
    )
    await _check(service, source_id)
    await _check(service, source_id)
    change = _stored_items(db, source_id)[0]
    produced.append((change["content_kind"], change["content_format"]))

    assert len(produced) == 4 + 1, f"a producer stopped emitting: {produced}"
    for pairing in produced:
        assert pairing in _VALID_PAIRINGS, f"{pairing} would be rejected on write"
        _validate_content_pairing(*pairing)


def test_every_content_kind_literal_in_the_package_is_from_the_vocabulary():
    """The typo guard, structurally.

    A producer that writes `"changed"` or `"plain"` instead of the real value
    does not fail at the write boundary (`_validate_content_pairing` raises,
    but only when that code path runs -- inside a scheduled fetch, on someone
    else's machine). Every `content_kind`/`content_format` value written as a
    dict-literal in the subscriptions/watchlists code must therefore be either
    one of `item_persist`'s named constants or a member of the vocabulary.
    Values that are expressions (e.g. `row.get("content_kind")` in
    `watchlist_normalizers`, which READS the field) are not producers and are
    skipped.

    Scoped to the trees that own this vocabulary: `content_format` is an
    unrelated, differently-valued key in the media-reading services
    (`Media/local_media_reading_service.py` and friends), and sweeping the
    whole package would be asserting something false about them.
    """
    kinds = {kind for kind, _fmt in _VALID_PAIRINGS}
    formats = {fmt for _kind, fmt in _VALID_PAIRINGS}
    allowed_names = {
        "CONTENT_KIND_ARTICLE",
        "CONTENT_KIND_CHANGE",
        "CONTENT_FORMAT_TEXT",
        "CONTENT_FORMAT_MARKDOWN",
        "CONTENT_FORMAT_DIFF",
    }
    package_root = Path(__file__).resolve().parents[2] / "tldw_chatbook"
    scanned = [
        *(package_root / "Subscriptions").rglob("*.py"),
        *(package_root / "UI" / "Watchlists_Modules").rglob("*.py"),
        package_root / "DB" / "Subscriptions_DB.py",
    ]
    assert len(scanned) > 20, "the scan lost its files"

    offenders: list[str] = []
    for path in scanned:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or key.value not in (
                    "content_kind",
                    "content_format",
                ):
                    continue
                allowed = kinds if key.value == "content_kind" else formats
                if isinstance(value, ast.Name):
                    if value.id not in allowed_names:
                        offenders.append(f"{path}:{value.lineno} {value.id}")
                elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if value.value not in allowed:
                        offenders.append(f"{path}:{value.lineno} {value.value!r}")

    assert not offenders, f"values outside the vocabulary: {offenders}"


_ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Anthropic News</title>
  <entry>
    <title>Opus 4.5 is now available</title>
    <link rel="alternate" href="https://example.com/news/opus-45"/>
    <summary>The model is available in the API today.</summary>
    <published>2026-07-28T09:00:00Z</published>
  </entry>
</feed>"""

_JSON_FEED = """{
  "version": "https://jsonfeed.org/version/1.1",
  "items": [
    {
      "id": "1",
      "title": "Opus 4.5 is now available",
      "url": "https://example.com/news/opus-45",
      "content_html": "<p>The model is available in the API today.</p>",
      "date_published": "2026-07-28T09:00:00Z"
    }
  ]
}"""
