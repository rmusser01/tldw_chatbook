"""Tests for the pure RSS feed builder (spec #2 phase 3, Task 3).

`build_feed_xml` turns a watchlist's audio episodes -- already shaped as
`FeedEpisode` -- into a self-contained RSS 2.0 podcast feed as bytes. It is
pure: no filesystem access, no ambient clock (`now` is an injected
parameter), so every test here is deterministic.

Every assertion parses the result back with `xml.etree.ElementTree` and
asserts on the parsed tree, never on a substring of the raw bytes -- a
substring test would happily pass on malformed XML, which is exactly the
failure this suite exists to exclude. The hostile-title test in particular
is a round-trip: build with a title containing `<`, `&` and `]]>`, parse it
back, and assert the parsed text equals the original exactly. Escaping is
ElementTree's job; this module must never hand-escape.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, unquote

import pytest

from tldw_chatbook.Subscriptions.briefing_feed import (
    FeedBuildError,
    FeedEpisode,
    build_feed_xml,
)

pytestmark = pytest.mark.unit

_ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
_NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


def _episode(**overrides: object) -> FeedEpisode:
    fields: dict = dict(
        title="Episode 1",
        filename="script-1-audio-2.wav",
        length_bytes=12345,
        duration_seconds=90.0,
        published=datetime(2026, 7, 30, 8, 0, 0, tzinfo=timezone.utc),
        guid="briefing-audio-2",
        description="A description of episode 1.",
    )
    fields.update(overrides)
    return FeedEpisode(**fields)


def _build(episodes: list[FeedEpisode], **overrides: object) -> bytes:
    kwargs: dict = dict(
        channel_title="My Watchlist",
        channel_description="Episodes from My Watchlist",
        episodes=episodes,
        now=_NOW,
    )
    kwargs.update(overrides)
    return build_feed_xml(**kwargs)


# --- document shape --------------------------------------------------------


def test_document_is_well_formed_rss_2_with_exactly_one_channel():
    root = ET.fromstring(_build([]))
    assert root.tag == "rss"
    assert root.get("version") == "2.0"
    channels = root.findall("channel")
    assert len(channels) == 1


def test_channel_carries_title_description_and_last_build_date():
    channel = ET.fromstring(_build([])).find("channel")
    assert channel.findtext("title") == "My Watchlist"
    assert channel.findtext("description") == "Episodes from My Watchlist"
    last_build_date = channel.findtext("lastBuildDate")
    assert last_build_date is not None
    # Round-trip through RFC-822 parsing, not a string-format assertion.
    assert parsedate_to_datetime(last_build_date) == _NOW


def test_channel_link_is_present_non_empty_and_a_relative_reference():
    """FIX C (whole-branch review): RSS 2.0 requires `title`, `link` and
    `description` on every channel; validators and Apple's podcast
    requirements flag a missing `<link>`. There is no server URL to point
    at (serving the exported folder is the user's own choice), so this is
    the feed's own filename as a relative reference -- consistent with the
    directory being self-contained."""
    link = ET.fromstring(_build([])).find("channel").findtext("link")
    assert link
    assert "://" not in link  # a relative reference, not an invented URL
    assert link == "feed.xml"


def test_empty_episodes_still_produces_a_valid_channel_with_zero_items():
    channel = ET.fromstring(_build([])).find("channel")
    assert channel is not None
    assert channel.find("title") is not None
    assert channel.findall("item") == []


# --- one episode -> one item -------------------------------------------------


def test_each_episode_yields_an_item_with_title_guid_and_pubdate():
    episode = _episode()
    channel = ET.fromstring(_build([episode])).find("channel")
    items = channel.findall("item")
    assert len(items) == 1
    item = items[0]
    assert item.findtext("title") == episode.title
    assert item.findtext("guid") == episode.guid
    pub_date = item.findtext("pubDate")
    assert pub_date is not None
    assert parsedate_to_datetime(pub_date) == episode.published


def test_two_episodes_yield_two_items_in_order():
    first = _episode(title="First", guid="g1")
    second = _episode(title="Second", guid="g2")
    channel = ET.fromstring(_build([first, second])).find("channel")
    items = channel.findall("item")
    assert [item.findtext("title") for item in items] == ["First", "Second"]


# --- enclosure ---------------------------------------------------------------


def test_enclosure_url_is_the_bare_relative_filename_never_absolute():
    episode = _episode(filename="script-9-audio-3.wav")
    item = ET.fromstring(_build([episode])).find("channel").find("item")
    enclosure = item.find("enclosure")
    assert enclosure is not None
    url = enclosure.get("url")
    assert url == "script-9-audio-3.wav"
    assert not url.startswith("/")
    assert ":" not in url  # excludes both an absolute POSIX path and a URL scheme


def test_enclosure_length_is_bytes_and_type_is_audio_wav():
    episode = _episode(length_bytes=98765)
    enclosure = ET.fromstring(_build([episode])).find("channel").find("item").find("enclosure")
    assert enclosure.get("length") == "98765"
    assert enclosure.get("type") == "audio/wav"


# --- FIX A (whole-branch review): enclosure URLs are percent-encoded -------
#
# `FeedEpisode.filename` comes from Task 4's `safe_export_stem`, which
# deliberately keeps spaces (pinned for its other caller by
# `test_stem_keeps_ordinary_characters` in
# `test_briefing_export_markdown.py`) and, via `str.isalnum()`, also admits
# non-ASCII letters. RFC 3986 forbids a raw space in a URI reference; a
# podcast client that sends it verbatim in the request line (rather than
# percent-encoding on resolve) gets a 400. The fix must be applied ONLY at
# emission -- `FeedEpisode.filename` itself, and the file Task 4 writes to
# disk under that name, must be untouched.


def test_enclosure_url_percent_encodes_a_space_and_carries_no_raw_space():
    episode = _episode(filename="Two Host Debate-42.wav")
    url = (
        ET.fromstring(_build([episode]))
        .find("channel")
        .find("item")
        .find("enclosure")
        .get("url")
    )
    assert "%20" in url
    assert " " not in url


def test_enclosure_url_percent_encodes_non_ascii_characters():
    episode = _episode(filename="Café Debate-7.wav")
    url = (
        ET.fromstring(_build([episode]))
        .find("channel")
        .find("item")
        .find("enclosure")
        .get("url")
    )
    assert "é" not in url
    assert "%C3%A9" in url  # UTF-8 bytes for "é", percent-encoded


@pytest.mark.parametrize(
    "filename",
    [
        "script-9-audio-3.wav",
        "Two Host Debate-42.wav",
        "Café Debate-7.wav",
        "already has, punctuation!.wav",
    ],
)
def test_enclosure_url_is_idempotently_encoded(filename):
    """The general property: the emitted `url` is already fully encoded --
    decoding then re-encoding it must be a no-op. A url that were only
    partially encoded (or double-encoded) would fail this even if the
    space/non-ASCII spot-checks above happened to pass."""
    episode = _episode(filename=filename)
    url = (
        ET.fromstring(_build([episode]))
        .find("channel")
        .find("item")
        .find("enclosure")
        .get("url")
    )
    assert url == quote(unquote(url), safe="")


def test_enclosure_url_percent_encoding_does_not_change_the_on_disk_filename():
    """`FeedEpisode.filename` -- what Task 4 writes to disk and what a
    future reader of this module should expect to find unmodified -- must
    stay exactly as given; only the emitted `<enclosure>` URL is encoded."""
    episode = _episode(filename="Two Host Debate-42.wav")
    assert episode.filename == "Two Host Debate-42.wav"
    _build([episode])  # building must not mutate the (frozen) episode
    assert episode.filename == "Two Host Debate-42.wav"


# --- itunes:duration ----------------------------------------------------------


def test_itunes_duration_emitted_when_duration_seconds_present():
    episode = _episode(duration_seconds=125.0)
    item = ET.fromstring(_build([episode])).find("channel").find("item")
    duration_el = item.find(f"{{{_ITUNES_NS}}}duration")
    assert duration_el is not None
    assert duration_el.text  # non-empty


def test_itunes_duration_omitted_when_duration_seconds_is_none():
    episode = _episode(duration_seconds=None)
    item = ET.fromstring(_build([episode])).find("channel").find("item")
    assert item.find(f"{{{_ITUNES_NS}}}duration") is None


# --- escaping: ElementTree's job, proven by round-trip -----------------------


def test_hostile_title_round_trips_exactly_through_parse():
    hostile_title = "Report: <Q3> Sales & Marketing ]]> Recap"
    episode = _episode(title=hostile_title)
    item = ET.fromstring(_build([episode])).find("channel").find("item")
    assert item.findtext("title") == hostile_title


def test_hostile_description_round_trips_exactly_through_parse():
    hostile_description = "A & B <together>, plus a stray ]]> marker."
    episode = _episode(description=hostile_description)
    item = ET.fromstring(_build([episode])).find("channel").find("item")
    assert item.findtext("description") == hostile_description


# --- the path-separator guard ------------------------------------------------


@pytest.mark.parametrize("bad_filename", ["../evil.wav", "sub/evil.wav", "sub\\evil.wav"])
def test_filename_with_a_path_separator_raises_feed_build_error_naming_it(bad_filename):
    episode = _episode(filename=bad_filename)
    with pytest.raises(FeedBuildError, match=re.escape(bad_filename)):
        _build([episode])


def test_path_separator_guard_rejects_before_emitting_any_xml():
    """A single bad episode among good ones must still raise -- not silently
    skip the bad one and emit a feed missing it (Task 4 is the layer that
    decides skip-vs-fail for I/O-level problems; a filename this shape is a
    programming error in the caller, not a runtime skip candidate)."""
    good = _episode(filename="ok.wav", guid="g-ok")
    bad = _episode(filename="../evil.wav", guid="g-bad")
    with pytest.raises(FeedBuildError):
        _build([good, bad])


# --- timezone-awareness: reject naive datetimes at the boundary -------------
#
# `email.utils.format_datetime` does NOT treat a naive datetime as local
# time -- it emits the correct wall-clock digits tagged "-0000" (RFC 2822's
# "unverified offset"). The hazard is one hop downstream: "-0000" round-trips
# through `parsedate_to_datetime` into a *naive* result, and the common
# `.astimezone()` idiom then silently reinterprets those same digits as
# local time -- same digits, wrong instant. `build_feed_xml` forecloses this
# by rejecting naive input outright rather than quietly coercing it to UTC.


def test_naive_now_raises_feed_build_error_naming_now():
    naive_now = datetime(2026, 8, 1, 12, 0, 0)  # no tzinfo
    with pytest.raises(FeedBuildError, match="now"):
        _build([], now=naive_now)


def test_naive_episode_published_raises_feed_build_error_naming_the_episode():
    naive_published = datetime(2026, 7, 30, 8, 0, 0)  # no tzinfo
    episode = _episode(published=naive_published, guid="briefing-audio-77")
    with pytest.raises(FeedBuildError, match="briefing-audio-77"):
        _build([episode])


def test_aware_utc_datetimes_produce_a_plus_zero_offset_not_unverified():
    """A `-0000` offset means "unverified" per RFC 2822 -- an aware UTC value
    must emit the unambiguous `+0000` instead, proving the guard above is
    actually necessary (not just a formality) and that the aware path is
    unaffected by it."""
    episode = _episode(published=_NOW)
    xml_bytes = _build([episode], now=_NOW)
    text = xml_bytes.decode("utf-8")
    channel = ET.fromstring(xml_bytes).find("channel")
    assert channel.findtext("lastBuildDate").endswith("+0000")
    assert channel.find("item").findtext("pubDate").endswith("+0000")
    assert "-0000" not in text


# --- itunes namespace: pin the raw-bytes declaration, not just the URI ------


def test_xmlns_itunes_is_declared_in_raw_bytes_when_a_duration_is_present():
    """Finding `itunes:duration` by namespace URI (as the tests above do)
    would still succeed even if the `xmlns:itunes` declaration were missing
    from the serialized bytes -- an ElementTree quirk that would produce
    invalid XML many real parsers reject. Assert the declaration itself is
    present in the actual output, not just that the parsed tree resolves the
    tag."""
    episode = _episode(duration_seconds=42.0)
    text = _build([episode]).decode("utf-8")
    assert f'xmlns:itunes="{_ITUNES_NS}"' in text


def test_xmlns_itunes_is_absent_from_raw_bytes_when_no_duration_is_present():
    episode = _episode(duration_seconds=None)
    text = _build([episode]).decode("utf-8")
    assert "xmlns:itunes" not in text
    assert "itunes:duration" not in text
