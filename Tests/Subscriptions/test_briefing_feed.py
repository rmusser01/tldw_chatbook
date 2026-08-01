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
