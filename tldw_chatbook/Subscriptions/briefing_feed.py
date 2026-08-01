"""Build a self-contained podcast RSS feed from a watchlist's audio episodes
(spec #2 phase 3, Task 3).

Task 2's `SubscriptionsDB.list_watchlist_audio_episodes` finds the rows;
Task 4 turns each row into a `FeedEpisode`, calls :func:`build_feed_xml`, and
writes the result to `feed.xml` alongside a copy of each episode's audio
file. This module does neither of those things -- it has no filesystem
access and no ambient clock, only pure data in and `bytes` out, which is
what makes it deterministic and unit-testable without a database or a temp
directory.

**No `feedgen`.** A survey for phase 3 confirmed nothing in the repo
generates RSS today (`rss_feed_generator.py` was deliberately deleted; the
`feedparser` dependency is declared but imported nowhere), so this is
written from scratch with stdlib `xml.etree.ElementTree`, following the
`SubElement` + `tostring` precedent at
`Web_Scraping/Article_Extractor_Lib.py:1141-1152`. A new third-party
dependency here would inherit phase 2b's three-job CI edit for no benefit.
`defusedxml` hardens *parsing*, not generation, so it adds nothing to a
module that only ever writes XML.

**Escaping is ElementTree's job.** Every piece of caller-supplied text
(titles, descriptions -- ultimately model output derived from remote feed
content) is set via an element's `.text` attribute and left to
`ET.tostring` to escape. Nothing in this module hand-escapes `&`, `<`, `>`
or `]]>`.

**Enclosure URLs are bare relative filenames, never absolute paths.** The
feed directory Task 4 writes is meant to be self-contained and shareable --
a user may hand the whole folder to a podcast client, sync it, or zip it up.
An absolute path would both break on any other machine and leak the user's
home directory into a file they distribute. `build_feed_xml` enforces this
by rejecting any `FeedEpisode.filename` containing a path separator before
emitting anything.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from email.utils import format_datetime
from typing import Sequence

_ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
_ITUNES_DURATION_TAG = f"{{{_ITUNES_NS}}}duration"

# Registering the prefix (rather than leaving ElementTree to invent "ns0")
# only affects how `itunes:duration` elements are *rendered* -- it has no
# bearing on parsing, which always resolves by URI regardless of prefix.
ET.register_namespace("itunes", _ITUNES_NS)


class FeedBuildError(RuntimeError):
    """Raised when episode data cannot be turned into a valid feed.

    Currently raised for exactly one reason: an episode's `filename`
    contains a path separator, which would let the feed reference a file
    outside its own directory.
    """


@dataclass(frozen=True)
class FeedEpisode:
    """One podcast episode, shaped from a `list_watchlist_audio_episodes` row.

    Attributes:
        title: Episode title, emitted verbatim as the `<item>`'s `<title>`.
        filename: Bare relative filename of the episode's audio file within
            the feed directory (e.g. `"script-1-audio-2.wav"`). Must not
            contain a path separator (`/` or `\\`) -- :func:`build_feed_xml`
            raises :class:`FeedBuildError` if it does.
        length_bytes: Size of the audio file in bytes, emitted as the
            `<enclosure>`'s `length` attribute.
        duration_seconds: Playback duration in seconds, or `None` if
            unknown. Emitted as `<itunes:duration>` only when present.
        published: The episode's publication timestamp, emitted as the
            `<item>`'s `<pubDate>` in RFC-822 form.
        guid: A stable identifier for the episode, emitted verbatim as the
            `<item>`'s `<guid>`.
        description: Episode description text, emitted verbatim as the
            `<item>`'s `<description>`.
    """

    title: str
    filename: str
    length_bytes: int
    duration_seconds: float | None
    published: datetime
    guid: str
    description: str


def build_feed_xml(
    *,
    channel_title: str,
    channel_description: str,
    episodes: Sequence[FeedEpisode],
    now: datetime,
) -> bytes:
    """Build a self-contained RSS 2.0 podcast feed as UTF-8 encoded bytes.

    Pure and deterministic: no filesystem access, no `datetime.now()` --
    `now` is the caller's injected timestamp for the channel's
    `<lastBuildDate>`. Every episode's `filename` is validated *before* any
    XML is emitted, so a single bad episode fails the whole build rather
    than silently emitting a feed that is missing it.

    Args:
        channel_title: Feed/channel title.
        channel_description: Feed/channel description.
        episodes: Episodes to include as `<item>` elements, emitted in the
            order given. An empty sequence still produces a valid channel
            with zero items.
        now: Timestamp for the channel's `<lastBuildDate>`.

    Returns:
        The complete RSS document, UTF-8 encoded.

    Raises:
        FeedBuildError: If any episode's `filename` contains a path
            separator (`/` or `\\`) -- naming the offending filename. A feed
            must never reference outside its own directory.
    """
    for episode in episodes:
        if "/" in episode.filename or "\\" in episode.filename:
            raise FeedBuildError(
                "episode filename must be a bare relative name with no "
                f"path separator, got: {episode.filename}"
            )

    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, "title").text = channel_title
    ET.SubElement(channel, "description").text = channel_description
    ET.SubElement(channel, "lastBuildDate").text = format_datetime(now)

    for episode in episodes:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = episode.title
        ET.SubElement(item, "description").text = episode.description
        guid_el = ET.SubElement(item, "guid")
        guid_el.text = episode.guid
        guid_el.set("isPermaLink", "false")
        ET.SubElement(item, "pubDate").text = format_datetime(episode.published)

        enclosure = ET.SubElement(item, "enclosure")
        enclosure.set("url", episode.filename)
        enclosure.set("length", str(episode.length_bytes))
        enclosure.set("type", "audio/wav")

        if episode.duration_seconds is not None:
            ET.SubElement(item, _ITUNES_DURATION_TAG).text = _format_itunes_duration(
                episode.duration_seconds
            )

    return ET.tostring(rss, encoding="utf-8")


def _format_itunes_duration(duration_seconds: float) -> str:
    """Format seconds as iTunes's `HH:MM:SS` duration convention.

    Args:
        duration_seconds: Non-negative duration in seconds.

    Returns:
        `"H:MM:SS"` (hours unpadded, minutes and seconds zero-padded to two
        digits), per the iTunes podcast duration convention.
    """
    total_seconds = int(round(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"
