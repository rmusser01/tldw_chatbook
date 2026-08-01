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

**Enclosure URLs are percent-encoded at emission -- the on-disk filename is
not.** Whole-branch review: `FeedEpisode.filename` comes from Task 4's
`safe_export_stem`, which deliberately KEEPS spaces (pinned by
`test_stem_keeps_ordinary_characters` for its other caller, the Markdown
export's save-dialog filename) and, via `str.isalnum()`, also admits
non-ASCII letters. A raw space in a URI reference is forbidden by RFC 3986;
a podcast client that sends it in the request line verbatim rather than
percent-encoding it on resolve gets a 400 and the episode never downloads.
The fix is applied ONLY where the filename becomes a URL -- `urllib.parse.
quote` wraps `episode.filename` when it is set as the `enclosure`'s `url`
attribute below -- never to `FeedEpisode.filename` itself or to any file
`Task 4` writes to disk: every static file server (including stdlib
`http.server`, the one this app's docs point users at) unquotes a request
path before looking up the file, so the file keeps its human-readable,
space-containing name on disk and the feed stays a valid URI reference.
`safe_export_stem` itself is untouched -- its space-keeping behavior is
correct for its other caller.

**All timestamps must be timezone-aware.** `email.utils.format_datetime`
does *not* treat a naive `datetime` as local time -- it emits the correct
wall-clock digits tagged `-0000` (RFC 2822's "unverified offset"). The real
hazard is one hop downstream: `-0000` round-trips through
`email.utils.parsedate_to_datetime` into a *naive* result, and the common
`dt.astimezone()` idiom then silently reinterprets those same digits as
local time -- same digits, wrong instant. `build_feed_xml` forecloses this
by rejecting a naive `now` or a naive `episode.published` outright rather
than quietly coercing to UTC (a caller passing naive input has a bug worth
surfacing, not hiding). **For Task 4:** SQLite's `CURRENT_TIMESTAMP` column
is UTC but comes back as a naive string -- attach `timezone.utc` explicitly
when parsing it into a `datetime` for this module.

**`<channel>` carries a `<link>`.** RSS 2.0 requires `title`, `link` and
`description` on every channel, and validators (and Apple's podcast
requirements) flag its absence. There is no server URL to point at --
serving the exported folder is the user's own choice (see the Docs/User
Guide's own "serve it yourself" note), so this module does not invent one.
`<link>` is set to `"feed.xml"`, the feed's own filename, as a relative
reference -- consistent with the whole directory being self-contained
(episodes are already referenced by bare relative filename, above). This
must match `briefing_export._FEED_XML_NAME`, the literal name Task 4 writes
the returned bytes to.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from email.utils import format_datetime
from typing import Sequence
from urllib.parse import quote

_ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
_ITUNES_DURATION_TAG = f"{{{_ITUNES_NS}}}duration"

# Registering the prefix (rather than leaving ElementTree to invent "ns0")
# only affects how `itunes:duration` elements are *rendered* -- it has no
# bearing on parsing, which always resolves by URI regardless of prefix.
ET.register_namespace("itunes", _ITUNES_NS)


class FeedBuildError(RuntimeError):
    """Raised when episode data cannot be turned into a valid feed.

    Raised when an episode's `filename` contains a path separator (which
    would let the feed reference a file outside its own directory), or when
    `now` or an episode's `published` is a naive `datetime` (no `tzinfo`) --
    see the module docstring's "All timestamps must be timezone-aware"
    section for why a naive input is rejected rather than coerced.
    """


@dataclass(frozen=True)
class FeedEpisode:
    """One podcast episode, shaped from a `list_watchlist_audio_episodes` row.

    Attributes:
        title: Episode title, emitted verbatim as the `<item>`'s `<title>`.
        filename: Bare relative filename of the episode's audio file within
            the feed directory (e.g. `"script-1-audio-2.wav"`, or
            `"Two Host Debate-42.wav"` -- Task 4's `safe_export_stem`
            deliberately keeps spaces). Must not contain a path separator
            (`/` or `\\`) -- :func:`build_feed_xml` raises
            :class:`FeedBuildError` if it does. Stored and emitted to disk
            exactly as given; :func:`build_feed_xml` percent-encodes a
            COPY of it for the `<enclosure>`'s `url` attribute only (see
            the module docstring) -- this field itself is never
            percent-encoded.
        length_bytes: Size of the audio file in bytes, emitted as the
            `<enclosure>`'s `length` attribute.
        duration_seconds: Playback duration in seconds, or `None` if
            unknown. Emitted as `<itunes:duration>` only when present.
        published: The episode's publication timestamp, emitted as the
            `<item>`'s `<pubDate>` in RFC-822 form. Must be timezone-aware --
            :func:`build_feed_xml` raises :class:`FeedBuildError` for a
            naive value (see the module docstring).
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
    `<lastBuildDate>`. Every episode's `filename` and every timestamp is
    validated *before* any XML is emitted, so a single bad episode fails the
    whole build rather than silently emitting a feed that is missing it.

    Args:
        channel_title: Feed/channel title.
        channel_description: Feed/channel description.
        episodes: Episodes to include as `<item>` elements, emitted in the
            order given. An empty sequence still produces a valid channel
            with zero items.
        now: Timestamp for the channel's `<lastBuildDate>`. Must be
            timezone-aware.

    Returns:
        The complete RSS document, UTF-8 encoded.

    Raises:
        FeedBuildError: If `now` is a naive `datetime` (naming `now`); if
            any episode's `published` is a naive `datetime` (naming the
            episode by its `guid`); or if any episode's `filename` contains
            a path separator (`/` or `\\`, naming the offending filename) --
            a feed must never reference outside its own directory.
    """
    if now.tzinfo is None:
        raise FeedBuildError(
            "'now' must be timezone-aware (tzinfo is None); a naive value "
            "would emit an RFC-2822 '-0000' (unverified offset) that later "
            "round-trips into a naive datetime and silently shifts under "
            "`.astimezone()` -- attach `timezone.utc` explicitly."
        )

    for episode in episodes:
        if episode.published.tzinfo is None:
            raise FeedBuildError(
                f"episode {episode.guid!r}'s `published` must be "
                "timezone-aware (tzinfo is None); a naive value would emit "
                "an RFC-2822 '-0000' (unverified offset) that later "
                "round-trips into a naive datetime and silently shifts "
                "under `.astimezone()` -- attach `timezone.utc` explicitly."
            )
        if "/" in episode.filename or "\\" in episode.filename:
            raise FeedBuildError(
                "episode filename must be a bare relative name with no "
                f"path separator, got: {episode.filename}"
            )

    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, "title").text = channel_title
    # RFC 2822/RSS-2.0 conventional order is title, link, description. A
    # relative reference to the feed's own filename -- there is no server
    # URL to point at; see the module docstring's "<channel> carries a
    # <link>" section. Must match `briefing_export._FEED_XML_NAME`.
    ET.SubElement(channel, "link").text = "feed.xml"
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
        # Percent-encode HERE ONLY -- `episode.filename` itself (and the
        # file Task 4 writes to disk under that exact name) is never
        # touched. `safe_export_stem` deliberately keeps spaces and,
        # via `str.isalnum()`, admits non-ASCII letters too; RFC 3986
        # forbids a raw space in a URI reference, so a client that does
        # not percent-encode on resolve would send it verbatim and get a
        # 400. `safe=""` encodes every character outside the unreserved
        # set (the guard above already rules out `/`, so there is nothing
        # this would need to leave alone).
        enclosure.set("url", quote(episode.filename, safe=""))
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
