# monitoring_engine.py
# Description: Core monitoring engine for RSS/Atom feeds and URL change detection
#
# This module provides secure feed parsing and URL monitoring capabilities with:
# - XXE protection for XML parsing
# - Change detection for URLs
# - Rate limiting
# - Circuit breaker pattern
# - Content extraction
#
# Imports
import hashlib
import json
import re
import textwrap
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from difflib import SequenceMatcher, unified_diff

#
# Third-Party Imports
import httpx
from loguru import logger

try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

    logger.warning(
        "defusedxml not available, using standard xml.etree. Install defusedxml for better security."
    )
from bs4 import BeautifulSoup
from loguru import logger

#
# Local Imports
from ..DB.Subscriptions_DB import (
    SubscriptionsDB,
    RateLimitError,
    AuthenticationError,
    SubscriptionError,
)
from ..Metrics.metrics_logger import log_histogram, log_counter
from .item_persist import (
    CONTENT_FORMAT_DIFF,
    CONTENT_FORMAT_TEXT,
    CONTENT_KIND_ARTICLE,
    CONTENT_KIND_CHANGE,
)
from .watchlist_rule_matching import RULE_MATCH_TEXT_KEY
from ..Utils.egress import (
    EgressBlockedError,
    EgressFetchError,
    MAX_FETCH_BYTES_PAGE,
    guarded_fetch_httpx_async,
    host_of,
    origin_set,
    warn_insecure_ssl,
)
from .security import SecurityValidator
#
########################################################################################################################
#
# Core Classes
#
########################################################################################################################


class FetchBlockedError(SubscriptionError):
    """A feed/URL fetch was blocked or failed at the egress (SSRF) guard.

    Mirrors ``RateLimitError``/``AuthenticationError`` as the module's existing
    failure-exception category so callers keep catching one family of
    ``SubscriptionError`` subclasses instead of a raw egress-layer exception.
    """

    pass


class RateLimiter:
    """Token bucket algorithm for rate limiting."""

    def __init__(self, tokens_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            tokens_per_minute: Maximum requests per minute
        """
        self.rate = tokens_per_minute / 60.0  # Tokens per second
        self.max_tokens = tokens_per_minute
        self.tokens = float(self.max_tokens)
        self.last_update = time.time()
        self.domain_buckets = {}  # Per-domain rate limiting

    async def acquire_token(self, domain: str) -> bool:
        """
        Try to acquire a token for the given domain.

        Args:
            domain: The domain to rate limit

        Returns:
            True if token acquired, False if rate limited
        """
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Refill tokens
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)

        # Check if we have tokens
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True

        return False

    def get_retry_after(self) -> float:
        """Get seconds until a token will be available."""
        if self.tokens >= 1.0:
            return 0.0
        tokens_needed = 1.0 - self.tokens
        return tokens_needed / self.rate


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def can_attempt(self) -> bool:
        """Check if we can attempt an operation."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if we should try recovery
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker entering half-open state")
                    return True
            return False

        # half_open state
        return True


class ContentExtractor:
    """Extract and process content from various sources."""

    @staticmethod
    def extract_text_from_html(html: str, ignore_selectors: List[str] = None) -> str:
        """
        Extract clean text from HTML.

        Args:
            html: HTML content
            ignore_selectors: CSS selectors to ignore

        Returns:
            Extracted text
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Remove elements matching ignore selectors
        if ignore_selectors:
            for selector in ignore_selectors:
                for element in soup.select(selector):
                    element.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    @staticmethod
    def calculate_content_hash(content: str) -> str:
        """Calculate SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def calculate_change_percentage(old_content: str, new_content: str) -> float:
        """
        Calculate percentage of change between two texts.

        Args:
            old_content: Previous content
            new_content: New content

        Returns:
            Change percentage (0.0 to 1.0)
        """
        if not old_content and not new_content:
            return 0.0
        if not old_content or not new_content:
            return 1.0

        matcher = SequenceMatcher(None, old_content, new_content)
        similarity = matcher.ratio()
        return 1.0 - similarity


########################################################################################################################
#
# content_kind / content_format production (TASK-1343)
#
########################################################################################################################

# `content_pane.render_change` styles a diff line by its leading `+` or `-`, so
# what `check_url` writes to `content` is a unified-diff *body*: `+`/`-` for
# changed segments, a leading space for context, `@@` for position. The
# `---`/`+++` file headers `difflib.unified_diff` emits first are dropped
# deliberately -- they begin with `-` and `+`, so the renderer would paint them
# red and green as if the header itself were part of the change.
#
# They are dropped POSITIONALLY (see `_HEADER_LINES`), never by pattern. Fix
# round 1, Important #1: filtering `line.startswith(("---", "+++"))` also
# deletes real content, because a removed segment beginning `--` becomes
# `---...` and an added one beginning `++` becomes `+++...`. A page dropping a
# literal `--- Deprecated notice ---` banner produced a persisted change whose
# body showed nothing removed and whose headline said "0 line(s) added, 0
# removed": the stored record misrepresented the change, which is worse than a
# rendering glitch.
_DIFF_CONTEXT_SEGMENTS = 1

# `difflib.unified_diff` yields `--- <fromfile>` and `+++ <tofile>` together,
# immediately before the first hunk, or yields nothing at all -- so when there
# is any output at all they are exactly positions 0 and 1.
_HEADER_LINES = 2

# `ContentExtractor.extract_text_from_html` joins every chunk of a page with a
# single space, so the extracted text of a whole page is ONE line containing no
# newlines at all. A line-based diff of two such snapshots is therefore always
# exactly `-<the entire old page>` / `+<the entire new page>`: the full text
# twice, which is simultaneously the least readable and the largest possible
# thing to store. Both sides are re-segmented before diffing -- see
# `_segment_for_diff`, which splits on real line breaks when the text has any
# and on sentence boundaries when it does not (sentences stay aligned under a
# local edit in a way fixed-width chunking does not).
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# A segment longer than this is wrapped at word boundaries, so no single diff
# line is wider than the narrow reader pane can show without wrapping mid-word.
_MAX_DIFF_SEGMENT_CHARS = 110

# Bounds on the stored diff. The body goes into a TEXT column and into a pane
# roughly nine rows tall, and the full page it was computed from is already
# kept in `url_snapshots` (`_store_snapshot`) -- so the diff is a summary, not
# an archive, and losing its tail loses nothing recoverable. 400 lines is far
# more than a reader scrolls and ~40x the pane's height; 20,000 characters
# keeps the item row small beside the snapshot. Whichever bound is reached
# first wins, and the truncation is stated IN the body (see
# `_DIFF_TRUNCATION_NOTICE`) so a partial change is never presented as a
# complete one.
_MAX_DIFF_LINES = 400
_MAX_DIFF_CHARS = 20_000

# Both notices are worded to start with `[` rather than `+`/`-`, so the
# renderer does not colour them as though they were themselves a change.
#
# The truncation notice goes FIRST, not last (fix round 1, Important #2): as
# the 401st line of 401 in a pane about nine rows tall it was unreachable
# exactly when it mattered, so a reader saw the head of a cut-down diff with
# nothing to say it had been cut. `_DIFF_TRUNCATION_SUMMARY_SUFFIX` puts it in
# the headline too, which is on screen without any scrolling at all.
_DIFF_TRUNCATION_NOTICE = (
    "[diff truncated: showing the first {kept} of {total} diff lines "
    "(cap {max_lines} lines / {max_chars} characters). This is a partial "
    "view of the change; the full page is in this source's snapshot history.]"
)
_DIFF_TRUNCATION_SUMMARY_SUFFIX = " (diff truncated)"
_NO_TEXTUAL_CHANGE_NOTICE = (
    "[the page changed, but its extracted text is identical once whitespace "
    "is normalized -- the difference was in markup or spacing only]"
)


def _segment_for_diff(text: str) -> List[str]:
    """Split one side of a comparison into diffable, pane-width segments.

    Splits on real line breaks when ``text`` contains any, and on sentence
    boundaries when it does not -- which is the case for a whole page captured
    through ``extract_text_from_html``, since that collapses everything onto
    one line. Segments longer than ``_MAX_DIFF_SEGMENT_CHARS`` are then wrapped
    at word boundaries, and blank segments are dropped.

    (Fix round 1, Minor #4: this used to be described in terms of the
    subscription's ``extraction_method``, which it never reads -- the only
    switch is whether the text already has newlines in it. A raw-extraction
    page usually does and a text-extracted one never does, but that is a
    consequence, not the condition.)

    Args:
        text: Extracted page text, which may be a single very long line.

    Returns:
        Non-empty, whitespace-trimmed segments, none longer than
        ``_MAX_DIFF_SEGMENT_CHARS``.
    """
    source = text or ""
    units = source.splitlines() if "\n" in source else _SENTENCE_BOUNDARY.split(source)
    segments: List[str] = []
    for unit in units:
        stripped = unit.strip()
        if not stripped:
            continue
        if len(stripped) <= _MAX_DIFF_SEGMENT_CHARS:
            segments.append(stripped)
            continue
        segments.extend(textwrap.wrap(stripped, _MAX_DIFF_SEGMENT_CHARS) or [stripped])
    return segments


def build_change_diff(previous_text: str, current_text: str) -> tuple[str, str]:
    """Produce the stored diff body and its one-line summary for a site change.

    Before TASK-1343 the site-change path stored the *entire new page text* as
    the item's ``content``, which meant the reader could see what the page says
    now but never what actually changed, and the change renderer it was written
    for was never even dispatched to (nothing wrote ``content_kind``).

    Args:
        previous_text: The previous snapshot's ``extracted_content``.
        current_text: The freshly fetched extracted text.

    Returns:
        ``(diff_body, diff_summary)``. ``diff_body`` is a bounded unified-diff
        body whose changed lines start with ``+``/``-`` for
        ``content_pane.render_change`` to colour; ``diff_summary`` is a single
        line naming how many lines were added and removed, counted over the
        *whole* diff so it stays true even when the body is truncated, and
        saying so when it was.
    """
    old_segments = _segment_for_diff(previous_text)
    new_segments = _segment_for_diff(current_text)

    # The generator is consumed ONCE and never materialized (PR #1092 review,
    # Bug #1): `list(unified_diff(...))` bounded what was *stored* but left peak
    # memory proportional to the whole diff. The fetch layer admits pages up to
    # `MAX_FETCH_BYTES_PAGE` (10 MB) and 110-char segmentation turns one of
    # those into a very long segment list, so the intermediate diff of two of
    # them can be enormous -- and this runs inside a scheduled fetch, where
    # memory pressure is both least visible and least welcome. Counters are
    # accumulated as the lines go past, and iteration DELIBERATELY continues
    # after a cap is hit so `total_lines`, `added` and `removed` still describe
    # the whole change rather than the retained slice.
    kept: List[str] = []
    chars = 0
    total_lines = 0
    added = 0
    removed = 0
    truncated = False
    for index, line in enumerate(
        unified_diff(
            old_segments,
            new_segments,
            n=_DIFF_CONTEXT_SEGMENTS,
            lineterm="",
        )
    ):
        # Drop the two file headers by POSITION, never by pattern -- see
        # `_HEADER_LINES` and `_DIFF_CONTEXT_SEGMENTS` for what a pattern match
        # deletes along with them.
        if index < _HEADER_LINES:
            continue
        total_lines += 1
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
        if truncated:
            continue
        if len(kept) >= _MAX_DIFF_LINES or chars + len(line) + 1 > _MAX_DIFF_CHARS:
            truncated = True
            continue
        kept.append(line)
        chars += len(line) + 1

    if not total_lines:
        # Reachable: the content hash is taken over the raw extracted text,
        # while segmentation trims and normalizes whitespace, so a
        # whitespace-only or markup-only change hashes differently and diffs
        # to nothing. Saying so beats an empty body, which `render_change`
        # would replace with "no body captured for this item" -- a claim that
        # content was never captured, when in fact it was and it matched.
        return _NO_TEXTUAL_CHANGE_NOTICE, "no textual change after normalization"

    summary = f"{added} line(s) added, {removed} removed"
    if truncated:
        # First line, not last -- see `_DIFF_TRUNCATION_NOTICE`. And in the
        # headline as well, which needs no scrolling to reach.
        kept.insert(
            0,
            _DIFF_TRUNCATION_NOTICE.format(
                kept=len(kept),
                total=total_lines,
                max_lines=_MAX_DIFF_LINES,
                max_chars=_MAX_DIFF_CHARS,
            ),
        )
        summary += _DIFF_TRUNCATION_SUMMARY_SUFFIX
    return "\n".join(kept), summary


def classify_change_type(previous_text: str, current_text: str) -> str:
    """Name what kind of change this is, from what ``check_url`` already has.

    This replaces a hardcoded ``"content"`` literal. Only the three cases the
    two snapshots can actually distinguish are reported: the richer vocabulary
    in ``baseline_manager.ChangeReport`` also carries ``'structural'`` and
    ``'semantic'``, but those need DOM-shape and embedding analysis that
    ``check_url`` does not do, so claiming them here would be a guess. (That
    module has no importers at all and its fate is TASK-1360; it is
    deliberately not extended from here.)

    Args:
        previous_text: The previous snapshot's extracted text.
        current_text: The freshly fetched extracted text.

    Returns:
        ``"new"`` when text appeared where there was none, ``"removed"`` when
        it disappeared entirely, otherwise ``"content"``.
    """
    had_text = bool((previous_text or "").strip())
    has_text = bool((current_text or "").strip())
    if not had_text and has_text:
        return "new"
    if had_text and not has_text:
        return "removed"
    return "content"


class FeedMonitor:
    """Monitor RSS/Atom feeds with security and performance features."""

    def __init__(
        self,
        rate_limiter: RateLimiter = None,
        security_validator: SecurityValidator = None,
    ):
        """
        Initialize feed monitor.

        Args:
            rate_limiter: Rate limiter instance
            security_validator: Security validator instance
        """
        self.rate_limiter = rate_limiter or RateLimiter()
        self.security_validator = security_validator
        self.circuit_breakers = {}  # Per-subscription circuit breakers

    async def check_feed(self, subscription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check a feed for new items.

        Args:
            subscription: Subscription dictionary from database

        Returns:
            List of new/updated items
        """
        start_time = time.time()
        subscription_id = subscription["id"]
        feed_url = subscription["source"]

        # Check circuit breaker
        if subscription_id not in self.circuit_breakers:
            self.circuit_breakers[subscription_id] = CircuitBreaker()

        breaker = self.circuit_breakers[subscription_id]
        if not breaker.can_attempt():
            raise RateLimitError(
                f"Circuit breaker open for subscription {subscription_id}"
            )

        try:
            # Parse URL for rate limiting
            parsed = urlparse(feed_url)
            domain = parsed.netloc

            # Check rate limit
            if not await self.rate_limiter.acquire_token(domain):
                retry_after = self.rate_limiter.get_retry_after()
                raise RateLimitError(
                    f"Rate limited. Retry after {retry_after:.1f} seconds"
                )

            # Fetch feed
            items = await self._fetch_and_parse_feed(subscription)

            # Record success
            breaker.record_success()

            # Log metrics
            duration = time.time() - start_time
            log_histogram(
                "subscription_check_duration",
                duration,
                labels={"type": subscription["type"], "status": "success"},
            )
            log_counter(
                "subscription_checks",
                labels={"type": subscription["type"], "status": "success"},
            )

            return items

        except Exception as e:
            # Record failure
            breaker.record_failure()

            # Log metrics
            duration = time.time() - start_time
            log_histogram(
                "subscription_check_duration",
                duration,
                labels={"type": subscription["type"], "status": "error"},
            )
            log_counter(
                "subscription_checks",
                labels={
                    "type": subscription["type"],
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )

            raise

    async def _fetch_and_parse_feed(
        self, subscription: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Fetch and parse a feed.

        Args:
            subscription: Subscription dictionary

        Returns:
            List of feed items
        """
        feed_url = subscription["source"]

        # Build headers
        headers = {
            "User-Agent": "tldw-chatbook/1.0 (+https://github.com/tldw/chatbook)",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
            "Accept-Encoding": "gzip, deflate",
        }

        # Add ETag/Last-Modified if available
        if subscription.get("etag"):
            headers["If-None-Match"] = subscription["etag"]
        if subscription.get("last_modified"):
            headers["If-Modified-Since"] = subscription["last_modified"]

        # Add custom headers
        if subscription.get("custom_headers"):
            try:
                custom = json.loads(subscription["custom_headers"])
                headers.update(custom)
            except (json.JSONDecodeError, TypeError):
                pass

        # Add authentication if configured
        auth = None
        if subscription.get("auth_config"):
            try:
                auth_config = json.loads(subscription["auth_config"])
                auth_type = auth_config.get("type")

                if auth_type == "basic":
                    auth = httpx.BasicAuth(
                        auth_config.get("username", ""), auth_config.get("password", "")
                    )
                elif auth_type == "bearer":
                    headers["Authorization"] = f"Bearer {auth_config.get('token', '')}"
                elif auth_type == "api_key":
                    key_header = auth_config.get("header", "X-API-Key")
                    headers[key_header] = auth_config.get("key", "")
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"Invalid auth config for subscription {subscription['id']}"
                )

        # Fetch feed
        feed_host = host_of(feed_url)
        if subscription.get("ssl_verify", True) == 0:
            warn_insecure_ssl(feed_host)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            verify=subscription.get("ssl_verify", True) != 0,
        ) as client:
            try:
                response = await guarded_fetch_httpx_async(
                    feed_url,
                    client=client,
                    max_bytes=MAX_FETCH_BYTES_PAGE,
                    trusted_origins=origin_set(feed_url),
                    headers=headers,
                    auth=auth,
                )
            except (EgressBlockedError, EgressFetchError) as e:
                logger.warning(f"Feed fetch blocked by egress policy: {e}")
                raise FetchBlockedError(
                    f"Feed URL blocked or oversize: {e}"
                ) from e

        # Handle response
        if response.status_code == 304:
            # Not modified
            logger.info(f"Feed not modified: {feed_url}")
            return []

        if response.status_code == 401:
            raise AuthenticationError("Authentication failed")

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            raise RateLimitError(
                f"Rate limited by server. Retry after {retry_after} seconds"
            )

        response.raise_for_status()

        # Parse feed based on type
        content_type = response.headers.get("content-type", "").lower()

        if "json" in content_type or subscription["type"] == "json_feed":
            return self._parse_json_feed(response.text)
        else:
            return self._parse_xml_feed(response.text, subscription["type"])

    def _parse_xml_feed(self, content: str, feed_type: str) -> List[Dict[str, Any]]:
        """
        Parse RSS/Atom XML feed with XXE protection.

        Args:
            content: Feed XML content
            feed_type: Type of feed (rss, atom)

        Returns:
            List of parsed items
        """
        try:
            # Parse XML (with defusedxml if available for XXE protection)
            root = ET.fromstring(content)

            items = []

            if feed_type == "atom" or root.tag.endswith("feed"):
                # Atom feed
                entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
                for entry in entries:
                    item = self._parse_atom_entry(entry)
                    if item:
                        items.append(item)
            else:
                # RSS feed
                channel = root.find(".//channel")
                if channel is not None:
                    for item_elem in channel.findall("item"):
                        item = self._parse_rss_item(item_elem)
                        if item:
                            items.append(item)

            return items

        except (ET.ParseError, Exception) as e:
            logger.error(f"XML parse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing feed: {e}")
            raise

    def _parse_rss_item(self, item_elem) -> Optional[Dict[str, Any]]:
        """Parse a single RSS item."""
        try:
            item = {
                "title": self._get_text(item_elem, "title"),
                "url": self._get_text(item_elem, "link"),
                "content": self._get_text(item_elem, "description"),
                "author": self._get_text(item_elem, "author")
                or self._get_text(item_elem, "dc:creator"),
                "published_date": self._parse_date(
                    self._get_text(item_elem, "pubDate")
                ),
                "categories": [
                    cat.text for cat in item_elem.findall("category") if cat.text
                ],
                "enclosures": [],
                # TASK-1343. Every feed item is an article, and `description`
                # is whatever the publisher wrote -- plain text or HTML.
                # Nothing on this path converts it to markdown, and
                # `_VALID_PAIRINGS` allows only "text" or "markdown" for an
                # article, so "text" is what was honestly captured; claiming
                # "markdown" would make `render_article` hand publisher HTML
                # to a CommonMark parser.
                "content_kind": CONTENT_KIND_ARTICLE,
                "content_format": CONTENT_FORMAT_TEXT,
            }

            # Get enclosures
            for enclosure in item_elem.findall("enclosure"):
                enc = {
                    "url": enclosure.get("url"),
                    "type": enclosure.get("type"),
                    "length": enclosure.get("length"),
                }
                if enc["url"]:
                    item["enclosures"].append(enc)

            # Calculate content hash
            content_for_hash = f"{item['title']}{item['content']}"
            item["content_hash"] = ContentExtractor.calculate_content_hash(
                content_for_hash
            )

            return item

        except Exception as e:
            logger.error(f"Error parsing RSS item: {e}")
            return None

    def _parse_atom_entry(self, entry) -> Optional[Dict[str, Any]]:
        """Parse a single Atom entry."""
        try:
            # Define Atom namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            item = {
                "title": self._get_text(entry, "atom:title", ns),
                "url": None,
                "content": self._get_text(entry, "atom:content", ns)
                or self._get_text(entry, "atom:summary", ns),
                "author": None,
                "published_date": self._parse_date(
                    self._get_text(entry, "atom:published", ns)
                ),
                "categories": [],
                "enclosures": [],
                # TASK-1343, same reasoning as `_parse_rss_item`: `atom:content`
                # arrives as the publisher's text or HTML, unconverted.
                "content_kind": CONTENT_KIND_ARTICLE,
                "content_format": CONTENT_FORMAT_TEXT,
            }

            # Get link
            link = entry.find('atom:link[@rel="alternate"]', ns)
            if link is None:
                link = entry.find("atom:link", ns)
            if link is not None:
                item["url"] = link.get("href")

            # Get author
            author = entry.find("atom:author", ns)
            if author is not None:
                item["author"] = self._get_text(author, "atom:name", ns)

            # Get categories
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term")
                if term:
                    item["categories"].append(term)

            # Calculate content hash
            content_for_hash = f"{item['title']}{item['content']}"
            item["content_hash"] = ContentExtractor.calculate_content_hash(
                content_for_hash
            )

            return item

        except Exception as e:
            logger.error(f"Error parsing Atom entry: {e}")
            return None

    def _parse_json_feed(self, content: str) -> List[Dict[str, Any]]:
        """Parse JSON Feed format."""
        try:
            feed = json.loads(content)
            items = []

            for feed_item in feed.get("items", []):
                item = {
                    "title": feed_item.get("title", "Untitled"),
                    "url": feed_item.get("url") or feed_item.get("external_url"),
                    "content": feed_item.get("content_html")
                    or feed_item.get("content_text", ""),
                    "author": None,
                    "published_date": self._parse_date(feed_item.get("date_published")),
                    "categories": feed_item.get("tags", []),
                    "enclosures": [],
                    # TASK-1343. JSON Feed's `content_html` is HTML by
                    # definition and `content_text` is plain; neither is
                    # markdown and nothing converts them, so "text" is the
                    # honest format for both.
                    "content_kind": CONTENT_KIND_ARTICLE,
                    "content_format": CONTENT_FORMAT_TEXT,
                }

                # Get author
                if "author" in feed_item:
                    item["author"] = feed_item["author"].get("name")
                elif "authors" in feed_item and feed_item["authors"]:
                    item["author"] = feed_item["authors"][0].get("name")

                # Get attachments
                for attachment in feed_item.get("attachments", []):
                    enc = {
                        "url": attachment.get("url"),
                        "type": attachment.get("mime_type"),
                        "length": attachment.get("size_in_bytes"),
                    }
                    if enc["url"]:
                        item["enclosures"].append(enc)

                # Calculate content hash
                content_for_hash = f"{item['title']}{item['content']}"
                item["content_hash"] = ContentExtractor.calculate_content_hash(
                    content_for_hash
                )

                items.append(item)

            return items

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing JSON feed: {e}")
            raise

    def _get_text(
        self, elem, tag: str, namespaces: Dict[str, str] = None
    ) -> Optional[str]:
        """Safely get text from XML element."""
        if elem is None:
            return None

        child = elem.find(tag, namespaces)
        if child is not None and child.text:
            return child.text.strip()
        return None

    def _parse_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse various date formats to ISO format."""
        if not date_str:
            return None

        # Common date formats
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 822
            "%a, %d %b %Y %H:%M:%S %Z",  # RFC 822 with timezone name
            "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 UTC
            "%Y-%m-%d %H:%M:%S",  # Simple format
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.isoformat()
            except ValueError:
                continue

        # If no format matches, return the original string
        logger.warning(f"Could not parse date: {date_str}")
        return date_str


class URLMonitor:
    """Monitor URLs for changes."""

    def __init__(
        self,
        db: SubscriptionsDB,
        rate_limiter: RateLimiter = None,
        *,
        persist_snapshots: bool = True,
    ):
        """
        Initialize URL monitor.

        Args:
            db: Subscriptions database instance
            rate_limiter: Rate limiter instance
            persist_snapshots: When ``False``, fetched snapshots are compared but
                not written to ``url_snapshots``. Useful for shadow/dry-run mode.
        """
        self.db = db
        self.rate_limiter = rate_limiter or RateLimiter()
        self.circuit_breakers = {}
        self.persist_snapshots = persist_snapshots

    async def check_url(self, subscription: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check a URL for changes.

        Args:
            subscription: Subscription dictionary

        Returns:
            Change information if changed, None otherwise
        """
        subscription_id = subscription["id"]
        url = subscription["source"]

        # Check circuit breaker
        if subscription_id not in self.circuit_breakers:
            self.circuit_breakers[subscription_id] = CircuitBreaker()

        breaker = self.circuit_breakers[subscription_id]
        if not breaker.can_attempt():
            raise RateLimitError(
                f"Circuit breaker open for subscription {subscription_id}"
            )

        try:
            # Fetch current content
            current_content = await self._fetch_url_content(subscription)

            # Get previous snapshot
            cursor = self.db.conn.cursor()
            cursor.execute(
                """
                SELECT content_hash, extracted_content
                FROM url_snapshots
                WHERE subscription_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (subscription_id,),
            )

            previous = cursor.fetchone()

            if not previous:
                # First check - store baseline
                await self._store_snapshot(subscription_id, url, current_content)
                breaker.record_success()
                return None

            # Calculate change
            current_hash = ContentExtractor.calculate_content_hash(
                current_content["text"]
            )

            if current_hash == previous["content_hash"]:
                # No change
                breaker.record_success()
                return None

            # Calculate change details
            previous_text = previous["extracted_content"] or ""
            change_percentage = ContentExtractor.calculate_change_percentage(
                previous_text, current_content["text"]
            )

            # Check if change exceeds threshold. Both sides of this comparison
            # are 0.0-1.0 ratios (`change_threshold` defaults to 0.1, i.e. 10%)
            # -- the scaling to a percentage happens only where the value is
            # handed to the reader, below.
            threshold = subscription.get("change_threshold", 0.1)
            if change_percentage < threshold:
                # Change too small
                breaker.record_success()
                return None

            # Significant change detected. TASK-1343: `content` is the DIFF, not
            # the new page. The full page continues to live in `url_snapshots`
            # (`_store_snapshot`, immediately below), which is where the
            # reader's `[full page]` / `[previous snapshot]` affordances read
            # from; storing it a second time here bought nothing and left the
            # reader unable to see what had actually changed.
            diff_body, diff_summary = build_change_diff(
                previous_text, current_content["text"]
            )
            change_info = {
                "type": "url_change",
                "url": url,
                "title": f"Change detected: {subscription['name']}",
                "content": diff_body,
                # Without these two, `content_pane.render_for` fell through to
                # the article renderer for every site change ever detected, and
                # `render_change` was unreachable in production.
                "content_kind": CONTENT_KIND_CHANGE,
                "content_format": CONTENT_FORMAT_DIFF,
                "content_hash": current_hash,
                "previous_hash": previous["content_hash"],
                # Scaled to a percentage, as the column name
                # (`change_percentage`), the reader's headline
                # (`f"{float(pct):.0f}% changed"`) and every renderer test
                # fixture all read it. `calculate_change_percentage` returns a
                # 0.0-1.0 ratio, so before TASK-1343 made the change renderer
                # reachable at all, a real 35% change would have displayed as
                # "0% changed" and a total rewrite as "1% changed".
                "change_percentage": change_percentage * 100.0,
                "change_type": classify_change_type(
                    previous_text, current_content["text"]
                ),
                "diff_summary": diff_summary,
                "published_date": datetime.now(timezone.utc).isoformat(),
                # Filters and content-alert rules are evaluated on the raw
                # fetched item, BEFORE persistence, and their haystack was
                # built from `content`. With `content` now a diff, a rule that
                # had matched a phrase anywhere on the page would silently have
                # narrowed to "matches a changed segment" -- a user's alert
                # quietly stopping after months, with nothing on screen to
                # explain it. So the full page text travels alongside the diff
                # for matching only; it is not a persisted column (see
                # `watchlist_rule_matching`).
                RULE_MATCH_TEXT_KEY: current_content["text"],
            }

            # Store new snapshot
            await self._store_snapshot(
                subscription_id, url, current_content, current_hash
            )

            breaker.record_success()
            return change_info

        except Exception:
            breaker.record_failure()
            raise

    async def _fetch_url_content(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch content from a URL.

        Args:
            subscription: Subscription dictionary

        Returns:
            Dictionary with content and metadata
        """
        url = subscription["source"]

        # Parse URL for rate limiting
        parsed = urlparse(url)
        domain = parsed.netloc

        # Check rate limit
        if not await self.rate_limiter.acquire_token(domain):
            retry_after = self.rate_limiter.get_retry_after()
            raise RateLimitError(f"Rate limited. Retry after {retry_after:.1f} seconds")

        # Build headers
        headers = {
            "User-Agent": "tldw-chatbook/1.0 (+https://github.com/tldw/chatbook)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        }

        # Add custom headers
        if subscription.get("custom_headers"):
            try:
                custom = json.loads(subscription["custom_headers"])
                headers.update(custom)
            except (json.JSONDecodeError, TypeError):
                pass

        # Fetch content
        url_host = host_of(url)
        if subscription.get("ssl_verify", True) == 0:
            warn_insecure_ssl(url_host)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            verify=subscription.get("ssl_verify", True) != 0,
        ) as client:
            try:
                response = await guarded_fetch_httpx_async(
                    url,
                    client=client,
                    max_bytes=MAX_FETCH_BYTES_PAGE,
                    trusted_origins=origin_set(url),
                    headers=headers,
                )
            except (EgressBlockedError, EgressFetchError) as e:
                logger.warning(f"URL fetch blocked by egress policy: {e}")
                raise FetchBlockedError(f"URL blocked or oversize: {e}") from e
            response.raise_for_status()

        # Extract content based on extraction method
        extraction_method = subscription.get("extraction_method", "auto")
        ignore_selectors = None

        if subscription.get("ignore_selectors"):
            ignore_selectors = [
                s.strip()
                for s in subscription["ignore_selectors"].split("\n")
                if s.strip()
            ]

        if extraction_method == "full" or extraction_method == "auto":
            # Extract text from HTML
            text = ContentExtractor.extract_text_from_html(
                response.text, ignore_selectors
            )
        else:
            # Raw content
            text = response.text

        return {
            "text": text,
            "html": response.text,
            "headers": dict(response.headers),
            "status_code": response.status_code,
        }

    async def _store_snapshot(
        self,
        subscription_id: int,
        url: str,
        content: Dict[str, Any],
        content_hash: str = None,
    ) -> None:
        """Store a URL snapshot."""
        if not self.persist_snapshots:
            return
        if not content_hash:
            content_hash = ContentExtractor.calculate_content_hash(content["text"])

        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO url_snapshots
                (subscription_id, url, content_hash, extracted_content, raw_html, headers)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    subscription_id,
                    url,
                    content_hash,
                    content["text"],
                    content["html"],
                    json.dumps(content["headers"]),
                ),
            )


# End of monitoring_engine.py
