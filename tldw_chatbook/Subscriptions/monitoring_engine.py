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
import asyncio
import hashlib
import json
import re
import textwrap
import time
from collections import Counter
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
from .db_offload import run_db_off_loop
from .item_persist import (
    CONTENT_FORMAT_DIFF,
    CONTENT_FORMAT_TEXT,
    CONTENT_KIND_ARTICLE,
    CONTENT_KIND_CHANGE,
)
from .noise_defaults import extraction_fingerprint, selector_parse_errors
from .watchlist_rule_matching import (
    RULE_MATCH_ADDED_TEXT_KEY,
    RULE_MATCH_REMOVED_TEXT_KEY,
    RULE_MATCH_TEXT_KEY,
)
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

# bs4 is extras-only (`[subscriptions]` among others) while this module is
# imported eagerly at boot via the scheduler handlers (app.py ->
# Scheduling/scheduler/handlers -> here), so a module-level
# `from bs4 import BeautifulSoup` made an install without the extra unable to
# import the app at all, and made every install pay the bs4+soupsieve import
# at boot (TASK-21104). BeautifulSoup is therefore resolved lazily at first
# HTML extraction; a missing install degrades to a per-check ImportError that
# the monitors' existing exception handling records against the subscription.
_BS4_INSTALL_HINT = (
    "beautifulsoup4 is required to extract text from HTML for "
    "watchlist/subscription monitoring, but it is not installed. "
    "Install it with: pip install tldw_chatbook[subscriptions]"
)


def _require_beautifulsoup() -> type:
    """Resolve the ``BeautifulSoup`` class lazily (TASK-21104).

    Returns:
        The ``bs4.BeautifulSoup`` class.

    Raises:
        ImportError: When beautifulsoup4 is not installed; the message names
            the feature and the exact install command so the per-check error
            surfaced on the subscription is actionable.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(_BS4_INSTALL_HINT) from exc
    return BeautifulSoup


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
        soup = _require_beautifulsoup()(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Remove elements matching ignore selectors.
        #
        # The noise filter must never break the thing it filters. Selectors are
        # user-typed (the create form and the Inspector both edit them), and
        # `soup.select` RAISES on anything CSS cannot parse -- so before this
        # guard one mistyped line aborted the whole URL check for that source,
        # every check, until the user guessed which line was bad. A bad line
        # may cost its own stripping and nothing more.
        if ignore_selectors:
            for selector in ignore_selectors:
                try:
                    matches = soup.select(selector)
                # `selector_parse_errors()` is lru_cached and shared with the
                # two UI save-path validators (`noise_defaults`) so the
                # definition of "the selector is malformed" cannot drift.
                # Called here rather than resolved at module import because it
                # probes soupsieve, which is extras-only like bs4 (TASK-21104).
                except selector_parse_errors() as exc:
                    # One line per bad selector per extraction: named, so the
                    # log says which rule to fix, not merely that one is broken.
                    logger.warning(
                        f"Skipping unparseable ignore selector {selector!r}: {exc}"
                    )
                    continue
                for element in matches:
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
    def calculate_change_percentage(
        old_content: str,
        new_content: str,
        *,
        old_segments: List[str] | None = None,
        new_segments: List[str] | None = None,
    ) -> float:
        """Estimate how much of a page's text changed, as a 0.0-1.0 ratio.

        Computed over ``_segment_for_diff`` segments -- the same
        sentence/line-sized units the stored diff body, ``diff_summary`` and
        ``added_and_removed_text`` are built from -- so the ratio means "the
        fraction of the page's segment content that appeared or disappeared"
        and agrees in granularity with everything else the reader sees about
        the change. The ratio is order-INSENSITIVE: a page that merely
        reorders its segments reports 0.0 -- see ``_segment_change_ratio``
        for that semantic decision and its rationale, and for the O(n) cost.

        TASK-16839. This was a character-level
        ``SequenceMatcher(None, old, new).ratio()`` with default autojunk,
        which had two entangled failure regimes over the unbounded inputs the
        10 MB fetch cap admits: for large Latin pages autojunk junks the
        entire alphabet and the value degenerates (a 5%-edited ~128 KB page
        measured pct=0.47 -- the 15764 review measured a full 1.0 -- while
        taking ~39 s), and for large character repertoires (CJK) nothing is
        junked and ``ratio()`` is quadratic (4x per doubling; ~7 minutes at
        the fetch cap, on a GIL-holding worker thread). Its consumers -- the
        ``change_threshold`` withhold comparison (default 0.0), the withheld
        disposition, and the reader's ``f"{pct:.0f}% changed"`` headline --
        all need a coarse, monotonic-ish magnitude, not char-exact ratios.

        Args:
            old_content: Previous extracted text.
            new_content: New extracted text.
            old_segments: ``old_content`` already run through
                ``_segment_for_diff``. Optional: segmentation is ~95% of this
                function's cost, so ``check_url`` segments each side once and
                shares the lists between this ratio and the significant-change
                details -- the same segment-once rule ``build_change_diff``
                documents. Segmented here when omitted, so callers passing
                only the texts are unchanged.
            new_segments: ``new_content`` likewise.

        Returns:
            Change ratio, 0.0 (identical) to 1.0 (nothing in common).
            Identical texts return 0.0; a whitespace-only difference also
            returns 0.0 (segmentation normalizes whitespace, deliberately
            agreeing with the "no textual change after normalization" path
            of ``build_change_diff``); one side empty returns 1.0.
        """
        if not old_content and not new_content:
            return 0.0
        if not old_content or not new_content:
            return 1.0

        if old_segments is None:
            old_segments = _segment_for_diff(old_content)
        if new_segments is None:
            new_segments = _segment_for_diff(new_content)
        return _segment_change_ratio(old_segments, new_segments)


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
#
# The second alternative splits AFTER a CJK sentence ender with no whitespace
# requirement (TASK-16839): CJK prose ends sentences with 。！？ and contains
# no spaces at all, so under the Latin-only rule an entire CJK page was ONE
# unit that fell to fixed-width wrapping -- every boundary after an edit
# shifted, which made the diff (and the change percentage now computed on the
# same segments) treat half the page as changed for a one-sentence edit.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[。．！？])")

# A segment longer than this is wrapped at word boundaries, so no single diff
# line is wider than the narrow reader pane can show without wrapping mid-word.
_MAX_DIFF_SEGMENT_CHARS = 110

# `textwrap.wrap` is quadratic in the length of a single unbreakable run:
# `_handle_long_word` re-slices the whole remainder once per emitted line, so
# one 3.4M-char spaceless unit (a 10 MB CJK page without sentence enders)
# costs minutes (TASK-16839). A unit containing any whitespace-free run longer
# than this is fixed-sliced to `_MAX_DIFF_SEGMENT_CHARS` instead -- O(n), and
# for a fully unbreakable unit the slices are exactly what `break_long_words`
# would have produced anyway. (Hyphens sometimes let textwrap break a
# spaceless run cheaply, but only between word characters -- whitespace is
# the only breaker this guard can trust.) Runs at or under this bound keep
# textwrap's word-boundary aesthetics at a bounded cost: each run re-slices
# at most ~1000 chars per emitted line.
_UNWRAPPABLE_RUN = re.compile(r"\S{1001,}")

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
    one line. Sentence boundaries include CJK enders with no trailing
    whitespace (see ``_SENTENCE_BOUNDARY``). Segments longer than
    ``_MAX_DIFF_SEGMENT_CHARS`` are then wrapped at word boundaries -- or
    fixed-sliced when they contain a run textwrap could only break
    quadratically (see ``_UNWRAPPABLE_RUN``) -- and blank segments are
    dropped.

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
        if _UNWRAPPABLE_RUN.search(stripped):
            # Bounded fixed-width slicing for units textwrap cannot break at
            # word boundaries anyway -- see `_UNWRAPPABLE_RUN` for why wrap
            # is quadratic on these.
            segments.extend(
                piece
                for start in range(0, len(stripped), _MAX_DIFF_SEGMENT_CHARS)
                if (piece := stripped[start : start + _MAX_DIFF_SEGMENT_CHARS].strip())
            )
            continue
        segments.extend(textwrap.wrap(stripped, _MAX_DIFF_SEGMENT_CHARS) or [stripped])
    return segments


def _segment_change_ratio(old_segments: List[str], new_segments: List[str]) -> float:
    """Order-insensitive change ratio between two segment lists. O(n), always.

    The fraction of segment occurrences (counting multiplicity) present on
    only one side: ``1 - 2*matches/total``, difflib's ``quick_ratio`` formula
    over segments.

    **Semantic decision (TASK-16839 fix round): a segment that merely MOVED
    is not a change.** A purely reordered page -- same segments, shuffled --
    reports 0.0 at every size. Rationale: this ratio's consumers (the
    ``change_threshold`` withhold comparison, the withheld disposition, the
    reader's "N% changed" headline) all read it as "how much of the page's
    content changed", and a re-sorted listing page whose content is intact is
    exactly the noise shape a raised threshold exists to withhold -- reporting
    near-100% for zero content change is the misleading-percentage defect
    this task exists to eliminate. Order is not lost to the reader: the
    stored diff body is position-aware and shows moved blocks as ``-``/``+``
    pairs, and a pure reorder is still *detected* (the content hash differs),
    so at the default threshold 0.0 it still produces an item -- headlined
    "0% changed", with the moves visible in the diff.

    Why one mechanism at every size, rather than an order-sensitive alignment
    below a cost bound: the reviewed revision ran a ``SequenceMatcher``
    alignment up to 4,000 total segments with this multiset ratio as the
    past-the-bound fallback, and the review reproduced the resulting cliff --
    a pure reorder reported 0.9925 at 4,000 total segments and 0.0000 at
    4,002, so one added sentence per side flipped "99% changed" to "0%
    changed". Any hard boundary between an order-sensitive and an
    order-insensitive tier flips like that on reorder-shaped edits, and
    order-sensitivity at *every* size is the unaffordable-quadratic shape
    this task retired -- so the order-insensitive ratio is the sole
    mechanism, and the reported quantity is continuous in page size for a
    fixed edit shape. For non-move edit shapes (in-place rewrites,
    insertions, deletions) this agrees with what the alignment tier
    reported anyway -- the review measured a scattered 5% edit at
    0.050000/0.049975 across that boundary -- so ordinary pages are
    unaffected by the tier's retirement; only moves are now consistently
    "not a content change" instead of order-dependently either.

    Args:
        old_segments: Previous text through ``_segment_for_diff``.
        new_segments: Current text likewise.

    Returns:
        0.0 (identical multisets) .. 1.0 (disjoint). Both sides empty is
        0.0; exactly one side empty is 1.0.
    """
    total = len(old_segments) + len(new_segments)
    if not old_segments or not new_segments:
        return 0.0 if total == 0 else 1.0

    old_counts = Counter(old_segments)
    new_counts = Counter(new_segments)
    matches = sum(
        min(count, new_counts.get(segment, 0)) for segment, count in old_counts.items()
    )
    return 1.0 - (2.0 * matches / total)


def build_change_diff(
    previous_text: str,
    current_text: str,
    *,
    old_segments: List[str] | None = None,
    new_segments: List[str] | None = None,
) -> tuple[str, str]:
    """Produce the stored diff body and its one-line summary for a site change.

    Before TASK-1343 the site-change path stored the *entire new page text* as
    the item's ``content``, which meant the reader could see what the page says
    now but never what actually changed, and the change renderer it was written
    for was never even dispatched to (nothing wrote ``content_kind``).

    Args:
        previous_text: The previous snapshot's ``extracted_content``.
        current_text: The freshly fetched extracted text.
        old_segments: ``previous_text`` already run through ``_segment_for_diff``.
            Optional: ``check_url`` also feeds the same segments to
            ``added_and_removed_text`` for the "appeared"/"disappeared" scopes
            (TASK-1363), so segmenting each side once and passing the result to
            both avoids a redundant pass over pages up to 10 MB (Qodo). Segmented
            here when omitted, so every other caller is unchanged.
        new_segments: ``current_text`` likewise.

    Returns:
        ``(diff_body, diff_summary)``. ``diff_body`` is a bounded unified-diff
        body whose changed lines start with ``+``/``-`` for
        ``content_pane.render_change`` to colour; ``diff_summary`` is a single
        line naming how many lines were added and removed, counted over the
        *whole* diff so it stays true even when the body is truncated, and
        saying so when it was.
    """
    if old_segments is None:
        old_segments = _segment_for_diff(previous_text)
    if new_segments is None:
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


def added_and_removed_text(
    previous_text: str,
    current_text: str,
    *,
    old_segments: List[str] | None = None,
    new_segments: List[str] | None = None,
) -> tuple[str, str]:
    """Split a site change into its added and removed text (TASK-1363).

    Feeds a content-alert rule scoped to "appeared" or "disappeared" (see
    ``watchlist_rule_matching.build_rule_haystack``) rather than the diff
    body ``build_change_diff`` renders for the reader -- the two exist for
    different consumers, but reuse the same ``_segment_for_diff``
    segmentation so "added"/"removed" line up with what the reader's diff
    pane shows, rather than a raw character-level diff that could slice a
    matched phrase mid-word.

    Args:
        previous_text: The previous snapshot's ``extracted_content``.
        current_text: The freshly fetched extracted text.
        old_segments: ``previous_text`` already segmented; ``new_segments``
            likewise. Optional, and shared with ``build_change_diff`` by
            ``check_url`` so the same page is segmented once, not twice (Qodo).
            Segmented here when omitted, so callers passing only the texts are
            unchanged.
        new_segments: See ``old_segments``.

    Returns:
        ``(added, removed)``: the new-side segments of every ``insert`` and
        ``replace`` opcode, and the old-side segments of every ``delete`` and
        ``replace`` opcode, each joined by a single space (matching the
        joining `build_rule_haystack` already uses). Either half is the empty
        string when nothing was added, or nothing was removed, respectively.
    """
    if old_segments is None:
        old_segments = _segment_for_diff(previous_text)
    if new_segments is None:
        new_segments = _segment_for_diff(current_text)
    matcher = SequenceMatcher(None, old_segments, new_segments)

    added: List[str] = []
    removed: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            added.extend(new_segments[j1:j2])
        if tag in ("delete", "replace"):
            removed.extend(old_segments[i1:i2])

    return " ".join(added), " ".join(removed)


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


def _change_percentage_with_segments(
    previous_text: str, current_text: str
) -> tuple[float, List[str], List[str]]:
    """Segment both sides once and compute the change ratio over the segments.

    ``check_url``'s percentage hop. Segmentation is ~95% of the ratio's cost
    (measured ~200 ms per side of a ~430 ms total at the 10 MB fetch cap;
    the multiset ratio itself is cheap), so the segment lists are returned for the
    details hop to reuse rather than re-segmenting -- extending the
    segment-once rule ``build_change_diff`` documents across the two
    ``asyncio.to_thread`` hops as well as within the second (TASK-16839 fix
    round, review finding 2: a significant change previously paid full
    segmentation twice end-to-end).
    """
    old_segments = _segment_for_diff(previous_text)
    new_segments = _segment_for_diff(current_text)
    percentage = ContentExtractor.calculate_change_percentage(
        previous_text,
        current_text,
        old_segments=old_segments,
        new_segments=new_segments,
    )
    return percentage, old_segments, new_segments


def _build_significant_change_details(
    previous_text: str,
    current_text: str,
    *,
    old_segments: List[str] | None = None,
    new_segments: List[str] | None = None,
) -> tuple[str, str, str, str, str]:
    """Build all significant-change diff details from one segmentation pass.

    Args:
        previous_text: The previous snapshot's ``extracted_content``.
        current_text: The freshly fetched extracted text.
        old_segments: ``previous_text`` already segmented; ``new_segments``
            likewise. Optional: ``check_url`` passes the lists its percentage
            hop already built (``_change_percentage_with_segments``), so a
            significant change segments each side once end-to-end. Segmented
            here when omitted, so direct callers are unchanged.
        new_segments: See ``old_segments``.
    """
    if old_segments is None:
        old_segments = _segment_for_diff(previous_text)
    if new_segments is None:
        new_segments = _segment_for_diff(current_text)
    diff_body, diff_summary = build_change_diff(
        previous_text,
        current_text,
        old_segments=old_segments,
        new_segments=new_segments,
    )
    added_text, removed_text = added_and_removed_text(
        previous_text,
        current_text,
        old_segments=old_segments,
        new_segments=new_segments,
    )
    change_type = classify_change_type(previous_text, current_text)
    return diff_body, diff_summary, added_text, removed_text, change_type


########################################################################################################################
#
# Check dispositions (TASK-1362, spec §4)
#
########################################################################################################################

#: A fresh snapshot was written and no comparison was made -- either the first
#: check of this URL or a re-baseline after an extraction-settings change. The
#: ``reason`` says which.
DISPOSITION_BASELINE_STORED = "baseline_stored"
#: The extracted text hashed identically to the previous snapshot.
DISPOSITION_UNCHANGED = "unchanged"
#: A real change, deliberately not reported because it fell under the source's
#: ``change_threshold``. Carries the percentage it measured.
DISPOSITION_WITHHELD = "withheld_below_threshold"
#: A change was detected and an item produced.
DISPOSITION_CHANGED = "changed"
#: `check_url` itself never returned -- it raised (timeout, SSRF block, HTTP
#: error, ...). Not one of the four outcomes above: those are `check_url`'s
#: own dispositions for a call that COMPLETED, while this one is synthesized
#: by the caller (`local_watchlists_service._default_run_executor`'s
#: `url_list`/`sitemap` loops) around a call that did not (task-1394). One
#: dead URL among many must not fail the whole run and discard what the
#: other URLs already collected; this is how that partial failure stays
#: visible instead of silently vanishing into "0 found".
DISPOSITION_ERROR = "error"
#: `check_url` was never called at all: another check of this exact
#: (subscription, url) pair was already in flight, so the entrant skipped
#: rather than double-checking (task-16838 -- a scheduled check and a UI
#: "Check Now" of the same source can otherwise interleave across the
#: network await and each report the same page change once, writing two
#: snapshots). Like `DISPOSITION_ERROR`, this is synthesized by the caller
#: (`local_watchlists_service._check_url_guarded`), never by `check_url`
#: itself. The concurrent check that held the claim reports the real
#: outcome; this disposition only records, honestly, that this run did not
#: check the URL.
DISPOSITION_SKIPPED_IN_FLIGHT = "skipped_in_flight"

#: ``reason`` values for ``DISPOSITION_BASELINE_STORED``. These two are NOT
#: interchangeable and must never be aggregated together (whole-branch review,
#: Critical 1): spec §3 accepts that a re-baseline throws away one diff window
#: -- a real change landing in it is never reported -- and it accepts that cost
#: *only because* "the Runs pane says why". A single ``baseline`` counter
#: cannot say why, and the difference is exactly the part the user needs:
#:
#: * ``first_check`` -- there was no previous snapshot, so nothing was
#:   discarded and no change could have been lost.
#: * ``extraction_settings_changed`` -- there WAS a snapshot with real prior
#:   content and it was discarded uncompared, so a change the page made in
#:   that window is gone.
#:
#: `local_watchlists_service._disposition_count_keys` binds each to its own
#: run counter (``baseline`` / ``rebaselined``) for that reason.
REASON_FIRST_CHECK = "first_check"
REASON_EXTRACTION_SETTINGS_CHANGED = "extraction_settings_changed"
#: ``reason`` for ``DISPOSITION_WITHHELD``.
REASON_BELOW_CHANGE_THRESHOLD = "below_change_threshold"


def _disposition(
    kind: str,
    *,
    reason: Optional[str] = None,
    withheld_percentage: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the record that says what a check DID, not merely what it returned.

    ``check_url`` used to return ``change_info | None``, and ``None`` meant four
    different things: first check, unchanged page, change withheld under the
    threshold, or (after TASK-1362) a re-baseline. The user could not tell
    "nothing happened" from "something happened and the app decided not to
    mention it" -- the same failure class as the watchlists that never ran at
    all (TASK-1210).

    Args:
        kind: One of the four ``DISPOSITION_*`` values.
        reason: Why, for the kinds that have more than one cause.
        withheld_percentage: The measured change, scaled ×100 for display to
            match ``change_percentage`` (TASK-1343's convention). ``None``
            except for ``DISPOSITION_WITHHELD``.

    Returns:
        A three-key dict; the shape is fixed so consumers can index it.
    """
    return {
        "kind": kind,
        "reason": reason,
        "withheld_percentage": withheld_percentage,
    }


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

        # Parse feed based on type.
        #
        # task-15463: the parse runs under `asyncio.to_thread`, the fetch above
        # does not. A scheduled check is dispatched straight onto the event
        # loop, and `ET.fromstring` (or `json.loads`) over a whole feed body is
        # synchronous CPU work that froze the UI for its duration -- while the
        # fetch either side of it was already non-blocking async httpx and
        # needs no help. Nothing here touches sqlite, so the thread hop is
        # unconditional: unlike the database hops in this module it has no
        # in-memory-connection hazard to respect.
        content_type = response.headers.get("content-type", "").lower()

        if "json" in content_type or subscription["type"] == "json_feed":
            return await asyncio.to_thread(self._parse_json_feed, response.text)
        return await asyncio.to_thread(
            self._parse_xml_feed, response.text, subscription["type"]
        )

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


#: How many snapshots `_store_snapshot` keeps per **(subscription, url)**
#: (TASK-1393). Nothing in the repo ever deleted from `url_snapshots` -- the
#: only DELETE lives in `baseline_manager.py`, which has zero importers
#: (TASK-1360) -- while every significant change stores a full row including
#: `raw_html`. TASK-1362's default `change_threshold` of 0.0 means every real
#: change persists one, and TASK-1361's per-URL baselines multiply that by a
#: source's URL count, so steady state was monotonic growth in the user's
#: private database.
#:
#: Three, and why each one is needed:
#:
#: 1. The live baseline -- the row the next `check_url` reads. Losing it
#:    re-baselines the URL and burns a diff window.
#: 2. The previous snapshot. The design spec's Content-pane mockup
#:    (`Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md`,
#:    "`[previous snapshot]` reading from `url_snapshots`") promises the reader
#:    an affordance that is **not built yet** -- there is no reference to it
#:    anywhere in `UI/`, and it is filed separately. Pruning must not foreclose
#:    it, so the second-newest row per URL survives.
#: 3. One row of slack for the same-second tie window of TASK-1361:
#:    `created_at` is a DATETIME with one-second resolution, so two checks
#:    inside one second are ordered only by the `id` tie-break.
#:
#: Deliberately NOT a config setting: there is no user question here that a
#: number answers, and a knob would be one more surface to migrate, validate
#: and document for a bound nobody has asked to move. `baseline_manager`'s
#: `retention_days` is orphaned code and stays untouched (TASK-1360).
_SNAPSHOTS_KEPT_PER_URL = 3


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

    async def check_url(
        self, subscription: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Check a URL for changes.

        Args:
            subscription: Subscription dictionary

        Returns:
            ``(change_info, disposition)``. ``change_info`` is the item to
            produce, or ``None`` when there is nothing to report. The
            disposition (see :func:`_disposition`) always says which of the four
            outcomes happened, because ``None`` alone is ambiguous between all
            four -- spec §4.
        """
        subscription_id = subscription["id"]
        url = subscription["source"]

        # The settings that shape extracted text, hashed. Computed before the
        # fetch so the value written to the new snapshot and the value compared
        # against the old one are provably the same one (spec §3).
        # The `"auto"` default is the SAME one `_fetch_url_content` applies
        # (whole-branch review, Minor 7): an absent key extracts HTML, an
        # explicit NULL does not, and the fingerprint has to agree with the
        # fetch on both or it hashes two different extractions alike.
        current_fingerprint = extraction_fingerprint(
            subscription.get("ignore_selectors"),
            subscription.get("extraction_method", "auto"),
        )

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

            # Get previous snapshot.
            #
            # TASK-1361. The `id DESC` tie-break is load-bearing, not tidiness:
            # `created_at` is a DATETIME defaulting to CURRENT_TIMESTAMP, which
            # has one-second resolution. Two checks of the same source inside
            # one second therefore share a `created_at`, and with only that
            # column in the ORDER BY, SQLite may return either row -- so a
            # check can measure the change against a *stale* baseline and
            # report the wrong percentage, the wrong diff, or a change that is
            # not there. `id` is INTEGER PRIMARY KEY AUTOINCREMENT, so it is
            # monotonic and breaks the tie by true insertion order. Same shape
            # as `Workspaces/registry_service.py:171`, which already pairs
            # `created_at` with a second key.
            #
            # Note for whoever resolves TASK-1360: `baseline_manager.py` has
            # four more instances of this same unqualified ordering, one of
            # them a pruning DELETE, so adopting that module means fixing
            # them too.
            # The `url` predicate is load-bearing for `url_list` and `sitemap`
            # sources: every URL of such a source shares one `subscription_id`,
            # so without it the "previous snapshot" is whichever URL was
            # checked last. Two URLs on one source then measured each other --
            # every URL after the first looked changed on the very first check,
            # and no URL was ever reported unchanged. Found while making the
            # per-run disposition counts (spec §4) come out right, which is
            # impossible while the baselines are shared.
            #
            # TASK-1393 ordering pact (one of two sites; grep that phrase for
            # the other). This ORDER BY is duplicated by the pruning DELETE in
            # `_store_snapshot`, at the end of this file, which keeps the first
            # `_SNAPSHOTS_KEPT_PER_URL` rows under exactly this ordering.
            # THE INVARIANT: survivor ordering == baseline ordering. That is
            # what makes the row this SELECT returns provably the first
            # survivor, and therefore never a row the prune deleted. Change
            # either ORDER BY (or either `url` predicate) and you must change
            # the other in the same commit; diverge them and this SELECT reads
            # a pruned row -- i.e. re-baselines, or diffs against stale text.
            # task-15463: the read hops to a worker thread (`run_db_off_loop`),
            # like every other sqlite call on the scheduled-check path. The row
            # is materialized by `fetchone` inside the hop, so what comes back
            # is plain data, not a live cursor.
            previous = await run_db_off_loop(
                self.db, self._select_latest_snapshot, subscription_id, url
            )

            if not previous:
                # First check - store baseline
                await self._store_snapshot(
                    subscription_id,
                    url,
                    current_content,
                    fingerprint=current_fingerprint,
                )
                breaker.record_success()
                return None, _disposition(
                    DISPOSITION_BASELINE_STORED, reason=REASON_FIRST_CHECK
                )

            # Spec §3, and BEFORE the hash comparison: a snapshot holds text
            # extracted under the settings in force when it was captured, so
            # once those settings change the stored hash describes a different
            # extraction and comparing it proves nothing. Equal hashes across a
            # settings change are luck, not evidence, and a differing hash is
            # the noise appearing or disappearing rather than anything the site
            # did -- which is the phantom item this prevents. Re-baseline
            # instead: one diff window is the honest, bounded cost.
            #
            # A stored NULL (every pre-migration snapshot) counts as a
            # mismatch, which makes the migration self-healing: each existing
            # source re-baselines exactly once and the Runs pane says why.
            #
            # That NULL case lands in the ``rebaselined`` counter with reason
            # ``extraction_settings_changed``, deliberately and not by
            # accident (whole-branch review, Important 2). Both halves are
            # honest: a pre-migration snapshot holds real prior content that
            # IS discarded uncompared -- a lost window the user must be
            # warned about, unlike a true `first_check` where nothing existed
            # -- and the settings really did change, because
            # `_ensure_watchlists_schema`'s one-time migration rewrote every
            # url-family source's `ignore_selectors` (to the shipped default
            # set) and `change_threshold` in the same breath as adding this
            # column. Guarding on truthiness instead (`if previous_fp and
            # previous_fp != current`) would compare text extracted WITHOUT
            # those selectors against text extracted WITH them and fire a
            # phantom item, on the first check of every migrated source.
            previous_fingerprint = previous["extraction_fingerprint"] or ""
            if previous_fingerprint != current_fingerprint:
                await self._store_snapshot(
                    subscription_id,
                    url,
                    current_content,
                    fingerprint=current_fingerprint,
                )
                breaker.record_success()
                return None, _disposition(
                    DISPOSITION_BASELINE_STORED,
                    reason=REASON_EXTRACTION_SETTINGS_CHANGED,
                )

            # Calculate change
            current_hash = ContentExtractor.calculate_content_hash(
                current_content["text"]
            )

            if current_hash == previous["content_hash"]:
                # No change
                breaker.record_success()
                return None, _disposition(DISPOSITION_UNCHANGED)

            # Calculate change details. The percentage hop returns the segment
            # lists it built alongside the ratio: segmentation dominates the
            # ratio's cost, and the details hop below consumes the very same
            # lists, so they are carried across the threshold check instead of
            # being rebuilt (TASK-16839 fix round, review finding 2). The lists
            # go out of scope with this call frame either way; holding them
            # across one cheap comparison does not change peak memory, which
            # the details hop's own segmentation already reached before.
            previous_text = previous["extracted_content"] or ""
            (
                change_percentage,
                old_segments,
                new_segments,
            ) = await asyncio.to_thread(
                _change_percentage_with_segments,
                previous_text,
                current_content["text"],
            )

            # Check if change exceeds threshold. Both sides of this comparison
            # are 0.0-1.0 ratios -- the scaling to a percentage happens only
            # where the value is handed to the reader (below, and in the
            # disposition).
            #
            # The default is 0.0 (spec §1): the threshold was a *volume* filter
            # being used as a *noise* filter, and at 0.1 a one-sentence edit to
            # a long page moved whole-page similarity far too little to be
            # reported at all. Noise is suppressed by `ignore_selectors`, which
            # strips named elements before anything is hashed. The identical-
            # hash check above already short-circuits unchanged pages, so 0.0
            # means "any real difference in extracted text".
            #
            # `.get("change_threshold", 0.0)` would NOT be enough: the key
            # exists whenever the row was read from the DB, so an explicit NULL
            # comes back as `None` and `change_percentage < None` is a
            # TypeError inside a scheduled fetch.
            raw_threshold = subscription.get("change_threshold")
            threshold = 0.0 if raw_threshold is None else float(raw_threshold)
            if change_percentage < threshold:
                # Change too small -- recorded rather than silent, so a user who
                # raised the threshold can see what it is holding back.
                breaker.record_success()
                return None, _disposition(
                    DISPOSITION_WITHHELD,
                    reason=REASON_BELOW_CHANGE_THRESHOLD,
                    withheld_percentage=change_percentage * 100.0,
                )

            # Significant change detected. TASK-1343: `content` is the DIFF, not
            # the new page. The full page continues to live in `url_snapshots`
            # (`_store_snapshot`, immediately below), which is where the
            # reader's `[full page]` / `[previous snapshot]` affordances read
            # from; storing it a second time here bought nothing and left the
            # reader unable to see what had actually changed.
            (
                diff_body,
                diff_summary,
                added_text,
                removed_text,
                change_type,
            ) = await asyncio.to_thread(
                _build_significant_change_details,
                previous_text,
                current_content["text"],
                old_segments=old_segments,
                new_segments=new_segments,
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
                "change_type": change_type,
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
                # A per-rule opt-in (TASK-1363): a rule scoped to "appeared" or
                # "disappeared" matches against just one of these instead of
                # the whole page above. Matching-only, like
                # `RULE_MATCH_TEXT_KEY` -- see `watchlist_rule_matching`.
                RULE_MATCH_ADDED_TEXT_KEY: added_text,
                RULE_MATCH_REMOVED_TEXT_KEY: removed_text,
            }

            # Store new snapshot
            await self._store_snapshot(
                subscription_id,
                url,
                current_content,
                current_hash,
                fingerprint=current_fingerprint,
            )

            breaker.record_success()
            return change_info, _disposition(DISPOSITION_CHANGED)

        except Exception:
            breaker.record_failure()
            raise

    def _select_latest_snapshot(self, subscription_id: int, url: str) -> Any:
        """The newest snapshot for one (subscription, url), or ``None``.

        Extracted only so `check_url`'s baseline read can take a
        `run_db_off_loop` hop (task-15463); the query, including its
        TASK-1393 ordering pact with `_store_snapshot`'s prune, is unchanged.
        """
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT content_hash, extracted_content, extraction_fingerprint
            FROM url_snapshots
            WHERE subscription_id = ? AND url = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """,
            (subscription_id, url),
        )
        return cursor.fetchone()

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
            text = await asyncio.to_thread(
                ContentExtractor.extract_text_from_html,
                response.text,
                ignore_selectors,
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
        fingerprint: Optional[str] = None,
    ) -> None:
        """Store a URL snapshot.

        Args:
            subscription_id: Owning subscription.
            url: The exact URL fetched. For ``url_list``/``sitemap`` sources
                many URLs share one ``subscription_id``, and this column is
                what keeps their baselines apart.
            content: The ``_fetch_url_content`` result.
            content_hash: Precomputed hash of ``content["text"]``, or ``None``
                to compute it here.
            fingerprint: The ``extraction_fingerprint`` in force at capture
                time. Stored so a later check can tell whether this snapshot's
                text is still comparable (spec §3). ``None`` is written as
                NULL, which every reader treats as a mismatch.
        """
        if not self.persist_snapshots:
            return
        if not content_hash:
            content_hash = ContentExtractor.calculate_content_hash(content["text"])

        # task-15463: the whole INSERT+prune transaction takes one worker-thread
        # hop, so the commit boundary the TASK-1393 comment below relies on is
        # untouched -- both statements are still inside one `db.transaction()`,
        # just not on the event loop.
        await run_db_off_loop(
            self.db,
            self._write_snapshot,
            subscription_id,
            url,
            content,
            content_hash,
            fingerprint,
        )

    def _write_snapshot(
        self,
        subscription_id: int,
        url: str,
        content: Dict[str, Any],
        content_hash: str,
        fingerprint: Optional[str],
    ) -> None:
        """Insert one snapshot and prune this URL's older ones, in one commit."""
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO url_snapshots
                (subscription_id, url, content_hash, extracted_content, raw_html,
                 headers, extraction_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    subscription_id,
                    url,
                    content_hash,
                    content["text"],
                    content["html"],
                    json.dumps(content["headers"]),
                    fingerprint,
                ),
            )

            # TASK-1393: prune here, and only here. This is the single live
            # write path into `url_snapshots`, and it already holds a
            # transaction -- so the INSERT and the DELETE share ONE commit
            # boundary (`SubscriptionsDB.transaction`). A crash before that
            # commit rolls back both, so the "row inserted, prune not yet run"
            # state is not merely benign, it is unrepresentable -- worth
            # stating because the neighbouring TASK-1362 fingerprint migration
            # (`DB/Subscriptions_DB.py`) had to take an explicit
            # `BEGIN IMMEDIATE` to get the same guarantee. The shadow-mode
            # guard above returns before both, so a dry run still deletes
            # nothing.
            #
            # TASK-1393 ordering pact (one of two sites; grep that phrase for
            # the other). Survivors are chosen by `ORDER BY created_at DESC,
            # id DESC` -- the SAME ordering as `check_url`'s baseline SELECT,
            # earlier in this file, the `SELECT content_hash, ... FROM
            # url_snapshots ... LIMIT 1` that runs right after the fetch
            # (TASK-1361's tie-break). THE INVARIANT: survivor ordering ==
            # baseline ordering. The row the next check will read is therefore,
            # by construction, the first survivor -- it can never be pruned,
            # whatever the cap. Change either ORDER BY (or either `url`
            # predicate) and you must change the other in the same commit;
            # diverging them lets this DELETE evict the very row the next check
            # is about to ask for.
            #
            # The `url` predicate is the load-bearing part, and it is on BOTH
            # halves. A `url_list` or `sitemap` source gives every one of its
            # URLs the same `subscription_id`, so pruning per subscription
            # would let a busy URL's snapshots evict a quiet URL's only
            # baseline -- and that URL would re-baseline on its next check,
            # for ever, reporting nothing each time. That is precisely the
            # defect in the orphaned `baseline_manager._cleanup_old_baselines`
            # (see TASK-1360).
            cursor.execute(
                """
                DELETE FROM url_snapshots
                WHERE subscription_id = ? AND url = ?
                  AND id NOT IN (
                      SELECT id FROM url_snapshots
                      WHERE subscription_id = ? AND url = ?
                      ORDER BY created_at DESC, id DESC
                      LIMIT ?
                  )
                """,
                (
                    subscription_id,
                    url,
                    subscription_id,
                    url,
                    _SNAPSHOTS_KEPT_PER_URL,
                ),
            )
            pruned = cursor.rowcount
            if pruned > 0:
                logger.debug(
                    "Pruned {} snapshot(s) for subscription {}, keeping the newest {}",
                    pruned,
                    subscription_id,
                    _SNAPSHOTS_KEPT_PER_URL,
                )


# End of monitoring_engine.py
