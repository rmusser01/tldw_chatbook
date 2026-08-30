"""Sync cores for web_* agent tools.

The SSRF guard below is written fresh for tldw_chatbook, using tldw_server's
tldw_Server_API/app/core/Web_Scraping/outbound_policy.py @ 5605b9d9906322c2e6b5342b48c391ae674d315e
(https://github.com/rmusser01/tldw_server, GPL-3.0-only) as the requirements
checklist — see re-plan spec §2.1 (2026-08-05).

web_fetch ports the *behaviors* (manual redirect loop with per-hop policy
checks, bounded streaming reads, per-domain rate limiting, TTL response cache,
trafilatura extraction with tag-strip fallback) of tldw_server's
tldw_Server_API/app/core/MCP_unified/modules/implementations/web_fetch_module.py,
web_rate_limit.py and web_cache.py @ 5605b9d9906322c2e6b5342b48c391ae674d315e —
rewritten as a small sync function, not a line port.
"""

import asyncio
import codecs
import html as html_lib
import importlib.util
import ipaddress
import re
import socket
import threading
import time
import zipfile
from collections import deque
from html.parser import HTMLParser
from io import BytesIO
from pathlib import PurePosixPath
from typing import NamedTuple, Optional
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from loguru import logger

from .local_tool_impls import LocalToolError
from ..Utils.tls_trust import build_httpx_client
from ..Web_Scraping.deep_search_citations import (
    summarize_for_footer as deep_search_citations_footer,
)

# XXE hardening for attacker-controlled sitemap XML (mirrors
# tldw_chatbook/Subscriptions/security.py's pattern): defusedxml is an
# optional dep (websearch/ebook/subscriptions extras), so fall back to the
# stdlib parser rather than hard-failing this core module when it's absent.
# defusedxml.ElementTree re-exports xml.etree.ElementTree's ParseError object
# unchanged (verified: `xET.ParseError is xml.etree.ElementTree.ParseError`
# == True across both branches). defusedxml's own refusals (EntitiesForbidden
# etc.) are a SEPARATE hierarchy that subclasses ValueError, not ParseError,
# so `_parse_sitemap` catches `(xET.ParseError, ValueError)` to cover both a
# malformed document and a hardening refusal with the same structured
# [crawl-failed] error, regardless of which library is in use.
try:
    import defusedxml.ElementTree as xET
except ImportError:
    import xml.etree.ElementTree as xET

    logger.warning(
        "defusedxml not available, using standard xml.etree for sitemap parsing. "
        "Install defusedxml for better security."
    )

_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Non-public ranges ipaddress does not flag on its own: CGNAT/shared space
# (RFC 6598 — Tailscale tailnets, carrier gear) is NOT private per
# ipaddress.is_private on Python 3.12, and 192.0.0.0/24 (IETF protocol
# assignments, RFC 6890) is only partially covered across versions.
_BLOCKED_EXTRA_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("192.0.0.0/24"),
)


def _is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:  # ::ffff:127.0.0.1 -> check the embedded v4 address
        ip = mapped
    if any(ip in net for net in _BLOCKED_EXTRA_NETWORKS):
        return False
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def validate_outbound_url(url: str) -> str:
    """Return ``url`` if it's safe to fetch; raise LocalToolError otherwise.

    Checks: scheme allowlist (http/https only), host resolves, and EVERY
    resolved IP is public (private/loopback/link-local/multicast/reserved/
    unspecified/CGNAT refused). Literal IPs are checked directly without DNS.

    Called for the initial URL AND every redirect hop. DNS-rebinding caveat:
    resolution happens again inside the HTTP client, so a hostile DNS server
    could return a public IP here and a private one at connection time;
    re-validating every redirect hop bounds that window. Pinning the
    connection to the validated IP is deliberately out of scope.

    Raises:
        LocalToolError: if the scheme is disallowed, the URL is malformed
            (bad port, bad IPv6 brackets), the host is missing or does not
            resolve, or any resolved IP is not public.
    """
    try:
        parts = urlsplit(url.strip())
        port = parts.port  # ValueError on out-of-range/non-numeric ports
    except ValueError as exc:
        raise LocalToolError(f"URL is malformed: {url!r}") from exc
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise LocalToolError(f"URL scheme not allowed (http/https only): {url!r}")
    host = parts.hostname
    if not host:
        raise LocalToolError(f"URL has no host: {url!r}")
    try:
        ipaddress.ip_address(host)  # literal IP: check directly
        candidates = [host]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, port or (443 if parts.scheme == "https" else 80),
                                       proto=socket.IPPROTO_TCP)
        except (socket.gaierror, UnicodeError, OSError) as exc:
            raise LocalToolError(f"host does not resolve: {host!r}") from exc
        candidates = [info[4][0] for info in infos]
    if not candidates or not all(_is_public_ip(ip) for ip in candidates):
        raise LocalToolError(f"host resolves to a private/internal address: {host!r}")
    return url


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------

FETCH_MAX_REDIRECTS = 5
FETCH_TIMEOUT_SECONDS = 30.0
FETCH_MAX_BYTES = 1 * 1024 * 1024          # default cap
FETCH_HARD_MAX_BYTES = 5 * 1024 * 1024     # absolute ceiling for max_bytes arg
FETCH_CACHE_TTL_SECONDS = 900.0
FETCH_CACHE_MAX_ENTRIES = 256
RATE_LIMIT_INTERVAL_SECONDS = 1.0          # per-domain min interval
PDF_MAX_BYTES = 20 * 1024 * 1024  # refusal threshold, never a truncation (spec §1)
# Refusal threshold for the OTHER allowlisted binary kinds (image/zip/audio):
# one shared ceiling, PDF keeps its own untouched (binary-fetch design doc
# ruling 4). Raised mid-stream on sniff/declared match, like pdf_max_bytes;
# a byte-truncated binary body is refused, never processed partially.
BINARY_MAX_BYTES = 10 * 1024 * 1024
ARCHIVE_LIST_MAX = 20  # display cap on ZIP member lines (design doc ruling 2, Minor 10)
# Per-member display cap (Qodo PR #1442): zip member names are attacker-
# controlled and the format allows up to 64 KiB per name — 20 such lines
# would blow the 32 KiB provider cap / 16,000-char runtime ceiling and get
# head-truncated, eating the "… and N more" marker off the end. Applied
# AFTER the suspicious-name repr-escaping, so it bounds that output too.
ARCHIVE_MEMBER_NAME_MAX = 200

_USER_AGENT = "tldw-chatbook-web-fetch/1.0"
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# Content types surfaced as text. Anything else is refused so the tool never
# returns binary blobs through the text tool contract.
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_PLAIN_TYPES = frozenset({
    "text/plain", "text/markdown", "application/json",
    "application/xml", "text/xml", "application/ld+json",
})

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t ]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")

# Module-level state (per-process). Cleared by _reset_state_for_tests().
# Keyed by (url, effective max_bytes): a small-cap fetch must not poison a
# later full-cap call. Bounded because web_crawl bulk-loads it (spec §1).
_fetch_cache: dict[tuple[str, int], tuple[float, str]] = {}
_domain_last_fetch: dict[str, float] = {}
# task-3770: tool calls each run on their own worker thread (same race
# Qodo's PR #1444 found for _search_cache below), so the eviction scan
# (min() ITERATES the dict) can race a concurrent put/pop into "dictionary
# changed size during iteration". One lock PER cache -- a robots fetch must
# never wait on a page-cache scan -- covering only the short cache ops
# (never a network call): see _cache_put and web_fetch's cache-hit branch.
_fetch_cache_lock = threading.Lock()

# robots.txt cache (task-2833 design doc §Mechanism), keyed by
# scheme://netloc -- host-level, not per-UA: only the can_fetch() *query*
# is per-caller-UA, the fetched/parsed policy is shared. Value is
# (expires_at_monotonic, parser_or_None); None means "unreachable or
# unparsable" -> fail open (ruling 2).
ROBOTS_MAX_BYTES = 64 * 1024
ROBOTS_CACHE_TTL_SECONDS = 1800.0
ROBOTS_CACHE_MAX_ENTRIES = 128

_robots_cache: dict[str, tuple[float, "RobotFileParser | None"]] = {}
# task-3770: separate from _fetch_cache_lock (see its comment above) -- a
# robots-cache op never blocks on a page-cache scan and vice versa. Covers
# only _robots_cache_put's body and _robots_allows' cache read/expiry
# check; the cache-miss robots.txt FETCH runs outside it (see _robots_allows).
_robots_cache_lock = threading.Lock()

# Test seam: tests set this to an httpx.MockTransport.
_transport: "httpx.BaseTransport | None" = None


def _reset_state_for_tests() -> None:
    """Clear the module-level fetch/search caches, rate-limit state, and robots cache."""
    _fetch_cache.clear()
    _domain_last_fetch.clear()
    _robots_cache.clear()
    _search_cache.clear()


def _cache_put(key: tuple[str, int], text: str) -> None:
    """Insert into cache, evicting earliest-expiry entry if at capacity.

    Locked (task-3770): the eviction scan (min() ITERATES the dict) and the
    insert are the whole critical section -- no network/extraction call is
    ever made under this lock.
    """
    with _fetch_cache_lock:
        if key not in _fetch_cache and len(_fetch_cache) >= FETCH_CACHE_MAX_ENTRIES:
            oldest = min(_fetch_cache, key=lambda k: _fetch_cache[k][0])
            _fetch_cache.pop(oldest)
        _fetch_cache[key] = (time.monotonic() + FETCH_CACHE_TTL_SECONDS, text)


def _robots_cache_put(key: str, parser: "RobotFileParser | None") -> None:
    """Insert into the robots cache, evicting earliest-expiry entry at capacity.

    Locked (task-3770), same shape as _cache_put: eviction scan + insert
    only, never a network call.
    """
    with _robots_cache_lock:
        if key not in _robots_cache and len(_robots_cache) >= ROBOTS_CACHE_MAX_ENTRIES:
            oldest = min(_robots_cache, key=lambda k: _robots_cache[k][0])
            _robots_cache.pop(oldest)
        _robots_cache[key] = (time.monotonic() + ROBOTS_CACHE_TTL_SECONDS, parser)


def _validate_hop(url: str) -> None:
    """validate_outbound_url with structured reason prefixes."""
    try:
        validate_outbound_url(url)
    except LocalToolError as exc:
        msg = str(exc)
        reason = "ssrf" if ("private" in msg or "internal" in msg) else "invalid-url"
        raise LocalToolError(f"[{reason}] {msg}") from exc


def _enforce_rate_limit(host: str) -> None:
    """Sleep until the per-domain minimum interval has elapsed.

    Raises LocalToolError("rate-limited") if sleeping did not advance the
    clock (pathological/frozen clock) instead of spinning forever.
    """
    last = _domain_last_fetch.get(host)
    now = time.monotonic()
    if last is not None:
        wait = RATE_LIMIT_INTERVAL_SECONDS - (now - last)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
            if now - last < RATE_LIMIT_INTERVAL_SECONDS:
                raise LocalToolError(
                    f"[rate-limited] per-domain interval not respected for {host!r}"
                )
    _domain_last_fetch[host] = now


_PDF_MAGIC = b"%PDF-"

# Declared content-type -> kind, resolved IMMEDIATELY with no byte
# confirmation (PDF-shortcut precedent, binary-fetch design doc ruling 3).
# Audio is deliberately absent: it has no reliable single magic, so it is
# matched by TOP-LEVEL type in _declared_kind() below instead of an exact
# subtype, accepting real-world variants (audio/mp3, audio/x-wav, ...).
_DECLARED_BINARY_KINDS = {
    "application/pdf": "pdf",
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "application/zip": "zip",
}

# WEBP is NOT a contiguous-prefix magic like the others (RIFF....WEBP, with
# a 4-byte size field at [4:8] ignored) -- it needs both anchors, which is
# why the minimum sniff buffering rises from 5 (%PDF-) to 12 (design doc
# ruling 3, Important 2).
_SNIFF_PREFIX_LEN = 12


def _top_level_type(content_type: str) -> str:
    """The substring before '/' of an already-lowercased, parameter-
    stripped main type (e.g. the ``declared`` value computed in
    ``_fetch_once``). Deliberately distinct from ``_extract_text``'s
    ``main_type``, which keeps the full type/subtype (design doc ruling on
    audio, Important 7 -- reusing that splitter here previously collided).
    """
    return content_type.split("/", 1)[0] if content_type else ""


def _declared_kind(declared: str) -> "str | None":
    """Resolve a binary ``kind`` from a declared content-type ALONE -- no
    byte confirmation, mirroring the PDF shortcut. Returns None when the
    declared type doesn't shortcut resolution (sniff decides instead)."""
    kind = _DECLARED_BINARY_KINDS.get(declared)
    if kind is not None:
        return kind
    if _top_level_type(declared) == "audio":
        return "audio"
    return None


def _sniff_kind(prefix: bytes) -> "str | None":
    """Identify a binary kind from the leading bytes of a response body.

    Audio has no reliable single magic (ID3/frame-sync ambiguity) and is
    declared-type-only -- never returned here. ``prefix`` may be shorter
    than ``_SNIFF_PREFIX_LEN`` for a short body; magics needing more bytes
    than are present simply fail to match (no crash, no false positive).
    """
    if prefix.startswith(_PDF_MAGIC):
        return "pdf"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "image"
    if prefix.startswith(b"GIF8"):
        return "image"
    if prefix.startswith(b"PK\x03\x04"):
        return "zip"
    if prefix[0:4] == b"RIFF" and prefix[8:12] == b"WEBP":
        return "image"
    return None


def _binary_read_cap(
    kind: "str | None",
    max_bytes: int,
    pdf_max_bytes: "int | None",
    binary_max_bytes: "int | None",
) -> int:
    """The read/truncation cap for a resolved ``kind`` (design doc ruling 4):
    PDF keeps its own ceiling; the other allowlisted binary kinds share
    ``binary_max_bytes``; anything else uses the caller's ``max_bytes``."""
    if kind == "pdf" and pdf_max_bytes is not None:
        return pdf_max_bytes
    if kind in ("image", "zip", "audio") and binary_max_bytes is not None:
        return binary_max_bytes
    return max_bytes


def _fetch_once(
    client: httpx.Client,
    url: str,
    max_bytes: int,
    *,
    pdf_max_bytes: "int | None" = None,
    binary_max_bytes: "int | None" = None,
    html_only: bool = False,
) -> tuple[int, httpx.Headers, bytes, bool, "str | None"]:
    """One GET with a bounded streaming read; redirects are NOT followed.

    Returns (status, headers, body, truncated, kind). ``kind`` is one of
    "pdf", "image", "zip", "audio", or None (no recognized binary kind --
    the body proceeds to ordinary text extraction). The read cap is
    decided MID-STREAM (spec §1, generalized by the binary-fetch design
    doc): a response resolving to "pdf" reads up to ``pdf_max_bytes``
    (PDF's own ceiling, unaffected by this generalization); a response
    resolving to "image"/"zip"/"audio" reads up to ``binary_max_bytes``
    (the shared ceiling); anything else uses ``max_bytes``. A byte-
    truncated binary body is refused by the caller, never processed
    partially.

    Kind resolution mirrors the PDF precedent (design doc ruling 3): a
    declared content-type in the allowlist resolves the kind IMMEDIATELY,
    no byte confirmation (see ``_declared_kind``); otherwise a magic-byte
    sniff on the first ``_SNIFF_PREFIX_LEN`` buffered bytes (see
    ``_sniff_kind``) OVERRIDES a wrong or absent declared type.

    ``html_only`` (web_crawl) stops the body read once the kind is KNOWN
    (after the sniff/declared-type lookup resolves) and it is not a plain
    HTML/absent-type response -- either a recognized kind was resolved, or
    a non-empty declared type is not an HTML type. A response that
    resolves to no recognized kind and declares HTML (or nothing) keeps
    reading for the caller's later ``<html`` sniff. Side effect of the
    12-byte sniff window (design doc, Blast radius): the minimum buffering
    before this early abort can fire rises from 5 to 12 bytes -- bounded
    and harmless, but real.
    """
    with client.stream("GET", url) as response:
        status = response.status_code
        if status in _REDIRECT_STATUSES:
            return status, response.headers, b"", False, None
        declared = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        chunks: list[bytes] = []
        downloaded = 0
        kind: "str | None" = _declared_kind(declared)
        resolved = kind is not None
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            downloaded += len(chunk)
            if not resolved and downloaded >= _SNIFF_PREFIX_LEN:
                kind = _sniff_kind(b"".join(chunks)[:_SNIFF_PREFIX_LEN])
                resolved = True
            # Review fix (Important 2): the abort predicate keys on
            # kind == "pdf" plus the DECLARED type only -- never on a
            # sniffed image/zip/audio kind -- so a binary served as
            # text/html during a crawl keeps today's full-read behavior
            # (the design doc's stated non-goal) instead of being cut at
            # the sniff window with truncated=False.
            if html_only and resolved and (kind == "pdf" or (declared and declared not in _HTML_TYPES)):
                break  # crawl only needs the type; don't drain the body
            # Review fix (Minor 3): while the kind is UNRESOLVED the loop
            # must not break before the sniff window fills -- a caller
            # max_bytes under _SNIFF_PREFIX_LEN could otherwise hand the
            # post-loop sniff a partial prefix that still resolves (e.g.
            # 5 of 12 bytes matching %PDF-), mis-computing truncated
            # against the raised ceiling.
            if resolved:
                cap = _binary_read_cap(kind, max_bytes, pdf_max_bytes, binary_max_bytes)
            else:
                cap = max(max_bytes, _SNIFF_PREFIX_LEN)
            if downloaded > cap:
                break  # overshoot by at most one chunk; sliced below
        if not resolved:  # body shorter than the sniff prefix
            kind = _sniff_kind(b"".join(chunks)[:_SNIFF_PREFIX_LEN])
        body = b"".join(chunks)
        cap = _binary_read_cap(kind, max_bytes, pdf_max_bytes, binary_max_bytes)
        truncated = len(body) > cap
        if truncated:
            body = body[:cap]
        return status, response.headers, body, truncated, kind


def _decode_body(body: bytes, content_type: str) -> str:
    """Decode using the charset advertised in the content-type header."""
    charset = "utf-8"
    if "charset=" in content_type.lower():
        candidate = content_type.lower().split("charset=", 1)[1].split(";", 1)[0].strip().strip('"')
        try:
            codecs.lookup(candidate)
            charset = candidate
        except (LookupError, ValueError):
            charset = "utf-8"
    return body.decode(charset, errors="replace")


def _strip_tags(html: str) -> str:
    """Fallback extraction: drop script/style, strip tags, collapse whitespace."""
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", without_scripts)
    text = html_lib.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def _format_size(num_bytes: int) -> str:
    """Human-readable byte count for binary-fetch metadata lines."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _member_display_name(name: str) -> str:
    """Traversal screen for one ZIP member name (design doc ruling 2):
    flag-and-show, not reject -- this module never extracts, so the only
    duty is to list a hostile name SAFELY (repr-escaped, not printed
    verbatim). Mirrors ``chatbook_importer._validated_archive_parts``'s
    checks (absolute path, ``..`` segments, a drive-letter-looking first
    segment -- ``PurePosixPath`` collapses ``""``/``"."`` segments before
    ``.parts``, so only ``..`` is live in that membership check) plus a
    screen up front for NUL, backslash (``PurePosixPath`` doesn't treat it
    as a separator), and ANY non-printable character: a member name is
    attacker-controlled text embedded in a structured listing, and a
    newline/ESC/RTL-override in it could forge listing rows or smuggle
    terminal control bytes into the model-facing transcript.
    ``str.isprintable()`` is False for all of those and True for ordinary
    Unicode names, so legitimate non-ASCII filenames list plainly.
    """
    if not name or not name.isprintable() or "\x00" in name or "\\" in name:
        return f"[suspicious name] {name!r}"
    posix = PurePosixPath(name)
    parts = posix.parts
    if (
        posix.is_absolute()
        or not parts
        or any(part in ("", ".", "..") for part in parts)
        or parts[0].endswith(":")
    ):
        return f"[suspicious name] {name!r}"
    return name


def _describe_image(body: bytes) -> str:
    """In-memory Pillow probe: format/size metadata only, never pixel
    data (design doc ruling 2; ``mode`` is deliberately not emitted --
    the spec's output format carries format/dimensions/bytes and nothing
    else). ``Image.MAX_IMAGE_PIXELS`` stays at its
    default -- this path never decodes pixel data, only headers, so
    declared-dimension abuse costs nothing beyond an honest metadata line.

    ``.verify()`` catches corruption (bad checksums raise SyntaxError,
    truncation raises OSError) without decoding pixel data; format/size
    survive it as cached header attributes, but a defensive re-open reads
    them anyway -- hygiene that matters for animated GIF/WEBP frame-
    seeking, since ``verify()`` leaves the file object unusable for
    anything further.
    """
    from PIL import Image, UnidentifiedImageError  # local import: keep module import cheap

    try:
        with Image.open(BytesIO(body)) as probe:
            probe.verify()
    except UnidentifiedImageError as exc:
        # Fixed message: Pillow's own text interpolates the BytesIO repr
        # (a heap address) -- noise the model can't use.
        raise LocalToolError("[image-error] could not identify image format") from exc
    except Exception as exc:
        raise LocalToolError(f"[image-error] could not read image: {exc}") from exc
    try:
        with Image.open(BytesIO(body)) as img:
            fmt = img.format or "unknown"
            width, height = img.size
    except Exception as exc:
        raise LocalToolError(f"[image-error] could not read image: {exc}") from exc
    return f"[image] {fmt} {width}×{height}, {_format_size(len(body))}"


def _describe_archive(body: bytes) -> str:
    """ZIP LISTING ONLY (design doc ruling 2) -- never extracts, never
    reads a member's body. Encrypted members are annotated via
    ``flag_bits & 0x1`` (``infolist()`` works fine under encryption; only
    a member's own ``.read()`` would need the password). Only a genuinely
    malformed archive raises; an encrypted-but-well-formed one lists fine.
    """
    try:
        with zipfile.ZipFile(BytesIO(body)) as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise LocalToolError(f"[archive-error] could not read ZIP: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — error contract: only LocalToolError escapes
        # Hostile central directories can raise beyond BadZipFile
        # (struct/Overflow/Value errors on absurd fields); normalize with a
        # fixed message — never interpolate an arbitrary exception string
        # derived from attacker-controlled bytes (Qodo PR #1442).
        raise LocalToolError("[archive-error] could not read ZIP (malformed metadata)") from exc
    lines = [f"[archive] ZIP, {_format_size(len(body))}, {len(infos)} members"]
    for info in infos[:ARCHIVE_LIST_MAX]:
        name = _member_display_name(info.filename)
        if len(name) > ARCHIVE_MEMBER_NAME_MAX:
            name = name[:ARCHIVE_MEMBER_NAME_MAX] + "… [name truncated]"
        encrypted = " (encrypted)" if (info.flag_bits & 0x1) else ""
        lines.append(f"{name} — {_format_size(info.file_size)}{encrypted}")
    if len(infos) > ARCHIVE_LIST_MAX:
        lines.append(f"… and {len(infos) - ARCHIVE_LIST_MAX} more")
    return "\n".join(lines)


def _describe_audio(content_type: str, size: int) -> str:
    """Metadata only, WITHOUT new dependencies (design doc ruling 2):
    mutagen is not a declared dep anywhere in pyproject and does not get
    added; richer audio metadata is a recorded non-goal."""
    main = content_type.split(";", 1)[0].strip().lower() or "audio/unknown"
    return f"[audio] {main}, {_format_size(size)}"


_BINARY_KIND_LABEL = {"image": "image", "zip": "archive", "audio": "audio"}


def _extract_text(body: bytes, content_type: str, kind: "str | None" = None) -> str:
    """Extract readable text, or metadata for a recognized binary ``kind``.

    ``kind`` (from ``_fetch_once``'s sniff+declared-type resolution)
    short-circuits straight to the matching binary describer BEFORE any
    decode is attempted -- binary bodies never round-trip through
    UTF-8-replace (design doc ruling 5). ``web_crawl`` deliberately never
    passes ``kind`` here: its own marker path already special-cases "pdf",
    and a wrong/absent declared type on any OTHER binary kind during a
    crawl keeps today's mojibake-decode behavior by design (design doc
    non-goals -- not rescued by the new sniff).
    """
    if kind == "image":
        return _describe_image(body)
    if kind == "zip":
        return _describe_archive(body)
    if kind == "audio":
        return _describe_audio(content_type, len(body))

    main_type = content_type.split(";", 1)[0].strip().lower()
    text = _decode_body(body, content_type)

    if main_type in _HTML_TYPES or (not main_type and "<html" in text.lower()):
        try:
            import trafilatura  # local import: heavy, keep module import cheap

            extracted = trafilatura.extract(
                text, include_comments=False, include_tables=False, include_images=False
            )
        except Exception:  # extractor failures fall back to tag strip
            extracted = None
        if extracted and extracted.strip():
            return extracted.strip()
        stripped = _strip_tags(text)
        if stripped:
            return stripped
        raise LocalToolError("[empty-content] no readable content could be extracted")

    if main_type and main_type not in _PLAIN_TYPES:
        raise LocalToolError(f"[empty-content] unsupported content type: {main_type}")
    cleaned = text.strip()
    if not cleaned:
        raise LocalToolError("[empty-content] response body was empty")
    return cleaned


def _pymupdf_available() -> bool:
    """Cheap availability probe (no import): the 20 MB PDF read ceiling and
    the [missing-dep] refusal must be decided before the GET starts — the probe
    chooses the read CAP; it does not skip the download itself. optional_deps.check_dependency()
    eagerly imports the module — wrong cost for the fetch hot path.

    Total, not just cheap: find_spec raises ValueError (not ImportError) when
    sys.modules already holds an entry for the name with __spec__ = None —
    e.g. a stubbed/partial module left behind by another import path. This
    module's failure contract is all-LocalToolError, so that must degrade to
    "unavailable" rather than escape as a raw ValueError.
    """
    try:
        return importlib.util.find_spec("pymupdf") is not None
    except (ImportError, ValueError):
        return False


def _extract_pdf_text(body: bytes, max_bytes: int) -> str:
    """Ephemeral PDF text extraction: bytes in, text out, nothing on disk.

    Stops the page loop as soon as accumulated text passes ``max_bytes`` —
    a 20 MB PDF can be thousands of pages and the tail is about to be
    thrown away anyway (spec §1).
    """
    try:
        import pymupdf  # local import: optional heavy dep (pdf extra)
    except ImportError as exc:
        raise LocalToolError(
            "[missing-dep] PDF support requires pymupdf — pip install tldw_chatbook[pdf]"
        ) from exc
    try:
        doc = pymupdf.open(stream=body, filetype="pdf")
    except Exception as exc:
        raise LocalToolError(f"[pdf-error] could not parse PDF: {exc}") from exc
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise LocalToolError("[pdf-error] PDF is encrypted")
        total_pages = doc.page_count
        parts: list[str] = []
        size = 0
        processed = 0
        for page in doc:
            text = page.get_text()
            processed += 1
            if text.strip():
                parts.append(text.strip())
                size += len(text.encode("utf-8"))
            if size > max_bytes:
                break
    except LocalToolError:
        raise
    except Exception as exc:  # damaged page trees surface mid-iteration
        raise LocalToolError(f"[pdf-error] could not parse PDF: {exc}") from exc
    finally:
        doc.close()
    joined = "\n\n".join(parts)
    if not joined:
        raise LocalToolError(
            "[empty-content] PDF contains no extractable text (scanned document?) "
            "— use media ingestion with OCR"
        )
    if size > max_bytes or processed < total_pages:
        raw = joined.encode("utf-8")[:max_bytes]
        joined = raw.decode("utf-8", errors="ignore") + (
            f"\n\n[... truncated: extracted text exceeded max_bytes={max_bytes}; "
            f"processed {processed} of {total_pages} pages ...]"
        )
    return joined


def _webfetch_settings() -> dict:
    """Resolve the ``[webfetch]`` config keys web_fetch/web_crawl need for
    robots.txt enforcement.

    A dedicated module function (mirroring ``_deep_search_settings()``
    below) so tests can monkeypatch config resolution wholesale rather than
    stubbing individual keys. ``_bool`` uses the same strict true-set
    coercion as ``_deep_search_settings``'s own helper: an actual bool
    value (the common case -- TOML parses `true`/`false` natively) passes
    through unchanged; a string coerces via ``"true"``/``"1"`` membership,
    so a stray quoted ``"false"`` reliably disables and a stray quoted
    ``"true"`` is never misread as disabled -- the same lesson
    ``_deep_search_settings`` recorded, applied in the opposite direction
    since THIS flag defaults to True rather than False. Anything else
    (missing key, wrong type) falls back to ``default``.

    Read once per tool invocation, not per hop (design doc ruling 6): both
    ``web_fetch`` and ``web_crawl`` call this exactly once at the top of
    the public function and thread the resolved flag through every hop.
    """
    from ..config import get_cli_setting  # local import: keep module import cheap

    def _bool(key: str, default: bool) -> bool:
        raw = get_cli_setting("webfetch", key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in ("true", "1")
        return default

    return {"respect_robots_txt": _bool("respect_robots_txt", True)}


def _robots_disallowed_message(url: str) -> str:
    """Structured refusal string (error contract style of [ssrf]/[invalid-url])."""
    host = urlsplit(url).hostname or url
    return (
        f"[robots-disallowed] {url} — {host}/robots.txt disallows this path "
        "for this tool's user agent; set [webfetch] respect_robots_txt = false to override"
    )


def _fetch_robots_parser(client: httpx.Client, cache_key: str) -> "RobotFileParser | None":
    """Fetch+parse the robots.txt at ``cache_key`` (``scheme://host[:port]``).

    Own bounded redirect loop (fix round 1, Important 1): a 3xx robots.txt
    response must not be treated as an unreachable fetch. ``_fetch_once``
    never follows redirects itself, so a bare non-2xx check previously
    negative-cached (i.e. DISABLED enforcement for) any host whose
    robots.txt redirects, for the full ``ROBOTS_CACHE_TTL_SECONDS`` --
    silently defeating the feature for exactly the hosts (CDN-fronted,
    HTTP->HTTPS canonicalized) most likely to redirect it. Up to
    ``FETCH_MAX_REDIRECTS`` hops, each hop re-validated (SSRF) before the
    GET; a chain that exhausts the cap still fails open, same as any other
    unreachable-robots outcome.

    Fail-open (design doc ruling 2, "compat" semantics): anything short of
    a clean 2xx fetch under the byte cap -- network error, non-2xx status
    after following redirects, a body TRUNCATED at ROBOTS_MAX_BYTES (a
    half-file could silently drop trailing Disallow lines -- refusing to
    trust a truncated policy is the honest reading), or a parse failure --
    returns ``None`` (no restrictions), via broad ``except Exception``
    around each network step: a robots.txt outage must not brick fetching,
    and in the shipped code this same broad catch is what keeps every
    pre-existing route-less-robots test passing unchanged once the fixture
    opts back into enforcement. Every genuine fail-open exit logs a debug
    line (fix round 1, Minor 5) so operators can distinguish "allowed
    because the site's policy says so" from "allowed because robots.txt
    was unreachable".

    Rate-limited per hop like any other request to that hop's host (design
    doc ruling 5 -- the politeness probe is itself polite), and — unlike
    the network/parse steps above — a ``[rate-limited]`` LocalToolError
    (pathological frozen clock) is NOT caught here: it propagates out of
    this function exactly like it does for an ordinary page fetch, instead
    of being swallowed into an accidental ``ROBOTS_CACHE_TTL_SECONDS``
    enforcement-off window (fix round 1, Minor 5).
    """
    current = f"{cache_key}/robots.txt"
    for _hop in range(FETCH_MAX_REDIRECTS + 1):
        try:
            _validate_hop(current)  # symmetry with every other request (design doc Minor 9)
        except Exception as exc:  # noqa: BLE001 - broad by design: fails open
            logger.debug(f"robots.txt unreachable for {cache_key}: {exc} — failing open")
            return None
        _enforce_rate_limit(urlsplit(current).hostname or "unknown")  # propagates; see docstring
        try:
            status, headers, body, truncated, _kind = _fetch_once(client, current, ROBOTS_MAX_BYTES)
        except Exception as exc:  # noqa: BLE001 - broad by design: fails open
            logger.debug(f"robots.txt unreachable for {cache_key}: {exc} — failing open")
            return None
        if status in _REDIRECT_STATUSES:
            location = headers.get("location")
            if not location:
                logger.debug(
                    f"robots.txt unreachable for {cache_key}: "
                    f"redirect (status {status}) without a Location header — failing open"
                )
                return None
            try:
                current = urljoin(current, location)
            except ValueError as exc:
                logger.debug(
                    f"robots.txt unreachable for {cache_key}: "
                    f"malformed redirect Location {location!r}: {exc} — failing open"
                )
                return None
            continue
        if not (200 <= status < 300) or truncated:
            reason = "truncated body" if truncated else f"status {status}"
            logger.debug(f"robots.txt unreachable for {cache_key}: {reason} — failing open")
            return None
        try:
            text = _decode_body(body, headers.get("content-type", ""))
            parser = RobotFileParser()
            parser.parse(text.splitlines())
        except Exception as exc:  # noqa: BLE001 - broad by design: fails open
            logger.debug(f"robots.txt for {cache_key} could not be parsed: {exc} — failing open")
            return None
        return parser
    logger.debug(
        f"robots.txt unreachable for {cache_key}: "
        f"exceeded {FETCH_MAX_REDIRECTS} redirects — failing open"
    )
    return None


def _robots_allows(client: httpx.Client, url: str, user_agent: str) -> bool:
    """True if ``user_agent`` may fetch ``url`` per its host's robots.txt.

    Cache lookup keyed by a NORMALIZED ``scheme://host[:port]`` (fix round
    1, Minor 4 -- lowercase host, no userinfo, matching the per-domain rate
    limiter's own normalization via ``urlsplit(...).hostname``): the UA is
    only a parameter to the *query*, ``can_fetch()``, not the cache key --
    the fetched policy is shared across callers, and the constructed
    robots.txt URL must never carry credentials that happened to be present
    in the original ``url``. Fetch/parse/redirect-following and fail-open
    semantics live in ``_fetch_robots_parser``.

    IPv6 literals (fix round 2, Qodo finding): ``urlsplit(...).hostname``
    strips the ``[...]`` brackets an IPv6 literal needs in a URL, returning
    the bare address (e.g. ``2606:4700::1111``). Re-bracketing it here is
    NOT cosmetic -- without it, the colons land straight in the assembled
    ``scheme://host:port`` string, ``urljoin``/``urlsplit`` misparse the
    result downstream in ``_fetch_robots_parser``, ``_validate_hop`` on the
    constructed robots.txt URL raises, and the broad fail-open catch there
    silently disables robots enforcement for every IPv6-literal host --
    an input class ``validate_outbound_url`` explicitly supports (checked
    directly, no DNS) and that ``_robots_allows`` must therefore support
    too.

    Cache bookkeeping locked, fetch not (task-3770): the cache read/expiry
    check below runs under ``_robots_cache_lock`` (matching the module's
    ``_search_cache_lock`` idiom -- see its comment), but ``can_fetch()``
    and a cache-miss's ``_fetch_robots_parser`` FETCH both run outside any
    lock, single call per tool invocation; a cross-call stampede still
    costs at most one duplicate robots.txt fetch, accepted, unchanged from
    before this task.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if ":" in host:  # IPv6 literal: bracket it back for URL assembly
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port else ""
    cache_key = f"{parts.scheme.lower()}://{host}{port}"
    with _robots_cache_lock:
        cached = _robots_cache.get(cache_key)
        now = time.monotonic()
        cache_hit = cached is not None and now < cached[0]
        parser = cached[1] if cache_hit else None
    if not cache_hit:
        parser = _fetch_robots_parser(client, cache_key)
        _robots_cache_put(cache_key, parser)
    if parser is None:
        return True
    return parser.can_fetch(user_agent, url)


def _new_web_fetch_client() -> httpx.Client:
    return build_httpx_client(
        follow_redirects=False,
        timeout=FETCH_TIMEOUT_SECONDS,
        headers={"User-Agent": _USER_AGENT},
        transport=_transport,
        # trust_env=False: with HTTP(S)_PROXY set, the proxy does its own DNS
        # and connects on our behalf, bypassing the SSRF guard entirely.
        trust_env=False,
    )


# task-3260: robots.txt parity for web_deep_search's scrape path. A truthful
# bot identity, distinct from _USER_AGENT/_CRAWL_USER_AGENT -- scrape_article
# itself masquerades as Chrome (pre-existing FIXME, Article_Extractor_Lib.py:192,
# out of scope here), but checking robots.txt with that same UA would evade
# every bot-scoped Disallow group; a truthful UA matches sites' `*` groups
# the way a real crawler would.
_DEEP_SEARCH_ROBOTS_UA = "tldw-chatbook-deep-search/1.0"


def robots_allows_for_scrape(url: str) -> bool:
    """True if ``_DEEP_SEARCH_ROBOTS_UA`` may fetch ``url`` per its host's
    robots.txt -- public helper for web_deep_search's scrape path (task-3260).

    Builds a short-lived ``httpx.Client`` on the module ``_transport`` seam,
    mirroring ``_new_web_fetch_client()``: ``trust_env=False`` (an honored
    ``HTTP(S)_PROXY`` would otherwise do its own DNS and connect on this
    process's behalf, silently defeating ``validate_outbound_url``'s SSRF
    check on the robots.txt URL itself) and the same bounded
    ``FETCH_TIMEOUT_SECONDS`` timeout. Delegates to ``_robots_allows``, so
    the module robots cache (30-min TTL, negative caching, redirect-
    following, IPv6-bracket handling, and every other #1427 hardening) is
    shared with ``web_fetch``/``web_crawl`` for free -- a host consulted by
    one path warms the cache for the others too.

    Fail-open semantics come from ``_robots_allows``/``_fetch_robots_parser``
    themselves: an unreachable/unparsable robots.txt returns True (no
    restrictions) from THIS function. Callers that want "treat a call that
    raises (e.g. a caller-side timeout wrapping this synchronous call) as
    allowed too" must implement that themselves -- this function only
    covers the fetch-level fail-open, not a caller's own offload timeout.

    Args:
        url: The page URL about to be scraped (not the robots.txt URL --
            that is derived internally, same as every other robots caller
            in this module).

    Returns:
        bool: True if the URL may be fetched per its host's robots.txt (or
        the policy was unreachable/unparsable); False if a fetched, parsed
        policy disallows it for ``_DEEP_SEARCH_ROBOTS_UA``.
    """
    client = build_httpx_client(
        follow_redirects=False,
        timeout=FETCH_TIMEOUT_SECONDS,
        headers={"User-Agent": _DEEP_SEARCH_ROBOTS_UA},
        transport=_transport,
        trust_env=False,
    )
    try:
        return _robots_allows(client, url, _DEEP_SEARCH_ROBOTS_UA)
    finally:
        client.close()


def web_fetch(url: str, *, max_bytes: int = FETCH_MAX_BYTES) -> str:
    """Fetch ``url`` and return extracted text (trafilatura for HTML, PyMuPDF
    for PDF) or compact metadata for other allowlisted binary kinds.

    Args:
        url: public http(s) URL to fetch.
        max_bytes: response-read cap for text, clamped to
            [1, FETCH_HARD_MAX_BYTES]; for PDFs it caps the EXTRACTED text,
            not the download.

    SSRF-guarded per hop (validate_outbound_url), redirect-capped,
    rate-limited per domain, cached in-memory by (url, max_bytes) key
    (256-entry bound, earliest-expiry eviction, FETCH_CACHE_TTL_SECONDS expiry).

    Robots-guarded per hop too (task-2833): each hop's host robots.txt is
    checked for ``_USER_AGENT``, honoring ``[webfetch] respect_robots_txt``
    (default true, fail-open on an unreachable/unparsable robots.txt). A
    cache hit re-checks robots the same way it re-checks SSRF policy.

    HTML/plain-text extracted via trafilatura with fallback tag-strip;
    script/style tags removed. Result ends with a truncation marker when
    capped at max_bytes (default FETCH_MAX_BYTES=1 MiB).

    PDF detection: declared type "application/pdf" or %PDF- magic sniff.
    Text extracted via PyMuPDF if available. The 20 MB PDF ceiling applies only
    when pymupdf is installed; when it is absent the body still downloads under
    the ordinary ``max_bytes`` cap and the fetch is then refused with
    ``[missing-dep]``. Extracted text is truncated per-page if total exceeds
    max_bytes; the result includes page count.

    Binary-file support (task-1359, in-memory only, zero disk writes):
    images (``image/png|jpeg|gif|webp``), ZIP archives, and audio return
    compact METADATA, not contents or extracted bytes -- never file
    contents. Images are probed with Pillow (format/size only, no pixel
    decode, no EXIF); ZIPs are LISTED via stdlib zipfile (never extracted,
    never member-read); audio gets a one-line content-type + size summary
    (no new dependency). Detection mirrors the PDF precedent: an
    allowlisted declared content-type resolves immediately, and a magic-
    byte sniff overrides a wrong or absent declared type (audio has no
    magic and is declared-type-only). All three binary kinds share a
    10 MB refusal ceiling (``BINARY_MAX_BYTES``) -- a byte-truncated
    binary body is refused, never processed partially. Any other content
    type keeps the pre-existing ``[empty-content] unsupported content
    type`` refusal.

    All failures raise LocalToolError with structured reasons:
        - "invalid-url", "ssrf", "robots-disallowed", "redirect-limit",
          "timeout", "http-<status>", "rate-limited", "fetch-failed" (general fetch)
        - "empty-content" (unextractable HTML/text/PDF, or unsupported content type)
        - "missing-dep" (PDF requires pymupdf extra)
        - "pdf-error" (PyMuPDF extraction or encryption failure)
        - "image-error" (corrupt/unidentifiable image bytes)
        - "archive-error" (malformed ZIP bytes)
        - "too-large" (PDF exceeds 20 MB, or image/zip/audio exceeds 10 MB)

    Returns:
        str: extracted text — trafilatura/tag-strip for HTML, raw for plain
        types, pymupdf text for PDFs, compact metadata for image/zip/audio
        — with a truncation marker when a text response was capped.

    Raises:
        LocalToolError: on invalid/SSRF URLs, redirect overflow, timeouts,
            HTTP error statuses, rate limiting, PDF/image/archive processing
            errors, oversized binary bodies, or unextractable content.
    """
    if not isinstance(url, str) or not url.strip():
        raise LocalToolError("[invalid-url] url must be a non-empty string")
    url = url.strip()
    try:
        max_bytes = max(1, min(int(max_bytes), FETCH_HARD_MAX_BYTES))
    except (TypeError, ValueError) as exc:
        raise LocalToolError(f"[invalid-url] max_bytes must be an integer: {max_bytes!r}") from exc

    # Read once per invocation, not per hop (design doc ruling 6).
    respect_robots = _webfetch_settings()["respect_robots_txt"]

    # Lazy client (fix round 1, Minor 6): a warm cache hit with robots
    # enforcement OFF must do zero client setup, same as before robots.txt
    # support existed -- httpx.Client() is built only where first actually
    # needed (a robots.txt consult on a cache hit, or the real fetch below).
    client: "httpx.Client | None" = None
    try:
        # Lock scope carved precisely (task-3770): only the dict .get(),
        # expiry comparison, and .pop() run under _fetch_cache_lock --
        # _validate_hop (live DNS), the robots re-check (_robots_allows can
        # trigger a full robots fetch + client construction), and the
        # `return cache_hit_text` below all run AFTER release. Never hold
        # this lock across a network call.
        cache_hit_text: "str | None" = None
        with _fetch_cache_lock:
            cached = _fetch_cache.get((url, max_bytes))
            if cached is not None:
                expires_at, text = cached
                if time.monotonic() < expires_at:
                    cache_hit_text = text
                else:
                    _fetch_cache.pop((url, max_bytes), None)

        if cache_hit_text is not None:
            _validate_hop(url)  # re-check policy on cache hits (cheap, no body)
            # Robots re-checked exactly like _validate_hop above: rules
            # may have changed since the body was cached (design doc
            # ruling 3). Known hole shared with the _validate_hop check
            # right above (fix round 1, Minor 3): this judges the
            # REQUESTED url only -- a cached body that was actually
            # fetched via a redirect is keyed and re-checked here under
            # the ORIGINAL start url, not the final redirect target.
            if respect_robots:
                client = _new_web_fetch_client()
                if not _robots_allows(client, url, _USER_AGENT):
                    raise LocalToolError(_robots_disallowed_message(url))
            return cache_hit_text

        client = _new_web_fetch_client()
        # Probed ONCE per call, not per redirect hop: a mid-call sys.modules
        # mutation could otherwise make the pre-fetch cap decision and the
        # post-fetch [missing-dep]-vs-[too-large] verdict disagree with each
        # other (and find_spec gains nothing from re-probing every hop).
        pymupdf_ok = _pymupdf_available()
        current_url = url
        for _hop in range(FETCH_MAX_REDIRECTS + 1):
            # Policy re-checked on EVERY hop: a permitted URL must not be able
            # to redirect into private/denied address space.
            _validate_hop(current_url)
            # Robots consulted at the same position as _validate_hop, for
            # every hop (design doc ruling 3): a redirect into a disallowed
            # path is a disallowed fetch.
            if respect_robots and not _robots_allows(client, current_url, _USER_AGENT):
                raise LocalToolError(_robots_disallowed_message(current_url))
            _enforce_rate_limit(urlsplit(current_url).hostname or "unknown")
            status, headers, body, truncated, kind = _fetch_once(
                client, current_url, max_bytes,
                pdf_max_bytes=PDF_MAX_BYTES if pymupdf_ok else None,
                binary_max_bytes=BINARY_MAX_BYTES,
            )
            if status in _REDIRECT_STATUSES:
                location = headers.get("location")
                if not location:
                    raise LocalToolError(f"[http-{status}] redirect without a Location header")
                current_url = urljoin(current_url, location)
                continue
            break
        else:
            raise LocalToolError(
                f"[redirect-limit] exceeded {FETCH_MAX_REDIRECTS} redirects for {url!r}"
            )
    except httpx.TimeoutException as exc:
        raise LocalToolError(
            f"[timeout] fetch timed out after {FETCH_TIMEOUT_SECONDS}s: {url!r}"
        ) from exc
    except httpx.InvalidURL as exc:  # NOT an HTTPError subclass — catch explicitly
        raise LocalToolError(f"[invalid-url] {exc}") from exc
    except httpx.HTTPError as exc:
        raise LocalToolError(f"[fetch-failed] {exc}") from exc
    finally:
        if client is not None:
            client.close()

    if status >= 400:
        raise LocalToolError(f"[http-{status}] upstream returned status {status} for {url!r}")

    if kind == "pdf":
        if not pymupdf_ok:
            # Decided BEFORE the size check: pdf_max_bytes was already None
            # for this fetch (above), so `truncated` reflects the ordinary
            # max_bytes cap, not the 20 MB ceiling — a [too-large] verdict
            # here would be meaningless (and dishonest about the cap used).
            raise LocalToolError(
                "[missing-dep] PDF support requires pymupdf — pip install tldw_chatbook[pdf]"
            )
        if truncated:  # hit the 20 MB PDF ceiling: refuse, never truncate bytes
            raise LocalToolError(
                f"[too-large] PDF exceeds {PDF_MAX_BYTES // (1024 * 1024)} MB — "
                "use media ingestion for large documents"
            )
        text = _extract_pdf_text(body, max_bytes)
    elif kind in ("image", "zip", "audio"):
        if truncated:  # hit the shared binary ceiling: refuse, never truncate bytes
            raise LocalToolError(
                f"[too-large] {_BINARY_KIND_LABEL[kind]} exceeds "
                f"{BINARY_MAX_BYTES // (1024 * 1024)} MB — use media ingestion for large files"
            )
        text = _extract_text(body, headers.get("content-type", ""), kind=kind)
    else:
        text = _extract_text(body, headers.get("content-type", ""))
        if truncated:
            text += f"\n\n[... truncated: response exceeded max_bytes={max_bytes} ...]"
    _cache_put((url, max_bytes), text)
    return text


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

SEARCH_DEFAULT_ENGINE = "duckduckgo"
SEARCH_ENGINES = ("google", "bing", "duckduckgo", "brave", "kagi", "tavily", "searx", "exa", "serper", "yandex")
SEARCH_DEFAULT_RESULT_COUNT = 5
SEARCH_MAX_RESULT_COUNT = 10
# Byte budgets (re-plan spec §2.2), matching the provider's byte-based
# 32 KiB result fitting: per-result bound and a total cap comfortably under
# it, so provider fitting never triggers even for multibyte (CJK) content.
SEARCH_RESULT_MAX_BYTES = 4 * 1024
SEARCH_TOTAL_MAX_BYTES = 24 * 1024
# task-2832: identical searches in a session waste provider quota and
# latency. Same shape as _fetch_cache (15-min TTL, bounded, earliest-expiry
# eviction, cleared by _reset_state_for_tests). Key is the POST-coercion
# argument tuple — (engine, whitespace-collapsed casefolded query, count) —
# and ONLY the genuine success-blocks output is ever stored (the design
# doc enumerates web_search's five other return shapes, all transient-
# failure-adjacent; none may pin for the TTL).
SEARCH_CACHE_TTL_SECONDS = 900.0
SEARCH_CACHE_MAX_ENTRIES = 128
_search_cache: dict[tuple[str, str, int], tuple[float, str]] = {}
# Qodo PR #1444: tool calls each run on their own worker thread, so the
# eviction scan (min() ITERATES the dict) can race a concurrent put/pop
# into "dictionary changed size during iteration". The lock covers only
# the short cache ops — never the backend call. The two older caches
# (_fetch_cache/_robots_cache) share this race pre-existing: task-3770.
_search_cache_lock = threading.Lock()


def _search_cache_put(key: tuple[str, str, int], text: str) -> None:
    """Insert into the search cache, evicting earliest-expiry at capacity."""
    with _search_cache_lock:
        if key not in _search_cache and len(_search_cache) >= SEARCH_CACHE_MAX_ENTRIES:
            oldest = min(_search_cache, key=lambda k: _search_cache[k][0])
            _search_cache.pop(oldest, None)
        _search_cache[key] = (time.monotonic() + SEARCH_CACHE_TTL_SECONDS, text)


_TRUNCATED_MARKER = "… [truncated]"


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    """Cap ``text`` at ``max_bytes`` of UTF-8, never splitting a codepoint."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore") + _TRUNCATED_MARKER


def web_search(
    query: str,
    *,
    search_engine: str = SEARCH_DEFAULT_ENGINE,
    result_count: int = SEARCH_DEFAULT_RESULT_COUNT,
) -> str:
    """Run a web search and return bounded, formatted results as text.

    Delegates to ``Web_Scraping.WebSearch_APIs.perform_websearch`` with the
    legacy ``Tools/web_search_tool.py`` config-default wiring (country US,
    English in/out, moderate safesearch, no advanced filters). Each result
    block is bounded to SEARCH_RESULT_MAX_BYTES and the whole output to
    SEARCH_TOTAL_MAX_BYTES (both UTF-8 byte budgets), so the provider's
    32 KiB byte fitting never triggers on search output. Backend failures
    and error envelopes return an error string rather than raising (legacy
    tool contract); only invalid arguments raise LocalToolError.

    Successful results are cached for SEARCH_CACHE_TTL_SECONDS keyed by
    the post-coercion (engine, normalized query, count) — identical
    searches within a session stop re-billing the provider (task-2832).
    Failure shapes and confirmed-empty results are never cached.

    Raises:
        LocalToolError: if ``query`` is empty.
    """
    if not isinstance(query, str) or not query.strip():
        raise LocalToolError("[invalid-args] query must be a non-empty string")
    query = query.strip()
    # Coerced like result_count below: garbage input degrades to the default.
    if isinstance(search_engine, str) and search_engine.strip():
        engine = search_engine.strip().lower()
    else:
        engine = SEARCH_DEFAULT_ENGINE
    try:
        count = int(result_count)
    except (TypeError, ValueError):
        count = SEARCH_DEFAULT_RESULT_COUNT
    if count < 1 or count > SEARCH_MAX_RESULT_COUNT:
        count = SEARCH_DEFAULT_RESULT_COUNT

    # Cache check AFTER validation/coercion (invalid args raise without
    # touching the cache), BEFORE the backend import/call. First populator's
    # raw query text wins for everyone sharing the normalized key — the
    # design doc records the trade-off.
    cache_key = (engine, " ".join(query.split()).casefold(), count)
    with _search_cache_lock:
        cached = _search_cache.get(cache_key)
        if cached is not None:
            expires_at, cached_text = cached
            if time.monotonic() < expires_at:
                return cached_text
            # pop-not-del (review Minor 3): concurrent tool threads can
            # both observe the same expired entry; the loser's del would
            # KeyError. The lock makes the observe+pop atomic anyway.
            _search_cache.pop(cache_key, None)

    # Local import: WebSearch_APIs pulls the config/metrics stack; keep this
    # module cheap to import and let tests monkeypatch the source attribute.
    from ..Web_Scraping.WebSearch_APIs import perform_websearch

    try:
        results = perform_websearch(
            search_engine=engine,
            search_query=query,
            content_country="US",
            search_lang="en",
            output_lang="en",
            result_count=count,
            date_range=None,
            safesearch="moderate",
            site_blacklist=None,
            exactTerms=None,
            excludeTerms=None,
            filter=None,
            geolocation=None,
            search_result_language=None,
            sort_results_by=None,
        )
    except Exception as exc:  # noqa: BLE001 — backend failure is a result string, not an exception
        logger.warning(f"web_search backend failure via {engine!r}: {exc}")
        return f"[search-failed] web search via {engine!r} failed: {exc}"

    if not isinstance(results, dict):
        return (
            f"No results found or unexpected response format from {engine!r} "
            f"(raw: {str(results)[:500]})"
        )
    # A well-formed envelope can still carry a failure: surface THAT reason.
    reason = results.get("processing_error") or results.get("error")
    if reason:
        return f"[search-failed] web search via {engine!r} reported an error: {reason}"
    if not isinstance(results.get("results"), list):
        return (
            f"No results found or unexpected response format from {engine!r} "
            f"(raw: {str(results)[:500]})"
        )
    items = [item if isinstance(item, dict) else {} for item in results["results"][:count]]
    if not items:
        return f"No results found for {query!r} via {engine!r}."

    blocks: list[str] = []
    total_bytes = 0
    for i, item in enumerate(items, 1):
        # Real standardized shape (process_web_search_results): body text is
        # top-level "content"; "snippet" lives under metadata. Accept both.
        snippet = item.get("snippet") or item.get("content") or "No description available"
        block = (
            f"{i}. {item.get('title') or 'No title'}\n"
            f"   URL: {item.get('url') or ''}\n"
            f"   {snippet}"
        )
        block = _truncate_to_bytes(block, SEARCH_RESULT_MAX_BYTES)
        block_bytes = len(block.encode("utf-8"))
        if total_bytes + block_bytes > SEARCH_TOTAL_MAX_BYTES:
            blocks.append("… [further results omitted: total size cap reached]")
            break
        blocks.append(block)
        total_bytes += block_bytes
    output = "\n\n".join(blocks)
    # The ONE cacheable point (design doc ruling 1): only the genuine
    # success-blocks output is stored — never the [search-failed] strings,
    # the unmarked malformed-response strings, or the confirmed-empty
    # message (a transient zero must not pin for the TTL).
    _search_cache_put(cache_key, output)
    return output


# ---------------------------------------------------------------------------
# web_crawl (spec 2026-08-06 §2)
# ---------------------------------------------------------------------------

CRAWL_DEFAULT_MAX_PAGES = 20
CRAWL_MAX_PAGES_CEILING = 40
CRAWL_DEFAULT_MAX_DEPTH = 2
CRAWL_MAX_DEPTH_CEILING = 5
CRAWL_DEADLINE_SECONDS = 120.0
CRAWL_PAGE_TIMEOUT_SECONDS = 10.0   # per page; a hung page must not eat the crawl
CRAWL_EXCERPT_MAX_CHARS = 200
CRAWL_RESULT_MAX_BYTES = 24 * 1024
CRAWL_BLOCK_MAX_BYTES = 1024
CRAWL_MAX_LINKS_PER_PAGE = 500      # frontier bound: cap links enqueued FROM one page
CRAWL_TITLE_MAX_CHARS = 512         # bound on <title> accumulation (see _CrawlLinkParser.handle_data)
SITEMAP_MAX_BYTES = 5 * 1024 * 1024
SITEMAP_MAX_CHILDREN = 20           # cap child sitemaps actually fetched from an index

_CRAWL_USER_AGENT = "tldw-chatbook-web-crawl/1.0"

_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


class _CrawlLinkParser(HTMLParser):
    """Collect <a href>, <base href>, and <title> text from one page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.base_href: "str | None" = None
        self.title: str = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        elif tag == "base" and self.base_href is None:
            href = dict(attrs).get("href")
            if href:
                self.base_href = href
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            # Bounded: an unclosed <title> would otherwise concatenate the
            # rest of the page's text data unboundedly (handle_data fires
            # once per chunk of text). The excerpt/title only ever needs a
            # display-sized prefix.
            self.title = (self.title + data)[:CRAWL_TITLE_MAX_CHARS]


def _crawl_host(url: str) -> str:
    """Lowercased host with a leading ``www.`` folded; '' when absent/bad."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _normalize_crawl_url(url: str) -> str:
    """Visited-set identity: scheme+folded host+path+query, no fragment.

    On malformed URLs (bad port, invalid IPv6, etc.), returns the input unchanged
    for stable visited-set identity; downstream egress guard rejects them as invalid.
    """
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        port = f":{parts.port}" if parts.port else ""
        path = parts.path or "/"
        query = f"?{parts.query}" if parts.query else ""
        return f"{parts.scheme.lower()}://{host}{port}{path}{query}"
    except ValueError:
        # Malformed URL (e.g., bad port, invalid IPv6): return unchanged
        return url


def _coerce_budget(value, default: int, ceiling: int) -> int:
    """v1 argument style: garbage degrades to the default, range clamps."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(result, ceiling))


def _parse_sitemap(xml_bytes: bytes) -> tuple[list[str], list[str]]:
    """Return (page_urls, child_sitemap_urls) from a urlset/sitemapindex.

    Sitemaps without the sitemaps.org namespace (some generators emit
    these) are also accepted: the namespaced findall is tried first, and
    an un-namespaced `.//loc` fallback runs only when it comes back empty.
    """
    try:
        root = xET.fromstring(xml_bytes)
    # defusedxml's refusals (EntitiesForbidden, etc.) subclass ValueError,
    # not xET.ParseError, so a hardening refusal must be caught here too —
    # otherwise it escapes as a raw, untyped exception instead of the
    # structured [crawl-failed] every other sitemap failure produces.
    except (xET.ParseError, ValueError) as exc:
        raise LocalToolError(f"[crawl-failed] sitemap could not be parsed: {exc}") from exc
    locs = [
        loc.text.strip()
        for loc in root.findall(f".//{_SITEMAP_NS}loc")
        if loc.text and loc.text.strip()
    ]
    if not locs:
        locs = [
            loc.text.strip()
            for loc in root.findall(".//loc")
            if loc.text and loc.text.strip()
        ]
    if root.tag.rsplit("}", 1)[-1] == "sitemapindex":
        return [], locs
    return locs, []


def _format_crawl_result(
    pages: list[dict],
    failed: int,
    blocked: int,
    stop_reason: str,
    children_skipped: int = 0,
    duplicates_skipped: int = 0,
    robots_disallowed: int = 0,
) -> str:
    blocks: list[str] = []
    total = 0
    for i, page in enumerate(pages, 1):
        if page["marker"]:
            block = f"{i}. {page['marker']}\n   URL: {page['url']}"
        else:
            block = f"{i}. {page['title'] or 'No title'}\n   URL: {page['url']}"
            if page["excerpt"]:
                block += f"\n   {page['excerpt']}"
        block = _truncate_to_bytes(block, CRAWL_BLOCK_MAX_BYTES)
        block_bytes = len(block.encode("utf-8"))
        if total + block_bytes > CRAWL_RESULT_MAX_BYTES:
            blocks.append("… [further pages omitted: total size cap reached]")
            break
        blocks.append(block)
        total += block_bytes
    counts = f"{failed} failed, {blocked} blocked"
    if robots_disallowed > 0:
        counts += f"; {robots_disallowed} robots-disallowed"
    if children_skipped > 0:
        counts += f"; {children_skipped} child sitemaps skipped"
    if duplicates_skipped > 0:
        counts += f"; {duplicates_skipped} duplicate redirects skipped"
    footer = f"Crawled {len(pages)} pages ({counts}). Stopped: {stop_reason}."
    return "\n\n".join(blocks + [footer]) if blocks else footer


class _CrawlDeadline(Exception):
    """Internal: the wall-clock budget expired mid-fetch."""


def _crawl_fetch_page(
    client: httpx.Client,
    url: str,
    deadline: float,
    *,
    max_bytes: int = FETCH_MAX_BYTES,
    html_only: bool = True,
    respect_robots: bool = True,
) -> tuple[str, "httpx.Headers", bytes, bool, "str | None"]:
    """Guarded, rate-limited GET with the crawl's redirect loop.

    Returns (final_url, headers, body, truncated, kind). Checks the
    deadline between redirect hops — one page's full chain must not
    overshoot the crawl budget by minutes (spec §2). ``kind`` is
    ``_fetch_once``'s sniff+declared-type result and MUST be consulted by
    callers before treating a response as HTML: a PDF mislabeled (or
    unlabeled) as text/html must not fall through to HTML extraction, or
    its raw bytes become the page excerpt and its garbage "extracted text"
    warm-writes the shared web_fetch cache (see web_crawl's marker
    branch). Binary-fetch design doc non-goal: crawl's own marker branch
    only special-cases ``kind == "pdf"`` -- other binary kinds (image/zip/
    audio) sniffed here are NOT rescued into a marker or into metadata
    extraction; they keep today's mojibake-decode behavior, unchanged.

    This is also the path every sitemap fetch takes (task-2833 design doc
    ruling 4: root + child sitemap fetches go through this same loop), so
    the ``respect_robots`` check below covers those for free -- always
    against ``_CRAWL_USER_AGENT``, the one UA this whole module uses for
    crawl-initiated fetches including sitemaps.
    """
    current = url
    for _hop in range(FETCH_MAX_REDIRECTS + 1):
        if time.monotonic() >= deadline:
            raise _CrawlDeadline()
        _validate_hop(current)
        # Robots consulted at the same position as _validate_hop, for every
        # hop (design doc ruling 3).
        if respect_robots and not _robots_allows(client, current, _CRAWL_USER_AGENT):
            raise LocalToolError(_robots_disallowed_message(current))
        _enforce_rate_limit(urlsplit(current).hostname or "unknown")
        try:
            status, headers, body, truncated, kind = _fetch_once(
                client, current, max_bytes, html_only=html_only
            )
        except httpx.TimeoutException as exc:
            raise LocalToolError(f"[timeout] fetch timed out: {current!r}") from exc
        except httpx.InvalidURL as exc:
            raise LocalToolError(f"[invalid-url] {exc}") from exc
        except httpx.HTTPError as exc:
            raise LocalToolError(f"[fetch-failed] {exc}") from exc
        if status in _REDIRECT_STATUSES:
            location = headers.get("location")
            if not location:
                raise LocalToolError(f"[http-{status}] redirect without a Location header")
            try:
                current = urljoin(current, location)
            except ValueError as exc:
                raise LocalToolError(f"[invalid-url] malformed redirect Location: {location!r}") from exc
            continue
        if status >= 400:
            raise LocalToolError(f"[http-{status}] upstream returned status {status} for {current!r}")
        return current, headers, body, truncated, kind
    raise LocalToolError(f"[redirect-limit] exceeded {FETCH_MAX_REDIRECTS} redirects for {url!r}")


class _SitemapSeed(NamedTuple):
    """Result of consulting a sitemap (or sitemapindex) for page URLs.

    children_capped: True when the SITEMAP_MAX_CHILDREN break fired, so the
        caller can report an honest stop reason instead of claiming the
        sitemap was exhausted when children were actually left unfetched.
    budget_truncated: True when `take()`'s max_pages cap stopped with
        candidates (page URLs or, in the child loop, child sitemaps)
        actually left over — i.e. the seed was cut short by the page
        budget, not because the sitemap was fully consulted. False when
        every candidate was considered, even if that consumed exactly
        max_pages slots.
    children_skipped: count of child sitemaps that were fetched-or-attempted
        but excluded from `urls` for a reason OTHER than the host filter or
        the SITEMAP_MAX_CHILDREN cap — i.e. every `continue` below that
        represents a child that failed to contribute: fetch/redirect error,
        a deadline expiry mid-fetch, oversized body, or a parse refusal
        (including a defusedxml hardening refusal).
    """

    urls: list[str]
    children_capped: bool
    budget_truncated: bool
    children_skipped: int


def _seed_from_sitemap(
    client: httpx.Client,
    sitemap_url: str,
    scope_host: str,
    max_pages: int,
    deadline: float,
    respect_robots: bool = True,
) -> _SitemapSeed:
    """Collect up to max_pages same-host page URLs from a sitemap.

    Sitemap fetches are discovery overhead — they do NOT consume the page
    budget; the deadline bounds a pathological index (spec §2). Host rules:
    child sitemaps must share sitemap_url's host; page URLs must share the
    crawl scope host.

    ``respect_robots`` is threaded into every ``_crawl_fetch_page`` call
    below (root sitemap and each child sitemap) -- a robots-disallowed
    root sitemap propagates uncaught out of this function (the caller
    wraps it into the structured ``[crawl-failed] sitemap could not be
    fetched: [robots-disallowed] ...`` refusal, matching every other
    seed-failure type); a disallowed child sitemap is caught by the
    existing broad ``except (LocalToolError, _CrawlDeadline)`` below and
    simply counted in ``children_skipped``, same as any other child
    fetch failure.
    """
    final_url, _headers, body, truncated, _kind = _crawl_fetch_page(
        client, sitemap_url, deadline, max_bytes=SITEMAP_MAX_BYTES, html_only=False,
        respect_robots=respect_robots,
    )
    if truncated:
        raise LocalToolError(f"[crawl-failed] sitemap exceeds {SITEMAP_MAX_BYTES} bytes: {sitemap_url!r}")
    page_urls, children = _parse_sitemap(body)
    sitemap_host = _crawl_host(final_url)

    urls: list[str] = []
    seen: set[str] = set()
    budget_truncated = False

    def take(candidates: list[str]) -> None:
        nonlocal budget_truncated
        for candidate in candidates:
            # Host/dedup filters run BEFORE the budget check: a trailing
            # off-host or duplicate candidate would be discarded anyway, so
            # it must not flip budget_truncated and claim the page budget —
            # not the sitemap itself — cut the seed short.
            if _crawl_host(candidate) != scope_host:
                continue
            norm = _normalize_crawl_url(candidate)
            if norm in seen:
                continue
            if len(urls) >= max_pages:
                # This candidate (and anything after it that would pass the
                # filters above) was never considered: the cap, not
                # exhaustion, ended this pass.
                budget_truncated = True
                return
            seen.add(norm)
            urls.append(candidate)

    take(page_urls)
    children_fetched = 0
    children_capped = False
    children_skipped = 0
    for child in children:
        if time.monotonic() >= deadline:
            break
        # Off-host filter runs BEFORE the budget check, mirroring take(): a
        # trailing off-host child would be skipped regardless, so it must
        # not flip budget_truncated on its own.
        if _crawl_host(child) != sitemap_host:
            continue
        if len(urls) >= max_pages:
            budget_truncated = True
            break
        if children_fetched >= SITEMAP_MAX_CHILDREN:
            # Amplification guard: a pathological same-host index (~119
            # children measured in the review, ~600 MB at 5 MiB each) is
            # bounded here instead of relying solely on the deadline.
            children_capped = True
            break
        children_fetched += 1
        try:
            _f, _h, child_body, child_truncated, _kind = _crawl_fetch_page(
                client, child, deadline, max_bytes=SITEMAP_MAX_BYTES, html_only=False,
                respect_robots=respect_robots,
            )
        except (LocalToolError, _CrawlDeadline):
            children_skipped += 1
            continue
        if child_truncated:
            children_skipped += 1
            continue
        try:
            child_pages, _nested = _parse_sitemap(child_body)  # one level: nested indexes ignored
        except LocalToolError:
            children_skipped += 1
            continue
        take(child_pages)
    return _SitemapSeed(urls, children_capped, budget_truncated, children_skipped)


def web_crawl(
    url: str,
    *,
    max_pages: int = CRAWL_DEFAULT_MAX_PAGES,
    max_depth: int = CRAWL_DEFAULT_MAX_DEPTH,
    sitemap_url: "str | None" = None,
) -> str:
    """Same-host breadth-first crawl returning a bounded page list.

    Each listed page carries URL, title, and a short excerpt; the model is
    expected to follow up with web_fetch on pages that matter (spec §2).
    Every URL is egress-guarded; budgets bound fetch ATTEMPTS; a wall-clock
    deadline bounds the whole crawl. Ephemeral: no database writes.

    Robots-guarded (task-2833): every hop -- BFS pages, and sitemap fetches
    (root + children) since they share the same fetch loop -- is checked
    against its host's robots.txt for ``_CRAWL_USER_AGENT``, honoring
    ``[webfetch] respect_robots_txt`` (default true). A disallowed page or
    child sitemap is SKIPPED and counted (footer's "robots-disallowed"
    clause), not fatal; a disallowed start URL or root sitemap is fatal,
    same as any other seed-fetch failure (see Raises below).

    Attempt/row invariant: A row-less attempt arises two ways: a redirect
    that lands on an already-listed final URL, or a plain fetch of a URL
    that an earlier page's redirect already listed (which occurs first
    depends on discovery order); both are surfaced in the footer's
    duplicate-redirects clause. Both cases are deduped against the same
    `listed` set, so "Crawled N pages" can legitimately be smaller than the
    number of fetch attempts spent.

    When ``sitemap_url`` is given, sitemap mode replaces link-discovery BFS:
    the page list comes from the sitemap (urlset, or a one-level
    sitemapindex); ``max_depth`` is ignored and links on seeded pages are
    not expanded.

    Args:
        url: Start URL; its host (www-folded) defines the crawl scope in
            both modes.
        max_pages: Fetch-attempt budget, clamped to
            [1, CRAWL_MAX_PAGES_CEILING]; garbage coerces to the default.
        max_depth: BFS link depth from the start URL (start = 0), clamped to
            [1, CRAWL_MAX_DEPTH_CEILING]; ignored in sitemap mode.
        sitemap_url: Optional sitemap URL; when given, the page list is
            seeded from the sitemap and links are not expanded.

    Returns:
        str: Numbered page list (URL, title, excerpt or type marker),
            byte-capped, ending with the status footer described above.

    Raises:
        LocalToolError: [invalid-args] for a bad url/host or a blank
            sitemap_url; [crawl-failed] when the START url cannot be
            fetched in BFS mode, or when the sitemap itself cannot be
            fetched/parsed (per-page failures inside the crawl are
            results, counted in the footer).
    """
    if not isinstance(url, str) or not url.strip():
        raise LocalToolError("[invalid-args] url must be a non-empty string")
    url = url.strip()
    max_pages = _coerce_budget(max_pages, CRAWL_DEFAULT_MAX_PAGES, CRAWL_MAX_PAGES_CEILING)
    max_depth = _coerce_budget(max_depth, CRAWL_DEFAULT_MAX_DEPTH, CRAWL_MAX_DEPTH_CEILING)
    scope_host = _crawl_host(url)
    if not scope_host:
        raise LocalToolError(f"[invalid-args] url has no host: {url!r}")

    # Read once per invocation, not per hop (design doc ruling 6).
    respect_robots = _webfetch_settings()["respect_robots_txt"]

    deadline = time.monotonic() + CRAWL_DEADLINE_SECONDS
    queue: "deque[tuple[str, int]]" = deque([(url, 0)])
    visited = {_normalize_crawl_url(url)}
    listed: set[str] = set()  # normalized final URLs actually appended to `pages`
    pages: list[dict] = []
    failed = blocked = 0
    robots_disallowed = 0
    attempts = 0
    stop_reason = "no more links within depth"
    children_skipped = 0
    duplicates_skipped = 0

    client = build_httpx_client(
        follow_redirects=False,
        timeout=CRAWL_PAGE_TIMEOUT_SECONDS,
        headers={"User-Agent": _CRAWL_USER_AGENT},
        transport=_transport,
        trust_env=False,
    )
    try:
        expand_links = sitemap_url is None
        if sitemap_url is not None:
            if not isinstance(sitemap_url, str) or not sitemap_url.strip():
                raise LocalToolError("[invalid-args] sitemap_url must be a non-empty string")
            try:
                seed = _seed_from_sitemap(
                    client, sitemap_url.strip(), scope_host, max_pages, deadline,
                    respect_robots=respect_robots,
                )
            except _CrawlDeadline:
                seed = _SitemapSeed(urls=[], children_capped=False, budget_truncated=False, children_skipped=0)
            except LocalToolError as exc:
                if "[crawl-failed]" in str(exc):
                    raise
                raise LocalToolError(f"[crawl-failed] sitemap could not be fetched: {exc}") from exc
            queue = deque((u, 0) for u in seed.urls)
            visited = {_normalize_crawl_url(u) for u in seed.urls}
            children_skipped = seed.children_skipped
            # Four non-exceptional paths can leave `seed.urls` short/empty for
            # a reason other than "the sitemap was fully consulted": the clock
            # ran out (root fetch's _CrawlDeadline, caught above, and the
            # child-sitemap loop's plain `break` on time.monotonic() >=
            # deadline both leave the clock past the deadline, read back
            # here); the SITEMAP_MAX_CHILDREN break fired and left child
            # sitemaps unfetched (children_capped); or `take()`'s max_pages
            # cap left page-URL or child-sitemap candidates unconsidered
            # (budget_truncated). Priority reflects which budget is "harder":
            # deadline (wall-clock, non-negotiable) > children_capped (an
            # amplification guard) > budget_truncated (the ordinary page
            # budget) > exhausted (every candidate was actually considered).
            if time.monotonic() >= deadline:
                stop_reason = "deadline reached"
            elif seed.children_capped:
                stop_reason = "sitemap child budget reached"
            elif seed.budget_truncated:
                stop_reason = "page budget reached"
            else:
                stop_reason = "sitemap exhausted"

        while queue:
            if attempts >= max_pages:
                stop_reason = "page budget reached"
                break
            if time.monotonic() >= deadline:
                stop_reason = "deadline reached"
                break
            current, depth = queue.popleft()
            is_start = attempts == 0
            attempts += 1
            try:
                final_url, headers, body, truncated, kind = _crawl_fetch_page(
                    client, current, deadline, respect_robots=respect_robots
                )
            except _CrawlDeadline:
                stop_reason = "deadline reached"
                break
            except LocalToolError as exc:
                if is_start and sitemap_url is None:
                    raise LocalToolError(f"[crawl-failed] start URL could not be fetched: {exc}") from exc
                # Prefix check, not substring: _validate_hop/_robots_allows put
                # the reason at position 0 of THEIR message, but a URL echoed
                # into an unrelated error (e.g. an http-404 message quoting
                # the failing URL) can contain the literal text "[ssrf]" or
                # "[robots-disallowed]" anywhere in the string without being
                # that kind of refusal.
                if str(exc).startswith("[ssrf]"):
                    blocked += 1
                elif str(exc).startswith("[robots-disallowed]"):
                    robots_disallowed += 1
                else:
                    failed += 1
                continue
            final_norm = _normalize_crawl_url(final_url)
            if final_norm in listed:
                # This exact final URL was already appended to `pages` —
                # via its own fetch or another page's redirect onto it.
                # `visited` is NOT the right set to dedup against here: it
                # holds every ENQUEUED url (marked at discovery time), so a
                # page that redirects onto a separately-enqueued sibling
                # link would otherwise be discarded even though nothing had
                # listed it yet — the start page's own final URL can't be
                # in `listed` before it's listed, so no `!= current` carve-out
                # is needed here. Counted (item 3): this attempt spent a
                # budget slot but produced no row, otherwise invisible to
                # the model — surfaced via the footer's "N duplicate
                # redirects skipped" clause.
                duplicates_skipped += 1
                continue
            visited.add(final_norm)

            ctype = (headers.get("content-type") or "").split(";", 1)[0].strip().lower()
            # kind == "pdf" (the %PDF- sniff, or the declared-type shortcut)
            # takes priority over the declared content-type: a PDF
            # mislabeled as text/html (or unlabeled) must never fall
            # through to HTML extraction below, or its raw bytes become
            # the excerpt and its "extracted text" warm-writes the shared
            # web_fetch cache with binary garbage. Matches the spec's own
            # detection rule ("the sniff wins over the declared type" §1):
            # when kind is "pdf" the marker is always "[application/pdf]",
            # regardless of what the server claimed; only a genuinely
            # non-PDF, non-HTML response is labeled with its own declared
            # type. `ctype` is guaranteed non-empty on the else side (the
            # `or` branch below required it truthy to enter this block at
            # all), so no `ctype or ...` fallback is needed. Deliberately
            # NOT generalized to "any recognized binary kind" (binary-fetch
            # design doc non-goal): a page whose body sniffs as image/zip
            # but is declared text/html (or unlabeled) is NOT rescued into
            # a marker here -- it falls through to the HTML/decode path
            # below exactly as it did before kind detection existed for
            # those other kinds, keeping crawl's pre-existing mojibake-
            # decode behavior for that edge case unchanged.
            if kind == "pdf" or (ctype and ctype not in _HTML_TYPES):
                marker = "[application/pdf]" if kind == "pdf" else f"[{ctype}]"
                pages.append({"url": final_url, "title": "", "excerpt": "", "marker": marker})
                listed.add(final_norm)
                continue

            html = _decode_body(body, headers.get("content-type", ""))
            parser = _CrawlLinkParser()
            try:
                parser.feed(html)
                parser.close()
            except Exception:  # noqa: BLE001 — keep whatever was collected
                pass
            try:
                # kind deliberately NOT passed here (see the comment on the
                # marker branch above): crawl never rescues a sniffed
                # non-PDF binary kind into metadata extraction either.
                full_text = _extract_text(body, headers.get("content-type", ""))
            except LocalToolError:
                full_text = ""
            if full_text and kind not in ("image", "zip", "audio"):
                # Parity with web_fetch: a body already sliced to FETCH_MAX_BYTES
                # must carry the same marker web_fetch would have appended, or a
                # default web_fetch() cache hit silently hands back a cut page.
                # Sniffed binary kinds are excluded from the warm-write
                # (task-3280 / Qodo PR #1442): crawl's mojibake decode of a
                # mislabeled binary must not occupy the cache key web_fetch
                # reads, or web_fetch returns garbage instead of its binary
                # metadata shape for the whole cache TTL.
                cache_text = full_text
                if truncated:
                    cache_text += f"\n\n[... truncated: response exceeded max_bytes={FETCH_MAX_BYTES} ...]"
                _cache_put((final_url, FETCH_MAX_BYTES), cache_text)
            pages.append({
                "url": final_url,
                "title": parser.title.strip(),
                "excerpt": full_text[:CRAWL_EXCERPT_MAX_CHARS].strip(),
                "marker": None,
            })
            listed.add(final_norm)

            # Expansion: same-host pages only, within the depth budget. A page
            # that redirected off-host is listed but its links are not followed.
            # Sitemap mode is discovery-complete: links are never expanded.
            if not expand_links:
                continue
            if depth >= max_depth or _crawl_host(final_url) != scope_host:
                continue
            base = final_url
            if parser.base_href:
                try:
                    base = urljoin(final_url, parser.base_href)
                except ValueError:
                    base = final_url  # malformed <base href>: fall back to the page URL
            enqueued_from_page = 0
            for href in parser.links:
                if enqueued_from_page >= CRAWL_MAX_LINKS_PER_PAGE:
                    # Frontier bound: a hostile/spammy page must not grow
                    # visited/queue without limit (52,975 entries measured
                    # from one page in the review).
                    break
                try:
                    absolute = urljoin(base, href)
                    scheme = urlsplit(absolute).scheme.lower()
                except ValueError:
                    continue  # malformed href (e.g. unbalanced IPv6 brackets): skip it
                if scheme not in _ALLOWED_SCHEMES:
                    continue
                if _crawl_host(absolute) != scope_host:
                    continue
                norm = _normalize_crawl_url(absolute)
                if norm in visited:
                    continue
                visited.add(norm)
                queue.append((absolute, depth + 1))
                enqueued_from_page += 1
    finally:
        client.close()

    return _format_crawl_result(
        pages, failed, blocked, stop_reason,
        children_skipped=children_skipped, duplicates_skipped=duplicates_skipped,
        robots_disallowed=robots_disallowed,
    )


# ---------------------------------------------------------------------------
# web_deep_search (task-1356 Task 5)
# ---------------------------------------------------------------------------

# 10 KiB / 15 KiB, not 16/24 (task-1356 review round 2, N4 + the finding-5
# residual -- RULING): the agent runtime truncates a tool RESULT to
# RunBudget.max_tool_result_chars = 16,000 chars HEAD-FIRST
# (Agents/agent_runtime.py:325-357 -- `content[:max_chars]` plus a trailer,
# never the tail), which would silently eat the Sources block and this
# tool's own honesty footer (confidence/warnings/deadline disclosure) off
# the END of any output that grew past that ceiling -- the footer is theater
# if it can't survive the delivery seam. DEEP_SEARCH_TOTAL_MAX_BYTES now
# really is a TOTAL: it bounds the combined answer + Sources block + footer
# (previously it capped the Sources block alone, which is why the "total"
# name was a lie -- an answer near its own 16 KiB cap plus a 24 KiB Sources
# block could still exceed 16,000 chars). 15 KiB keeps the whole output
# under the 16,000-char ceiling with headroom.
DEEP_SEARCH_ANSWER_MAX_BYTES = 10 * 1024
DEEP_SEARCH_SOURCES_MAX = 20
DEEP_SEARCH_TOTAL_MAX_BYTES = 15 * 1024
# Grace period asyncio.wait_for() adds on top of the remaining deadline when
# awaiting analyze_and_aggregate. What this DOES bound: awaited work that
# keeps yielding control back to the event loop between awaits (e.g. every
# `await` inside search_result_relevance's per-result loop) -- the pipeline
# has no circuit breaker (a dead relevance-LLM provider costs N sequential
# llm_timeout_s calls), so cooperative cancellation via cancel_event is the
# primary stop signal and this wait_for grace is the backstop for a call
# that never checks it, PROVIDED it still yields. What this does NOT bound:
# a synchronous call that blocks the loop thread outright -- wait_for's
# timeout is delivered via a scheduled callback that can only run once the
# loop is idle, and a blocking call never gives it that chance (task-1356
# review, reproduced live: a blocking aggregate_results call made this
# grace period a no-op, silently returning a LATE result instead of timing
# out). That is why analyze_and_aggregate offloads its one genuinely
# synchronous step via asyncio.to_thread. Bounding any ONE blocking call's
# own wall-clock time (an LLM transport's connect/read timeout, say) is
# that call's own job, not this watchdog's -- and if a pipeline call
# somehow still doesn't yield in time, _run_coro_loop_safe's thread-join
# path is the one backstop that CAN preempt it (see its docstring).
_DEEP_SEARCH_DEADLINE_GRACE_S = 30.0
# Extra slack given to the loop-safe runner's thread.join() backstop on top
# of the coroutine's own wait_for(deadline + grace) so the join timeout
# never races the coroutine's own timeout under normal scheduling jitter.
_DEEP_SEARCH_THREAD_JOIN_SLACK_S = 5.0
# Additional margin on top of the two constants above, folded into
# deep_search_outer_timeout_s() below -- covers ordinary scheduling jitter
# between the agent runtime's own timer and this tool's internal one (two
# independent clocks measuring "the same" deadline never fire at the exact
# same instant).
_DEEP_SEARCH_SCHEDULING_JITTER_S = 15.0


def _deep_search_settings() -> dict:
    """Resolve the ``[SearchSettings]`` config keys web_deep_search needs.

    A dedicated module function (rather than inline ``get_cli_setting``
    calls) so tests can monkeypatch config resolution wholesale instead of
    stubbing individual keys. Defaults mirror task-4's config revival.

    ``get_cli_setting("SearchSettings", key, default)`` reads the RAW TOML
    value with no type validation of its own -- it is NOT the same as the
    typed ``search_settings_general`` bucket ``config.load_settings()``
    builds (this seam deliberately bypasses that bucket, since it uses
    different key names for some fields and this tool wants direct
    ``[SearchSettings]`` control). Left unvalidated, a malformed TOML value
    reaches the tool unfiltered (task-1356 review, three reproduced harms):
    a non-numeric timeout crashes ``float(...)`` with a bare ``ValueError``;
    a quoted ``search_enable_subquery = "false"`` reads as Python-truthy
    (``bool("false") is True``), silently ENABLING paid sub-query
    generation; and a negative timeout collapses the deadline to ~0. So
    every value resolved here is coerced before being handed back:
    ints/timeouts through the same reject-bool / reject-nonpositive /
    default-on-malformed semantics as ``config._get_int_timeout_value``
    (imported directly -- config.py has no dependency on this module, so
    there is no import cycle); the one bool through a strict true-set
    (``"true"`` / ``"1"`` / literal ``True`` -- deliberately narrower than
    ``config._get_typed_value``'s bool coercion, since this flag gates a
    paid LLM call); provider/engine strings stripped of surrounding
    whitespace.
    """
    from ..config import _get_int_timeout_value, get_cli_setting  # local: keep module import cheap

    def _str(key: str, default: str) -> str:
        # Substitute the default ONLY for a missing key or a non-string
        # value (matching config._get_typed_value's own contract) -- an
        # EXPLICIT empty/whitespace string must pass through as "" so the
        # spend-check-before-spend gate below still fires on it. Silently
        # replacing "" with "openai" (task-1356 review, N1) let a probe
        # with [SearchSettings] relevance_analysis_llm = "" make a REAL
        # provider call against a provider the user never named.
        raw = get_cli_setting("SearchSettings", key, default)
        if not isinstance(raw, str):
            return default
        return raw.strip()

    def _bool(key: str, default: bool) -> bool:
        raw = get_cli_setting("SearchSettings", key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in ("true", "1")
        return default

    def _int(key: str, default: int) -> int:
        raw = get_cli_setting("SearchSettings", key, default)
        return _get_int_timeout_value({key: raw}, key, default)

    return {
        "search_provider_default": _str("search_provider_default", "google"),
        "relevance_analysis_llm": _str("relevance_analysis_llm", "openai"),
        "final_answer_llm": _str("final_answer_llm", "openai"),
        "search_enable_subquery": _bool("search_enable_subquery", False),
        "search_default_max_queries": _int("search_default_max_queries", 5),
        "search_result_max": _int("search_result_max", 10),
        "relevance_llm_timeout_s": _int("relevance_llm_timeout_s", 30),
        "relevance_scrape_timeout_s": _int("relevance_scrape_timeout_s", 30),
        # 240 default (task-1356 review ruling). Fix round 1: this used to
        # need to undercut the agent runtime's 300s max_tool_call_seconds so
        # a deadline-hit run could still return its partial synthesis
        # instead of being killed first -- that is no longer an operator
        # obligation. deep_search_outer_timeout_s() below now DERIVES the
        # runtime's per-call ceiling from whatever this resolves to, so any
        # configured value (not just ones under 300) keeps that guarantee.
        "deep_search_timeout_s": _int("deep_search_timeout_s", 240),
    }


def deep_search_outer_timeout_s() -> float:
    """Outer per-call timeout for the ``web_deep_search`` tool (task-1356
    fix round 1).

    Single source of truth: reads the SAME coerced settings seam the tool
    itself uses (``_deep_search_settings()``), so a malformed
    ``deep_search_timeout_s`` degrades identically for both -- the tool's
    own internal deadline and this outer override both fall back to the
    240 default, never disagreeing about a bad config value.

    Derived, not pinned: returns ``deep_search_timeout_s`` plus the tool's
    own worst-case internal overrun (``_DEEP_SEARCH_DEADLINE_GRACE_S`` +
    ``_DEEP_SEARCH_THREAD_JOIN_SLACK_S``) plus a scheduling-jitter margin
    (``_DEEP_SEARCH_SCHEDULING_JITTER_S``) -- so the outer ceiling exceeds
    the tool's own graceful deadline/grace/join sequence for ANY configured
    value, not only the shipped default. At the 240 default this still
    returns 290.0, exactly the constant this function replaces
    (``Agents/local_tool_provider.py``'s former ``_WEB_DEEP_SEARCH_
    TIMEOUT_S``) -- default behavior is unchanged. Consulted by
    ``LocalToolProvider.timeout_for`` for ``web_deep_search`` only.

    Returns:
        Outer per-call ceiling in seconds: the configured (coerced)
        ``deep_search_timeout_s`` plus the tool's worst-case internal
        overrun plus jitter margin (290.0 at the shipped defaults).
    """
    settings = _deep_search_settings()
    return (
        settings["deep_search_timeout_s"]
        + _DEEP_SEARCH_DEADLINE_GRACE_S
        + _DEEP_SEARCH_THREAD_JOIN_SLACK_S
        + _DEEP_SEARCH_SCHEDULING_JITTER_S
    )


def _run_coro_loop_safe(coro, timeout_s: float):
    """Run ``coro`` to completion regardless of whether a loop is already running.

    No running loop on this thread (the common case -- a worker thread or a
    plain sync caller): ``asyncio.run(coro)`` directly. Measured reality
    (task-1356 review, N3 -- this paragraph previously overstated it):
    ``coro``'s own internal ``asyncio.wait_for`` bounds the RETURNED VALUE
    (or raised exception) only, and that part IS prompt -- but
    ``asyncio.run()``'s own shutdown sequence still waits for any orphaned
    ``asyncio.to_thread`` worker a cancelled-but-already-running call left
    behind inside ``coro`` (same limitation as the dedicated-thread path
    below: Python cannot forcibly kill a running thread), so the WALL-CLOCK
    time of this whole call can exceed whatever timeout ``coro`` enforced
    on itself by up to that blocking call's own remaining duration.
    Empirically confirmed in ``test_analyze_and_aggregate_offloads_
    aggregate_results_so_wait_for_can_fire``: a `wait_for(timeout=0.05)`
    around a 0.3s-blocking call raised its `TimeoutError` promptly, but the
    enclosing ``asyncio.run()`` call still took the full ~0.3s to return.
    Nothing here bounds that overrun on the no-loop path -- only ``coro``'s
    own internal deadline logic determines when its RESULT is ready; there
    is no second thread to join for a hard wall-clock backstop, unlike the
    path below. A loop IS already running (the tool invoked from inside
    async agent-runtime code): ``asyncio.run`` cannot nest, so ``coro``
    runs on a dedicated daemon thread with its own fresh loop, and THIS
    thread blocks in ``thread.join(timeout_s)``.

    That join is a genuinely different kind of backstop from ``coro``'s own
    internal ``wait_for``: it is enforced by the OS thread scheduler on the
    CALLING thread, not by the dedicated thread's event loop -- so it fires
    on schedule even if that loop is completely wedged (e.g. stuck inside a
    synchronous call that starves its own timeout callbacks; see
    ``_DEEP_SEARCH_DEADLINE_GRACE_S``'s note on why that can happen). What
    it does NOT do: reach into the dedicated thread and stop ``coro``. If
    ``coro`` never returns, that thread keeps running as an orphaned daemon
    after this function raises -- Python cannot forcibly kill a running
    thread, and a real analyze_and_aggregate call would already be spending
    real provider tokens on it by that point.

    Args:
        coro: A not-yet-awaited coroutine object.
        timeout_s: Hard wall-clock backstop for the ``thread.join()`` on
            the dedicated-thread path ONLY -- bounds how long THIS calling
            thread waits, not how long ``coro`` itself is allowed to run.
            Ignored entirely on the direct ``asyncio.run(coro)`` path
            (there is no second thread to join).

    Returns:
        Whatever ``coro`` returns.

    Raises:
        LocalToolError: ``[deep-search-failed] timeout: ...`` if the
            dedicated thread does not finish within ``timeout_s``.
        BaseException: whatever ``coro`` itself raised, re-raised as-is on
            the caller's thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome: dict = {}

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            outcome["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise LocalToolError(
            f"[deep-search-failed] timeout: analysis did not finish within {timeout_s:.0f}s "
            "(loop-safe thread backstop)"
        )
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def deep_search_pipeline_params(
    *,
    engine: Optional[str] = None,
    max_results: Optional[int] = None,
    subquery: Optional[bool] = None,
    max_queries: Optional[int] = None,
    respect_robots: Optional[bool] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Assemble the deep-search pipeline search_params from [SearchSettings]
    (task-16484) -- ONE assembly shared by the web_deep_search tool, the
    Console /research command, and the baseline script, with per-key
    overrides for callers that need tighter bounds (e.g. spend-capped
    baseline runs force subquery off and one query).
    """
    settings = _deep_search_settings()

    try:
        result_ceiling = int(settings.get("search_result_max", SEARCH_MAX_RESULT_COUNT))
    except (TypeError, ValueError):
        result_ceiling = SEARCH_MAX_RESULT_COUNT
    if result_ceiling < 1:
        result_ceiling = SEARCH_MAX_RESULT_COUNT
    resolved_max_results = max_results if max_results is not None else result_ceiling
    try:
        resolved_max_results = max(1, min(int(resolved_max_results), result_ceiling))
    except (TypeError, ValueError):
        resolved_max_results = result_ceiling

    deadline_s = float(settings.get("deep_search_timeout_s", 240) or 240)

    params: dict = {
        "engine": engine or settings.get("search_provider_default", SEARCH_DEFAULT_ENGINE),
        "content_country": "US",
        "search_lang": "en",
        "output_lang": "en",
        "result_count": resolved_max_results,
        "subquery_generation": bool(
            settings.get("search_enable_subquery", False)
            if subquery is None
            else subquery
        ),
        "subquery_generation_llm": settings.get("relevance_analysis_llm"),
        "relevance_analysis_llm": settings.get("relevance_analysis_llm"),
        "final_answer_llm": settings.get("final_answer_llm"),
        # CRITICAL: analyze_and_aggregate reads these two straight out of
        # search_params -- omitting them means the config knobs silently
        # never reach the pipeline and it falls back to its own 30s defaults.
        "relevance_llm_timeout_s": settings.get("relevance_llm_timeout_s", 30),
        "relevance_scrape_timeout_s": settings.get("relevance_scrape_timeout_s", 30),
        # generate_and_search reads this to cap total fan-out.
        "search_default_max_queries": (
            settings.get("search_default_max_queries", 5)
            if max_queries is None
            else max_queries
        ),
        # The caller's remaining deadline (full configured timeout when the
        # run has not started yet).
        "phase1_time_budget_s": deadline_s,
        "respect_robots_txt": (
            _webfetch_settings()["respect_robots_txt"]
            if respect_robots is None
            else respect_robots
        ),
        "deep_search_timeout_s": deadline_s,
    }
    if extra:
        params.update(extra)
    return params


def web_deep_search(question: str, engine: Optional[str] = None, max_results: Optional[int] = None) -> str:
    """Multi-query web research: sub-questions, relevance filtering, a cited answer.

    Two-phase pipeline (``Web_Scraping.WebSearch_APIs``): phase 1
    (``generate_and_search``, sync) optionally expands ``question`` into
    sub-queries and fans out searches; phase 2 (``analyze_and_aggregate``,
    async) scores results for relevance, scrapes/summarizes the relevant
    ones, and synthesizes a cited answer. Both LLM endpoints (relevance
    analysis, final synthesis) come from ``[SearchSettings]`` config
    (``_deep_search_settings``) and are validated BEFORE phase 1 runs --
    missing config must never cost a search-provider call.

    Phase 2 runs under a wall-clock deadline (``deep_search_timeout_s``
    minus phase 1's elapsed time): a ``cancel_event`` is set when the
    deadline is reached, which the pipeline checks cooperatively between
    per-result relevance calls (task-1356's no-circuit-breaker design means
    a dead provider otherwise costs N sequential LLM timeouts) so a
    deadline hit still yields a partial synthesis rather than nothing.
    ``asyncio.wait_for`` at deadline + grace is a second backstop, but it
    only bounds work that keeps yielding control back to the loop -- see
    ``_DEEP_SEARCH_DEADLINE_GRACE_S``'s note on why a synchronous pipeline
    call would otherwise make it a no-op, and ``_run_coro_loop_safe``'s
    docstring for the third, genuinely preemptive thread-join backstop that
    applies when this tool is invoked from inside an already-running loop.
    If the deadline fires before ANY result was confirmed relevant, the
    zero-relevant return below says so explicitly and does not claim
    coverage the run never had.

    Args:
        question: The research question. Must be non-empty.
        engine: Search engine name, validated against ``SEARCH_ENGINES``;
            defaults to ``search_provider_default`` from config.
        max_results: Results per query, clamped to ``[1, search_result_max]``
            (config); defaults to ``search_result_max``.

    Returns:
        str: On success, the synthesized answer (capped at
        ``DEEP_SEARCH_ANSWER_MAX_BYTES``) followed by a ``Sources:`` list
        (evidence id/title/url, at most ``DEEP_SEARCH_SOURCES_MAX`` entries)
        and a one-line status footer -- with the COMBINED output (footer +
        answer + sources) held under ``DEEP_SEARCH_TOTAL_MAX_BYTES``, since
        that is what actually has to survive the agent runtime's
        head-first tool-result truncation (see the constant's own
        docstring); an omission marker in the Sources block distinguishes a
        size-budget cutoff from the ordinary ``DEEP_SEARCH_SOURCES_MAX``
        count cap. When phase 2 finds no relevant results, returns an
        explanatory (non-error) string -- listing the queries tried, or, on
        a deadline hit, disclosing the cutoff instead -- rather than raising;
        the model is expected to read and act on it.

    Raises:
        LocalToolError: ``[invalid-args]`` for an empty question or a
            caller-supplied engine not in ``SEARCH_ENGINES``;
            ``[deep-search-failed] relevance: ...`` / ``synthesis: ...`` if
            the corresponding LLM is not configured (checked before any
            spend); ``[deep-search-failed] search: ...`` for a phase-1
            failure, zero search results, a misconfigured
            ``search_provider_default``, or a malformed pipeline result;
            ``[deep-search-failed] relevance: ...`` for an unexpected
            phase-2 failure; ``[deep-search-failed] timeout: ...`` if the
            deadline backstop is hit.
    """
    if not isinstance(question, str) or not question.strip():
        raise LocalToolError("[invalid-args] question must be a non-empty string")
    question = question.strip()

    settings = _deep_search_settings()

    # Track provenance: an invalid engine the CALLER passed is the caller's
    # mistake ([invalid-args], the model should retry with a different
    # value); an invalid engine that came from [SearchSettings]
    # search_provider_default is a config problem the caller had no part
    # in -- blaming the caller's (absent) argument would misdirect a model
    # into "fixing" an argument it never supplied (task-1356 review minor).
    engine_from_caller = engine is not None
    if engine is None:
        engine = settings.get("search_provider_default", SEARCH_DEFAULT_ENGINE)
    if not isinstance(engine, str) or engine.strip().lower() not in SEARCH_ENGINES:
        if engine_from_caller:
            raise LocalToolError(f"[invalid-args] engine must be one of {SEARCH_ENGINES}: {engine!r}")
        raise LocalToolError(
            f"[deep-search-failed] search: configured [SearchSettings] "
            f"search_provider_default {engine!r} is not a supported engine "
            f"(one of {SEARCH_ENGINES})"
        )
    engine = engine.strip().lower()

    try:
        result_ceiling = int(settings.get("search_result_max", SEARCH_MAX_RESULT_COUNT))
    except (TypeError, ValueError):
        result_ceiling = SEARCH_MAX_RESULT_COUNT
    if result_ceiling < 1:
        result_ceiling = SEARCH_MAX_RESULT_COUNT
    if max_results is None:
        max_results = result_ceiling
    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = result_ceiling
    max_results = max(1, min(max_results, result_ceiling))

    # Spend-check-before-spend: both LLM endpoints must be configured before
    # phase 1 (a real search-provider call) ever runs.
    relevance_llm = settings.get("relevance_analysis_llm")
    if not relevance_llm or not str(relevance_llm).strip():
        raise LocalToolError("[deep-search-failed] relevance: no relevance_analysis_llm configured")
    final_answer_llm = settings.get("final_answer_llm")
    if not final_answer_llm or not str(final_answer_llm).strip():
        raise LocalToolError("[deep-search-failed] synthesis: no final_answer_llm configured")

    deadline_s = float(settings.get("deep_search_timeout_s", 240) or 240)

    # task-16484: ONE shared assembly (this tool, the Console /research
    # command, and the baseline script all build these params).
    search_params = deep_search_pipeline_params(
        engine=engine, max_results=max_results
    )

    from ..Web_Scraping import WebSearch_APIs  # local import: keep module import cheap

    start = time.monotonic()
    try:
        phase1 = WebSearch_APIs.generate_and_search(question, search_params)
    except Exception as exc:  # noqa: BLE001 - structured per behavior 6
        raise LocalToolError(f"[deep-search-failed] search: {exc}") from exc

    if (
        not isinstance(phase1, dict)
        or not isinstance(phase1.get("web_search_results_dict"), dict)
        or not isinstance(phase1.get("sub_query_dict"), dict)
    ):
        raise LocalToolError("[deep-search-failed] search: malformed pipeline result")

    wsr = phase1["web_search_results_dict"]
    sub_query_dict = phase1["sub_query_dict"]
    results = wsr.get("results") or []
    warnings = list(wsr.get("warnings") or [])

    if not results:
        warn_suffix = f" (warnings: {'; '.join(warnings)})" if warnings else ""
        raise LocalToolError(f"[deep-search-failed] search: no results{warn_suffix}")

    remaining = max(0.0, deadline_s - (time.monotonic() - start))
    hard_timeout = remaining + _DEEP_SEARCH_DEADLINE_GRACE_S

    async def _phase2():
        cancel_event = asyncio.Event()

        async def _watchdog() -> None:
            try:
                await asyncio.sleep(remaining)
            except asyncio.CancelledError:
                return
            cancel_event.set()

        watchdog_task = asyncio.ensure_future(_watchdog())
        try:
            pipeline_result = await asyncio.wait_for(
                WebSearch_APIs.analyze_and_aggregate(wsr, sub_query_dict, search_params, cancel_event=cancel_event),
                timeout=hard_timeout,
            )
        finally:
            watchdog_task.cancel()
        return pipeline_result, cancel_event.is_set()

    try:
        phase2_result, deadline_hit = _run_coro_loop_safe(
            _phase2(), timeout_s=hard_timeout + _DEEP_SEARCH_THREAD_JOIN_SLACK_S
        )
    except LocalToolError:
        raise  # already structured (e.g. the loop-safe runner's own timeout)
    except asyncio.TimeoutError as exc:
        raise LocalToolError(
            f"[deep-search-failed] timeout: exceeded {deadline_s:.0f}s deep-search deadline"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - structured per behavior 6
        # analyze_and_aggregate's own two steps (relevance scoring,
        # synthesis) each swallow their internal failures and degrade
        # gracefully rather than raising (see WebSearch_APIs.py); an
        # exception escaping the combined call is therefore attributed to
        # the earlier, I/O-heavier relevance step.
        raise LocalToolError(f"[deep-search-failed] relevance: {exc}") from exc

    final_answer = phase2_result.get("final_answer") or {}
    relevant_results = phase2_result.get("relevant_results") or {}
    warnings = list((phase2_result.get("web_search_results_dict") or {}).get("warnings") or warnings)

    sub_questions = list(sub_query_dict.get("sub_questions") or [])
    n_queries = 1 + len(sub_questions)
    query_plural = "y" if n_queries == 1 else "ies"

    if not relevant_results:
        if deadline_hit:
            # CRITICAL fix (task-1356 review): a watchdog firing before the
            # relevance loop confirms even one result must not claim
            # "Analyzed N results" (it analyzed none) and must not advise
            # rephrasing outright (the cause might be a timeout, not a bad
            # query). Round-2 fix (task-1356 review, N2): the FIRST version
            # of this message claimed "none were scored in time" -- also
            # false. The pipeline exposes NO "how many were actually
            # scored before cancellation" signal (search_result_relevance
            # returns only the relevant subset, not an attempted count), so
            # a probe with 19 of 40 genuinely scored (just none relevant)
            # got the exact same byte-identical message as a cancel at the
            # very top of the loop. This version claims only what's
            # knowable -- N raw results found, an unknown number scored --
            # and gives advice for both worlds.
            return (
                f"Deep search for {question!r} was cut off by the {deadline_s:.0f}s "
                "deep-search deadline before any result was confirmed relevant. "
                f"Found {len(results)} raw result(s) across {n_queries} quer{query_plural}; "
                "an unknown number were scored before the cutoff. A longer "
                "deep_search_timeout_s allows more results to be scored; if many were "
                "scored but none proved relevant, rephrasing may help."
            )
        queries_tried = "; ".join([question, *sub_questions]) if sub_questions else question
        return (
            f"No relevant results found for {question!r}. Analyzed {len(results)} "
            f"result(s) across {n_queries} quer{query_plural} "
            f"tried: {queries_tried}. Try rephrasing the question or broadening it."
        )

    # Footer built first (task-1356 review): it always survives regardless
    # of how large the sources block below turns out to be.
    try:
        confidence = float(final_answer.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0

    chunks = final_answer.get("chunks") or []
    # "fallback" (not "generated") is the failure signal (final review,
    # Important 1): "generated" only means "an LLM produced this summary" --
    # the single-chunk skip path in aggregate_results also sets it False
    # with nothing having failed, so reading it here falsely called the
    # healthiest possible run (single chunk, synthesis succeeded) a
    # "fallback summary". "fallback": True is set ONLY when a per-chunk
    # summarization call actually failed and truncated raw text was
    # substituted (WebSearch_APIs.py's per-chunk except branch).
    fallback_count = sum(1 for c in chunks if isinstance(c, dict) and c.get("fallback"))
    fallback_note = f" · {fallback_count} chunk(s) used a fallback summary" if fallback_count else ""
    warning_note = f" · {len(warnings)} search warning(s)" if warnings else ""
    # "may be incomplete", not a definite "partial synthesis" claim
    # (task-1356 review, N2): the b2 probe showed a run whose watchdog
    # fired mid-call but which still went on to complete fully and
    # successfully -- deadline_hit only means the deadline was reached,
    # not that anything was actually cut short.
    deadline_note = " · deadline reached — results may be incomplete" if deadline_hit else ""
    # task-16333: a gate-fallback report must never masquerade as a
    # relevance-verified one.
    gate_block = final_answer.get("gate") or {}
    gate_note = (
        " · evidence not relevance-verified (gate fallback)" if gate_block.get("fallback") else ""
    )
    # "scored" is only accurate when the relevance loop ran to completion --
    # a deadline hit means some of `results` were never examined at all, so
    # say "found" instead of implying full coverage the run never had
    # (task-1356 review minor; the wording was "Analyzed K of N", but K is
    # the RELEVANT count, not an analyzed count either).
    coverage_verb = "found" if deadline_hit else "scored"

    # Citation verification (task-16331): when the pipeline ran its
    # citation/quote check (LLM-success branch only), surface the counts in
    # the footer so the model can weigh the answer's grounding; absent on
    # fallback/failure branches, which have no verdict to report.
    citation_note = ""
    citation_summary = deep_search_citations_footer(final_answer.get("citation_verification"))
    if citation_summary:
        citation_note = f" · {citation_summary}"

    footer = (
        f"Confidence: {confidence:.2f} · Engine: {engine} · Sub-queries: {len(sub_questions)} · "
        f"Relevant: {len(relevant_results)} of {len(results)} {coverage_verb}"
        f"{fallback_note}{warning_note}{deadline_note}{citation_note}{gate_note}"
    )

    text = _truncate_to_bytes(str(final_answer.get("text") or ""), DEEP_SEARCH_ANSWER_MAX_BYTES)

    # Sources gets whatever's left of the REAL total after the footer
    # (built above, always emitted in full) and the answer (already capped)
    # are reserved -- this is what makes DEEP_SEARCH_TOTAL_MAX_BYTES an
    # actual total (task-1356 review round 2, N4): previously the Sources
    # block had its own INDEPENDENT budget on top of the answer cap, so a
    # maxed-out answer plus a maxed-out Sources block could together still
    # exceed what the agent runtime's head-first tool-result truncation
    # (RunBudget.max_tool_result_chars) actually preserves.
    footer_bytes = len(footer.encode("utf-8"))
    answer_bytes = len(text.encode("utf-8"))
    section_separator_bytes = len("\n\n".encode("utf-8")) * 2  # joins the 3 sections below
    sources_budget = max(
        0, DEEP_SEARCH_TOTAL_MAX_BYTES - footer_bytes - answer_bytes - section_separator_bytes
    )

    # Count-capped (DEEP_SEARCH_SOURCES_MAX) THEN byte-budget-capped
    # (sources_budget, derived above). Both title AND url are truncated via
    # the house byte-truncation helper -- a pathologically long field in
    # EITHER previously reproduced a runaway result (title first, then a
    # long-URL variant once title truncation alone defeated the original
    # reproduction). Two DISTINCT, honest omission markers: "size cap" for
    # count-capped candidates that still didn't fit the byte budget
    # (covers the single-oversized-line case too -- when even the FIRST
    # candidate doesn't fit, every one of them is reported omitted rather
    # than the block silently rendering "Sources: (none)" as if no
    # evidence existed at all) and "count cap" for evidence beyond
    # DEEP_SEARCH_SOURCES_MAX that was never even considered.
    evidence = final_answer.get("evidence") or []
    candidates = [item for item in evidence[:DEEP_SEARCH_SOURCES_MAX] if isinstance(item, dict)]
    count_omitted = max(0, len(evidence) - DEEP_SEARCH_SOURCES_MAX)
    source_lines: list = []
    sources_bytes = 0
    emitted = 0
    for i, item in enumerate(candidates, 1):
        sid = item.get("id", i)
        title = _truncate_to_bytes(str(item.get("title") or item.get("url") or "Untitled"), 200)
        url = _truncate_to_bytes(str(item.get("url") or ""), 500)
        line = f"[{sid}] {title} — {url}"
        line_bytes = len(line.encode("utf-8"))
        if sources_bytes + line_bytes > sources_budget:
            break
        source_lines.append(line)
        sources_bytes += line_bytes
        emitted += 1
    size_omitted = len(candidates) - emitted
    if size_omitted > 0:
        source_lines.append(f"… [{size_omitted} further source(s) omitted: size cap reached]")
    if count_omitted > 0:
        source_lines.append(
            f"… [{count_omitted} further source(s) omitted: count cap reached "
            f"({DEEP_SEARCH_SOURCES_MAX} max)]"
        )
    sources_block = "Sources:\n" + "\n".join(source_lines) if source_lines else "Sources: (none)"

    return f"{text}\n\n{sources_block}\n\n{footer}"
