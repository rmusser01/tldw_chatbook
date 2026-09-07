"""Egress-gated stream-URL resolution for ``/stream-video`` (task-3401.11,
ADR-044 decision 5).

The pipeline the typed URL walks before ffmpeg ever sees a URL:

1. **Shape**: http/https only.
2. **Redirect walk (hop 1)**: the user-typed URL carries the trust seed --
   its own hostname is trusted because the USER typed it
   (``origin_set(typed)``; metadata endpoints stay blocked regardless).
3. **Redirect walk (hops 2+)**: every vendor-supplied ``Location`` is
   validated WITHOUT trust -- a redirect cannot bounce the player into a
   private network the user never asked for.
4. **Direct-media detection**: a video/audio/octet-stream content type
   means the final URL is the stream itself.
5. **yt-dlp fallback (HTML pages)**: yt-dlp runs ONLY as a subprocess
   (never imported as a library) with a progressive, single-URL format
   selector. Its output URL is vendor-controlled and gets the same
   untrusted egress validation (AC2). ``.m3u8``/DASH output is refused:
   separate audio/video muxing is the documented follow-up, not v1.
6. **Seekability**: an ``Accept-Ranges: bytes`` probe decides whether the
   player offers seek (AC4).

Nothing is ever written to disk in any path.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 # yt-dlp invoked as a fixed-argv subprocess
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from tldw_chatbook.Utils import egress
from tldw_chatbook.Utils.egress import EgressBlockedError

MAX_REDIRECT_HOPS = 10
RESOLVE_TIMEOUT_SECONDS = 15.0
YTDLP_TIMEOUT_SECONDS = 30.0

#: Time-box for one stream session (AC5/ADR residual risk: no byte cap on
#: streams; sessions are time-boxed and user-terminated instead).
MAX_STREAM_SECONDS = 2 * 60 * 60

#: Bias yt-dlp toward a single progressive https URL (no separate a/v).
YTDLP_PROGRESSIVE_FORMAT = (
    "best[ext=mp4][protocol=https]/best[protocol=https]/best"
)

HLS_FOLLOW_UP_NOTE = (
    "HLS/DASH streams with separate audio/video tracks are not supported "
    "yet -- dual-input muxing is a documented follow-up (ADR-044, "
    "task-3401.11 v1 scope: progressive single-URL streams only)."
)

_MEDIA_CONTENT_PREFIXES = ("video/", "audio/")


class StreamResolutionError(RuntimeError):
    """Raised when a URL cannot be resolved to a playable stream."""


@dataclass(frozen=True)
class StreamResolution:
    """The validated, playable result of resolving one typed URL."""

    final_url: str
    content_type: str
    seekable: bool
    via_ytdlp: bool
    content_length: int | None = None


def _check_media_content_type(content_type: str) -> bool:
    lowered = (content_type or "").split(";")[0].strip().lower()
    return lowered.startswith(_MEDIA_CONTENT_PREFIXES) or lowered in {
        "application/octet-stream",
        "application/vnd.apple.mpegurl",
        "application/x-mpegurl",
    }


def _is_hls_url(url: str, content_type: str) -> bool:
    lowered_type = (content_type or "").lower()
    return (
        urlparse(url).path.lower().endswith(".m3u8")
        or "mpegurl" in lowered_type
    )


def _validate_first_hop(url: str) -> None:
    """Egress-validate the typed URL with the user's own trust seed."""
    egress.check_url_or_raise(url, trusted_origins=egress.origin_set(url))


def _validate_later_hop(url: str) -> None:
    """Egress-validate a vendor-supplied hop WITHOUT any trust."""
    egress.check_url_or_raise(url, trusted_origins=frozenset())


def _walk_redirects(raw_url: str) -> tuple[str, httpx.Response]:
    """Walk the typed URL's redirect chain to the final response.

    Every hop is egress-validated (first hop with the typed trust, later
    hops without) BEFORE it is requested -- ffmpeg would otherwise follow
    these redirects internally, outside the policy (AC2). HEAD is used so
    nothing downloads; servers that refuse HEAD get one streaming GET with
    a closed body instead.

    Returns:
        ``(final_url, final_response)``.

    Raises:
        StreamResolutionError: On shape errors, blocked hops, a redirect
            without a Location, or too many hops.
    """
    parsed = urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise StreamResolutionError(
            f"not an http(s) URL: {raw_url!r} -- usage: /stream-video <url>"
        )
    _validate_first_hop(raw_url)
    current = raw_url
    with httpx.Client(follow_redirects=False, timeout=RESOLVE_TIMEOUT_SECONDS) as client:
        for hop in range(MAX_REDIRECT_HOPS + 1):
            try:
                response = client.head(current)
                if response.status_code in {405, 403, 400, 404} and hop == 0:
                    # Some file hosts reject HEAD; fall back to a bounded GET.
                    with client.stream("GET", current) as get_response:
                        response = get_response
            except httpx.HTTPError as exc:
                raise StreamResolutionError(f"could not reach {current!r}: {exc}") from exc
            if response.is_redirect:
                location = response.headers.get("location") or response.headers.get("Location")
                if not location:
                    raise StreamResolutionError("redirect without a Location header")
                current = urljoin(str(response.url), location)
                _validate_later_hop(current)
                continue
            return str(response.url), response
    raise StreamResolutionError("too many redirects")


def _probe_ranges(url: str) -> tuple[bool, int | None]:
    """Probe Accept-Ranges/Content-Length on the final URL (untrusted hop)."""
    _validate_later_hop(url)
    try:
        with httpx.Client(follow_redirects=False, timeout=RESOLVE_TIMEOUT_SECONDS) as client:
            response = client.head(url)
    except httpx.HTTPError:
        return False, None
    accept_ranges = (response.headers.get("accept-ranges") or "").lower()
    length_raw = response.headers.get("content-length")
    length: int | None = None
    if length_raw is not None:
        try:
            length = int(length_raw)
        except ValueError:
            length = None
    return accept_ranges == "bytes", length


def _resolve_with_ytdlp(url: str) -> str:
    """Resolve a page URL to a direct progressive stream via yt-dlp SUBPROCESS.

    yt-dlp is never imported as a library (dependency weight + licensing,
    ADR-044); it is a user-installed runtime binary like ffmpeg.
    """
    if shutil.which("yt-dlp") is None:
        raise StreamResolutionError(
            "This page is not a direct media URL, and yt-dlp is not "
            "installed to resolve it (e.g. 'pipx install yt-dlp' or 'brew "
            "install yt-dlp'). Direct .mp4/.webm links always work without it."
        )
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "-f", YTDLP_PROGRESSIVE_FORMAT,
        "-g",
        url,
    ]
    try:
        result = subprocess.run(  # nosec B603 # fixed argv, probed binary
            cmd, capture_output=True, text=True, timeout=YTDLP_TIMEOUT_SECONDS
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise StreamResolutionError(f"yt-dlp resolution failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        raise StreamResolutionError(
            "yt-dlp could not resolve this page"
            + (f": {detail[-1][:200]}" if detail else "")
        )
    resolved = (result.stdout or "").strip().splitlines()
    if not resolved or not resolved[0].strip():
        raise StreamResolutionError("yt-dlp returned no stream URL")
    final = resolved[0].strip()
    # Vendor-controlled output: full untrusted egress validation (AC2).
    try:
        _validate_later_hop(final)
    except EgressBlockedError as exc:
        raise StreamResolutionError(
            f"yt-dlp's stream URL failed the egress policy: {exc}"
        ) from exc
    return final


def resolve_stream_url(raw_url: str) -> StreamResolution:
    """Resolve one typed URL to a validated, playable stream.

    Args:
        raw_url: The user-typed URL from ``/stream-video``.

    Returns:
        A :class:`StreamResolution` naming the final stream URL, its content
        type, seekability, and whether yt-dlp produced it.

    Raises:
        StreamResolutionError: On shape errors, egress refusals, HLS/DASH
            (v1 scope), or unresolvable pages.
    """
    final_url, response = _walk_redirects(raw_url)
    content_type = response.headers.get("content-type", "")

    via_ytdlp = False
    if not _check_media_content_type(content_type):
        # Not a direct media response: try yt-dlp against the ORIGINAL typed
        # URL (page URLs are what yt-dlp understands, not our walked hops).
        final_url = _resolve_with_ytdlp(raw_url)
        via_ytdlp = True
        content_type = ""

    if _is_hls_url(final_url, content_type):
        raise StreamResolutionError(HLS_FOLLOW_UP_NOTE)

    if via_ytdlp:
        seekable, content_length = _probe_ranges(final_url)
    else:
        accept_ranges = (response.headers.get("accept-ranges") or "").lower()
        length_raw = response.headers.get("content-length")
        content_length = None
        if length_raw is not None:
            try:
                content_length = int(length_raw)
            except ValueError:
                content_length = None
        seekable = accept_ranges == "bytes"

    return StreamResolution(
        final_url=final_url,
        content_type=content_type or "application/octet-stream",
        seekable=seekable,
        via_ytdlp=via_ytdlp,
        content_length=content_length,
    )
