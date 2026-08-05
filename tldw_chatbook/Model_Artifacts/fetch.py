"""Streaming guarded fetch for managed artifact acquisition (TASK-595).

The egress hop loop (SSRF policy, hop cap, credential stripping) re-shaped
to stream to disk under a hard byte bound. The fetch-state sidecar is owned
by acquisition, NOT this module.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import httpx
from loguru import logger

from tldw_chatbook.Utils.egress import (
    MAX_REDIRECT_HOPS,
    check_url_or_raise_async,
)

# Mirrors egress._STRIP_HEADERS (private there); the drift-guard test in
# test_stream_fetch.py fails if the two sets ever diverge.
_STRIP_HEADERS = ("authorization", "cookie", "proxy-authorization", "x-goog-api-key")

_CHUNK_BYTES = 1024 * 1024


class FetchError(Exception):
    """Base for streaming-fetch failures. Messages never carry headers."""


class FetchRestartRequired(FetchError):
    """Resume impossible (validators changed/weak, or no Range support)."""


class FetchTooLargeError(FetchError):
    """The transfer would exceed the declared byte bound."""


class FetchTransportError(FetchError):
    """Network-level failure, wrapping httpx errors without header data."""


@dataclass(frozen=True)
class FetchValidators:
    """HTTP validators captured from a response, used for Range resume."""

    etag: str | None
    last_modified: str | None

    @property
    def strong(self) -> bool:
        """True when resuming on these validators is safe (strong ETag or
        a Last-Modified date; weak `W/` ETags never qualify)."""
        if self.etag and not self.etag.startswith("W/"):
            return True
        return self.etag is None and self.last_modified is not None


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one stream_fetch call."""

    bytes_written: int
    validators: FetchValidators
    resumed: bool


def _same_origin(a: httpx.URL, b: httpx.URL) -> bool:
    return (a.scheme, a.host, a.port) == (b.scheme, b.host, b.port)


# ``Content-Range: bytes <start>-<end>/<total>`` (RFC 9110 14.4) or a
# ``*``-total variant (``bytes <start>-<end>/*``); ``total`` is optional to
# capture for callers that don't need it.
_CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


def _parse_content_range_start(header_value: str | None) -> int | None:
    """Extract the START byte offset from a ``Content-Range`` header.

    A 206 response alone does not prove the body actually starts at the
    ``Range`` request's offset -- only ``Content-Range`` does, and only when
    it parses. Anything else (missing header, unparseable value) must be
    treated by the caller as "cannot be trusted", never as "assume it's
    fine".

    Args:
        header_value: The raw ``Content-Range`` header value, or ``None``
            when the header was absent.

    Returns:
        The parsed start offset, or ``None`` when the header is missing or
        does not match the expected ``bytes <start>-<end>/<total>`` shape.
    """
    if not header_value:
        return None
    match = _CONTENT_RANGE_RE.match(header_value.strip())
    if not match:
        return None
    return int(match.group(1))


async def stream_fetch(
    url: str,
    destination: Path,
    *,
    client: httpx.AsyncClient,
    max_bytes: int,
    resume_from: int = 0,
    validators: FetchValidators | None = None,
    headers: Mapping[str, str] | None = None,
    trusted_origins: frozenset[str] = frozenset(),
    on_chunk: Callable[[int], None] | None = None,
) -> FetchResult:
    """Stream a URL to disk with egress guards, byte bounds, and resume.

    Args:
        url: Source URL (http/https; egress policy enforced per hop).
        destination: File appended to (created if absent).
        client: Caller-owned AsyncClient (connection reuse, test injection).
        max_bytes: Hard bound on the FINAL file size (resume_from + written).
        resume_from: Durable bytes already on disk; sends a Range request.
        validators: Validators the existing bytes were fetched under; resume
            requires them strong and matching the server's current ones.
        headers: Extra headers (credentials); stripped on cross-origin hops.
        trusted_origins: Egress-policy trust additions (fixture servers).
        on_chunk: Called with each chunk's byte count (progress).

    Returns:
        FetchResult with bytes written THIS call, captured validators, and
        whether a Range continuation was used.

    Raises:
        FetchRestartRequired: Resume requested but unsafe/unsupported.
        FetchTooLargeError: Bound exceeded (pre-declared or mid-stream).
        FetchTransportError: Connection/protocol failures.
        EgressBlockedError: Policy rejection (from egress helpers).
    """
    if resume_from and not (validators and validators.strong):
        raise FetchRestartRequired("resume requires strong validators")
    if resume_from >= max_bytes:
        raise FetchTooLargeError("resume offset already at or past the bound")

    current = httpx.URL(url)
    origin = current
    request_headers: dict[str, str] = dict(headers or {})
    if resume_from:
        request_headers["Range"] = f"bytes={resume_from}-"
        # If-Range must carry WHICHEVER validator makes resuming safe: a
        # strong ETag if we have one, else the Last-Modified date (RFC 9110
        # 13.1.5 -- If-Range also accepts an HTTP-date). Sending only the
        # etag branch would silently drop the Last-Modified-only case that
        # FetchValidators.strong explicitly allows (etag=None, last_modified
        # set): the server would then see a bare Range with no If-Range,
        # honor it unconditionally, and a changed resource's bytes would be
        # appended to stale on-disk data with no mismatch ever detected.
        if validators and validators.etag:
            request_headers["If-Range"] = validators.etag
        elif validators and validators.last_modified:
            request_headers["If-Range"] = validators.last_modified

    written = 0
    for _hop in range(MAX_REDIRECT_HOPS + 1):
        await check_url_or_raise_async(str(current), trusted_origins=trusted_origins)
        is_same_origin = _same_origin(origin, current)
        send_headers = dict(request_headers)
        if not is_same_origin:
            for name in list(send_headers):
                if name.lower() in _STRIP_HEADERS:
                    del send_headers[name]
        # Built explicitly (not passed straight to client.stream/client.get)
        # so cross-origin credential stripping can also reach headers the
        # CALLER attached at the client-construction level (e.g.
        # httpx.AsyncClient(headers={"Authorization": ...})) -- httpx merges
        # those client-default headers onto the request during send, AFTER
        # send_headers above was already stripped, so send_headers alone
        # never sees them. Mirrors egress.py's guarded_fetch_httpx_async.
        request = client.build_request("GET", current, headers=send_headers)
        if not is_same_origin:
            for _h in _STRIP_HEADERS:
                request.headers.pop(_h, None)
        try:
            response = await client.send(request, stream=True, follow_redirects=False)
        except httpx.HTTPError as exc:
            raise FetchTransportError(type(exc).__name__) from exc
        try:
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get("location")
                if not location:
                    raise FetchTransportError("redirect without location")
                current = current.join(location)
                if origin.scheme == "https" and current.scheme != "https":
                    raise FetchTransportError("HTTPS redirect downgrade")
                continue
            if resume_from and response.status_code != 206:
                # Server ignored Range (200 full body / no support).
                raise FetchRestartRequired("server did not honor Range")
            if response.status_code == 401 or response.status_code == 403:
                raise FetchTransportError(f"HTTP {response.status_code}")
            if response.status_code >= 400:
                raise FetchTransportError(f"HTTP {response.status_code}")
            got = FetchValidators(
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
            )
            if resume_from and validators and validators.etag:
                # A missing ETag on the resumed response is NOT "no
                # information" -- it is treated as a mismatch. Accepting it
                # as a match would let a 206 that simply omits the header
                # (a real, under-implemented server behavior, not just a
                # theoretical one) get silently appended to stale on-disk
                # bytes fetched under a DIFFERENT resource version.
                if not got.etag or got.etag != validators.etag:
                    raise FetchRestartRequired("validators changed upstream")
            if (
                resume_from
                and validators
                and not validators.etag
                and validators.last_modified
                and got.last_modified
                and got.last_modified != validators.last_modified
            ):
                # Symmetric to the ETag check above, and NOT redundant
                # with the earlier status-code check: that check only
                # catches a server that answers a stale If-Range with a
                # full 200. A server that IGNORES If-Range entirely (a
                # real and, for date-based conditionals specifically,
                # under-implemented failure mode -- If-Range is far more
                # commonly wired up for ETags than for Last-Modified)
                # still answers 206, and would otherwise have its
                # mismatched bytes appended with no detection at all.
                raise FetchRestartRequired("validators changed upstream")
            if resume_from and response.status_code == 206:
                # A 206 alone does not prove the body starts at
                # resume_from -- only Content-Range does, and only when it
                # parses AND its start matches. A missing/unparseable
                # header or a start that disagrees with the Range we asked
                # for must be rejected BEFORE the destination is ever
                # opened for append, or a server's mismatched bytes would
                # be silently appended to stale on-disk data.
                range_start = _parse_content_range_start(
                    response.headers.get("content-range")
                )
                if range_start != resume_from:
                    raise FetchRestartRequired(
                        "Content-Range start does not match the requested resume offset"
                    )
            mode = "ab" if resume_from else "wb"
            with open(destination, mode) as fh:
                async for chunk in response.aiter_bytes(_CHUNK_BYTES):
                    if resume_from + written + len(chunk) > max_bytes:
                        raise FetchTooLargeError("byte bound exceeded")
                    fh.write(chunk)
                    written += len(chunk)
                    if on_chunk:
                        on_chunk(len(chunk))
                fh.flush()
                os.fsync(fh.fileno())
            return FetchResult(written, got, resumed=bool(resume_from))
        except httpx.HTTPError as exc:
            raise FetchTransportError(type(exc).__name__) from exc
        finally:
            await response.aclose()
    raise FetchTransportError("redirect hop limit exceeded")
