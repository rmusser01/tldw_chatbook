"""Pre-flight analyzer for library ingestion sources.

Provides a lightweight, dependency-cheap way to inspect a local path or URL
before starting an ingest job. It reports discovered file type groups,
estimated size, tooling warnings, and any errors that would prevent ingestion.
"""

from __future__ import annotations

import socket
import ssl
from pathlib import Path
from typing import Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from loguru import logger

from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.Library.ingest_capabilities import (
    UNSUPPORTED_GROUP,
    get_tooling_warnings,
    get_type_group,
)
from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Local_Ingestion.local_file_ingestion import is_http_url
from tldw_chatbook.Utils.egress import EgressBlockedError, check_url_or_raise
from tldw_chatbook.Utils.input_validation import validate_url
from tldw_chatbook.Utils.path_validation import validate_path_simple


def _safe_size(path: Path) -> int:
    """Return the size of ``path`` in bytes, or ``0`` on ``OSError``."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _statted_size(path: Path) -> int | None:
    """Size in bytes from a SUCCESSFUL stat, else ``None``.

    (task-2160 Qodo round) The empty-file classifier must not treat an
    unstatable file as "0 B" -- ``_safe_size``'s error fallback of ``0``
    would mislabel it; an unreadable file stays in its type group so the
    pipeline surfaces the real error at ingest time.
    """
    try:
        return path.stat().st_size
    except OSError:
        return None


def collect_directory_files(directory: Path, scan_limit: int) -> tuple[list[Path], bool]:
    """Expand a directory into the files an ingest submission should cover.

    The public seam over :func:`_collect_files`, so that submitting a folder
    and pre-flighting a folder walk it identically -- same recursion, same
    symlink/hidden-entry skipping, same ``scan_limit``. A summary that
    promises N files and a submission that queues a different N would be
    worse than either alone.

    Args:
        directory: Directory to walk.
        scan_limit: Maximum number of files to collect.

    Returns:
        A tuple of ``(files, truncated)``; ``truncated`` is ``True`` when
        files beyond ``scan_limit`` were left uncollected.
    """
    files, truncated, _skipped = _collect_files(directory, scan_limit)
    return files, truncated


def _collect_files(p: Path, scan_limit: int) -> tuple[list[Path], bool, int]:
    """Recursively collect files under ``p`` up to ``scan_limit``.

    Symlinks and hidden entries (names starting with ``.``) are skipped to avoid
    cycles, system files, and unexpected traversal. Directories that raise
    ``PermissionError`` are skipped.

    Args:
        p: Directory to scan.
        scan_limit: Maximum number of files to collect.

    Returns:
        A tuple of ``(files, truncated, skipped)``. ``truncated`` is ``True``
        when there were additional files beyond ``scan_limit`` that could not
        be collected. ``skipped`` counts the entries the scan passed over
        without collecting anything from them -- symlinks, dot-entries, and
        unreadable folders (xhigh review of task-14823: ``total_files == 0``
        alone cannot tell an EMPTY folder from one whose every entry was
        skipped, and the ingest gate asserted "This folder is empty" about
        both). Entries left uncollected because the limit was reached are
        ``truncated``, not ``skipped``.
    """
    files: list[Path] = []
    truncated = False
    skipped = 0

    try:
        entries = list(p.iterdir())
    except OSError:
        # The directory itself could not be read. It yielded nothing, but it
        # is emphatically not EMPTY -- report it as one skipped entry so the
        # caller never diagnoses "this folder is empty" from a permission
        # problem (the parent adds this to its own tally when the
        # unreadable directory is a child).
        return files, truncated, 1

    for entry in entries:
        if entry.is_symlink() or entry.name.startswith("."):
            skipped += 1
            continue

        try:
            if entry.is_dir():
                remaining = scan_limit - len(files)
                if remaining <= 0:
                    # The limit is already reached; only mark truncated if this
                    # directory actually contains files we cannot collect.
                    sub_files, _, _ = _collect_files(entry, 1)
                    if sub_files:
                        truncated = True
                        break
                    continue
                sub_files, sub_truncated, sub_skipped = _collect_files(
                    entry, remaining
                )
                files.extend(sub_files)
                skipped += sub_skipped
                if sub_truncated:
                    truncated = True
                    break
            elif entry.is_file():
                if len(files) >= scan_limit:
                    truncated = True
                    break
                files.append(entry)
            else:
                # Neither a file nor a directory (a socket, a FIFO, a
                # vanished entry): nothing to collect, and the user should
                # not be told the folder was empty because of it.
                skipped += 1
        except PermissionError:
            skipped += 1
            continue

    return files, truncated, skipped


#: HTTP statuses that mean "this resource is not there", as distinct from "the
#: host answered but would not confirm it for us". Only the former justifies
#: refusing to start.
_ABSENT_STATUSES = frozenset({404, 410})

#: Seconds allowed for the (opt-in) URL probe.
_PROBE_TIMEOUT_SECONDS = 5

#: Config gate for the URL probe. OFF by default (TASK-19556).
#:
#: The probe used to fire from ``library_screen``'s 0.8 s typing debounce,
#: which meant a link pasted into the ingest field became an HTTP request
#: before the user had asked for anything to be imported. That is the wrong
#: default even for hosts the egress policy allows: the user has not chosen
#: to contact them yet, and repeated pauses mid-typing hit them repeatedly.
#: With the gate off the pre-flight for a URL is pure classification --
#: exactly what the local-path arm does with ``stat`` -- and a URL that
#: cannot actually be fetched is reported by the ingest job, where the
#: failure carries a real reason (``_probe_url``'s own docstring already
#: said so).
_PROBE_ENABLED_SECTION = "library"
_PROBE_ENABLED_KEY = "ingest_url_preflight_probe"

#: The single note returned for EVERY URL the egress policy declines.
#:
#: Collapsing the vocabulary is the point (TASK-19556). The old probe
#: returned three differentiable outcomes -- an ``error`` for a refused
#: connection, a ``warning`` naming the status code for an answered one, and
#: a clean type-group echo for a 2xx -- so a pasted link read out the state
#: of an internal host+port. Every declined reason (private, loopback,
#: link-local, CGNAT, multicast, cloud metadata, bad scheme, DNS failure)
#: now produces this one string, so there is nothing left to difference.
_UNVERIFIABLE_NOTE = (
    "The link could not be checked ahead of time. The import will still be "
    "attempted."
)


def url_probe_enabled() -> bool:
    """Whether the pre-flight may issue a network request for a URL.

    Returns:
        ``True`` only when ``[library] ingest_url_preflight_probe`` is
        explicitly enabled. Defaults to ``False``; see
        :data:`_PROBE_ENABLED_KEY` for why.
    """
    value = get_cli_setting(_PROBE_ENABLED_SECTION, _PROBE_ENABLED_KEY, False)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


class _NoRedirectHandler(HTTPRedirectHandler):
    """Refuse to follow redirects during the probe.

    ``urlopen``'s default opener follows them, so a public URL answering
    ``302 Location: http://10.0.0.5:8080/`` walked the probe into the
    internal network *after* the egress check had already passed on the
    original target. Returning ``None`` here makes urllib surface the 3xx as
    an ``HTTPError``, which the probe reports as "the site answered {code}"
    -- an outcome about the public host the user typed, and nothing else.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _open_probe(request: Request):
    """Issue the probe request with redirects disabled (test seam)."""
    return build_opener(_NoRedirectHandler).open(
        request, timeout=_PROBE_TIMEOUT_SECONDS
    )


class UrlProbe(NamedTuple):
    """The outcome of probing a URL before ingest.

    Attributes:
        error: A reason to refuse the source outright. ``None`` when the source
            should be attempted.
        note: A reason the probe could not confirm the URL, worth telling the
            user without blocking them. ``None`` when there is nothing to say.
    """

    error: str | None = None
    note: str | None = None


def _plain_unreachable_reason(exc: URLError) -> str:
    """Plain-language line for a URL that produced no HTTP response.

    (task-3305, MI-13) ``URLError``'s ``str`` is an exception repr
    (``<urlopen error [Errno 8] nodename nor servname provided, or not
    known>``) -- users were shown raw Python. The transport failure's KIND
    is what they can act on; the raw detail goes to the debug log at the
    call site.

    Args:
        exc: The caught ``URLError``.

    Returns:
        A complete user-facing sentence, never containing a repr.
    """
    reason = getattr(exc, "reason", None)
    if isinstance(reason, socket.gaierror):
        # DNS: [Errno 8] nodename nor servname provided / [Errno -2]
        # Name or service not known.
        return "URL unreachable — the server name could not be found."
    if isinstance(reason, ConnectionRefusedError):
        return "URL unreachable — the connection was refused."
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "URL unreachable — the connection timed out."
    if isinstance(reason, ssl.SSLError):
        return "URL unreachable — the secure connection (TLS) failed."
    return "URL unreachable — the server could not be contacted."


def _probe_url(url: str) -> UrlProbe:
    """Probe ``url`` with a HEAD request.

    A probe that cannot verify a URL must not get to veto it. Any HTTP response
    -- including a refusal -- proves the host resolved and answered, and the
    configured backend may well fetch what the probe could not: sites commonly
    refuse ``HEAD`` (405) or unrecognised clients (403), and a tldw server's
    browser-based clipper succeeds on pages our own client is refused (verified
    on a Wikipedia article that answers 403 to us even with a browser
    User-Agent, and that the server clipped at 200). Blocking those was task-697.

    A 404/410 is different in kind: the host is telling us the resource is not
    there, so refusing is right. A failure to *fetch* during ingest is reported
    as a failed job, where it carries a real reason.

    (TASK-19556) The egress policy is consulted BEFORE any transport call,
    with **no** trusted origins. An automatic probe is not a user asking to
    contact a host, so it may not seed trust from its own input URL -- the
    rule ``Utils/egress.py`` states for all shared pipeline code. That is
    also what makes the collapse below meaningful: with self-trust the check
    would be a no-op for exactly the private hosts at issue. The deliberate
    ingest that follows is a different matter and keeps its own trust
    (``[web_security]``: URLs you explicitly configure may be private).

    Args:
        url: URL to probe. Must already have passed ``validate_url``.

    Returns:
        A ``UrlProbe``. An empty one means the URL verified cleanly.
    """
    try:
        check_url_or_raise(url)
    except EgressBlockedError as exc:
        # ONE outcome for every declined reason. `exc.reason` is deliberately
        # not read: "private" vs "dns_failure" vs "metadata" is precisely the
        # difference an internal scan wants, and the user cannot act on it
        # here anyway.
        logger.debug(f"URL probe declined by egress policy: {exc}")
        return UrlProbe(note=_UNVERIFIABLE_NOTE)
    except Exception as exc:  # policy evaluation itself failed
        logger.debug(f"URL probe policy check failed for {url}: {exc!r}")
        return UrlProbe(note=_UNVERIFIABLE_NOTE)

    try:
        request = Request(url, method="HEAD")
        with _open_probe(request):
            return UrlProbe()
    except TimeoutError:
        return UrlProbe(error="URL probe timed out after 5 seconds")
    except HTTPError as exc:
        # An HTTP status means the host answered.
        if exc.code in _ABSENT_STATUSES:
            # (task-3305, MI-13) Plain language, not the exception's own
            # "HTTP Error 404: Not Found" phrasing.
            return UrlProbe(
                error=(
                    "URL unreachable — the server says this page does not "
                    f"exist (HTTP {exc.code})."
                )
            )
        return UrlProbe(
            note=(
                f"The site answered {exc.code} to our check, so it could not be "
                "confirmed ahead of time. The import will still be attempted."
            )
        )
    except URLError as exc:
        # No HTTP response at all: DNS failure, refused connection, bad TLS.
        # (task-3305, MI-13) The user-facing line names the failure's KIND;
        # the raw exception detail is debug-log material, never UI copy.
        logger.debug(f"URL probe failure for {url}: {exc!r}")
        return UrlProbe(error=_plain_unreachable_reason(exc))
    except Exception as exc:
        logger.debug(f"URL probe unexpected failure for {url}: {exc!r}")
        return UrlProbe(
            error="URL probe failed — the address could not be checked."
        )


def analyze_path(
    path_or_url: str, scan_limit: int = 1000, *, probe_url: bool | None = None
) -> PreflightResult:
    """Analyze a local path or URL before ingestion.

    Args:
        path_or_url: Local file path, directory path, or HTTP(S) URL.
        scan_limit: Maximum number of files to enumerate for directories.
            Must be greater than zero.
        probe_url: Whether a URL source may be probed over the network.
            ``None`` (the default) consults :func:`url_probe_enabled`;
            ``False`` forbids it outright. The while-typing caller in
            ``library_screen`` passes ``False`` so that even a user who has
            opted the probe in is not made to contact a host on every
            keystroke pause -- the probe then runs from the deliberate
            triggers (blur, Enter, Browse…, the retry button) instead.

    Returns:
        A ``PreflightResult`` describing the discovered source.

    Raises:
        ValueError: If ``scan_limit`` is less than or equal to zero.
    """
    if scan_limit <= 0:
        raise ValueError("scan_limit must be greater than zero")

    type_groups: dict[str, list[str]] = {}
    warnings: list[dict[str, Any]] = []
    errors: list[str] = []
    total_size = 0
    empty_files: list[str] = []
    truncated = False
    total_files = 0
    skipped_entries = 0
    path_invalid = False
    source_is_url = is_http_url(path_or_url)

    if source_is_url:
        if not validate_url(path_or_url):
            # (TASK-19556) Validation precedes every network request. At the
            # base of that task the probe fired first, so a malformed URL --
            # including one carrying embedded credentials
            # (``http://user:pass@host/``, which ``validate_url`` refuses
            # exactly because they end up in logs and forwarded URLs) -- was
            # put on the wire before anything looked at it.
            errors.append(
                "Invalid URL — check the address (it must be a plain http(s) "
                "link with no spaces or embedded credentials)."
            )
            path_invalid = True
        else:
            may_probe = url_probe_enabled() if probe_url is None else probe_url
            probe = _probe_url(path_or_url) if may_probe else UrlProbe()
            if probe.error:
                errors.append(probe.error)
            else:
                group = get_type_group(path_or_url)
                type_groups.setdefault(group, []).append(path_or_url)
                total_files = 1
                if probe.note:
                    warnings.append(
                        {"label": "Could not check the link", "hint": probe.note}
                    )
                warnings.extend(get_tooling_warnings(group))
    else:
        try:
            p = validate_path_simple(path_or_url, require_exists=False)
        except ValueError as e:
            errors.append(f"Invalid path: {e}")
            return PreflightResult(
                type_groups=type_groups,
                warnings=warnings,
                errors=errors,
                total_size=total_size,
                truncated=truncated,
                total_files=total_files,
                path_invalid=True,
            )
        if not p.exists():
            errors.append(f"Path not found: {path_or_url}")
            path_invalid = True
        elif p.is_file():
            size = _statted_size(p)
            total_size = size or 0
            total_files = 1
            if size == 0:
                empty_files.append(str(p))
            else:
                group = get_type_group(str(p))
                type_groups.setdefault(group, []).append(str(p))
                warnings.extend(get_tooling_warnings(group))
        elif p.is_dir():
            files, truncated, skipped_entries = _collect_files(p, scan_limit)
            total_files = len(files)
            for file_path in files:
                size = _statted_size(file_path)
                total_size += size or 0
                if size == 0:
                    empty_files.append(str(file_path))
                    continue
                group = get_type_group(str(file_path))
                type_groups.setdefault(group, []).append(str(file_path))
            for group in type_groups:
                if group == UNSUPPORTED_GROUP:
                    # No amount of installing makes these ingestible, so there
                    # is no tooling warning to raise for them -- they are
                    # counted separately in the summary instead.
                    continue
                warnings.extend(get_tooling_warnings(group))
        else:
            errors.append(f"Path is neither a file nor a directory: {path_or_url}")
            path_invalid = True

    return PreflightResult(
        type_groups=type_groups,
        warnings=warnings,
        errors=errors,
        total_size=total_size,
        truncated=truncated,
        total_files=total_files,
        path_invalid=path_invalid,
        empty_files=tuple(empty_files),
        source_is_url=source_is_url,
        skipped_entries=skipped_entries,
    )
