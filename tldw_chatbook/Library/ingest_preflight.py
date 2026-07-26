"""Pre-flight analyzer for library ingestion sources.

Provides a lightweight, dependency-cheap way to inspect a local path or URL
before starting an ingest job. It reports discovered file type groups,
estimated size, tooling warnings, and any errors that would prevent ingestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tldw_chatbook.Library.ingest_capabilities import (
    UNSUPPORTED_GROUP,
    get_tooling_warnings,
    get_type_group,
)
from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Local_Ingestion.local_file_ingestion import is_http_url
from tldw_chatbook.Utils.path_validation import validate_path_simple


def _safe_size(path: Path) -> int:
    """Return the size of ``path`` in bytes, or ``0`` on ``OSError``."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


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
    return _collect_files(directory, scan_limit)


def _collect_files(p: Path, scan_limit: int) -> tuple[list[Path], bool]:
    """Recursively collect files under ``p`` up to ``scan_limit``.

    Symlinks and hidden entries (names starting with ``.``) are skipped to avoid
    cycles, system files, and unexpected traversal. Directories that raise
    ``PermissionError`` are skipped.

    Args:
        p: Directory to scan.
        scan_limit: Maximum number of files to collect.

    Returns:
        A tuple of ``(files, truncated)``. ``truncated`` is ``True`` when there
        were additional files beyond ``scan_limit`` that could not be collected.
    """
    files: list[Path] = []
    truncated = False

    try:
        entries = list(p.iterdir())
    except OSError:
        return files, truncated

    for entry in entries:
        if entry.is_symlink() or entry.name.startswith("."):
            continue

        try:
            if entry.is_dir():
                remaining = scan_limit - len(files)
                if remaining <= 0:
                    # The limit is already reached; only mark truncated if this
                    # directory actually contains files we cannot collect.
                    sub_files, _ = _collect_files(entry, 1)
                    if sub_files:
                        truncated = True
                        break
                    continue
                sub_files, sub_truncated = _collect_files(entry, remaining)
                files.extend(sub_files)
                if sub_truncated:
                    truncated = True
                    break
            elif entry.is_file():
                if len(files) >= scan_limit:
                    truncated = True
                    break
                files.append(entry)
        except PermissionError:
            continue

    return files, truncated


#: HTTP statuses that mean "this resource is not there", as distinct from "the
#: host answered but would not confirm it for us". Only the former justifies
#: refusing to start.
_ABSENT_STATUSES = frozenset({404, 410})


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

    Args:
        url: URL to probe.

    Returns:
        A ``UrlProbe``. An empty one means the URL verified cleanly.
    """
    try:
        request = Request(url, method="HEAD")
        with urlopen(request, timeout=5):
            return UrlProbe()
    except TimeoutError:
        return UrlProbe(error="URL probe timed out after 5 seconds")
    except HTTPError as exc:
        # An HTTP status means the host answered.
        if exc.code in _ABSENT_STATUSES:
            return UrlProbe(error=f"URL unreachable: {exc}")
        return UrlProbe(
            note=(
                f"The site answered {exc.code} to our check, so it could not be "
                "confirmed ahead of time. The import will still be attempted."
            )
        )
    except URLError as exc:
        # No HTTP response at all: DNS failure, refused connection, bad TLS.
        return UrlProbe(error=f"URL unreachable: {exc}")
    except Exception as exc:
        return UrlProbe(error=f"URL probe failed: {exc}")


def analyze_path(path_or_url: str, scan_limit: int = 1000) -> PreflightResult:
    """Analyze a local path or URL before ingestion.

    Args:
        path_or_url: Local file path, directory path, or HTTP(S) URL.
        scan_limit: Maximum number of files to enumerate for directories.
            Must be greater than zero.

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
    truncated = False
    total_files = 0
    path_invalid = False

    if is_http_url(path_or_url):
        probe = _probe_url(path_or_url)
        if probe.error:
            errors.append(probe.error)
        else:
            group = get_type_group(path_or_url)
            type_groups.setdefault(group, []).append(path_or_url)
            total_files = 1
            if probe.note:
                warnings.append({"label": "Could not check the link", "hint": probe.note})
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
            group = get_type_group(str(p))
            type_groups.setdefault(group, []).append(str(p))
            total_size = _safe_size(p)
            total_files = 1
            warnings.extend(get_tooling_warnings(group))
        elif p.is_dir():
            files, truncated = _collect_files(p, scan_limit)
            total_files = len(files)
            for file_path in files:
                group = get_type_group(str(file_path))
                type_groups.setdefault(group, []).append(str(file_path))
                total_size += _safe_size(file_path)
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
    )
