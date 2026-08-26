"""Lightweight shared types for library ingestion workflows.

This module exists so that state/UI modules can share pre-flight and job
result shapes without importing the heavy analysis modules that build them.
Keep it stdlib-only and free of optional dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PreflightResult:
    """Result of a pre-flight ingestion analysis.

    Args:
        type_groups: Mapping from capability group (``pdf``, ``audio_video``,
            ``ebook``, ``generic``) to the paths or URLs assigned to that group.
            An ``unsupported`` group may also be present for files that have no
            handler; callers that need to render supported groups separately
            should pop it before passing the dict to the UI.
        warnings: Tooling availability warnings from the capability layer.
        errors: Human-readable errors that would prevent ingestion.
        total_size: Sum of file sizes in bytes; ``0`` for URLs where the size
            is not known from the probe.
        truncated: ``True`` when a directory scan reached ``scan_limit``.
        total_files: Number of files discovered (``1`` for a reachable URL).
        path_invalid: ``True`` when the errors are about the *path itself* --
            missing, malformed, or neither a file nor a directory. Those are
            not worth retrying: the same path will fail the same way, and the
            fix is to correct it. A URL that failed to respond, by contrast,
            may well succeed on a second attempt.
    """

    type_groups: dict[str, list[str]]
    warnings: list[dict[str, Any]]
    errors: list[str]
    total_size: int
    truncated: bool
    total_files: int
    path_invalid: bool = False
    #: (task-2160) 0-byte files, pulled out of their type group at
    #: analysis time: the pipeline is guaranteed to fail them ("<name> is
    #: empty; there was nothing to ingest"), so the forecast must say so
    #: instead of promising "1 will import" for a file it measured at 0 B.
    empty_files: tuple[str, ...] = ()
    #: (task-2043) How many staged files appear to already exist in the
    #: Library (content-hash match, generic/text group only -- the DB hashes
    #: PARSED content, so only read≈parse types can be checked pre-parse).
    already_in_library: int = 0
    #: (task-2130) True when the duplicate check hit its candidate cap --
    #: ``already_in_library`` is then a floor, not a total, and the UI must
    #: say "at least N" rather than presenting the cap as the truth (an
    #: 80-duplicate folder read "20 files appear to already be…").
    already_in_library_capped: bool = False
    #: (task-3305, MI-19) True when the analyzed source was an http(s) URL.
    #: A URL's size is unknown to the probe, so ``total_size`` is 0 by
    #: construction -- the UI must not present that as "1 file · 0 B".
    source_is_url: bool = False
    #: (xhigh review of task-14823) Directory entries the scan passed over
    #: without collecting a file from them: symlinks, dot-entries, and
    #: folders it could not read. ``total_files == 0`` alone cannot tell
    #: "this folder holds nothing" from "this folder's entries were all
    #: skipped", and the ingest gate asserted the first about both -- a
    #: folder of symlinked media was told it was empty AND (since
    #: task-14823's submit gate) refused outright. ``0`` for non-directory
    #: sources, which have no entries to skip.
    skipped_entries: int = 0
