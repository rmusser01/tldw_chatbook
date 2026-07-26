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
