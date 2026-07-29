"""
Path validation utilities to prevent directory traversal attacks.

This module provides functions to validate file paths and ensure they don't
escape allowed directories.
"""

import os
import time
from pathlib import Path
from typing import Optional, Sequence, Union
from loguru import logger
from ..Metrics.metrics_logger import log_counter, log_histogram

#: Ultra-short recovery pointer, LED with (round 1 review CRITICAL 2): the
#: full `ROOT_DENIAL_RECOVERY_HINT` sentence below is itself too long to
#: survive the Console transcript's live tool-step-marker truncation
#: (``console_agent_bridge._STEP_MARKER_RESULT_LIMIT`` = 160 chars) once
#: the tool's own "Failed to {read/write/list} ... " prefix and the path
#: itself are accounted for -- measured: for a realistic 45-60 char path,
#: the visible (pre-truncation) budget is only ~155-158 chars, and the
#: prefix + "Path '<path>' is outside every allowed root." core sentence
#: alone already consumes ~95-135 of that, leaving too little room for
#: "Settings > Workspaces" to ever appear. This pointer is deliberately the
#: FIRST thing after the bare "outside every allowed root" acknowledgement
#: -- before the path is even repeated -- so it is what survives
#: truncation, not what gets cut. See `test_recovery_pointer_survives_real_
#: transcript_truncation` (Tests/Utils/test_path_validation_multi.py) for
#: the actual truncation math, not just an estimate.
#:
#: Qodo PR #1074 finding 3: the ORIGINAL pointer ("create a workspace +
#: bind a folder") baked in the Default-workspace assumption unconditionally
#: -- misleading for the common case of a run already in a normal, named
#: workspace, where the actual fix is just "bind a folder" (creating
#: another workspace would be actively wrong advice there). This function's
#: caller, `validate_path_multi` below, has no cheap way to know which
#: workspace the denied run belongs to: it is a generic multi-root path
#: validator with no workspace awareness, called from three sites in
#: `Tools/file_operation_tools.py` that would each need to resolve and
#: thread the run's *effective* workspace id here -- duplicating
#: `workspace_file_roots.allowed_file_roots`'s own None-falls-back-to-
#: active-workspace logic, across a Utils -> Tools/Workspaces layering
#: boundary this module does not otherwise cross, just to pick a copy
#: variant. So the pointer stays workspace-agnostic and universally
#: correct instead, and the Default-specific caveat moves to
#: `ROOT_DENIAL_RECOVERY_HINT` below, reworded as an explicit conditional
#: rather than an assertion.
ROOT_DENIAL_RECOVERY_POINTER = "Fix: bind a folder in Settings > Workspaces."

#: Fuller explanation appended AFTER the pointer above and the (now
#: second-priority) path/consulted-roots detail (TASK-1231, fleet-UX review
#: F3; reworded to a conditional per Qodo PR #1074 finding 3): on a fresh
#: install every session starts on the Default workspace, which cannot
#: hold folder bindings -- the FIRST file-tool call a model makes there is
#: always rejected. Phrased as an "if" (not "you are in Default") because
#: this exact denial also fires for a normal, already-named workspace that
#: simply has no folder bound yet -- for that run, "create a NEW
#: workspace" would be wrong; it only needs the bind-a-folder step the
#: pointer above already covers. This is the part truncation is allowed to
#: eat into (along with the consulted-roots list) -- it is not the user's
#: only route to the fix, `ROOT_DENIAL_RECOVERY_POINTER` above is.
ROOT_DENIAL_RECOVERY_HINT = (
    "The Default workspace cannot hold folder bindings -- create a named "
    "workspace first if this run is in Default, then bind a folder to it "
    "and use a session in that workspace."
)


def validate_path(
    user_path: Union[str, Path], base_directory: Union[str, Path]
) -> Path:
    """
    Validates that a user-provided path is within the allowed base directory.

    Args:
        user_path: The path provided by the user
        base_directory: The allowed base directory

    Returns:
        Path: The validated absolute path

    Raises:
        ValueError: If the path is invalid or attempts directory traversal
    """
    start_time = time.time()
    log_counter("path_validation_validate_path_attempt")

    try:
        # Convert to Path objects
        user_path = Path(user_path)
        base_directory = Path(base_directory).resolve()

        # Resolve the full path (follows symlinks and resolves ..)
        if user_path.is_absolute():
            full_path = user_path.resolve()
        else:
            full_path = (base_directory / user_path).resolve()

        # Check if the resolved path is within the base directory
        try:
            full_path.relative_to(base_directory)
        except ValueError:
            logger.warning(
                f"Path traversal attempt detected: {user_path} -> {full_path}"
            )
            log_counter(
                "path_validation_security_violation",
                labels={"type": "directory_traversal"},
            )
            raise ValueError(f"Path '{user_path}' is outside the allowed directory")

        # Additional checks for safety.
        # Hidden-file check applies only to the user-supplied portion (relative
        # to base_directory) — a base dir that itself lives under a dotted
        # ancestor (e.g. ~/.local/share/...) must not falsely trip this.
        relative_parts = full_path.relative_to(base_directory).parts
        if any(part.startswith(".") for part in relative_parts if part != "."):
            logger.warning(f"Hidden file/directory access attempt: {full_path}")
            log_counter(
                "path_validation_security_violation",
                labels={"type": "hidden_file_access"},
            )
            raise ValueError("Access to hidden files/directories is not allowed")

        # Some callers pass the destination's own immediate parent as
        # base_directory (e.g. to validate an arbitrary user-chosen export
        # destination while still using this function for traversal/symlink
        # checks, rather than confining to one fixed app-data root). In that
        # pattern a hidden final directory is folded into base_directory
        # itself, so it never appears in relative_parts above and the check
        # is silently bypassed. Catch that by also rejecting a base
        # directory whose own final component is dotted. This deliberately
        # does not walk base_directory's ancestors, so a base dir that lives
        # *under* a dotted ancestor (e.g. ~/.local/share/...) is unaffected.
        if base_directory.name.startswith("."):
            logger.warning(f"Hidden base directory rejected: {base_directory}")
            log_counter(
                "path_validation_security_violation",
                labels={"type": "hidden_file_access"},
            )
            raise ValueError("Access to hidden files/directories is not allowed")

        # Log success
        duration = time.time() - start_time
        log_histogram(
            "path_validation_validate_path_duration",
            duration,
            labels={"status": "success"},
        )
        log_counter("path_validation_validate_path_success")

        return full_path

    except Exception as e:
        # Log error
        duration = time.time() - start_time
        log_histogram(
            "path_validation_validate_path_duration",
            duration,
            labels={"status": "error"},
        )
        log_counter(
            "path_validation_validate_path_error",
            labels={"error_type": type(e).__name__},
        )

        logger.error(f"Path validation error for '{user_path}': {e}")
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Invalid path: {user_path}")


def validate_filename(filename: str) -> str:
    """
    Validates a filename to ensure it doesn't contain path separators or other dangerous characters.

    Args:
        filename: The filename to validate

    Returns:
        str: The validated filename

    Raises:
        ValueError: If the filename is invalid
    """
    start_time = time.time()
    log_counter("path_validation_validate_filename_attempt")

    if not filename:
        log_counter(
            "path_validation_validate_filename_error",
            labels={"error_type": "empty_filename"},
        )
        raise ValueError("Filename cannot be empty")

    # Check for path separators
    if os.path.sep in filename or "/" in filename or "\\" in filename:
        log_counter(
            "path_validation_security_violation",
            labels={"type": "path_separator_in_filename"},
        )
        raise ValueError("Filename cannot contain path separators")

    # Check for parent directory references
    if ".." in filename:
        log_counter(
            "path_validation_security_violation",
            labels={"type": "parent_directory_reference"},
        )
        raise ValueError("Filename cannot contain parent directory references")

    # Check for null bytes
    if "\x00" in filename:
        log_counter(
            "path_validation_security_violation",
            labels={"type": "null_byte_in_filename"},
        )
        raise ValueError("Filename cannot contain null bytes")

    # Check for reserved names on Windows
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }

    name_without_ext = filename.split(".")[0].upper()
    if name_without_ext in reserved_names:
        log_counter(
            "path_validation_security_violation", labels={"type": "reserved_filename"}
        )
        raise ValueError(f"'{filename}' is a reserved filename")

    # Log success
    duration = time.time() - start_time
    log_histogram("path_validation_validate_filename_duration", duration)
    log_counter("path_validation_validate_filename_success")

    return filename


def safe_join_path(base_directory: Union[str, Path], *paths: Union[str, Path]) -> Path:
    """
    Safely joins paths ensuring the result stays within the base directory.

    Args:
        base_directory: The base directory
        *paths: Path components to join

    Returns:
        Path: The safely joined path

    Raises:
        ValueError: If the resulting path would escape the base directory
    """
    base = Path(base_directory).resolve()

    # Start with the base directory
    result = base

    for path_component in paths:
        # Validate each component
        if isinstance(path_component, str):
            validate_filename(path_component)
        result = result / path_component

    # Validate the final path
    return validate_path(result, base)


def is_safe_path(user_path: Union[str, Path], base_directory: Union[str, Path]) -> bool:
    """
    Checks if a path is safe without raising exceptions.

    Args:
        user_path: The path to check
        base_directory: The allowed base directory

    Returns:
        bool: True if the path is safe, False otherwise
    """
    try:
        validate_path(user_path, base_directory)
        return True
    except ValueError:
        return False


def get_safe_relative_path(
    full_path: Union[str, Path], base_directory: Union[str, Path]
) -> Optional[Path]:
    """
    Gets the relative path from base_directory, or None if the path is unsafe.

    Args:
        full_path: The full path
        base_directory: The base directory

    Returns:
        Optional[Path]: The relative path, or None if unsafe
    """
    try:
        full_path = Path(full_path).resolve()
        base_directory = Path(base_directory).resolve()
        return full_path.relative_to(base_directory)
    except ValueError:
        return None


def validate_path_simple(
    user_path: Union[str, Path],
    require_exists: bool = False,
    *,
    probe_existing: bool = True,
) -> Path:
    """
    Simple path validation that checks for common security issues without requiring a base directory.

    Args:
        user_path: The path to validate
        require_exists: Whether to require the path exists
        probe_existing: Whether to inspect and resolve an existing selected path.
            Disable this when a later no-follow boundary owns link validation.

    Returns:
        Path: The validated path

    Raises:
        ValueError: If the path contains security risks
    """
    start_time = time.time()
    log_counter("path_validation_validate_path_simple_attempt")

    try:
        path_str = str(user_path)

        # Check for null bytes
        if "\x00" in path_str:
            log_counter(
                "path_validation_security_violation", labels={"type": "null_byte"}
            )
            raise ValueError("Path cannot contain null bytes")

        # Check for obvious traversal attempts
        dangerous_patterns = [
            "../..",  # Multiple parent refs (POSIX)
            "..\\..\\",  # Multiple parent refs (Windows) -- kept in parity
            # with the POSIX pattern above; a single "..\" segment is a
            # legitimate, unresolved component (e.g. "nested\..\locks") and
            # must not be flagged here. This function has no base directory
            # to resolve against, so genuine traversal outside an intended
            # base is the sibling validate_path()/validate_path_safety()'s
            # job; this raw substring scan only catches egregious,
            # unresolvable inputs.
            "~/",  # Home directory expansion
            "~\\",  # Windows home
            "\x00",  # Null byte
            "|",  # Pipe (command injection)
            ";",  # Command separator
            "&&",  # Command chaining
            "||",  # Command chaining
            "`",  # Command substitution
            "$(",  # Command substitution
            "${",  # Variable expansion
        ]

        for pattern in dangerous_patterns:
            if pattern in path_str:
                log_counter(
                    "path_validation_security_violation",
                    labels={"type": "dangerous_pattern", "pattern": pattern},
                )
                raise ValueError(f"Path contains dangerous pattern: {pattern}")

        # Convert to Path object and check basic validity
        path = Path(user_path)

        if probe_existing:
            # If path exists, resolve it to catch symlink attacks
            if path.exists():
                resolved = path.resolve()
                # Check if resolution changed the path significantly (possible symlink attack)
                if path.is_absolute() and resolved != path:
                    logger.warning("Path resolution changed during validation")
            elif require_exists:
                raise ValueError("Path does not exist")
        elif require_exists:
            raise ValueError("require_exists requires probe_existing=True")

        # Log success
        duration = time.time() - start_time
        log_histogram("path_validation_validate_path_simple_duration", duration)
        log_counter("path_validation_validate_path_simple_success")

        return path

    except Exception as e:
        # Log error
        duration = time.time() - start_time
        log_histogram(
            "path_validation_validate_path_simple_duration",
            duration,
            labels={"status": "error"},
        )
        log_counter(
            "path_validation_validate_path_simple_error",
            labels={"error_type": type(e).__name__},
        )

        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Invalid path: {user_path}")


def validate_path_multi(
    user_path: Union[str, Path], roots: Sequence[Union[str, Path]]
) -> Path:
    """Validate ``user_path`` against several allowed roots (first match wins).

    Relative paths resolve against ``roots[0]`` (the primary root — callers
    pass the tool sandbox first so legacy relative-path behavior is
    unchanged). The rejection message names every consulted root so a
    denial is actionable.

    Args:
        user_path: The path provided by the user or model.
        roots: Allowed base directories, in priority order.

    Returns:
        The validated absolute path.

    Raises:
        ValueError: No roots given, or the path escapes all of them.
    """
    root_list = [Path(root) for root in roots]
    if not root_list:
        raise ValueError("No allowed roots configured for path validation.")
    candidate = Path(user_path)
    for index, root in enumerate(root_list):
        if index > 0 and not candidate.is_absolute():
            continue  # relative paths anchor to the primary root only
        try:
            return validate_path(user_path, root)
        except ValueError:
            continue
    consulted = ", ".join(str(root.resolve()) for root in root_list)
    # Ordering is load-bearing (round 1 review CRITICAL 2): the ultra-short
    # POINTER comes first, before the path is even repeated, because it is
    # the one piece of this message that MUST survive the transcript's
    # 160-char truncation -- the path and consulted-roots list may not.
    raise ValueError(
        f"Outside every allowed root. {ROOT_DENIAL_RECOVERY_POINTER} "
        f"Path: '{user_path}'. {ROOT_DENIAL_RECOVERY_HINT} "
        f"(Checked: {consulted})"
    )
