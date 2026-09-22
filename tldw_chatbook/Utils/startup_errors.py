"""Plain-language diagnostics for private-path startup refusals (task-32900).

A fresh install on a machine where an ancestor of the config/data directories
(e.g. ``~/.config`` at 0775, typically created by an earlier tool under a
permissive umask) used to die at ``import tldw_chatbook.config`` with a bare
``PrivatePathError: unsafe_parent: shared_writable_parent`` traceback. The
refusal is correct and stays fail-closed (ADR-029/ADR-127); what was missing
was telling the user *which* directory, *why*, and the exact repair command.

This module must stay importable before the heavy chain, exactly like
``startup_logging``: it imports nothing from tldw_chatbook except the
stdlib-only ``Utils.private_paths`` leaf.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .private_paths import PrivatePathError, PrivatePathResult

_WIDTH = 72

#: Refusals caused by HOME living in a shared sticky directory (/tmp & co):
#: relocating the config under that same HOME cannot help, so the alternative
#: block must talk about HOME itself instead of TLDW_CONFIG_PATH.
_STICKY_HOME_REASONS = frozenset(
    {
        "missing_component_in_shared_sticky_parent",
        "missing_leaf_in_shared_sticky_parent",
        "shared_sticky_directory_not_allowed",
    }
)

#: Refusals about which data root to use: the config-file alternative is
#: irrelevant (the guidance already names the [paths] data_dir choice).
_DATA_ROOT_REASONS = frozenset({"ambiguous_default_data_roots"})

_emitted = False


def _shell_path(path: Path | str) -> str:
    """Quote a filesystem path for safe copy-paste into a POSIX shell.

    Paths come from the user's own filesystem and are pasted verbatim, so a
    path containing whitespace, quotes, or shell metacharacters must arrive at
    chmod/chown as a single operand rather than as injected syntax.
    """

    return shlex.quote(str(path))


def _shared_writable_guidance(result: PrivatePathResult) -> list[str]:
    offender = result.offender_path
    lines = [
        "Remove group/world write permission from the blocked directory",
        "shown above, then start Chatbook again:",
    ]
    if offender is not None:
        lines += ["", f"    chmod g-w,o-w -- {_shell_path(offender)}"]
    else:
        lines += [
            "",
            "    namei -l <location-above>   # Linux: shows each directory's mode",
            "    ls -ld <each directory on the path>",
        ]
    lines += [
        "",
        "This commonly happens when an earlier tool created the directory",
        "under a permissive umask (mode 0775 means group-writable).",
    ]
    return lines


def _sticky_shared_guidance(result: PrivatePathResult) -> list[str]:
    return [
        "The blocked directory lives in a shared, sticky directory such as",
        "/tmp, which any local user can read and fill. HOME (or the parent",
        "shown above) must be an ordinary private directory:",
        "",
        "    * log in as a regular user with its own home directory, or",
        "    * point HOME somewhere that is not shared before starting.",
    ]


def _ownership_guidance(result: PrivatePathResult) -> list[str]:
    offender = result.offender_path
    lines = [
        "The blocked directory belongs to a different user. This usually",
        "means Chatbook is running under sudo/su while HOME points at",
        "another account's files. Run Chatbook as the user who owns that",
        "directory, or change who owns it:",
    ]
    if offender is not None:
        lines += ["", f'    sudo chown "$USER" -- {_shell_path(offender)}']
    # A foreign-owned directory can also be group/world-writable; repairing
    # only the owner would leave the very next check failing.
    if result.offender_mode is not None and result.offender_mode & 0o022:
        lines += [
            "",
            "That directory is also group- or world-writable, so also remove",
            "the write bits or startup will refuse again:",
        ]
        if offender is not None:
            lines += ["", f"    chmod g-w,o-w -- {_shell_path(offender)}"]
    return lines


def _ambiguous_roots_guidance(result: PrivatePathResult) -> list[str]:
    return [
        "Both ~/.local/share/tldw_cli and ~/.tldw_cli-data exist, so Chatbook",
        "cannot tell which one holds your data. Remove the one you do not",
        "want, or set an explicit choice in the config file:",
        "",
        "    [paths]",
        "    data_dir = \"/absolute/path/to/the/root/you/want\"",
    ]


def _generic_guidance(result: PrivatePathResult) -> list[str]:
    return [
        "Check the ownership and permissions of every directory on the path",
        "shown above. On Linux, `namei -l <path>` lists them one line each;",
        "every directory must belong to root or to you, and must not be",
        "group- or world-writable (unless it is a sticky directory like /tmp).",
    ]


_GUIDANCE: dict[str, Callable[[PrivatePathResult], list[str]]] = {
    "shared_writable_parent": _shared_writable_guidance,
    "missing_component_in_shared_sticky_parent": _sticky_shared_guidance,
    "missing_leaf_in_shared_sticky_parent": _sticky_shared_guidance,
    "shared_sticky_directory_not_allowed": _sticky_shared_guidance,
    "untrusted_directory_owner": _ownership_guidance,
    "application_directory_wrong_owner": _ownership_guidance,
    "sticky_child_wrong_owner": _ownership_guidance,
    "wrong_owner": _ownership_guidance,
    "ambiguous_default_data_roots": _ambiguous_roots_guidance,
}


def _rule(character: str = "=") -> str:
    return character * _WIDTH


def _alternative_lines(result: PrivatePathResult) -> list[str]:
    """Reason-aware escape hatch that does not repeat the failed check."""

    reason = result.reason or ""
    if reason in _STICKY_HOME_REASONS:
        return [
            "ALTERNATIVE",
            "Start Chatbook with HOME pointing at a private directory you",
            "own (replace the example with your own path):",
            "",
            '    HOME=/home/you tldw-cli',
        ]
    if reason in _DATA_ROOT_REASONS:
        # The HOW TO FIX block above already names the [paths] data_dir
        # choice; a config-path alternative cannot resolve a data-root
        # refusal, so do not offer one.
        return []
    return [
        "ALTERNATIVE",
        "Keep the directory as it is and point the config elsewhere instead",
        '(a file inside a directory only you can write):',
        "",
        '    TLDW_CONFIG_PATH="$HOME/tldw-config.toml" tldw-cli',
        "",
        "If the blocked path above belongs to Chatbook's data storage",
        "rather than its config, set [paths] data_dir in the config file to",
        "a private directory instead.",
    ]


def format_private_path_error(exc: PrivatePathError) -> str:
    """Render a private-path refusal as an actionable first-run diagnostic.

    Args:
        exc: The refusal to explain. ``exc.result`` supplies the refused
            location, the machine-stable reason, and any diagnostics-only
            offender path/mode captured by the refusing walk.

    Returns:
        The full diagnostic as a single newline-joined string, wrapped at
        72 columns, ready to print to stderr.
    """

    result = exc.result
    reason = result.reason or result.status.value
    lines = [
        _rule(),
        " Chatbook cannot start: its private storage location is not",
        " secure on this machine",
        _rule(),
        "",
        "Chatbook stores API keys and other secrets in its config and data",
        "directories, so it refuses to run from any location that other",
        "users on this machine could modify. Nothing outside Chatbook's own",
        "config/data directories was created or changed.",
        "",
        f"Location it was trying to use: {result.lexical_path}",
    ]
    if result.offender_path is not None:
        detail = f" ({result.offender_detail})" if result.offender_detail else ""
        lines.append(f"Blocked by: {result.offender_path}{detail}")
    lines += [f"Reason: {reason}", ""]

    lines += ["HOW TO FIX"]
    lines += _GUIDANCE.get(reason, _generic_guidance)(result)
    alternative = _alternative_lines(result)
    if alternative:
        lines += [""]
        lines += alternative
    lines += [
        "",
        "Technical detail for bug reports:",
        f"    {exc}",
    ]
    return "\n".join(lines)


def emit_private_path_startup_error(exc: PrivatePathError) -> None:
    """Print the diagnostic to stderr once per process.

    Args:
        exc: The refusal to explain, rendered by
            :func:`format_private_path_error`.

    config.py emits before re-raising the import-time refusal, and cli.py
    emits again from its handler in case the failure came from a later
    startup stage; the guard keeps the two from duplicating on screen.
    """

    global _emitted
    if _emitted:
        return
    _emitted = True
    print(format_private_path_error(exc), file=sys.stderr)


def reset_emit_guard_for_tests() -> None:
    """Reset the once-per-process guard between tests."""

    global _emitted
    _emitted = False


__all__: Sequence[str] = (
    "emit_private_path_startup_error",
    "format_private_path_error",
    "reset_emit_guard_for_tests",
)
