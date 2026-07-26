# tldw_chatbook/Utils/sensitive_paths.py
"""Paths no agent tool may read or write, regardless of configured root.

Shared by the `files` pack and (from Phase 4) ``run_command``, so the two
cannot drift. Two distinct reasons a path lands here:

1. **Credentials.** ``read_file`` carries no risk tag, so it resolves to
   the built-in ``allow`` floor and executes with no prompt. An unconfined
   read is therefore a zero-prompt path from a private key into a
   persisted transcript that may be sent to any provider.
2. **This application's own gate state.** A tool able to rewrite
   ``mcp_permissions.json`` or ``config.toml`` can turn every ``ask`` into
   ``allow`` -- a one-step bypass of the permission system.

This is a guardrail, not a security boundary: it stops accidents and naive
injected payloads, not a determined ``python -c``. The sandbox track is
the real answer for shell execution.
"""

from __future__ import annotations

from pathlib import Path

#: Directory prefixes that are refused along with everything beneath them.
_SENSITIVE_DIRS = (
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    "~/.config/gcloud",
    "~/.docker",
    "~/.kube",
    "~/.local/share/keyrings",
)

#: Individual files that are refused.
_SENSITIVE_FILES = (
    "~/.config/tldw_cli/config.toml",
    "~/.config/tldw_cli/mcp_permissions.json",
)


def _resolved(path_str: str) -> Path | None:
    try:
        return Path(path_str).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def is_sensitive_path(candidate: Path) -> bool:
    """Whether ``candidate`` is a credential or gate-state path.

    Comparison is by RESOLVED ancestry, never by string prefix, so
    ``~/.sshfoo`` is not mistaken for ``~/.ssh`` and a symlink cannot
    smuggle a path past the check.

    Args:
        candidate: The path a tool intends to touch.

    Returns:
        True when the path is refused. Fails CLOSED: a path that cannot be
        resolved is treated as sensitive.
    """
    resolved = _resolved(str(candidate))
    if resolved is None:
        return True

    for entry in _SENSITIVE_FILES:
        target = _resolved(entry)
        if target is not None and resolved == target:
            return True

    for entry in _SENSITIVE_DIRS:
        root = _resolved(entry)
        if root is not None and (resolved == root or root in resolved.parents):
            return True

    return False
