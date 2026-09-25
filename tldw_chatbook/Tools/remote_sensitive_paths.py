"""Remote-home-relative sensitive paths denied to the pinned remote worker.

Pure data (Phase 1d, Task 8): the locations on a REMOTE host — relative
to the remote user's home directory — that the remote worker bundle must
never read, list, or write, regardless of which workspace root a binding
pins. The parent's own denylist (``Utils/sensitive_paths.py``) resolves
THIS application's config/database paths and cannot travel: none of them
exist on the remote host, and resolving them would need the very config
machinery the bundle must not carry. This list is the remote-side floor.

The worker bundle embeds this tuple verbatim (rendered by
``Tools/build_remote_worker_bundle.py`` from this module — extract, not
copy-paste), so the data has exactly one definition site. Enforcement
wiring lands with the remote binding tasks; this module ships the data
and its embed.

Entries are ``PurePosixPath``-form relative strings: remote targets are
POSIX hosts reached over SSH, and matching joins them onto the resolved
remote home directory.
"""

from __future__ import annotations

REMOTE_SENSITIVE_PATHS: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".gnupg",
    ".config/gcloud",
    ".kube",
    ".docker",
    ".netrc",
)

__all__ = ["REMOTE_SENSITIVE_PATHS"]
