"""Admitted workspace-root descriptors: ``LocalRoot | RemoteRoot`` (Phase 3a).

Spec: ``Docs/superpowers/specs/2026-09-24-ssh-remote-workspace-bindings-design.md``
§ "Remote roots never touch the laptop's disk". Giving remote roots their
own TYPE is what makes "tools execute server-side" literally true at the
call site: a :class:`RemoteRoot` can never be mistaken for a laptop
``Path``, so every consumer that still does laptop-disk work fails loudly
at the boundary instead of silently reading the laptop's filesystem.

Migration shape (Phases 3b/3c tighten it):

- ``RunAdmittedWorkspaceRoot.root`` accepts the :data:`AdmittedRoot` union.
  A plain ``Path`` remains an implicit local root during the migration --
  every existing construction site (Console composition, agent worktrees,
  tests) passes one, and their behavior is pinned byte-identical. Task 18
  is the first caller that puts a :class:`RemoteRoot` in the slot.
- Consumers dispatch with :func:`is_remote` / :func:`local_root_path`:
  unwrap local roots, raise loud for remote roots that reached a
  laptop-disk site (Phase 3b/3c migrate those sites to worker-reported
  values).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Union

__all__ = [
    "AdmittedRoot",
    "LocalRoot",
    "RemoteRoot",
    "display_uri",
    "is_remote",
    "local_root_path",
]


@dataclass(frozen=True, slots=True)
class LocalRoot:
    """A workspace root that lives on THIS machine's filesystem."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class RemoteRoot:
    """A workspace root pinned on a remote host over SSH.

    A pure-Python descriptor: it carries no filesystem presence on the
    laptop, cannot be opened, stated, or resolved locally, and exists so
    laptop-disk call sites reject it at the type boundary. All path
    resolution for a remote root happens transport-side (the remote
    worker), and identity comes from the binding status cache.

    Attributes:
        alias: The stable run-level binding alias (``root_alias``).
        canonical_locator: The canonical SSH locator string (the
            context-note form Task 18 renders with :func:`display_uri`).
        root: The root's absolute path ON THE REMOTE HOST. POSIX only --
            SSH targets are POSIX hosts.
        binding_id: The registry binding this root was admitted through
            (the status-cache key for identity and guard state).
    """

    alias: str
    canonical_locator: str
    root: PurePosixPath
    binding_id: str

    def __post_init__(self) -> None:
        for field_name in ("alias", "canonical_locator", "binding_id"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        root = PurePosixPath(self.root)
        if not root.is_absolute():
            raise ValueError("root must be an absolute POSIX path")
        object.__setattr__(self, "root", root)


#: The typed slot every run-admitted authority carries from Phase 3a on.
#: Plain ``Path`` values remain valid (implicit local roots) while the
#: Phase 3 tasks migrate the remaining construction sites.
AdmittedRoot = Union[LocalRoot, RemoteRoot]


def is_remote(root: "Path | AdmittedRoot") -> bool:
    """Whether ``root`` is a remote (server-side) workspace root."""
    return isinstance(root, RemoteRoot)


def display_uri(root: "Path | AdmittedRoot") -> str:
    """Render one admitted root as its model-facing URI.

    ``LocalRoot``/``Path`` → ``file://`` + the (absolute) local path;
    ``RemoteRoot`` → ``ssh://`` + the canonical locator -- the form the
    workspace context note shows (Task 18 consumes this).
    """
    if isinstance(root, RemoteRoot):
        return f"ssh://{root.canonical_locator}"
    if isinstance(root, LocalRoot):
        return root.path.as_uri()
    return Path(root).as_uri()


def local_root_path(root: "Path | AdmittedRoot", *, site: str) -> Path:
    """Unwrap a local root to its ``Path``; refuse remote roots loudly.

    The Phase 3a boundary helper for every consumer that is about to do
    laptop-disk work with an admitted root. A ``RemoteRoot`` here is a
    composition bug (or an un-migrated Phase 3b/3c site) and must never
    degrade into a silent laptop read of a same-named local path.

    Args:
        root: The admitted-root slot value (``Path``, ``LocalRoot``, or
            ``RemoteRoot``).
        site: Short label naming the calling site -- it rides in the
            exception so the failure says WHICH laptop-disk path was
            reached and gives the migration tasks a grep anchor.

    Returns:
        The laptop ``Path`` for local roots; plain paths pass through
        unchanged (identity-preserving for the byte-identical LocalRoot
        contract).

    Raises:
        TypeError: ``root`` is a :class:`RemoteRoot`.
    """
    if isinstance(root, RemoteRoot):
        raise TypeError(f"remote root reached laptop-disk path: {site}")
    if isinstance(root, LocalRoot):
        return root.path
    if isinstance(root, Path):
        return root
    return Path(root)
