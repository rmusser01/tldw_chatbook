"""Remote skill fetch: URL classification, hardened download, bounded re-root.

The install path for "paste a GitHub link": classify -> fetch (SSRF-hardened)
-> re-root the archive to a single-skill zip -> the EXISTING hardened
``import_skill_file`` seam. This module owns all skill-install network I/O;
``LocalSkillsService`` stays network-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class GitHubZipSource:
    """A GitHub repo/tree URL normalized for zipball download."""

    owner: str
    repo: str
    tree_tail: tuple[str, ...]
    suggested_name: str


@dataclass(frozen=True)
class DirectZipSource:
    """A direct https zip URL (GitHub release asset or third-party)."""

    url: str
    github_auth: bool
    suggested_name: str


class RemoteSkillError(ValueError):
    """User-presentable remote-install failure."""


def _zip_basename(path_segment: str) -> str:
    name = path_segment.rsplit("/", 1)[-1]
    return name[:-4] if name.lower().endswith(".zip") else name


def classify_skill_source_url(url: str) -> GitHubZipSource | DirectZipSource:
    """Classify a pasted URL into a fetchable skill source.

    Args:
        url: The pasted URL.

    Returns:
        A ``GitHubZipSource`` (repo/tree form, zipball-normalized later) or a
        ``DirectZipSource`` (release asset / any https ``.zip``).

    Raises:
        RemoteSkillError: Non-https, non-GitHub non-zip, or malformed input.
    """
    # Deferred import: input_validation pulls metrics plumbing.
    from ..Utils.input_validation import validate_url

    if not validate_url(url):
        raise RemoteSkillError("That doesn't look like a valid URL.")
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RemoteSkillError("Only https:// URLs are supported.")
    host = (parsed.hostname or "").lower()
    path = parsed.path.strip("/")
    segments = tuple(s for s in path.split("/") if s)

    if host == "github.com":
        if len(segments) >= 4 and segments[2] == "releases" and segments[3] == "download":
            if not path.lower().endswith(".zip"):
                raise RemoteSkillError("Only .zip release assets are supported.")
            return DirectZipSource(
                url=url, github_auth=True, suggested_name=_zip_basename(path)
            )
        if len(segments) == 2:
            owner, repo = segments
            return GitHubZipSource(
                owner=owner, repo=repo, tree_tail=(), suggested_name=repo
            )
        if len(segments) >= 4 and segments[2] == "tree":
            owner, repo = segments[0], segments[1]
            tail = segments[3:]
            return GitHubZipSource(
                owner=owner,
                repo=repo,
                tree_tail=tail,
                suggested_name=tail[-1] if len(tail) > 1 else repo,
            )
        raise RemoteSkillError(
            "Unsupported GitHub URL — use a repo, /tree/<branch>/<subdir>, "
            "or a .zip release asset."
        )

    if path.lower().endswith(".zip"):
        return DirectZipSource(
            url=url, github_auth=False, suggested_name=_zip_basename(path)
        )
    raise RemoteSkillError(
        "Unsupported URL — paste a GitHub repo/subdirectory URL or a direct .zip link."
    )


async def resolve_ref_and_subdir(
    source: GitHubZipSource, list_branches
) -> tuple[str, str]:
    """Split a ``/tree/`` tail into (ref, subdir) per the spec's §2 rules.

    Single-segment tails are the ref outright. Multi-segment tails try a
    longest-prefix match against the repo's branch list (possibly capped at
    100 — a missing match degrades to the first-segment heuristic, never a
    wrong silent match). Branch-list failures degrade the same way.

    Args:
        source: The classified GitHub source (``tree_tail`` may be empty).
        list_branches: Async callable ``(owner, repo) -> list[str]``.

    Returns:
        ``(ref, subdir)`` — subdir is ``""`` or a POSIX relative path.
    """
    tail = source.tree_tail
    if not tail:
        return "HEAD", ""
    if len(tail) == 1:
        return tail[0], ""
    try:
        branches = await list_branches(source.owner, source.repo)
    except Exception:
        branches = []
    joined = "/".join(tail)
    best = ""
    for branch in branches:
        if (joined == branch or joined.startswith(branch + "/")) and len(branch) > len(best):
            best = branch
    if best:
        subdir = joined[len(best):].strip("/")
        return best, subdir
    return tail[0], "/".join(tail[1:])
