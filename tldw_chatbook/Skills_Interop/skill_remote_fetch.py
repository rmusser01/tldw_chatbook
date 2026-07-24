"""Remote skill fetch: URL classification, hardened download, bounded re-root.

The install path for "paste a GitHub link": classify -> fetch (SSRF-hardened)
-> re-root the archive to a single-skill zip -> the EXISTING hardened
``import_skill_file`` seam. This module owns all skill-install network I/O;
``LocalSkillsService`` stays network-free.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx


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


REMOTE_FETCH_MAX_BYTES = 30 * 1024 * 1024
REMOTE_FETCH_MAX_HOPS = 3
_FETCH_CHUNK = 65536
GITHUB_AUTH_HOSTS = frozenset(
    {"github.com", "api.github.com", "codeload.github.com",
     "objects.githubusercontent.com"}
)


def _default_resolver(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    return [info[4][0] for info in infos]


def _assert_host_allowed(host: str, resolver) -> None:
    """Reject unless EVERY resolved address is public (mixed sets reject)."""
    try:
        literal = ipaddress.ip_address(host)
        addresses = [str(literal)]
    except ValueError:
        try:
            addresses = resolver(host)
        except OSError as exc:
            raise RemoteSkillError(f"Could not resolve {host}.") from exc
    if not addresses:
        raise RemoteSkillError(f"Could not resolve {host}.")
    for raw in addresses:
        addr = ipaddress.ip_address(raw.split("%", 1)[0])
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            raise RemoteSkillError("That host is not reachable from here.")


async def fetch_zip_bytes(
    url: str, *, token: str | None = None, transport=None, resolver=None
) -> bytes:
    """Download a zip with SSRF hardening, manual redirects, and a size cap.

    Args:
        url: The (already classified/normalized) https URL to download.
        token: Optional GitHub token; attached ONLY on GITHUB_AUTH_HOSTS hops.
        transport: httpx transport override (tests).
        resolver: ``(host) -> list[str]`` override (tests); defaults to
            ``socket.getaddrinfo``.

    Returns:
        The raw (compressed) zip bytes, at most ``REMOTE_FETCH_MAX_BYTES``.

    Raises:
        RemoteSkillError: On every rejection class (scheme, host, hops, cap,
            HTTP status, timeouts).
    """
    from ..Utils.input_validation import validate_url

    resolver = resolver or _default_resolver
    timeout = httpx.Timeout(60.0, connect=10.0)
    current = url
    async with httpx.AsyncClient(
        transport=transport, timeout=timeout, follow_redirects=False
    ) as client:
        for _hop in range(REMOTE_FETCH_MAX_HOPS + 1):
            if not validate_url(current):
                raise RemoteSkillError("Invalid URL after redirect.")
            parsed = httpx.URL(current)
            if parsed.scheme != "https":
                raise RemoteSkillError("Only https:// is allowed (redirect downgraded).")
            host = (parsed.host or "").lower()
            _assert_host_allowed(host, resolver)
            headers = {}
            if token and host in GITHUB_AUTH_HOSTS:
                headers["Authorization"] = f"token {token}"
            try:
                async with client.stream("GET", current, headers=headers) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise RemoteSkillError("Redirect without a location.")
                        current = str(parsed.join(location))
                        continue
                    if response.status_code != 200:
                        raise RemoteSkillError(
                            f"Download failed (HTTP {response.status_code})."
                        )
                    chunks: list[bytes] = []
                    received = 0
                    async for chunk in response.aiter_bytes(_FETCH_CHUNK):
                        received += len(chunk)
                        if received > REMOTE_FETCH_MAX_BYTES:
                            raise RemoteSkillError(
                                "Download too large (over 30 MB compressed)."
                            )
                        chunks.append(chunk)
                    return b"".join(chunks)
            except httpx.TimeoutException as exc:
                raise RemoteSkillError("Download timed out.") from exc
            except httpx.HTTPError as exc:
                raise RemoteSkillError(f"Download failed: {exc}") from exc
        raise RemoteSkillError("Too many redirects.")
