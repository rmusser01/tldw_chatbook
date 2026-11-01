"""Closed source locators; remote declarations never select host filesystem paths."""

import re
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

from tldw_chatbook.Utils.path_validation import validate_canonical_directory

from .models import FrozenModel
from .package_files import validate_relative_member

ALLOWED_GIT_TRANSPORTS = frozenset({"https", "ssh", "local"})


class SourceLocator(FrozenModel):
    """Acquisition provenance only; none of these fields grants activation."""

    transport: Literal["https", "ssh", "local"]
    origin: str
    repository: str
    ref: str = "HEAD"
    subdirectory: str | None = None

    @property
    def url(self) -> str:
        return self.origin + self.repository


def require_transport(transport: str) -> None:
    if transport not in ALLOWED_GIT_TRANSPORTS:
        raise ValueError("plugin_source_transport_unsupported")


def _selectors(ref: str, subdirectory: str | None) -> None:
    if (
        not ref
        or len(ref.encode("utf-8")) > 1024
        or ref.startswith(("-", "/"))
        or ref.endswith((".", "/"))
        or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in ref)
        or any(c in ref for c in ("..", "@{", "//", "~", "^", ":", "?", "*", "[", "\\"))
        or any(
            part.startswith(".") or part.endswith(".lock") for part in ref.split("/")
        )
        or ref == "@"
    ):
        raise ValueError("plugin_source_ref_invalid")
    if (
        subdirectory is not None
        and validate_relative_member(subdirectory).as_posix() != subdirectory
    ):
        raise ValueError("plugin_source_subdirectory_invalid")


def parse_source_locator(value: str) -> SourceLocator:
    """Parse HTTPS/SSH or GitHub shorthand with #ref=…&subdir=… selectors.

    Local sources require ``select_local_source`` from an explicit user choice;
    catalog strings cannot grant authority to a host path. Errors are codes only.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("plugin_source_invalid")
    try:
        if len(value.encode("utf-8")) > 4096:
            raise ValueError("plugin_source_invalid")
    except UnicodeError:
        raise ValueError("plugin_source_invalid") from None
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("plugin_source_transport_unsupported")
    if value.startswith("github:"):
        path, separator, selectors = value[7:].partition("#")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path):
            raise ValueError("plugin_source_invalid")
        value = "https://github.com/" + path.removesuffix(".git") + ".git"
        if separator:
            value += "#" + selectors
    require_transport(value.partition(":")[0])
    try:
        parsed = urlsplit(value)
        require_transport(parsed.scheme)
        if parsed.scheme == "local":
            raise ValueError("plugin_source_local_selection_required")
        if (
            not parsed.hostname
            or parsed.password is not None
            or (parsed.username is not None and parsed.scheme != "ssh")
            or (parsed.scheme == "ssh" and parsed.username is None)
            or parsed.query
            or not re.fullmatch(r"[A-Za-z0-9.-]+", parsed.hostname)
            or (
                parsed.username is not None
                and not re.fullmatch(r"[A-Za-z0-9_-]+", parsed.username)
            )
            or not re.fullmatch(r"/[A-Za-z0-9_./~@+-]+", parsed.path)
            or any(part in {"", ".", ".."} for part in parsed.path[1:].split("/"))
        ):
            raise ValueError("plugin_source_invalid")
        port = parsed.port
        if port is not None and not 0 < port <= 65535:
            raise ValueError("plugin_source_invalid")
        pairs = parse_qsl(parsed.fragment, keep_blank_values=True, strict_parsing=True)
        if len(pairs) != len(dict(pairs)) or any(
            key not in {"ref", "subdir"} for key, _ in pairs
        ):
            raise ValueError("plugin_source_selectors_invalid")
        selectors = dict(pairs)
        ref, subdirectory = selectors.get("ref", "HEAD"), selectors.get("subdir")
        _selectors(ref, subdirectory)
        host = parsed.hostname.lower()
        if port is not None:
            host += f":{port}"
        if parsed.username is not None:
            host = parsed.username + "@" + host
        return SourceLocator(
            transport=parsed.scheme,
            origin=parsed.scheme + "://" + host,
            repository=parsed.path,
            ref=ref,
            subdirectory=subdirectory,
        )
    except (ValueError, UnicodeError):
        raise ValueError("plugin_source_invalid") from None


def select_local_source(
    path: Path, *, ref: str = "HEAD", subdirectory: str | None = None
) -> SourceLocator:
    """Record an explicitly selected, canonical existing host directory."""
    _selectors(ref, subdirectory)
    path = validate_canonical_directory(Path(path))
    return SourceLocator(
        transport="local",
        origin="",
        repository=str(path),
        ref=ref,
        subdirectory=subdirectory,
    )
