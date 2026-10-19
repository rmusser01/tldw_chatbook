"""Bounded marketplace discovery; locators are declarations, never host authority."""

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from .git_source import parse_source_locator
from .inspection import catalog_identity
from .models import Diagnostic, FrozenModel
from .package_files import (
    PackageFileError,
    _unique_pairs,
    canonical_json,
    parse_document,
    validate_relative_member,
)

CATALOG_PATHS = (
    ".agents/plugins/marketplace.json",
    ".cursor-plugin/marketplace.json",
    ".claude-plugin/marketplace.json",
)
MAX_CATALOG_BYTES = 5 * 1024 * 1024
MAX_CATALOG_ENTRIES = 10_000
MAX_SOURCES = 50


class CatalogEntry(FrozenModel):
    source_id: str = Field(max_length=100)
    identity: str = Field(max_length=256)
    name: str = Field(max_length=100)
    package_locator: str | None = Field(default=None, max_length=8192)
    local_path: str | None = Field(default=None, max_length=4096)
    overlays: tuple[str, ...] = ()
    support: Literal["supported", "unsupported"] = "supported"
    diagnostics: tuple[Diagnostic, ...] = ()
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class CatalogSnapshot(FrozenModel):
    source_id: str = Field(max_length=100)
    catalog_commit: str | None = Field(max_length=40)
    digest: str = Field(pattern=r"^(?:|[0-9a-f]{64})$")
    fetched_at: str = Field(max_length=64)
    entries: tuple[CatalogEntry, ...] = Field(max_length=MAX_CATALOG_ENTRIES)
    refresh_error: str | None = Field(default=None, max_length=100)


def catalog_local_path(root: Path, relative: str) -> Path:
    """Resolve a catalog declaration against its repository root, never metadata."""
    member = validate_relative_member(relative)
    root = root.resolve()
    candidate = (root / member).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("catalog_path_outside_root")
    return candidate


def select_catalog(root: Path, selected: str | None = None) -> str:
    present = tuple(path for path in CATALOG_PATHS if (root / path).is_file())
    if selected is not None:
        if selected not in present:
            raise ValueError("catalog_selection_missing")
        return selected
    if len(present) != 1:
        raise ValueError("catalog_selection_required")
    return present[0]


def _entry(source_id, index, raw, prefix):
    encoded = canonical_json(raw)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    name = raw.get("name") if isinstance(raw, dict) else None
    valid_name = isinstance(name, str) and re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}", name
    )
    name = name if valid_name else f"invalid:{index}"
    identity = f"{source_id}:{name}"
    try:
        if not valid_name:
            raise ValueError("catalog_name_invalid")
        # Reuse the package JSON bounds independently for each sibling.
        parse_document(encoded.encode())
        source = raw.get("source")
        local = locator = None
        if isinstance(source, str) and source.startswith("./"):
            local = source
        elif isinstance(source, dict):
            kind = source.get("source")
            if kind == "local":
                local = source.get("path")
            elif kind in {"url", "git-subdir", "github"}:
                ref, sha = source.get("ref"), source.get("sha")
                if ref is not None and sha is not None:
                    raise ValueError("catalog_conflicting_selectors")
                if sha is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
                    raise ValueError("catalog_sha_invalid")
                url = (
                    source.get("url")
                    if kind != "github"
                    else "github:" + source.get("repo", "")
                )
                selected = parse_source_locator(url)
                if "#" in url and (
                    ref is not None or sha is not None or "path" in source
                ):
                    raise ValueError("catalog_conflicting_selectors")
                from urllib.parse import urlencode

                selectors = {"ref": ref or sha or selected.ref}
                subdirectory = source.get("path", selected.subdirectory)
                if kind == "git-subdir" and not subdirectory:
                    raise ValueError("catalog_subdirectory_required")
                if subdirectory is not None:
                    selectors["subdir"] = validate_relative_member(
                        subdirectory
                    ).as_posix()
                locator = parse_source_locator(
                    selected.url + "#" + urlencode(selectors)
                ).model_dump_json()
            else:
                raise ValueError("catalog_source_unsupported")
        else:
            raise ValueError("catalog_source_unsupported")
        if local is not None:
            local = validate_relative_member(local).as_posix()
            if prefix:
                local = validate_relative_member(prefix + "/" + local).as_posix()
        elif locator is None:
            raise ValueError("catalog_source_invalid")
        overlay = {
            key: value for key, value in raw.items() if key not in {"source", "strict"}
        }
        return CatalogEntry(
            source_id=source_id,
            identity=identity,
            name=name,
            package_locator=locator,
            local_path=local,
            overlays=(catalog_identity(overlay),),
            digest=digest,
        )
    except (TypeError, ValueError, AttributeError):
        return CatalogEntry(
            source_id=source_id,
            identity=identity,
            name=name,
            support="unsupported",
            diagnostics=(Diagnostic(code="catalog_entry_unsupported"),),
            digest=digest,
        )


def parse_catalog(
    data: bytes,
    *,
    source_id: str,
    catalog_path: str,
    catalog_commit: str | None = None,
    fetched_at: str = "",
) -> CatalogSnapshot:
    """Normalize all supported shapes while retaining malformed entry diagnostics."""
    if catalog_path not in CATALOG_PATHS or len(data) > MAX_CATALOG_BYTES:
        raise ValueError("catalog_limit_or_path_invalid")
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("catalog_nonfinite")
            ),
        )
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("catalog_document_invalid") from None
    pending = [(value, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > 32:
            raise ValueError("catalog_depth_limit")
        if isinstance(node, dict):
            pending.extend((child, depth + 1) for child in node.values())
        elif isinstance(node, list):
            pending.extend((child, depth + 1) for child in node)
        elif isinstance(node, str):
            try:
                node.encode("utf-8")
            except UnicodeError:
                raise ValueError("catalog_unicode_invalid") from None
    if not isinstance(value, dict) or not isinstance(value.get("plugins"), list):
        raise PackageFileError("catalog_plugins_required")
    if len(value["plugins"]) > MAX_CATALOG_ENTRIES:
        raise ValueError("catalog_entries_limit")
    metadata = value.get("metadata", {})
    prefix = metadata.get("pluginRoot", "") if isinstance(metadata, dict) else ""
    if prefix:
        prefix = validate_relative_member(prefix).as_posix()
    entries = tuple(
        _entry(source_id, i, entry, prefix) for i, entry in enumerate(value["plugins"])
    )
    counts = {}
    for entry in entries:
        counts[entry.identity] = counts.get(entry.identity, 0) + 1
    entries = tuple(
        (
            entry
            if counts[entry.identity] == 1
            else entry.model_copy(
                update={
                    "identity": f"{entry.identity}:{i}",
                    "support": "unsupported",
                    "diagnostics": (Diagnostic(code="catalog_duplicate_name"),),
                }
            )
        )
        for i, entry in enumerate(entries)
    )
    return CatalogSnapshot(
        source_id=source_id,
        catalog_commit=catalog_commit,
        digest=hashlib.sha256(data).hexdigest(),
        fetched_at=fetched_at,
        entries=entries,
    )
