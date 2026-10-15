"""Explicit foreign-root previews; no profile scanning or imported grants."""

import hashlib
import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from tldw_chatbook.Utils.path_validation import validate_canonical_directory

from .catalogs import (
    CATALOG_PATHS,
    MAX_CATALOG_BYTES,
    catalog_local_path,
    parse_catalog,
)
from .models import AcquiredSnapshot, FrozenModel
from .package_files import (
    _directory_fd,
    _member_fd,
    canonical_json,
    capture_package,
    validate_relative_member,
)

EXCLUSIONS = (
    "credentials",
    "approvals",
    "trust",
    "enablement",
    "runtime_cache_overlays",
)


class ImportCandidate(FrozenModel):
    kind: Literal["catalog", "package", "source"]
    path: str
    digest: str
    catalog_path: str | None = None
    package_paths: tuple[str, ...] = ()


class ImportFailure(FrozenModel):
    path: Literal[
        ".",
        ".agents/plugins/marketplace.json",
        ".cursor-plugin/marketplace.json",
        ".claude-plugin/marketplace.json",
    ]
    code: Literal[
        "clean_source_required", "import_item_unavailable", "import_copy_failed"
    ]


class ImportPreview(FrozenModel):
    app: Literal["cursor", "codex"]
    selected_root: str
    root_identity: tuple[int, int]
    candidates: tuple[ImportCandidate, ...]
    exclusions: tuple[str, ...]
    reviewed_digest: str
    failures: tuple[ImportFailure, ...] = Field(default=(), max_length=3)


def read_selected_file(
    root: Path, path: str, *, maximum: int = MAX_CATALOG_BYTES
) -> bytes:
    """Read one bounded no-follow file beneath an explicitly selected directory."""
    with (
        _directory_fd(root) as root_fd,
        _member_fd(root_fd, validate_relative_member(path)) as fd,
    ):
        before = os.fstat(fd)
        import stat

        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError("import_file_invalid")
        chunks = []
        size = 0
        while chunk := os.read(fd, min(65536, maximum + 1 - size)):
            size += len(chunk)
            if size > maximum:
                raise ValueError("import_file_limit")
            chunks.append(chunk)
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("import_file_changed")
        return b"".join(chunks)


def _package_digest(root):
    capture = capture_package(root)
    if capture.errors:
        raise ValueError("import_package_invalid")
    # Profile files are not package declarations; request a clean selected root.
    if any(
        Path(name).name
        in {"auth.json", "credentials.json", "state.vscdb", "config.toml", ".env"}
        for name in capture.files
    ):
        raise ValueError("import_profile_state")
    return capture.digest


def inspect_app_import(app: str, selected_root: Path) -> ImportPreview:
    """Preview only known catalogs or one explicitly selected complete package."""
    if app not in {"cursor", "codex"}:
        raise ValueError("import_app_unsupported")
    root = validate_canonical_directory(selected_root)
    info = root.stat()
    candidates = []
    failures = []
    excluded = list(EXCLUSIONS)
    paths = (
        (".cursor-plugin/marketplace.json",)
        if app == "cursor"
        else (CATALOG_PATHS[0], CATALOG_PATHS[2])
    )
    for path in paths:
        if not os.path.lexists(root / path):
            continue
        try:
            data = read_selected_file(root, path)
            snapshot = parse_catalog(data, source_id="import", catalog_path=path)
            package_digests = {}
            for entry in snapshot.entries:
                if entry.support == "supported" and entry.local_path is not None:
                    package = catalog_local_path(root, entry.local_path)
                    package_digests[entry.local_path] = _package_digest(package)
            digest = hashlib.sha256(
                canonical_json(
                    {"catalog": snapshot.digest, "packages": package_digests}
                ).encode()
            ).hexdigest()
            candidates.append(
                ImportCandidate(
                    kind="catalog",
                    path=path,
                    digest=digest,
                    catalog_path=path,
                    package_paths=tuple(sorted(package_digests)),
                )
            )
        except (OSError, ValueError):
            excluded.append("clean_source_required")
            failures.append(ImportFailure(path=path, code="clean_source_required"))
    manifest = (
        ".cursor-plugin/plugin.json" if app == "cursor" else ".codex-plugin/plugin.json"
    )
    if (root / manifest).is_file() or (
        app == "codex" and (root / "plugin.json").is_file()
    ):
        try:
            candidates.append(
                ImportCandidate(kind="package", path=".", digest=_package_digest(root))
            )
        except (OSError, ValueError):
            excluded.append("clean_source_required")
            failures.append(ImportFailure(path=".", code="clean_source_required"))
    if not candidates:
        excluded.append("clean_source_required")
    content = {
        "app": app,
        "selected_root": str(root),
        "root_identity": (info.st_dev, info.st_ino),
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "exclusions": sorted(set(excluded)),
        "failures": [item.model_dump(mode="json") for item in failures],
    }
    return ImportPreview(
        **content,
        reviewed_digest=hashlib.sha256(canonical_json(content).encode()).hexdigest(),
    )


class ImportResult(FrozenModel):
    source_ids: tuple[str, ...] = ()
    packages: tuple[AcquiredSnapshot, ...] = ()
    exclusions: tuple[str, ...] = ()
    failures: tuple[ImportFailure, ...] = Field(default=(), max_length=3)
