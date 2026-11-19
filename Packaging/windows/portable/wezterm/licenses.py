"""Preserve license declarations and notices from the exact available sources."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tomllib
from pathlib import Path


def preserve_licenses(source: Path, cargo_home: Path, destination: Path) -> Path:
    """Copy notices with a source index; do not interpret license obligations."""
    destination.mkdir(parents=True, exist_ok=True)
    roots = [("upstream", source)]
    for registry in (cargo_home / "registry/src").glob("*"):
        roots.extend(
            (f"registry/{package.name}", package) for package in registry.iterdir()
        )
    packages, notices = [], []
    for label, root in roots:
        for directory, children, files in os.walk(root):
            children[:] = [name for name in children if name not in {"target", ".git"}]
            for name in files:
                path = Path(directory) / name
                relative = f"{label}/{path.relative_to(root).as_posix()}"
                if name == "Cargo.toml":
                    manifest = tomllib.loads(path.read_text(encoding="utf-8"))
                    package = manifest.get("package", {})
                    if package:
                        packages.append(
                            {
                                "manifest": relative,
                                "name": package.get("name"),
                                "version": package.get("version"),
                                "license": package.get("license"),
                                "license_file": package.get("license-file"),
                            }
                        )
                if not (
                    name.upper().startswith(
                        ("LICENSE", "COPYING", "COPYRIGHT", "NOTICE", "OFL")
                    )
                    and path.suffix.lower() in {"", ".txt", ".md", ".rst"}
                ):
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                output = destination / f"{digest}.txt"
                if not output.exists():
                    shutil.copyfile(path, output)
                notices.append(
                    {"source": relative, "file": output.name, "sha256": digest}
                )
    index = destination / "index.json"
    index.write_text(
        json.dumps({"packages": packages, "notices": notices}, indent=2) + "\n",
        encoding="utf-8",
    )
    return index
