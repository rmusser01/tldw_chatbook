"""Setuptools commands for explicit, qualified native backup-helper wheels."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Any

from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


_BUILD_HELPER_SPEC = importlib.util.spec_from_file_location(
    "_tldw_backup_age_build_helper", Path(__file__).with_name("build_helper.py")
)
if _BUILD_HELPER_SPEC is None or _BUILD_HELPER_SPEC.loader is None:
    raise RuntimeError("backup helper build module is unavailable")
_BUILD_HELPER_MODULE = importlib.util.module_from_spec(_BUILD_HELPER_SPEC)
_BUILD_HELPER_SPEC.loader.exec_module(_BUILD_HELPER_MODULE)
TARGETS = _BUILD_HELPER_MODULE.TARGETS
build_helper = _BUILD_HELPER_MODULE.build_helper

_TARGET_ENV = "TLDW_BACKUP_HELPER_TARGET"
_PLATFORM_TAGS = {
    ("darwin", "arm64"): "macosx_12_0_arm64",
    ("darwin", "amd64"): "macosx_12_0_x86_64",
    ("linux", "amd64"): "manylinux_2_28_x86_64",
    ("linux", "arm64"): "manylinux_2_28_aarch64",
    ("windows", "amd64"): "win_amd64",
}


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _qualification() -> dict[str, Any]:
    return json.loads(
        (_source_root() / "Packaging/backup_age/qualification.json").read_text(
            encoding="utf-8"
        )
    )


def _selected_target() -> tuple[str, str] | None:
    raw = os.environ.get(_TARGET_ENV)
    if raw is None:
        return None
    parts = raw.split("/")
    if len(parts) != 2 or tuple(parts) not in TARGETS:
        raise ValueError(f"{_TARGET_ENV} must name a candidate GOOS/GOARCH tuple")
    target = (parts[0], parts[1])
    record = next(
        (
            item
            for item in _qualification()["targets"]
            if (item["os"], item["arch"]) == target
        ),
        None,
    )
    if record is None or record["status"] != "qualified":
        raise ValueError(f"backup helper target is not qualified: {raw}")
    return target


def _qualified_record(target: tuple[str, str]) -> dict[str, Any]:
    record = next(
        item
        for item in _qualification()["targets"]
        if (item["os"], item["arch"]) == target
    )
    if record["wheel_platform_tag"] != _PLATFORM_TAGS[target]:
        raise ValueError("qualified wheel platform tag differs from build policy")
    return record


class BackupHelperBuildPy(build_py):
    """Inject a helper only for an explicitly selected qualified native wheel."""

    def run(self) -> None:
        super().run()
        target = _selected_target()
        package_root = Path(self.build_lib) / "tldw_chatbook/Backup_Recovery"
        resource_root = package_root / "_age"
        shutil.rmtree(resource_root, ignore_errors=True)
        if target is None:
            return

        goos, goarch = target
        record = _qualified_record(target)
        resource_root.mkdir(mode=0o755)
        binary_name = "backup-age.exe" if goos == "windows" else "backup-age"
        binary = build_helper(goos, goarch, resource_root / binary_name)
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise RuntimeError("built helper digest differs from qualification record")

        for name in ("LICENSE.age.txt", "THIRD_PARTY_NOTICES.txt"):
            shutil.copy2(_source_root() / "Packaging/backup_age" / name, resource_root)
        source_manifest = json.loads(
            (
                _source_root() / "tldw_chatbook/Backup_Recovery/helper_manifest.json"
            ).read_text(encoding="utf-8")
        )
        qualified_entry = {
            "status": "qualified",
            "protocol": 1,
            "helper_version": "1",
            "age_version": "v1.3.2",
            "os": goos,
            "arch": goarch,
            "resource": f"_age/{binary_name}",
            "sha256": digest,
        }
        source_manifest["helpers"] = [
            qualified_entry if (item["os"], item["arch"]) == target else item
            for item in source_manifest["helpers"]
        ]
        (package_root / "helper_manifest.json").write_text(
            json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
        )


class BackupHelperBdistWheel(bdist_wheel):
    """Tag selected helper wheels for their qualified native platform."""

    def finalize_options(self) -> None:
        target = _selected_target()
        if target is not None:
            self.distribution.has_ext_modules = lambda: True
        super().finalize_options()
        if target is not None:
            self.root_is_pure = False
            self.plat_name = _qualified_record(target)["wheel_platform_tag"]

    def get_tag(self) -> tuple[str, str, str]:
        target = _selected_target()
        if target is None:
            return super().get_tag()
        return "py3", "none", _qualified_record(target)["wheel_platform_tag"]
