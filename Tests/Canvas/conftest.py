"""Focused fixtures for Canvas runtime-profile tests."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.Canvas.profiles import (
    ProfileRecord,
    ProfileSnapshot,
    load_profile_snapshot,
)

_STATIC = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Canvas" / "static"


@pytest.fixture
def damage_canvas_package(tmp_path, monkeypatch):
    """Install one copied, deliberately damaged Canvas package closure."""

    def damage(kind: str) -> Path:
        from tldw_chatbook.Canvas import profiles, runtime_assets

        package_root = tmp_path / f"damaged-{kind}" / "Canvas"
        shutil.copytree(_STATIC, package_root / "static")
        monkeypatch.setattr(profiles, "files", lambda _package: package_root)
        monkeypatch.setattr(runtime_assets, "files", lambda _package: package_root)
        static = package_root / "static"
        if kind == "missing-v2-library":
            static.joinpath("mermaid-subset.json").unlink()
        elif kind == "tampered-v2-worker":
            worker = static / "canvas_runtime_worker_v2.js"
            worker.write_bytes(worker.read_bytes() + b"\n/* tampered */\n")
        elif kind == "missing-catalog":
            static.joinpath("profile-catalog.json").unlink()
        elif kind == "malformed-catalog":
            static.joinpath("profile-catalog.json").write_text(
                '{"schema_version": 1,', encoding="utf-8"
            )
        else:
            raise ValueError(f"unknown package damage: {kind}")
        return static

    return damage


@pytest.fixture
def profile_snapshot():
    return ProfileSnapshot(
        build_id="a" * 64,
        policy_id="b" * 64,
        profiles=(
            ProfileRecord("canvas-v1", "c" * 64, True, None, 0),
            ProfileRecord("canvas-v2-mermaid-1", "d" * 64, True, None, 77817),
        ),
        default_diagram_profile="canvas-v2-mermaid-1",
    )


@pytest.fixture
def candidate_snapshot():
    base = load_profile_snapshot()
    candidate = "canvas-v2-mermaid-1"
    assert any(row.profile_id == candidate for row in base.profiles)
    return replace(
        base,
        profiles=tuple(
            replace(row, executable=True, reason=None)
            if row.profile_id == candidate
            else row
            for row in base.profiles
        ),
        default_diagram_profile=candidate,
    )
