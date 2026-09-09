"""Behavioral tests for immutable Canvas runtime-profile admission."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from tldw_chatbook.Canvas.profiles import (
    ProfileRecord,
    load_profile_snapshot,
    resolve_profile,
    runtime_assets_for,
    runtime_snapshot_id,
)

STATIC = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Canvas" / "static"


def test_removing_diagrams_retains_parent_profile(profile_snapshot):
    result = resolve_profile(
        profile_snapshot,
        operation="update",
        parent_profile="canvas-v2-mermaid-1",
        has_diagrams=False,
    )

    assert result.profile_id == "canvas-v2-mermaid-1"
    assert result.executable


@pytest.mark.parametrize(
    ("operation", "parent", "has_diagrams", "expected"),
    [
        ("create", None, False, "canvas-v1"),
        ("create", None, True, "canvas-v2-mermaid-1"),
        ("update", "canvas-v1", False, "canvas-v1"),
        ("update", "canvas-v1", True, "canvas-v2-mermaid-1"),
        ("update", "canvas-v2-mermaid-1", False, "canvas-v2-mermaid-1"),
        ("update", "canvas-v2-mermaid-1", True, "canvas-v2-mermaid-1"),
        ("rename", "canvas-v1", True, "canvas-v1"),
        ("load", "canvas-v2-mermaid-1", False, "canvas-v2-mermaid-1"),
    ],
)
def test_profile_selection_uses_exact_parent_or_admitted_default(
    profile_snapshot, operation, parent, has_diagrams, expected
):
    result = resolve_profile(
        profile_snapshot,
        operation=operation,
        parent_profile=parent,
        has_diagrams=has_diagrams,
    )

    assert result.profile_id == expected
    assert result.executable is True
    assert result.reason is None


@pytest.mark.parametrize("operation", ["update", "rename", "load"])
@pytest.mark.parametrize(
    ("profile_id", "reason"),
    [
        ("canvas-retired", "profile-retired"),
        ("canvas-revoked", "profile-revoked"),
        ("canvas-unknown", "profile-unavailable"),
    ],
)
def test_identity_preserving_operations_keep_unavailable_profile_source_only(
    profile_snapshot, operation, profile_id, reason
):
    records = profile_snapshot.profiles
    if profile_id != "canvas-unknown":
        records += (ProfileRecord(profile_id, "e" * 64, False, reason, 0),)
    snapshot = replace(profile_snapshot, profiles=records)

    result = resolve_profile(
        snapshot,
        operation=operation,
        parent_profile=profile_id,
        has_diagrams=False,
    )

    assert result.profile_id == profile_id
    assert result.executable is False
    assert result.reason == reason


def test_missing_admitted_diagram_default_does_not_fall_back_to_v1(
    profile_snapshot,
):
    snapshot = replace(profile_snapshot, default_diagram_profile=None)

    result = resolve_profile(
        snapshot, operation="create", parent_profile=None, has_diagrams=True
    )

    assert result.profile_id == "profile-unavailable"
    assert result.executable is False
    assert result.reason == "profile-unavailable"


@pytest.mark.parametrize(
    ("operation", "parent"),
    [("update", None), ("rename", None), ("load", None), ("delete", "canvas-v1")],
)
def test_invalid_operation_or_missing_required_parent_is_rejected(
    profile_snapshot, operation, parent
):
    with pytest.raises(ValueError):
        resolve_profile(
            profile_snapshot,
            operation=operation,
            parent_profile=parent,
            has_diagrams=False,
        )


def test_candidate_profile_identity_is_validated_before_resolution(profile_snapshot):
    with pytest.raises(ValueError):
        resolve_profile(
            profile_snapshot,
            operation="load",
            parent_profile="../canvas-v1",
            has_diagrams=False,
        )


def test_snapshot_identity_is_canonical_and_covers_execution_policy(profile_snapshot):
    reordered = replace(
        profile_snapshot, profiles=tuple(reversed(profile_snapshot.profiles))
    )
    expected = hashlib.sha256(
        json.dumps(
            {
                "build_id": "a" * 64,
                "default_diagram_profile": "canvas-v2-mermaid-1",
                "policy_id": "b" * 64,
                "profiles": [
                    {
                        "executable": True,
                        "library_bytes": 0,
                        "manifest_sha256": "c" * 64,
                        "profile_id": "canvas-v1",
                        "reason": None,
                    },
                    {
                        "executable": True,
                        "library_bytes": 77817,
                        "manifest_sha256": "d" * 64,
                        "profile_id": "canvas-v2-mermaid-1",
                        "reason": None,
                    },
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert runtime_snapshot_id(profile_snapshot) == expected
    assert runtime_snapshot_id(reordered) == expected
    assert (
        runtime_snapshot_id(replace(profile_snapshot, policy_id="f" * 64)) != expected
    )


def test_profile_records_and_snapshots_are_immutable(profile_snapshot):
    with pytest.raises(FrozenInstanceError):
        profile_snapshot.default_diagram_profile = None
    with pytest.raises(FrozenInstanceError):
        profile_snapshot.profiles[0].executable = False


def _isolated_static(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from tldw_chatbook.Canvas import profiles, runtime_assets

    package_root = tmp_path / "Canvas"
    shutil.copytree(STATIC, package_root / "static")
    monkeypatch.setattr(profiles, "files", lambda _package: package_root)
    monkeypatch.setattr(runtime_assets, "files", lambda _package: package_root)
    return package_root / "static"


def test_packaged_snapshot_admits_exact_qualified_mermaid_profile():
    snapshot = load_profile_snapshot()

    assert (
        snapshot.build_id
        == "5cdfdfcf09ed257bce900fa94472cc9ff79eb58506ec86e0a55ce0cd2d7e95d8"
    )
    assert (
        snapshot.policy_id
        == "cd4f0cdd756732e686b05031ce12c6bd086473cc72ff2f9d58340d8528b40f15"
    )
    assert snapshot.default_diagram_profile == "canvas-v2-mermaid-1"
    assert [(record.profile_id, record.executable) for record in snapshot.profiles] == [
        ("canvas-v1", True),
        ("canvas-v2-mermaid-1", True),
    ]
    assert snapshot.profiles[0].library_bytes == 0
    assert snapshot.profiles[1].reason is None


def test_owned_snapshot_does_not_reread_mutated_packaged_inputs(tmp_path, monkeypatch):
    static = _isolated_static(tmp_path, monkeypatch)
    snapshot = load_profile_snapshot()
    identity = runtime_snapshot_id(snapshot)

    static.joinpath("runtime-manifest.json").write_bytes(b"tampered")
    static.joinpath("profile-catalog.json").write_bytes(b"tampered")

    assert runtime_snapshot_id(snapshot) == identity
    assert resolve_profile(
        snapshot, operation="load", parent_profile="canvas-v1", has_diagrams=False
    ).executable
    owned = runtime_assets_for(snapshot, "canvas-v1")
    assert owned is not None
    assert hashlib.sha256(owned.manifest_bytes).hexdigest() == (
        snapshot.profiles[0].manifest_sha256
    )


def test_snapshot_owns_exact_manifest_and_library_bytes_for_each_profile(
    tmp_path, monkeypatch
):
    static = _isolated_static(tmp_path, monkeypatch)
    catalog_path = static / "profile-catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        static.joinpath("runtime-manifest.json").read_text(encoding="utf-8")
    )
    manifest["runtime_profile"] = "canvas-test-library"
    second_manifest = static / "canvas-test-library-manifest.json"
    second_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    library_bytes = b"verified-library-fixture"
    library = static / "canvas-test-library.js"
    library.write_bytes(library_bytes)
    catalog["profiles"].append(
        {
            "executable": False,
            "library": {
                "bytes": len(library_bytes),
                "files": {
                    library.name: {
                        "bytes": len(library_bytes),
                        "sha256": hashlib.sha256(library_bytes).hexdigest(),
                    }
                },
            },
            "manifest": second_manifest.name,
            "manifest_sha256": hashlib.sha256(second_manifest.read_bytes()).hexdigest(),
            "profile_id": "canvas-test-library",
            "reason": "profile-unqualified",
        }
    )
    catalog["build_id"] = _catalog_build_id(catalog)
    catalog["policy_id"] = _catalog_policy_id(catalog)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    snapshot = load_profile_snapshot()
    owned = runtime_assets_for(snapshot, "canvas-test-library")

    assert owned is not None
    assert owned.manifest_name == second_manifest.name
    assert owned.manifest["runtime_profile"] == "canvas-test-library"
    assert owned.library_files[library.name] == library_bytes
    assert owned.javascript == static.joinpath("quickjs-runtime.js").read_bytes()
    assert (
        owned.worker_javascript
        == static.joinpath("canvas_runtime_worker.js").read_bytes()
    )
    assert (
        owned.renderer_javascript == static.joinpath("canvas_renderer.js").read_bytes()
    )
    with pytest.raises(TypeError):
        owned.library_files[library.name] = b"replacement"

    second_manifest.write_bytes(b"changed after snapshot")
    library.write_bytes(b"changed after snapshot")
    static.joinpath("quickjs-runtime.js").write_bytes(b"changed after snapshot")
    assert owned.manifest["runtime_profile"] == "canvas-test-library"
    assert owned.library_files[library.name] == library_bytes
    assert (
        hashlib.sha256(owned.javascript).hexdigest()
        == owned.manifest["outputs"]["quickjs-runtime.js"]["sha256"]
    )


def _catalog_build_id(catalog: dict) -> str:
    projection = [
        {
            "library": entry["library"],
            "manifest_sha256": entry["manifest_sha256"],
            "profile_id": entry["profile_id"],
        }
        for entry in catalog["profiles"]
    ]
    return hashlib.sha256(
        json.dumps(
            sorted(projection, key=lambda item: item["profile_id"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _catalog_policy_id(catalog: dict) -> str:
    projection = {
        "default_diagram_profile": catalog["default_diagram_profile"],
        "profiles": sorted(
            (
                {
                    "executable": entry["executable"],
                    "profile_id": entry["profile_id"],
                    "reason": entry["reason"],
                }
                for entry in catalog["profiles"]
            ),
            key=lambda item: item["profile_id"],
        ),
    }
    return hashlib.sha256(
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        "byte-tamper",
        "missing-manifest",
        "duplicate-key",
        "duplicate-id",
        "unknown-field",
        "id-reuse",
        "missing-contract",
        "bad-library-inventory",
    ],
)
def test_packaged_snapshot_rejects_untrusted_or_ambiguous_inputs(
    tmp_path, monkeypatch, mutation
):
    static = _isolated_static(tmp_path, monkeypatch)
    catalog_path = static / "profile-catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    manifest_path = static / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if mutation == "byte-tamper":
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    elif mutation == "missing-manifest":
        manifest_path.unlink()
    elif mutation == "duplicate-key":
        catalog_path.write_text(
            catalog_path.read_text(encoding="utf-8").replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
            ),
            encoding="utf-8",
        )
    elif mutation == "duplicate-id":
        catalog["profiles"].append(catalog["profiles"][0])
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    elif mutation == "unknown-field":
        catalog["profiles"][0]["surprise"] = True
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    elif mutation == "id-reuse":
        catalog["profiles"][0]["profile_id"] = "canvas-v2"
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    elif mutation == "missing-contract":
        del manifest["profile_contract"]["unicode"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        catalog["profiles"][0]["manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    else:
        catalog["profiles"][0]["library"] = {
            "bytes": 1,
            "files": {"missing.js": {"bytes": 1, "sha256": "f" * 64}},
        }
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(
        ValueError, match="Canvas runtime profile catalog is unavailable"
    ):
        load_profile_snapshot()


def test_oversized_catalog_is_read_only_to_runtime_manifest_ceiling(
    tmp_path, monkeypatch
):
    static = _isolated_static(tmp_path, monkeypatch)
    from tldw_chatbook.Canvas import profiles

    static.joinpath("profile-catalog.json").write_bytes(
        b"{" + b" " * profiles.PROFILE_CATALOG_BYTES + b"}"
    )

    with pytest.raises(
        ValueError, match="Canvas runtime profile catalog is unavailable"
    ):
        load_profile_snapshot()


@pytest.mark.parametrize(
    "package_damage",
    [
        "missing-v2-library",
        "tampered-v2-worker",
        "missing-catalog",
        "malformed-catalog",
    ],
)
def test_strict_loader_rejects_damaged_v2_closure_or_catalog(
    damage_canvas_package, package_damage
) -> None:
    """Application recovery must not weaken exact packaged-byte verification."""
    damage_canvas_package(package_damage)

    with pytest.raises(
        ValueError, match="Canvas runtime profile catalog is unavailable"
    ):
        load_profile_snapshot()
