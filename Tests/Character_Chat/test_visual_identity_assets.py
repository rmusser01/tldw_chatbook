"""Asset loading and complete-candidate validation for Visual Identity packs."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

import tldw_chatbook.Character_Chat.visual_identity as visual_identity
from tldw_chatbook.Character_Chat.visual_identity import (
    SAMIRA_MANIFEST_SCHEMA_ID,
    SAMIRA_SERVER_COMMIT,
    compute_pack_content_sha256,
    load_visual_identity_asset,
    validate_visual_identity_assets,
    validate_visual_identity_manifest,
)


def _image_bytes(
    *, image_format: str = "WEBP", size: tuple[int, int] = (8, 8)
) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (12, 34, 56)).save(buffer, format=image_format)
    return buffer.getvalue()


def _animated_gif_bytes() -> bytes:
    buffer = BytesIO()
    first = Image.new("RGB", (8, 8), (12, 34, 56))
    second = Image.new("RGB", (8, 8), (65, 43, 21))
    first.save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=[second],
        duration=50,
        loop=0,
    )
    return buffer.getvalue()


def _manifest_for_bytes(
    image_bytes: bytes,
    *,
    storage_relpath: str = "characters/samira/expressions/neutral.webp",
    content_type: str = "image/webp",
    width: int = 8,
    height: int = 8,
    is_animated: bool = False,
    frame_count: int = 1,
    duration_ms: int | None = None,
    second_missing_asset: bool = False,
):
    assets: list[dict[str, object]] = [
        {
            "expression_key": "neutral",
            "original_label": "neutral",
            "display_label": "Neutral",
            "storage_relpath": storage_relpath,
            "content_type": content_type,
            "bytes": len(image_bytes),
            "width": width,
            "height": height,
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "is_animated": is_animated,
            "frame_count": frame_count,
            "duration_ms": duration_ms,
        }
    ]
    if second_missing_asset:
        assets.append(
            {
                "expression_key": "thinking",
                "original_label": "thinking",
                "display_label": "Thinking",
                "storage_relpath": "characters/samira/expressions/missing.webp",
                "content_type": "image/webp",
                "bytes": 1,
                "width": 8,
                "height": 8,
                "sha256": "0" * 64,
                "is_animated": False,
                "frame_count": 1,
                "duration_ms": None,
            }
        )
    data: dict[str, object] = {
        "schema_id": SAMIRA_MANIFEST_SCHEMA_ID,
        "pack_id": "user.example.pack",
        "title": "Example",
        "license": "MIT",
        "default_expression_key": "neutral",
        "source_server_commit": SAMIRA_SERVER_COMMIT,
        "assets": assets,
    }
    data["pack_content_sha256"] = compute_pack_content_sha256(data)
    return validate_visual_identity_manifest(data)


def _write_builtin_asset(package_root: Path, relpath: str, data: bytes) -> Path:
    path = package_root.joinpath("assets", *relpath.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_builtin_load_uses_importlib_package_assets_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)
    package_root = tmp_path / "installed-package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
    calls: list[str] = []

    def fake_files(package: str):
        calls.append(package)
        return package_root

    monkeypatch.setattr(visual_identity.resources, "files", fake_files)

    loaded = load_visual_identity_asset(manifest.assets[0], source_kind="builtin")

    assert calls == ["tldw_chatbook"]
    assert loaded.asset is manifest.assets[0]
    assert loaded.data == image_bytes
    assert not hasattr(loaded, "__dict__")


def test_user_load_is_confined_below_injected_visual_identities_root(
    tmp_path: Path,
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/v1/neutral.webp"
    )
    user_data_dir = tmp_path / "profile"
    path = user_data_dir / "visual_identities" / "packs/example/v1/neutral.webp"
    path.parent.mkdir(parents=True)
    path.write_bytes(image_bytes)

    loaded = load_visual_identity_asset(
        manifest.assets[0], source_kind="manual", user_data_dir=user_data_dir
    )

    assert loaded.data == image_bytes


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        ".",
        "/absolute.webp",
        "../escape.webp",
        "packs/../escape.webp",
        r"packs\escape.webp",
        "packs/escape\x00.webp",
        "packs//escape.webp",
        "packs/./escape.webp",
    ],
)
def test_unsafe_builtin_paths_are_rejected_before_resource_access(
    unsafe_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)
    asset = manifest.assets[0]
    object.__setattr__(asset, "storage_relpath", unsafe_path)
    calls: list[str] = []

    def fake_files(package: str):
        calls.append(package)
        return tmp_path

    monkeypatch.setattr(visual_identity.resources, "files", fake_files)

    with pytest.raises(ValueError, match="^visual_identity_path_invalid$"):
        load_visual_identity_asset(asset, source_kind="builtin")

    assert calls == []


def test_user_symlink_escape_is_rejected_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/link.webp"
    )
    user_data_dir = tmp_path / "profile"
    link = user_data_dir / "visual_identities/packs/example/link.webp"
    link.parent.mkdir(parents=True)
    outside = tmp_path / "outside.webp"
    outside.write_bytes(image_bytes)
    link.symlink_to(outside)
    reads: list[Path] = []
    original_read_bytes = Path.read_bytes

    def recording_read_bytes(path: Path) -> bytes:
        reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)

    with pytest.raises(ValueError, match="^visual_identity_path_invalid$"):
        load_visual_identity_asset(
            manifest.assets[0], source_kind="manual", user_data_dir=user_data_dir
        )

    assert reads == []


def test_selected_runtime_load_reads_only_the_selected_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes, second_missing_asset=True)
    package_root = tmp_path / "package"
    selected_path = _write_builtin_asset(
        package_root, manifest.assets[0].storage_relpath, image_bytes
    )
    reads: list[Path] = []
    original_read_bytes = Path.read_bytes

    def recording_read_bytes(path: Path) -> bytes:
        reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )
    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)

    loaded = load_visual_identity_asset(manifest.assets[0], source_kind="builtin")

    assert loaded.data == image_bytes
    assert reads == [selected_path]


@pytest.mark.parametrize(
    ("field", "value", "category"),
    [
        ("bytes", 1, "visual_identity_asset_size_mismatch"),
        ("sha256", "0" * 64, "visual_identity_asset_sha256_mismatch"),
    ],
)
def test_selected_runtime_load_verifies_size_and_hash(
    field: str,
    value: object,
    category: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)
    object.__setattr__(manifest.assets[0], field, value)
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )

    with pytest.raises(ValueError, match=f"^{category}$"):
        load_visual_identity_asset(manifest.assets[0], source_kind="builtin")


def test_unsupported_source_kind_fails_closed_without_path_details() -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)

    with pytest.raises(ValueError) as error:
        load_visual_identity_asset(manifest.assets[0], source_kind="remote")

    assert str(error.value) == "visual_identity_source_kind_unsupported"
    assert manifest.assets[0].storage_relpath not in str(error.value)
    assert "source_kind" not in manifest.assets[0].__dataclass_fields__


def test_complete_validation_checks_webp_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    png_bytes = _image_bytes(image_format="PNG")
    manifest = _manifest_for_bytes(png_bytes)
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, png_bytes)
    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )

    with pytest.raises(ValueError, match="^visual_identity_asset_format_mismatch$"):
        validate_visual_identity_assets(manifest, source_kind="builtin")


def test_complete_validation_checks_exact_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes(size=(7, 8))
    manifest = _manifest_for_bytes(image_bytes, width=8, height=8)
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )

    with pytest.raises(ValueError, match="^visual_identity_asset_dimensions_mismatch$"):
        validate_visual_identity_assets(manifest, source_kind="builtin")


def test_complete_validation_checks_declared_frame_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, is_animated=True, frame_count=2, duration_ms=100
    )
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )

    with pytest.raises(ValueError, match="^visual_identity_asset_frame_mismatch$"):
        validate_visual_identity_assets(manifest, source_kind="builtin")


def test_complete_validation_returns_every_verified_candidate(
    tmp_path: Path,
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/v1/neutral.webp"
    )
    user_data_dir = tmp_path / "profile"
    path = user_data_dir / "visual_identities/packs/example/v1/neutral.webp"
    path.parent.mkdir(parents=True)
    path.write_bytes(image_bytes)

    loaded = validate_visual_identity_assets(
        manifest,
        source_kind="profile",
        user_data_dir=user_data_dir,
        directory_bytes=len(image_bytes),
    )

    assert len(loaded) == 1
    assert loaded[0].data == image_bytes


def test_complete_validation_accepts_consistent_animation_fields(
    tmp_path: Path,
) -> None:
    image_bytes = _animated_gif_bytes()
    manifest = _manifest_for_bytes(
        image_bytes,
        storage_relpath="packs/example/v1/animated.gif",
        content_type="image/gif",
        is_animated=True,
        frame_count=2,
        duration_ms=100,
    )
    user_data_dir = tmp_path / "profile"
    path = user_data_dir / "visual_identities/packs/example/v1/animated.gif"
    path.parent.mkdir(parents=True)
    path.write_bytes(image_bytes)

    loaded = validate_visual_identity_assets(
        manifest,
        source_kind="manual",
        user_data_dir=user_data_dir,
    )

    assert loaded[0].data == image_bytes
