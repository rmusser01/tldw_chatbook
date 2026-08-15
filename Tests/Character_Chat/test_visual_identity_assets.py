"""Asset loading and complete-candidate validation for Visual Identity packs."""

from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path
import zipfile

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

MAX_EXPRESSION_ASSET_BYTES = 25 * 1024 * 1024
MAX_EXPRESSION_IMAGE_DIMENSION = 4096
MAX_EXPRESSION_FRAME_COUNT = 512


def _image_bytes(
    *, image_format: str = "WEBP", size: tuple[int, int] = (8, 8)
) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (12, 34, 56)).save(buffer, format=image_format)
    return buffer.getvalue()


def _animated_gif_bytes(*, frame_count: int = 2) -> bytes:
    buffer = BytesIO()
    frames = [
        Image.new(
            "RGB",
            (8, 8),
            (index % 256, (index // 256) % 256, (index * 17) % 256),
        )
        for index in range(frame_count)
    ]
    first, *remaining = frames
    first.save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=remaining,
        duration=50,
        loop=0,
        optimize=False,
    )
    return buffer.getvalue()


class _RecordingTraversable:
    def __init__(
        self,
        node,
        events: list[tuple[str, str]],
        parts: tuple[str, ...] = (),
    ) -> None:
        self._node = node
        self._events = events
        self._parts = parts

    def joinpath(self, *descendants: str):
        node = self._node
        for descendant in descendants:
            node = node.joinpath(descendant)
        return _RecordingTraversable(
            node,
            self._events,
            (*self._parts, *descendants),
        )

    def open(self, mode: str = "r", *args, **kwargs):
        self._events.append(("open", "/".join(self._parts)))
        return self._node.open(mode, *args, **kwargs)

    def read_bytes(self) -> bytes:
        self._events.append(("read_bytes", "/".join(self._parts)))
        return self._node.read_bytes()


class _BoundedStream(BytesIO):
    def __init__(self, data: bytes, read_sizes: list[int]) -> None:
        super().__init__(data)
        self._read_sizes = read_sizes

    def read(self, size: int = -1) -> bytes:
        self._read_sizes.append(size)
        if size < 0:
            raise AssertionError("unbounded read")
        return super().read(size)


class _StreamOnlyTraversable:
    def __init__(self, data: bytes, read_sizes: list[int]) -> None:
        self._data = data
        self._read_sizes = read_sizes

    def joinpath(self, *descendants: str):
        return self

    def open(self, mode: str = "r"):
        assert mode == "rb"
        return _BoundedStream(self._data, self._read_sizes)

    def read_bytes(self) -> bytes:
        raise AssertionError("read_bytes is unbounded")


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


@pytest.mark.skipif(os.name != "posix", reason="os.read is used by POSIX fd loading")
def test_user_fd_stream_read_is_bounded_to_expected_bytes_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/v1/neutral.webp"
    )
    user_data_dir = tmp_path / "profile"
    path = user_data_dir / "visual_identities/packs/example/v1/neutral.webp"
    path.parent.mkdir(parents=True)
    path.write_bytes(image_bytes)
    read_sizes: list[int] = []
    original_read = os.read

    def recording_read(fd: int, size: int) -> bytes:
        read_sizes.append(size)
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", recording_read)

    loaded = load_visual_identity_asset(
        manifest.assets[0], source_kind="manual", user_data_dir=user_data_dir
    )

    assert loaded.data == image_bytes
    assert read_sizes
    assert all(
        0 < size <= min(manifest.assets[0].bytes + 1, MAX_EXPRESSION_ASSET_BYTES + 1)
        for size in read_sizes
    )


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
    tmp_path: Path,
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
    with pytest.raises(ValueError, match="^visual_identity_path_invalid$"):
        load_visual_identity_asset(
            manifest.assets[0], source_kind="manual", user_data_dir=user_data_dir
        )


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor-bound no-follow walk is POSIX-specific",
)
def test_user_symlink_leaf_inside_root_is_rejected_by_nofollow(tmp_path: Path) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/link.webp"
    )
    user_data_dir = tmp_path / "profile"
    link = user_data_dir / "visual_identities/packs/example/link.webp"
    link.parent.mkdir(parents=True)
    inside_target = user_data_dir / "visual_identities/inside.webp"
    inside_target.write_bytes(image_bytes)
    link.symlink_to(inside_target)

    with pytest.raises(ValueError, match="^visual_identity_path_invalid$"):
        load_visual_identity_asset(
            manifest.assets[0], source_kind="manual", user_data_dir=user_data_dir
        )


def test_selected_runtime_load_reads_only_the_selected_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes, second_missing_asset=True)
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
    events: list[tuple[str, str]] = []
    traversable = _RecordingTraversable(package_root, events)
    monkeypatch.setattr(visual_identity.resources, "files", lambda package: traversable)

    loaded = load_visual_identity_asset(manifest.assets[0], source_kind="builtin")

    assert loaded.data == image_bytes
    assert events == [
        (
            "open",
            "assets/characters/samira/expressions/neutral.webp",
        )
    ]


def test_builtin_loader_supports_real_non_path_traversable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)
    archive_path = tmp_path / "package.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "assets/characters/samira/expressions/neutral.webp",
            image_bytes,
        )
    with zipfile.ZipFile(archive_path) as archive:
        package_root = zipfile.Path(archive)
        monkeypatch.setattr(
            visual_identity.resources, "files", lambda package: package_root
        )

        loaded = load_visual_identity_asset(manifest.assets[0], source_kind="builtin")

    assert loaded.data == image_bytes


def test_builtin_stream_read_is_bounded_to_expected_bytes_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)
    read_sizes: list[int] = []
    traversable = _StreamOnlyTraversable(image_bytes + b"external", read_sizes)
    monkeypatch.setattr(visual_identity.resources, "files", lambda package: traversable)

    with pytest.raises(ValueError, match="^visual_identity_asset_size_mismatch$"):
        load_visual_identity_asset(manifest.assets[0], source_kind="builtin")

    assert read_sizes
    assert all(
        0 < size <= min(manifest.assets[0].bytes + 1, MAX_EXPRESSION_ASSET_BYTES + 1)
        for size in read_sizes
    )


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


@pytest.mark.parametrize("source_kind", ["remote", "profile"])
def test_unsupported_source_kind_fails_closed_without_path_details(
    source_kind: str,
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)

    with pytest.raises(ValueError) as error:
        load_visual_identity_asset(manifest.assets[0], source_kind=source_kind)

    assert str(error.value) == "visual_identity_source_kind_unsupported"
    assert manifest.assets[0].storage_relpath not in str(error.value)
    assert "source_kind" not in manifest.assets[0].__dataclass_fields__


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor-bound no-follow walk is POSIX-specific",
)
def test_user_leaf_swap_to_external_symlink_fails_without_leaking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/v1/neutral.webp"
    )
    user_data_dir = tmp_path / "profile"
    path = user_data_dir / "visual_identities/packs/example/v1/neutral.webp"
    path.parent.mkdir(parents=True)
    path.write_bytes(image_bytes)
    outside = tmp_path / "private-secret.webp"
    outside.write_bytes(b"PRIVATE-EXTERNAL-CONTENT")
    original_read = os.read
    swapped = False

    def swap_then_read(fd: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            path.unlink()
            path.symlink_to(outside)
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", swap_then_read)

    with pytest.raises(ValueError) as error:
        load_visual_identity_asset(
            manifest.assets[0], source_kind="manual", user_data_dir=user_data_dir
        )

    assert str(error.value) == "visual_identity_path_invalid"
    assert error.value.__cause__ is None
    assert str(outside) not in str(error.value)
    assert "PRIVATE-EXTERNAL-CONTENT" not in str(error.value)


def test_user_platform_fallback_rejects_symlink_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/link.webp"
    )
    user_data_dir = tmp_path / "profile"
    link = user_data_dir / "visual_identities/packs/example/link.webp"
    link.parent.mkdir(parents=True)
    inside_target = user_data_dir / "visual_identities/inside.webp"
    inside_target.write_bytes(image_bytes)
    link.symlink_to(inside_target)
    monkeypatch.setattr(visual_identity, "_supports_secure_dir_fd", lambda: False)

    with pytest.raises(ValueError, match="^visual_identity_path_invalid$"):
        load_visual_identity_asset(
            manifest.assets[0], source_kind="manual", user_data_dir=user_data_dir
        )


def test_user_platform_fallback_rejects_symlink_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/neutral.webp"
    )
    user_data_dir = tmp_path / "profile"
    actual_dir = user_data_dir / "visual_identities/actual"
    actual_dir.mkdir(parents=True)
    (actual_dir / "neutral.webp").write_bytes(image_bytes)
    packs_dir = user_data_dir / "visual_identities/packs"
    packs_dir.mkdir()
    (packs_dir / "example").symlink_to(actual_dir, target_is_directory=True)
    monkeypatch.setattr(visual_identity, "_supports_secure_dir_fd", lambda: False)

    with pytest.raises(ValueError, match="^visual_identity_path_invalid$"):
        load_visual_identity_asset(
            manifest.assets[0], source_kind="manual", user_data_dir=user_data_dir
        )


def test_user_resolve_runtime_error_is_normalized_without_raw_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/neutral.webp"
    )

    def broken_resolve(path: Path, *, strict: bool = False):
        raise RuntimeError("/private/raw/loop")

    monkeypatch.setattr(Path, "resolve", broken_resolve)

    with pytest.raises(ValueError) as error:
        load_visual_identity_asset(
            manifest.assets[0], source_kind="manual", user_data_dir=tmp_path
        )

    assert str(error.value) == "visual_identity_path_invalid"
    assert error.value.__cause__ is None
    assert "/private/raw/loop" not in str(error.value)


def test_user_open_oserror_is_normalized_without_raw_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/neutral.webp"
    )
    root = tmp_path / "profile/visual_identities"
    root.mkdir(parents=True)

    def denied(*args, **kwargs):
        raise PermissionError("/private/raw/denied")

    monkeypatch.setattr(os, "open", denied)

    with pytest.raises(ValueError) as error:
        load_visual_identity_asset(
            manifest.assets[0], source_kind="manual", user_data_dir=tmp_path / "profile"
        )

    assert str(error.value) == "visual_identity_asset_unavailable"
    assert error.value.__cause__ is None
    assert "/private/raw/denied" not in str(error.value)


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


def test_complete_validation_rejects_unsupported_but_decodable_mime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)
    object.__setattr__(manifest.assets[0], "content_type", "image/x-webp")
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
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


def test_complete_validation_fails_fast_before_opening_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrupt = b"not-an-image"
    manifest = _manifest_for_bytes(corrupt, second_missing_asset=True)
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, corrupt)
    events: list[tuple[str, str]] = []
    traversable = _RecordingTraversable(package_root, events)
    monkeypatch.setattr(visual_identity.resources, "files", lambda package: traversable)

    with pytest.raises(ValueError, match="^visual_identity_asset_decode_invalid$"):
        validate_visual_identity_assets(manifest, source_kind="builtin")

    assert events == [
        (
            "open",
            "assets/characters/samira/expressions/neutral.webp",
        )
    ]


def test_complete_validation_rejects_actual_dimension_over_server_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes(size=(MAX_EXPRESSION_IMAGE_DIMENSION + 1, 1))
    manifest = _manifest_for_bytes(image_bytes, width=8, height=8)
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )

    with pytest.raises(ValueError, match="^visual_identity_asset_limits_exceeded$"):
        validate_visual_identity_assets(manifest, source_kind="builtin")


def test_complete_validation_rejects_actual_frame_count_before_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _animated_gif_bytes(frame_count=MAX_EXPRESSION_FRAME_COUNT + 1)
    manifest = _manifest_for_bytes(
        image_bytes,
        storage_relpath="characters/samira/expressions/neutral.gif",
        content_type="image/gif",
    )
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )
    monkeypatch.setattr(
        visual_identity,
        "_image_duration_ms",
        lambda image, frame_count: (_ for _ in ()).throw(
            AssertionError("iterated over-limit frames")
        ),
    )

    with pytest.raises(ValueError, match="^visual_identity_asset_limits_exceeded$"):
        validate_visual_identity_assets(manifest, source_kind="builtin")


@pytest.mark.parametrize("max_pixels", [40, 10])
def test_complete_validation_normalizes_pillow_decompression_bombs(
    max_pixels: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", max_pixels)

    with pytest.raises(ValueError) as error:
        validate_visual_identity_assets(manifest, source_kind="builtin")

    assert str(error.value) == "visual_identity_asset_decode_invalid"
    assert error.value.__cause__ is None
    assert "pixels" not in str(error.value)


def test_complete_validation_loads_and_rejects_corrupt_later_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _animated_gif_bytes()[:-3]
    manifest = _manifest_for_bytes(
        image_bytes,
        storage_relpath="characters/samira/expressions/neutral.gif",
        content_type="image/gif",
        is_animated=True,
        frame_count=2,
        duration_ms=100,
    )
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )

    with pytest.raises(ValueError, match="^visual_identity_asset_decode_invalid$"):
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
        source_kind="manual",
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
