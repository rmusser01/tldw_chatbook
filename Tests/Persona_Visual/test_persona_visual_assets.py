"""Tests for the profile-owned Persona Visual raster boundary."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tldw_chatbook.Persona_Visual.assets import (
    ASSET_INVALID_REASON,
    PersonaVisualAssetError,
    PersonaVisualAssetMetadata,
    load_persona_visual_asset,
    validate_persona_visual_asset_set,
)
from tldw_chatbook.Persona_Visual.contracts import (
    MAX_ASSET_COUNT,
    MAX_ASSET_DIMENSION,
    MAX_ASSET_TOTAL_BYTES,
    MAX_FRAMES_PER_ANIMATION,
)


def _image_bytes(
    image_format: str = "PNG",
    *,
    size: tuple[int, int] = (2, 3),
    frames: int = 1,
) -> bytes:
    images = [Image.new("RGBA", size, (index, 2, 3, 255)) for index in range(frames)]
    output = BytesIO()
    if image_format == "JPEG":
        images[0].convert("RGB").save(output, format=image_format)
    elif frames > 1:
        images[0].save(
            output,
            format=image_format,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
    else:
        images[0].save(output, format=image_format)
    return output.getvalue()


def _metadata(
    data: bytes,
    *,
    asset_key: str = "idle",
    mime_type: str = "image/png",
    width: int = 2,
    height: int = 3,
    frame_count: int | None = 1,
    duration_ms: int | None = None,
) -> PersonaVisualAssetMetadata:
    return PersonaVisualAssetMetadata(
        asset_key=asset_key,
        role="sprite",
        mime_type=mime_type,
        byte_count=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
        frame_count=frame_count,
        duration_ms=duration_ms,
    )


@pytest.mark.parametrize(
    ("image_format", "suffix", "mime_type"),
    [
        ("PNG", ".png", "image/png"),
        ("JPEG", ".jpg", "image/jpeg"),
        ("WEBP", ".webp", "image/webp"),
        ("GIF", ".gif", "image/gif"),
    ],
)
def test_load_accepts_the_four_pinned_raster_formats(
    tmp_path: Path,
    image_format: str,
    suffix: str,
    mime_type: str,
) -> None:
    data = _image_bytes(image_format)
    path = tmp_path / f"idle{suffix}"
    path.write_bytes(data)

    asset = load_persona_visual_asset(
        tmp_path,
        storage_key=path.name,
        metadata=_metadata(data, mime_type=mime_type),
    )

    assert asset.data == data
    assert asset.metadata.mime_type == mime_type
    assert asset.selected_frame == 0
    assert not hasattr(asset, "storage_key")
    assert not hasattr(asset, "path")


@pytest.mark.parametrize(
    ("actual_format", "suffix", "declared_mime"),
    [
        ("PNG", ".jpg", "image/jpeg"),
        ("JPEG", ".png", "image/png"),
        ("WEBP", ".gif", "image/gif"),
        ("GIF", ".webp", "image/webp"),
        ("PNG", ".png", "image/gif"),
    ],
)
def test_load_rejects_extension_mime_and_decoder_disagreement(
    tmp_path: Path,
    actual_format: str,
    suffix: str,
    declared_mime: str,
) -> None:
    data = _image_bytes(actual_format)
    path = tmp_path / f"idle{suffix}"
    path.write_bytes(data)

    with pytest.raises(PersonaVisualAssetError, match=f"^{ASSET_INVALID_REASON}$"):
        load_persona_visual_asset(
            tmp_path,
            storage_key=path.name,
            metadata=_metadata(data, mime_type=declared_mime),
        )


def test_metadata_set_enforces_file_and_total_byte_budgets() -> None:
    item = _metadata(_image_bytes())
    maximum = [
        replace(item, asset_key=f"asset-{index}") for index in range(MAX_ASSET_COUNT)
    ]
    assert len(validate_persona_visual_asset_set(maximum)) == 256

    with pytest.raises(PersonaVisualAssetError):
        validate_persona_visual_asset_set(
            maximum + [replace(item, asset_key="overflow")]
        )
    with pytest.raises(PersonaVisualAssetError):
        validate_persona_visual_asset_set(
            [replace(item, byte_count=MAX_ASSET_TOTAL_BYTES), item]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asset_key", ""),
        ("role", ""),
        ("mime_type", "text/plain"),
        ("byte_count", 0),
        ("byte_count", True),
        ("sha256", "A" * 64),
        ("sha256", "0" * 63),
        ("width", 0),
        ("height", 0),
        ("width", MAX_ASSET_DIMENSION + 1),
        ("height", MAX_ASSET_DIMENSION + 1),
        ("frame_count", 0),
        ("frame_count", MAX_FRAMES_PER_ANIMATION + 1),
        ("duration_ms", 0),
    ],
)
def test_metadata_set_rejects_invalid_or_unbounded_fields(
    field: str, value: object
) -> None:
    item = replace(_metadata(_image_bytes()), **{field: value})

    with pytest.raises(PersonaVisualAssetError, match=f"^{ASSET_INVALID_REASON}$"):
        validate_persona_visual_asset_set([item])


def test_metadata_and_loaded_asset_are_deeply_immutable(tmp_path: Path) -> None:
    data = _image_bytes()
    (tmp_path / "idle.png").write_bytes(data)
    metadata = _metadata(data)
    asset = load_persona_visual_asset(
        tmp_path, storage_key="idle.png", metadata=metadata
    )

    with pytest.raises(FrozenInstanceError):
        metadata.width = 7  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        asset.data = b"changed"  # type: ignore[misc]
    assert type(asset.data) is bytes


@pytest.mark.parametrize(
    "storage_key",
    [
        "../idle.png",
        "pack/../idle.png",
        "/private/idle.png",
        "C:/idle.png",
        "C:\\idle.png",
        "//server/share/idle.png",
        "pack\\idle.png",
        "idle.png\x00suffix",
        "./idle.png",
        "pack//idle.png",
        "NUL.png",
        "pack/CON",
        "pack/COM1.gif",
    ],
)
def test_load_rejects_unsafe_relative_storage_keys(
    tmp_path: Path, storage_key: str
) -> None:
    data = _image_bytes()

    with pytest.raises(PersonaVisualAssetError, match=f"^{ASSET_INVALID_REASON}$"):
        load_persona_visual_asset(
            tmp_path, storage_key=storage_key, metadata=_metadata(data)
        )


def test_load_rejects_file_and_directory_symlink_aliases(tmp_path: Path) -> None:
    data = _image_bytes()
    outside = tmp_path.parent / "private-idle.png"
    outside.write_bytes(data)
    (tmp_path / "file-alias.png").symlink_to(outside)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "idle.png").write_bytes(data)
    (tmp_path / "dir-alias").symlink_to(real_dir, target_is_directory=True)

    for key in ("file-alias.png", "dir-alias/idle.png"):
        with pytest.raises(PersonaVisualAssetError):
            load_persona_visual_asset(
                tmp_path, storage_key=key, metadata=_metadata(data)
            )


def test_load_rejects_a_symlinked_profile_root(tmp_path: Path) -> None:
    data = _image_bytes()
    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "idle.png").write_bytes(data)
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(PersonaVisualAssetError):
        load_persona_visual_asset(
            alias_root, storage_key="idle.png", metadata=_metadata(data)
        )


def test_load_reads_only_the_declared_bounded_size(tmp_path: Path) -> None:
    data = _image_bytes()
    (tmp_path / "idle.png").write_bytes(data + b"unexpected")

    with pytest.raises(PersonaVisualAssetError):
        load_persona_visual_asset(
            tmp_path, storage_key="idle.png", metadata=_metadata(data)
        )


def test_load_rejects_missing_digest_and_decode_mismatch(tmp_path: Path) -> None:
    data = _image_bytes()
    missing = _metadata(data)
    with pytest.raises(PersonaVisualAssetError):
        load_persona_visual_asset(tmp_path, storage_key="missing.png", metadata=missing)

    (tmp_path / "idle.png").write_bytes(data)
    with pytest.raises(PersonaVisualAssetError):
        load_persona_visual_asset(
            tmp_path,
            storage_key="idle.png",
            metadata=replace(missing, sha256="0" * 64),
        )

    corrupt = b"not a png"
    (tmp_path / "idle.png").write_bytes(corrupt)
    with pytest.raises(PersonaVisualAssetError):
        load_persona_visual_asset(
            tmp_path,
            storage_key="idle.png",
            metadata=_metadata(corrupt),
        )


def test_load_rejects_declared_and_decoded_metadata_mismatch(tmp_path: Path) -> None:
    data = _image_bytes()
    (tmp_path / "idle.png").write_bytes(data)
    base = _metadata(data)

    for changed in (
        replace(base, width=1),
        replace(base, height=1),
        replace(base, frame_count=2),
    ):
        with pytest.raises(PersonaVisualAssetError):
            load_persona_visual_asset(
                tmp_path, storage_key="idle.png", metadata=changed
            )


def test_load_decodes_a_bounded_selected_gif_frame(tmp_path: Path) -> None:
    data = _image_bytes("GIF", frames=3)
    (tmp_path / "idle.gif").write_bytes(data)
    metadata = _metadata(
        data,
        mime_type="image/gif",
        frame_count=3,
        duration_ms=300,
    )

    asset = load_persona_visual_asset(
        tmp_path,
        storage_key="idle.gif",
        metadata=metadata,
        selected_frame=2,
    )

    assert asset.selected_frame == 2
    with pytest.raises(PersonaVisualAssetError):
        load_persona_visual_asset(
            tmp_path,
            storage_key="idle.gif",
            metadata=metadata,
            selected_frame=3,
        )


def test_load_rejects_more_than_the_manifest_frame_budget(tmp_path: Path) -> None:
    data = _image_bytes("GIF", frames=MAX_FRAMES_PER_ANIMATION + 1)
    (tmp_path / "idle.gif").write_bytes(data)

    with pytest.raises(PersonaVisualAssetError):
        load_persona_visual_asset(
            tmp_path,
            storage_key="idle.gif",
            metadata=_metadata(
                data,
                mime_type="image/gif",
                frame_count=MAX_FRAMES_PER_ANIMATION,
            ),
        )


def test_load_detects_final_name_to_inode_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _image_bytes()
    (tmp_path / "idle.png").write_bytes(data)
    real_stat = os.stat
    calls = 0

    def swapped_stat(*args: object, **kwargs: object) -> os.stat_result:
        nonlocal calls
        result = real_stat(*args, **kwargs)
        if kwargs.get("dir_fd") is not None and kwargs.get("follow_symlinks") is False:
            calls += 1
            values = list(result)
            values[1] += 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(os, "stat", swapped_stat)
    with pytest.raises(PersonaVisualAssetError):
        load_persona_visual_asset(
            tmp_path, storage_key="idle.png", metadata=_metadata(data)
        )
    assert calls == 1


def test_errors_never_expose_private_paths_or_exception_text(tmp_path: Path) -> None:
    private_root = tmp_path / "private-user-name" / "secret-pack"
    private_root.mkdir(parents=True)
    key = "sensitive-file-name.png"

    with pytest.raises(PersonaVisualAssetError) as caught:
        load_persona_visual_asset(
            private_root, storage_key=key, metadata=_metadata(_image_bytes())
        )

    rendered = repr(caught.value) + str(caught.value)
    assert rendered == (
        "PersonaVisualAssetError('persona_visual_asset_invalid')"
        "persona_visual_asset_invalid"
    )
    assert "private-user-name" not in rendered
    assert "sensitive-file-name" not in rendered


def test_metadata_boundary_redacts_arbitrary_sequence_failures() -> None:
    class HostileSequence(Sequence[PersonaVisualAssetMetadata]):
        def __getitem__(self, index: int) -> PersonaVisualAssetMetadata:
            raise RuntimeError("/private/user/asset.png")

        def __len__(self) -> int:
            raise RuntimeError("/private/user/asset.png")

    with pytest.raises(PersonaVisualAssetError) as caught:
        validate_persona_visual_asset_set(HostileSequence())

    assert str(caught.value) == ASSET_INVALID_REASON
