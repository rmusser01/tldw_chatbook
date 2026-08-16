"""Asset loading and complete-candidate validation for Visual Identity packs."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile

import pytest
from PIL import Image

import tldw_chatbook.Character_Chat.visual_identity as visual_identity
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    extract_json_from_image_file,
    load_character_card_from_file,
)
from tldw_chatbook.Character_Chat.visual_identity import (
    SAMIRA_EXPRESSION_KEYS,
    SAMIRA_LICENSE,
    SAMIRA_MANIFEST_SCHEMA_ID,
    SAMIRA_PACK_ID,
    SAMIRA_REACTION_LABELS,
    SAMIRA_SERVER_COMMIT,
    compute_pack_content_sha256,
    load_visual_identity_asset,
    parse_visual_identity_manifest_json,
    validate_visual_identity_assets,
    validate_visual_identity_manifest,
)

MAX_EXPRESSION_ASSET_BYTES = 25 * 1024 * 1024
MAX_EXPRESSION_IMAGE_DIMENSION = 4096
MAX_EXPRESSION_FRAME_COUNT = 512

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "tldw_chatbook"
SAMIRA_ASSET_DIR = PACKAGE_ROOT / "assets/characters/samira"
SAMIRA_SOURCE_PORTRAIT_SHA256 = (
    "0b86569c3f419836a8e867b035136195a95345b2704ffc28d640849629905bed"
)
SAMIRA_RGB_SHA256 = "77452a48101e437834dedaa09ec5121d524c39ea9a13b02f87a158af80d3185f"
EXPECTED_SAMIRA_TOP_LEVEL = {
    "ASSET_LICENSE.md",
    "Samira.character.json",
    "Sammy.png",
    "expressions",
    "visual_identity_pack.json",
}
EXPECTED_VISUAL_DIRECTIONS = {
    "admiration": "softened eyes and quiet impressed respect",
    "amusement": "restrained closed-mouth smile and bright eyes",
    "anger": "controlled lowered brow and firm jaw, never rage",
    "annoyance": "slight brow pinch and restrained exasperation, milder than anger",
    "approval": "small affirming nod and composed positive regard",
    "caring": "attentive concern and gentle protective warmth",
    "confusion": "asymmetrical brow and searching focus",
    "curiosity": "alert eyes, slight head angle, investigative interest",
    "desire": "focused yearning toward an idea or objective, never flirtation",
    "disappointment": "lowered gaze and restrained letdown",
    "disapproval": "steady evaluative gaze and lightly pressed lips",
    "disgust": "subtle nose tension and aversion without caricature",
    "embarrassment": "averted gaze and contained self-consciousness",
    "excitement": "widened bright eyes and energized posture, not theatrical",
    "fear": "guarded eyes and contained alarm",
    "gratitude": "softened eyes and sincere appreciative warmth",
    "grief": "heavy eyes and controlled deep loss, no melodrama",
    "joy": "genuine warm smile and open eyes",
    "love": "deep warm regard and trust, explicitly nonromantic",
    "nervousness": "slight tension and uncertain focus, no cartoon cues",
    "neutral": "canonical quiet recognition and composed warmth",
    "optimism": "lifted focus and restrained confidence about what comes next",
    "pride": "upright composure and earned satisfaction, never smugness",
    "realization": "newly focused eyes and subtle I see it recognition",
    "relief": "released facial tension and a small exhale",
    "remorse": "lowered gaze and accountable regret",
    "sadness": "quiet sorrow and softened posture",
    "surprise": "widened eyes and subtly parted lips, controlled intensity",
    "thinking": "pensive focus and slight off-axis gaze",
    "speaking": "natural mid-sentence engagement with restrained mouth opening",
    "error": "concerned, apologetic recovery focus without a sweatdrop or symbol",
}


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(
        image_bytes, storage_relpath="packs/example/v1/neutral.webp"
    )
    user_data_dir = tmp_path / "profile"
    path = user_data_dir / "visual_identities" / "packs/example/v1/neutral.webp"
    path.parent.mkdir(parents=True)
    path.write_bytes(image_bytes)
    validations: list[tuple[Path, Path, bool, bool]] = []

    def validate_path(
        candidate: Path,
        root: Path,
        *,
        redact_paths: bool,
        allow_hidden: bool,
    ) -> Path:
        validations.append((candidate, root, redact_paths, allow_hidden))
        return candidate.resolve()

    monkeypatch.setattr(visual_identity, "validate_path", validate_path, raising=False)

    loaded = load_visual_identity_asset(
        manifest.assets[0], source_kind="manual", user_data_dir=user_data_dir
    )

    assert loaded.data == image_bytes
    assert validations == [(path, user_data_dir / "visual_identities", True, True)]


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


def test_builtin_loader_normalizes_corrupt_zip_crc_without_member_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)
    member = "assets/characters/samira/expressions/neutral.webp"
    archive_path = tmp_path / "corrupt-package.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member, image_bytes)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        data_offset = (
            info.header_offset
            + 30
            + len(info.filename.encode("utf-8"))
            + len(info.extra)
        )
    with archive_path.open("r+b") as archive_file:
        archive_file.seek(data_offset)
        original = archive_file.read(1)
        archive_file.seek(data_offset)
        archive_file.write(bytes([original[0] ^ 0xFF]))

    with zipfile.ZipFile(archive_path) as archive:
        monkeypatch.setattr(
            visual_identity.resources,
            "files",
            lambda package: zipfile.Path(archive),
        )
        with pytest.raises(ValueError) as error:
            load_visual_identity_asset(manifest.assets[0], source_kind="builtin")

    assert str(error.value) == "visual_identity_asset_unavailable"
    assert error.value.__cause__ is None
    assert member not in str(error.value)


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


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "mkfifo"),
    reason="FIFO behavior is POSIX-specific",
)
@pytest.mark.parametrize("force_fallback", [False, True])
def test_user_fifo_is_rejected_without_blocking(
    force_fallback: bool, tmp_path: Path
) -> None:
    user_data_dir = tmp_path / "profile"
    fifo = user_data_dir / "visual_identities/packs/example/neutral.webp"
    fifo.parent.mkdir(parents=True)
    os.mkfifo(fifo)
    script = "\n".join(
        [
            "import sys",
            "import tldw_chatbook.Character_Chat.visual_identity as module",
            "from tldw_chatbook.Character_Chat.visual_identity import (",
            "    VisualIdentityManifestAsset, load_visual_identity_asset)",
            f"module._supports_secure_dir_fd = lambda: {not force_fallback!r}",
            "asset = VisualIdentityManifestAsset(",
            "    expression_key='neutral', original_label='neutral',",
            "    display_label='Neutral',",
            "    storage_relpath='packs/example/neutral.webp',",
            "    content_type='image/webp', bytes=1, width=1, height=1,",
            "    sha256='0' * 64, is_animated=False, frame_count=1,",
            "    duration_ms=None)",
            "try:",
            "    load_visual_identity_asset(",
            "        asset, source_kind='manual', user_data_dir=sys.argv[1])",
            "except ValueError as error:",
            "    assert error.__cause__ is None",
            "    print(str(error))",
            "else:",
            "    raise AssertionError('FIFO was accepted')",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(user_data_dir)],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "visual_identity_path_invalid"
    assert str(fifo) not in completed.stdout


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


def test_complete_validation_rejects_actual_decoded_work_before_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, manifest.assets[0].storage_relpath, image_bytes)
    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )
    loaded = load_visual_identity_asset(manifest.assets[0], source_kind="builtin")
    monkeypatch.setattr(visual_identity, "MAX_EXPRESSION_ASSET_DECODED_PIXELS", 63)
    monkeypatch.setattr(
        visual_identity,
        "_image_duration_ms",
        lambda image, frame_count: (_ for _ in ()).throw(
            AssertionError("iterated over decoded-work limit")
        ),
    )

    with pytest.raises(ValueError, match="^visual_identity_budget_exceeded$"):
        visual_identity._validate_image_bytes(loaded)


def test_complete_validation_rechecks_cumulative_actual_decoded_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_bytes = _image_bytes()
    manifest = _manifest_for_bytes(image_bytes)
    first = manifest.assets[0]
    second = replace(
        first,
        expression_key="thinking",
        original_label="thinking",
        display_label="Thinking",
        storage_relpath="characters/samira/expressions/thinking.webp",
        width=1,
        height=1,
    )
    manifest = replace(manifest, assets=(first, second))
    package_root = tmp_path / "package"
    _write_builtin_asset(package_root, first.storage_relpath, image_bytes)
    _write_builtin_asset(package_root, second.storage_relpath, image_bytes)
    monkeypatch.setattr(
        visual_identity.resources, "files", lambda package: package_root
    )
    monkeypatch.setattr(visual_identity, "MAX_EXPRESSION_PACK_DECODED_PIXELS", 100)
    original_duration = visual_identity._image_duration_ms
    iterations: list[int] = []

    def record_iteration(image: Image.Image, frame_count: int) -> int:
        iterations.append(frame_count)
        return original_duration(image, frame_count)

    monkeypatch.setattr(visual_identity, "_image_duration_ms", record_iteration)

    with pytest.raises(ValueError, match="^visual_identity_budget_exceeded$"):
        validate_visual_identity_assets(manifest, source_kind="builtin")

    assert iterations == [1]


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


def _samira_directory_bytes() -> int:
    return sum(
        path.stat().st_size for path in SAMIRA_ASSET_DIR.rglob("*") if path.is_file()
    )


def _samira_manifest_data() -> dict[str, object]:
    return json.loads(
        (SAMIRA_ASSET_DIR / "visual_identity_pack.json").read_text(encoding="utf-8")
    )


def _walk_mapping_keys(value: object):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_mapping_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_mapping_keys(nested)


def _assert_no_private_json_metadata_keys(*values: object) -> None:
    private_keys = sorted(
        {
            key
            for value in values
            for key in _walk_mapping_keys(value)
            if "private" in key.casefold()
        }
    )
    assert not private_keys, f"private JSON metadata key: {', '.join(private_keys)}"


def _assert_no_webp_metadata(raw: bytes, image_info: dict[str, object]) -> None:
    chunk_markers = (b"EXIF", b"ICCP", b"XMP ")
    metadata_keys = {"exif", "icc_profile", "xmp", "xmp_metadata"}
    found_chunks = [marker.decode("ascii") for marker in chunk_markers if marker in raw]
    found_keys = sorted(metadata_keys.intersection(image_info))
    assert not (found_chunks or found_keys), (
        f"WebP metadata: chunks={found_chunks}, Pillow keys={found_keys}"
    )


@pytest.mark.parametrize(
    "private_metadata",
    [
        {"private": True},
        {"provenance": {"MiXeD_PrIvAtE_Flag": "legacy"}},
    ],
    ids=("top-level-private", "nested-mixed-case-private"),
)
def test_packaging_key_guard_rejects_private_manifest_metadata_even_when_parser_accepts(
    private_metadata: dict[str, object],
) -> None:
    manifest = _samira_manifest_data()
    manifest.update(private_metadata)

    parsed = parse_visual_identity_manifest_json(
        json.dumps(manifest).encode("utf-8"),
        require_samira_bundle=True,
        directory_bytes=_samira_directory_bytes(),
    )
    assert parsed.pack_id == SAMIRA_PACK_ID
    with pytest.raises(AssertionError, match="private JSON metadata key"):
        _assert_no_private_json_metadata_keys(manifest)


def test_webp_metadata_guard_rejects_exif_chunk_and_pillow_key() -> None:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(
        buffer,
        format="WEBP",
        exif=b"Exif\x00\x00legacy-metadata",
    )
    raw = buffer.getvalue()
    with Image.open(BytesIO(raw)) as image:
        image.load()
        assert b"EXIF" in raw
        assert "exif" in image.info
        with pytest.raises(AssertionError, match="WebP metadata"):
            _assert_no_webp_metadata(raw, image.info)


def test_bundled_samira_package_inventory_is_exact() -> None:
    assert {
        path.name for path in SAMIRA_ASSET_DIR.iterdir()
    } == EXPECTED_SAMIRA_TOP_LEVEL
    assert {path.name for path in (SAMIRA_ASSET_DIR / "expressions").iterdir()} == {
        f"{label}.webp" for label in SAMIRA_REACTION_LABELS
    }


def test_bundled_samira_manifest_is_strict_valid_and_content_addressed() -> None:
    raw = (SAMIRA_ASSET_DIR / "visual_identity_pack.json").read_bytes()
    directory_bytes = _samira_directory_bytes()
    manifest = parse_visual_identity_manifest_json(
        raw,
        require_samira_bundle=True,
        directory_bytes=directory_bytes,
    )
    data = _samira_manifest_data()

    assert manifest.pack_content_sha256 == compute_pack_content_sha256(data)
    assert manifest.pack_content_sha256 == (
        "5993ec841ca635d99ca83691c3ac284db1b14bff35978c72edad12df04a917c8"
    )
    assert directory_bytes <= 20 * 1024 * 1024
    assert data["normalization_contract"] == {
        "source_commit": SAMIRA_SERVER_COMMIT,
        "source_module": "app/core/Visual_Identities/expression_slots.py",
        "source_repository": "tldw_server",
    }
    assert validate_visual_identity_assets(
        manifest,
        source_kind="builtin",
        directory_bytes=directory_bytes,
    )


def test_bundled_samira_manifest_rows_match_packaged_webps() -> None:
    data = _samira_manifest_data()
    assets = data["assets"]
    assert isinstance(assets, list)
    assert [asset["original_label"] for asset in assets] == list(SAMIRA_REACTION_LABELS)
    assert {
        asset["original_label"]: asset["expression_key"] for asset in assets
    } == SAMIRA_EXPRESSION_KEYS

    total_expression_bytes = 0
    for asset in assets:
        label = asset["original_label"]
        path = SAMIRA_ASSET_DIR / "expressions" / f"{label}.webp"
        raw = path.read_bytes()
        total_expression_bytes += len(raw)
        with Image.open(BytesIO(raw)) as image:
            image.load()
            assert image.format == "WEBP"
            assert image.size == (1024, 1024)
            assert image.mode == "RGB"
            assert getattr(image, "n_frames", 1) == 1
            _assert_no_webp_metadata(raw, image.info)

        assert asset == {
            "bytes": len(raw),
            "content_type": "image/webp",
            "display_label": visual_identity.display_label_for_expression_key(
                SAMIRA_EXPRESSION_KEYS[label]
            ),
            "duration_ms": None,
            "expression_key": SAMIRA_EXPRESSION_KEYS[label],
            "frame_count": 1,
            "generation": asset["generation"],
            "height": 1024,
            "is_animated": False,
            "original_label": label,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "storage_relpath": f"characters/samira/expressions/{label}.webp",
            "width": 1024,
        }
        assert len(raw) <= 1024 * 1024
        generation = asset["generation"]
        assert generation["date"] == "2026-08-15"
        assert generation["source_portrait"] == {
            "filename": "Sammy.png",
            "sha256": SAMIRA_SOURCE_PORTRAIT_SHA256,
        }
        assert generation["visual_direction"] == EXPECTED_VISUAL_DIRECTIONS[label]
        if label == "neutral":
            assert generation["strategy"] == (
                "deterministic 1024x1024 LANCZOS derivative of the source portrait"
            )
            assert generation["tool"] == "Pillow 11.2.1"
        else:
            assert generation["strategy"] == (
                "independent edit from the original source portrait"
            )
            assert generation["tool"] == "built-in image_gen"

    assert total_expression_bytes <= 16 * 1024 * 1024


def test_bundled_samira_json_and_png_cards_are_exactly_equivalent() -> None:
    json_path = SAMIRA_ASSET_DIR / "Samira.character.json"
    png_path = SAMIRA_ASSET_DIR / "Sammy.png"
    card_text = json_path.read_text(encoding="utf-8")
    card = json.loads(card_text)
    embedded_text = extract_json_from_image_file(png_path.read_bytes())

    assert (
        card_text
        == json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    assert embedded_text == card_text
    assert json.loads(embedded_text) == card

    parsed_json = load_character_card_from_file(json_path)
    parsed_png = load_character_card_from_file(png_path)
    assert parsed_json is not None
    assert parsed_png == parsed_json

    with Image.open(png_path) as image:
        image.load()
        assert image.format == "PNG"
        assert image.size == (1254, 1254)
        assert image.mode == "RGB"
        assert set(image.info) == {"chara"}
        assert hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() == (
            SAMIRA_RGB_SHA256
        )


def test_bundled_samira_card_has_only_public_namespaced_metadata() -> None:
    card = json.loads(
        (SAMIRA_ASSET_DIR / "Samira.character.json").read_text(encoding="utf-8")
    )
    assert card["spec"] == "chara_card_v2"
    assert card["spec_version"] == "2.0"
    data = card["data"]
    assert data["name"] == "Samira “Sammy” Vadem"
    assert data["creator"] == "tldw_chatbook"
    assert data["creator_notes"] == (
        "An included, editable demonstration character for tldw_chatbook with a "
        "complete Visual Identity reaction pack."
    )
    assert set(data["tags"]) == {
        "built-in",
        "character",
        "demonstration",
        "editable",
        "editorial",
        "knowledge",
        "reaction-pack",
        "research",
        "samira",
        "synthesis",
        "visual-identity",
        "writing",
        "ambiguous-ai",
        "chatbook",
        "collaborator",
        "curator",
    }
    assert data["extensions"] == {
        "tldw/builtin_id": "samira",
        "tldw/license": SAMIRA_LICENSE,
        "tldw/nature": "ambiguous-living-index",
        "tldw/personality_mix": {
            "curious_collaborator": 25,
            "dry_wit": 15,
            "knowing_curator": 60,
        },
        "tldw/role": "curator-collaborator",
        "tldw/visual_identity_pack_id": SAMIRA_PACK_ID,
    }
    assert "image" not in data
    assert "data:image" not in json.dumps(card)
    _assert_no_private_json_metadata_keys(card)
    assert not any(
        key.casefold().startswith("vademhq/") for key in _walk_mapping_keys(card)
    )


def test_bundled_samira_assets_have_public_license_and_no_legacy_provenance() -> None:
    license_text = (SAMIRA_ASSET_DIR / "ASSET_LICENSE.md").read_text(encoding="utf-8")
    manifest = _samira_manifest_data()
    card = json.loads(
        (SAMIRA_ASSET_DIR / "Samira.character.json").read_text(encoding="utf-8")
    )
    inventory = {
        "ASSET_LICENSE.md",
        "Samira.character.json",
        "Sammy.png",
        "visual_identity_pack.json",
        *(f"expressions/{label}.webp" for label in SAMIRA_REACTION_LABELS),
    }

    assert "SPDX-License-Identifier: AGPL-3.0-or-later" in license_text
    assert "Sammy.png" in license_text
    assert SAMIRA_SOURCE_PORTRAIT_SHA256 in license_text
    assert "built-in image_gen" in license_text
    assert "2026-08-15" in license_text
    assert "independent edit from the original source portrait" in license_text
    declared_inventory = {
        line.removeprefix("- `").removesuffix("`")
        for line in license_text.splitlines()
        if line.startswith("- `") and line.endswith("`")
    }
    assert declared_inventory == inventory
    assert manifest["license"] == SAMIRA_LICENSE
    assert card["data"]["extensions"]["tldw/license"] == SAMIRA_LICENSE

    forbidden = (b"vademhq", b"vademhq/", b"private_is_canonical", b"private_source")
    for path in SAMIRA_ASSET_DIR.rglob("*"):
        if path.is_file():
            lowered = path.read_bytes().lower()
            assert not any(token in lowered for token in forbidden), path.name
    png_metadata = extract_json_from_image_file(
        (SAMIRA_ASSET_DIR / "Sammy.png").read_bytes()
    )
    assert png_metadata is not None
    embedded_card = json.loads(png_metadata)
    _assert_no_private_json_metadata_keys(card, embedded_card, manifest)
    assert not any(token in png_metadata.lower().encode() for token in forbidden)


def test_samira_package_data_patterns_are_explicit_and_bounded() -> None:
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    setuptools = pyproject["tool"]["setuptools"]
    assert setuptools["include-package-data"] is False
    assert setuptools["package-data"]["tldw_chatbook"] == [
        "assets/characters/samira/*.json",
        "assets/characters/samira/*.png",
        "assets/characters/samira/*.md",
        "assets/characters/samira/expressions/*.webp",
    ]
