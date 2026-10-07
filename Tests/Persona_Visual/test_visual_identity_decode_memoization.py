"""Task 16 (F13): visual-identity decode memoization and portrait version tokens.

Covers two cost centers:

1. ``visual_identity._inspect_image_bytes`` decode work is memoized per
   (source identity, stat signature), and the deliberate corruption-check
   decode shares its composited frames with ``prepare_expression`` so one
   resolution plus one preparation performs exactly one full decode.
2. ``persona_visual_identity`` revalidation reuses the linked-portrait
   content hash while the character card stat (row version) is unchanged.
"""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tldw_chatbook.Character_Chat import (
    persona_visual_identity,
    visual_identity,
)
from tldw_chatbook.Chat import character_expression_playback as playback
from tldw_chatbook.Character_Chat.persona_visual_identity import (
    capture_local_persona_visual_identity,
    local_persona_visual_identity_is_current,
    resolve_persona_visual_identity,
)
from tldw_chatbook.Character_Chat.visual_identity import resolve_visual_identity
from tldw_chatbook.Chat.character_expression_playback import prepare_expression
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (16, 12)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def _gif_bytes(
    colors: tuple[str, ...] = ("red", "blue", "green"),
    durations: tuple[int, ...] = (100, 200, 300),
    frame_size: tuple[int, int] = (24, 16),
) -> bytes:
    frames = [Image.new("RGBA", frame_size, color) for color in colors]
    out = BytesIO()
    frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=list(durations),
        loop=0,
        lossless=True,
    )
    return out.getvalue()


def _corrupt_bytes() -> bytes:
    """Container header intact, interior payload corrupt (fails at decode)."""

    data = bytearray(_png_bytes((10, 20, 30)))
    for offset in range(64, min(len(data), 160)):
        data[offset] ^= 0xFF
    return bytes(data)


def _asset_record(
    relpath: str,
    data: bytes,
    *,
    expression_key: str,
    original_label: str,
    content_type: str,
    is_animated: bool,
    frame_count: int,
    duration_ms: int | None,
) -> dict[str, Any]:
    with Image.open(BytesIO(data)) as image:
        width, height = image.size
    return {
        "expression_key": expression_key,
        "original_expression_key": original_label,
        "display_label": original_label.title(),
        "source_filename": relpath.rsplit("/", 1)[-1],
        "storage_relpath": relpath,
        "content_type": content_type,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": width,
        "height": height,
        "source_context": {"fixture": True},
        "is_animated": is_animated,
        "frame_count": frame_count,
        "duration_ms": duration_ms,
    }


def _write_user_asset(user_root: Path, relpath: str, data: bytes) -> None:
    path = user_root / "visual_identities" / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture(autouse=True)
def _reset_decode_caches() -> None:
    visual_identity._reset_inspection_memo()
    playback._reset_shared_decode_state()
    persona_visual_identity._reset_portrait_memo()
    yield
    visual_identity._reset_inspection_memo()
    playback._reset_shared_decode_state()
    persona_visual_identity._reset_portrait_memo()


class _DecodeSpy:
    """Count decode passes at the two module seams that drive PIL frames."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.inspect_passes = 0
        self.playback_frames = 0
        inspect_original = visual_identity._image_duration_ms
        playback_original = playback._decode_frame

        def inspect_wrapper(
            image: Any, frame_count: int, retention: Any = None
        ) -> int:
            self.inspect_passes += 1
            return inspect_original(image, frame_count, retention=retention)

        def playback_wrapper(image: Any, index: int) -> None:
            self.playback_frames += 1
            return playback_original(image, index)

        monkeypatch.setattr(visual_identity, "_image_duration_ms", inspect_wrapper)
        monkeypatch.setattr(playback, "_decode_frame", playback_wrapper)


@pytest.fixture
def pack_environment(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "resolve.db", "resolve-test")
    user_root = tmp_path / "profile"
    actor_id = db.add_character_card({"name": "Memo actor"})
    assert actor_id is not None
    neutral = _png_bytes((20, 30, 40))
    anim = _gif_bytes()
    neutral_relpath = "characters/memo/expressions/neutral.png"
    anim_relpath = "characters/memo/expressions/anim.gif"
    _write_user_asset(user_root, neutral_relpath, neutral)
    _write_user_asset(user_root, anim_relpath, anim)
    VisualIdentityRepository(db).activate_pack(
        pack={
            "title": "Memo pack",
            "description": "",
            "default_expression_key": "neutral",
            "source_kind": "manual",
            "source_context": {"source_id": "fixture.memo"},
        },
        manifest={"fixture": "memo"},
        assets=[
            _asset_record(
                neutral_relpath,
                neutral,
                expression_key="neutral",
                original_label="neutral",
                content_type="image/png",
                is_animated=False,
                frame_count=1,
                duration_ms=None,
            ),
            _asset_record(
                anim_relpath,
                anim,
                expression_key="custom:anim",
                original_label="anim",
                content_type="image/gif",
                is_animated=True,
                frame_count=3,
                duration_ms=600,
            ),
        ],
        actor_kind="character",
        actor_id=actor_id,
    )
    try:
        yield {
            "db": db,
            "actor_id": actor_id,
            "user_root": user_root,
            "anim_path": user_root / "visual_identities" / anim_relpath,
            "anim_bytes": anim,
        }
    finally:
        db.close_connection()


def _resolve_anim(env: dict[str, Any]):
    return resolve_visual_identity(
        env["db"],
        actor_kind="character",
        actor_id=env["actor_id"],
        requested_state="idle",
        manual_expression_key="custom:anim",
        user_data_dir=env["user_root"],
    )


def test_second_resolution_and_prepare_decode_exactly_once(
    pack_environment, monkeypatch
) -> None:
    spy = _DecodeSpy(monkeypatch)
    env = pack_environment

    first = _resolve_anim(env)
    assert first.image_bytes == env["anim_bytes"]
    assert first.is_animated
    assert first.decode_identity is not None
    prepared = prepare_expression(
        first.image_bytes, (12, 8), animate=True, identity=first.decode_identity
    )
    try:
        assert len(prepared.frames) == 3
        assert prepared.durations_ms == (100, 200, 300)
        assert prepared.plays == 0
    finally:
        prepared.close()
    assert spy.inspect_passes == 1
    assert spy.playback_frames == 0

    second = _resolve_anim(env)
    prepared_again = prepare_expression(
        second.image_bytes, (12, 8), animate=True, identity=second.decode_identity
    )
    prepared_again.close()

    assert spy.inspect_passes == 1
    assert spy.playback_frames == 0
    assert second.cache_identity == first.cache_identity
    assert second.decode_identity == first.decode_identity


def test_repeated_preparations_of_one_resolution_never_redecode(
    pack_environment, monkeypatch
) -> None:
    spy = _DecodeSpy(monkeypatch)
    env = pack_environment
    resolution = _resolve_anim(env)
    for size in ((12, 8), (6, 4), (24, 16), (48, 32)):
        prepared = prepare_expression(
            resolution.image_bytes, size, animate=True, identity=resolution.decode_identity
        )
        try:
            assert len(prepared.frames) == 3
            assert prepared.durations_ms == (100, 200, 300)
        finally:
            prepared.close()
    static = prepare_expression(
        resolution.image_bytes, (5, 5), animate=False, identity=resolution.decode_identity
    )
    try:
        assert len(static.frames) == 1
        assert static.durations_ms == (100,)
    finally:
        static.close()
    assert spy.inspect_passes == 1
    assert spy.playback_frames == 0


def test_mtime_bump_invalidates_inspection_and_redecodes(
    pack_environment, monkeypatch
) -> None:
    spy = _DecodeSpy(monkeypatch)
    env = pack_environment

    first = _resolve_anim(env)
    assert spy.inspect_passes == 1

    stat_result = os.stat(env["anim_path"])
    os.utime(
        env["anim_path"],
        ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000_000),
    )

    second = _resolve_anim(env)
    assert spy.inspect_passes == 2
    assert second.image_bytes == first.image_bytes
    assert second.decode_identity == first.decode_identity


def test_shared_prepare_is_pixel_identical_to_full_decode(
    pack_environment, monkeypatch
) -> None:
    env = pack_environment
    resolution = _resolve_anim(env)
    data = resolution.image_bytes
    identity = resolution.decode_identity
    assert identity is not None

    def snapshot(prepared: Any) -> dict[str, Any]:
        try:
            return {
                "count": len(prepared.frames),
                "durations": tuple(prepared.durations_ms),
                "plays": prepared.plays,
                "reason": prepared.fallback_reason,
                "sizes": tuple(frame.size for frame in prepared.frames),
                "pixels": tuple(frame.tobytes() for frame in prepared.frames),
            }
        finally:
            prepared.close()

    # Force the full-decode path with a key that can never hit the store.
    golden_animated = snapshot(
        prepare_expression(
            data, (10, 6), animate=True, identity=("vi-decode-v1", "content", "golden-a")
        )
    )
    golden_static = snapshot(
        prepare_expression(
            data, (7, 3), animate=False, identity=("vi-decode-v1", "content", "golden-s")
        )
    )

    shared_animated = snapshot(
        prepare_expression(data, (10, 6), animate=True, identity=identity)
    )
    shared_static = snapshot(
        prepare_expression(data, (7, 3), animate=False, identity=identity)
    )

    assert shared_animated == golden_animated
    assert shared_static == golden_static


def test_corrupt_payload_is_never_memoized(pack_environment) -> None:
    corrupt = _corrupt_bytes()
    key = ("vi-decode-v1", "content", "corrupt-fixture")
    for _ in range(2):
        with pytest.raises(ValueError, match="visual_identity_asset_decode_invalid"):
            visual_identity._inspect_image_bytes(corrupt, key=key)
    assert key not in visual_identity._inspection_memo


def test_corrupt_pack_asset_falls_back_every_resolution(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "corrupt.db", "corrupt-test")
    user_root = tmp_path / "profile"
    actor_id = db.add_character_card({"name": "Corrupt actor"})
    assert actor_id is not None
    corrupt = _corrupt_bytes()
    relpath = "characters/corrupt/expressions/broken.png"
    _write_user_asset(user_root, relpath, corrupt)
    VisualIdentityRepository(db).activate_pack(
        pack={
            "title": "Corrupt pack",
            "description": "",
            "default_expression_key": "custom:broken",
            "source_kind": "manual",
            "source_context": {"source_id": "fixture.corrupt"},
        },
        manifest={"fixture": "corrupt"},
        assets=[
            _asset_record(
                relpath,
                corrupt,
                expression_key="custom:broken",
                original_label="broken",
                content_type="image/png",
                is_animated=False,
                frame_count=1,
                duration_ms=None,
            )
        ],
        actor_kind="character",
        actor_id=actor_id,
    )
    try:

        def resolve() -> Any:
            return resolve_visual_identity(
                db,
                actor_kind="character",
                actor_id=actor_id,
                requested_state="idle",
                manual_expression_key="custom:broken",
                user_data_dir=user_root,
            )

        # The corrupt candidate is re-attempted on every resolution: failures
        # are never memoized, so the deliberate corruption check still runs.
        for _ in range(2):
            resolution = resolve()
            assert resolution.resolution_source == "placeholder"
            assert resolution.fallback_reason == "portrait_unavailable"
            assert resolution.image_bytes is None
    finally:
        db.close_connection()


def test_inspection_memo_is_bounded_and_lru(tmp_path: Path, monkeypatch) -> None:
    spy = _DecodeSpy(monkeypatch)
    db = CharactersRAGDB(tmp_path / "bounded.db", "bounded-test")
    user_root = tmp_path / "profile"
    actor_id = db.add_character_card({"name": "Bounded actor"})
    assert actor_id is not None
    total = visual_identity._INSPECTION_MEMO_LIMIT + 1
    assets = []
    for index in range(total):
        data = _png_bytes((index % 256, (index * 7) % 256, (index * 13) % 256))
        relpath = f"characters/bounded/expressions/k{index:02d}.png"
        _write_user_asset(user_root, relpath, data)
        assets.append(
            _asset_record(
                relpath,
                data,
                expression_key=f"custom:k{index:02d}",
                original_label=f"k{index:02d}",
                content_type="image/png",
                is_animated=False,
                frame_count=1,
                duration_ms=None,
            )
        )
    VisualIdentityRepository(db).activate_pack(
        pack={
            "title": "Bounded pack",
            "description": "",
            "default_expression_key": "custom:k00",
            "source_kind": "manual",
            "source_context": {"source_id": "fixture.bounded"},
        },
        manifest={"fixture": "bounded"},
        assets=assets,
        actor_kind="character",
        actor_id=actor_id,
    )
    try:
        for index in range(total):
            resolution = resolve_visual_identity(
                db,
                actor_kind="character",
                actor_id=actor_id,
                requested_state="idle",
                manual_expression_key=f"custom:k{index:02d}",
                user_data_dir=user_root,
            )
            assert resolution.image_bytes is not None
        assert spy.inspect_passes == total
        assert len(visual_identity._inspection_memo) <= visual_identity._INSPECTION_MEMO_LIMIT

        # k01 is still resident (LRU keeps the newest 32 of 33).
        resolve_visual_identity(
            db,
            actor_kind="character",
            actor_id=actor_id,
            requested_state="idle",
            manual_expression_key="custom:k01",
            user_data_dir=user_root,
        )
        assert spy.inspect_passes == total

        # k00 was the oldest entry: it was evicted and must re-decode.
        resolve_visual_identity(
            db,
            actor_kind="character",
            actor_id=actor_id,
            requested_state="idle",
            manual_expression_key="custom:k00",
            user_data_dir=user_root,
        )
        assert spy.inspect_passes == total + 1
    finally:
        db.close_connection()


class _FakePersonaService:
    """Minimal local Persona profile service surface used by the authority."""

    def __init__(self, portrait: bytes, *, character_version: int = 3) -> None:
        self._persona = {
            "backend": "local",
            "id": "p1",
            "version": 9,
            "deleted": False,
            "is_active": True,
            "character_card_id": 5,
        }
        self.character = {
            "id": 5,
            "version": character_version,
            "deleted": False,
            "image": portrait,
        }

    def get_persona_profile(self, persona_id: str) -> dict[str, Any]:
        return dict(self._persona)

    def get_character(self, character_id: int) -> dict[str, Any]:
        return dict(self.character)


@pytest.fixture
def portrait_sha_spy(monkeypatch: pytest.MonkeyPatch):
    counter = {"portrait_hashes": 0}
    portrait_holder: dict[str, bytes] = {}
    real_sha256 = hashlib.sha256

    def counting_sha256(data: Any = b"", **kwargs: Any) -> Any:
        if (
            isinstance(data, (bytes, bytearray, memoryview))
            and "portrait" in portrait_holder
            and bytes(data) == portrait_holder["portrait"]
        ):
            counter["portrait_hashes"] += 1
        return real_sha256(data, **kwargs)

    monkeypatch.setattr(persona_visual_identity.hashlib, "sha256", counting_sha256)
    return counter, portrait_holder


def test_persona_revalidation_skips_portrait_rehash(
    portrait_sha_spy, monkeypatch
) -> None:
    counter, holder = portrait_sha_spy
    portrait = _png_bytes((90, 40, 200), size=(32, 32))
    holder["portrait"] = portrait
    service = _FakePersonaService(portrait)

    authority = capture_local_persona_visual_identity(service, "p1")
    assert authority is not None and authority.portrait is not None
    assert authority.portrait.data == portrait
    assert counter["portrait_hashes"] == 1

    for _ in range(3):
        assert local_persona_visual_identity_is_current(service, authority) is True
    assert counter["portrait_hashes"] == 1

    # A changed card stat (row version bump) forces one recapture and rehash.
    service.character["version"] += 1
    assert local_persona_visual_identity_is_current(service, authority) is False
    assert counter["portrait_hashes"] == 2

    refreshed = capture_local_persona_visual_identity(service, "p1")
    assert refreshed is not None and refreshed.portrait is not None
    assert refreshed.portrait.revision == 4
    assert refreshed.portrait.sha256 == authority.portrait.sha256
    for _ in range(2):
        assert local_persona_visual_identity_is_current(service, refreshed) is True
    assert counter["portrait_hashes"] == 2


def test_persona_resolution_reuses_portrait_across_resolves(
    tmp_path: Path, portrait_sha_spy, monkeypatch
) -> None:
    counter, holder = portrait_sha_spy
    portrait = _gif_bytes(frame_size=(20, 20))
    holder["portrait"] = portrait
    service = _FakePersonaService(portrait)
    spy = _DecodeSpy(monkeypatch)
    db = CharactersRAGDB(tmp_path / "persona.db", "persona-test")
    try:
        first = resolve_persona_visual_identity(
            db, service, persona_id="p1", requested_state="idle"
        )
        assert first.image_bytes == portrait
        assert first.resolution_source == "persona_portrait"
        assert first.decode_identity is not None
        second = resolve_persona_visual_identity(
            db, service, persona_id="p1", requested_state="idle"
        )
        assert second.cache_identity == first.cache_identity
        assert second.decode_identity == first.decode_identity

        # One portrait hash and one portrait decode across two full resolves
        # (each resolve revalidates the authority several times).
        assert counter["portrait_hashes"] == 1
        assert spy.inspect_passes == 1

        prepared = prepare_expression(
            second.image_bytes, (10, 10), animate=True, identity=second.decode_identity
        )
        try:
            assert len(prepared.frames) == 3
            assert prepared.durations_ms == (100, 200, 300)
        finally:
            prepared.close()
        assert spy.inspect_passes == 1
        assert spy.playback_frames == 0
    finally:
        db.close_connection()
