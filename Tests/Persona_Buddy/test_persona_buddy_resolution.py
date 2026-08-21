"""Async Persona Buddy resolution, rendering, and lifetime contracts."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import threading
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Buddy.controller import (
    BuddyDrainResult,
    PersonaBuddyController,
)
from tldw_chatbook.Persona_Buddy.preferences import (
    PersonaBuddyPreferences,
    PersonaBuddySelection,
)
from tldw_chatbook.Persona_Buddy.rendering import (
    PersonaBuddyFrameError,
    prepare_persona_buddy_frame,
)
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
from tldw_chatbook.Persona_Visual.runtime import (
    PersonaVisualCacheAsset,
    PersonaVisualCacheIdentity,
    PersonaVisualPortrait,
    PersonaVisualResolvedFrame,
)


def _png(colour: tuple[int, int, int, int], size: tuple[int, int] = (6, 8)) -> bytes:
    output = BytesIO()
    Image.new("RGBA", size, colour).save(output, format="PNG")
    return output.getvalue()


def _gif() -> bytes:
    output = BytesIO()
    frames = [
        Image.new("RGBA", (4, 4), (220, 20, 20, 255)),
        Image.new("RGBA", (4, 4), (20, 20, 220, 255)),
    ]
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )
    return output.getvalue()


def _manifest() -> dict[str, object]:
    states = {
        "idle": {"animation_id": "idle"},
        "wake_armed": {"animation_id": "idle"},
        "listening": {"animation_id": "listening"},
        "thinking": {"animation_id": "idle"},
        "speaking": {"animation_id": "idle"},
        "tool_running": {"animation_id": "idle"},
        "approval_needed": {"animation_id": "idle"},
        "error": {"animation_id": "idle"},
        "offline": {"animation_id": "idle"},
    }
    return {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": states,
        "animations": {
            "idle": {"frames": [{"asset_id": "idle"}]},
            "listening": {
                "frames": [
                    {"asset_id": "listen-a", "duration_ms": 80},
                    {"asset_id": "listen-b", "duration_ms": 120},
                ],
                "frame_rate": 12,
            },
        },
        "fallbacks": {},
        "state_catalog": {},
        "authored_triggers": [],
    }


def _write_personas(path: Path, *, active: bool = True, deleted: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "id": "persona-local-1",
                        "name": "Local operator",
                        "is_active": active,
                        "deleted": deleted,
                        "version": 7,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _runtime(
    tmp_path: Path,
    *,
    active: bool = True,
    deleted: bool = False,
    bound: bool = True,
    reduced_motion: bool = False,
    phase_barrier=None,
    portrait: bytes | None = None,
) -> tuple[PersonaBuddyController, CharactersRAGDB, object | None]:
    root = tmp_path / "profile"
    root.mkdir()
    persona_path = root / "personas.json"
    _write_personas(persona_path, active=active, deleted=deleted)
    service = LocalCharacterPersonaService(None, persona_store_path=persona_path)
    db = CharactersRAGDB(tmp_path / "persona-visual.db", client_id="buddy-test")
    graph = None
    if bound:
        assets: list[dict[str, object]] = []
        for key, colour in (
            ("idle", (20, 160, 80, 255)),
            ("listen-a", (220, 20, 20, 255)),
            ("listen-b", (20, 20, 220, 255)),
        ):
            data = _png(colour)
            relpath = f"persona_visual/buddy/v1/{key}.png"
            target = root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            assets.append(
                {
                    "asset_key": key,
                    "role": "frame",
                    "storage_relpath": relpath,
                    "mime_type": "image/png",
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "width": 6,
                    "height": 8,
                    "frame_count": 1,
                    "duration_ms": None,
                }
            )
        graph = PersonaVisualRepository(db).activate_new_pack(
            persona_id="persona-local-1",
            title="Buddy",
            manifest=_manifest(),
            manifest_storage_relpath="persona_visual/buddy/v1/manifest.json",
            assets=assets,
            expected_persona_revision=7,
            authority_guard=lambda: True,
        )
    preferences = PersonaBuddyPreferences(
        enabled=True,
        selection=PersonaBuddySelection("local", "persona-local-1"),
    )
    controller = PersonaBuddyController(
        preferences=preferences,
        local_persona_service=service,
        profile_db=db,
        profile_root=root,
        reduced_motion=reduced_motion,
        phase_barrier=phase_barrier,
        portrait_loader=(
            lambda _record: (
                PersonaVisualPortrait(
                    portrait_id="local-portrait",
                    revision=7,
                    mime_type="image/png",
                    sha256=hashlib.sha256(portrait).hexdigest(),
                    data=portrait,
                )
                if portrait is not None
                else None
            )
        ),
    )
    return controller, db, graph


@pytest.mark.asyncio
async def test_resolve_selected_local_persona_from_real_active_binding(
    tmp_path: Path,
) -> None:
    controller, db, graph = _runtime(tmp_path)
    assert graph is not None
    controller.acquire_state(source="voice", owner="turn", state="listening")
    try:
        visual = await controller.resolve_current_visual(cols=24, lines=10)
    finally:
        await controller.shutdown()
        db.close_connection()

    assert visual.available is True
    assert visual.source == "persona_visual"
    assert visual.graph_identity == graph.identity
    assert visual.cache_identity.graph == graph.identity
    assert visual.cache_identity.requested_state == "listening"
    assert visual.cache_identity.resolved_state == "listening"
    assert visual.cache_identity.animation_id == "listening"
    assert visual.requested_state == "listening"
    assert visual.resolved_state == "listening"
    assert visual.animation_id == "listening"
    assert visual.animate is True
    assert tuple(frame.asset_id for frame in visual.frames) == tuple(
        asset.id for asset in graph.assets if asset.asset_key.startswith("listen-")
    )
    assert tuple(frame.manifest_frame_index for frame in visual.frames) == (0, 1)
    assert tuple(frame.selected_frame for frame in visual.frames) == (0, 0)
    assert visual.cache_identity.assets == tuple(
        PersonaVisualCacheAsset(
            asset_id=frame.asset_id,
            asset_key=frame.asset_key,
            sha256=frame.asset_sha256,
            manifest_frame_index=frame.manifest_frame_index,
            selected_frame=frame.selected_frame,
        )
        for frame in visual.frames
    )


@pytest.mark.parametrize("case", ("disabled", "deleted", "missing", "unbound"))
@pytest.mark.asyncio
async def test_disabled_deleted_missing_or_unbound_selection_preserves_enabled_but_hides(
    tmp_path: Path,
    case: str,
) -> None:
    controller, db, _graph = _runtime(
        tmp_path,
        active=case != "disabled",
        deleted=case == "deleted",
        bound=case != "unbound",
    )
    if case == "missing":
        controller.select_local_persona("persona-missing")
    try:
        visual = await controller.resolve_current_visual(cols=20, lines=8)
        snapshot = controller.snapshot()
    finally:
        await controller.shutdown()
        db.close_connection()

    assert visual.available is False
    assert visual.frames == ()
    assert visual.reason in {
        "persona_buddy_persona_unavailable",
        "persona_buddy_binding_unavailable",
    }
    assert snapshot.enabled is True
    assert snapshot.selection is not None


@pytest.mark.asyncio
async def test_state_idle_portrait_fallback_never_blanks(tmp_path: Path) -> None:
    portrait = _png((120, 90, 210, 255))
    controller, db, _graph = _runtime(tmp_path, portrait=portrait)
    (tmp_path / "profile/persona_visual/buddy/v1/idle.png").write_bytes(b"broken")
    try:
        fallback = await controller.resolve_current_visual(cols=20, lines=8)
    finally:
        await controller.shutdown()
        db.close_connection()

    assert fallback.available is True
    assert fallback.source == "persona_portrait"
    assert fallback.requested_state == "idle"
    assert fallback.frames[0].cells
    assert fallback.cache_identity is not None
    assert fallback.cache_identity.portrait_id == "local-portrait"


def _resolved_frame(
    data: bytes, *, selected_frame: int = 0
) -> PersonaVisualResolvedFrame:
    return PersonaVisualResolvedFrame(
        asset_id=9,
        asset_key="sprite",
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
        duration_ms=80,
        region=None,
        manifest_frame_index=3,
        selected_frame=selected_frame,
    )


def _cache(data: bytes, *, selected_frame: int = 0) -> PersonaVisualCacheIdentity:
    return PersonaVisualCacheIdentity(
        graph=None,
        requested_state="listening",
        resolved_state="listening",
        animation_id="listen",
        reduced_motion=False,
        assets=(
            PersonaVisualCacheAsset(
                asset_id=9,
                asset_key="sprite",
                sha256=hashlib.sha256(data).hexdigest(),
                manifest_frame_index=3,
                selected_frame=selected_frame,
            ),
        ),
    )


def test_sprite_frames_prepare_distinct_painted_frames() -> None:
    data = _gif()
    first = prepare_persona_buddy_frame(
        _resolved_frame(data), resolution_cache_identity=_cache(data), cols=8, lines=4
    )
    second = prepare_persona_buddy_frame(
        _resolved_frame(data, selected_frame=1),
        resolution_cache_identity=_cache(data, selected_frame=1),
        cols=8,
        lines=4,
    )

    assert first.cells and second.cells
    assert first.cells != second.cells
    assert first.selected_frame == 0
    assert second.selected_frame == 1


@pytest.mark.asyncio
async def test_reduced_motion_prepares_only_frame_zero(tmp_path: Path) -> None:
    controller, db, _graph = _runtime(tmp_path, reduced_motion=True)
    controller.acquire_state(source="voice", owner="turn", state="listening")
    try:
        visual = await controller.resolve_current_visual(cols=20, lines=8)
    finally:
        await controller.shutdown()
        db.close_connection()

    assert len(visual.frames) == 1
    assert visual.frames[0].manifest_frame_index == 0
    assert visual.frames[0].selected_frame == 0
    assert visual.animate is False
    assert visual.cache_identity.reduced_motion is True


@pytest.mark.asyncio
async def test_decode_failure_keeps_previous_or_portrait_frame(tmp_path: Path) -> None:
    controller, db, _graph = _runtime(tmp_path)
    first = await controller.resolve_current_visual(cols=20, lines=8)
    assert first.available and first.frames
    (tmp_path / "profile/persona_visual/buddy/v1/idle.png").write_bytes(b"broken")
    controller.invalidate_profile()
    try:
        failed = await controller.resolve_current_visual(cols=20, lines=8)
    finally:
        await controller.shutdown()
        db.close_connection()

    assert failed.available is True
    assert failed.frames == first.frames
    assert failed.reason == "persona_buddy_frame_unavailable"


def test_direct_decode_failure_is_path_free() -> None:
    with pytest.raises(
        PersonaBuddyFrameError, match="^persona_buddy_frame_unavailable$"
    ):
        prepare_persona_buddy_frame(
            _resolved_frame(b"/private/profile/not-an-image"),
            resolution_cache_identity=_cache(b"/private/profile/not-an-image"),
            cols=8,
            lines=4,
        )


def test_render_snapshot_repr_is_byte_and_path_free() -> None:
    data = _png((1, 2, 3, 255))
    prepared = prepare_persona_buddy_frame(
        _resolved_frame(data), resolution_cache_identity=_cache(data), cols=8, lines=4
    )

    rendered = repr(prepared)
    assert prepared.cells
    assert prepared.renderable is not None
    assert "renderable=" not in rendered
    assert "data=" not in rendered
    assert "/private/" not in rendered
    assert str(data) not in rendered


async def _assert_stale_after_phase(tmp_path: Path, phase: str) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    visited: list[str] = []

    async def barrier(current: str) -> None:
        visited.append(current)
        if current == phase:
            entered.set()
            await release.wait()

    controller, db, _graph = _runtime(tmp_path, phase_barrier=barrier)
    downstream: list[str] = []
    if phase == "persona_read":
        original = controller._read_graph

        def track_graph(*args: object) -> object:
            downstream.append("graph_read")
            return original(*args)

        controller._read_graph = track_graph  # type: ignore[method-assign]
    elif phase == "graph_read":
        original = controller._resolve_runtime

        def track_runtime(*args: object) -> object:
            downstream.append("runtime_resolve")
            return original(*args)  # type: ignore[arg-type]

        controller._resolve_runtime = track_runtime  # type: ignore[method-assign]
    elif phase == "runtime_resolve":
        original = controller._prepare_resolution

        def track_prepare(*args: object) -> object:
            downstream.append("frame_prepare")
            return original(*args)  # type: ignore[arg-type]

        controller._prepare_resolution = track_prepare  # type: ignore[method-assign]
    task = asyncio.create_task(controller.resolve_current_visual(cols=20, lines=8))
    await asyncio.wait_for(entered.wait(), 2)
    controller.select_local_persona("replacement")
    release.set()
    try:
        result = await asyncio.wait_for(task, 2)
    finally:
        await controller.shutdown()
        db.close_connection()

    assert result.reason == "persona_buddy_resolution_stale"
    assert controller.snapshot().visual is None
    ordered = ["persona_read", "graph_read", "runtime_resolve", "frame_prepare"]
    assert visited == ordered[: ordered.index(phase) + 1]
    assert downstream == []


@pytest.mark.asyncio
async def test_stale_after_persona_read_cannot_apply(tmp_path: Path) -> None:
    await _assert_stale_after_phase(tmp_path, "persona_read")


@pytest.mark.asyncio
async def test_stale_after_graph_read_cannot_apply(tmp_path: Path) -> None:
    await _assert_stale_after_phase(tmp_path, "graph_read")


@pytest.mark.asyncio
async def test_stale_after_runtime_resolve_cannot_apply(tmp_path: Path) -> None:
    await _assert_stale_after_phase(tmp_path, "runtime_resolve")


@pytest.mark.asyncio
async def test_stale_after_frame_prepare_cannot_apply(tmp_path: Path) -> None:
    await _assert_stale_after_phase(tmp_path, "frame_prepare")


@pytest.mark.asyncio
async def test_binding_version_change_between_reads_cannot_apply(
    tmp_path: Path,
) -> None:
    changed = False
    controller: PersonaBuddyController
    db: CharactersRAGDB

    async def barrier(phase: str) -> None:
        nonlocal changed
        if phase != "graph_read" or changed:
            return
        changed = True

        def replace_binding_version() -> None:
            with db.transaction():
                db.execute_query(
                    "UPDATE persona_visual_bindings SET version = version + 1 "
                    "WHERE persona_id = ? AND status = 'active'",
                    ("persona-local-1",),
                )

        await asyncio.to_thread(replace_binding_version)

    controller, db, _graph = _runtime(tmp_path, phase_barrier=barrier)
    try:
        visual = await controller.resolve_current_visual(cols=20, lines=8)
    finally:
        await controller.shutdown()
        db.close_connection()

    assert changed is True
    assert visual.available is False
    assert visual.reason == "persona_buddy_frame_unavailable"
    assert controller.snapshot().visual is not None
    assert controller.snapshot().visual.available is False


@pytest.mark.asyncio
async def test_drain_distinguishes_successful_none_from_child_self_cancel() -> None:
    controller = PersonaBuddyController()

    async def none() -> None:
        return None

    async def child_cancel() -> None:
        raise asyncio.CancelledError

    completed = await controller._drain_owned(none(), name="none")
    cancelled = await controller._drain_owned(child_cancel(), name="child-cancel")
    await controller.shutdown()

    assert completed == BuddyDrainResult(completed=True, value=None)
    assert cancelled.completed is False
    assert cancelled.error_category == "persona_buddy_operation_cancelled"


@pytest.mark.asyncio
async def test_repeated_cancel_drains_before_next_owner() -> None:
    controller = PersonaBuddyController()
    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def blocking() -> str:
        order.append("first-start")
        entered.set()
        release.wait(2)
        order.append("first-end")
        return "done"

    first = asyncio.create_task(controller.run_serialized(blocking, name="first"))
    assert await asyncio.to_thread(entered.wait, 2)
    first.cancel()
    first.cancel()
    second = asyncio.create_task(
        controller.run_serialized(lambda: order.append("second"), name="second")
    )
    await asyncio.sleep(0)
    assert order == ["first-start"]
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    await controller.shutdown()

    assert order == ["first-start", "first-end", "second"]


@pytest.mark.asyncio
async def test_preference_commit_survives_outer_cancellation() -> None:
    entered = threading.Event()
    release = threading.Event()

    def writer(_preferences: PersonaBuddyPreferences) -> bool:
        entered.set()
        release.wait(2)
        return True

    controller = PersonaBuddyController(preference_writer=writer)
    updated = PersonaBuddyPreferences(
        enabled=True,
        selection=PersonaBuddySelection("local", "persona-local-1"),
    )
    operation = asyncio.create_task(controller.update_preferences(updated))
    assert await asyncio.to_thread(entered.wait, 2)
    operation.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation

    assert controller.snapshot().enabled is True
    assert controller.snapshot().selection == updated.selection
    await controller.shutdown()


@pytest.mark.asyncio
async def test_shutdown_drains_before_profile_db_closes() -> None:
    controller = PersonaBuddyController()
    entered = threading.Event()
    release = threading.Event()
    closed = False

    def blocking() -> None:
        entered.set()
        release.wait(2)
        assert closed is False

    operation = asyncio.create_task(
        controller.run_serialized(blocking, name="profile-read")
    )
    assert await asyncio.to_thread(entered.wait, 2)
    shutdown = asyncio.create_task(controller.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    closed = True
    # This models the forbidden app ordering and must still be visible to the
    # worker; restore ownership before releasing it.
    closed = False
    release.set()
    await operation
    await shutdown


def test_controller_resolution_contract_has_no_textual_dependency() -> None:
    source = inspect.getsource(PersonaBuddyController)
    assert "textual" not in source.lower()
    assert "screen" not in source.lower()
    assert "widget" not in source.lower()
