"""App-owned Buddy authority, persistence, state leases, and cancellation tests."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
from dataclasses import FrozenInstanceError, replace

import pytest

from Tests.Persona_Visual.test_persona_visual_publication import _snapshot
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Persona_Visual.publication import publish_persona_visual
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository

PERSONA = "persona-local-1"


@pytest.fixture
def environment(tmp_path):
    source = tmp_path / "source"
    profile = tmp_path / "profile"
    source.mkdir()
    profile.mkdir(mode=0o700)
    db = CharactersRAGDB(tmp_path / "buddy.db", "buddy-test")
    service = LocalCharacterPersonaService(None)
    service.create_persona_profile({"id": PERSONA, "name": "Migu"})
    repo = PersonaVisualRepository(db)
    snapshot = _snapshot(source, persona_revision=1)
    publish_persona_visual(
        repo,
        snapshot,
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: True,
    )
    yield service, repo, profile, source, snapshot
    db.close_connection()


def controller(environment, **kwargs):
    from tldw_chatbook.Persona_Visual.buddy import BuddyController

    return BuddyController(*environment[:3], **kwargs)


async def test_default_off_requires_explicit_eligible_local_selection(environment):
    assert importlib.util.find_spec("tldw_chatbook.Persona_Visual.buddy") is not None
    buddy = controller(environment)
    assert await buddy.refresh(28, 16) is None
    assert not buddy.preferences.enabled
    assert not await buddy.select(PERSONA, source="server")
    assert buddy.reason == "persona_buddy_local_required"
    assert not buddy.preferences.enabled
    assert await buddy.select(PERSONA)
    rendered = await buddy.refresh(28, 16)
    assert rendered.persona_id == PERSONA
    assert rendered.name == "Migu"
    assert rendered.resolution.frames
    assert buddy.preferences.enabled
    with pytest.raises(FrozenInstanceError):
        buddy.preferences.enabled = False


async def test_profile_preferences_survive_new_controller_without_retarget(environment):
    buddy = controller(environment)
    await buddy.select(PERSONA)
    await buddy.update_preferences(x=14, y=8, width=35, height=19, collapsed=True)
    restored = controller(environment)
    await restored.load_preferences()
    assert restored.preferences == buddy.preferences
    assert restored.preferences.source == "local"
    saved = environment[2] / "persona_buddy.json"
    assert saved.stat().st_mode & 0o777 == 0o600
    assert "Migu" not in saved.read_text()
    assert str(environment[2]) not in saved.read_text()
    with pytest.raises(ValueError, match="^persona_buddy_preferences_invalid$"):
        await restored.update_preferences(local_persona_id="other")


async def test_disabled_deleted_missing_and_unbound_hide_but_restore(environment):
    buddy = controller(environment)
    service, repo = environment[:2]
    await buddy.select(PERSONA)
    first = await buddy.refresh(28, 16)
    service.update_persona_profile(PERSONA, {"is_active": False})
    assert await buddy.refresh(28, 16) is None
    assert buddy.reason == "persona_buddy_unavailable"
    assert buddy.preferences.enabled
    service.update_persona_profile(PERSONA, {"is_active": True})
    restored = await buddy.refresh(28, 16)
    assert restored is not None
    assert restored.generation > first.generation
    service.delete_persona_profile(PERSONA)
    assert await buddy.refresh(28, 16) is None
    service.restore_persona_profile(PERSONA, expected_version=4)
    assert await buddy.refresh(28, 16) is not None
    with repo.db.transaction() as cursor:
        cursor.execute("UPDATE persona_visual_bindings SET status='deleted'")
    assert await buddy.refresh(28, 16) is None
    assert buddy.preferences.enabled


def test_source_scoped_leases_priority_expiry_and_wake_live_guard(environment):
    clock = [100.0]
    buddy = controller(environment, clock=lambda: clock[0])
    buddy.signal("network", "offline")
    buddy.signal("wake", "wake_armed")
    assert buddy.requested_state == "wake_armed"
    buddy.signal("voice", "speaking")
    assert buddy.requested_state == "speaking"
    buddy.signal("tool-a", "tool_running")
    buddy.signal("tool-b", "tool_running")
    buddy.release("tool-a")
    assert buddy.requested_state == "tool_running"
    buddy.signal("explicit", "reaction.happy", ttl=2)
    assert buddy.requested_state == "reaction.happy"
    buddy.signal("approval-a", "approval_needed")
    buddy.signal("approval-b", "approval_needed")
    buddy.release("approval-a")
    assert buddy.requested_state == "approval_needed"
    buddy.signal("fault", "error", ttl=1)
    assert buddy.requested_state == "error"
    clock[0] += 1.1
    assert buddy.requested_state == "approval_needed"
    buddy.release("approval-b")
    clock[0] += 1
    assert buddy.requested_state == "tool_running"
    buddy.release("tool-b")
    buddy.release("voice")
    assert buddy.requested_state == "wake_armed"
    buddy.release("wake")
    assert buddy.requested_state == "offline"
    buddy.release("network")
    assert buddy.requested_state == "idle"
    buddy.signal("mic", "listening")
    assert buddy.requested_state == "listening"
    buddy.signal("model", "thinking")
    assert buddy.requested_state == "thinking"
    with pytest.raises(ValueError, match="^persona_buddy_signal_invalid$"):
        buddy.signal("model", "<emote>bad</emote>")
    with pytest.raises(ValueError, match="^persona_buddy_signal_invalid$"):
        buddy.signal("explicit", "reaction.happy", ttl=None)


async def test_resolution_preparation_is_off_loop_and_rechecks_local_authority(
    environment,
):
    buddy = controller(environment)
    await buddy.select(PERSONA)
    main_thread = threading.get_ident()

    def prepare(resolution):
        assert threading.get_ident() != main_thread
        environment[0].update_persona_profile(PERSONA, {"is_active": False})
        return tuple(frame.sha256 for frame in resolution.frames)

    assert await buddy.refresh(28, 16, prepare=prepare) is None
    assert buddy.reason == "persona_buddy_unavailable"


async def test_publication_during_decode_drops_old_binding_and_advances_generation(
    environment,
):
    buddy = controller(environment)
    await buddy.select(PERSONA)
    initial = await buddy.refresh(28, 16)
    _service, repo, profile, source, old = environment

    def prepare(resolution):
        snapshot = replace(old, expected_identity=initial.identity)
        publish_persona_visual(
            repo,
            snapshot,
            source_root=source,
            profile_root=profile,
            authority_guard=lambda: True,
        )
        return "old"

    assert await buddy.refresh(29, 16, prepare=prepare) is None
    current = await buddy.refresh(28, 16)
    assert current.identity != initial.identity
    assert current.generation > initial.generation


async def test_cancelled_decode_drains_before_replacement_work(environment):
    buddy = controller(environment)
    await buddy.select(PERSONA)
    started = threading.Event()
    finish = threading.Event()
    replacement_prepared = threading.Event()

    def prepare(resolution):
        started.set()
        assert finish.wait(5)
        return "old"

    old = asyncio.create_task(buddy.refresh(28, 16, prepare=prepare))
    assert await asyncio.to_thread(started.wait, 5)
    old.cancel()
    await asyncio.sleep(0)
    old.cancel()  # A second navigation cancellation must not break the drain.
    new = asyncio.create_task(
        buddy.refresh(
            35,
            16,
            prepare=lambda result: replacement_prepared.set() or "new",
        )
    )
    await asyncio.sleep(0.03)
    try:
        assert not old.done()
        assert not replacement_prepared.is_set()
    finally:
        finish.set()
    with pytest.raises(asyncio.CancelledError):
        await old
    result = await new
    assert result.prepared == "new"
    await buddy.shutdown()
    assert await buddy.refresh(28, 16) is None


async def test_signal_or_selection_change_during_decode_never_returns_stale_snapshot(
    environment,
):
    buddy = controller(environment)
    await buddy.select(PERSONA)
    started = threading.Event()
    finish = threading.Event()

    def prepare(resolution):
        started.set()
        assert finish.wait(5)
        return "stale"

    pending = asyncio.create_task(buddy.refresh(28, 16, prepare=prepare))
    assert await asyncio.to_thread(started.wait, 5)
    buddy.signal("tool", "tool_running")
    finish.set()
    assert await pending is None
    assert (await buddy.refresh(28, 16)).requested_state == "tool_running"
    await buddy.select(PERSONA)
    assert buddy.requested_state == "idle"


async def test_malformed_persisted_preferences_fail_closed_without_path_text(
    environment,
):
    path = environment[2] / "persona_buddy.json"
    path.write_text(
        json.dumps({"enabled": True, "source": "server", "local_persona_id": PERSONA})
    )
    buddy = controller(environment)
    assert await buddy.refresh(28, 16) is None
    assert not buddy.preferences.enabled
    assert buddy.reason == "persona_buddy_preferences_invalid"
    assert str(path) not in repr(buddy.preferences)


async def test_authored_trigger_is_exact_scoped_bounded_and_below_explicit(environment):
    _service, repo, profile, source, original = environment
    current = repo.get_active_persona_pack(PERSONA).identity
    manifest = json.loads(original.manifest_json)
    manifest["authored_triggers"] = [
        {
            "id": "live-speaking",
            "source": "live_state",
            "match": "speaking",
            "state": "listening",
            "duration_ms": 1000,
            "priority": 40,
        },
        {
            "id": "notes",
            "source": "tool_category",
            "match": "notes",
            "state": "thinking",
            "duration_ms": 2000,
            "priority": 20,
        },
    ]
    publish_persona_visual(
        repo,
        replace(
            original,
            expected_identity=current,
            manifest_json=json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        ),
        source_root=source,
        profile_root=profile,
        authority_guard=lambda: True,
    )
    clock = [1.0]
    buddy = controller(environment, clock=lambda: clock[0])
    await buddy.select(PERSONA)
    buddy.signal("voice", "speaking", ttl=None)
    assert buddy.requested_state == "listening"
    buddy.signal("tool", "tool_running", ttl=None)
    assert not buddy.trigger("model_text", "notes", owner="a")
    assert not buddy.trigger("tool_category", "notes extra", owner="a")
    assert buddy.trigger("tool_category", "notes", owner="a")
    assert buddy.trigger("tool_category", "notes", owner="b")
    clock[0] += 1.1
    buddy.signal("voice", "speaking", ttl=None)
    assert buddy.requested_state == "thinking"
    buddy.signal("override", "idle", ttl=1, explicit=True)
    assert buddy.requested_state == "idle"
    buddy.release("override")
    buddy.release("a")
    assert buddy.requested_state == "thinking"
    buddy.release("b")
    assert buddy.requested_state == "tool_running"
    buddy.release("tool")
    assert buddy.requested_state == "speaking"


async def test_persona_replacement_requested_mid_decode_fences_old_frame(environment):
    buddy = controller(environment)
    await buddy.select(PERSONA)
    started, finish = threading.Event(), threading.Event()

    def prepare(resolution):
        started.set()
        assert finish.wait(5)
        return "old"

    pending = asyncio.create_task(buddy.refresh(28, 16, prepare=prepare))
    assert await asyncio.to_thread(started.wait, 5)
    replacement = asyncio.create_task(buddy.select(PERSONA))
    await asyncio.sleep(0)
    finish.set()
    assert await pending is None
    assert await replacement
    assert (await buddy.refresh(28, 16)).prepared is None


async def test_portrait_fallback_resolves_when_bound_idle_asset_is_unavailable(
    environment,
):
    import hashlib

    from Tests.Persona_Visual.test_persona_visual_publication import _png_bytes
    from tldw_chatbook.Persona_Visual.runtime import PersonaVisualPortrait

    buddy = controller(environment)
    await buddy.select(PERSONA)
    data = _png_bytes()
    portrait = PersonaVisualPortrait(
        "persona-portrait", 1, "image/png", hashlib.sha256(data).hexdigest(), data
    )
    for frame in environment[2].glob("persona_visual/packs/*/versions/*/assets/*"):
        frame.unlink()
    resolved = await buddy.refresh(28, 16, portrait=portrait)
    assert resolved.resolution.source == "persona_portrait"
    assert resolved.resolution.portrait == portrait


def test_renewing_same_visible_state_does_not_restart_current_animation(environment):
    clock = [1.0]
    buddy = controller(environment, clock=lambda: clock[0])
    buddy.signal("voice", "speaking", ttl=2)
    generation = buddy.generation
    clock[0] += 1
    buddy.signal("voice", "speaking", ttl=2)
    buddy.signal("voice-second", "speaking", ttl=2)
    buddy.release("voice-second")
    assert buddy.generation == generation
    clock[0] += 1.1
    assert buddy.requested_state == "speaking"
    clock[0] += 1
    assert buddy.requested_state == "idle"
    assert buddy.generation > generation


async def test_symlink_preferences_fail_closed_without_reading_target(
    environment, tmp_path
):
    external = tmp_path / "external.json"
    external.write_text(json.dumps({"enabled": True, "local_persona_id": PERSONA}))
    (environment[2] / "persona_buddy.json").symlink_to(external)
    buddy = controller(environment)
    await buddy.load_preferences()
    assert not buddy.preferences.enabled
    assert buddy.reason == "persona_buddy_preferences_invalid"
    assert json.loads(external.read_text())["enabled"] is True


async def test_fifo_preferences_fail_closed_and_allow_shutdown(environment):
    import os

    fifo = environment[2] / "persona_buddy.json"
    os.mkfifo(fifo, 0o600)
    buddy = controller(environment)
    pending = asyncio.create_task(buddy.load_preferences())
    completed = False
    try:
        done, _ = await asyncio.wait({pending}, timeout=0.5)
        completed = pending in done
    finally:
        # The RED implementation is blocked opening the FIFO. Pair with its
        # reader and close, so an assertion failure cannot strand the executor.
        if not completed:
            descriptor = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            os.close(descriptor)
        await pending
        await asyncio.wait_for(buddy.shutdown(), timeout=1)
    assert completed, "A FIFO must be rejected before opening a blocking reader"
    assert not buddy.preferences.enabled
    assert buddy.reason == "persona_buddy_preferences_invalid"


@pytest.mark.parametrize("unsafe", ["permissions", "hardlink", "oversized"])
async def test_nonprivate_preferences_fail_closed(environment, unsafe):
    import os

    path = environment[2] / "persona_buddy.json"
    contents = json.dumps({"enabled": True, "local_persona_id": PERSONA})
    path.write_text(contents)
    path.chmod(0o600)
    if unsafe == "permissions":
        path.chmod(0o644)
    elif unsafe == "hardlink":
        os.link(path, environment[2] / "alias.json")
    else:
        path.write_text(contents + " " * 8192)
    buddy = controller(environment)
    await buddy.load_preferences()
    assert not buddy.preferences.enabled
    assert buddy.reason == "persona_buddy_preferences_invalid"


@pytest.mark.parametrize("unsafe", ["symlink", "permissions"])
async def test_unsafe_profile_root_cannot_receive_preferences(environment, unsafe):
    from tldw_chatbook.Persona_Visual.buddy import BuddyController

    root = environment[2]
    if unsafe == "symlink":
        root = root.parent / "linked-profile"
        root.symlink_to(environment[2], target_is_directory=True)
    else:
        root.chmod(0o755)
    buddy = BuddyController(environment[0], environment[1], root)
    assert not await buddy.select(PERSONA)
    assert buddy.reason == "persona_buddy_preferences_invalid"
    assert not (environment[2] / "persona_buddy.json").exists()


async def test_missing_profile_below_symlink_does_not_create_external_directory(
    environment,
):
    from tldw_chatbook.Persona_Visual.buddy import BuddyController

    linked_parent = environment[2].parent / "linked-parent"
    linked_parent.symlink_to(environment[2], target_is_directory=True)
    buddy = BuddyController(
        environment[0], environment[1], linked_parent / "new-profile"
    )
    assert not await buddy.select(PERSONA)
    assert not (environment[2] / "new-profile").exists()


async def test_corrupt_asset_keeps_current_binding_snapshot_for_static_fallback(
    environment,
):
    buddy = controller(environment)
    await buddy.select(PERSONA)
    first = await buddy.refresh(28, 16)
    assert first.resolution.frames
    assets = tuple(environment[2].glob("persona_visual/packs/*/versions/*/assets/*"))
    assert assets
    for asset in assets:
        asset.write_bytes(b"invalid raster")
    current = await buddy.refresh(
        28,
        16,
        prepare=lambda resolution: tuple(frame.sha256 for frame in resolution.frames),
    )
    assert current is not None
    assert current.identity == first.identity
    assert current.persona_id == PERSONA
    assert current.resolution.source == "unavailable"
    assert current.resolution.reason == "persona_visual_unavailable"
    assert current.resolution.frames == ()
    assert current.prepared == ()
    assert buddy.reason == "persona_visual_unavailable"
    assert str(environment[2]) not in repr(current)
