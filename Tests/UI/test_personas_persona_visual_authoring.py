"""Mounted orchestration tests for Persona Visual authoring in Personas."""

from __future__ import annotations

import asyncio
import threading
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
import tldw_chatbook.UI.Screens.personas_screen as personas_screen_module
from Tests.UI.test_personas_workbench import (
    CHARACTERS,
    PROFILE,
    PersonasTestApp,
    _mounted,
)
from tldw_chatbook.Persona_Visual.importer import PersonaVisualImportReview
from tldw_chatbook.Persona_Visual.publication import (
    PersonaVisualPublicationError,
    PersonaVisualPublicationResult,
)
from tldw_chatbook.Persona_Visual.repository import PersonaVisualIdentity
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Utils.paths import get_user_data_dir
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    EditPersonaProfileRequested,
    PersonaProfileSaveRequested,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_persona_visual_pack_widget import (
    PersonasPersonaVisualPackWidget,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def stub_characters(monkeypatch):
    from Tests.UI.test_personas_dictionaries import patch_character_paging

    monkeypatch.setattr(
        character_handler_module,
        "fetch_all_characters",
        lambda: [dict(character) for character in CHARACTERS],
    )
    monkeypatch.setattr(
        character_handler_module,
        "fetch_character_by_id",
        lambda character_id: next(
            dict(character)
            for character in CHARACTERS
            if str(character["id"]) == str(character_id)
        ),
    )
    patch_character_paging(monkeypatch)


class _Repository:
    def __init__(self, _db):
        pass

    def get_active_persona_pack(self, _persona_id):
        return None


def _identity(version: int = 1) -> PersonaVisualIdentity:
    return PersonaVisualIdentity(
        persona_id="p-1",
        persona_revision=2,
        binding_id=11,
        binding_version=version,
        pack_id=12,
        pack_revision=version,
        pack_version_id=13,
        version_number=version,
        manifest_sha256="a" * 64,
    )


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGBA", (4, 4), (20, 40, 60, 255)).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def local_scope(mock_app_instance):
    record = {**PROFILE, "version": 2, "is_active": True, "deleted": False}
    local = Mock()
    local.get_persona_profile.return_value = dict(record)
    scope = Mock()
    scope.local_service = local
    scope.list_persona_profiles = AsyncMock(
        return_value={"items": [dict(record)], "total": 1}
    )
    scope.get_persona_profile = AsyncMock(return_value=dict(record))
    scope.update_persona_profile = AsyncMock(return_value=dict(record))
    scope.create_persona_profile = AsyncMock(return_value=dict(record))
    mock_app_instance.character_persona_scope_service = scope
    mock_app_instance.chachanotes_db = object()
    return scope


async def _open_editor(pilot) -> PersonasScreen:
    screen = await _mounted(pilot)
    await pilot.click("#personas-mode-personas")
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.click("#personas-library-row-persona-p-1")
    await pilot.pause()
    screen.post_message(EditPersonaProfileRequested("p-1"))
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    return screen


async def test_local_editor_loads_isolated_unbound_draft(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        state = screen._persona_visual_authoring
        browser = screen.query_one(PersonasPersonaVisualPackWidget)

        assert state is not None
        assert state.draft.persona_id == "p-1"
        assert state.draft.persona_revision == 2
        assert state.dirty is False
        assert browser.availability == "available"


async def test_custom_edit_is_staged_and_blocks_profile_save(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)
    notifications: list[str] = []
    app.notify = lambda message, **_kwargs: notifications.append(str(message))

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        assert await screen._stage_persona_visual_custom(
            "deep_focus", "Deep focus", "mood"
        )
        screen.post_message(
            PersonaProfileSaveRequested(
                {"id": "p-1", "version": 2, "name": "Archivist"}
            )
        )
        await pilot.pause()

        assert screen._persona_visual_authoring is not None
        assert screen._persona_visual_authoring.dirty is True
        assert local_scope.update_persona_profile.await_count == 0
        assert any("Save or Cancel Persona Visual" in item for item in notifications)


async def test_import_review_replaces_draft_without_publication(
    monkeypatch, mock_app_instance, stub_characters, local_scope, tmp_path
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)
    publish = Mock()
    monkeypatch.setattr(personas_screen_module, "publish_persona_visual", publish)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        original = screen._persona_visual_authoring
        imported = personas_screen_module.create_persona_visual_draft(
            persona_id="p-1", persona_revision=2, title="Imported"
        )
        review = PersonaVisualImportReview(
            schema_version="tldw.persona_visual_pack.v1",
            archive_sha256="b" * 64,
            pack_title="Imported",
            asset_count=0,
            state_count=0,
            draft=imported,
            cleanup_candidate="pvi1:" + "c" * 64 + ":.import-" + "d" * 32,
            _candidate_name=".import-" + "d" * 32,
            _candidate_identity=(1, 2),
        )
        monkeypatch.setattr(
            personas_screen_module,
            "import_persona_visual_pack",
            lambda *_args, **_kwargs: review,
        )
        monkeypatch.setattr(
            personas_screen_module,
            "persona_visual_import_source_root",
            lambda *_args, **_kwargs: tmp_path,
        )

        assert await screen._import_persona_visual_from_path(
            "ignored.tldw-persona-vpack"
        )
        assert screen._persona_visual_authoring is original
        assert original.draft.title == "Imported"
        assert original.dirty is True
        publish.assert_not_called()


async def test_save_publishes_once_then_invalidates_exact_old_and_new(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)
    invalidated: list[tuple[str, tuple[PersonaVisualIdentity, ...]]] = []
    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        screen._session = SimpleNamespace(
            invalidate_persona_visual_identities=lambda persona_id, identities: (
                invalidated.append((persona_id, identities))
            )
        )
        state = screen._persona_visual_authoring
        assert state is not None
        state.dirty = True
        monkeypatch.setattr(
            personas_screen_module,
            "persona_visual_draft_publication_snapshot",
            lambda _draft: object(),
        )
        calls: list[object] = []

        def publish(*args, **_kwargs):
            calls.append(args)
            return PersonaVisualPublicationResult(_identity(), _identity(2), None)

        monkeypatch.setattr(personas_screen_module, "publish_persona_visual", publish)
        monkeypatch.setattr(screen, "_configure_persona_visual", AsyncMock())

        assert await screen._save_persona_visual_pack()

        assert len(calls) == 1
        assert invalidated == [("p-1", (_identity(), _identity(2)))]
        assert screen._persona_visual_authoring is None


async def test_failed_publication_keeps_draft_and_invalidates_nothing(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)
    invalidation = AsyncMock()

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        screen._session = SimpleNamespace(
            invalidate_persona_visual_identities=invalidation
        )
        state = screen._persona_visual_authoring
        assert state is not None
        state.dirty = True
        monkeypatch.setattr(
            personas_screen_module,
            "persona_visual_draft_publication_snapshot",
            lambda _draft: object(),
        )

        def fail(*_args, **_kwargs):
            raise PersonaVisualPublicationError("persona_visual_authority_changed")

        monkeypatch.setattr(personas_screen_module, "publish_persona_visual", fail)

        assert await screen._save_persona_visual_pack() is False
        assert screen._persona_visual_authoring is state
        invalidation.assert_not_awaited()


async def test_persona_authority_guard_rejects_revision_and_eligibility_changes(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        state = screen._persona_visual_authoring
        assert state is not None
        assert screen._persona_visual_authority_guard(state.snapshot) is True

        local_scope.local_service.get_persona_profile.return_value = {
            **PROFILE,
            "version": 3,
            "is_active": False,
            "deleted": False,
        }
        assert screen._persona_visual_authority_guard(state.snapshot) is False


async def test_cancel_signals_active_operation_and_discards_only_draft(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        state = screen._persona_visual_authoring
        assert state is not None
        state.dirty = True
        event = asyncio.Event()

        async def blocked():
            await event.wait()

        task = asyncio.create_task(blocked())
        screen._persona_visual_operation_task = task
        screen._persona_visual_operation_event = threading.Event()
        monkeypatch.setattr(screen, "_configure_persona_visual", AsyncMock())

        cancel = asyncio.create_task(screen._cancel_persona_visual_authoring())
        await asyncio.sleep(0)
        assert screen._persona_visual_operation_event.is_set()
        event.set()
        await cancel

        assert screen._persona_visual_authoring is None
        assert screen._configure_persona_visual.await_count == 1


async def test_stale_import_is_cleaned_and_cannot_repaint(
    monkeypatch, mock_app_instance, stub_characters, local_scope, tmp_path
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        original = screen._persona_visual_authoring
        assert original is not None
        review = PersonaVisualImportReview(
            schema_version="tldw.persona_visual_pack.v1",
            archive_sha256="b" * 64,
            pack_title="Imported",
            asset_count=0,
            state_count=0,
            draft=original.draft,
            cleanup_candidate="pvi1:" + "c" * 64 + ":.import-" + "d" * 32,
            _candidate_name=".import-" + "d" * 32,
            _candidate_identity=(1, 2),
        )
        started = threading.Event()
        release = threading.Event()

        def blocked_import(*_args, **_kwargs):
            started.set()
            release.wait(timeout=5)
            return review

        cleaned: list[PersonaVisualImportReview] = []
        monkeypatch.setattr(
            personas_screen_module, "import_persona_visual_pack", blocked_import
        )
        monkeypatch.setattr(
            personas_screen_module,
            "cleanup_persona_visual_import_review",
            lambda candidate, **_kwargs: cleaned.append(candidate) or True,
        )
        task = asyncio.create_task(
            screen._import_persona_visual_from_path("ignored.tldw-persona-vpack")
        )
        while not started.is_set():
            await asyncio.sleep(0)
        screen._persona_visual_generation += 1
        release.set()

        assert await task is False
        assert screen._persona_visual_authoring is original
        assert cleaned == [review]


async def test_outer_cancelled_import_drains_and_discards_review(
    monkeypatch, mock_app_instance, stub_characters, local_scope, tmp_path
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        original = screen._persona_visual_authoring
        assert original is not None
        imported = personas_screen_module.create_persona_visual_draft(
            persona_id="p-1", persona_revision=2, title="Cancelled import"
        )
        review = PersonaVisualImportReview(
            schema_version="tldw.persona_visual_pack.v1",
            archive_sha256="b" * 64,
            pack_title="Cancelled import",
            asset_count=0,
            state_count=0,
            draft=imported,
            cleanup_candidate="pvi1:" + "c" * 64 + ":.import-" + "d" * 32,
            _candidate_name=".import-" + "d" * 32,
            _candidate_identity=(1, 2),
        )
        started = threading.Event()
        release = threading.Event()

        def blocked_import(*_args, **_kwargs):
            started.set()
            release.wait(timeout=5)
            return review

        cleaned: list[PersonaVisualImportReview] = []
        monkeypatch.setattr(
            personas_screen_module, "import_persona_visual_pack", blocked_import
        )
        monkeypatch.setattr(
            personas_screen_module,
            "persona_visual_import_source_root",
            lambda *_args, **_kwargs: tmp_path,
        )
        monkeypatch.setattr(
            personas_screen_module,
            "cleanup_persona_visual_import_review",
            lambda candidate, **_kwargs: cleaned.append(candidate) or True,
        )

        operation = asyncio.create_task(
            screen._import_persona_visual_from_path("ignored.tldw-persona-vpack")
        )
        while not started.is_set():
            await asyncio.sleep(0)
        operation.cancel()
        await asyncio.sleep(0)
        assert not operation.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation

        assert screen._persona_visual_authoring is original
        assert original.draft.title != "Cancelled import"
        assert cleaned == [review]


async def test_unexpected_import_error_category_cannot_reach_notification(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)
    notifications: list[str] = []
    app.notify = lambda message, **_kwargs: notifications.append(str(message))
    private_marker = "/Users/alice/private/reactions.tldw-persona-vpack"

    class UnexpectedImportError(Exception):
        category = private_marker

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        monkeypatch.setattr(
            personas_screen_module,
            "import_persona_visual_pack",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(UnexpectedImportError()),
        )

        assert not await screen._import_persona_visual_from_path(
            "ignored.tldw-persona-vpack"
        )

        assert notifications
        assert all(private_marker not in item for item in notifications)


async def test_duplicate_save_is_rejected_while_first_publication_drains(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        state = screen._persona_visual_authoring
        assert state is not None
        state.dirty = True
        monkeypatch.setattr(
            personas_screen_module,
            "persona_visual_draft_publication_snapshot",
            lambda _draft: object(),
        )
        started = threading.Event()
        release = threading.Event()
        calls = 0

        def publish(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            started.set()
            release.wait(timeout=5)
            return PersonaVisualPublicationResult(None, _identity(), None)

        monkeypatch.setattr(personas_screen_module, "publish_persona_visual", publish)
        monkeypatch.setattr(screen, "_configure_persona_visual", AsyncMock())
        first = asyncio.create_task(screen._save_persona_visual_pack())
        while not started.is_set():
            await asyncio.sleep(0)

        assert await screen._save_persona_visual_pack() is False
        release.set()
        assert await first is True
        assert calls == 1


async def test_dirty_navigation_decline_preserves_draft_and_approval_discards_it(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        state = screen._persona_visual_authoring
        assert state is not None
        state.dirty = True
        assert screen._persona_visual_has_unsaved_authoring() is True
        continued: list[str] = []

        async def continuation():
            continued.append("continued")

        monkeypatch.setattr(
            screen, "_confirm_discard_unsaved", AsyncMock(return_value=False)
        )
        await screen._confirm_then_run(continuation)
        assert screen._persona_visual_authoring is state
        assert continued == []

        monkeypatch.setattr(
            screen, "_confirm_discard_unsaved", AsyncMock(return_value=True)
        )
        await screen._confirm_then_run(continuation)
        assert screen._persona_visual_authoring is None
        assert continued == ["continued"]


async def test_navigation_drain_defers_outer_cancellation_until_operation_settles(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        release = asyncio.Event()

        async def blocked_operation():
            await release.wait()

        operation = asyncio.create_task(blocked_operation())
        screen._persona_visual_operation_task = operation
        screen._persona_visual_operation_event = threading.Event()
        discard = AsyncMock()
        monkeypatch.setattr(screen, "_discard_persona_visual_authoring_async", discard)

        drain = asyncio.create_task(screen._drain_persona_visual_authoring())
        await asyncio.sleep(0)
        drain.cancel()
        await asyncio.sleep(0)

        assert not drain.done()
        discard.assert_not_awaited()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await drain
        discard.assert_awaited_once()


async def test_replacement_stages_private_asset_and_preview_loads_selected_only(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        assert await screen._stage_persona_visual_replacement("idle", _png())
        state = screen._persona_visual_authoring
        assert state is not None and state.workspace is not None
        first_workspace = state.workspace
        assert await screen._stage_persona_visual_replacement("idle", _png())
        assert state.workspace is not None and state.workspace != first_workspace
        assert not (
            first_workspace.profile_root / first_workspace.relative_root
        ).exists()
        assert state.dirty is True
        loads: list[str] = []
        original_loader = personas_screen_module.load_persona_visual_asset

        def observed_loader(root, *, storage_key, metadata, selected_frame=0):
            loads.append(metadata.asset_key)
            return original_loader(
                root,
                storage_key=storage_key,
                metadata=metadata,
                selected_frame=selected_frame,
            )

        monkeypatch.setattr(
            personas_screen_module, "load_persona_visual_asset", observed_loader
        )

        assert await screen._preview_persona_visual_state("idle")
        assert len(loads) == 1


async def test_replacement_cancellation_drains_workspace_creation_and_cleans_it(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        original_prepare = screen._prepare_persona_visual_workspace
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocked_prepare(state, profile_root):
            started.set()
            release.wait(timeout=5)
            try:
                return original_prepare(state, profile_root)
            finally:
                finished.set()

        monkeypatch.setattr(
            screen, "_prepare_persona_visual_workspace", blocked_prepare
        )
        replacement = asyncio.create_task(
            screen._stage_persona_visual_replacement("idle", _png())
        )
        while not started.is_set():
            await asyncio.sleep(0)
        replacement.cancel()
        await asyncio.sleep(0)

        assert not replacement.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await replacement
        assert finished.is_set()
        authoring_root = get_user_data_dir() / "persona_visual" / "authoring"
        assert not authoring_root.exists() or list(authoring_root.iterdir()) == []


async def test_invalid_replacement_leaves_no_private_workspace(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)

        assert not await screen._stage_persona_visual_replacement(
            "idle", b"not an image"
        )

        authoring_root = get_user_data_dir() / "persona_visual" / "authoring"
        assert not authoring_root.exists() or list(authoring_root.iterdir()) == []


async def test_preview_cancellation_drains_selected_decode_before_release(
    monkeypatch, mock_app_instance, stub_characters, local_scope
):
    monkeypatch.setattr(personas_screen_module, "PersonaVisualRepository", _Repository)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_editor(pilot)
        assert await screen._stage_persona_visual_replacement("idle", _png())
        original_loader = personas_screen_module.load_persona_visual_asset
        started = threading.Event()
        release = threading.Event()

        def blocked_loader(*args, **kwargs):
            started.set()
            release.wait(timeout=5)
            return original_loader(*args, **kwargs)

        monkeypatch.setattr(
            personas_screen_module, "load_persona_visual_asset", blocked_loader
        )
        preview = asyncio.create_task(screen._preview_persona_visual_state("idle"))
        while not started.is_set():
            await asyncio.sleep(0)
        preview.cancel()
        await asyncio.sleep(0)

        assert not preview.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await preview
