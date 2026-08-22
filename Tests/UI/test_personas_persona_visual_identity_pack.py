"""Persona editor Shared Visual Identity surface contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Static

import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler_module
import tldw_chatbook.UI.Screens.personas_screen as personas_screen_module
from Tests.UI.test_personas_workbench import (
    CHARACTERS,
    PROFILE,
    PersonasTestApp,
    _mounted,
)
from tldw_chatbook.Character_Chat.visual_identity import (
    CANONICAL_EXPRESSION_SLOTS,
    VisualIdentityPublicationError,
    VisualIdentityPublicationResult,
    VisualIdentityResolution,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    EditPersonaProfileRequested,
    VisualIdentityAssetMetadata,
    VisualIdentityPackMetadata,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_persona_visual_pack_widget import (
    PersonasPersonaVisualPackWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_visual_identity_pack_widget import (
    PersonasVisualIdentityPackWidget,
)


def _pack(*, bound: bool = True) -> VisualIdentityPackMetadata:
    return VisualIdentityPackMetadata(
        binding_id=5 if bound else 0,
        pack_id=10 if bound else 0,
        pack_version_id=20 if bound else 0,
        title="Shared Visual Identity reactions",
        source_kind="manual" if bound else "unbound",
        default_expression_key="neutral",
        assets=tuple(
            VisualIdentityAssetMetadata(
                asset_id=index if bound else -index,
                expression_key=key,
                original_label=key,
                display_label=key.replace("custom:", "").replace("-", " ").title(),
                content_type="image/png" if bound else "",
                is_animated=False,
            )
            for index, key in enumerate(CANONICAL_EXPRESSION_SLOTS, start=1)
        ),
    )


class _EditorApp(App):
    def compose(self) -> ComposeResult:
        yield PersonaProfileEditorWidget()


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


@pytest.fixture
def local_scope(mock_app_instance):
    record = {
        **PROFILE,
        "backend": "local",
        "version": 2,
        "is_active": True,
        "deleted": False,
    }
    local = Mock()
    local.get_persona_profile.return_value = dict(record)
    scope = Mock()
    scope.local_service = local
    scope.list_persona_profiles = AsyncMock(
        return_value={"items": [dict(record)], "total": 1}
    )
    scope.get_persona_profile = AsyncMock(return_value=dict(record))
    mock_app_instance.character_persona_scope_service = scope
    mock_app_instance.chachanotes_db = object()
    return scope


class _PersonaVisualRepository:
    def __init__(self, _db):
        pass

    def get_active_persona_pack(self, _persona_id):
        return None


class _SharedRepository:
    def __init__(self, _db):
        pass

    def get_active_actor_pack(self, actor_kind, actor_id):
        assert (actor_kind, actor_id) == ("persona", "p-1")
        return None


class _BoundSharedRepository(_SharedRepository):
    def get_active_actor_pack(self, actor_kind, actor_id):
        assert (actor_kind, actor_id) == ("persona", "p-1")
        return {
            "binding": {"id": 5},
            "pack": {"id": 10, "title": "Persona reactions", "source_kind": "manual"},
            "version": {"id": 20, "default_expression_key": "neutral"},
            "assets": [
                {
                    "id": 30,
                    "expression_key": "neutral",
                    "original_expression_key": "neutral",
                    "display_label": "Neutral",
                    "content_type": "image/png",
                    "is_animated": False,
                }
            ],
        }


def _resolution() -> VisualIdentityResolution:
    return VisualIdentityResolution(
        actor_kind="persona",
        actor_id="p-1",
        requested_expression_key="neutral",
        manual_expression_key="neutral",
        resolved_expression_key="neutral",
        pack_id=10,
        pack_version_id=20,
        asset_id=30,
        expression_id=None,
        storage_source="manual",
        storage_relpath=None,
        content_type="image/png",
        is_animated=False,
        resolution_source="manual",
        fallback_reason="",
        cache_identity=("persona", "p-1", "20", "30"),
        image_bytes=b"image",
    )


class _Candidate:
    old_binding_id = None
    old_pack_id = None
    old_version_id = None

    def __init__(self):
        self.replacements = []
        self.clears = []
        self.cancelled = False

    def stage_replacement(self, expression_key, data, *, source):
        self.replacements.append((expression_key, bytes(data), source))

    def stage_clear(self, expression_key):
        self.clears.append(expression_key)

    def cancel(self):
        self.cancelled = True


async def _open_persona_editor(pilot):
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


@pytest.mark.asyncio
async def test_persona_editor_keeps_shared_identity_and_persona_visual_as_separate_sections() -> (
    None
):
    app = _EditorApp()
    async with app.run_test(size=(100, 40)):
        editor = app.query_one(PersonaProfileEditorWidget)
        await editor.show_shared_visual_identity_pack(_pack())

        shared = editor.query_one(PersonasVisualIdentityPackWidget)
        operational = editor.query_one(PersonasPersonaVisualPackWidget)
        assert shared is not operational
        assert "Shared Visual Identity reactions" in str(
            shared.query_one("#personas-visual-identity-title", Static).renderable
        )
        assert "Persona Visual operational states" in str(
            editor.query_one("#personas-editor-persona-visual-title", Static).renderable
        )


@pytest.mark.asyncio
async def test_local_persona_shows_path_free_metadata_lazy_preview_and_manual_labels() -> (
    None
):
    app = _EditorApp()
    async with app.run_test(size=(100, 40)):
        editor = app.query_one(PersonaProfileEditorWidget)
        browser = await editor.show_shared_visual_identity_pack(_pack())
        assert browser is not None
        assert browser.selected_asset is not None
        assert browser.selected_asset.expression_key == "neutral"
        assert "/Users/" not in repr(browser.pack)
        preview = browser.query_one(
            "#personas-visual-identity-preview-image", Container
        )
        assert "Loading" in str(preview.children[0].renderable)


@pytest.mark.asyncio
async def test_unbound_local_persona_offers_create_replace_clear_save_cancel() -> None:
    app = _EditorApp()
    async with app.run_test(size=(100, 40)):
        editor = app.query_one(PersonaProfileEditorWidget)
        browser = await editor.show_shared_visual_identity_pack(_pack(bound=False))
        assert browser is not None
        assert (
            tuple(asset.expression_key for asset in browser.pack.assets)
            == CANONICAL_EXPRESSION_SLOTS
        )
        assert (
            browser.query_one("#personas-visual-identity-replace", Button).disabled
            is False
        )
        assert (
            browser.query_one("#personas-visual-identity-clear", Button).disabled
            is False
        )
        browser.set_staged_change("neutral", "replace")
        assert (
            browser.query_one("#personas-visual-identity-save", Button).disabled
            is False
        )
        assert (
            browser.query_one("#personas-visual-identity-cancel", Button).display
            is True
        )


@pytest.mark.asyncio
async def test_server_persona_disables_shared_identity_with_save_local_copy_first() -> (
    None
):
    app = _EditorApp()
    async with app.run_test(size=(100, 40)):
        editor = app.query_one(PersonaProfileEditorWidget)
        editor.load_persona({"id": "server-p"}, runtime_source="server")

        host = editor.query_one(
            "#personas-editor-shared-visual-identity-host", Container
        )
        assert "Save a local copy first" in str(host.children[0].renderable)
        assert not host.query(PersonasVisualIdentityPackWidget)


@pytest.mark.asyncio
async def test_workbench_loads_canonical_unbound_shared_identity_for_local_persona(
    monkeypatch, mock_app_instance, stub_characters, local_scope
) -> None:
    monkeypatch.setattr(
        personas_screen_module, "PersonaVisualRepository", _PersonaVisualRepository
    )
    monkeypatch.setattr(
        personas_screen_module, "VisualIdentityRepository", _SharedRepository
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        await _open_persona_editor(pilot)
        browser = pilot.app.screen.query_one(PersonasVisualIdentityPackWidget)

        assert browser.pack is not None
        assert browser.pack.source_kind == "unbound"
        assert (
            tuple(asset.expression_key for asset in browser.pack.assets)
            == CANONICAL_EXPRESSION_SLOTS
        )


@pytest.mark.asyncio
async def test_workbench_discards_shared_identity_when_persona_revision_changes(
    monkeypatch, mock_app_instance, stub_characters, local_scope
) -> None:
    monkeypatch.setattr(
        personas_screen_module, "PersonaVisualRepository", _PersonaVisualRepository
    )
    monkeypatch.setattr(
        personas_screen_module, "VisualIdentityRepository", _SharedRepository
    )
    calls = 0
    original = personas_screen_module.capture_local_persona_visual_identity

    def changing_capture(service, persona_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            service.get_persona_profile.return_value = {
                **PROFILE,
                "backend": "local",
                "version": 3,
                "is_active": True,
                "deleted": False,
            }
        return original(service, persona_id)

    monkeypatch.setattr(
        personas_screen_module,
        "capture_local_persona_visual_identity",
        changing_capture,
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        await _open_persona_editor(pilot)
        host = pilot.app.screen.query_one(
            "#personas-editor-shared-visual-identity-host", Container
        )

        assert not host.query(PersonasVisualIdentityPackWidget)
        assert "unavailable" in str(host.children[0].renderable).lower()


@pytest.mark.asyncio
async def test_persona_preview_stale_after_resolve_does_not_paint(
    monkeypatch, mock_app_instance, stub_characters, local_scope
) -> None:
    monkeypatch.setattr(
        personas_screen_module, "PersonaVisualRepository", _PersonaVisualRepository
    )
    monkeypatch.setattr(
        personas_screen_module, "VisualIdentityRepository", _BoundSharedRepository
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_persona_editor(pilot)
        browser = screen.query_one(PersonasVisualIdentityPackWidget)
        snapshot = screen._persona_shared_visual_identity_author_snapshot()
        assert snapshot is not None
        asset = browser.pack.assets[0]

        def stale_resolve(*_args, **_kwargs):
            local_scope.local_service.get_persona_profile.return_value = {
                **PROFILE,
                "backend": "local",
                "version": 3,
                "is_active": True,
                "deleted": False,
            }
            return _resolution()

        monkeypatch.setattr(
            personas_screen_module, "resolve_persona_visual_identity", stale_resolve
        )
        screen._avatar_render_cache = SimpleNamespace(
            prepare=lambda *_args: True,
            get_pil=lambda *_args: None,
        )
        unavailable = Mock(wraps=browser.set_preview_unavailable)
        monkeypatch.setattr(browser, "set_preview_unavailable", unavailable)

        await screen._render_persona_shared_visual_identity_preview(snapshot, asset)

        unavailable.assert_called_with(asset_id=asset.asset_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((100, 40), (80, 24)))
async def test_normal_and_80x24_compact_layout_paints_labelled_focusable_actions(
    size: tuple[int, int],
) -> None:
    app = _EditorApp()
    async with app.run_test(size=size):
        editor = app.query_one(PersonaProfileEditorWidget)
        browser = await editor.show_shared_visual_identity_pack(_pack(bound=False))
        assert browser is not None
        labels = {
            str(button.label)
            for button in browser.query("#personas-visual-identity-actions Button")
        }
        assert {
            "Replace…",
            "Generate",
            "Generate All",
            "Clear",
            "Save",
            "Cancel",
        } <= labels
        assert all(button.can_focus for button in browser.query(Button))


@pytest.mark.asyncio
async def test_persona_replace_clear_save_and_cancel_own_one_unpublished_candidate(
    monkeypatch, mock_app_instance, stub_characters, local_scope
) -> None:
    monkeypatch.setattr(
        personas_screen_module, "PersonaVisualRepository", _PersonaVisualRepository
    )
    monkeypatch.setattr(
        personas_screen_module, "VisualIdentityRepository", _SharedRepository
    )
    candidate = _Candidate()
    monkeypatch.setattr(
        personas_screen_module,
        "create_visual_identity_candidate",
        lambda *_args, **_kwargs: candidate,
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_persona_editor(pilot)
        browser = screen.query_one(PersonasVisualIdentityPackWidget)
        asset = browser.pack.assets[0]

        assert await screen._stage_persona_shared_visual_identity_replacement(
            asset, b"image"
        )
        assert await screen._stage_persona_shared_visual_identity_clear(asset)
        assert candidate.replacements == [("neutral", b"image", "upload")]
        assert candidate.clears == ["neutral"]
        assert screen._persona_shared_visual_identity_has_unsaved_authoring()

        screen._request_persona_shared_visual_identity_cancel()
        assert candidate.cancelled is True
        assert screen._persona_shared_visual_identity_authoring is None


@pytest.mark.asyncio
async def test_failed_persona_publication_preserves_draft_and_active_pack(
    monkeypatch, mock_app_instance, stub_characters, local_scope
) -> None:
    monkeypatch.setattr(
        personas_screen_module, "PersonaVisualRepository", _PersonaVisualRepository
    )
    monkeypatch.setattr(
        personas_screen_module, "VisualIdentityRepository", _SharedRepository
    )
    candidate = _Candidate()
    monkeypatch.setattr(
        personas_screen_module,
        "create_visual_identity_candidate",
        lambda *_args, **_kwargs: candidate,
    )

    async def failed_publish(*_args, **_kwargs):
        return personas_screen_module._DrainedTaskResult(
            error=VisualIdentityPublicationError("visual_identity_actor_changed")
        )

    monkeypatch.setattr(personas_screen_module, "_drain_to_thread", failed_publish)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_persona_editor(pilot)
        browser = screen.query_one(PersonasVisualIdentityPackWidget)
        authoritative_pack = browser.pack
        assert await screen._stage_persona_shared_visual_identity_replacement(
            browser.pack.assets[0], b"image"
        )

        assert await screen._save_persona_shared_visual_identity_pack() is False
        assert screen._persona_shared_visual_identity_authoring is not None
        assert browser.pack == authoritative_pack


@pytest.mark.asyncio
async def test_successful_persona_save_invalidates_only_exact_actor_result(
    monkeypatch, mock_app_instance, stub_characters, local_scope
) -> None:
    monkeypatch.setattr(
        personas_screen_module, "PersonaVisualRepository", _PersonaVisualRepository
    )
    monkeypatch.setattr(
        personas_screen_module, "VisualIdentityRepository", _SharedRepository
    )
    candidate = _Candidate()
    monkeypatch.setattr(
        personas_screen_module,
        "create_visual_identity_candidate",
        lambda *_args, **_kwargs: candidate,
    )
    result = VisualIdentityPublicationResult(
        actor_kind="persona",
        actor_id="p-1",
        old_pack_id=None,
        old_version_id=None,
        new_pack_id=10,
        new_version_id=20,
        version_directory=Path("unused"),
    )

    async def successful_publish(*_args, **_kwargs):
        return personas_screen_module._DrainedTaskResult(completed=True, value=result)

    monkeypatch.setattr(personas_screen_module, "_drain_to_thread", successful_publish)
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_persona_editor(pilot)
        browser = screen.query_one(PersonasVisualIdentityPackWidget)
        assert await screen._stage_persona_shared_visual_identity_replacement(
            browser.pack.assets[0], b"image"
        )
        invalidate = AsyncMock()
        refresh = AsyncMock()
        screen._invalidate_visual_identity_publication = invalidate
        screen._configure_persona_shared_visual_identity = refresh

        assert await screen._save_persona_shared_visual_identity_pack() is True
        invalidate.assert_awaited_once_with(result)
        refresh.assert_awaited_once()
        assert screen._persona_shared_visual_identity_authoring is None


@pytest.mark.asyncio
async def test_dirty_navigation_decline_preserves_persona_reaction_draft_and_accept_drains(
    monkeypatch, mock_app_instance, stub_characters, local_scope
) -> None:
    monkeypatch.setattr(
        personas_screen_module, "PersonaVisualRepository", _PersonaVisualRepository
    )
    monkeypatch.setattr(
        personas_screen_module, "VisualIdentityRepository", _SharedRepository
    )
    candidate = _Candidate()
    monkeypatch.setattr(
        personas_screen_module,
        "create_visual_identity_candidate",
        lambda *_args, **_kwargs: candidate,
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _open_persona_editor(pilot)
        browser = screen.query_one(PersonasVisualIdentityPackWidget)
        assert await screen._stage_persona_shared_visual_identity_replacement(
            browser.pack.assets[0], b"image"
        )
        continuation = AsyncMock()

        screen._confirm_discard_unsaved = AsyncMock(return_value=False)
        await screen._confirm_then_run(continuation)
        continuation.assert_not_awaited()
        assert screen._persona_shared_visual_identity_authoring is not None
        assert candidate.cancelled is False

        screen._confirm_discard_unsaved = AsyncMock(return_value=True)
        await screen._confirm_then_run(continuation)
        continuation.assert_awaited_once()
        assert candidate.cancelled is True
        assert screen._persona_shared_visual_identity_authoring is None
