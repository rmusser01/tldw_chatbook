from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass

import pytest
from textual.app import App
from textual.widgets import Button, Select, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import LibraryHarness
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterRepairCandidate,
    CharacterRepairPage,
    CharacterRepairRequest,
    CharacterRepairResult,
    ResolvedLocalCharacterKey,
    UnresolvedConversationKey,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Library_Modules.library_character_repair_controller import (
    LibraryCharacterRepairController,
    LibraryCharacterRepairDialog,
)
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    LibraryCharacterRepairContext,
    RoleplayReturnTarget,
    serialize_library_character_repair_context,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

CONTEXT = LibraryCharacterRepairContext(
    unresolved=UnresolvedConversationKey("authority-A", "conversation-X"),
    expected_conversation_version=3,
    historical_display_snapshot="Historical Ada",
    return_target=RoleplayReturnTarget.personas_conversations(),
)


@dataclass
class _Service:
    candidates: tuple[CharacterRepairCandidate, ...]
    result: CharacterRepairResult = CharacterRepairResult.APPLIED
    request: CharacterRepairRequest | None = None

    def repair_candidates(self, _key, *, offset=0, limit=20):
        candidates = self.candidates[offset : offset + limit]
        next_offset = offset + len(candidates)
        return CharacterRepairPage(
            candidates,
            len(self.candidates),
            next_offset if next_offset < len(self.candidates) else None,
        )

    def repair(self, request: CharacterRepairRequest) -> CharacterRepairResult:
        self.request = request
        return self.result


def _controller(service: _Service):
    invalidations: list[str] = []
    returns: list[RoleplayReturnTarget] = []
    refresh_focus: list[bool] = []
    controller = LibraryCharacterRepairController(
        service=service,
        invalidate_keyword=lambda: invalidations.append("keyword"),
        invalidate_semantic=lambda: invalidations.append("semantic"),
        return_to_anchor=returns.append,
        focus_refresh=lambda: refresh_focus.append(True),
    )
    return controller, invalidations, returns, refresh_focus


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((52, 20), (120, 50)))
async def test_repair_continuation_reaches_every_candidate_and_restarts_after_mutation(
    tmp_path,
    size,
) -> None:
    """Page one must not strand later cards; source changes invalidate offsets."""
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationNavigationService,
    )

    db = CharactersRAGDB(tmp_path / "repair-pages.sqlite", client_id="repair-pages")
    try:
        ids = {db.add_character_card({"name": f"Candidate {i:02d}"}) for i in range(45)}
        with db.transaction() as connection:
            ids = {
                row[0]
                for row in connection.execute(
                    "SELECT id FROM character_cards WHERE deleted = 0"
                )
            }
        authority = db.get_local_authority_id()
        db.add_conversation(
            {
                "id": "broken",
                "title": "Broken link",
                "assistant_kind": "character",
                "character_id": min(ids),
                "assistant_id": str(min(ids)),
                "assistant_authority_id": authority,
            }
        )
        with db.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE conversations SET assistant_id = ?, assistant_authority_id = NULL WHERE id = ?",
                ("Historical card", "broken"),
            )
        original = db.get_conversation_by_id("broken")
        context = LibraryCharacterRepairContext(
            UnresolvedConversationKey(authority, "broken"),
            original["version"],
            "Historical card",
            RoleplayReturnTarget.personas_conversations(),
        )
        controller, invalidations, returns, _ = _controller(
            CharacterConversationNavigationService(db)
        )
        # The production Library provides the active source revision as well.
        controller._source_revision = db.get_character_conversation_search_revision
        app = App()
        async with app.run_test(size=size) as pilot:
            dialog = LibraryCharacterRepairDialog(controller, context)
            await app.push_screen(dialog)
            await pilot.pause()
            assert len(controller.candidates) == 20
            seen = {item.key.character_id for item in controller.candidates}
            next_button = dialog.query_one("#library-character-repair-next", Button)
            assert not next_button.disabled
            await pilot.click(next_button)
            await pilot.pause()
            seen.update(item.key.character_id for item in controller.candidates)
            assert len(seen) == 40
            assert (
                dialog.query_one("#library-character-repair-candidate", Select).value
                is Select.NULL
            )
            next_button.press()
            await pilot.pause()
            seen.update(item.key.character_id for item in controller.candidates)
            assert seen == ids
            assert next_button.disabled

            await pilot.click("#library-character-repair-refresh")
            await pilot.pause()
            new_id = db.add_character_card({"name": "AAAA inserted"})
            await pilot.click(next_button)
            await pilot.pause()
            assert controller.candidates[0].key.character_id == new_id
            assert len(controller.candidates) == 20
            assert (
                "changed"
                in str(
                    dialog.query_one(
                        "#library-character-repair-status", Static
                    ).renderable
                ).lower()
            )
            assert db.get_conversation_by_id("broken")["version"] == original["version"]
            assert not invalidations and not returns
    finally:
        db.close_connection()


def test_candidates_are_same_authority_and_name_match_is_not_preselected() -> None:
    same = CharacterRepairCandidate(
        ResolvedLocalCharacterKey("authority-A", 7), "Historical Ada", 1
    )
    foreign = CharacterRepairCandidate(
        ResolvedLocalCharacterKey("authority-B", 8), "Historical Ada", 1
    )
    controller, *_ = _controller(_Service((same, foreign)))

    candidates = controller.accept(CONTEXT)

    assert candidates == (same,)
    assert controller.selected_candidate is None
    assert controller.historical_identity_copy == "Historical Ada"


def test_old_and_selected_identity_are_visible_and_confirmation_is_required() -> None:
    candidate = CharacterRepairCandidate(
        ResolvedLocalCharacterKey("authority-A", 7), "Current Ada", 1
    )
    service = _Service((candidate,))
    controller, *_ = _controller(service)
    controller.accept(CONTEXT)
    controller.select(candidate.key)

    assert controller.identity_comparison == (
        "Historical Ada",
        "Current Ada · local character 7",
    )
    assert controller.apply_confirmed() is None
    assert service.request is None
    assert controller.request_confirmation()
    assert controller.apply_confirmed() is CharacterRepairResult.APPLIED


def test_stale_cas_preserves_context_and_focuses_refresh() -> None:
    candidate = CharacterRepairCandidate(
        ResolvedLocalCharacterKey("authority-A", 7), "Current Ada", 1
    )
    service = _Service((candidate,), CharacterRepairResult.STALE_VERSION)
    controller, invalidations, returns, refresh_focus = _controller(service)
    controller.accept(CONTEXT)
    controller.select(candidate.key)
    controller.request_confirmation()

    result = controller.apply_confirmed()

    assert result is CharacterRepairResult.STALE_VERSION
    assert controller.context == CONTEXT
    assert controller.status_copy == "Conversation changed. Refresh before repairing."
    assert refresh_focus == [True]
    assert invalidations == [] and returns == []


def test_success_invalidates_indexes_and_returns_to_source_anchor() -> None:
    candidate = CharacterRepairCandidate(
        ResolvedLocalCharacterKey("authority-A", 7), "Current Ada", 1
    )
    service = _Service((candidate,))
    controller, invalidations, returns, refresh_focus = _controller(service)
    controller.accept(CONTEXT)
    controller.select(candidate.key)
    controller.request_confirmation()

    result = controller.apply_confirmed()

    assert result is CharacterRepairResult.APPLIED
    assert service.request == CharacterRepairRequest(
        unresolved=CONTEXT.unresolved,
        replacement=candidate.key,
        expected_conversation_version=3,
    )
    assert invalidations == ["keyword", "semantic"]
    assert returns == [CONTEXT.return_target]
    assert refresh_focus == []


def test_cancel_keeps_navigation_context_and_conversation_unchanged() -> None:
    candidate = CharacterRepairCandidate(
        ResolvedLocalCharacterKey("authority-A", 7), "Current Ada", 1
    )
    service = _Service((candidate,))
    controller, invalidations, returns, _ = _controller(service)
    controller.accept(CONTEXT)
    controller.select(candidate.key)

    controller.cancel_confirmation()

    assert controller.context == CONTEXT
    assert service.request is None
    assert invalidations == [] and returns == []


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", (Select.NULL, "8"), ids=("blank", "other"))
async def test_candidate_change_revokes_armed_repair_in_real_dialog(
    replacement,
) -> None:
    service = _Service(
        tuple(
            CharacterRepairCandidate(
                ResolvedLocalCharacterKey("authority-A", key), name, 1
            )
            for key, name in ((7, "Current Ada"), (8, "Other Ada"))
        )
    )
    controller, *_ = _controller(service)
    app = App()
    async with app.run_test() as pilot:
        await app.push_screen(LibraryCharacterRepairDialog(controller, CONTEXT))
        await app.workers.wait_for_complete()
        await pilot.pause()
        select = app.screen.query_one("#library-character-repair-candidate", Select)
        apply = app.screen.query_one("#library-character-repair-apply", Button)
        select.value = "7"
        await pilot.pause()
        apply.press()
        await pilot.pause()
        assert str(apply.label) == "Confirm repair"
        select.value = replacement
        await pilot.pause()
        assert str(apply.label) == "Repair"
        assert not controller._confirmation_requested
        if replacement is Select.NULL:
            assert controller.selected_candidate is None
            apply.press()
            await pilot.pause()
            assert service.request is None
            select.value = "7"
            await pilot.pause()
        apply.press()
        await pilot.pause()
        assert str(apply.label) == "Confirm repair"
        assert service.request is None
        apply.press()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert service.request is not None
        assert service.request.replacement.character_id == (
            8 if replacement == "8" else 7
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((52, 20), (120, 50)))
async def test_real_textual_pilot_shows_explicit_repair_and_stale_refresh(
    size,
) -> None:
    """The Library-owned CAS dialog remains complete at both required sizes."""

    candidate = CharacterRepairCandidate(
        ResolvedLocalCharacterKey("authority-A", 7), "Current Ada", 1
    )
    controller, *_ = _controller(
        _Service((candidate,), CharacterRepairResult.STALE_VERSION)
    )
    app = App()
    async with app.run_test(size=size) as pilot:
        await app.push_screen(LibraryCharacterRepairDialog(controller, CONTEXT))
        await pilot.pause()
        select = app.screen.query_one("#library-character-repair-candidate", Select)
        assert select.value is Select.NULL
        select.value = "7"
        await pilot.pause()
        status = app.screen.query_one("#library-character-repair-status", Static)
        assert "Historical Ada" in str(status.renderable)
        assert "Current Ada" in str(status.renderable)

        await pilot.click("#library-character-repair-apply")
        await pilot.pause()
        apply_button = app.screen.query_one("#library-character-repair-apply", Button)
        assert str(apply_button.label) == "Confirm repair"
        apply_button.press()
        await pilot.pause(0.05)
        refresh = app.screen.query_one("#library-character-repair-refresh", Button)
        assert app.focused is refresh, (
            controller.status_copy,
            str(status.renderable),
            str(apply_button.label),
        )
        assert "Refresh" in str(status.renderable)
        if qa_root := os.environ.get("TASK_31243_QA_DIR"):
            app.save_screenshot(
                filename=f"library-repair-stale-{size[0]}x{size[1]}.svg",
                path=qa_root,
            )


@pytest.mark.asyncio
async def test_mounted_library_refreshes_real_cas_version_then_retry_succeeds(
    tmp_path,
) -> None:
    """Mounted Library owns refresh and retry through the real SQLite CAS path."""

    db = CharactersRAGDB(tmp_path / "library-repair.sqlite", client_id="repair")
    try:
        replacement_id = db.add_character_card({"name": "Current Ada"})
        authority = db.get_local_authority_id()
        assert replacement_id
        assert db.add_conversation(
            {
                "id": "unresolved",
                "character_id": replacement_id,
                "assistant_kind": "character",
                "assistant_id": str(replacement_id),
                "assistant_authority_id": authority,
                "title": "Repair me",
            }
        )
        with db.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE conversations SET assistant_id = ?, "
                "assistant_authority_id = NULL WHERE id = ?",
                ("Historical Ada", "unresolved"),
            )
        original = db.get_conversation_by_id("unresolved")
        assert original
        context = LibraryCharacterRepairContext(
            unresolved=UnresolvedConversationKey(authority, "unresolved"),
            expected_conversation_version=original["version"],
            historical_display_snapshot="Historical Ada",
            return_target=RoleplayReturnTarget.personas_conversations(),
        )
        with db.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE conversations SET version = version + 1 WHERE id = ?",
                ("unresolved",),
            )

        app_instance = _build_test_app()
        app_instance.chachanotes_db = db
        screen = LibraryScreen(app_instance)
        from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_CHARACTER_REPAIR

        screen.apply_navigation_context(
            {
                LIBRARY_NAV_CONTEXT_CHARACTER_REPAIR: serialize_library_character_repair_context(
                    context
                )
            }
        )
        harness = LibraryHarness(app_instance, screen=screen)
        async with harness.run_test(size=(120, 50)) as pilot:
            for _ in range(100):
                await pilot.pause(0.05)
                if isinstance(harness.screen, LibraryCharacterRepairDialog):
                    break
            dialog = harness.screen
            assert isinstance(dialog, LibraryCharacterRepairDialog)
            controller = screen._navigation_controller.repair_controller
            assert controller is not None
            select = dialog.query_one("#library-character-repair-candidate", Select)
            assert not select.disabled
            assert controller.context is not None
            assert (
                controller.context.expected_conversation_version
                == original["version"] + 1
            )
            select.value = str(replacement_id)
            await pilot.pause()
            apply = dialog.query_one("#library-character-repair-apply", Button)
            apply.press()
            await pilot.pause()
            assert str(apply.label) == "Confirm repair"
            apply.press()
            await dialog.workers.wait_for_complete()
            await pilot.pause()

        assert screen._navigation_controller.pending_repair_context is None
        assert screen._navigation_controller.keyword_generation == 1
        assert screen._navigation_controller.semantic_generation == 1
        repaired = db.get_conversation_by_id("unresolved")
        assert repaired and repaired["character_id"] == replacement_id
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_delayed_cas_disables_cancel_and_unmount_blocks_ui_side_effects() -> None:
    candidate = CharacterRepairCandidate(
        ResolvedLocalCharacterKey("authority-A", 7), "Current Ada", 1
    )
    started = threading.Event()
    release = threading.Event()

    class _DelayedService(_Service):
        def repair(self, request):
            self.request = request
            started.set()
            assert release.wait(5)
            return CharacterRepairResult.APPLIED

    controller, invalidations, returns, _ = _controller(_DelayedService((candidate,)))
    app = App()
    async with app.run_test(size=(120, 50)) as pilot:
        await app.push_screen(LibraryCharacterRepairDialog(controller, CONTEXT))
        await pilot.pause()
        app.screen.query_one("#library-character-repair-candidate", Select).value = "7"
        await pilot.pause()
        apply = app.screen.query_one("#library-character-repair-apply", Button)
        apply.press()
        await pilot.pause()
        apply.press()
        assert await asyncio.to_thread(started.wait, 5)
        assert app.screen.query_one("#library-character-repair-cancel", Button).disabled
        await app.pop_screen()
        release.set()
        await pilot.pause(0.1)

    assert invalidations == []
    assert returns == []


@pytest.mark.asyncio
async def test_cas_exception_restores_retry_controls() -> None:
    candidate = CharacterRepairCandidate(
        ResolvedLocalCharacterKey("authority-A", 7), "Current Ada", 1
    )

    class _RaisingService(_Service):
        def repair(self, request):
            raise RuntimeError("database unavailable")

    controller, *_ = _controller(_RaisingService((candidate,)))
    app = App()
    async with app.run_test(size=(120, 50)) as pilot:
        await app.push_screen(LibraryCharacterRepairDialog(controller, CONTEXT))
        await pilot.pause()
        dialog = app.screen
        dialog.query_one("#library-character-repair-candidate", Select).value = "7"
        await pilot.pause()
        apply = dialog.query_one("#library-character-repair-apply", Button)
        apply.press()
        await pilot.pause()
        apply.press()
        await dialog.workers.wait_for_complete()
        await pilot.pause()

        status = str(
            dialog.query_one("#library-character-repair-status", Static).renderable
        )
        assert status == "Repair failed. Retry or cancel."
        assert str(apply.label) == "Repair"
        assert not apply.disabled
        assert not dialog.query_one("#library-character-repair-cancel", Button).disabled
        assert dialog._operation_token is None
