"""TASK-32108 disposable, source-bound headless qualification captures."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from textual.widgets import Button, Select, Switch

from Tests.UI.test_console_navigation_decisions import _until
from Tests.UI.test_console_screen_reuse import (
    _boot_settled,
    _press_until_screen,
    _scratch_env,
)


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.parametrize("profile_kind", ["fresh", "upgrade"])
async def test_buddy_v1_rendered_profile_journey(monkeypatch, tmp_path, profile_kind):
    """Render the production app with real bundled art and local SQLite."""
    _scratch_env(monkeypatch, tmp_path)
    monkeypatch.delenv("NO_COLOR", raising=False)
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.config import get_chachanotes_db_path
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
    from tldw_chatbook.UI.Navigation.buddy_management import get_buddy_management
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
        BuddyManagementModal,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
        PersonaBuddyWidget,
    )

    assert os.environ["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    database_path = get_chachanotes_db_path()
    assert database_path.is_relative_to(tmp_path)
    assert not database_path.exists()
    legacy = None
    if profile_kind == "upgrade":
        from Tests.Persona_Visual.test_persona_visual_repository import _activate

        with monkeypatch.context() as old_schema:
            old_schema.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 69)
            db = CharactersRAGDB(database_path, "qualification-upgrade")
            legacy = _activate(PersonaVisualRepository(db))
            db.close_connection()

    output = Path(
        os.environ.get("TLDW_BUDDY_QUALIFICATION_OUTPUT", tmp_path / "captures")
    )
    output.mkdir(exist_ok=True)
    evidence = {"profile": profile_kind, "headless": True, "frames": {}}
    app = TldwCli()
    async with app.run_test(size=(150, 45)) as pilot:
        await _boot_settled(app, pilot)
        assert (
            app.chachanotes_db.execute_query(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()[0]
            == 70
        )
        if legacy is not None:
            assert (
                PersonaVisualRepository(app.chachanotes_db).get_active_persona_pack(
                    legacy.identity.persona_id
                )
                == legacy
            )
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        controller = app.screen._ensure_console_chat_controller()
        target = controller.store.ensure_session()
        persona_before = await asyncio.to_thread(
            app.local_character_persona_service.list_persona_profiles
        )
        manager = get_buddy_management(app)
        for mode in ("static", "dynamic"):
            manager.request_open()
            await _until(lambda: isinstance(app.screen, BuddyManagementModal))
            await pilot.pause()
            buddies = await asyncio.to_thread(manager.library.list_buddies)
            assert buddies
            app.screen.query_one("#buddy-artwork", Select).value = buddies[0].id
            app.screen.query_one(
                "#buddy-follow", Select
            ).value = f"conversation:{target.id}"
            app.screen.query_one("#buddy-motion", Select).value = mode
            app.screen.query_one("#buddy-enabled", Switch).value = True
            await pilot.pause()
            (output / f"{profile_kind}-{mode}-management.svg").write_text(
                app.export_screenshot()
            )
            app.screen.query_one("#buddy-apply", Button).press()
            await _until(lambda: not isinstance(app.screen, BuddyManagementModal))
            await pilot.pause()
            assert manager.preferences.animated is (mode == "dynamic")
            await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
            await _until(
                lambda: bool(list(app.screen.query(PersonaBuddyWidget))),
                detail=lambda: (
                    repr(manager.controller.snapshot())
                    + repr(app._persona_buddy_overlay.__dict__)
                ),
            )
            widget = app.screen.query_one(PersonaBuddyWidget)
            await _until(lambda widget=widget: widget._accepted_render is not None)
            observed = set()
            sequence = []
            for _ in range(40):
                await pilot.pause(0.1)
                observed.add(widget.frame_index)
                sequence.append(widget.frame_index)
            visual = widget._accepted_render.visual
            evidence["frames"][mode] = {
                "frame_count": len(visual.frames),
                "observed_indices": sorted(observed),
                "sequence": sequence,
                "animate": visual.animate,
                "loop": visual.loop,
                "buddy_owner": visual.graph_identity.buddy_id,
                "persona_owner": visual.persona_id,
            }
            assert visual.persona_id is None
            assert (len(observed) > 1) if mode == "dynamic" else observed == {0}
            (output / f"{profile_kind}-{mode}-home.svg").write_text(
                app.export_screenshot()
            )
        assert (
            await asyncio.to_thread(
                app.local_character_persona_service.list_persona_profiles
            )
            == persona_before
        )
        assert manager.preferences.binding.target_id == target.id
        manager.request_interaction()
        await _until(lambda: type(app.screen).__name__ == "BuddyConversationModal")
        await pilot.pause()
        assert app.screen.binding.target_id == target.id
        (output / f"{profile_kind}-conversation.svg").write_text(
            app.export_screenshot()
        )
        app.screen.query_one("#buddy-close", Button).press()
        await _until(lambda: type(app.screen).__name__ == "HomeScreen")
        workspace = await asyncio.to_thread(
            app.workspace_registry_service.create_workspace,
            workspace_id="qualification-workspace",
            name="Qualification workspace",
            assistant_defaults=None,
        )
        manager.request_open()
        await _until(lambda: isinstance(app.screen, BuddyManagementModal))
        await pilot.pause()
        app.screen.query_one(
            "#buddy-follow", Select
        ).value = f"workspace:{workspace.workspace_id}"
        app.screen.query_one("#buddy-persona", Select).value = "#none"
        await pilot.pause()
        (output / f"{profile_kind}-workspace-management.svg").write_text(
            app.export_screenshot()
        )
        app.screen.query_one("#buddy-apply", Button).press()
        await _until(lambda: not isinstance(app.screen, BuddyManagementModal))
        assert manager.preferences.binding.kind == "workspace"
        assert manager.preferences.binding.target_id == workspace.workspace_id
        assert (
            app.workspace_registry_service.get_workspace(
                workspace.workspace_id
            ).assistant_defaults
            is None
        )
        manager.request_interaction()
        await _until(lambda: type(app.screen).__name__ == "BuddyWorkspaceModal")
        await _until(lambda: app.screen._loaded and app.screen._fresh)
        await pilot.pause()
        (output / f"{profile_kind}-workspace-inbox.svg").write_text(
            app.export_screenshot()
        )
        app.screen.query_one("#buddy-inbox-close", Button).press()
        await _until(lambda: type(app.screen).__name__ == "HomeScreen")
        assert controller.store.active_session_id == target.id
        evidence["workspace_explicit_none"] = True
        evidence["conversation_id"] = target.id
        evidence["schema_version"] = 70
        evidence["persona_count_unchanged"] = True
    evidence["source_commit"] = (
        await asyncio.to_thread(
            subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
        )
    ).strip()
    evidence["captures"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.glob(f"{profile_kind}-*.svg")
    }
    (output / f"{profile_kind}-evidence.json").write_text(
        json.dumps(evidence, indent=2)
    )
