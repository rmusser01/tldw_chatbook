"""TASK-32108 disposable, source-bound headless qualification captures."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import pytest
from textual.widgets import Button, Select, Switch

from Tests.UI.test_console_navigation_decisions import _until
from Tests.UI.test_console_screen_reuse import (
    _boot_settled,
    _press_until_screen,
    _scratch_env,
)

_ARTIFACT_ROOT = Path(tempfile.gettempdir()) / "chatbook-buddy-v1-qualification-32108"


def _capture_output(tmp_path: Path) -> Path:
    """Resolve captures inside the test profile or the disposable export root."""
    from tldw_chatbook.Utils.path_validation import validate_path

    configured = os.environ.get("TLDW_BUDDY_QUALIFICATION_OUTPUT")
    return validate_path(
        configured if configured is not None else tmp_path / "captures",
        _ARTIFACT_ROOT if configured is not None else tmp_path,
        redact_paths=True,
    )


def _write_artifact(output: Path, name: str, content: str) -> str:
    """Write one validated artifact leaf and return its generated filename."""
    from tldw_chatbook.Utils.path_validation import validate_path

    destination = validate_path(name, output, redact_paths=True)
    destination.write_text(content, encoding="utf-8")
    return name


def _capture_hashes(output: Path, names: list[str]) -> dict[str, str]:
    """Hash only this run's generated captures, revalidating each read target."""
    from tldw_chatbook.Utils.path_validation import validate_path

    return {
        name: hashlib.sha256(
            validate_path(name, output, redact_paths=True).read_bytes()
        ).hexdigest()
        for name in names
    }


def _prepare_upgrade(monkeypatch: pytest.MonkeyPatch, database_path: Path):
    """Create a schema69 predecessor and release its handle on every exit."""
    from Tests.Persona_Visual.test_persona_visual_repository import _activate
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository

    with monkeypatch.context() as old_schema:
        old_schema.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 69)
        db = CharactersRAGDB(database_path, "qualification-upgrade")
        try:
            with db.transaction() as cursor:
                assert (
                    cursor.execute(
                        "SELECT version FROM db_schema_version WHERE schema_name=?",
                        ("rag_char_chat_schema",),
                    ).fetchone()[0]
                    == 69
                )
                assert not cursor.execute(
                    "SELECT name FROM sqlite_master WHERE name IN (?, ?)",
                    ("buddy_profiles", "buddy_visual_bindings"),
                ).fetchall()
            return _activate(PersonaVisualRepository(db))
        finally:
            db.close_connection()


def _has_complete_cycle(sequence: list[int], frame_count: int) -> bool:
    """Require every ordered frame and the transition back to the first one."""
    expected = [*range(frame_count), 0]
    return any(
        sequence[index : index + len(expected)] == expected
        for index in range(len(sequence) - frame_count)
    )


@pytest.mark.parametrize("kind", ["absolute", "traversal", "symlink"])
def test_capture_output_rejects_escape(monkeypatch, tmp_path, kind):
    """Reject configured exports outside the allowed scratch artifact root.

    Args:
        monkeypatch: Scoped environment and artifact-root overrides.
        tmp_path: Disposable root for the allowed and outside directories.
        kind: Absolute, traversal, or symlink escape to reject.
    """
    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setitem(globals(), "_ARTIFACT_ROOT", root)
    candidate = {
        "absolute": str(outside),
        "traversal": "../outside",
        "symlink": "link",
    }[kind]
    monkeypatch.setenv("TLDW_BUDDY_QUALIFICATION_OUTPUT", candidate)
    with pytest.raises(ValueError, match="outside the allowed directory"):
        _capture_output(tmp_path)


def test_capture_output_accepts_only_scoped_destinations(monkeypatch, tmp_path):
    """Allow the fixture default and relative/absolute descendants of the export root.

    Args:
        monkeypatch: Scoped environment and artifact-root overrides.
        tmp_path: Disposable profile and export parent.
    """
    monkeypatch.delenv("TLDW_BUDDY_QUALIFICATION_OUTPUT", raising=False)
    assert _capture_output(tmp_path) == tmp_path / "captures"
    root = tmp_path / "artifacts"
    monkeypatch.setitem(globals(), "_ARTIFACT_ROOT", root)
    for configured in ("follow-up", str(root / "follow-up")):
        monkeypatch.setenv("TLDW_BUDDY_QUALIFICATION_OUTPUT", configured)
        assert _capture_output(tmp_path) == root / "follow-up"


@pytest.mark.parametrize("name", ["fresh-static-management.svg", "fresh-evidence.json"])
def test_artifact_write_rejects_symlink_leaf(tmp_path: Path, name: str) -> None:
    """Reject SVG and receipt links without overwriting their outside target.

    Args:
        tmp_path: Disposable capture root and outside target parent.
        name: The generated SVG or JSON filename to replace with a symlink.
    """
    output = tmp_path / "captures"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("preserve outside bytes")
    (output / name).symlink_to(outside)

    with pytest.raises(ValueError, match="outside the allowed directory"):
        _write_artifact(output, name, "replacement")
    assert outside.read_text() == "preserve outside bytes"


def test_capture_hashes_ignore_unrelated_matching_symlinks(tmp_path: Path) -> None:
    """Ignore unknown matching files and reject a generated leaf replaced by a link.

    Args:
        tmp_path: Disposable output directory and dangling outside link target.
    """
    output = tmp_path / "captures"
    output.mkdir()
    generated = _write_artifact(output, "fresh-static-home.svg", "captured frame")
    outside = tmp_path / "never-created.svg"
    (output / "fresh-unrelated.svg").symlink_to(outside)

    assert _capture_hashes(output, [generated]) == {
        generated: hashlib.sha256(b"captured frame").hexdigest()
    }
    (output / generated).unlink()
    (output / generated).symlink_to(outside)
    with pytest.raises(ValueError, match="outside the allowed directory"):
        _capture_hashes(output, [generated])
    assert not outside.exists()


def test_upgrade_setup_closes_connection_after_activation_failure(
    monkeypatch, tmp_path
):
    """Close the real SQLite handle even after activation writes then raises.

    Args:
        monkeypatch: Scoped scratch profile and activation failure injection.
        tmp_path: Disposable predecessor database location.
    """
    _scratch_env(monkeypatch, tmp_path)
    from Tests.Persona_Visual import test_persona_visual_repository as fixture

    activate = fixture._activate
    connections = []

    def fail_after_activation(repository):
        activate(repository)
        connections.append(repository.db.get_connection())
        raise RuntimeError("qualification activation failed")

    monkeypatch.setattr(fixture, "_activate", fail_after_activation)
    with pytest.raises(RuntimeError, match="qualification activation failed"):
        _prepare_upgrade(monkeypatch, tmp_path / "upgrade.db")
    assert len(connections) == 1
    # Probe the retained raw handle: entering a new transaction could reopen it.
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connections[0].execute("SELECT 1")


@pytest.mark.parametrize(
    ("sequence", "complete"),
    [
        ([0, 1], False),
        ([0, 1, 2, 3], False),
        ([0, 1, 3, 0], False),
        ([0, 2, 1, 3, 0], False),
        ([0, 1, 2, 3, 0], True),
        ([2, 3, 0, 1, 2, 3, 0], True),
    ],
)
def test_dynamic_qualification_requires_complete_ordered_loop(sequence, complete):
    """Distinguish full loops from partial, skipped, stopped or reordered frames.

    Args:
        sequence: Observed frame transitions without repeated samples.
        complete: Whether this sequence contains the full four-frame loop.
    """
    assert _has_complete_cycle(sequence, 4) is complete


@pytest.mark.asyncio
@pytest.mark.ui
@pytest.mark.parametrize("profile_kind", ["fresh", "upgrade"])
async def test_buddy_v1_rendered_profile_journey(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile_kind: str
) -> None:
    """Render the production app with real bundled art and local SQLite.

    Args:
        monkeypatch: Scoped disposable profile and predecessor-schema overrides.
        tmp_path: Isolated HOME, config, data, database and default capture root.
        profile_kind: Fresh setup or synthetic schema69-to70 upgrade journey.
    """
    _scratch_env(monkeypatch, tmp_path)
    monkeypatch.delenv("NO_COLOR", raising=False)
    output = _capture_output(tmp_path)
    output.mkdir(parents=True, exist_ok=True)
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.config import get_chachanotes_db_path
    from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
    from tldw_chatbook.UI.Navigation.buddy_management import get_buddy_management
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_management_modal import (
        BuddyManagementModal,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
        PersonaBuddyWidget,
    )
    from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults

    assert os.environ["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
    database_path = get_chachanotes_db_path()
    assert database_path.is_relative_to(tmp_path)
    assert not database_path.exists()
    legacy = None
    if profile_kind == "upgrade":
        legacy = _prepare_upgrade(monkeypatch, database_path)

    evidence = {"profile": profile_kind, "headless": True, "frames": {}}
    generated_captures = []
    app = TldwCli()

    def capture(view: str) -> None:
        generated_captures.append(
            _write_artifact(
                output, f"{profile_kind}-{view}.svg", app.export_screenshot()
            )
        )

    async with app.run_test(size=(150, 45)) as pilot:
        await _boot_settled(app, pilot)
        with app.chachanotes_db.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT version FROM db_schema_version WHERE schema_name=?",
                    ("rag_char_chat_schema",),
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
            capture(f"{mode}-management")
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
            visual = widget._accepted_render.visual
            assert len(visual.frames) == (4 if mode == "dynamic" else 1)
            sequence = [widget.frame_index]
            deadline = asyncio.get_running_loop().time() + (
                8 if mode == "dynamic" else 1
            )
            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
                if widget.frame_index != sequence[-1]:
                    sequence.append(widget.frame_index)
                if mode == "dynamic" and _has_complete_cycle(sequence, 4):
                    break
            observed = set(sequence)
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
            if mode == "dynamic":
                assert visual.animate and visual.loop
                assert _has_complete_cycle(sequence, 4), sequence
            else:
                assert not visual.animate and sequence == [0]
            capture(f"{mode}-home")
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
        capture("conversation")
        app.screen.query_one("#buddy-close", Button).press()
        await _until(lambda: type(app.screen).__name__ == "HomeScreen")
        workspace_persona = await asyncio.to_thread(
            app.local_character_persona_service.create_persona_profile,
            {
                "id": "qualification-workspace-persona",
                "name": "Qualification workspace Persona",
                "system_prompt": "Be a helpful guide.",
            },
        )
        workspace = await asyncio.to_thread(
            app.workspace_registry_service.create_workspace,
            workspace_id="qualification-workspace",
            name="Qualification workspace",
            assistant_defaults=WorkspaceAssistantDefaults(
                assistant_id=workspace_persona["id"]
            ),
        )
        assert workspace.assistant_defaults.assistant_id == workspace_persona["id"]
        assert not workspace.assistant_defaults_explicit_none
        manager.request_open()
        await _until(lambda: isinstance(app.screen, BuddyManagementModal))
        await pilot.pause()
        app.screen.query_one(
            "#buddy-follow", Select
        ).value = f"workspace:{workspace.workspace_id}"
        await pilot.pause()
        persona_control = app.screen.query_one("#buddy-persona", Select)
        assert not persona_control.disabled
        persona_control.value = "#none"
        persona_control.scroll_visible(animate=False)
        await pilot.pause()
        assert persona_control.value == "#none"
        capture("workspace-management")
        app.screen.query_one("#buddy-apply", Button).press()
        await _until(lambda: not isinstance(app.screen, BuddyManagementModal))
        assert manager.preferences.binding.kind == "workspace"
        assert manager.preferences.binding.target_id == workspace.workspace_id
        cleared_workspace = app.workspace_registry_service.get_workspace(
            workspace.workspace_id
        )
        assert cleared_workspace.assistant_defaults is None
        assert cleared_workspace.assistant_defaults_explicit_none
        manager.request_interaction()
        await _until(lambda: type(app.screen).__name__ == "BuddyWorkspaceModal")
        await _until(lambda: app.screen._loaded and app.screen._fresh)
        await pilot.pause()
        capture("workspace-inbox")
        app.screen.query_one("#buddy-inbox-close", Button).press()
        await _until(lambda: type(app.screen).__name__ == "HomeScreen")
        assert controller.store.active_session_id == target.id
        evidence["workspace_explicit_none"] = True
        evidence["workspace_persona_before_clear"] = workspace_persona["id"]
        evidence["workspace_persona_after_clear"] = None
        evidence["workspace_explicit_none_before"] = (
            workspace.assistant_defaults_explicit_none
        )
        evidence["workspace_explicit_none_after"] = (
            cleared_workspace.assistant_defaults_explicit_none
        )
        evidence["conversation_id"] = target.id
        evidence["schema_version"] = 70
        evidence["persona_count_unchanged_during_artwork_selection"] = True
    evidence["source_commit"] = (
        await asyncio.to_thread(
            subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
        )
    ).strip()
    evidence["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    evidence["captures"] = _capture_hashes(output, generated_captures)
    _write_artifact(
        output, f"{profile_kind}-evidence.json", json.dumps(evidence, indent=2)
    )
