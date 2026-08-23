"""Application ownership and teardown order for Actor Pack exports."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


def test_app_constructs_one_export_controller_after_profile_services() -> None:
    from tldw_chatbook.app import TldwCli

    wiring = inspect.getsource(TldwCli._wire_character_persona_services)
    repository = wiring.index("self.actor_pack_repository =")
    service = wiring.index("self.actor_pack_export_service =")
    controller = wiring.index("self.actor_pack_export_controller =")

    assert repository < service < controller
    assert "PersonaVisualRepository(self.chachanotes_db)" in wiring
    assert "VisualIdentityRepository(self.chachanotes_db)" in wiring


def test_app_constructs_import_pipeline_after_profile_services() -> None:
    from tldw_chatbook.app import TldwCli

    wiring = inspect.getsource(TldwCli._wire_character_persona_services)
    repository = wiring.index("self.actor_pack_repository =")
    importer = wiring.index("self.actor_pack_import_service =")
    activation = wiring.index("self.actor_pack_activation_service =")
    controller = wiring.index("self.actor_pack_import_controller =")

    assert repository < importer < activation < controller
    assert "local_service=self.local_character_persona_service" in wiring
    assert "refresh_callbacks=(self._refresh_after_actor_pack_import,)" in wiring


def test_persona_import_fences_mounted_buddy_state() -> None:
    from tldw_chatbook.app import TldwCli

    invalidated: list[str] = []
    app = object.__new__(TldwCli)
    app.persona_buddy_controller = SimpleNamespace(
        invalidate_profile=lambda: invalidated.append("persona")
    )

    TldwCli._refresh_after_actor_pack_import(
        app, SimpleNamespace(actor_kind="character")
    )
    TldwCli._refresh_after_actor_pack_import(app, SimpleNamespace(actor_kind="persona"))

    assert invalidated == ["persona"]


def test_export_controller_is_in_app_owned_shutdown_before_profile_teardown() -> None:
    from tldw_chatbook.app import TldwCli

    owned = inspect.getsource(TldwCli._shutdown_app_owned_lifecycles)
    assert owned.index("_shutdown_actor_pack_import") < owned.index(
        "_shutdown_actor_pack_export"
    )
    assert owned.index("_shutdown_actor_pack_export") < owned.index(
        "_shutdown_console_runtime"
    )
    shutdown = inspect.getsource(TldwCli._shutdown_actor_pack_export)
    assert "controller.shutdown" in shutdown
    import_shutdown = inspect.getsource(TldwCli._shutdown_actor_pack_import)
    assert "controller.shutdown" in import_shutdown


@pytest.mark.asyncio
async def test_export_shutdown_finishes_before_later_app_owners() -> None:
    from tldw_chatbook.app import TldwCli

    events: list[str] = []

    class ExportController:
        async def shutdown(self) -> None:
            events.append("export")

    class ImportController:
        async def shutdown(self) -> None:
            events.append("import")

    class AsyncOwner:
        async def shutdown(self) -> None:
            events.append("later-owner")

    async def record(name: str) -> None:
        events.append(name)

    app = object.__new__(TldwCli)
    app.actor_pack_import_controller = ImportController()
    app._actor_pack_import_shutdown_task = None
    app.actor_pack_export_controller = ExportController()
    app._actor_pack_export_shutdown_task = None
    app._shutdown_notes_sync_runtime = lambda: record("notes")
    app._shutdown_console_runtime = lambda: record("console")
    app._shutdown_persona_buddy = lambda: record("buddy")
    app._audio_cpp_artifact_lease_coordinator = None
    app.audio_cpp_model_install_owner = AsyncOwner()
    app._shutdown_console_image_edits = lambda: record("image")
    app._shutdown_file_notes_session_owner = lambda: record("file-notes")

    await TldwCli._shutdown_app_owned_lifecycles(app)

    assert events == [
        "notes",
        "import",
        "export",
        "console",
        "buddy",
        "later-owner",
        "image",
        "file-notes",
    ]
