"""Production app wiring for the Research Workspace foundation."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Research_Workspace import (
    QualifiedWorkspaceRef,
    ResearchPanePreferences,
    ResearchPresentationOverlayStore,
    ResearchWorkspaceCatalogState,
    ResearchWorkspaceController,
    ResearchWorkspaceSummary,
    WorkspaceDataSource,
)
from tldw_chatbook.UI.Screens.research_workspace_screen import (
    ResearchWorkspaceScreen,
)
from tldw_chatbook.Workspaces.models import WorkspaceRecord
from tldw_chatbook.app import TldwCli


class _LocalWorkspaceService:
    def __init__(self, workspace_id: str = "local-research") -> None:
        self.workspace_id = workspace_id
        self.list_calls = 0

    def list_workspaces(self, *, include_archived: bool = False):
        self.list_calls += 1
        return [
            WorkspaceRecord(
                workspace_id=self.workspace_id,
                name="Local notebook",
                archived=False,
            )
        ]


class _ServerWorkspaceService:
    def __init__(self) -> None:
        self.list_calls = 0

    async def list_workspaces(self):
        self.list_calls += 1
        return [
            {
                "id": "server-research",
                "name": "Server notebook",
                "archived": False,
                "version": 3,
            }
        ]


class _ServerContextProvider:
    def __init__(self, *, unavailable: bool = False) -> None:
        capabilities = {
            "server_configured": True,
            "reachability": "unreachable" if unavailable else "reachable",
            "auth_state": "authenticated",
            "revision": "context-1",
        }
        self.context = SimpleNamespace(
            active_server_id="server-profile-a",
            auth_token="not-a-secret-test-token",
            credential_source="test",
            capabilities=capabilities,
        )

    def get_active_context(self):
        return self.context


class _DeferredCatalogPort:
    def __init__(self) -> None:
        self.results: list[asyncio.Future] = []

    async def list_workspaces(self, *, include_archived: bool = False):
        future = asyncio.get_running_loop().create_future()
        self.results.append(future)
        return await future


def _unmounted_app(
    *,
    local_service: object | None,
    server_service: object | None,
    server_context_provider: object | None,
) -> TldwCli:
    app = TldwCli.__new__(TldwCli)
    app.workspace_registry_service = local_service
    app.server_notes_workspace_service = server_service
    app.server_context_provider = server_context_provider
    return app


def _rendered_text(widget: Static) -> str:
    return str(widget.render())


def test_research_screen_dependencies_are_fresh_and_late_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module

    first_local = _LocalWorkspaceService("local-first")
    first_server = _ServerWorkspaceService()
    first_provider = _ServerContextProvider()
    app = _unmounted_app(
        local_service=first_local,
        server_service=first_server,
        server_context_provider=first_provider,
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)

    first = app._create_navigation_screen("research_workspace", ResearchWorkspaceScreen)

    first_local_port = first.controller.port_for_data_source(WorkspaceDataSource.LOCAL)
    first_server_port = first.controller.port_for_data_source(
        WorkspaceDataSource.SERVER
    )
    assert first_local_port._service is first_local
    assert first_server_port._service is first_server
    assert first_server_port._context_provider is first_provider
    assert first.overlay_store.path == tmp_path / "research_workspace_overlay.json"
    assert not first.overlay_store.path.exists()

    second_local = _LocalWorkspaceService("local-second")
    second_server = _ServerWorkspaceService()
    second_provider = _ServerContextProvider()
    app.workspace_registry_service = second_local
    app.server_notes_workspace_service = second_server
    app.server_context_provider = second_provider

    second = app._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )

    assert second is not first
    assert (
        second.controller.port_for_data_source(WorkspaceDataSource.LOCAL)._service
        is second_local
    )
    assert (
        second.controller.port_for_data_source(WorkspaceDataSource.SERVER)._service
        is second_server
    )
    assert (
        second.controller.port_for_data_source(
            WorkspaceDataSource.SERVER
        )._context_provider
        is second_provider
    )
    assert second.controller is not first.controller
    assert second.overlay_store is not first.overlay_store


@pytest.mark.asyncio
async def test_missing_foundation_services_return_typed_recovery_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module

    app = _unmounted_app(
        local_service=None,
        server_service=None,
        server_context_provider=None,
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)

    screen = app._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )
    local_state = await screen.controller.refresh_workspace_catalog()
    screen.controller.select_data_source(WorkspaceDataSource.SERVER)
    server_state = await screen.controller.refresh_workspace_catalog()

    assert local_state.data_source is WorkspaceDataSource.LOCAL
    assert local_state.workspaces == ()
    assert local_state.recovery is not None
    assert local_state.recovery.reason_code == "local_service_unavailable"
    assert server_state.data_source is WorkspaceDataSource.SERVER
    assert server_state.workspaces == ()
    assert server_state.recovery is not None
    assert server_state.recovery.reason_code == "server_service_unavailable"


class _ResearchHarness(ConsolidatedCSSApp):
    def __init__(self, screen: ResearchWorkspaceScreen) -> None:
        super().__init__()
        self._research_screen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self._research_screen)


class _ProductionResearchHarness(_ResearchHarness):
    CSS_PATH = TldwCli.CSS_PATH


@pytest.mark.asyncio
async def test_mounted_selector_switches_controller_and_qualified_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module

    local = _LocalWorkspaceService()
    server = _ServerWorkspaceService()
    provider = _ServerContextProvider()
    app_owner = _unmounted_app(
        local_service=local,
        server_service=server,
        server_context_provider=provider,
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    screen = app_owner._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )
    app = _ResearchHarness(screen)

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        assert screen.controller.selected_data_source is WorkspaceDataSource.LOCAL
        local_state = screen.controller.catalog_state
        assert local_state is not None
        assert local_state.workspaces[0].ref.data_source is WorkspaceDataSource.LOCAL
        assert local_state.workspaces[0].ref.workspace_id == "local-research"
        assert "Local notebook" in _rendered_text(
            screen.query_one("#research-workspace-selection", Static)
        )

        screen.query_one("#research-data-source-server", Button).press()
        await pilot.pause(0.1)

        assert screen.controller.selected_data_source is WorkspaceDataSource.SERVER
        server_state = screen.controller.catalog_state
        assert server_state is not None
        server_ref = server_state.workspaces[0].ref
        assert server_ref.data_source is WorkspaceDataSource.SERVER
        assert server_ref.workspace_id == "server-research"
        assert server_ref.server_profile_id == "server-profile-a"
        assert server_ref.principal_id.startswith("credential-fingerprint:test:")
        assert "not-a-secret-test-token" not in server_ref.principal_id
        assert "Server notebook" in _rendered_text(
            screen.query_one("#research-workspace-selection", Static)
        )
        assert screen.query_one("#research-data-source-server", Button).has_class(
            "is-active"
        )
        assert "Server catalog ready" in _rendered_text(
            screen.query_one("#research-workspace-status", Static)
        )
        assert local.list_calls == 1
        assert server.list_calls == 1


@pytest.mark.asyncio
async def test_unavailable_server_stays_selected_with_recovery_and_no_local_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module

    local = _LocalWorkspaceService()
    server = _ServerWorkspaceService()
    provider = _ServerContextProvider(unavailable=True)
    app_owner = _unmounted_app(
        local_service=local,
        server_service=server,
        server_context_provider=provider,
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    screen = app_owner._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )
    app = _ResearchHarness(screen)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.1)
        local_calls_before_switch = local.list_calls
        screen.query_one("#research-data-source-server", Button).press()
        await pilot.pause(0.1)

        state = screen.controller.catalog_state
        assert state is not None
        assert state.data_source is WorkspaceDataSource.SERVER
        assert state.workspaces == ()
        assert state.recovery is not None
        assert state.recovery.reason_code == "server_unavailable"
        assert screen.query_one("#research-data-source-server", Button).has_class(
            "is-active"
        )
        recovery = _rendered_text(
            screen.query_one("#research-authority-recovery", Static)
        )
        assert "selected server is unavailable" in recovery.lower()
        assert "retry or change" in recovery.lower()
        status = _rendered_text(screen.query_one("#research-workspace-status", Static))
        assert "Server selected" in status
        assert "Recovery required" in status
        assert local.list_calls == local_calls_before_switch
        assert server.list_calls == 0


@pytest.mark.asyncio
async def test_selected_workspace_loads_and_saves_device_only_pane_preferences(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module

    app_owner = _unmounted_app(
        local_service=_LocalWorkspaceService(),
        server_service=_ServerWorkspaceService(),
        server_context_provider=_ServerContextProvider(),
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    screen = app_owner._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )
    local_ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "local-research")
    screen.overlay_store.save(
        local_ref,
        ResearchPanePreferences(sources_open=False, studio_open=True),
        expected_revision=0,
        timestamp="2026-08-24T00:00:00Z",
    )
    app = _ResearchHarness(screen)

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        assert screen.controller.selected_ref == local_ref
        assert screen.pane_preferences.sources_open is False
        assert not screen.query_one("#research-sources-pane").display

        screen.query_one("#research-sources-reveal", Button).press()
        await pilot.pause(0.1)

        saved = screen.overlay_store.load(local_ref)
        assert saved is not None
        assert saved.revision == 2
        assert saved.preferences.sources_open is True


def test_foundation_screen_does_not_construct_future_phase_coordinators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module

    app = _unmounted_app(
        local_service=_LocalWorkspaceService(),
        server_service=_ServerWorkspaceService(),
        server_context_provider=_ServerContextProvider(),
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    screen = app._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )

    for attribute in (
        "source_coordinator",
        "chat_coordinator",
        "studio_coordinator",
        "sharing_coordinator",
        "transfer_coordinator",
        "ingestion_coordinator",
    ):
        assert not hasattr(screen, attribute)


@pytest.mark.asyncio
async def test_catalog_aba_does_not_repaint_newer_local_selection() -> None:
    port = _DeferredCatalogPort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    screen = ResearchWorkspaceScreen(SimpleNamespace(), controller=controller)
    app = _ResearchHarness(screen)

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        controller.select_data_source(WorkspaceDataSource.SERVER)
        controller.select_data_source(WorkspaceDataSource.LOCAL)
        new_request = asyncio.create_task(screen._refresh_workspace_catalog())
        await asyncio.sleep(0)

        new_workspace = ResearchWorkspaceSummary(
            ref=QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "new"),
            name="New notebook",
        )
        port.results[1].set_result((new_workspace,))
        await new_request
        assert "New notebook" in _rendered_text(
            screen.query_one("#research-workspace-selection", Static)
        )

        old_workspace = ResearchWorkspaceSummary(
            ref=QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "old"),
            name="Old notebook",
        )
        port.results[0].set_result((old_workspace,))
        await pilot.pause()

        assert "New notebook" in _rendered_text(
            screen.query_one("#research-workspace-selection", Static)
        )
        assert screen.controller.catalog_state is not None
        assert screen.controller.catalog_state.workspaces == (new_workspace,)


@pytest.mark.asyncio
async def test_local_overlay_preferences_do_not_carry_to_server_without_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module

    app_owner = _unmounted_app(
        local_service=_LocalWorkspaceService(),
        server_service=_ServerWorkspaceService(),
        server_context_provider=_ServerContextProvider(),
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    screen = app_owner._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )
    local_ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "local-research")
    screen.overlay_store.save(
        local_ref,
        ResearchPanePreferences(sources_open=False, studio_open=False),
        expected_revision=0,
        timestamp="2026-08-24T00:00:00Z",
    )
    app = _ResearchHarness(screen)

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        assert screen.pane_preferences == ResearchPanePreferences(
            sources_open=False, studio_open=False
        )
        screen.query_one("#research-data-source-server", Button).press()
        await pilot.pause(0.1)

        assert screen.controller.selected_data_source is WorkspaceDataSource.SERVER
        assert screen.pane_preferences == ResearchPanePreferences()
        assert screen.query_one("#research-sources-pane").display
        assert screen.query_one("#research-studio-pane").display


@pytest.mark.asyncio
async def test_server_overlay_preferences_do_not_carry_back_to_local_without_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module

    app_owner = _unmounted_app(
        local_service=_LocalWorkspaceService(),
        server_service=_ServerWorkspaceService(),
        server_context_provider=_ServerContextProvider(),
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    screen = app_owner._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )
    server_adapter = screen.controller.port_for_data_source(WorkspaceDataSource.SERVER)
    server_ref = (await server_adapter.list_workspaces())[0].ref
    screen.overlay_store.save(
        server_ref,
        ResearchPanePreferences(sources_open=False, studio_open=False),
        expected_revision=0,
        timestamp="2026-08-24T00:00:00Z",
    )
    app = _ResearchHarness(screen)

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        screen.query_one("#research-data-source-server", Button).press()
        await pilot.pause(0.1)
        assert screen.pane_preferences == ResearchPanePreferences(
            sources_open=False, studio_open=False
        )

        screen.query_one("#research-data-source-local", Button).press()
        await pilot.pause(0.1)
        assert screen.controller.selected_data_source is WorkspaceDataSource.LOCAL
        assert screen.pane_preferences == ResearchPanePreferences()


@pytest.mark.asyncio
async def test_medium_companion_change_persists_qualified_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module

    app_owner = _unmounted_app(
        local_service=_LocalWorkspaceService(),
        server_service=_ServerWorkspaceService(),
        server_context_provider=_ServerContextProvider(),
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    screen = app_owner._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )
    app = _ResearchHarness(screen)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause(0.1)
        screen.query_one("#research-pane-mode-studio", Button).press()
        await pilot.pause(0.1)

        assert screen.controller.selected_ref is not None
        saved = screen.overlay_store.load(screen.controller.selected_ref)
        assert saved is not None
        assert saved.preferences.preferred_companion == "studio"


class _BlockingOverlayStore:
    def __init__(self, local_ref: QualifiedWorkspaceRef) -> None:
        self.local_ref = local_ref
        self.started = threading.Event()
        self.release = threading.Event()

    def load(self, ref: QualifiedWorkspaceRef):
        if ref != self.local_ref:
            return None
        self.started.set()
        assert self.release.wait(2)
        return SimpleNamespace(
            ref=ref,
            revision=1,
            preferences=ResearchPanePreferences(sources_open=False, studio_open=False),
        )


class _CommitThenPauseOverlayStore:
    """Let the first real save commit, then delay its coroutine result."""

    def __init__(self, path: Path) -> None:
        self._store = ResearchPresentationOverlayStore(path)
        self.first_committed = threading.Event()
        self.release_first = threading.Event()

    def load(self, ref: QualifiedWorkspaceRef):
        return self._store.load(ref)

    def save(
        self,
        ref: QualifiedWorkspaceRef,
        preferences: ResearchPanePreferences,
        *,
        expected_revision: int,
    ):
        saved = self._store.save(
            ref,
            preferences,
            expected_revision=expected_revision,
        )
        if saved.revision == 1:
            self.first_committed.set()
            assert self.release_first.wait(2)
        return saved


@pytest.mark.asyncio
async def test_rapid_medium_companion_saves_follow_committed_revision_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A committed thread write must finish before the final choice is saved."""
    import tldw_chatbook.app as app_module

    app_owner = _unmounted_app(
        local_service=_LocalWorkspaceService(),
        server_service=_ServerWorkspaceService(),
        server_context_provider=_ServerContextProvider(),
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    screen = app_owner._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )
    store = _CommitThenPauseOverlayStore(tmp_path / "race-overlay.json")
    screen.overlay_store = store
    warnings: list[str] = []
    monkeypatch.setattr(
        screen,
        "notify",
        lambda message, **kwargs: (
            warnings.append(str(message))
            if kwargs.get("severity") == "warning"
            else None
        ),
    )
    app = _ResearchHarness(screen)

    async with app.run_test(size=(120, 30)) as pilot:
        for _ in range(100):
            await pilot.pause(0.02)
            if screen.controller.selected_ref is not None:
                break
        ref = screen.controller.selected_ref
        assert ref is not None

        screen.query_one("#research-pane-mode-studio", Button).press()
        assert await asyncio.to_thread(store.first_committed.wait, 1)
        screen.query_one("#research-pane-mode-sources", Button).press()
        await pilot.pause()
        store.release_first.set()

        saved = None
        for _ in range(100):
            await pilot.pause(0.02)
            saved = store.load(ref)
            if (
                saved is not None
                and saved.revision == 2
                and saved.preferences.preferred_companion == "sources"
            ):
                break

        assert saved is not None
        assert saved.revision == 2
        assert saved.preferences == screen.pane_preferences
        assert saved.preferences.preferred_companion == "sources"
        assert screen._overlay_revision == 2
        assert warnings == []


@pytest.mark.asyncio
async def test_queued_overlay_save_recaptures_owner_after_authority_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Queued work saves the current qualified owner, never the prior ref."""
    import tldw_chatbook.app as app_module

    app_owner = _unmounted_app(
        local_service=_LocalWorkspaceService(),
        server_service=_ServerWorkspaceService(),
        server_context_provider=_ServerContextProvider(),
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    screen = app_owner._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )
    store = _CommitThenPauseOverlayStore(tmp_path / "owner-switch-overlay.json")
    screen.overlay_store = store
    app = _ResearchHarness(screen)

    async with app.run_test(size=(120, 30)) as pilot:
        for _ in range(100):
            await pilot.pause(0.02)
            if screen.controller.selected_ref is not None:
                break
        local_ref = screen.controller.selected_ref
        assert local_ref is not None

        screen.query_one("#research-pane-mode-studio", Button).press()
        assert await asyncio.to_thread(store.first_committed.wait, 1)
        screen.query_one("#research-pane-mode-sources", Button).press()
        await pilot.pause()
        screen.query_one("#research-data-source-server", Button).press()
        for _ in range(100):
            await pilot.pause(0.02)
            if (
                screen._overlay_ref is not None
                and screen._overlay_ref.data_source is WorkspaceDataSource.SERVER
            ):
                break
        server_ref = screen._overlay_ref
        assert server_ref is not None
        assert server_ref.data_source is WorkspaceDataSource.SERVER
        store.release_first.set()

        server_saved = None
        for _ in range(100):
            await pilot.pause(0.02)
            server_saved = store.load(server_ref)
            if server_saved is not None:
                break

        local_saved = store.load(local_ref)
        assert local_saved is not None
        assert local_saved.revision == 1
        assert local_saved.preferences.preferred_companion == "studio"
        assert server_saved is not None
        assert server_saved.revision == 1
        assert server_saved.preferences == ResearchPanePreferences()
        assert screen._overlay_revision == 1


@pytest.mark.asyncio
async def test_stale_overlay_load_cannot_repaint_new_qualified_workspace() -> None:
    local_ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "local")
    server_ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "server",
        server_profile_id="profile-a",
        principal_id="principal-a",
    )
    store = _BlockingOverlayStore(local_ref)
    controller = ResearchWorkspaceController({})
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(), controller=controller, overlay_store=store
    )
    app = _ResearchHarness(screen)

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        local_state = ResearchWorkspaceCatalogState(
            data_source=WorkspaceDataSource.LOCAL,
            context_revision=controller.context_revision,
            catalog_generation=controller.catalog_generation,
            workspaces=(ResearchWorkspaceSummary(ref=local_ref, name="Local"),),
        )
        old_load = asyncio.create_task(screen._apply_catalog_state(local_state))
        assert await asyncio.to_thread(store.started.wait, 1)

        controller.select_data_source(WorkspaceDataSource.SERVER)
        server_state = ResearchWorkspaceCatalogState(
            data_source=WorkspaceDataSource.SERVER,
            context_revision=controller.context_revision,
            catalog_generation=controller.catalog_generation,
            workspaces=(ResearchWorkspaceSummary(ref=server_ref, name="Server"),),
        )
        await screen._apply_catalog_state(server_state)
        assert screen.controller.selected_ref == server_ref
        assert screen.pane_preferences == ResearchPanePreferences()

        store.release.set()
        await old_load

        assert screen.controller.selected_ref == server_ref
        assert screen.pane_preferences == ResearchPanePreferences()


@pytest.mark.asyncio
async def test_mismatched_overlay_ref_cannot_repaint_selected_workspace() -> None:
    selected_ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "selected")
    wrong_ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "wrong")

    class _MismatchedOverlayStore:
        def load(self, ref: QualifiedWorkspaceRef):
            return SimpleNamespace(
                ref=wrong_ref,
                revision=1,
                preferences=ResearchPanePreferences(
                    sources_open=False, studio_open=False
                ),
            )

    controller = ResearchWorkspaceController({})
    screen = ResearchWorkspaceScreen(
        SimpleNamespace(),
        controller=controller,
        overlay_store=_MismatchedOverlayStore(),
    )
    app = _ResearchHarness(screen)

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        state = ResearchWorkspaceCatalogState(
            data_source=WorkspaceDataSource.LOCAL,
            context_revision=controller.context_revision,
            catalog_generation=controller.catalog_generation,
            workspaces=(ResearchWorkspaceSummary(ref=selected_ref, name="Selected"),),
        )
        await screen._apply_catalog_state(state)

        assert screen.controller.selected_ref == selected_ref
        assert screen.pane_preferences == ResearchPanePreferences()


@pytest.mark.asyncio
async def test_inactive_authority_focus_does_not_impersonate_active_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module

    app_owner = _unmounted_app(
        local_service=_LocalWorkspaceService(),
        server_service=_ServerWorkspaceService(),
        server_context_provider=_ServerContextProvider(),
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    screen = app_owner._create_navigation_screen(
        "research_workspace", ResearchWorkspaceScreen
    )
    app = _ProductionResearchHarness(screen)

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        server = screen.query_one("#research-data-source-server", Button)
        local = screen.query_one("#research-data-source-local", Button)
        server.press()
        await pilot.pause(0.1)
        local.focus()
        await pilot.pause()

        assert server.has_class("is-active")
        assert not local.has_class("is-active")
        assert server.styles.background != local.styles.background
