"""Personal Context lifecycle controls in canonical Settings."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Select, Static
from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileProvenance,
    ProfileRecord,
    ProfileScope,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
)

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _visible_text,
)
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.Personal_Context.runtime_policy import AgentAuthority
from tldw_chatbook.Personal_Context.key_protector import (
    InMemoryProfileKeyProtector,
    ProfileLockedError,
)
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.service import (
    PersonalContextSettingsSnapshot,
    PersonalContextService,
    ProfileConflictError,
    ProfileKeyCollisionError,
    ProfileOperationalState,
    ProfileOperationalStatus,
    SettingsScopeSnapshot,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Widgets.Settings_Widgets.personal_context_panel import (
    PersonalContextSettingsPanel,
    RecoveryPassphraseDialog,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave


NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


class _ProfileServiceStub:
    def __init__(self, snapshot: PersonalContextSettingsSnapshot) -> None:
        self.snapshot = snapshot
        self.runtime_changes: list[bool] = []
        self.authority_changes: list[tuple[str, AgentAuthority, str | None]] = []
        self.finish_secure_removal_calls = 0

    def settings_snapshot(self) -> PersonalContextSettingsSnapshot:
        return self.snapshot

    def set_runtime_enabled(self, enabled: bool) -> None:
        self.runtime_changes.append(enabled)

    def set_scope_authority(
        self,
        scope_id: str,
        authority: AgentAuthority,
        *,
        expected_policy_version_id: str | None,
    ) -> None:
        self.authority_changes.append((scope_id, authority, expected_policy_version_id))

    def finish_secure_removal(self) -> None:
        self.finish_secure_removal_calls += 1


class _ExplodingProfileService:
    def settings_snapshot(self):
        raise RuntimeError("private-value-that-must-never-render")


class _ConflictProfileService(_ProfileServiceStub):
    def update_record(self, *_args, **_kwargs):
        raise ProfileConflictError("private-conflict-detail")


class _ExportCaptureService(_ProfileServiceStub):
    def __init__(self, snapshot: PersonalContextSettingsSnapshot) -> None:
        super().__init__(snapshot)
        self.plaintext_requests: list[object] = []
        self.recovery_requests: list[object] = []

    def export_plaintext(self, request):
        self.plaintext_requests.append(request)

    def export_recovery(self, request):
        self.recovery_requests.append(request)


class _CollisionProfileService(_ProfileServiceStub):
    def _collision(self, *_args, **_kwargs):
        raise ProfileKeyCollisionError("private-colliding-record-id")

    create_manual_record = _collision
    update_record = _collision
    restore_record = _collision


def _status(state: ProfileOperationalState) -> ProfileOperationalStatus:
    return ProfileOperationalStatus(
        state=state,
        profile_present=state
        not in {ProfileOperationalState.ABSENT, ProfileOperationalState.REMOVED},
        locked=state is ProfileOperationalState.LOCKED,
        runtime_enabled=state is ProfileOperationalState.READY,
        reason_code=(
            "personal_context_disabled"
            if state is ProfileOperationalState.DISABLED
            else None
        ),
    )


def _ready_snapshot() -> PersonalContextSettingsSnapshot:
    scope = ProfileScope(
        scope_id="scope-global",
        profile_id="profile-local",
        kind=ScopeKind.GLOBAL,
        version_id="scope-version",
        created_at=NOW,
        updated_at=NOW,
    )
    record = ProfileRecord(
        profile_id="profile-local",
        record_id="record-1",
        scope_id=scope.scope_id,
        kind="preference",
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise answers"
        ),
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
        state=RecordState.ACTIVE,
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
        provenance=ProfileProvenance(
            source="manual", actor="user", reason_code="settings_edit"
        ),
        version_id="record-version",
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
    )
    return PersonalContextSettingsSnapshot(
        status=_status(ProfileOperationalState.READY),
        scopes=(
            SettingsScopeSnapshot(
                scope=scope,
                label="Global",
                linked=True,
                authority=AgentAuthority.PROPOSE,
            ),
        ),
        records=(record,),
    )


def _multi_scope_snapshot() -> PersonalContextSettingsSnapshot:
    ready = _ready_snapshot()
    global_scope = ready.scopes[0]
    linked_scope = ProfileScope(
        scope_id="scope-workspace-linked",
        profile_id="profile-local",
        kind=ScopeKind.WORKSPACE,
        version_id="scope-linked-version",
        created_at=NOW,
        updated_at=NOW,
    )
    unlinked_scope = ProfileScope(
        scope_id="scope-workspace-unlinked",
        profile_id="profile-local",
        kind=ScopeKind.WORKSPACE,
        version_id="scope-unlinked-version",
        created_at=NOW,
        updated_at=NOW,
    )
    base_record = ready.records[0]
    linked_record = base_record.model_copy(
        update={
            "record_id": "record-workspace-linked",
            "scope_id": linked_scope.scope_id,
            "payload": PreferencePayload(
                subject="project.detail", polarity="like", value="project context"
            ),
            "semantic_key": SemanticKey(
                namespace="preference", subject="project.detail"
            ),
            "version_id": "record-linked-version",
        }
    )
    unlinked_record = base_record.model_copy(
        update={
            "record_id": "record-workspace-unlinked",
            "scope_id": unlinked_scope.scope_id,
            "payload": PreferencePayload(
                subject="legacy.detail", polarity="like", value="retained context"
            ),
            "semantic_key": SemanticKey(
                namespace="preference", subject="legacy.detail"
            ),
            "version_id": "record-unlinked-version",
        }
    )
    return replace(
        ready,
        scopes=(
            global_scope,
            SettingsScopeSnapshot(
                scope=linked_scope,
                label="Project Atlas",
                linked=True,
                authority=AgentAuthority.PROPOSE,
            ),
            SettingsScopeSnapshot(
                scope=unlinked_scope,
                label="Unlinked workspace",
                linked=False,
                authority=AgentAuthority.PROPOSE,
            ),
        ),
        records=(base_record, linked_record, unlinked_record),
    )


def _real_service(tmp_path, *, with_record: bool = False) -> PersonalContextService:
    service = PersonalContextService(
        PersonalContextRepository(
            tmp_path / "personal-context.db",
            key_protector=InMemoryProfileKeyProtector(),
        )
    )
    service.create_profile()
    service.set_runtime_enabled(True)
    if with_record:
        scope = service.list_scopes()[0]
        service.create_manual_record(
            scope_id=scope.scope_id,
            payload=PreferencePayload(
                subject="response.detail", polarity="like", value="concise answers"
            ),
            semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
            controls=ProfileControls(
                sync_mode=SyncMode.SYNCABLE,
                agent_visibility=AgentVisibility.AGENT_VISIBLE,
            ),
        )
    return service


async def _wait_for_record_rows(panel, pilot, count: int) -> None:
    for _ in range(20):
        await pilot.pause()
        if len(panel.query(".personal-context-record-row")) == count:
            return
    assert len(panel.query(".personal-context-record-row")) == count


def test_my_profile_is_registered_in_data_privacy_and_settings_contracts() -> None:
    screen = SettingsScreen(_build_test_app())

    summaries = {summary.category: summary for summary in screen._category_summaries()}
    groups = dict(screen._category_groups())

    assert summaries[SettingsCategoryId.PERSONAL_CONTEXT].title == "My Profile"
    assert SettingsCategoryId.PERSONAL_CONTEXT in groups["Data & Privacy"]
    assert SettingsCategoryId.PERSONAL_CONTEXT in screen._build_ownership_by_category()
    assert screen._inspector_guidance(SettingsCategoryId.PERSONAL_CONTEXT)
    assert (
        screen._persistence_badge(SettingsCategoryId.PERSONAL_CONTEXT)
        == "Applies immediately"
    )
    assert "Encrypted local profile" in screen._category_state_scope_text(
        SettingsCategoryId.PERSONAL_CONTEXT
    )
    assert callable(getattr(TldwCli, "launch_personal_context_link", None))


def test_profile_detail_compose_does_not_resolve_app_service() -> None:
    app = _build_test_app()
    resolutions: list[object] = []

    def resolve_service():
        resolutions.append(object())
        return _ProfileServiceStub(_ready_snapshot())

    app.get_personal_context_service = resolve_service
    screen = SettingsScreen(app)
    screen.active_category = SettingsCategoryId.PERSONAL_CONTEXT.value

    rendered = list(screen._render_detail_pane())

    assert resolutions == []
    assert len(rendered) == 1
    assert isinstance(rendered[0], PersonalContextSettingsPanel)


def test_app_personal_context_service_is_lazy_and_reused(monkeypatch) -> None:
    app = _build_test_app()
    created: list[object] = []

    def bootstrap():
        service = SimpleNamespace(marker=len(created))
        created.append(service)
        return service

    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.bootstrap.bootstrap_personal_context_service",
        bootstrap,
    )

    assert created == []
    assert app.get_personal_context_service() is app.get_personal_context_service()
    assert len(created) == 1


@pytest.mark.asyncio
async def test_fresh_profile_settings_visit_retries_a_locked_app_service(
    monkeypatch,
) -> None:
    app = _build_test_app()
    locked = PersonalContextService.locked("profile_locked", profile_present=True)
    available = _ProfileServiceStub(_ready_snapshot())
    app._personal_context_service = locked
    calls: list[None] = []

    def bootstrap():
        calls.append(None)
        return available

    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.bootstrap.bootstrap_personal_context_service",
        bootstrap,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(100, 30)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)

        assert (
            str(screen.query_one("#personal-context-status").renderable) == "Available"
        )
        assert app._personal_context_service is available
        assert calls == [None]


@pytest.mark.asyncio
async def test_my_profile_renders_content_safe_status_records_and_local_actions() -> (
    None
):
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(_ready_snapshot())
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert (
            str(screen.query_one("#personal-context-status").renderable) == "Available"
        )
        assert screen.query(".personal-context-record-row")
        assert screen.query("#personal-context-remove-local")
        assert not screen.query("#personal-context-delete-everywhere")
        assert "concise answers" in _visible_text(screen)


@pytest.mark.asyncio
async def test_scope_filter_browses_global_linked_and_unlinked_records() -> None:
    snapshot = _multi_scope_snapshot()
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(snapshot)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 38)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        screen = _active_destination_screen(host)
        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        assert len(panel.query(".personal-context-record-row")) == 3
        assert "Global" in _visible_text(panel)
        assert "Project Atlas" in _visible_text(panel)
        assert "Unlinked workspace" in _visible_text(panel)

        scope_filter = panel.query_one("#personal-context-scope-filter", Select)
        scope_filter.value = "scope-workspace-linked"
        await _wait_for_record_rows(panel, pilot, 1)

        rows = list(panel.query(".personal-context-record-row"))
        assert len(rows) == 1
        assert "Project Atlas" in str(rows[0].label)
        assert panel.selected_record_id == "record-workspace-linked"

        scope_filter = panel.query_one("#personal-context-scope-filter", Select)
        scope_filter.value = "__all__"
        await _wait_for_record_rows(panel, pilot, 3)
        assert panel.selected_record_id == "record-workspace-linked"

        scope_filter = panel.query_one("#personal-context-scope-filter", Select)
        scope_filter.value = "scope-workspace-unlinked"
        await _wait_for_record_rows(panel, pilot, 1)

        rows = list(panel.query(".personal-context-record-row"))
        assert len(rows) == 1
        assert "Unlinked workspace" in str(rows[0].label)
        assert panel.selected_record_id == "record-workspace-unlinked"


@pytest.mark.asyncio
async def test_run_interview_again_passes_selected_linked_scope_audience_and_mode() -> (
    None
):
    launched: list[tuple[str, str, str]] = []
    panel = PersonalContextSettingsPanel(
        _ProfileServiceStub(_multi_scope_snapshot()),
        interview_launcher=lambda kind, scope_id, mode: launched.append(
            (kind, scope_id, mode)
        ),
    )

    class _PanelHost(ConsolidatedCSSApp):
        def compose(self):
            yield panel

    host = _PanelHost()
    async with host.run_test(size=(100, 30)) as pilot:
        await host.workers.wait_for_complete()
        panel.selected_scope_id = "scope-workspace-linked"
        await pilot.pause()
        panel.query_one("#personal-context-interview-mode", Select).value = "adaptive"
        panel.action_run_interview()

        assert launched == [("workspace", "scope-workspace-linked", "adaptive")]


@pytest.mark.asyncio
async def test_run_interview_again_is_disabled_for_unlinked_scope() -> None:
    launched: list[tuple[str, str, str]] = []
    panel = PersonalContextSettingsPanel(
        _ProfileServiceStub(_multi_scope_snapshot()),
        interview_launcher=lambda kind, scope_id, mode: launched.append(
            (kind, scope_id, mode)
        ),
    )

    class _PanelHost(ConsolidatedCSSApp):
        def compose(self):
            yield panel

    host = _PanelHost()
    async with host.run_test(size=(100, 30)) as pilot:
        await host.workers.wait_for_complete()
        panel.selected_scope_id = "scope-workspace-unlinked"
        await pilot.pause()

        assert panel.query_one("#personal-context-run-interview", Button).disabled
        panel.action_run_interview()
        assert launched == []


@pytest.mark.asyncio
async def test_run_interview_does_not_fall_back_to_global_for_selected_unlinked_record() -> (
    None
):
    launched: list[tuple[str, str, str]] = []
    panel = PersonalContextSettingsPanel(
        _ProfileServiceStub(_multi_scope_snapshot()),
        interview_launcher=lambda kind, scope_id, mode: launched.append(
            (kind, scope_id, mode)
        ),
    )

    class _PanelHost(ConsolidatedCSSApp):
        def compose(self):
            yield panel

    host = _PanelHost()
    async with host.run_test(size=(100, 30)) as pilot:
        await host.workers.wait_for_complete()
        panel.selected_scope_id = "__all__"
        panel.selected_record_id = "record-workspace-unlinked"
        await pilot.pause()

        assert panel.query_one("#personal-context-run-interview", Button).disabled
        panel.action_run_interview()
        assert launched == []


@pytest.mark.asyncio
async def test_missing_interview_launcher_reports_content_safe_unavailable_state() -> (
    None
):
    panel = PersonalContextSettingsPanel(_ProfileServiceStub(_ready_snapshot()))
    notices: list[tuple[str, str]] = []

    class _PanelHost(ConsolidatedCSSApp):
        def compose(self):
            yield panel

    host = _PanelHost()
    async with host.run_test(size=(100, 30)):
        await host.workers.wait_for_complete()
        panel.notify = lambda message, *, severity="information", **_kwargs: (
            notices.append((message, severity))
        )
        panel.action_run_interview()

        assert notices == [
            (
                "Interview setup is unavailable in this Settings session.",
                "warning",
            )
        ]


@pytest.mark.asyncio
async def test_link_action_is_exposed_only_on_canonical_profile_panel() -> None:
    launched: list[str] = []
    panel = PersonalContextSettingsPanel(
        _ProfileServiceStub(_ready_snapshot()),
        link_launcher=lambda: launched.append("link"),
    )

    class _PanelHost(ConsolidatedCSSApp):
        def compose(self):
            yield panel

    host = _PanelHost()
    async with host.run_test(size=(100, 34)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.click("#personal-context-link-server")
        await pilot.pause()

        assert launched == ["link"]


@pytest.mark.asyncio
async def test_plaintext_export_uses_selected_scope_or_all_scopes(
    tmp_path, monkeypatch
) -> None:
    service = _ExportCaptureService(_multi_scope_snapshot())
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 38)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        selected_paths = iter((tmp_path / "workspace.json", True))

        async def workspace_prompt(_prompt):
            return next(selected_paths)

        monkeypatch.setattr(host, "push_screen_wait", workspace_prompt)
        panel.query_one(
            "#personal-context-scope-filter", Select
        ).value = "scope-workspace-linked"
        await pilot.pause()
        await panel._choose_plaintext_export()
        await host.workers.wait_for_complete()

        assert service.plaintext_requests[-1].scope_ids == ("scope-workspace-linked",)

        selected_paths = iter((tmp_path / "all.json", True))
        panel.query_one("#personal-context-scope-filter", Select).value = "__all__"
        await pilot.pause()
        await panel._choose_plaintext_export()
        await host.workers.wait_for_complete()

        assert service.plaintext_requests[-1].scope_ids is None


@pytest.mark.asyncio
async def test_plaintext_export_captures_scope_when_export_is_confirmed(
    tmp_path, monkeypatch
) -> None:
    service = _ExportCaptureService(_multi_scope_snapshot())
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 38)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        selected_paths = iter((tmp_path / "workspace.json", True))

        async def prompt(_prompt):
            return next(selected_paths)

        pending_operations = []
        monkeypatch.setattr(host, "push_screen_wait", prompt)
        monkeypatch.setattr(
            panel,
            "_run_export",
            lambda operation, _message: pending_operations.append(operation),
        )
        panel.query_one(
            "#personal-context-scope-filter", Select
        ).value = "scope-workspace-linked"
        await pilot.pause()

        await panel._choose_plaintext_export()
        panel.query_one("#personal-context-scope-filter", Select).value = "__all__"
        await pilot.pause()
        pending_operations[0]()

        assert service.plaintext_requests[-1].scope_ids == ("scope-workspace-linked",)


@pytest.mark.asyncio
async def test_plaintext_export_confirmation_locks_scope_and_explicit_overwrite(
    tmp_path, monkeypatch
) -> None:
    service = _ExportCaptureService(_multi_scope_snapshot())
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")
    destination = tmp_path / "workspace.json"
    destination.write_text("existing", encoding="utf-8")
    prompts: list[object] = []

    async def prompt(dialog):
        prompts.append(dialog)
        return destination if len(prompts) == 1 else True

    async with host.run_test(size=(120, 38)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        panel.query_one(
            "#personal-context-scope-filter", Select
        ).value = "scope-workspace-linked"
        await pilot.pause()
        assert "Project Atlas" in str(
            panel.query_one("#personal-context-export-plaintext", Button).label
        )
        monkeypatch.setattr(host, "push_screen_wait", prompt)

        await panel._choose_plaintext_export()
        await host.workers.wait_for_complete()

        confirmation = prompts[1]
        assert isinstance(confirmation, ConfirmationDialog)
        assert "Project Atlas" in confirmation.message
        assert "file already exists" in confirmation.message
        request = service.plaintext_requests[-1]
        assert request.scope_ids == ("scope-workspace-linked",)
        assert request.confirm_overwrite is True


@pytest.mark.asyncio
async def test_plaintext_export_does_not_authorize_a_file_appearing_after_selection(
    tmp_path, monkeypatch
) -> None:
    service = _ExportCaptureService(_ready_snapshot())
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")
    destination = tmp_path / "appears-late.json"
    prompts: list[object] = []

    async def prompt(dialog):
        prompts.append(dialog)
        if len(prompts) == 1:
            return destination
        destination.write_text("raced", encoding="utf-8")
        return True

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        monkeypatch.setattr(host, "push_screen_wait", prompt)

        await panel._choose_plaintext_export()
        await host.workers.wait_for_complete()

        assert service.plaintext_requests[-1].confirm_overwrite is False


@pytest.mark.asyncio
async def test_recovery_export_requires_explicit_existing_file_replacement(
    tmp_path, monkeypatch
) -> None:
    service = _ExportCaptureService(_ready_snapshot())
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")
    destination = tmp_path / "recovery.json"
    destination.write_text("existing", encoding="utf-8")
    prompts: list[object] = []

    async def prompt(dialog):
        prompts.append(dialog)
        if len(prompts) == 1:
            return destination
        if len(prompts) == 2:
            return True
        return "correct horse battery staple"

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        monkeypatch.setattr(host, "push_screen_wait", prompt)

        await panel._choose_recovery_export()
        await host.workers.wait_for_complete()

        assert isinstance(prompts[1], ConfirmationDialog)
        assert "file already exists" in prompts[1].message
        assert isinstance(prompts[2], RecoveryPassphraseDialog)
        assert service.recovery_requests[-1].confirm_overwrite is True


@pytest.mark.asyncio
async def test_profile_panel_discards_stale_background_snapshot() -> None:
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(_ready_snapshot())
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        current = panel.snapshot
        assert current is not None
        stale = replace(
            current,
            status=_status(ProfileOperationalState.LOCKED),
            records=(),
        )

        panel._apply_snapshot(panel._load_generation - 1, stale)
        await pilot.pause()

        assert panel.snapshot is current


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "label", "expected_action"),
    [
        (ProfileOperationalState.ABSENT, "Empty", "#personal-context-create"),
        (ProfileOperationalState.DISABLED, "Disabled", "#personal-context-runtime"),
        (ProfileOperationalState.LOCKED, "Locked", None),
        (ProfileOperationalState.REMOVED, "Removed", "#personal-context-start-fresh"),
    ],
)
async def test_my_profile_named_states_are_safe_and_actionable(
    state: ProfileOperationalState, label: str, expected_action: str | None
) -> None:
    snapshot = PersonalContextSettingsSnapshot(status=_status(state))
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(snapshot)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(100, 30)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert str(screen.query_one("#personal-context-status").renderable) == label
        if expected_action is not None:
            assert screen.query(expected_action)
        if state is ProfileOperationalState.DISABLED:
            assert "Agent use is disabled" in _visible_text(screen)
        if state in {ProfileOperationalState.LOCKED, ProfileOperationalState.REMOVED}:
            assert "concise answers" not in _visible_text(screen)


@pytest.mark.asyncio
async def test_profile_load_error_is_content_free() -> None:
    app = _build_test_app()
    app._personal_context_service = _ExplodingProfileService()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(100, 30)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert str(screen.query_one("#personal-context-status").renderable) == "Error"
        assert "private-value-that-must-never-render" not in _visible_text(screen)


@pytest.mark.asyncio
async def test_unlinked_workspace_authority_is_legible_but_not_editable() -> None:
    ready = _ready_snapshot()
    unlinked = replace(ready.scopes[0], label="Unlinked workspace", linked=False)
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(
        replace(ready, scopes=(unlinked,))
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(100, 30)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        await pilot.pause()

        authority = screen.query_one("#personal-context-authority-0", Select)
        assert authority.value == AgentAuthority.PROPOSE.value
        assert authority.disabled is True
        assert "Unlinked workspace" in _visible_text(screen)


@pytest.mark.asyncio
async def test_add_editor_offers_only_linked_scopes_and_explains_sync_privacy() -> None:
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(_multi_scope_snapshot())
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 38)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        panel.action_add_record()
        await pilot.pause()

        scope = panel.query_one("#personal-context-scope", Select)
        assert set(scope._legal_values) == {
            "scope-global",
            "scope-workspace-linked",
        }
        assert scope.disabled is False
        assert (
            "An authorized home server can read syncable content. "
            "Device-only content stays on this device." in _visible_text(panel)
        )


@pytest.mark.asyncio
async def test_editor_has_persistent_labels_and_kind_specific_controls() -> None:
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(_ready_snapshot())
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 42)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        panel.action_add_record()
        await pilot.pause()

        labels = {
            str(widget.renderable)
            for widget in panel.query("#personal-context-editor .settings-input-label")
            if isinstance(widget, Static)
        }
        assert {
            "Kind",
            "Scope",
            "Subject",
            "Value",
            "Polarity",
            "Syncability",
            "Visibility",
            "Retention",
        } <= labels
        polarity = panel.query_one("#personal-context-polarity", Select)
        retention = panel.query_one("#personal-context-retention", Select)
        assert polarity.disabled is False
        assert retention.disabled is True

        panel.query_one("#personal-context-kind", Select).value = "working_context"
        await pilot.pause()
        assert panel.query_one("#personal-context-polarity", Select).disabled is True
        assert panel.query_one("#personal-context-retention", Select).disabled is False
        assert panel.query_one("#personal-context-retention", Select).value == "30_days"


@pytest.mark.asyncio
async def test_working_context_retention_defaults_can_be_overridden_and_are_preserved(
    tmp_path,
) -> None:
    service = _real_service(tmp_path)
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(130, 44)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        panel.action_add_record()
        await pilot.pause()
        panel.query_one("#personal-context-kind", Select).value = "working_context"
        panel.query_one("#personal-context-subject", Input).value = "current task"
        panel.query_one("#personal-context-value", Input).value = "draft"
        await pilot.pause()
        panel._save_editor()
        await host.workers.wait_for_complete()
        await pilot.pause()
        bounded = service.settings_snapshot().records[0]
        assert bounded.expires_at == bounded.created_at + timedelta(days=30)
        assert bounded.no_expiry is False

        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        panel.action_edit_record()
        await pilot.pause()
        assert panel.query_one("#personal-context-kind", Select).disabled is True
        assert (
            panel.query_one("#personal-context-retention", Select).value == "preserve"
        )
        panel.query_one("#personal-context-value", Input).value = "revised"
        panel._save_editor()
        await host.workers.wait_for_complete()
        await pilot.pause()
        preserved = service.settings_snapshot().records[0]
        assert preserved.expires_at == bounded.expires_at

        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        panel.action_add_record()
        await pilot.pause()
        panel.query_one("#personal-context-kind", Select).value = "working_context"
        panel.query_one("#personal-context-subject", Input).value = "long term"
        panel.query_one("#personal-context-value", Input).value = "keep"
        panel.query_one("#personal-context-retention", Select).value = "no_expiry"
        panel._save_editor()
        await host.workers.wait_for_complete()
        records = service.settings_snapshot().records
        forever = next(
            record for record in records if record.payload.subject == "long term"
        )
        assert forever.expires_at is None
        assert forever.no_expiry is True


@pytest.mark.asyncio
async def test_edit_scope_is_immutable_and_scope_moves_are_rejected() -> None:
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(_multi_scope_snapshot())
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 38)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        panel.selected_record_id = "record-workspace-linked"

        panel.action_edit_record()
        await pilot.pause()

        scope = panel.query_one("#personal-context-scope", Select)
        assert scope.value == "scope-workspace-linked"
        assert scope.disabled is True
        kind = panel.query_one("#personal-context-kind", Select)
        assert kind.value == "preference"
        assert kind.disabled is True

        kind.disabled = False
        kind.value = "goal"
        panel._save_editor()
        await pilot.pause()
        assert panel.query("#personal-context-editor")
        assert "Kind cannot be changed while editing" in _visible_text(panel)
        kind = panel.query_one("#personal-context-kind", Select)
        kind.value = "preference"

        scope.disabled = False
        scope.value = "scope-global"
        panel._save_editor()
        await pilot.pause()

        assert panel.query("#personal-context-editor")
        assert "Scope cannot be changed while editing" in _visible_text(panel)


@pytest.mark.asyncio
async def test_unlinked_scope_records_browse_but_cannot_be_added_or_edited() -> None:
    snapshot = _multi_scope_snapshot()
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(snapshot)
    host = DestinationHarness(app, "settings")
    toasts: list[str] = []
    host.notify = lambda message, **_kwargs: toasts.append(str(message))

    async with host.run_test(size=(120, 38)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        panel.query_one(
            "#personal-context-scope-filter", Select
        ).value = "scope-workspace-unlinked"
        await pilot.pause()

        assert "retained context" in _visible_text(panel)
        assert panel.query_one("#personal-context-edit", Button).disabled is True
        assert (
            panel.query_one("#personal-context-archive-restore", Button).disabled
            is True
        )
        assert panel.query_one("#personal-context-delete", Button).disabled is True
        panel.action_edit_record()
        await pilot.pause()
        assert not panel.query("#personal-context-editor")
        assert any("no longer linked" in toast for toast in toasts)

        toasts.clear()
        panel._archive_or_restore()
        panel.action_delete_record()
        await pilot.pause()
        assert not isinstance(host.screen, ConfirmationDialog)
        assert len(toasts) == 2
        assert all("browse and export only" in toast for toast in toasts)

        panel.snapshot = replace(
            snapshot,
            scopes=(snapshot.scopes[2],),
            records=(snapshot.records[2],),
        )
        await pilot.pause()
        toasts.clear()
        panel.action_add_record()
        await pilot.pause()

        assert not panel.query("#personal-context-editor")
        assert any("No linked scope" in toast for toast in toasts)


@pytest.mark.asyncio
async def test_edit_conflict_reloads_latest_and_shows_content_free_recovery() -> None:
    app = _build_test_app()
    service = _ConflictProfileService(_ready_snapshot())
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")
    toasts: list[str] = []
    host.notify = lambda message, **_kwargs: toasts.append(str(message))

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        panel.action_edit_record()
        await pilot.pause()
        panel.query_one("#personal-context-value", Input).value = "new value"
        panel._save_editor()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert not screen.query("#personal-context-editor")
        assert any("reloaded the latest version" in toast for toast in toasts)
        assert all("private-conflict-detail" not in toast for toast in toasts)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["add", "edit"])
async def test_record_collision_preserves_editor_with_actionable_safe_copy(
    mode: str,
) -> None:
    app = _build_test_app()
    app._personal_context_service = _CollisionProfileService(_ready_snapshot())
    host = DestinationHarness(app, "settings")
    toasts: list[str] = []
    host.notify = lambda message, **_kwargs: toasts.append(str(message))

    async with host.run_test(size=(120, 38)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        if mode == "add":
            panel.action_add_record()
            await pilot.pause()
            panel.query_one(
                "#personal-context-subject", Input
            ).value = "response.detail"
            panel.query_one("#personal-context-value", Input).value = "brief"
        else:
            panel.action_edit_record()
            await pilot.pause()
            panel.query_one("#personal-context-value", Input).value = "brief"

        panel._save_editor()
        await host.workers.wait_for_complete()
        await pilot.pause()

        copy = _visible_text(panel)
        assert panel.query("#personal-context-editor")
        assert "same kind and subject is already active in this scope" in copy
        assert "Change the kind or subject, or archive the other record" in copy
        assert "private-colliding-record-id" not in copy
        assert all("private-colliding-record-id" not in toast for toast in toasts)


@pytest.mark.asyncio
async def test_restore_collision_keeps_record_archived_and_gives_safe_recovery() -> (
    None
):
    ready = _ready_snapshot()
    archived = ready.records[0].model_copy(update={"state": RecordState.ARCHIVED})
    service = _CollisionProfileService(replace(ready, records=(archived,)))
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")
    toasts: list[str] = []
    host.notify = lambda message, **_kwargs: toasts.append(str(message))

    async with host.run_test(size=(120, 38)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        panel._archive_or_restore()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert panel._selected_record().state is RecordState.ARCHIVED
        assert any(
            "same kind and subject is already active in this scope" in toast
            for toast in toasts
        )
        assert all("private-colliding-record-id" not in toast for toast in toasts)


def test_my_profile_footer_contract_advertises_only_working_actions() -> None:
    shortcuts = SettingsScreen._category_footer_shortcuts(
        SettingsCategoryId.PERSONAL_CONTEXT
    )
    assert shortcuts == (
        ("a", "add record"),
        ("e", "edit record"),
        ("d", "delete record"),
        ("x", "export profile"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (
            _ready_snapshot(),
            (
                ("a", "add record"),
                ("e", "edit record"),
                ("d", "delete record"),
                ("x", "export profile"),
            ),
        ),
        (
            replace(_ready_snapshot(), records=()),
            (("a", "add record"), ("x", "export profile")),
        ),
        (
            replace(
                _multi_scope_snapshot(),
                scopes=(_multi_scope_snapshot().scopes[2],),
                records=(_multi_scope_snapshot().records[2],),
            ),
            (("x", "export profile"),),
        ),
        (
            PersonalContextSettingsSnapshot(
                status=_status(ProfileOperationalState.ABSENT)
            ),
            (),
        ),
        (
            PersonalContextSettingsSnapshot(
                status=_status(ProfileOperationalState.LOCKED)
            ),
            (),
        ),
        (
            PersonalContextSettingsSnapshot(
                status=_status(ProfileOperationalState.REMOVED)
            ),
            (),
        ),
    ],
)
async def test_profile_footer_and_f1_help_match_current_working_actions(
    snapshot: PersonalContextSettingsSnapshot,
    expected: tuple[tuple[str, str], ...],
) -> None:
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(snapshot)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)

        assert screen._footer_shortcut_entries() == expected
        screen.action_show_workbench_help()
        await pilot.pause()
        assert isinstance(host.screen, WorkbenchHelpPanel)
        assert host.screen.state.shortcuts == expected


def test_profile_loading_and_error_states_have_no_advertised_mutation_shortcuts() -> (
    None
):
    panel = PersonalContextSettingsPanel(_ProfileServiceStub(_ready_snapshot()))

    assert panel.available_shortcuts() == ()
    panel.load_failed = True
    assert panel.available_shortcuts() == ()


@pytest.mark.asyncio
async def test_profile_record_add_edit_archive_restore_and_delete_use_service(
    tmp_path,
) -> None:
    service = _real_service(tmp_path)
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(140, 42)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        panel.action_add_record()
        await pilot.pause()
        panel.query_one("#personal-context-subject", Input).value = "response.detail"
        panel.query_one("#personal-context-value", Input).value = "brief"
        panel._save_editor()
        await host.workers.wait_for_complete()
        await pilot.pause()
        record = service.settings_snapshot().records[0]
        assert record.payload.value == "brief"

        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        panel.action_edit_record()
        await pilot.pause()
        panel.query_one("#personal-context-value", Input).value = "detailed"
        panel.query_one("#personal-context-sync-mode", Select).value = "device_only"
        panel.query_one("#personal-context-visibility", Select).value = "user_only"
        panel._save_editor()
        await host.workers.wait_for_complete()
        await pilot.pause()
        record = service.settings_snapshot().records[0]
        assert record.payload.value == "detailed"
        assert record.controls.sync_mode is SyncMode.DEVICE_ONLY
        assert record.controls.agent_visibility is AgentVisibility.USER_ONLY

        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        panel._archive_or_restore()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert service.settings_snapshot().records[0].state is RecordState.ARCHIVED
        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        panel._archive_or_restore()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert service.settings_snapshot().records[0].state is RecordState.ACTIVE

        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        panel.action_delete_record()
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        await pilot.click("#confirm-button")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert not service.settings_snapshot().records


@pytest.mark.asyncio
async def test_delete_confirmation_identifies_exact_safe_selected_target(
    monkeypatch,
) -> None:
    service = _ProfileServiceStub(_multi_scope_snapshot())
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")
    prompts: list[object] = []

    async def prompt(dialog):
        prompts.append(dialog)
        return False

    async with host.run_test(size=(120, 38)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        panel.selected_record_id = "record-workspace-linked"
        monkeypatch.setattr(host, "push_screen_wait", prompt)

        panel.action_delete_record()
        await host.workers.wait_for_complete()

        assert len(prompts) == 1
        dialog = prompts[0]
        assert isinstance(dialog, ConfirmationDialog)
        assert "Project Atlas · Preference · project.detail" in dialog.message
        assert "project context" not in dialog.message
        assert "response.detail" not in dialog.message


@pytest.mark.asyncio
async def test_runtime_and_scope_authority_controls_show_exact_state_and_persist(
    tmp_path,
) -> None:
    service = _real_service(tmp_path, with_record=True)
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        assert "Disable agent use" in str(
            panel.query_one("#personal-context-runtime").label
        )
        panel._toggle_runtime()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert service.status().state is ProfileOperationalState.DISABLED

        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        authority = panel.query_one("#personal-context-authority-0", Select)
        assert authority.value == AgentAuthority.PROPOSE.value
        authority.value = AgentAuthority.READ_ONLY.value
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert (
            service.get_scope_authority(service.list_scopes()[0].scope_id)
            is AgentAuthority.READ_ONLY
        )


@pytest.mark.asyncio
async def test_scope_authority_control_carries_snapshot_policy_version() -> None:
    snapshot = _ready_snapshot()
    snapshot = replace(
        snapshot,
        scopes=(replace(snapshot.scopes[0], policy_version_id="scope-policy-version"),),
    )
    service = _ProfileServiceStub(snapshot)
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        authority = panel.query_one("#personal-context-authority-0", Select)
        authority.value = AgentAuthority.READ_ONLY.value
        await host.workers.wait_for_complete()

    assert service.authority_changes == [
        (
            snapshot.scopes[0].scope.scope_id,
            AgentAuthority.READ_ONLY,
            "scope-policy-version",
        )
    ]


@pytest.mark.asyncio
async def test_profile_panel_is_contained_and_actions_reachable_at_80x24(
    tmp_path,
) -> None:
    service = _real_service(tmp_path, with_record=True)
    app = _build_test_app()
    app._personal_context_service = service
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=(80, 24)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        await pilot.pause()
        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        detail_body = screen.query_one("#settings-detail-pane-body")

        assert panel.content_size.width <= detail_body.content_size.width
        assert panel.region.x >= detail_body.region.x
        assert panel.region.right <= detail_body.region.right

        screen.query_one("#settings-category-personal-context").focus()
        await pilot.press("a")
        await pilot.pause()
        assert panel.query("#personal-context-editor")
        panel.editor_mode = ""
        await pilot.pause()

        screen.query_one("#settings-category-personal-context").focus()
        await pilot.press("e")
        await pilot.pause()
        assert panel.query("#personal-context-editor")
        panel.editor_mode = ""
        await pilot.pause()

        screen.query_one("#settings-category-personal-context").focus()
        await pilot.press("d")
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        await pilot.click("#cancel-button")

        screen.query_one("#settings-category-personal-context").focus()
        await pilot.press("x")
        await pilot.pause()
        assert isinstance(host.screen, EnhancedFileSave)
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_local_removal_is_confirmed_and_only_start_fresh_recreates_storage(
    tmp_path,
) -> None:
    service = _real_service(tmp_path, with_record=True)
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        panel._confirm_remove_local()
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        assert "not Delete Everywhere" in host.screen.message
        await pilot.click("#confirm-button")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert service.status().state is ProfileOperationalState.REMOVED
        assert screen.query("#personal-context-start-fresh")
        assert not screen.query("#personal-context-delete-everywhere")

        await pilot.click("#personal-context-start-fresh")
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        await pilot.click("#confirm-button")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert service.status().state is ProfileOperationalState.DISABLED


@pytest.mark.asyncio
async def test_partial_local_removal_failure_redacts_stale_snapshot_and_reloads_removed(
    tmp_path, monkeypatch
) -> None:
    protector = InMemoryProfileKeyProtector()
    service = PersonalContextService(
        PersonalContextRepository(
            tmp_path / "personal-context.db", key_protector=protector
        )
    )
    service.create_profile()
    scope = service.list_scopes()[0]
    service.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="removal.canary", polarity="like", value="private removal canary"
        ),
        semantic_key=SemanticKey(namespace="preference", subject="removal.canary"),
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=AgentVisibility.AGENT_VISIBLE,
        ),
    )
    monkeypatch.setattr(
        protector,
        "delete",
        lambda _profile_ref: (_ for _ in ()).throw(
            ProfileLockedError("injected key deletion failure")
        ),
    )
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        panel = _active_destination_screen(host).query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )
        assert "private removal canary" in _visible_text(panel)

        panel._confirm_remove_local()
        await pilot.pause()
        await pilot.click("#confirm-button")
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)

        assert service.status().state is ProfileOperationalState.REMOVED
        assert "private removal canary" not in _visible_text(screen)
        assert screen.query("#personal-context-finish-removal")
        assert screen.query("#personal-context-start-fresh")


@pytest.mark.asyncio
async def test_removed_state_can_finish_secure_removal_without_starting_fresh() -> None:
    service = _ProfileServiceStub(
        PersonalContextSettingsSnapshot(status=_status(ProfileOperationalState.REMOVED))
    )
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(100, 30)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = _active_destination_screen(host)

        assert "retry deletion of the old encryption keys" in _visible_text(screen)
        assert screen.query("#personal-context-finish-removal")
        await pilot.click("#personal-context-finish-removal")
        await host.workers.wait_for_complete()

        assert service.finish_secure_removal_calls == 1
        assert service.snapshot.status.state is ProfileOperationalState.REMOVED


@pytest.mark.asyncio
async def test_exports_require_selected_destination_warning_and_masked_confirmation(
    tmp_path, monkeypatch
) -> None:
    service = _real_service(tmp_path, with_record=True)
    app = _build_test_app()
    app._personal_context_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)
        await host.workers.wait_for_complete()
        panel = screen.query_one(
            "#personal-context-settings-panel", PersonalContextSettingsPanel
        )

        plaintext_path = tmp_path / "profile.json"
        plaintext_prompts: list[object] = []

        async def plaintext_prompt(prompt):
            plaintext_prompts.append(prompt)
            return plaintext_path if len(plaintext_prompts) == 1 else True

        monkeypatch.setattr(host, "push_screen_wait", plaintext_prompt)
        await panel._choose_plaintext_export()
        await host.workers.wait_for_complete()
        assert isinstance(plaintext_prompts[0], EnhancedFileSave)
        assert isinstance(plaintext_prompts[1], ConfirmationDialog)
        assert "not encrypted" in plaintext_prompts[1].message
        assert plaintext_path.is_file()

        recovery_path = tmp_path / "recovery.json"
        recovery_prompts: list[object] = []

        async def recovery_prompt(prompt):
            recovery_prompts.append(prompt)
            return (
                recovery_path
                if len(recovery_prompts) == 1
                else "correct horse battery staple"
            )

        monkeypatch.setattr(host, "push_screen_wait", recovery_prompt)
        await panel._choose_recovery_export()
        await host.workers.wait_for_complete()
        assert isinstance(recovery_prompts[0], EnhancedFileSave)
        assert isinstance(recovery_prompts[1], RecoveryPassphraseDialog)
        assert recovery_path.is_file()


@pytest.mark.asyncio
async def test_recovery_passphrase_fields_are_masked() -> None:
    host = ConsolidatedCSSApp()
    async with host.run_test(size=(80, 24)):
        await host.push_screen(RecoveryPassphraseDialog())
        inputs = list(host.screen.query(Input))
        assert inputs
        assert all(widget.password for widget in inputs)
        labels = {
            str(widget.renderable)
            for widget in host.screen.query(".settings-input-label")
            if isinstance(widget, Static)
        }
        assert labels == {"Recovery passphrase", "Confirm recovery passphrase"}


@pytest.mark.asyncio
async def test_my_profile_is_discoverable_through_live_category_search() -> None:
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(_ready_snapshot())
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        screen = _active_destination_screen(host)

        await pilot.press("/")
        await pilot.pause()
        search = screen.query_one("#settings-category-search", Input)
        assert search.has_focus

        await pilot.press(*"my profile")
        await pilot.pause()
        await pilot.pause()

        assert screen._filtered_category_values() == ["personal-context"]
        assert screen.query_one("#settings-category-personal-context").display

        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert screen.active_category == SettingsCategoryId.PERSONAL_CONTEXT.value
        assert screen.query("#personal-context-settings-panel")


@pytest.mark.asyncio
async def test_my_profile_f1_help_advertises_exact_working_category_actions() -> None:
    app = _build_test_app()
    app._personal_context_service = _ProfileServiceStub(_ready_snapshot())
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "personal-context")
        screen = _active_destination_screen(host)

        screen.action_show_workbench_help()
        await pilot.pause()

        panel = host.screen
        assert isinstance(panel, WorkbenchHelpPanel)
        assert panel.state.shortcuts == SettingsScreen.PERSONAL_CONTEXT_SHORTCUTS
        assert panel.state.render_text().splitlines()[-4:] == [
            "- a: add record",
            "- e: edit record",
            "- d: delete record",
            "- x: export profile",
        ]
