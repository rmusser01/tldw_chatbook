"""Settings Tool Profiles presentation and workflow regressions."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static, TextArea

import pytest

from tldw_chatbook.Tool_Packs.service import (
    ToolProfileListing,
    ToolProfilePresentation,
)
from tldw_chatbook.Tool_Packs.activation import (
    InstalledToolProfile,
    ToolPackActivationResult,
)
from tldw_chatbook.Tool_Packs.binding import (
    ToolProfileBindingReview,
    ToolProfileBindingSummary,
)
from tldw_chatbook.Tool_Packs.contracts import PortableFallback, PortableToolRule
from tldw_chatbook.Tool_Packs.export import (
    ToolPackExportReview,
    ToolPackExportSnapshot,
)
from tldw_chatbook.Tool_Packs.publication import ToolPackPublicationResult
from tldw_chatbook.Tool_Packs.removal import (
    RemovedToolProfile,
    ToolProfileRemovalResult,
)
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.UI.Screens import settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.enhanced_file_picker import (
    EnhancedFileOpen,
    EnhancedFileSave,
)
from tldw_chatbook.Tool_Packs.importer import (
    MappedToolRule,
    ServerMapping,
    ToolPackImportReview,
)
from tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review import (
    ToolPackExportReviewModal,
    ToolPackImportOptions,
    ToolPackImportOptionsModal,
    ToolPackImportReviewModal,
    ToolProfileFirstBindReviewModal,
)
from tldw_chatbook.Workspaces.models import WorkspaceAssistantDefaults
from tldw_chatbook.Widgets.Settings_Widgets.tool_profiles_panel import (
    ToolProfilesPanel,
)

from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
)


def _profile(
    profile_id: str,
    *,
    origin: str = "local",
    lifecycle_valid: bool = True,
    binding_state: str = "bound",
    receipt_health: str = "not_applicable",
    removal_eligible: bool = True,
    removal_blocker: str | None = None,
    revision: int | None = 3,
    policy_digest: str | None = "a" * 64,
    references: tuple[int, int] = (0, 0),
) -> ToolProfilePresentation:
    return ToolProfilePresentation(
        profile_id=profile_id,
        origin=origin,  # type: ignore[arg-type]
        lifecycle_valid=lifecycle_valid,
        binding_state=binding_state,  # type: ignore[arg-type]
        first_bind_confirmation_required=binding_state == "unbound",
        reference_counts=references,
        posture_counts=(4, 3, 2),
        receipt_health=receipt_health,  # type: ignore[arg-type]
        removal_eligible=removal_eligible,
        removal_blocker=removal_blocker,
        revision=revision,
        policy_digest=policy_digest,
    )


class _PanelHarness(App[None]):
    def __init__(self, panel: ToolProfilesPanel) -> None:
        super().__init__()
        self.panel = panel
        self.events: list[object] = []

    def compose(self) -> ComposeResult:
        yield self.panel

    def on_tool_profiles_panel_import_requested(
        self, event: ToolProfilesPanel.ImportRequested
    ) -> None:
        self.events.append(event)

    def on_tool_profiles_panel_export_requested(
        self, event: ToolProfilesPanel.ExportRequested
    ) -> None:
        self.events.append(event)

    def on_tool_profiles_panel_edit_policy_requested(
        self, event: ToolProfilesPanel.EditPolicyRequested
    ) -> None:
        self.events.append(event)

    def on_tool_profiles_panel_bind_requested(
        self, event: ToolProfilesPanel.BindRequested
    ) -> None:
        self.events.append(event)

    def on_tool_profiles_panel_remove_requested(
        self, event: ToolProfilesPanel.RemoveRequested
    ) -> None:
        self.events.append(event)


class _ModalHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield Button("Open", id="open")


class _WorkflowService:
    def __init__(self, listing: ToolProfileListing) -> None:
        self.listing = listing
        self.calls: list[tuple[object, ...]] = []

    def list_profiles(self) -> ToolProfileListing:
        return self.listing

    def inspect_import(
        self,
        archive_path: Path,
        *,
        destination_id: str,
        mappings: tuple[ServerMapping, ...],
    ) -> ToolPackImportReview:
        self.calls.append(("inspect", destination_id, mappings))
        return replace(
            _import_review(),
            archive_path=archive_path,
            destination_id=destination_id,
            mappings=mappings,
        )

    def import_unbound(self, review: ToolPackImportReview) -> ToolPackActivationResult:
        self.calls.append(("import", review.destination_id))
        return ToolPackActivationResult(
            InstalledToolProfile(
                review.destination_id,
                "a" * 64,
                1,
                "tp-" + "1" * 32,
            ),
            "store-generation",
        )

    def capture_export(self, profile_id: str, **context) -> ToolPackExportReview:
        self.calls.append(("capture", profile_id, context))
        return _export_review()

    def publish_export(
        self, _review: ToolPackExportReview, _destination: object, **context
    ) -> ToolPackPublicationResult:
        self.calls.append(("publish", context, context["cancelled"]()))
        return ToolPackPublicationResult("9" * 64, True, False)

    def remove_profile(
        self, profile_id: str, *, expected_revision: int
    ) -> ToolProfileRemovalResult:
        self.calls.append(("remove", profile_id, expected_revision))
        return ToolProfileRemovalResult(
            RemovedToolProfile(
                profile_id,
                "tool_pack_tombstone",
                expected_revision + 1,
                "b" * 64,
                "tp-" + "2" * 32,
                "c" * 64,
            ),
            "store-generation",
        )


def _text(panel: ToolProfilesPanel) -> str:
    return "\n".join(str(widget.renderable) for widget in panel.query(Static))


def _rule(
    server_key: str,
    tool_name: str,
    state: str,
    *,
    digest: str | None = "b" * 64,
) -> PortableToolRule:
    return PortableToolRule(
        "mcp",
        server_key,
        tool_name,
        state,  # type: ignore[arg-type]
        digest,
    )


def _import_review() -> ToolPackImportReview:
    connected = _rule("source:one", "lookup", "allow")
    cached = _rule("source:two", "write", "ask")
    changed = _rule("source:three", "changed", "allow")
    missing = _rule("source:four", "missing", "deny", digest=None)
    omitted = _rule("source:five", "omitted", "ask")
    return ToolPackImportReview(
        archive_path=Path("/private/never-render-this/pack.tldw-tool-pack"),
        archive_sha256="1" * 64,
        manifest_sha256="2" * 64,
        payload_sha256="3" * 64,
        destination_id="research",
        store_generation="store-generation",
        inventory_digest="4" * 64,
        mappings=(ServerMapping("source:one", "destination:one"),),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        matched=(
            MappedToolRule(
                connected,
                ("mcp", "destination:one", "lookup"),
                "b" * 64,
                True,
            ),
            MappedToolRule(
                cached,
                ("mcp", "source:two", "write"),
                "b" * 64,
                False,
            ),
        ),
        changed=(changed,),
        missing=(missing,),
        pending_denies=(missing,),
        omitted_allow_ask=(changed, omitted),
        content_digest="5" * 64,
        display_name="[bold]Research tools[/bold]",
        producer=("publisher", "1.2.3"),
        fallbacks=(
            PortableFallback("mcp", "*", "ask"),
            PortableFallback("builtin", "agent:builtin", "deny"),
        ),
    )


def _binding_review() -> ToolProfileBindingReview:
    return ToolProfileBindingReview(
        workspace_id="workspace-1",
        action="replace",
        intended_defaults_digest="6" * 64,
        profile_id="research",
        policy_digest="a" * 64,
        revision=3,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        summary=ToolProfileBindingSummary(
            global_fallback="ask",
            builtin_fallback="deny",
            allow_server_fallbacks=("mcp:source:one",),
            stored_exact_allows=(("mcp:source:one", "lookup"),),
            effective_allows=(("mcp:source:one", "lookup"),),
            unavailable_allows=(("mcp:source:two", "cached"),),
            downgraded_allows=(("mcp:source:three", "changed"),),
            high_risk_allows=(("mcp:source:one", "lookup"),),
            allow_count=1,
            ask_count=2,
            deny_count=3,
            inventory_digest="4" * 64,
            effective_asks=(("mcp:source:four", "confirm"),),
            effective_denies=(("mcp:source:five", "blocked"),),
        ),
    )


def _export_review() -> ToolPackExportReview:
    return ToolPackExportReview(
        snapshot=ToolPackExportSnapshot(
            manifest=type(
                "Manifest",
                (),
                {
                    "display_name": "Research tools",
                    "suggested_id": "research",
                    "producer_name": "tldw_chatbook",
                    "producer_version": "1",
                    "content_digest": "5" * 64,
                },
            )(),  # type: ignore[arg-type]
            payload=type(
                "Payload",
                (),
                {
                    "fallbacks": (
                        PortableFallback("mcp", "*", "ask"),
                        PortableFallback("builtin", "agent:builtin", "deny"),
                    ),
                    "rules": (
                        _rule("source:one", "lookup", "allow"),
                        _rule("source:two", "write", "ask"),
                        _rule("source:three", "erase", "deny"),
                    ),
                },
            )(),  # type: ignore[arg-type]
        ),
        inventory_digest="4" * 64,
        excluded_counts=(("unsupported_authority", 2),),
        omitted_allow_ask=(("source:gone", "missing"),),
        pending_denies=(("source:gone", "old-deny"),),
    )


@pytest.mark.asyncio
async def test_panel_lists_truthful_profiles_without_policy_editor() -> None:
    listing = ToolProfileListing(
        profiles=(
            _profile(
                "default",
                removal_eligible=False,
                removal_blocker="default_profile",
                references=(2, 1),
            ),
            _profile(
                "research",
                origin="imported",
                binding_state="unbound",
                receipt_health="available",
            ),
            _profile(
                "ws-w-1",
                origin="workspace-managed",
                lifecycle_valid=False,
                removal_eligible=False,
                removal_blocker="profile_invalid",
                revision=None,
                policy_digest=None,
            ),
        )
    )
    panel = ToolProfilesPanel(listing)

    async with _PanelHarness(panel).run_test(size=(120, 36)) as pilot:
        await pilot.pause()

        assert panel.profile_ids == ("default", "research", "ws-w-1")
        assert panel.row("research").origin == "Imported Tool Pack"
        assert panel.row("research").first_bind_confirmation_required is True
        assert panel.row("research").reference_counts == (0, 0)
        assert panel.row("default").reference_counts == (2, 1)
        assert panel.has_policy_editor is False
        assert not panel.query(Input)
        assert not panel.query(TextArea)
        visible = _text(panel)
        assert "Imported Tool Pack" in visible
        assert "Workspace-managed" in visible
        assert "2 active · 1 archived" in visible
        assert "Invalid policy lifecycle" in visible
        assert "First bind review required" in visible

        for action in ("export", "edit", "bind", "remove"):
            assert panel.query_one(f"#tool-profile-{action}-2", Button).disabled, action


@pytest.mark.asyncio
async def test_profile_ids_render_as_plain_text() -> None:
    profile_id = "[bold]local-profile[/bold]"
    panel = ToolProfilesPanel(
        ToolProfileListing(
            profiles=(
                _profile(
                    profile_id,
                    removal_eligible=False,
                    removal_blocker="not_imported",
                    revision=None,
                ),
            )
        )
    )

    async with _PanelHarness(panel).run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        title = panel.query_one(".tool-profile-title", Static)
        assert str(title.renderable) == profile_id
        assert title.render().plain == profile_id


@pytest.mark.asyncio
async def test_receipt_unavailable_preserves_non_removal_actions() -> None:
    panel = ToolProfilesPanel(
        ToolProfileListing(
            profiles=(
                _profile(
                    "portable",
                    origin="imported",
                    binding_state="unbound",
                    receipt_health="unavailable",
                    removal_eligible=False,
                    removal_blocker="receipt_unavailable",
                ),
            )
        )
    )

    async with _PanelHarness(panel).run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        assert "Receipt unavailable" in _text(panel)
        for action in ("export", "edit", "bind"):
            assert not panel.query_one(f"#tool-profile-{action}-0", Button).disabled, (
                action
            )
        assert panel.query_one("#tool-profile-remove-0", Button).disabled


@pytest.mark.asyncio
async def test_panel_emits_exact_captured_action_contexts() -> None:
    profile = _profile("research", origin="imported", binding_state="unbound")
    panel = ToolProfilesPanel(ToolProfileListing(profiles=(profile,)))
    app = _PanelHarness(panel)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.click("#tool-profiles-import")
        await pilot.click("#tool-profile-export-0")
        await pilot.click("#tool-profile-edit-0")
        await pilot.click("#tool-profile-bind-0")
        await pilot.click("#tool-profile-remove-0")
        await pilot.pause()

    assert isinstance(app.events[0], ToolProfilesPanel.ImportRequested)
    for event_type, event in zip(
        (
            ToolProfilesPanel.ExportRequested,
            ToolProfilesPanel.EditPolicyRequested,
            ToolProfilesPanel.BindRequested,
            ToolProfilesPanel.RemoveRequested,
        ),
        app.events[1:],
        strict=True,
    ):
        assert isinstance(event, event_type)
        assert event.profile_id == "research"
        assert event.revision == 3
        assert event.policy_digest == "a" * 64


@pytest.mark.asyncio
async def test_panel_has_explicit_empty_and_unavailable_states() -> None:
    empty = ToolProfilesPanel(ToolProfileListing())
    async with _PanelHarness(empty).run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        assert "No visible Tool profiles" in _text(empty)
        assert not empty.query(".tool-profile-row")

    unavailable = ToolProfilesPanel(
        ToolProfileListing(unavailable_category="store_invalid")
    )
    async with _PanelHarness(unavailable).run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        assert "Profiles unavailable · store_invalid" in _text(unavailable)
        assert unavailable.query_one("#tool-profiles-import", Button).disabled


@pytest.mark.asyncio
async def test_panel_can_apply_one_fresh_immutable_listing() -> None:
    panel = ToolProfilesPanel(ToolProfileListing(unavailable_category="loading"))
    async with _PanelHarness(panel).run_test(size=(90, 24)) as pilot:
        await panel.apply_listing(ToolProfileListing(profiles=(_profile("fresh"),)))
        await pilot.pause()
        assert panel.profile_ids == ("fresh",)
        assert "Profiles unavailable" not in _text(panel)
        assert panel.query_one("#tool-profile-edit-0", Button).disabled is False


@pytest.mark.asyncio
async def test_canonical_settings_category_loads_app_owned_listing_off_thread() -> None:
    listing = ToolProfileListing(
        profiles=(_profile("research", origin="imported", binding_state="unbound"),)
    )
    app = _build_test_app()
    app.tool_pack_service = type(
        "ToolPackServiceStub", (), {"list_profiles": lambda self: listing}
    )()
    app.tool_pack_service_unavailable_reason = None
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        panel = screen.query_one("#settings-tool-profiles-panel", ToolProfilesPanel)
        assert panel.profile_ids == ("research",)
        assert screen.active_category == "tool-profiles"


@pytest.mark.asyncio
async def test_tool_profiles_category_retries_failed_service_composition_once() -> None:
    app = _build_test_app()
    app.tool_pack_service = None
    app.tool_pack_service_unavailable_reason = "composition_unavailable"
    attempts: list[bool] = []

    def retry() -> None:
        attempts.append(True)
        app.tool_pack_service_unavailable_reason = "starting"

    app._deferred_wire_tool_pack_service = retry
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    assert attempts == [True]


@pytest.mark.asyncio
async def test_tool_profiles_first_use_refreshes_after_composition() -> None:
    listing = ToolProfileListing(
        profiles=(_profile("research", origin="imported", binding_state="unbound"),)
    )
    app = _build_test_app()
    app.tool_pack_service = None
    app.tool_pack_service_unavailable_reason = "not_ready"

    class PendingComposition:
        async def wait(self) -> None:
            app.tool_pack_service = type(
                "ToolPackServiceStub", (), {"list_profiles": lambda self: listing}
            )()
            app.tool_pack_service_unavailable_reason = None

    app._deferred_wire_tool_pack_service = PendingComposition
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        panel = screen.query_one("#settings-tool-profiles-panel", ToolProfilesPanel)
        assert panel.profile_ids == ("research",)


@pytest.mark.asyncio
async def test_tool_profiles_refresh_when_settings_resumes_after_policy_edit() -> None:
    service = _WorkflowService(
        ToolProfileListing(
            profiles=(_profile("research", origin="imported", binding_state="unbound"),)
        )
    )
    app = _build_test_app()
    app.tool_pack_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        service.listing = ToolProfileListing(
            profiles=(
                _profile(
                    "research",
                    origin="imported",
                    binding_state="unbound",
                    revision=4,
                    policy_digest="b" * 64,
                ),
            )
        )
        screen.on_screen_resume()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        panel = screen.query_one("#settings-tool-profiles-panel", ToolProfilesPanel)
        assert panel.row("research").revision == 4
        assert panel.row("research").policy_digest == "b" * 64


@pytest.mark.asyncio
async def test_listing_worker_does_not_surface_private_exception_categories() -> None:
    class SensitiveFailure(RuntimeError):
        category = "/private/path/API_KEY=secret"

    def fail_listing(_self):
        raise SensitiveFailure("credential contents")

    app = _build_test_app()
    app.tool_pack_service = type(
        "ToolPackServiceStub", (), {"list_profiles": fail_listing}
    )()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        panel = screen.query_one("#settings-tool-profiles-panel", ToolProfilesPanel)
        visible = _text(panel)
        assert "Profiles unavailable · store_invalid" in visible
        assert "private" not in visible.casefold()
        assert "secret" not in visible.casefold()


def test_tool_profiles_category_is_searchable_by_user_vocabulary() -> None:
    screen = SettingsScreen(_build_test_app())
    matches = screen._filtered_category_summaries("portable tool permission")

    assert matches
    assert matches[0].category.value == "tool-profiles"


@pytest.mark.asyncio
async def test_tool_profile_actions_remain_focusable_at_narrow_size() -> None:
    listing = ToolProfileListing(
        profiles=(_profile("research", origin="imported", binding_state="unbound"),)
    )
    app = _build_test_app()
    app.tool_pack_service = _WorkflowService(listing)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        for action in ("export", "edit", "bind", "remove"):
            button = screen.query_one(f"#tool-profile-{action}-0", Button)
            assert not button.disabled
            assert button.can_focus
            assert button.display
            assert button.region.width > 0
            assert button.region.x + button.region.width <= screen.size.width


@pytest.mark.asyncio
async def test_edit_policy_deep_link_captures_exact_profile_authority() -> None:
    listing = ToolProfileListing(
        profiles=(_profile("research", origin="imported", binding_state="unbound"),)
    )
    app = _build_test_app()
    app.tool_pack_service = type(
        "ToolPackServiceStub", (), {"list_profiles": lambda self: listing}
    )()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#tool-profile-edit-0")
        await pilot.pause()

    assert host.seen_routes[-1] == "mcp"
    assert host.seen_contexts[-1] == {
        "mode": "permissions",
        "tool_policy_profile_id": "research",
        "profile_revision": 3,
        "profile_policy_digest": "a" * 64,
    }


@pytest.mark.asyncio
async def test_import_review_discloses_policy_identity_mapping_and_no_install() -> None:
    review = _import_review()
    app = _ModalHarness()
    async with app.run_test(size=(100, 32)) as pilot:
        app.push_screen(ToolPackImportReviewModal(review))
        await pilot.pause()

        modal = app.screen
        text = "\n".join(str(widget.renderable) for widget in modal.query(Static))
        assert "[bold]Research tools[/bold]" in text
        assert "Producer: publisher 1.2.3" in text
        assert f"Content digest: {'5' * 64}" in text
        assert "Proposed profile id: research" in text
        assert "Fallback · mcp/*: Ask" in text
        assert "Fallback · builtin/agent:builtin: Deny" in text
        assert "Source rules · Allow 2 · Ask 2 · Deny 1" in text
        assert "Exact matches: 2" in text
        assert "Changed contracts: 1" in text
        assert "Missing tools: 1" in text
        assert "Pending Denies: 1" in text
        assert "Omitted Ask/Allow: 2" in text
        assert "destination:one/lookup · connected" in text
        assert "source:two/write · disconnected (cached definition)" in text
        assert "source:one → destination:one" in text
        assert "Imported unbound" in text
        assert "does not install tools" in text
        assert "/private/never-render-this" not in text

        buttons = tuple(modal.query(Button))
        assert {button.label.plain for button in buttons} == {
            "Cancel",
            "Change id or mappings",
            "Import unbound profile",
        }
        assert not any("bind" in (button.id or "") for button in buttons)


@pytest.mark.asyncio
async def test_import_review_returns_only_explicit_unbound_confirmation() -> None:
    app = _ModalHarness()
    results: list[ToolPackImportReview | None] = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(ToolPackImportReviewModal(_import_review()), results.append)
        await pilot.pause()
        assert app.focused is app.screen.query_one("#tool-pack-import-unbound", Button)
        await pilot.press("enter")
        await pilot.pause()

    assert len(results) == 1
    assert results[0] is not None
    assert results[0].destination_id == "research"


@pytest.mark.asyncio
async def test_import_options_capture_destination_and_explicit_mappings() -> None:
    results: list[ToolPackImportOptions | None] = []
    app = _ModalHarness()
    options = ToolPackImportOptions(
        destination_id="research",
        mappings=(ServerMapping("source:old", "destination:old"),),
    )

    async with app.run_test(size=(90, 28)) as pilot:
        app.push_screen(ToolPackImportOptionsModal(options), results.append)
        await pilot.pause()
        app.screen.query_one(
            "#tool-pack-import-profile-id", Input
        ).value = "research-v2"
        app.screen.query_one(
            "#tool-pack-import-server-mappings", TextArea
        ).text = "source:one -> destination:one\nsource:two=destination:two"
        await pilot.click("#tool-pack-import-options-review")
        await pilot.pause()

    assert results == [
        ToolPackImportOptions(
            destination_id="research-v2",
            mappings=(
                ServerMapping("source:one", "destination:one"),
                ServerMapping("source:two", "destination:two"),
            ),
        )
    ]


@pytest.mark.asyncio
async def test_import_review_change_action_never_returns_a_commit_review() -> None:
    results: list[ToolPackImportReview | str | None] = []
    app = _ModalHarness()

    async with app.run_test(size=(90, 28)) as pilot:
        app.push_screen(ToolPackImportReviewModal(_import_review()), results.append)
        await pilot.pause()
        await pilot.click("#tool-pack-import-revise")
        await pilot.pause()

    assert results == ["revise"]


@pytest.mark.asyncio
async def test_export_review_discloses_portable_scope_and_exact_source() -> None:
    review = _export_review()
    results: list[ToolPackExportReview | None] = []
    app = _ModalHarness()

    async with app.run_test(size=(100, 34)) as pilot:
        app.push_screen(
            ToolPackExportReviewModal(
                review,
                profile_id="research",
                revision=3,
                policy_digest="a" * 64,
            ),
            results.append,
        )
        await pilot.pause()
        visible = "\n".join(
            str(widget.renderable) for widget in app.screen.query(Static)
        )
        assert "Profile: research · revision 3" in visible
        assert f"Policy digest: {'a' * 64}" in visible
        assert "Name: Research tools" in visible
        assert "Producer: tldw_chatbook 1" in visible
        assert "Allow 1 · Ask 1 · Deny 1" in visible
        assert "Omitted Ask/Allow: 1" in visible
        assert "Pending Denies: 1" in visible
        assert "unsupported_authority: 2" in visible
        assert "permission policy only" in visible
        assert "does not include or install tools" in visible
        await pilot.click("#tool-pack-export-continue")
        await pilot.pause()

    assert results == [review]


@pytest.mark.asyncio
async def test_first_bind_review_discloses_exact_authority_and_separate_memory_gate() -> (
    None
):
    review = _binding_review()
    intended = WorkspaceAssistantDefaults(
        assistant_kind="persona",
        assistant_id="persona-1",
        persona_memory_mode="read_write",
        tool_policy_profile_id="research",
    )
    app = _ModalHarness()

    async with app.run_test(size=(110, 38)) as pilot:
        app.push_screen(ToolProfileFirstBindReviewModal(review, intended))
        await pilot.pause()
        modal = app.screen
        visible = "\n".join(str(widget.renderable) for widget in modal.query(Static))

        assert "Workspace: workspace-1" in visible
        assert "Action: Replace" in visible
        assert "Profile: research · revision 3" in visible
        assert f"Policy digest: {'a' * 64}" in visible
        assert "Persona: persona-1" in visible
        assert "Memory: read_write" in visible
        assert "Assistant kind: persona" in visible
        assert "Voice: Default" in visible
        assert "Style: Default" in visible
        assert "Global fallback: Ask" in visible
        assert "Built-in fallback: Deny" in visible
        assert "Allow 1 · Ask 2 · Deny 3" in visible
        assert "Unavailable allows: mcp:source:two/cached" in visible
        assert "Downgraded allows: mcp:source:three/changed" in visible
        assert "High-risk allows: mcp:source:one/lookup" in visible
        assert "mcp:source:four/confirm" in visible
        assert "mcp:source:five/blocked" in visible
        assert "separate from the read_write memory acknowledgement" in visible
        assert "Any change requires a new review" in visible
        assert not modal.query("Checkbox")


@pytest.mark.asyncio
async def test_first_bind_review_returns_only_the_exact_review_object() -> None:
    review = _binding_review()
    intended = WorkspaceAssistantDefaults(
        assistant_kind="persona",
        assistant_id="persona-1",
        persona_memory_mode="read_only",
        tool_policy_profile_id="research",
    )
    results: list[ToolProfileBindingReview | None] = []
    app = _ModalHarness()

    async with app.run_test(size=(90, 30)) as pilot:
        app.push_screen(
            ToolProfileFirstBindReviewModal(review, intended),
            results.append,
        )
        await pilot.pause()
        assert app.focused is app.screen.query_one("#tool-profile-bind-confirm", Button)
        await pilot.press("enter")
        await pilot.pause()

    assert results == [review]


def test_management_operations_are_exclusive_workers() -> None:
    for name in (
        "_tool_profile_import_flow",
        "_tool_profile_export_flow",
        "_tool_profile_remove_flow",
    ):
        assert getattr(SettingsScreen.__dict__[name], "__wrapped__", None) is not None


@pytest.mark.asyncio
async def test_import_worker_reinspects_revised_options_then_imports_unbound(
    tmp_path: Path,
) -> None:
    listing = ToolProfileListing(
        profiles=(_profile("research", origin="imported", binding_state="unbound"),)
    )
    service = _WorkflowService(listing)
    app = _build_test_app()
    app.tool_pack_service = service
    host = DestinationHarness(app, "settings")
    archive = tmp_path / "research.tldw-tool-pack"
    archive.touch()
    review_count = 0

    async def choose(screen):
        nonlocal review_count
        if isinstance(screen, EnhancedFileOpen):
            return archive
        if isinstance(screen, ToolPackImportOptionsModal):
            if review_count:
                return ToolPackImportOptions(
                    "research-v2",
                    (ServerMapping("source:one", "destination:one"),),
                )
            return screen.options
        if isinstance(screen, ToolPackImportReviewModal):
            review_count += 1
            return "revise" if review_count == 1 else screen.review
        raise AssertionError(type(screen).__name__)

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        host.push_screen_wait = AsyncMock(side_effect=choose)
        await pilot.click("#tool-profiles-import")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert service.calls[:2] == [
            ("inspect", "research", ()),
            (
                "inspect",
                "research-v2",
                (ServerMapping("source:one", "destination:one"),),
            ),
        ]
        assert service.calls[2] == ("import", "research-v2")
        assert "Imported research-v2 unbound" in screen._tool_profiles_result


@pytest.mark.asyncio
async def test_export_worker_uses_exact_profile_context_and_captured_destination(
    tmp_path: Path,
) -> None:
    listing = ToolProfileListing(
        profiles=(_profile("research", origin="imported", binding_state="unbound"),)
    )
    service = _WorkflowService(listing)
    app = _build_test_app()
    app.tool_pack_service = service
    host = DestinationHarness(app, "settings")
    destination = tmp_path.resolve() / "research.tldw-tool-pack"

    async def choose(screen):
        if isinstance(screen, ToolPackExportReviewModal):
            return screen.review
        if isinstance(screen, EnhancedFileSave):
            return destination
        raise AssertionError(type(screen).__name__)

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        host.push_screen_wait = AsyncMock(side_effect=choose)
        await pilot.click("#tool-profile-export-0")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert service.calls[0] == (
            "capture",
            "research",
            {
                "display_name": "research",
                "suggested_id": "research",
                "expected_revision": 3,
                "expected_policy_digest": "a" * 64,
            },
        )
        assert service.calls[1][0] == "publish"
        assert service.calls[1][2] is False
        assert "overwrite_token" not in service.calls[1][1]
        assert "Exported Tool Pack" in screen._tool_profiles_result


@pytest.mark.asyncio
async def test_removal_worker_confirms_and_uses_the_rendered_revision() -> None:
    listing = ToolProfileListing(
        profiles=(_profile("research", origin="imported", binding_state="unbound"),)
    )
    service = _WorkflowService(listing)
    app = _build_test_app()
    app.tool_pack_service = service
    host = DestinationHarness(app, "settings")

    async def choose(screen):
        assert isinstance(screen, ConfirmationDialog)
        return True

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        host.push_screen_wait = AsyncMock(side_effect=choose)
        screen.query_one("#tool-profile-remove-0", Button).press()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert service.calls[0] == ("remove", "research", 3)
        assert "id permanently reserved" in screen._tool_profiles_result


@pytest.mark.asyncio
async def test_removal_worker_rechecks_cancellation_after_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listing = ToolProfileListing(
        profiles=(_profile("research", origin="imported", binding_state="unbound"),)
    )
    service = _WorkflowService(listing)
    app = _build_test_app()
    app.tool_pack_service = service
    host = DestinationHarness(app, "settings")
    monkeypatch.setattr(
        settings_screen_module,
        "get_current_worker",
        lambda: SimpleNamespace(is_cancelled=True),
    )

    async def choose(screen):
        assert isinstance(screen, ConfirmationDialog)
        return True

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        host.push_screen_wait = AsyncMock(side_effect=choose)
        screen.query_one("#tool-profile-remove-0", Button).press()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert not service.calls
        assert screen._tool_profiles_result == "Remove cancelled"


@pytest.mark.asyncio
async def test_removal_confirmation_does_not_interpolate_the_profile_id() -> None:
    profile_id = "[bold]local-profile[/bold]"
    listing = ToolProfileListing(
        profiles=(_profile(profile_id, origin="imported", binding_state="unbound"),)
    )
    service = _WorkflowService(listing)
    app = _build_test_app()
    app.tool_pack_service = service
    host = DestinationHarness(app, "settings")

    async def choose(screen):
        assert isinstance(screen, ConfirmationDialog)
        assert profile_id not in screen.message
        assert "local-profile" in screen.message
        return False

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        host.push_screen_wait = AsyncMock(side_effect=choose)
        screen.query_one("#tool-profile-remove-0", Button).press()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert not service.calls
        assert screen._tool_profiles_result == "Remove cancelled"


@pytest.mark.asyncio
async def test_bind_request_stages_profile_in_active_workspace_without_binding() -> (
    None
):
    listing = ToolProfileListing(
        profiles=(_profile("research", origin="imported", binding_state="unbound"),)
    )
    service = _WorkflowService(listing)
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-active", name="Active")
    registry.set_active_workspace("ws-active")
    app.tool_pack_service = service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        screen.query_one("#tool-profile-bind-0", Button).press()
        await pilot.pause(0.3)

        assert screen.active_category == "workspaces"
        assert screen._settings_selected_workspace_id == "ws-active"
        assert screen._settings_workspace_assistant_pending == {
            "workspace_id": "ws-active",
            "persona_id": None,
            "memory_mode": "read_only",
            "profile_id": "research",
        }
        assert registry.get_workspace("ws-active").assistant_defaults is None
        assert "staged" in screen._settings_workspaces_result.casefold()


@pytest.mark.asyncio
async def test_bind_request_without_explicit_workspace_shows_recovery_guidance() -> (
    None
):
    listing = ToolProfileListing(
        profiles=(_profile("research", origin="imported", binding_state="unbound"),)
    )
    app = _build_test_app()
    app.tool_pack_service = _WorkflowService(listing)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await pilot.app.workers.wait_for_complete()
        screen.query_one("#tool-profile-bind-0", Button).press()
        await pilot.pause(0.3)

        assert screen.active_category == "workspaces"
        result = screen.query_one("#settings-workspaces-result", Static)
        assert "Choose a non-default workspace" in str(result.renderable)


def test_publication_unsupported_has_a_distinct_truthful_outcome() -> None:
    copy = SettingsScreen._tool_pack_failure_copy(
        "export", ToolPackError("export", "publication_unsupported")
    )
    assert "publication_unsupported" in copy
    assert "publication_failed" not in copy
