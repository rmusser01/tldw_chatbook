from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Input, Static
from tldw_profile_core import (
    ActorType,
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileProposal,
    ProfileProvenance,
    ProfileRecord,
    ProfileScope,
    ProposalOperation,
    RecordState,
    ScopeKind,
    SemanticKey,
    SyncMode,
)

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Personal_Context.runtime_policy import AgentAuthority
from tldw_chatbook.Personal_Context.service import (
    PersonalContextSettingsSnapshot,
    ProfileConflictError,
    ProfileOperationalState,
    ProfileOperationalStatus,
    SettingsScopeSnapshot,
)
from tldw_chatbook.Widgets.Settings_Widgets.personal_context_panel import (
    PersonalContextSettingsPanel,
)
from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
    PersonalContextProposalReviewModal,
    ProposalReviewResult,
)


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


def _record(
    value: str = "concise",
    *,
    record_id: str = "record-proposed",
    visibility: AgentVisibility = AgentVisibility.AGENT_VISIBLE,
) -> ProfileRecord:
    record = ProfileRecord(
        profile_id="profile-1",
        record_id=record_id,
        scope_id="scope-global",
        kind="preference",
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value=value
        ),
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
        state=RecordState.ACTIVE,
        controls=ProfileControls(
            sync_mode=SyncMode.SYNCABLE,
            agent_visibility=visibility,
        ),
        provenance=ProfileProvenance(
            source="agent", actor="agent", reason_code="conversation_learning"
        ),
        version_id="record-version-proposed",
        parent_version_id=None,
        created_at=NOW,
        updated_at=NOW,
    )
    return record


def _proposal(value: str = "concise") -> ProfileProposal:
    record = _record(value)
    return ProfileProposal(
        proposal_id="proposal-1",
        profile_id="profile-1",
        scope_id="scope-global",
        operation=ProposalOperation.CREATE,
        target_record_id=None,
        base_version_id=None,
        proposed_record=record,
        provenance=ProfileProvenance(
            source="agent",
            actor="agent",
            reason_code="conversation_learning",
            source_references=("message-1",),
            source_hashes=("0" * 64,),
        ),
        created_at=NOW,
        expires_at=NOW + timedelta(days=90),
    )


def _existing_record_proposal(
    operation: ProposalOperation = ProposalOperation.ARCHIVE,
) -> tuple[ProfileProposal, ProfileRecord]:
    target = _record("target value", record_id="record-target")
    return (
        ProfileProposal(
            proposal_id="proposal-existing",
            profile_id="profile-1",
            scope_id=target.scope_id,
            operation=operation,
            target_record_id=target.record_id,
            base_version_id=target.version_id,
            proposed_record=None,
            provenance=ProfileProvenance(
                source="agent",
                actor="agent",
                reason_code="conversation_learning",
                source_references=("message-1",),
                source_hashes=("0" * 64,),
            ),
            created_at=NOW,
            expires_at=NOW + timedelta(days=90),
        ),
        target,
    )


class _ProposalService:
    def __init__(self, proposal: ProfileProposal) -> None:
        self.proposal = proposal
        self.accept_calls: list[dict] = []
        self.reject_calls: list[str] = []

    def accept(self, proposal_id: str, **kwargs):
        self.accept_calls.append({"proposal_id": proposal_id, **kwargs})
        assert self.proposal.proposed_record is not None
        return self.proposal.proposed_record

    def reject(self, proposal_id: str):
        self.reject_calls.append(proposal_id)
        return self.proposal


class _BlockingProposalService(_ProposalService):
    def __init__(self, proposal: ProfileProposal) -> None:
        super().__init__(proposal)
        self.entered = threading.Event()
        self.release = threading.Event()

    def accept(self, proposal_id: str, **kwargs):
        self.entered.set()
        assert self.release.wait(5)
        return super().accept(proposal_id, **kwargs)


class _FailingProposalService(_ProposalService):
    def __init__(self, proposal: ProfileProposal, error: Exception) -> None:
        super().__init__(proposal)
        self.error = error

    def accept(self, proposal_id: str, **kwargs):
        del proposal_id, kwargs
        raise self.error


class _Host(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[ProposalReviewResult | None] = []


@pytest.mark.asyncio
async def test_agent_proposal_review_shows_source_scope_and_private_safety() -> None:
    proposal = _proposal()
    service = _ProposalService(proposal)
    host = _Host()

    async with host.run_test(size=(100, 32)) as pilot:
        modal = PersonalContextProposalReviewModal(
            service,
            proposal=proposal,
            scope_label="Global",
        )
        await host.push_screen(modal, callback=host.results.append)
        await pilot.pause()

        text = " ".join(
            str(widget.renderable) for widget in modal.query(Static)
        ).lower()
        assert "agent proposal" in text
        assert "global" in text
        assert "create" in text
        assert "private record" in text
        assert "private-hidden-canary" not in text
        assert modal.query_one("#personal-context-proposal-value", Input).value == (
            "concise"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", [ProposalOperation.ARCHIVE, ProposalOperation.PROMOTE]
)
async def test_existing_record_proposal_names_exact_eligible_target(operation) -> None:
    proposal, target = _existing_record_proposal(operation)
    host = _Host()

    async with host.run_test(size=(100, 32)) as pilot:
        modal = PersonalContextProposalReviewModal(
            _ProposalService(proposal),
            proposal=proposal,
            scope_label="Global",
            target_record=target,
        )
        await host.push_screen(modal, callback=host.results.append)
        await pilot.pause()

        text = " ".join(str(widget.renderable) for widget in modal.query(Static))
        assert "response.detail" in text
        assert "target value" in text
        assert not modal.query_one("#personal-context-proposal-accept", Button).disabled


@pytest.mark.asyncio
async def test_existing_record_proposal_never_discloses_user_only_target() -> None:
    proposal, _ = _existing_record_proposal()
    private_target = _record(
        "PRIVATE-TARGET-CANARY",
        record_id="record-target",
        visibility=AgentVisibility.USER_ONLY,
    )
    host = _Host()

    async with host.run_test(size=(100, 32)) as pilot:
        modal = PersonalContextProposalReviewModal(
            _ProposalService(proposal),
            proposal=proposal,
            scope_label="Global",
            target_record=private_target,
        )
        await host.push_screen(modal, callback=host.results.append)
        await pilot.pause()

        text = " ".join(str(widget.renderable) for widget in modal.query(Static))
        assert "PRIVATE-TARGET-CANARY" not in text
        assert "target unavailable" in text.lower()
        assert modal.query_one("#personal-context-proposal-accept", Button).disabled


@pytest.mark.asyncio
async def test_proposal_review_actions_remain_contained_at_80x24() -> None:
    proposal = _proposal()
    host = _Host()

    async with host.run_test(size=(80, 24)) as pilot:
        modal = PersonalContextProposalReviewModal(
            _ProposalService(proposal),
            proposal=proposal,
            scope_label="Global",
        )
        await host.push_screen(modal, callback=host.results.append)
        await pilot.pause()

        container = modal.query_one("#personal-context-review-modal")
        for button in modal.query(Button):
            assert button.region.x >= container.region.x
            assert button.region.right <= container.region.right
            assert button.region.y >= container.region.y
            assert button.region.bottom <= container.region.bottom


@pytest.mark.asyncio
async def test_edit_and_accept_freezes_every_dismissal_until_commit_finishes() -> None:
    proposal = _proposal()
    service = _BlockingProposalService(proposal)
    host = _Host()

    async with host.run_test(size=(100, 32)) as pilot:
        modal = PersonalContextProposalReviewModal(
            service,
            proposal=proposal,
            scope_label="Global",
        )
        await host.push_screen(modal, callback=host.results.append)
        await pilot.pause()
        modal.query_one(
            "#personal-context-proposal-value", Input
        ).value = "edited by user"
        await pilot.click("#personal-context-proposal-accept-edited")
        assert service.entered.wait(1)

        assert all(button.disabled for button in modal.query(Button))
        await pilot.press("escape")
        await pilot.click(offset=(0, 0))
        assert host.results == []

        service.release.set()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert len(service.accept_calls) == 1
        call = service.accept_calls[0]
        assert call["user_actor"] is ActorType.USER
        assert call["edited_payload"].value == "edited by user"
        assert host.results == [
            ProposalReviewResult(
                proposal_id=proposal.proposal_id,
                state="accepted",
                record_id="record-proposed",
            )
        ]


@pytest.mark.asyncio
async def test_reject_resolves_proposal_and_returns_content_free_result() -> None:
    proposal = _proposal("REJECT-UI-CANARY")
    service = _ProposalService(proposal)
    host = _Host()

    async with host.run_test(size=(100, 32)) as pilot:
        modal = PersonalContextProposalReviewModal(
            service,
            proposal=proposal,
            scope_label="Global",
        )
        await host.push_screen(modal, callback=host.results.append)
        await pilot.pause()
        await pilot.click("#personal-context-proposal-reject")
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert service.reject_calls == [proposal.proposal_id]
        assert host.results == [
            ProposalReviewResult(
                proposal_id=proposal.proposal_id,
                state="rejected",
                record_id=None,
            )
        ]
        assert "REJECT-UI-CANARY" not in repr(host.results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_copy"),
    [
        (ProfileConflictError("PRIVATE-CONFLICT-CANARY"), "changed"),
        (ValueError("proposal_expired"), "expired"),
    ],
)
async def test_known_resolution_failures_are_safe_to_close_and_reload(
    error: Exception, expected_copy: str
) -> None:
    proposal = _proposal()
    service = _FailingProposalService(proposal, error)
    host = _Host()

    async with host.run_test(size=(100, 32)) as pilot:
        modal = PersonalContextProposalReviewModal(
            service,
            proposal=proposal,
            scope_label="Global",
        )
        await host.push_screen(modal, callback=host.results.append)
        await pilot.pause()
        await pilot.click("#personal-context-proposal-accept")
        await host.workers.wait_for_complete()
        await pilot.pause()

        status = str(
            modal.query_one("#personal-context-proposal-status", Static).renderable
        ).lower()
        assert expected_copy in status
        assert "private" not in status
        assert modal.query_one("#personal-context-proposal-close", Button).disabled is (
            False
        )
        assert host.results == []


@pytest.mark.asyncio
async def test_unknown_resolution_outcome_disables_retries_until_reload() -> None:
    proposal = _proposal()
    service = _FailingProposalService(proposal, RuntimeError("PRIVATE-OUTCOME-CANARY"))
    host = _Host()

    async with host.run_test(size=(100, 32)) as pilot:
        modal = PersonalContextProposalReviewModal(
            service,
            proposal=proposal,
            scope_label="Global",
        )
        await host.push_screen(modal, callback=host.results.append)
        await pilot.pause()
        await pilot.click("#personal-context-proposal-accept")
        await host.workers.wait_for_complete()
        await pilot.pause()

        status = str(
            modal.query_one("#personal-context-proposal-status", Static).renderable
        ).lower()
        assert "could not be confirmed" in status
        assert "private" not in status
        assert modal.query_one("#personal-context-proposal-accept", Button).disabled
        assert modal.query_one("#personal-context-proposal-reject", Button).disabled
        assert not modal.query_one("#personal-context-proposal-close", Button).disabled


class _SettingsService:
    def __init__(
        self,
        proposal: ProfileProposal,
        *,
        records: tuple[ProfileRecord, ...] = (),
    ) -> None:
        scope = ProfileScope(
            scope_id="scope-global",
            profile_id="profile-1",
            kind=ScopeKind.GLOBAL,
            version_id="scope-version",
            created_at=NOW,
            updated_at=NOW,
        )
        self.snapshot = PersonalContextSettingsSnapshot(
            status=ProfileOperationalStatus(
                state=ProfileOperationalState.READY,
                profile_present=True,
                locked=False,
                runtime_enabled=True,
                reason_code=None,
            ),
            scopes=(
                SettingsScopeSnapshot(
                    scope=scope,
                    label="Global",
                    linked=True,
                    authority=AgentAuthority.PROPOSE,
                ),
            ),
            records=records,
            proposals=(proposal,),
        )
        self.proposals = _ProposalService(proposal)
        self.settings_reads = 0

    def settings_snapshot(self) -> PersonalContextSettingsSnapshot:
        self.settings_reads += 1
        return self.snapshot

    def proposal_service(self):
        return self.proposals


class _PanelHost(ConsolidatedCSSApp):
    def __init__(self, panel: PersonalContextSettingsPanel) -> None:
        super().__init__()
        self.panel = panel

    def compose(self) -> ComposeResult:
        yield self.panel


@pytest.mark.asyncio
async def test_settings_lists_pending_proposals_and_opens_review() -> None:
    proposal = _proposal()
    service = _SettingsService(proposal)
    panel = PersonalContextSettingsPanel(service)  # type: ignore[arg-type]
    host = _PanelHost(panel)

    async with host.run_test(size=(110, 40)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()

        button = panel.query_one("#personal-context-proposal-0", Button)
        assert "Create" in str(button.label)
        assert "response.detail" in str(button.label)
        await pilot.click(button)
        await pilot.pause()
        assert isinstance(host.screen, PersonalContextProposalReviewModal)


@pytest.mark.asyncio
async def test_settings_resolves_existing_target_from_its_exact_snapshot() -> None:
    proposal, target = _existing_record_proposal()
    service = _SettingsService(proposal, records=(target,))
    panel = PersonalContextSettingsPanel(service)  # type: ignore[arg-type]
    host = _PanelHost(panel)

    async with host.run_test(size=(110, 40)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()

        button = panel.query_one("#personal-context-proposal-0", Button)
        assert "response.detail" in str(button.label)
        await pilot.click(button)
        await pilot.pause()

        text = " ".join(str(widget.renderable) for widget in host.screen.query(Static))
        assert "Target subject: response.detail" in text
        assert "Target value: target value" in text


@pytest.mark.asyncio
async def test_settings_does_not_resolve_private_target_for_agent_proposal() -> None:
    proposal, _ = _existing_record_proposal()
    private_target = _record(
        "PRIVATE-SETTINGS-TARGET-CANARY",
        record_id="record-target",
        visibility=AgentVisibility.USER_ONLY,
    )
    service = _SettingsService(proposal, records=(private_target,))
    panel = PersonalContextSettingsPanel(service)  # type: ignore[arg-type]
    host = _PanelHost(panel)

    async with host.run_test(size=(110, 40)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()

        button = panel.query_one("#personal-context-proposal-0", Button)
        assert "PRIVATE-SETTINGS-TARGET-CANARY" not in str(button.label)
        assert "target unavailable" in str(button.label).lower()
        await pilot.click(button)
        await pilot.pause()

        text = " ".join(str(widget.renderable) for widget in host.screen.query(Static))
        assert "PRIVATE-SETTINGS-TARGET-CANARY" not in text


@pytest.mark.asyncio
async def test_proposal_review_dismissal_always_reloads_settings_snapshot() -> None:
    proposal = _proposal()
    service = _SettingsService(proposal)
    panel = PersonalContextSettingsPanel(service)  # type: ignore[arg-type]
    host = _PanelHost(panel)

    async with host.run_test(size=(110, 40)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert service.settings_reads == 1

        await pilot.click("#personal-context-proposal-0")
        await pilot.pause()
        await pilot.click("#personal-context-proposal-close")
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert service.settings_reads == 2
