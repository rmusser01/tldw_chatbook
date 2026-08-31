"""Manual Sync v2 preview and explicit execution control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, MutableMapping, Sequence

from tldw_chatbook.Sync_Interop.conflict_review import (
    SyncV2ConflictReviewItem,
    SyncV2ConflictReviewService,
)
from tldw_chatbook.Sync_Interop.notes_organization import NOTES_ORGANIZATION_DOMAINS
from tldw_chatbook.Sync_Interop.sync_state import is_local_first_sync_profile_mode

ManualSyncStatus = Literal[
    "ready",
    "empty",
    "blocked",
    "success",
    "partial-failure",
    "conflict",
    "failed",
]

DEFAULT_MANUAL_SYNC_DOMAINS: tuple[str, ...] = ("notes", "chat")


@dataclass(frozen=True, slots=True)
class ManualSyncPreview:
    """User-facing state shown before a manual Sync v2 mutation is allowed."""

    status: ManualSyncStatus
    can_run: bool
    pending_total: int
    pending_by_domain: dict[str, int]
    user_message: str
    profile: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ManualSyncRunResult:
    """User-facing state after an explicit manual sync run."""

    status: ManualSyncStatus
    user_message: str
    summary: dict[str, Any]
    preview: ManualSyncPreview
    conflict_reviews: tuple[SyncV2ConflictReviewItem, ...] = ()


class ManualSyncControlService:
    """Build manual Sync v2 previews and execute sync only on explicit request."""

    def __init__(
        self,
        *,
        state_repository: Any,
        local_first_sync_service: Any,
        dataset_keys: MutableMapping[str, bytes] | None = None,
        default_domains: Sequence[str] = DEFAULT_MANUAL_SYNC_DOMAINS,
        notes_organization_sync_service: Any | None = None,
        notes_repository: Any | None = None,
    ) -> None:
        """Initialize the manual sync control service.

        Args:
            state_repository: Repository used to read Sync v2 profile and outbox state.
            local_first_sync_service: Service that executes the explicit Sync v2 run.
            dataset_keys: Shared mutable dataset-key cache. This reference is retained so
                dynamically loaded keys are visible to manual sync previews and runs.
            default_domains: Sync domains included when the caller does not override them.

        Returns:
            None.

        Raises:
            None.
        """

        self.state_repository = state_repository
        self.local_first_sync_service = local_first_sync_service
        self.dataset_keys = dataset_keys if dataset_keys is not None else {}
        self.default_domains = tuple(default_domains)
        self.notes_organization_sync_service = notes_organization_sync_service
        self.notes_repository = notes_repository

    def preview(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        domains: Sequence[str] | None = None,
    ) -> ManualSyncPreview:
        """Return local-only manual sync readiness and pending outbox counts.

        This method must not call server transport or drain the durable outbox.

        Args:
            server_profile_id: Stable identifier for the configured server profile.
            authenticated_principal_id: Optional authenticated user or account scope.
            workspace_scope: Optional workspace scope for workspace-specific datasets.
            domains: Optional sync domains to preview. Defaults to Notes and Chat.

        Returns:
            Manual sync readiness, pending counts, and user-facing copy.

        Raises:
            None.
        """

        selected_domains = self._domains(domains)
        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile is None:
            return self._blocked("No Sync v2 server profile is configured.")
        if not is_local_first_sync_profile_mode(profile.get("profile_mode")):
            return self._blocked(
                "Manual Sync v2 requires a local-first sync profile.", profile
            )

        dataset_id = str(profile.get("dataset_id") or "").strip()
        device_id = str(profile.get("device_id") or "").strip()
        if not dataset_id or not device_id:
            return self._blocked(
                "Manual Sync v2 requires dataset and device identity.", profile
            )
        if dataset_id not in self.dataset_keys:
            return self._blocked(
                "Manual Sync v2 is blocked because the dataset key is unavailable.",
                profile,
            )
        if (
            self.local_first_sync_service is None
            or getattr(self.local_first_sync_service, "local_store", None) is None
        ):
            return self._blocked(
                "Manual Sync v2 is blocked because the local apply store is unavailable.",
                profile,
            )

        entries = self.state_repository.list_pending_sync_v2_outbox_envelopes(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            dataset_id=dataset_id,
            domains=list(selected_domains),
        )
        pending_by_domain = {domain: 0 for domain in selected_domains}
        for entry in entries:
            domain = str(entry.get("domain") or "")
            if domain in pending_by_domain:
                pending_by_domain[domain] += 1
        pending_by_domain = {
            domain: count for domain, count in pending_by_domain.items() if count > 0
        }
        pending_total = sum(pending_by_domain.values())
        if pending_total == 0:
            return ManualSyncPreview(
                status="empty",
                can_run=True,
                pending_total=0,
                pending_by_domain={},
                user_message="Manual Sync preview: no pending Notes or Chat changes; pull can still check for server updates.",
                profile=profile,
            )
        domain_copy = ", ".join(
            f"{domain}: {count}" for domain, count in pending_by_domain.items()
        )
        return ManualSyncPreview(
            status="ready",
            can_run=True,
            pending_total=pending_total,
            pending_by_domain=pending_by_domain,
            user_message=f"Manual Sync preview: {pending_total} pending outgoing changes ({domain_copy}).",
            profile=profile,
        )

    async def run_once(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        domains: Sequence[str] | None = None,
        display_name: str = "Chatbook",
        enrolled_note_ids: set[str] | None = None,
        enrolled_conversation_ids: set[str] | None = None,
    ) -> ManualSyncRunResult:
        """Execute one manual Sync v2 cycle after preflight allows it.

        Args:
            server_profile_id: Stable identifier for the configured server profile.
            authenticated_principal_id: Optional authenticated user or account scope.
            workspace_scope: Optional workspace scope for workspace-specific datasets.
            domains: Optional sync domains to run. Defaults to Notes and Chat.

        Returns:
            Manual sync outcome, summary, and a post-run preview reflecting current
            pending outbox state when execution succeeds.

        Raises:
            None. Sync transport and local apply exceptions are returned as failed
            ManualSyncRunResult values.
        """

        selected_domains = self._domains(domains)
        if self.notes_organization_sync_service is not None:
            durable_note_ids, durable_conversation_ids = self._durable_dependency_ids()
            organization = self.notes_organization_sync_service.for_server_profile(
                server_profile_id
            )
            enrollment = await organization.advance_enrollment(
                server_service=self.local_first_sync_service.server_service,
                local_first_service=self.local_first_sync_service,
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                display_name=display_name,
                enrolled_note_ids=set(
                    durable_note_ids if enrolled_note_ids is None else enrolled_note_ids
                ),
                enrolled_conversation_ids=set(
                    durable_conversation_ids
                    if enrolled_conversation_ids is None
                    else enrolled_conversation_ids
                ),
            )
            if enrollment.get("status") != "ready":
                profile = self.state_repository.get_sync_v2_profile_state(
                    server_profile_id=server_profile_id,
                    authenticated_principal_id=authenticated_principal_id,
                    workspace_scope=workspace_scope,
                )
                reviews = self._conflict_review_items(
                    profile=profile,
                    domains=selected_domains,
                )
                preview = self._blocked(
                    "Manual Sync is waiting for Notes organization enrollment.",
                    profile,
                )
                return ManualSyncRunResult(
                    status="conflict" if reviews else "blocked",
                    user_message=preview.user_message,
                    summary={"notes_organization_enrollment": dict(enrollment)},
                    preview=preview,
                    conflict_reviews=reviews,
                )
        preview = self.preview(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domains=selected_domains,
        )
        if not preview.can_run:
            return ManualSyncRunResult(
                status="blocked",
                user_message=preview.user_message,
                summary={},
                preview=preview,
            )
        try:
            summary = await self.local_first_sync_service.sync_once(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                domains=list(selected_domains),
            )
        except Exception as exc:
            return ManualSyncRunResult(
                status="failed",
                user_message=f"Manual Sync failed: {exc}",
                summary={"error": str(exc), "error_type": type(exc).__name__},
                preview=preview,
                conflict_reviews=self._conflict_review_items(
                    profile=preview.profile,
                    domains=selected_domains,
                ),
            )
        status, message = self._result_copy(summary)
        post_preview = self.preview(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            domains=selected_domains,
        )
        return ManualSyncRunResult(
            status=status,
            user_message=message,
            summary=dict(summary),
            preview=post_preview,
            conflict_reviews=self._conflict_review_items(
                profile=post_preview.profile or preview.profile,
                domains=selected_domains,
            ),
        )

    def list_conflict_reviews(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        domains: Sequence[str] | None = None,
    ) -> tuple[SyncV2ConflictReviewItem, ...]:
        """Expose generic and Notes-owned reviews for the active profile."""

        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        return self._conflict_review_items(
            profile=profile,
            domains=self._domains(domains),
        )

    def resolve_notes_organization_adoption(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None = None,
        workspace_scope: str | None = None,
        review_id: str,
        action: str,
        new_name: str | None = None,
    ) -> bool:
        """Resolve one Notes-owned review through the active-profile seam."""

        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile is None or not profile.get("dataset_id"):
            raise ValueError("persisted Notes organization profile is required")
        repository = self._notes_repository_for_profile(server_profile_id)
        if repository is None:
            raise ValueError("Notes organization repository is required")
        return SyncV2ConflictReviewService(
            state_repository=self.state_repository,
            notes_repository=repository,
            notes_organization_sync_service=self.notes_organization_sync_service,
        ).resolve_notes_organization_adoption(
            review_id=review_id,
            action=action,
            new_name=new_name,
        )

    def _blocked(
        self,
        message: str,
        profile: Mapping[str, Any] | None = None,
    ) -> ManualSyncPreview:
        return ManualSyncPreview(
            status="blocked",
            can_run=False,
            pending_total=0,
            pending_by_domain={},
            user_message=message,
            profile=profile,
        )

    def _domains(self, domains: Sequence[str] | None) -> tuple[str, ...]:
        selected = tuple(
            str(domain).strip() for domain in (domains or self.default_domains)
        )
        filtered = tuple(domain for domain in selected if domain)
        if self.notes_organization_sync_service is None:
            return filtered
        return tuple(dict.fromkeys((*filtered, *NOTES_ORGANIZATION_DOMAINS)))

    def _durable_dependency_ids(self) -> tuple[set[str], set[str]]:
        """Return current note/conversation identities from durable sync heads."""

        repository = self.notes_repository
        if repository is None:
            return set(), set()
        rows = (
            repository.db.get_connection()
            .execute(
                "SELECT head.entity, head.entity_id FROM sync_log AS head "
                "JOIN ("
                "SELECT entity, entity_id, MAX(change_id) AS change_id "
                "FROM sync_log WHERE entity IN ('notes', 'conversations') "
                "GROUP BY entity, entity_id"
                ") AS latest ON latest.change_id = head.change_id "
                "WHERE head.operation <> 'delete' "
                "ORDER BY head.entity, head.entity_id"
            )
            .fetchall()
        )
        note_ids = {
            str(row[1]) for row in rows if str(row[0]) == "notes" and str(row[1])
        }
        conversation_ids = {
            str(row[1])
            for row in rows
            if str(row[0]) == "conversations" and str(row[1])
        }
        return note_ids, conversation_ids

    def _conflict_review_items(
        self,
        *,
        profile: Mapping[str, Any] | None,
        domains: Sequence[str],
    ) -> tuple[SyncV2ConflictReviewItem, ...]:
        if profile is None:
            return ()
        dataset_id = str(profile.get("dataset_id") or "").strip()
        if not dataset_id:
            return ()
        repository = self._notes_repository_for_profile(
            str(profile["server_profile_id"])
        )
        service = SyncV2ConflictReviewService(
            state_repository=self.state_repository,
            notes_repository=repository,
            notes_organization_sync_service=self.notes_organization_sync_service,
        )
        generic = service.build_review_items(
            server_profile_id=str(profile["server_profile_id"]),
            authenticated_principal_id=profile.get("authenticated_principal_id"),
            workspace_scope=profile.get("workspace_scope"),
            dataset_id=dataset_id,
            domains=domains,
        )
        return generic + service.build_notes_organization_adoption_items(
            dataset_id=dataset_id
        )

    def _notes_repository_for_profile(self, server_profile_id: str) -> Any | None:
        repository = self.notes_repository
        if repository is None:
            return None
        if repository.server_profile_id == server_profile_id:
            return repository
        return type(repository)(
            repository.db,
            server_profile_id=server_profile_id,
        )

    @staticmethod
    def _result_copy(summary: Mapping[str, Any]) -> tuple[ManualSyncStatus, str]:
        conflicts = list(summary.get("push_conflicts") or []) + list(
            summary.get("conflicts") or []
        )
        if conflicts:
            return (
                "conflict",
                f"Manual Sync found {len(conflicts)} conflict(s); review is required before completion.",
            )
        retained = int(summary.get("outbox_retained") or 0)
        rejected = list(summary.get("rejected_envelopes") or [])
        if retained or rejected:
            return (
                "partial-failure",
                (
                    "Manual Sync partially completed: "
                    f"{summary.get('outbox_dispatched', 0)} outgoing dispatched, "
                    f"{retained} retained for retry."
                ),
            )
        return (
            "success",
            (
                "Manual Sync completed: "
                f"{summary.get('outbox_dispatched', 0)} outgoing dispatched, "
                f"{summary.get('pulled_envelopes', 0)} pulled, "
                f"{summary.get('applied_envelopes', 0)} applied."
            ),
        )
