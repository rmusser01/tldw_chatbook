"""Manual Sync v2 preview and explicit execution control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, MutableMapping, Sequence

from tldw_chatbook.Sync_Interop.conflict_review import (
    SyncV2ConflictReviewItem,
    SyncV2ConflictReviewService,
)
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
            return self._blocked("Manual Sync v2 requires a local-first sync profile.", profile)

        dataset_id = str(profile.get("dataset_id") or "").strip()
        device_id = str(profile.get("device_id") or "").strip()
        if not dataset_id or not device_id:
            return self._blocked("Manual Sync v2 requires dataset and device identity.", profile)
        if dataset_id not in self.dataset_keys:
            return self._blocked("Manual Sync v2 is blocked because the dataset key is unavailable.", profile)
        if (
            self.local_first_sync_service is None
            or getattr(self.local_first_sync_service, "local_store", None) is None
        ):
            return self._blocked("Manual Sync v2 is blocked because the local apply store is unavailable.", profile)

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
            domain: count
            for domain, count in pending_by_domain.items()
            if count > 0
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
            f"{domain}: {count}"
            for domain, count in pending_by_domain.items()
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
        selected = tuple(str(domain).strip() for domain in (domains or self.default_domains))
        return tuple(domain for domain in selected if domain)

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
        service = SyncV2ConflictReviewService(state_repository=self.state_repository)
        return service.build_review_items(
            server_profile_id=str(profile["server_profile_id"]),
            authenticated_principal_id=profile.get("authenticated_principal_id"),
            workspace_scope=profile.get("workspace_scope"),
            dataset_id=dataset_id,
            domains=domains,
        )

    @staticmethod
    def _result_copy(summary: Mapping[str, Any]) -> tuple[ManualSyncStatus, str]:
        conflicts = list(summary.get("push_conflicts") or []) + list(summary.get("conflicts") or [])
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
