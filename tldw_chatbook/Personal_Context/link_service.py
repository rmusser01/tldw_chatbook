"""Reviewed, resumable Personal Context first-link coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .reconciliation import (
    CanonicalBootstrapSnapshot,
    ReconciliationPlan,
    build_reconciliation_plan,
    canonical_snapshot_heads,
)
from .key_protector import ProfileLockedError
from .repository import ProfileKeyActivationPendingError


@dataclass(frozen=True, slots=True)
class PersonalContextLinkReceipt:
    """Bounded content-free result returned to Settings."""

    profile_id: str
    dataset_id: str
    device_id: str
    bootstrap_cursor: str
    sync_transport_cursor: str
    confirmed_cursor: str
    rebaseline_version: int


class PersonalContextLinkAttentionRequired(Exception):
    """Carry one validated content-free bootstrap attention result to Settings."""

    def __init__(self, attention: Any) -> None:
        super().__init__("personal_context_link_attention_required")
        self.attention = attention


def cleanup_completed_link_artifacts(
    profile: Any,
    state: Mapping[str, Any],
) -> None:
    """Remove only artifacts owned by one exact completed link plan."""

    if state.get("state") != "complete":
        raise ValueError("personal_context_link_cleanup_mismatch")
    plan_id = str(state["plan_id"])
    marker_binding = {
        "plan_id": plan_id,
        "target_profile_id": str(state["profile_id"]),
        "target_integrity_key_id": str(state["integrity_key_id"]),
        "target_key_record_id": str(state["key_record_id"]),
        "target_purge_generation": int(state["purge_generation"]),
        "rebaseline_version": int(state["rebaseline_version"]),
    }
    marker_owner_reader = getattr(
        profile, "first_link_rebaseline_commit_plan_id", None
    )
    marker_owner = marker_owner_reader() if callable(marker_owner_reader) else None
    if marker_owner not in {None, plan_id}:
        raise ValueError("personal_context_link_cleanup_mismatch")
    clear = getattr(profile, "clear_first_link_rebaseline_commit", None)
    if marker_owner == plan_id or not callable(marker_owner_reader):
        if callable(clear):
            clear(**marker_binding)
    if callable(marker_owner_reader) and marker_owner_reader() == plan_id:
        repair = getattr(
            profile, "authenticate_legacy_first_link_rebaseline_commit", None
        )
        if callable(repair) and repair(**marker_binding) and callable(clear):
            clear(**marker_binding)
    if callable(marker_owner_reader) and marker_owner_reader() is not None:
        raise ValueError("personal_context_link_cleanup_mismatch")

    freeze_owner_reader = getattr(profile, "first_link_freeze_plan_id", None)
    freeze_owner = freeze_owner_reader() if callable(freeze_owner_reader) else None
    if freeze_owner not in {None, plan_id}:
        raise ValueError("personal_context_link_cleanup_mismatch")
    release = getattr(profile, "release_first_link_freeze", None)
    if freeze_owner == plan_id or not callable(freeze_owner_reader):
        if callable(release):
            release(plan_id=plan_id)
    if callable(freeze_owner_reader) and freeze_owner_reader() is not None:
        raise ValueError("personal_context_link_cleanup_mismatch")


class PersonalContextLinkService:
    """Own planning, explicit approval, local rebaseline, and server completion."""

    def __init__(
        self,
        *,
        personal_context_service: Any,
        server_sync_service: Any,
        state_repository: Any,
        wrapping_key_provider: Any,
        key_custodian: Any,
        freeze_release_fallback: Any | None = None,
        first_link_sync: Any | None = None,
        local_first_sync_service: Any | None = None,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        display_name: str,
        client_version: str | None = None,
    ) -> None:
        self._profile = personal_context_service
        self._server = server_sync_service
        self._state = state_repository
        self._wrapping = wrapping_key_provider
        self._custodian = key_custodian
        self._freeze_release_fallback = freeze_release_fallback
        self._first_link_sync = first_link_sync
        self._local_first_sync = local_first_sync_service
        self._server_profile_id = server_profile_id
        self._principal_id = authenticated_principal_id
        self._display_name = display_name
        self._client_version = client_version
        self._plans: dict[str, tuple[ReconciliationPlan, CanonicalBootstrapSnapshot]] = {}

    @property
    def _scope(self) -> dict[str, str | None]:
        return {
            "server_profile_id": self._server_profile_id,
            "authenticated_principal_id": self._principal_id,
        }

    async def plan(
        self,
        *,
        required_schema_version: int | None = 1,
        required_quotas: Mapping[str, int] | None = None,
        expected_purge_generation: int | None = None,
    ) -> ReconciliationPlan:
        """Fetch and compare one snapshot without upload or completion calls."""

        self._discard_expired_review()
        if required_quotas is None:
            from tldw_chatbook.Sync_Interop.sync_readiness import (
                PERSONAL_CONTEXT_MINIMUM_QUOTAS,
            )

            required_quotas = PERSONAL_CONTEXT_MINIMUM_QUOTAS
        from ..tldw_api.exceptions import PersonalContextBootstrapAttentionError

        try:
            response = await self._server.bootstrap_personal_context_link(
                server_profile_id=self._server_profile_id,
                authenticated_principal_id=self._principal_id,
                display_name=self._display_name,
                wrapping_key_provider=self._wrapping,
                client_version=self._client_version,
                required_schema_version=required_schema_version,
                required_quotas=dict(required_quotas),
                expected_purge_generation=expected_purge_generation,
            )
        except PersonalContextBootstrapAttentionError as exc:
            raise PersonalContextLinkAttentionRequired(exc.attention) from None
        if not isinstance(response, Mapping):
            device_id = str(getattr(response, "device_id"))
        else:
            device_id = str(response["device_id"])
        remote = CanonicalBootstrapSnapshot.from_response(response)
        manifest, scopes, records, proposals, bindings = (
            self._profile.first_link_snapshot()
        )
        plan = build_reconciliation_plan(
            local_manifest=manifest,
            local_scopes=scopes,
            local_records=records,
            local_proposals=proposals,
            remote=remote,
            local_workspace_bindings=bindings,
            required_schema_version=required_schema_version,
            required_quotas=required_quotas,
        )
        state = "attention_required" if plan.attention_codes else "review_required"
        capabilities = {
            "personal_context": {"schema_version": remote.schema_version},
            "supported_domains": [
                "personal_context.manifest",
                "personal_context.scope",
                "personal_context.record",
                "personal_context.proposal",
                "personal_context.purge",
            ],
        }
        if isinstance(response, Mapping):
            sync_capabilities = response.get("_sync_capabilities")
            if isinstance(sync_capabilities, Mapping):
                max_batch_size = sync_capabilities.get("max_batch_size")
                if type(max_batch_size) is not int or max_batch_size < 1:
                    raise ValueError("personal_context_sync_capabilities_invalid")
                capabilities["max_batch_size"] = max_batch_size
        self._seed_sync_profile(
            device_id=device_id,
            dataset_id=remote.dataset_id,
            capabilities=capabilities,
        )
        freeze_acquired = False
        if state == "review_required":
            acquire_freeze = getattr(self._profile, "acquire_first_link_freeze", None)
            if callable(acquire_freeze):
                acquire_freeze(
                    plan_id=plan.plan_id,
                    snapshot_token=plan.local_snapshot_token,
                )
                freeze_acquired = True
        try:
            self._state.set_personal_context_link_state(
                **self._scope,
                state=state,
                device_id=device_id,
                dataset_id=remote.dataset_id,
                authority_id=remote.authority_id,
                profile_id=remote.manifest.profile_id,
                integrity_key_id=remote.integrity_key_id,
                key_record_id=remote.key_record_id,
                purge_generation=remote.purge_generation,
                bootstrap_cursor=remote.cursor,
                sync_transport_cursor=remote.sync_transport_cursor,
                bootstrap_heads=canonical_snapshot_heads(
                    remote.manifest, remote.scopes, remote.records, remote.proposals
                ),
                reviewed_lineage=self._bootstrap_reviewed_lineage(remote),
                plan_id=plan.plan_id,
                rebaseline_version=1,
                attention_code=(
                    plan.attention_codes[0] if plan.attention_codes else None
                ),
            )
        except BaseException:
            if freeze_acquired:
                self._release_freeze(plan.plan_id)
            raise
        self._plans[plan.plan_id] = (plan, remote)
        return plan

    def _discard_expired_review(self) -> None:
        """Release a restarted review whose canonical snapshot is no longer in memory."""

        existing = self._state.get_personal_context_link_state(**self._scope)
        if existing is None:
            freeze_owner = getattr(
                self._profile,
                "first_link_freeze_plan_id",
                None,
            )
            if callable(freeze_owner) and (plan_id := freeze_owner()) is not None:
                self._release_freeze(str(plan_id))
            return
        if existing.get("state") == "attention_required":
            self._recover_attention_artifacts(existing, original_error=None)
            return
        if existing.get("state") != "review_required":
            return
        plan_id = str(existing["plan_id"])
        if plan_id in self._plans:
            raise ValueError("personal_context_link_review_active")
        if self._state.cancel_personal_context_link_plan(
            **self._scope, plan_id=plan_id
        ):
            self._release_freeze(plan_id)

    def cancel(self, plan_id: str) -> bool:
        """Discard an unapproved in-memory snapshot and its content-free state."""

        cancelled = self._state.cancel_personal_context_link_plan(
            **self._scope, plan_id=plan_id
        )
        if cancelled:
            self._plans.pop(plan_id, None)
            self._release_freeze(plan_id)
        return cancelled

    def _release_freeze(self, plan_id: str) -> None:
        release = getattr(self._profile, "release_first_link_freeze", None)
        if callable(release):
            try:
                release(plan_id=plan_id)
                return
            except ProfileLockedError:
                if self._freeze_release_fallback is None:
                    raise
        if self._freeze_release_fallback is not None:
            self._freeze_release_fallback(plan_id)

    @staticmethod
    def _key_binding(state: Mapping[str, Any]) -> dict[str, str]:
        return {
            "server_profile_id": str(state["server_profile_id"]),
            "dataset_id": str(state["dataset_id"]),
            "device_id": str(state["device_id"]),
            "profile_id": str(state["profile_id"]),
            "integrity_key_id": str(state["integrity_key_id"]),
            "key_record_id": str(state["key_record_id"]),
        }

    async def apply(
        self, plan_id: str, decisions: Mapping[str, str]
    ) -> PersonalContextLinkReceipt:
        """Apply reviewed decisions locally, then acknowledge the exact cursor."""

        state = self._state.get_personal_context_link_state(**self._scope)
        if state is None or state["plan_id"] != plan_id:
            raise ValueError("personal_context_link_plan_stale")
        if state["state"] == "complete":
            return self._cleanup_complete(state)
        if state["state"] == "local_rebaseline_complete":
            return await self.resume()
        if state["state"] == "reconciling":
            return await self.resume()
        if state["state"] != "review_required":
            raise ValueError("personal_context_link_not_approvable")
        try:
            plan, remote = self._plans[plan_id]
        except KeyError:
            raise ValueError("personal_context_link_review_expired") from None
        if not plan.can_approve or set(decisions) != set(plan.required_decision_ids):
            raise ValueError("personal_context_link_decisions_incomplete")

        applying = self._state.set_personal_context_link_state(
            **self._scope,
            **{key: state[key] for key in (
                "device_id", "dataset_id", "authority_id", "profile_id",
                "integrity_key_id", "key_record_id", "purge_generation",
                "bootstrap_cursor", "sync_transport_cursor", "plan_id", "rebaseline_version",
                "bootstrap_heads", "expected_heads", "reviewed_lineage",
                "attention_code",
            )},
            state="applying",
            expected_states=("review_required",),
        )
        binding = self._key_binding(applying)
        try:
            integrity_key = self._wrapping.unwrap_integrity_key(
                remote.wrapped_key_blob,
                integrity_key_id=remote.integrity_key_id,
            )
        except Exception as exc:
            self._fail_provisional_apply(
                applying,
                attention_code="local_key_unwrap_failed",
                original_error=exc,
            )
            raise
        try:
            self._custodian.stage(**binding, integrity_key=integrity_key)
        except Exception as exc:
            self._fail_provisional_apply(
                applying,
                attention_code="local_key_stage_failed",
                original_error=exc,
            )
            raise
        try:
            result = self._profile.apply_reviewed_link(
                plan=plan,
                remote=remote,
                decisions=dict(decisions),
                integrity_key=integrity_key,
            )
            self._clear_stale_destination(applying)
        except ProfileKeyActivationPendingError:
            # The database transaction committed under the staged key, but the
            # independent secure key-store activation did not. Retain both the
            # exact staged binding and the applying gate for restart recovery.
            raise
        except Exception as exc:
            self._fail_provisional_apply(
                applying,
                attention_code="local_apply_failed",
                original_error=exc,
            )
            raise
        locally_complete = self._state.set_personal_context_link_state(
            **self._scope,
            **{key: applying[key] for key in (
                "device_id", "dataset_id", "authority_id", "profile_id",
                "integrity_key_id", "key_record_id", "purge_generation",
                "bootstrap_cursor", "sync_transport_cursor", "plan_id",
                "bootstrap_heads",
            )},
            state="local_rebaseline_complete",
            rebaseline_version=int(result["rebaseline_version"]),
            expected_heads=self._expected_heads(result),
            reviewed_lineage=self._reviewed_lineage(applying, result=result),
            attention_code=None,
            expected_states=("applying",),
        )
        return await self._complete(locally_complete)

    async def resume(self) -> PersonalContextLinkReceipt:
        """Resume only the safe server-completion half of an interrupted link."""

        state = self._state.get_personal_context_link_state(**self._scope)
        if state is None:
            raise ValueError("personal_context_link_missing")
        if state["state"] == "complete":
            return self._cleanup_complete(state)
        if state["state"] not in {"local_rebaseline_complete", "reconciling"}:
            raise ValueError("personal_context_link_requires_review")
        return await self._complete(state)

    async def resume_after_local_activation(
        self, *, rebaseline_version: int
    ) -> PersonalContextLinkReceipt:
        """Advance an authenticated staged-key recovery, then complete the server."""

        if type(rebaseline_version) is not int or rebaseline_version < 1:
            raise ValueError("personal_context_rebaseline_version_invalid")
        state = self._state.get_personal_context_link_state(**self._scope)
        if state is None or state["state"] != "applying":
            raise ValueError("personal_context_link_recovery_stale")
        if self.authenticated_committed_rebaseline_version() != rebaseline_version:
            raise ValueError("personal_context_link_recovery_unconfirmed")
        self._clear_stale_destination(state)
        locally_complete = self._state.set_personal_context_link_state(
            **self._scope,
            **{
                key: state[key]
                for key in (
                    "device_id",
                    "dataset_id",
                    "authority_id",
                    "profile_id",
                    "integrity_key_id",
                    "key_record_id",
                    "purge_generation",
                    "bootstrap_cursor",
                    "sync_transport_cursor",
                    "plan_id",
                    "bootstrap_heads",
                    "expected_heads",
                )
            },
            state="local_rebaseline_complete",
            rebaseline_version=rebaseline_version,
            reviewed_lineage=self._reviewed_lineage(state),
            attention_code=None,
            expected_states=("applying",),
        )
        return await self._complete(locally_complete)

    def _clear_stale_destination(self, state: Mapping[str, Any]) -> None:
        """Idempotently discard only stale pending PC copies for this binding."""

        clear_stale = getattr(
            self._state, "clear_pending_personal_context_outbox", None
        )
        if callable(clear_stale):
            clear_stale(
                **self._scope,
                workspace_scope=None,
                dataset_id=str(state["dataset_id"]),
                device_id=str(state["device_id"]),
            )

    def abandon_uncommitted_apply(self) -> bool:
        """Release a staged key only when the authenticated DB stayed provisional."""

        state = self._state.get_personal_context_link_state(**self._scope)
        if state is None or state["state"] != "applying":
            return False
        reader = getattr(self._profile, "first_link_apply_recovery_state", None)
        if not callable(reader):
            return False
        try:
            recovery_state, _version = reader(
                plan_id=str(state["plan_id"]),
                target_profile_id=str(state["profile_id"]),
                target_integrity_key_id=str(state["integrity_key_id"]),
                target_key_record_id=str(state["key_record_id"]),
                target_purge_generation=int(state["purge_generation"]),
            )
        except (ProfileLockedError, RuntimeError, ValueError):
            return False
        if recovery_state != "uncommitted":
            return False
        attention = self._state.set_personal_context_link_state(
            **self._scope,
            **{
                key: state[key]
                for key in (
                    "device_id",
                    "dataset_id",
                    "authority_id",
                    "profile_id",
                    "integrity_key_id",
                    "key_record_id",
                    "purge_generation",
                    "bootstrap_cursor",
                    "sync_transport_cursor",
                    "plan_id",
                    "rebaseline_version",
                    "bootstrap_heads",
                    "expected_heads",
                    "reviewed_lineage",
                )
            },
            state="attention_required",
            attention_code="local_apply_interrupted",
            expected_states=("applying",),
        )
        self._recover_attention_artifacts(attention, original_error=None)
        return True

    def mark_ambiguous_apply_attention(self) -> bool:
        """Persist retryable attention without releasing ambiguous recovery custody."""

        state = self._state.get_personal_context_link_state(**self._scope)
        if state is None or state["state"] != "applying":
            return False
        fields = {
            key: state[key]
            for key in (
                "device_id",
                "dataset_id",
                "authority_id",
                "profile_id",
                "integrity_key_id",
                "key_record_id",
                "purge_generation",
                "bootstrap_cursor",
                "plan_id",
                "rebaseline_version",
                "bootstrap_heads",
                "expected_heads",
                "reviewed_lineage",
            )
        }
        transport_cursor = state.get("sync_transport_cursor")
        if isinstance(transport_cursor, str) and transport_cursor:
            fields["sync_transport_cursor"] = transport_cursor
        self._state.set_personal_context_link_state(
            **self._scope,
            **fields,
            state="applying",
            attention_code="local_apply_recovery_ambiguous",
            expected_states=("applying",),
        )
        return True

    def authenticated_committed_rebaseline_version(self) -> int | None:
        """Return the exact committed generation only when active keys authenticate it."""

        state = self._state.get_personal_context_link_state(**self._scope)
        if state is None or state["state"] != "applying":
            return None
        reader = getattr(self._profile, "first_link_apply_recovery_state", None)
        version_reader = getattr(self._profile, "first_link_rebaseline_version", None)
        if not callable(reader) or not callable(version_reader):
            return None
        try:
            recovery_state, version = reader(
                plan_id=str(state["plan_id"]),
                target_profile_id=str(state["profile_id"]),
                target_integrity_key_id=str(state["integrity_key_id"]),
                target_key_record_id=str(state["key_record_id"]),
                target_purge_generation=int(state["purge_generation"]),
            )
            manifest = self._profile.get_manifest()
            active_version = int(version_reader())
        except (ProfileLockedError, RuntimeError, ValueError, TypeError):
            return None
        if (
            recovery_state == "committed"
            and version is not None
            and manifest.profile_id == state["profile_id"]
            and active_version == version
        ):
            return version
        return None

    def _fail_provisional_apply(
        self,
        state: Mapping[str, Any],
        *,
        attention_code: str,
        original_error: Exception,
    ) -> None:
        """Release the provisional apply gate before best-effort key cleanup."""

        reader = getattr(self._profile, "first_link_apply_recovery_state", None)
        if not callable(reader):
            original_error.add_note(
                "Interrupted Personal Context apply evidence is unavailable; "
                "recovery material was preserved."
            )
            return
        try:
            recovery_state, _version = reader(
                plan_id=str(state["plan_id"]),
                target_profile_id=str(state["profile_id"]),
                target_integrity_key_id=str(state["integrity_key_id"]),
                target_key_record_id=str(state["key_record_id"]),
                target_purge_generation=int(state["purge_generation"]),
            )
        except (ProfileLockedError, RuntimeError, ValueError, TypeError):
            recovery_state = "ambiguous"
        if recovery_state != "uncommitted":
            original_error.add_note(
                "Interrupted Personal Context apply may be committed; "
                "recovery material was preserved."
            )
            return

        attention = self._state.set_personal_context_link_state(
            **self._scope,
            **{
                key: state[key]
                for key in (
                    "device_id",
                    "dataset_id",
                    "authority_id",
                    "profile_id",
                    "integrity_key_id",
                    "key_record_id",
                    "purge_generation",
                    "bootstrap_cursor",
                    "sync_transport_cursor",
                    "plan_id",
                    "rebaseline_version",
                    "bootstrap_heads",
                    "expected_heads",
                    "reviewed_lineage",
                )
            },
            state="attention_required",
            attention_code=attention_code,
            expected_states=("applying",),
        )
        self._recover_attention_artifacts(
            attention,
            original_error=original_error,
        )

    def _recover_attention_artifacts(
        self,
        state: Mapping[str, Any],
        *,
        original_error: Exception | None,
    ) -> None:
        """Best-effort cleanup after the durable link state is already safe."""

        if not self._delete_staged_key(state, original_error=original_error):
            return
        try:
            self._release_freeze(str(state["plan_id"]))
        except Exception:
            if original_error is not None:
                original_error.add_note(
                    "Personal Context review freeze release remains pending."
                )
            return
        self._plans.pop(str(state["plan_id"]), None)
        self._clear_rebaseline_marker(state)

    def _delete_staged_key(
        self,
        state: Mapping[str, Any],
        *,
        original_error: Exception | None,
    ) -> bool:
        """Attempt cleanup without undoing an already durable safe state."""

        try:
            self._custodian.delete(**self._key_binding(state))
        except Exception:
            if original_error is None:
                return False
            original_error.add_note(
                "Staged Personal Context key cleanup remains pending after safe state recovery."
            )
            return False
        return True

    def _clear_rebaseline_marker(self, state: Mapping[str, Any]) -> None:
        clear = getattr(self._profile, "clear_first_link_rebaseline_commit", None)
        if callable(clear):
            clear(
                plan_id=str(state["plan_id"]),
                target_profile_id=str(state["profile_id"]),
                target_integrity_key_id=str(state["integrity_key_id"]),
                target_key_record_id=str(state["key_record_id"]),
                target_purge_generation=int(state["purge_generation"]),
                rebaseline_version=int(state["rebaseline_version"]),
            )

    async def _complete(
        self, state: Mapping[str, Any]
    ) -> PersonalContextLinkReceipt:
        try:
            self._ensure_sync_runtime(state)
            if state["state"] == "local_rebaseline_complete":
                storage_key = self._custodian.load_or_create_storage_key(
                    **self._key_binding(state)
                )
            else:
                storage_key = self._custodian.load_storage_key(
                    **self._key_binding(state)
                )
            activate = getattr(self._first_link_sync, "activate_storage_key", None)
            if callable(activate):
                activate(str(state["dataset_id"]), storage_key)
            if state["state"] == "local_rebaseline_complete":
                await self._server.complete_personal_context_link(
                    device_id=str(state["device_id"]),
                    dataset_id=str(state["dataset_id"]),
                    bootstrap_cursor=str(state["bootstrap_cursor"]),
                )
                state = self._state.set_personal_context_link_state(
                    **self._scope,
                    **{
                        key: state[key]
                        for key in (
                            "device_id",
                            "dataset_id",
                            "authority_id",
                            "profile_id",
                            "integrity_key_id",
                            "key_record_id",
                            "purge_generation",
                            "bootstrap_cursor",
                            "sync_transport_cursor",
                            "plan_id",
                            "rebaseline_version",
                            "bootstrap_heads",
                            "expected_heads",
                            "reviewed_lineage",
                        )
                    },
                    state="reconciling",
                    confirmed_cursor=None,
                    attention_code=None,
                    expected_states=("local_rebaseline_complete",),
                )
            if self._first_link_sync is None:
                raise RuntimeError("personal_context_first_link_sync_unavailable")
            convergence = await self._first_link_sync.converge(
                server_profile_id=self._server_profile_id,
                authenticated_principal_id=self._principal_id,
                device_id=str(state["device_id"]),
                dataset_id=str(state["dataset_id"]),
                profile_id=str(state["profile_id"]),
                integrity_key_id=str(state["integrity_key_id"]),
                key_record_id=str(state["key_record_id"]),
                purge_generation=int(state["purge_generation"]),
                bootstrap_cursor=str(state["bootstrap_cursor"]),
                sync_transport_cursor=str(state["sync_transport_cursor"]),
                expected_heads=state["expected_heads"],
                bootstrap_heads=state["bootstrap_heads"],
                reviewed_lineage=state["reviewed_lineage"],
            )
            confirmed_cursor = convergence.get("confirmed_cursor")
            confirmed_heads = convergence.get("confirmed_heads")
            if (
                not isinstance(confirmed_cursor, str)
                or not confirmed_cursor
                or confirmed_heads != state["expected_heads"]
            ):
                raise RuntimeError("personal_context_convergence_unconfirmed")
            profile = self._state.get_sync_v2_profile_state(
                **self._scope, workspace_scope=None
            )
            if (
                profile is None
                or profile.get("dataset_id") != state["dataset_id"]
                or profile.get("device_id") != state["device_id"]
            ):
                raise RuntimeError("personal_context_sync_profile_binding_conflict")
            dataset_cursors = dict(profile.get("dataset_cursors") or {})
            dataset_cursors["sync_v2"] = confirmed_cursor
            self._state.set_sync_v2_profile_state(
                **self._scope,
                workspace_scope=None,
                profile_mode="local_first_sync",
                device_id=str(state["device_id"]),
                dataset_id=str(state["dataset_id"]),
                dataset_cursors=dataset_cursors,
                capabilities=profile.get("capabilities"),
                dry_run_metadata=profile.get("dry_run_metadata"),
            )
            complete = self._state.set_personal_context_link_state(
                **self._scope,
                **{key: state[key] for key in (
                    "device_id", "dataset_id", "authority_id", "profile_id",
                    "integrity_key_id", "key_record_id", "purge_generation",
                    "bootstrap_cursor", "sync_transport_cursor", "plan_id", "rebaseline_version",
                    "expected_heads", "bootstrap_heads", "reviewed_lineage",
                )},
                state="complete",
                confirmed_cursor=confirmed_cursor,
                attention_code=None,
                expected_states=("reconciling",),
            )
        except (RuntimeError, ValueError) as exc:
            attention_code = self._completion_attention_code(exc)
            if attention_code is None:
                raise
            self._move_completion_to_attention(
                state,
                attention_code,
                original_error=exc,
            )
            raise
        return self._cleanup_complete(complete)

    @staticmethod
    def _completion_attention_code(exc: Exception) -> str | None:
        code = str(exc)
        known = {
            "personal_context_convergence_unconfirmed": "server_snapshot_changed",
            "personal_context_reconciliation_push_rejected": (
                "reconciliation_push_rejected"
            ),
            "personal_context_reconciliation_apply_failed": (
                "reconciliation_apply_failed"
            ),
            "personal_context_reconciliation_version_missing": (
                "reconciliation_version_missing"
            ),
            "personal_context_reviewed_lineage_changed": "server_snapshot_changed",
            "personal_context_reconciliation_binding_stale": (
                "reconciliation_binding_stale"
            ),
            "personal_context_staging_key_unavailable": "staging_key_unavailable",
            "personal_context_staging_key_invalid": "staging_key_unavailable",
            "personal_context_sync_profile_binding_conflict": (
                "sync_profile_binding_conflict"
            ),
            "personal_context_first_link_sync_unavailable": (
                "reconciliation_runtime_unavailable"
            ),
        }
        if code in known:
            return known[code]
        if isinstance(exc, ValueError):
            return "reconciliation_validation_failed"
        return None

    def _move_completion_to_attention(
        self,
        state: Mapping[str, Any],
        attention_code: str,
        *,
        original_error: Exception,
    ) -> None:
        current = self._state.get_personal_context_link_state(**self._scope)
        if (
            current is None
            or current.get("plan_id") != state.get("plan_id")
            or current.get("state")
            not in {"local_rebaseline_complete", "reconciling"}
        ):
            return
        attention = self._state.set_personal_context_link_state(
            **self._scope,
            **{key: current[key] for key in (
                "device_id", "dataset_id", "authority_id", "profile_id",
                "integrity_key_id", "key_record_id", "purge_generation",
                "bootstrap_cursor", "sync_transport_cursor", "plan_id", "rebaseline_version",
                "bootstrap_heads", "expected_heads", "reviewed_lineage",
            )},
            state="attention_required",
            confirmed_cursor=None,
            attention_code=attention_code,
            expected_states=(str(current["state"]),),
        )
        self._recover_attention_artifacts(
            attention,
            original_error=original_error,
        )

    def _cleanup_complete(
        self, state: Mapping[str, Any]
    ) -> PersonalContextLinkReceipt:
        """Retry exception-safe local cleanup after a durable complete receipt."""

        self._custodian.load_storage_key(**self._key_binding(state))
        cleanup_completed_link_artifacts(self._profile, state)
        self._custodian.delete(**self._key_binding(state))
        self._plans.pop(str(state["plan_id"]), None)
        return self._receipt(state)

    @staticmethod
    def _receipt(state: Mapping[str, Any]) -> PersonalContextLinkReceipt:
        return PersonalContextLinkReceipt(
            profile_id=str(state["profile_id"]),
            dataset_id=str(state["dataset_id"]),
            device_id=str(state["device_id"]),
            bootstrap_cursor=str(state["bootstrap_cursor"]),
            sync_transport_cursor=str(state["sync_transport_cursor"]),
            confirmed_cursor=str(state["confirmed_cursor"]),
            rebaseline_version=int(state["rebaseline_version"]),
        )

    def _ensure_sync_runtime(self, state: Mapping[str, Any]) -> None:
        if self._first_link_sync is not None:
            return
        if self._local_first_sync is None:
            return
        from tldw_chatbook.Sync_Interop.personal_context_first_link_sync import (
            PersonalContextFirstLinkSync,
        )

        dispatcher = self._profile.build_personal_context_outbox_dispatcher(
            state_repository=self._state,
            integrity_key_id=str(state["integrity_key_id"]),
        )
        self._local_first_sync.personal_context_outbox_dispatcher = dispatcher
        self._local_first_sync.personal_context_service = self._profile
        self._first_link_sync = PersonalContextFirstLinkSync(
            server_service=self._server,
            state_repository=self._state,
            dispatcher=dispatcher,
            personal_context_service=self._profile,
            local_store=self._local_first_sync.local_store,
            dataset_keys=self._local_first_sync.dataset_keys,
        )

    def _expected_heads(self, result: Mapping[str, Any]) -> Mapping[str, Mapping[str, str]]:
        heads = result.get("expected_heads")
        if isinstance(heads, Mapping):
            return heads
        reader = getattr(self._profile, "first_link_sync_heads", None)
        if callable(reader):
            heads = reader()
            if isinstance(heads, Mapping) and heads:
                return heads
        raise RuntimeError("personal_context_expected_heads_unavailable")

    def _reviewed_lineage(
        self,
        state: Mapping[str, Any],
        *,
        result: Mapping[str, Any] | None = None,
    ) -> list[list[str]]:
        """Merge durable review ancestry with authenticated local materialization."""

        lineage = {
            tuple(item)
            for item in state.get("reviewed_lineage", ())
            if isinstance(item, (list, tuple)) and len(item) == 3
        }
        if result is not None:
            lineage.update(
                tuple(item)
                for item in result.get("reviewed_lineage", ())
                if isinstance(item, (list, tuple)) and len(item) == 3
            )
        reader = getattr(self._profile, "first_link_reviewed_lineage", None)
        if callable(reader):
            lineage.update(
                tuple(item)
                for item in reader()
                if isinstance(item, (list, tuple)) and len(item) == 3
            )
        return [list(item) for item in sorted(lineage)]

    @staticmethod
    def _bootstrap_reviewed_lineage(remote: CanonicalBootstrapSnapshot) -> list[list[str]]:
        """Persist the exact reviewed server heads and declared record ancestors."""

        heads = canonical_snapshot_heads(
            remote.manifest,
            remote.scopes,
            remote.records,
            remote.proposals,
        )
        lineage = {
            (domain, object_id, version_id)
            for domain, objects in heads.items()
            for object_id, version_id in objects.items()
        }
        lineage.update(
            (
                "personal_context.record",
                record.record_id,
                record.parent_version_id,
            )
            for record in remote.records
            if record.parent_version_id is not None
        )
        return [list(item) for item in sorted(lineage)]

    def _seed_sync_profile(
        self,
        *,
        device_id: str,
        dataset_id: str,
        capabilities: Mapping[str, Any],
    ) -> None:
        existing = self._state.get_sync_v2_profile_state(
            **self._scope, workspace_scope=None
        )
        if existing is not None and (
            existing.get("device_id") not in {None, device_id}
            or existing.get("dataset_id") not in {None, dataset_id}
        ):
            raise ValueError("personal_context_sync_profile_binding_conflict")
        cursors = dict((existing or {}).get("dataset_cursors") or {})
        merged_capabilities = dict((existing or {}).get("capabilities") or {})
        merged_domains = list(merged_capabilities.get("supported_domains") or [])
        for domain in capabilities["supported_domains"]:
            if domain not in merged_domains:
                merged_domains.append(domain)
        merged_capabilities.update(capabilities)
        merged_capabilities["supported_domains"] = merged_domains
        self._state.set_sync_v2_profile_state(
            **self._scope,
            workspace_scope=None,
            profile_mode="local_first_sync",
            device_id=device_id,
            dataset_id=dataset_id,
            dataset_cursors=cursors,
            capabilities=merged_capabilities,
            dry_run_metadata=(existing or {}).get("dry_run_metadata"),
            last_error=(existing or {}).get("last_error"),
            last_mirror_report_id=(existing or {}).get("last_mirror_report_id"),
        )


__all__ = ["PersonalContextLinkReceipt", "PersonalContextLinkService"]
