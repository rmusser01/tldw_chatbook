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
from .repository import ProfileKeyActivationPendingError


@dataclass(frozen=True, slots=True)
class PersonalContextLinkReceipt:
    """Bounded content-free result returned to Settings."""

    profile_id: str
    dataset_id: str
    device_id: str
    bootstrap_cursor: str
    confirmed_cursor: str
    rebaseline_version: int


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
            cursor=remote.cursor,
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
                bootstrap_heads=canonical_snapshot_heads(
                    remote.manifest, remote.scopes, remote.records, remote.proposals
                ),
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
            release(plan_id=plan_id)

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
                "bootstrap_cursor", "plan_id", "rebaseline_version",
                "bootstrap_heads", "expected_heads",
                "attention_code",
            )},
            state="applying",
            expected_states=("review_required",),
        )
        integrity_key = self._wrapping.unwrap_integrity_key(
            remote.wrapped_key_blob,
            integrity_key_id=remote.integrity_key_id,
        )
        binding = self._key_binding(applying)
        self._custodian.stage(**binding, integrity_key=integrity_key)
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
        except Exception:
            self._custodian.delete(**binding)
            self._state.set_personal_context_link_state(
                **self._scope,
                **{key: applying[key] for key in (
                    "device_id", "dataset_id", "authority_id", "profile_id",
                    "integrity_key_id", "key_record_id", "purge_generation",
                    "bootstrap_cursor", "plan_id", "rebaseline_version",
                    "bootstrap_heads", "expected_heads",
                )},
                state="attention_required",
                attention_code="local_apply_failed",
                expected_states=("applying",),
            )
            self._release_freeze(plan_id)
            raise
        locally_complete = self._state.set_personal_context_link_state(
            **self._scope,
            **{key: applying[key] for key in (
                "device_id", "dataset_id", "authority_id", "profile_id",
                "integrity_key_id", "key_record_id", "purge_generation",
                "bootstrap_cursor", "plan_id",
                "bootstrap_heads",
            )},
            state="local_rebaseline_complete",
            rebaseline_version=int(result["rebaseline_version"]),
            expected_heads=self._expected_heads(result),
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
                    "plan_id",
                    "bootstrap_heads",
                    "expected_heads",
                )
            },
            state="local_rebaseline_complete",
            rebaseline_version=rebaseline_version,
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
        manifest = self._profile.get_manifest()
        if manifest.profile_id == state["profile_id"]:
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
                    "plan_id",
                    "rebaseline_version",
                    "bootstrap_heads",
                    "expected_heads",
                )
            },
            state="attention_required",
            attention_code="local_apply_interrupted",
            expected_states=("applying",),
        )
        self._custodian.delete(**self._key_binding(attention))
        self._release_freeze(str(attention["plan_id"]))
        return True

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
                            "plan_id",
                            "rebaseline_version",
                            "bootstrap_heads",
                            "expected_heads",
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
                expected_heads=state["expected_heads"],
                bootstrap_heads=state["bootstrap_heads"],
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
                    "bootstrap_cursor", "plan_id", "rebaseline_version",
                    "expected_heads", "bootstrap_heads",
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
            self._move_completion_to_attention(state, attention_code)
            raise
        self._release_freeze(str(complete["plan_id"]))
        self._custodian.delete(**self._key_binding(complete))
        self._plans.pop(str(complete["plan_id"]), None)
        return self._receipt(complete)

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
        self, state: Mapping[str, Any], attention_code: str
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
                "bootstrap_cursor", "plan_id", "rebaseline_version",
                "bootstrap_heads", "expected_heads",
            )},
            state="attention_required",
            confirmed_cursor=None,
            attention_code=attention_code,
            expected_states=(str(current["state"]),),
        )
        self._release_freeze(str(attention["plan_id"]))
        self._custodian.delete(**self._key_binding(attention))
        self._plans.pop(str(attention["plan_id"]), None)

    def _cleanup_complete(
        self, state: Mapping[str, Any]
    ) -> PersonalContextLinkReceipt:
        """Retry exception-safe local cleanup after a durable complete receipt."""

        self._release_freeze(str(state["plan_id"]))
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

    def _seed_sync_profile(
        self,
        *,
        device_id: str,
        dataset_id: str,
        cursor: str,
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
        cursors["sync_v2"] = cursor
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
