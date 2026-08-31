"""Reviewed, resumable Personal Context first-link coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .reconciliation import (
    CanonicalBootstrapSnapshot,
    ReconciliationPlan,
    build_reconciliation_plan,
)
from .repository import ProfileKeyActivationPendingError


@dataclass(frozen=True, slots=True)
class PersonalContextLinkReceipt:
    """Bounded content-free result returned to Settings."""

    profile_id: str
    dataset_id: str
    device_id: str
    bootstrap_cursor: str
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

        response = await self._server.bootstrap_personal_context_link(
            server_profile_id=self._server_profile_id,
            authenticated_principal_id=self._principal_id,
            display_name=self._display_name,
            wrapping_key_provider=self._wrapping,
            client_version=self._client_version,
            required_schema_version=required_schema_version,
            required_quotas=dict(required_quotas or {}),
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
        )
        state = "attention_required" if plan.attention_codes else "review_required"
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
            plan_id=plan.plan_id,
            rebaseline_version=1,
            attention_code=(
                plan.attention_codes[0] if plan.attention_codes else None
            ),
        )
        self._plans[plan.plan_id] = (plan, remote)
        return plan

    def cancel(self, plan_id: str) -> bool:
        """Discard an unapproved in-memory snapshot and its content-free state."""

        cancelled = self._state.cancel_personal_context_link_plan(
            **self._scope, plan_id=plan_id
        )
        if cancelled:
            self._plans.pop(plan_id, None)
        return cancelled

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
            return self._receipt(state)
        if state["state"] == "local_rebaseline_complete":
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
                )},
                state="attention_required",
                attention_code="local_apply_failed",
                expected_states=("applying",),
            )
            raise
        locally_complete = self._state.set_personal_context_link_state(
            **self._scope,
            **{key: applying[key] for key in (
                "device_id", "dataset_id", "authority_id", "profile_id",
                "integrity_key_id", "key_record_id", "purge_generation",
                "bootstrap_cursor", "plan_id",
            )},
            state="local_rebaseline_complete",
            rebaseline_version=int(result["rebaseline_version"]),
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
            return self._receipt(state)
        if state["state"] != "local_rebaseline_complete":
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
                )
            },
            state="local_rebaseline_complete",
            rebaseline_version=rebaseline_version,
            attention_code=None,
            expected_states=("applying",),
        )
        return await self._complete(locally_complete)

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
                )
            },
            state="attention_required",
            attention_code="local_apply_interrupted",
            expected_states=("applying",),
        )
        self._custodian.delete(**self._key_binding(attention))
        return True

    async def _complete(
        self, state: Mapping[str, Any]
    ) -> PersonalContextLinkReceipt:
        await self._server.complete_personal_context_link(
            device_id=str(state["device_id"]),
            dataset_id=str(state["dataset_id"]),
            bootstrap_cursor=str(state["bootstrap_cursor"]),
        )
        complete = self._state.set_personal_context_link_state(
            **self._scope,
            **{key: state[key] for key in (
                "device_id", "dataset_id", "authority_id", "profile_id",
                "integrity_key_id", "key_record_id", "purge_generation",
                "bootstrap_cursor", "plan_id", "rebaseline_version",
            )},
            state="complete",
            attention_code=None,
            expected_states=("local_rebaseline_complete",),
        )
        self._custodian.delete(**self._key_binding(complete))
        self._plans.pop(str(complete["plan_id"]), None)
        return self._receipt(complete)

    @staticmethod
    def _receipt(state: Mapping[str, Any]) -> PersonalContextLinkReceipt:
        return PersonalContextLinkReceipt(
            profile_id=str(state["profile_id"]),
            dataset_id=str(state["dataset_id"]),
            device_id=str(state["device_id"]),
            bootstrap_cursor=str(state["bootstrap_cursor"]),
            rebaseline_version=int(state["rebaseline_version"]),
        )


__all__ = ["PersonalContextLinkReceipt", "PersonalContextLinkService"]
