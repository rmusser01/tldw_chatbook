"""App-owned presentation awaiters over the existing plugin operation owners."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ...Plugins.review import OperationReceipt


def operation_label(
    *, committed: bool, stopped: bool | None, cleanup_pending: bool
) -> str:
    if not committed:
        return "Disable not saved (blocked for this session)"
    if cleanup_pending or stopped is not True:
        return "Disabled — cleanup pending"
    return "Disabled"


@dataclass
class PluginOperationViewModel:
    receipt: OperationReceipt
    runtime_status: dict
    installation_id: str | None = None
    workspace_id: str | None = None
    component_id: str | None = None
    action: str = "commit"
    review: object | None = field(default=None, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def label(self):
        receipt = self.receipt
        if self.action == "disable":
            return operation_label(
                committed=receipt.committed,
                stopped=receipt.runtime_stopped,
                cleanup_pending=receipt.cleanup_pending,
            )
        saved = "Saved" if receipt.committed else "Not saved"
        cleanup = " · cleanup pending" if receipt.cleanup_pending else ""
        stopped = " · runtime status unknown" if receipt.runtime_stopped is None else ""
        return f"{self.action}: {saved} · {receipt.phase}{cleanup}{stopped}"


class PluginOperations:
    """View state only; no replay, policy, durable store or execution ownership."""

    def __init__(self, service):
        self.service = service
        self.jobs = {}
        self.submissions = {}
        self.drafts = {}
        self.profile_tasks = {}
        self.profile_results = {}
        self.variable_tasks = {}
        self.variable_results = {}

    async def save_variable(self, token, installation_id, name, value):
        """Retain an already-started save; results contain reference metadata only."""
        task = self.variable_tasks.get(token)
        if task is None:

            async def save(submitted_value):
                try:
                    result = await self.service.create_variable_reference(
                        installation_id, name, submitted_value
                    )
                # Discard protected backend exception bodies.
                except Exception:  # noqa: BLE001
                    result = None
                finally:
                    submitted_value = None
                self.variable_results[token] = result
                return result

            task = self.variable_tasks[token] = asyncio.create_task(
                save(value),
                name="plugin-variable-save",
            )
        value = None
        result = await asyncio.shield(task)
        if result is None:
            raise ValueError("plugin_variable_save_unavailable")
        return result

    async def revoke_variable(self, installation_id, reference_id):
        """Navigation cannot cancel an explicitly confirmed protected revocation."""
        token = (installation_id, reference_id)
        task = self.variable_tasks.get(token)
        if task is None:
            task = self.variable_tasks[token] = asyncio.create_task(
                self.service.revoke_variable_reference(installation_id, reference_id),
                name="plugin-variable-revoke",
            )
            task.add_done_callback(
                lambda finished: (
                    finished.exception() if not finished.cancelled() else None
                )
            )
        return await asyncio.shield(task)

    def submit(self, action, review, *, installation_id=None, workspace_id=None):
        from ...Plugins.revocation import RevocationFailure

        submitted = self.submissions.get(id(review))
        if submitted is not None and submitted.review is review:
            return submitted
        if action in {"disable-all", "uninstall"}:
            request = self.service.begin_reviewed_removal(review)
            identity = request.request_id
            callback = lambda: self.service.finish_revocation(request)
            initial = self.service.revocation_status(request)
        else:
            identity = review.operation_id
            initial = OperationReceipt(identity, "submitted", False)
            if action == "create-data":

                async def callback():
                    await self.service.create_data(review, identity)
                    return await self.service.lookup_operation(identity)

            elif action in {"delete-data", "resume-data"}:
                from ...Plugins.data_cleanup import root_ref

                refs = tuple(root_ref(row) for row in review.targets)
                callback = lambda: self.service.delete_data(refs, identity)
            elif action == "reconcile-data":

                async def callback():
                    await self.service.reconcile_data(review, identity)
                    return await self.service.lookup_operation(identity)

            else:
                callback = lambda: self.service.commit(review, identity)
        if identity in self.jobs:
            return self.jobs[identity]
        model = PluginOperationViewModel(
            initial, {}, installation_id, workspace_id, action=action, review=review
        )
        self.jobs[identity] = model
        self.submissions[id(review)] = model

        async def run():
            try:
                model.receipt = await callback()
            except RevocationFailure as error:
                model.receipt = error.receipt
            except Exception as error:  # noqa: BLE001 - retain owner outcome
                # Only bounded exception types enter presentation, never payloads.
                model.receipt = OperationReceipt(
                    identity,
                    "outcome_unconfirmed",
                    False,
                    persistence_error=type(error).__name__,
                    cleanup_pending=True,
                )

        model.task = asyncio.create_task(run(), name="plugin-operation")
        return model

    async def save_profile(self, draft, owner):
        """Retain the data-only MCP save; saving never grants a plugin mapping."""
        task = self.profile_tasks.get(draft.token)
        if task is None:

            async def save():
                details = await self.service.management_details(
                    draft.installation_id, workspace_id=draft.workspace_id
                )
                from ...Plugins.review import inspection_identity

                if inspection_identity(details["inspection"]) != inspection_identity(
                    draft.inspection
                ) or (
                    draft.data_root is not None
                    and draft.data_root not in details["roots"]
                ):
                    raise ValueError("plugin_configuration_review_changed")
                record = await asyncio.to_thread(
                    owner.save_owned_profile,
                    installation_id=draft.installation_id,
                    inspection=draft.inspection,
                    component_id=draft.component_id,
                    data_root=draft.data_root,
                    credential_reference=draft.credential_reference,
                    credential_generation=draft.credential_generation,
                    session_isolation=draft.session_isolation,
                    protocol_version=draft.protocol_version,
                )
                self.profile_results[draft.token] = {
                    "installation_id": draft.installation_id,
                    "component_id": draft.component_id,
                    "profile_id": record["profile_id"],
                }
                return self.profile_results[draft.token]

            task = self.profile_tasks[draft.token] = asyncio.create_task(
                save(), name="plugin-configuration-save"
            )
        return await asyncio.shield(task)

    async def refresh(self, *, offset=0, limit=50):
        page = await self.service.list_operations(offset=offset, limit=limit)
        page_ids = []
        for row in page:
            receipt = row["receipt"]
            if receipt.operation_id is not None:
                page_ids.append(receipt.operation_id)
            if (
                receipt.operation_id is not None
                and receipt.operation_id not in self.jobs
            ):
                self.jobs[receipt.operation_id] = PluginOperationViewModel(
                    receipt,
                    {},
                    installation_id=row["installation_id"],
                    action=row["kind"],
                )
        for identity, job in tuple(self.jobs.items()):
            state = await self.service.operation_status(identity)
            if state["receipt"].phase != "unavailable_or_expired":
                job.receipt, job.runtime_status = state["receipt"], state
        return tuple(self.jobs[key] for key in page_ids)

    async def cancel(self, identity, *, affected_work=False):
        job = self.jobs[identity]
        if getattr(job.review, "kind", None) in {"update", "select"}:
            callback = (
                self.service.cancel_revision_work
                if affected_work
                else self.service.cancel_revision
            )
            callback(job.review)
        elif job.action in {"delete-data", "resume-data"}:
            callback = (
                self.service.cancel_data_work
                if affected_work
                else self.service.cancel_data_deletion
            )
            await callback(identity)
        else:
            raise ValueError("operation_cannot_be_cancelled")
