"""Lazy app-owned plugin storage worker and typed Console facade."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from .admission import LivePluginFences, PluginAdmission, PluginUnavailable
from .review import OperationReceipt, PluginReview


@dataclass(frozen=True)
class PluginRunOwnership:
    """Exact host-owned run association consumed by revocation and drain."""

    installation_id: str
    workspace_id: str | None
    revision_digest: str
    pending_id: str
    run_id: str
    handle_id: str | None
    lease_token: str
    cancel: Callable[[], object]
    completed: threading.Event
    turn_id: str | None = None
    parent_run_id: str | None = None


class PluginService:
    """Own one persistent worker; construction performs no IO or keyring work."""

    def __init__(
        self,
        profile_root: Path,
        *,
        workspace_lookup: Callable[[str], object | None],
        marker_store_factory: Callable[[Path], Any] | None = None,
        accept_reduced_protection: bool = False,
    ) -> None:
        self.profile_root = Path(profile_root)
        self.workspace_lookup = workspace_lookup
        self._marker_factory = marker_store_factory
        self._reduced = accept_reduced_protection
        self.fences = LivePluginFences()
        self._lock = self.fences.live_lock
        self._thread = None
        self._closed = False
        self._ready = Future()
        self._catalog = {}
        self._details = ()
        self._snapshots = {}
        self._admitted = {}
        self._admission_keys = {}
        self._retired = set()
        self._pending_custody = {}
        self._root_runs = {}
        self._bound_runs = {}
        self._terminal_runs = set()
        self._live_runs = {}

    def _start(self):
        with self._lock:
            if self._closed:
                raise PluginUnavailable("plugin_service_closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._worker, name="chatbook-plugin-storage", daemon=True
                )
                self._thread.start()
        return self._ready

    def _worker(self):
        from tldw_chatbook.DB.base_db import operation_owned_connection
        from tldw_chatbook.Skills_Interop.local_skills_service import (
            default_local_skills_store_dir,
        )

        from .authority_store import (
            KeyringPluginMarkerStore,
            PluginAuthorityStore,
            default_plugin_authority_dir,
        )
        from .coordinator import PluginCoordinator
        from .registry import PluginRegistry
        from .runtime_owner import PluginRuntimeOwner

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        owner = registry = None
        try:
            owner = PluginRuntimeOwner(self.profile_root / "plugins")
            if not owner.try_acquire():
                raise PluginUnavailable("plugin_owner_busy")
            registry = PluginRegistry(owner.root / "registry.sqlite3", owner=owner)
            trust_root = default_plugin_authority_dir(
                default_local_skills_store_dir(self.profile_root)
            )
            marker = (self._marker_factory or KeyringPluginMarkerStore)(trust_root)
            authority = PluginAuthorityStore(
                trust_root, marker, accept_reduced_protection=self._reduced
            )
            self._coordinator = PluginCoordinator(registry, authority, owner)
            self._admission = PluginAdmission(
                self._coordinator, self.workspace_lookup, fences=self.fences
            )
            self._loop = loop
            self._ready.set_result(loop)
            workspace_db = getattr(
                getattr(self.workspace_lookup, "__self__", None), "db", None
            )
            with operation_owned_connection(workspace_db):
                loop.run_forever()
        # Publish all startup failures to waiters before releasing the owner.
        except BaseException as error:  # noqa: BLE001
            if not self._ready.done():
                self._ready.set_exception(error)
        finally:
            if registry is not None:
                registry.close()
            if owner is not None:
                owner.close()
            loop.close()

    async def _call(self, callback):
        loop = await asyncio.wrap_future(self._start())

        async def invoke():
            result = callback()
            return await result if inspect.isawaitable(result) else result

        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(invoke(), loop)
        )

    def _call_from_agent(self, callback):
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("plugin synchronous checks require an agent worker")
        loop = self._start().result()

        async def invoke():
            return callback()

        return asyncio.run_coroutine_threadsafe(invoke(), loop).result()

    async def bootstrap(self, passphrase: str) -> None:
        def operation():
            self._coordinator.bootstrap(passphrase)
            self._refresh()

        await self._call(operation)

    async def unlock(self, passphrase: str) -> None:
        async def operation():
            self._coordinator.authority.unlock(passphrase)
            await self._coordinator.recover()
            self._refresh()

        await self._call(operation)

    async def review_install(
        self, root: Path, *, selection: tuple[str, ...], workspace_id: str | None
    ) -> PluginReview:
        def operation():
            from .inspection import inspect_package

            self._admission._workspace(workspace_id)
            return self._coordinator.review(
                inspect_package(root), selection=selection, workspace_id=workspace_id
            )

        return await self._call(operation)

    async def review_trust(self, installation_id: str) -> PluginReview:
        return await self._call(lambda: self._coordinator.review_trust(installation_id))

    async def review_activation(
        self, installation_id: str, *, workspace_id: str | None, intent: str
    ) -> PluginReview:
        def operation():
            self._admission._workspace(workspace_id)
            return self._coordinator.review_activation(
                installation_id, workspace_id=workspace_id, intent=intent
            )

        return await self._call(operation)

    async def commit(self, review: PluginReview, operation_id: str) -> OperationReceipt:
        async def operation():
            self._admission._workspace(review.workspace_id)
            try:
                return await self._coordinator.commit(review, operation_id)
            finally:
                self._refresh()

        return await self._call(operation)

    def _refresh(self):
        """Publish metadata eligibility, never package bodies or a runtime grant."""
        from .recovery import retained_inspections
        from .skill_provider import skill_summary

        catalog, details = {}, []
        try:
            authority = self._coordinator.published_snapshot()
            inspections = retained_inspections(authority)
            for installation in authority["installations"]:
                identity = installation["installation_id"]
                inspection = inspections[(identity, installation["revision_digest"])]
                for component in inspection.inventory.values():
                    if component.kind == "skill":
                        details.append(
                            skill_summary(
                                identity,
                                inspection,
                                component.component_id,
                                installation.get("alias") or identity,
                            )
                        )
                scopes = {None} | {
                    row["workspace_id"]
                    for row in authority["activation"]
                    if row["installation_id"] == identity
                }
                for workspace in scopes:
                    rows = []
                    try:
                        snapshot = self._admission.capture(
                            identity, workspace, "catalog"
                        )
                        token = uuid4().hex
                        self._snapshots[token] = snapshot
                        for component_id in snapshot.selection:
                            row = skill_summary(
                                identity, inspection, component_id, snapshot.alias
                            )
                            row.update(
                                plugin_ceiling=token,
                                plugin_workspace_id=workspace,
                                trust_blocked=False,
                            )
                            rows.append(row)
                    except PluginUnavailable:
                        pass
                    catalog[(identity, workspace)] = tuple(rows)
        except (ValueError, OSError, PermissionError):
            catalog, details = {}, []
        with self._lock:
            self._catalog = catalog
            self._details = tuple(details)

    def capture_maximum(self, workspace_id: str | None) -> dict:
        """Read the worker-published immutable metadata ceiling without IO."""
        with self._lock:
            catalog = self._catalog
            identities = sorted({identity for identity, _ in catalog})
            rows = []
            for identity in identities:
                try:
                    self.fences.check(identity, workspace_id)
                except PluginUnavailable:
                    continue
                selected = catalog.get(
                    (identity, workspace_id), catalog.get((identity, None), ())
                )
                rows.extend(
                    dict(deepcopy(row), plugin_workspace_id=workspace_id)
                    for row in selected
                )
        return {
            "backend": "local",
            "available_skills": rows,
            "blocked_skills": [],
            "context_text": "",
        }

    def list_skills(self) -> list[dict]:
        with self._lock:
            return [deepcopy(row) for row in self._details]

    def owns(self, name: str) -> bool:
        from .skill_provider import owned_identifier

        return owned_identifier(name) or any(
            name in {row["name"], row["tool_name"], row["record_id"]}
            for row in self.list_skills()
        )

    async def admit(self, maximum: Mapping[str, Any], run_id: str) -> dict:
        def operation():
            if run_id in self._retired:
                raise PluginUnavailable("plugin_turn_retired")
            turn_id = maximum.get("plugin_turn_id")
            ceiling = frozenset(
                (
                    row.get("plugin_ceiling"),
                    row.get("plugin_component_id"),
                    row.get("plugin_workspace_id"),
                )
                for row in maximum.get("available_skills", ())
                if row.get("plugin_owned")
            )
            prior_custody = self._pending_custody.get(run_id)
            if prior_custody is not None:
                if prior_custody[0] != turn_id:
                    raise PluginUnavailable("plugin_turn_identity_changed")
                if not ceiling <= prior_custody[1]:
                    raise PluginUnavailable("plugin_ceiling_changed")
            result = dict(maximum)
            rows = []
            for entry in maximum.get("available_skills", ()):
                row = dict(entry)
                if row.get("plugin_owned"):
                    prior = self._snapshots.get(row.get("plugin_ceiling"))
                    if prior is None:
                        raise PluginUnavailable("plugin_ceiling_unavailable")
                    workspace = row.get("plugin_workspace_id")
                    generations = prior.generations
                    if workspace is not None and prior.workspace_id is None:
                        generations = tuple(
                            sorted((*generations, ("workspace", workspace, 0)))
                        )
                    snapshot = replace(
                        prior,
                        workspace_id=workspace,
                        run_id=run_id,
                        generations=generations,
                    )
                    self._admission.check(snapshot, row["plugin_component_id"])
                    key = (run_id, row["plugin_ceiling"], workspace)
                    token = self._admission_keys.get(key)
                    if token is None:
                        token = uuid4().hex
                        self._admission_keys[key] = token
                        self._admitted[token] = snapshot
                    from .skill_provider import skill_summary

                    row = dict(
                        skill_summary(
                            snapshot.installation_id,
                            snapshot.inspection,
                            row["plugin_component_id"],
                            snapshot.alias,
                        ),
                        plugin_ceiling=entry["plugin_ceiling"],
                        plugin_workspace_id=workspace,
                        plugin_admission=token,
                        plugin_run_id=run_id,
                        trust_blocked=False,
                    )
                rows.append(row)
            self._pending_custody.setdefault(run_id, (turn_id, ceiling))
            result["available_skills"] = rows
            return result

        return await self._call(operation)

    def _checked(self, name: str, token: str):
        from .skill_provider import skill_summary

        snapshot = self._admitted.get(token)
        if snapshot is None:
            raise PluginUnavailable("plugin_run_admission_required")
        from tldw_chatbook.Agents.run_context import current_run_id

        actor_id = current_run_id()
        with self._lock:
            if snapshot.run_id in self._retired and not any(
                record.pending_id == snapshot.run_id and record.run_id == actor_id
                for record in self._live_runs.values()
            ):
                raise PluginUnavailable("plugin_turn_retired")
        for component_id in snapshot.selection:
            row = skill_summary(
                snapshot.installation_id,
                snapshot.inspection,
                component_id,
                snapshot.alias,
            )
            if name in {row["name"], row["tool_name"], row["record_id"]}:
                self._admission.check(snapshot, component_id)
                return snapshot, component_id, row
        raise PluginUnavailable("plugin_component_not_admitted")

    async def get_skill(self, name: str) -> dict:
        """Explicit detail reads remain owned and never imply standalone trust."""

        def operation():
            from .package_files import capture_package
            from .recovery import retained_inspections
            from .skill_provider import skill_summary

            authority = self._coordinator.published_snapshot()
            inspections = retained_inspections(authority)
            for installed in authority["installations"]:
                identity = installed["installation_id"]
                inspection = inspections[(identity, installed["revision_digest"])]
                for component in inspection.inventory.values():
                    if component.kind != "skill":
                        continue
                    row = skill_summary(
                        identity,
                        inspection,
                        component.component_id,
                        installed.get("alias") or identity,
                    )
                    if name in {row["name"], row["tool_name"], row["record_id"]}:
                        capture = capture_package(
                            Path(inspection.materialized_identity)
                        )
                        if (
                            capture.errors
                            or capture.digest != inspection.content_digest
                        ):
                            raise PluginUnavailable("plugin_material_changed")
                        return dict(
                            row, content=capture.read(component.path).decode("utf-8")
                        )
            raise PluginUnavailable("plugin_skill_unavailable")

        return await self._call(operation)

    async def execute_skill(
        self, name: str, *, admission_token: str, args: str | None = None
    ) -> dict:
        def operation():
            from .skill_provider import render_skill

            snapshot, component_id, row = self._checked(name, admission_token)
            result = render_skill(snapshot, component_id, row, args or "")
            self._admission.check(snapshot, component_id)
            return result

        return await self._call(operation)

    async def read_skill_file(
        self, name: str, path: str, *, admission_token: str
    ) -> dict:
        def operation():
            from .context import instruction_block
            from .package_files import capture_package, validate_relative_member

            snapshot, component_id, row = self._checked(name, admission_token)
            relative = validate_relative_member(path)
            component = snapshot.inspection.inventory[component_id]
            member = str(Path(component.path).parent / relative)
            capture = capture_package(Path(snapshot.inspection.materialized_identity))
            if capture.errors or capture.digest != snapshot.inspection.content_digest:
                raise PluginUnavailable("plugin_material_changed")
            data = capture.read(member)
            try:
                body = data.decode("utf-8")
            except UnicodeError:
                raise PluginUnavailable("plugin_file_not_text") from None
            result = {
                "content": instruction_block(
                    snapshot.installation_id,
                    component_id,
                    snapshot.revision_digest,
                    body,
                ),
                "size": len(data),
                "truncated": False,
                "record_id": row["record_id"],
                "plugin_owned": True,
            }
            self._admission.check(snapshot, component_id)
            return result

        return await self._call(operation)

    def current_definition(self, name: str, admission_token: str) -> str:
        return self._call_from_agent(
            lambda: self._checked(name, admission_token)[2]["definition_digest"]
        )

    async def check_entries(self, entries: Sequence[Mapping[str, Any]]) -> None:
        await self._call(
            lambda: [
                self._checked(row["name"], row["plugin_admission"])
                for row in entries
                if row.get("plugin_owned")
            ]
        )

    def check_entries_from_agent(self, entries: Sequence[Mapping[str, Any]]) -> None:
        self._call_from_agent(
            lambda: [
                self._checked(row["name"], row["plugin_admission"])
                for row in entries
                if row.get("plugin_owned")
            ]
        )

    def live_runs(self) -> tuple[PluginRunOwnership, ...]:
        """Read exact cancellation/completion handles without waiting for storage."""
        with self._lock:
            return tuple(self._live_runs.values())

    def bind_run(
        self,
        entries: Sequence[Mapping[str, Any]],
        run_id: str,
        cancel: Callable[[], object],
        handle_id: str | None = None,
        *,
        parent_run_id: str | None = None,
    ) -> None:
        """Bind a real host run before effects, preserving its originating scope."""
        if not run_id:
            raise PluginUnavailable("plugin_run_identity_required")

        def operation():
            for row in entries:
                if not row.get("plugin_owned"):
                    continue
                snapshot, _, _ = self._checked(row["name"], row["plugin_admission"])
                key = (snapshot.installation_id, run_id)
                with self._lock:
                    if run_id in self._terminal_runs:
                        raise PluginUnavailable("plugin_run_terminal")
                    existing = self._live_runs.get(key)
                    if existing is not None:
                        if existing.pending_id != snapshot.run_id:
                            raise PluginUnavailable("plugin_run_identity_reused")
                        continue
                    if snapshot.run_id in self._retired:
                        raise PluginUnavailable("plugin_turn_retired")
                    if parent_run_id is None:
                        root = self._root_runs.get(snapshot.run_id)
                        if root is not None and root != run_id:
                            raise PluginUnavailable("plugin_root_identity_reused")
                    elif parent_run_id not in self._bound_runs.get(snapshot.run_id, ()):
                        raise PluginUnavailable("plugin_parent_identity_unavailable")
                    self.fences.check(snapshot.installation_id, snapshot.workspace_id)
                # Durable reservation can block; never hold the live seal lock
                # while performing SQLite or filesystem work.
                owner = self._coordinator.owner
                token = owner.reserve_launch(
                    snapshot.run_id,
                    snapshot.installation_id,
                    snapshot.workspace_id,
                    snapshot.revision_digest,
                )
                try:
                    owner.publish_process(
                        token,
                        {
                            "runtime": "console",
                            "run_id": run_id,
                            "handle_id": handle_id,
                            "pending_id": snapshot.run_id,
                        },
                    )
                    with self._lock:
                        if run_id in self._terminal_runs:
                            raise PluginUnavailable("plugin_run_terminal")
                        if snapshot.run_id in self._retired:
                            raise PluginUnavailable("plugin_turn_retired")
                        self.fences.check(
                            snapshot.installation_id, snapshot.workspace_id
                        )
                        self._live_runs[key] = PluginRunOwnership(
                            snapshot.installation_id,
                            snapshot.workspace_id,
                            snapshot.revision_digest,
                            snapshot.run_id,
                            run_id,
                            handle_id,
                            token,
                            cancel,
                            threading.Event(),
                            self._pending_custody[snapshot.run_id][0],
                            parent_run_id,
                        )
                        if parent_run_id is None:
                            self._root_runs[snapshot.run_id] = run_id
                        self._bound_runs.setdefault(snapshot.run_id, set()).add(run_id)
                except BaseException:
                    # bind_run has not returned: the host cannot have launched
                    # this run, so this unused reservation has no live writers.
                    owner.settle_process(token, True)
                    raise

        self._call_from_agent(operation)

    def complete_run(self, run_id: str) -> None:
        """Record actual host terminal evidence; coroutine cancellation is insufficient."""
        with self._lock:
            self._terminal_runs.add(run_id)
            records = [
                record for record in self._live_runs.values() if record.run_id == run_id
            ]
            for record in records:
                record.completed.set()
                self._live_runs.pop((record.installation_id, run_id), None)
        if records:
            self._call_from_agent(
                lambda: [
                    self._coordinator.owner.settle_process(record.lease_token, True)
                    for record in records
                ]
            )

    async def retire_pending(self, pending_id: str) -> None:
        """Fence reuse of a completed/rejected turn while retaining live children."""
        with self._lock:
            self._retired.add(pending_id)
        # Existing bound runs retain their immutable admissions until the host
        # reports terminal evidence. Merely returning from submit cannot drain.

    async def aclose(self) -> None:
        """Close after Console's owning lifecycle has drained its work."""
        with self._lock:
            if self._closed:
                return
            if self._live_runs:
                raise PluginUnavailable("plugin_owned_work_not_drained")
            self._closed = True
            thread = self._thread
            self._catalog = {}
        if thread is not None:
            try:
                loop = await asyncio.wrap_future(self._ready)
            except Exception:  # noqa: BLE001 -- a failed worker still needs joining.
                loop = None
            if loop is not None:
                loop.call_soon_threadsafe(loop.stop)
            await asyncio.to_thread(thread.join)
