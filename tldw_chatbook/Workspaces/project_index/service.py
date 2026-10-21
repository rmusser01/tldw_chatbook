"""One application-owned, explicitly started writer for folder-bound indexes."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext, ExitStack
from dataclasses import dataclass, replace
import os
import re
import stat
from pathlib import Path
import threading
from uuid import uuid4

from tldw_chatbook.DB.Workspace_Project_Index_DB import (
    prepare_scan_page,
    WorkspaceProjectIndexDB,
)
from .lifetime import BindingLifetime, LifetimeBusy
from .models import (
    MAX_CACHE_BYTES,
    IndexAuthorityFence,
    IndexAuthorityReader,
    ProjectIndexJobClaim,
    ProjectIndexKey,
    ProjectIndexPolicy,
    ProjectIndexScope,
    ProjectIndexStatus,
)


@dataclass
class _Run:
    claim: ProjectIndexJobClaim
    stop: threading.Event
    writer: threading.Thread | None = None
    supervisor: threading.Thread | None = None
    error: BaseException | None = None
    state: str = "Interrupted"
    db_closed: bool = False
    native_retired: bool = False
    pin: object | None = None
    lease: BindingLifetime | None = None


def binding_bytes(directory: Path) -> int:
    """Count every owned namespace/sidecar, with bounded traversal and no aliases."""
    total = count = 0
    pending = [directory]
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                count += 1
                if count > 4096:
                    return MAX_CACHE_BYTES
                info = entry.stat(follow_symlinks=False)
                if entry.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise ValueError("cache_alias_refused")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                else:
                    raise ValueError("cache_alias_refused")
    return total


class ProjectIndexService:
    """Synchronous management runs off the UI thread; only aclose is asynchronous.

    A service serializes its supplied scanner. Multiple applications coordinate
    with the stable binding capability and durable registry/store CAS.
    """

    def __init__(
        self,
        store_factory,
        scanner,
        authority_reader: IndexAuthorityReader,
        authority_fence: IndexAuthorityFence,
    ):
        self._factory = store_factory
        self._scanner = scanner
        self._reader = authority_reader
        self._fence = authority_fence
        self._lock = threading.RLock()
        self._runs: dict[str, _Run] = {}
        self._closed = False

    @contextmanager
    def _store(self, key):
        pin = self._retain(key)
        try:
            with self._bind(pin):
                with self._opened_store(key) as store:
                    yield store
        finally:
            if pin is not None:
                pin.close()

    def _retain(self, key):
        retain = getattr(self._factory, "retain", None)
        return retain(key) if retain is not None else None

    @staticmethod
    def _bind(pin):
        return pin.bind() if pin is not None else nullcontext()

    @contextmanager
    def _opened_store(self, key):
        if self._closed:
            raise RuntimeError("project_index_closed")
        store = self._factory(key)
        try:
            store.status_cache_bytes = binding_bytes(self._binding_root(store))
            yield store
        finally:
            store.close_native()

    @staticmethod
    def _binding_root(store):
        return store.db_path.parent.parent

    def _lifetime(self, store, key):
        lease = BindingLifetime.acquire(self._binding_root(store) / key.lifetime_name)
        return lease

    @staticmethod
    def _claim_contract(claim, key):
        if claim is not None and not claim.owner_token.startswith(
            "native-v1:" + key.binding_namespace + ":"
        ):
            raise ValueError("legacy_retirement_unproven")

    @contextmanager
    def _admitted(self, store, key):
        with self._fence(key) as authority:
            if authority is None:
                old = store.load_status(key)
                if old.state != "Off":
                    store.revoke(key, clear=False, expected_scope=old.scope)
                yield None
            else:
                store.reconcile_authority(key, authority.authority_revision)
                yield authority

    def status(self, key: ProjectIndexKey) -> ProjectIndexStatus:
        with self._store(key) as store:
            with self._admitted(store, key):
                result = store.load_status(key)
                claim = store.read_claim(result.scope)
            # Reacquisition establishes that a durable Building claim is inert;
            # no automatic Resume, even when no local notification ever arrived.
            run = self._runs.get(result.job_id)
            if (
                claim is not None
                and result.state == "Building"
                and (run is None or run.native_retired)
            ):
                try:
                    lease = self._lifetime(store, key)
                except LifetimeBusy:
                    pass
                else:
                    try:
                        with self._admitted(store, key):
                            if store.read_claim(result.scope) == claim:
                                store.suspend_job(
                                    claim, state="Interrupted", advance=False
                                )
                                result = store.load_status(key)
                    finally:
                        lease.close()
            return replace(result, cache_bytes=binding_bytes(self._binding_root(store)))

    def enable(
        self,
        key: ProjectIndexKey,
        policy: ProjectIndexPolicy,
        *,
        expected_scope: ProjectIndexScope,
    ) -> ProjectIndexStatus:
        with self._lock, self._store(key) as store:
            lease = self._lifetime(store, key)
            try:
                with self._admitted(store, key) as authority:
                    store.assert_native_lifetime()
                    if authority is None:
                        raise ValueError("authority_unavailable")
                    self._claim_contract(store.read_claim(expected_scope), key)
                    store.configure(
                        key, policy, enabled=True, expected_scope=expected_scope
                    )
                    return store.load_status(key)
            finally:
                lease.close()

    def _stop_local(self, runs: tuple[_Run, ...]) -> None:
        for run in runs:
            run.stop.set()
        for run in runs:
            if run.supervisor is not None:
                run.supervisor.join()
            if not run.native_retired:
                raise RuntimeError("index_retirement_unproven") from run.error

    def _revoke(self, key, expected_scope, *, clear):
        with self._store(key) as store:
            # Serialize capture with _start's claim/dispatch interval, but never
            # hold either lock while signalling or joining native borrowers.
            with self._lock, self._fence(key) as authority:
                if expected_scope is not None and authority is None:
                    raise ValueError("authority_unavailable")
                if authority is not None:
                    store.reconcile_authority(key, authority.authority_revision)
                revoked = store.revoke(key, clear=False, expected_scope=expected_scope)
                affected = tuple(
                    run for run in self._runs.values() if run.claim.scope.key == key
                )
            self._stop_local(affected)
            if clear:
                # Revocation is already durable if external retirement or cleanup
                # fails. Retry Clear explicitly; there is no background reopen.
                self._clear_binding(store, key, revoked)
                store.status_cache_bytes = binding_bytes(self._binding_root(store))
            return store.load_status(key)

    def _clear_binding(self, current_store, key, revoked):
        group = self._binding_root(current_store)
        # This inventory is bounded and grants cleanup ownership only. Unknown
        # layouts, aliases and legacy namespaces never become inferred owners.
        binding_bytes(group)

        def inventory():
            namespaces = []
            leases = {group / key.lifetime_name}
            for count, entry in enumerate(group.iterdir(), 1):
                if count > 4096:
                    raise ValueError("index_cleanup_inventory_limit")
                if entry.is_file() and re.fullmatch(
                    r"[0-9a-f]{64}\.lifetime\.lock", entry.name
                ):
                    leases.add(entry)
                elif entry.is_dir() and re.fullmatch(r"[0-9a-f]{64}", entry.name):
                    path = entry / "workspace_project_index.db"
                    allowed = {
                        path.name + suffix
                        for suffix in ("", "-journal", "-wal", "-shm")
                    }
                    if not path.is_file() or any(
                        child.name not in allowed for child in entry.iterdir()
                    ):
                        raise ValueError("index_cleanup_namespace_unknown")
                    namespaces.append(path)
                else:
                    raise ValueError("index_cleanup_namespace_unknown")
            return namespaces, leases

        namespaces, leases = inventory()
        with ExitStack() as resources:
            for path in sorted(leases):
                lease = BindingLifetime.acquire(path)
                resources.callback(lease.close)
            namespaces, observed_leases = inventory()
            if observed_leases != leases:
                raise ValueError("index_cleanup_inventory_changed")
            with self._fence(key) as authority:
                if authority is None:
                    raise ValueError("authority_unavailable")
                current = current_store.load_status(key)
                if current.scope != revoked or current.state != "Off":
                    raise ValueError("stale_scope")
            # Preflight all namespaces before any derived-data cleanup. In
            # particular, opening a v1 namespace for cleanup cannot migrate it.
            for path in namespaces:
                store = WorkspaceProjectIndexDB(path, cleanup_only=True)
                try:
                    store.cleanup_eligible(path.parent.name, key.binding_namespace)
                finally:
                    store.close_native()
            for path in sorted(namespaces, key=lambda value: value.stat().st_size):
                total = binding_bytes(group)
                size = path.stat().st_size
                # A DELETE-journal transaction cannot journal a page twice.
                # Reserve all existing DB pages plus 8-byte records and headers.
                headroom = size + size // 64 + 65536
                if total + headroom > MAX_CACHE_BYTES:
                    raise ValueError("index_cleanup_headroom_required")
                store = (
                    current_store
                    if path == current_store.db_path
                    else WorkspaceProjectIndexDB(path, cleanup_only=True)
                )
                try:
                    with store.connection() as conn:
                        pages = conn.execute("PRAGMA page_count").fetchone()[0]
                        conn.execute(f"PRAGMA max_page_count={pages}")
                    if store is current_store:
                        if not store.clear_revoked(revoked):
                            raise ValueError("stale_scope")
                    else:
                        store.clear_owned_namespace(
                            path.parent.name, key.binding_namespace
                        )
                    store.reclaim_free_pages()
                finally:
                    if store is not current_store:
                        store.close_native()

    def disable(
        self, key: ProjectIndexKey, *, expected_scope: ProjectIndexScope
    ) -> ProjectIndexStatus:
        return self._revoke(key, expected_scope, clear=False)

    def clear(
        self, key: ProjectIndexKey, *, expected_scope: ProjectIndexScope
    ) -> ProjectIndexStatus:
        return self._revoke(key, expected_scope, clear=True)

    def revoke(self, key: ProjectIndexKey) -> None:
        self._revoke(key, None, clear=False)

    def pause(
        self,
        job_id: str,
        *,
        expected_scope: ProjectIndexScope,
        expected_job_revision: int,
    ) -> ProjectIndexStatus:
        key = expected_scope.key
        with self._store(key) as store:
            with self._lock, self._admitted(store, key):
                current = store.load_status(key)
                claim = store.read_claim(expected_scope)
                if current.scope != expected_scope:
                    raise ValueError("stale_scope")
                if (
                    claim is None
                    or claim.job_id != job_id
                    or claim.job_revision != expected_job_revision
                    or current.state != "Building"
                ):
                    raise ValueError("stale_job")
                store.suspend_job(claim, state="Paused")
                affected = tuple(
                    run for run in self._runs.values() if run.claim == claim
                )
            self._stop_local(affected)
            return store.load_status(key)

    def refresh(
        self,
        key: ProjectIndexKey,
        *,
        expected_scope: ProjectIndexScope,
        expected_job_revision: int,
    ) -> str:
        if expected_scope.key != key:
            raise ValueError("stale_scope")
        return self._start(expected_scope, expected_job_revision, None)

    def resume(
        self,
        job_id: str,
        *,
        expected_scope: ProjectIndexScope,
        expected_job_revision: int,
    ) -> str:
        return self._start(expected_scope, expected_job_revision, job_id)

    def _start(self, scope, revision, resume_id):
        pin = self._retain(scope.key)
        try:
            with self._bind(pin):
                return self._start_pinned(scope, revision, resume_id, pin)
        except BaseException:
            if pin is not None and not any(
                run.pin is pin for run in self._runs.values()
            ):
                pin.close()
            raise

    def _start_pinned(self, scope, revision, resume_id, pin):
        with self._lock:
            if any(not run.native_retired for run in self._runs.values()):
                raise LifetimeBusy("scanner_busy")
            store = self._factory(scope.key)
            lease = None
            run = None
            try:
                store.status_cache_bytes = binding_bytes(self._binding_root(store))
                if self._closed:
                    raise RuntimeError("project_index_closed")
                lease = self._lifetime(store, scope.key)
                with self._admitted(store, scope.key) as authority:
                    if authority is None:
                        raise ValueError("authority_unavailable")
                    current = store.load_status(scope.key)
                    if current.scope != scope:
                        raise ValueError("stale_scope")
                    if current.job_revision != revision:
                        raise ValueError("stale_job")
                    old = store.read_claim(scope)
                    store.assert_native_lifetime()
                    self._claim_contract(old, scope.key)
                    token = (
                        "native-v1:" + scope.key.binding_namespace + ":" + uuid4().hex
                    )
                    if resume_id is not None or old is not None:
                        if (
                            old is None
                            or (resume_id is not None and old.job_id != resume_id)
                            or current.state
                            not in {"Paused", "Interrupted", "Partial", "Error"}
                        ):
                            raise ValueError("stale_job")
                        # The held exclusive acquisition proves every prior
                        # inherited holder retired. This callback performs no I/O.
                        claim = store.replace_retired_job(
                            old,
                            owner_token=token,
                            authority_revision=authority.authority_revision,
                            retirement_verifier=lambda observed: observed == old
                            and not lease.closed,
                            restart=resume_id is None,
                        )
                    else:
                        if old is not None or current.state not in {
                            "Paused",
                            "Ready",
                            "Partial",
                            "Interrupted",
                            "Error",
                        }:
                            raise ValueError("stale_job")
                        claim = store.claim_job(
                            scope,
                            owner_token=token,
                            authority_revision=authority.authority_revision,
                            expected_job_revision=revision,
                        )
                    policy = store.policy(scope)
                # The management thread's native connection ends before dispatch.
                store.close_native()
                run = _Run(claim, threading.Event(), pin=pin, lease=lease)
                self._runs[claim.job_id] = run
                self._scanner.lifetime = lease
                run.writer = threading.Thread(
                    target=self._write,
                    args=(run, store, authority, policy, pin),
                    name="project-index-writer",
                )
                run.supervisor = threading.Thread(
                    target=self._retire,
                    args=(run, store, lease, pin),
                    name="project-index-owner",
                )
                run.writer.start()
                run.supervisor.start()
                return claim.job_id
            except BaseException:
                if run is not None and run.writer.ident is not None:
                    # A failed supervisor launch cannot release a live writer's
                    # capability. The dispatch caller becomes its physical owner.
                    run.stop.set()
                    run.supervisor = None
                    self._retire(run, store, lease, pin)
                else:
                    store.close_native()
                    if lease is not None:
                        lease.close()
                    if run is not None:
                        run.native_retired = True
                        run.supervisor = None
                        if pin is not None:
                            pin.close()
                raise

    def _write(self, run, store, authority, policy, pin):
        with self._bind(pin):
            self._write_bound(run, store, authority, policy)

    def _write_bound(self, run, store, authority, policy):
        from tldw_chatbook.Backup_Recovery.activation import execution_scope
        from .models import OWNER_ID

        claim = run.claim
        attempted = set()

        def frontier():
            while not run.stop.is_set():
                after = None
                while True:
                    pending = store.pending_directories(claim.scope, after=after)
                    fresh = [path for path in pending if path not in attempted]
                    if fresh or len(pending) < 256:
                        break
                    after = pending[-1]
                if not fresh:
                    break
                for path in fresh:
                    attempted.add(path)
                    yield path

        def admit(directory):
            if run.stop.is_set():
                return None
            with self._admitted(store, claim.scope.key) as current:
                if (
                    run.stop.is_set()
                    or current is None
                    or current.authority_revision != claim.authority_revision
                    or current.exclusions != authority.exclusions
                    or not store.admit_job(claim)
                ):
                    return None
                return current

        def emit(page):
            prepared = prepare_scan_page(page)
            store.publication_cache_bytes = binding_bytes(self._binding_root(store))
            # Reserve rollback headroom for this namespace after all obsolete
            # namespaces and the stable capability have been counted.
            with store.connection() as conn:
                page_size = conn.execute("PRAGMA page_size").fetchone()[0]
                other = max(0, store.publication_cache_bytes - store._cache_bytes())
                pages = max(1, (MAX_CACHE_BYTES - other - 1024**2) // (2 * page_size))
                conn.execute(f"PRAGMA max_page_count={pages}")
            with self._admitted(store, claim.scope.key) as live:
                if (
                    run.stop.is_set()
                    or live is None
                    or live.authority_revision != claim.authority_revision
                    or live.exclusions != authority.exclusions
                ):
                    return False
                return store.commit_scan_page(
                    claim.scope,
                    claim.job_id,
                    prepared,
                    expected_job_revision=claim.job_revision,
                    authority_revision=live.authority_revision,
                )

        try:
            with execution_scope((OWNER_ID,), store.db_path) as allowed:
                if not allowed:
                    raise RuntimeError("index_execution_not_admitted")
                result = self._scanner.scan(
                    authority,
                    claim.scope,
                    claim.job_id,
                    frontier(),
                    emit,
                    run.stop,
                    admit_unit=admit,
                    policy=policy,
                )
                state = store.load_status(claim.scope.key).state
                run.state = (
                    "Partial"
                    if state == "Partial"
                    else {
                        "complete": "Ready",
                        "partial": "Partial",
                        "paused": "Paused",
                        "interrupted": "Interrupted",
                        "error": "Error",
                    }[result.status]
                )
                if run.state == "Ready" and store.pending_directories(claim.scope):
                    run.state = "Partial"
                store.close_native()
                run.db_closed = True
        except BaseException as error:
            run.state = "Error"
            run.error = error
        finally:
            try:
                store.close_native()
                run.db_closed = True
            except BaseException as error:
                run.error = error

    def _retire(self, run, store, lease, pin):
        with self._bind(pin):
            self._retire_bound(run, store, lease, pin)

    def _retire_bound(self, run, store, lease, pin):
        run.writer.join()  # actual thread exit, separately from coroutine return
        try:
            if not run.db_closed or not self._scanner.retired:
                raise RuntimeError("index_retirement_unproven")
            with self._admitted(store, run.claim.scope.key) as authority:
                current = store.read_claim(run.claim.scope)
                if authority is not None and current == run.claim:
                    if run.state == "Ready":
                        store.retire_job(
                            run.claim.scope,
                            run.claim.job_id,
                            owner_token=run.claim.owner_token,
                            expected_job_revision=run.claim.job_revision,
                            state="Ready",
                        )
                    else:
                        store.suspend_job(run.claim, state=run.state, advance=False)
        except BaseException as error:
            run.error = error
        finally:
            try:
                store.close_native()
                if run.db_closed and self._scanner.retired:
                    self._scanner.lifetime = None
                    lease.close()
                    if pin is not None:
                        pin.close()
                    run.native_retired = True
            except BaseException as error:
                run.error = error

    def binding_changed(self, binding_id: str) -> None:
        """Post-commit hint only: stop local work; next admission reads the witness."""
        for run in tuple(self._runs.values()):
            if run.claim.scope.key.binding_id == binding_id:
                run.stop.set()

    async def aclose(self) -> None:
        """Stop admission and await actual owner threads despite waiter cancellation."""
        with self._lock:
            self._closed = True
            runs = tuple(self._runs.values())
            for run in runs:
                run.stop.set()

        async def drain():
            for run in runs:
                if run.supervisor is not None:
                    await asyncio.to_thread(run.supervisor.join)
                if not run.native_retired:
                    raise RuntimeError("index_retirement_unproven") from run.error

        task = asyncio.create_task(drain())
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError
