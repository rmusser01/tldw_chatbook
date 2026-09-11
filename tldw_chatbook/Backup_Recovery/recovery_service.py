"""One app-owned worker composes existing local recovery engines."""

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock
from types import MappingProxyType
from uuid import uuid4

from . import archive_reader
from .limits import ArchiveLimits
from .native_files import create_private_directory, pinned_directory
from .recovery_copies import hold_recovery_copy, list_recovery_copies
from .service_storage import default_control_root, ensure_storage

__all__ = ["RecoveryService", "default_control_root", "issue_code"]


def issue_code(error: Exception, *, kind: str = "") -> str:
    """Only fixed local codes cross into a view; arbitrary exception text stays out."""
    from .capture import CaptureReviewRequired
    from .crypto import CryptoError

    if isinstance(error, InterruptedError):
        return "cancelled"
    if isinstance(error, CaptureReviewRequired):
        return "review_required"
    if isinstance(error, archive_reader.CompressionReviewRequired):
        return "compression_review_required"
    if isinstance(error, CryptoError):
        code = error.args[0] if error.args and type(error.args[0]) is str else None
        if code == "cancelled":
            return "cancelled"
        if code == "invalid_password":
            return "password_required"
        if code in {"helper_unavailable", "helper_integrity_mismatch"}:
            return "encryption_unavailable"
        return (
            "archive_unlock_failed"
            if kind.startswith("inspect")
            else "encryption_failed"
        )
    if isinstance(error, FileExistsError):
        return "destination_exists"
    if isinstance(error, PermissionError):
        return "permission_denied"
    known = {
        "insufficient_space",
        "scope_changed",
        "target_changed",
        "sealed_changed",
        "output_overlap",
        "invalid_backup_suffix",
        "path_collision",
        "encryption_password_required",
        "rollback_password_required",
        "recovery_copy_held",
        "recovery_copy_pending",
        "recovery_action_unavailable",
        "preview_required",
        "archive_plan_mismatch",
        "recovery_operation_running",
    }
    if (
        isinstance(error, ValueError)
        and len(error.args) == 1
        and type(error.args[0]) is str
        and error.args[0] in known
    ):
        return error.args[0]
    return "backup_operation_failed"


class RecoveryService:
    """Views observe this worker; closing a view never cancels its operation.

    Passwords belong only to queued/running call frames. Completed futures contain
    no results or exception tracebacks that could retain those frames.
    """

    issue_code = staticmethod(issue_code)

    def __init__(self, control_root):
        self.control_root = Path(control_root)
        if not self.control_root.is_absolute() or ".." in self.control_root.parts:
            raise ValueError("invalid_control_root")
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="backup-recovery"
        )
        self._closed = False
        self._states = {}
        self._futures = {}
        self._cancellations = {}
        self._archives = {}
        self._workspaces = {}
        self._retained_workspaces = set()

    def _update(self, operation, **changes):
        with self._lock:
            current = dict(self._states[operation])
            if "result" in changes:
                changes["result"] = MappingProxyType(dict(changes["result"]))
            current.update(changes)
            self._states[operation] = MappingProxyType(current)

    def _start(self, kind, function):
        from .capture import CaptureReviewRequired

        with self._lock:
            if self._closed:
                raise ValueError("recovery_service_closed")
            if any(not future.done() for future in self._futures.values()):
                raise ValueError("recovery_operation_running")
            operation = uuid4().hex
            cancel = Event()
            self._cancellations[operation] = cancel
            self._states[operation] = MappingProxyType(
                {
                    "operation_id": operation,
                    "kind": kind,
                    "state": "running",
                    "phase": "preparing",
                    "cancellation_requested": False,
                    "result": MappingProxyType({}),
                    "issues": (),
                    "review_issues": (),
                }
            )

            def run():
                try:
                    archive_reader._check(cancel)
                    function(operation, cancel)
                except InterruptedError:
                    self._failed(operation, "cancelled")
                except CaptureReviewRequired as error:
                    self._update(operation, review_issues=error.issues)
                    self._failed(operation, "review_required")
                except Exception as error:  # noqa: BLE001 - discard secret-bearing tracebacks after mapping a fixed code.
                    self._failed(operation, issue_code(error, kind=kind))
                else:
                    self._update(operation, state="succeeded")

            self._futures[operation] = self._executor.submit(run)
            return operation

    def _failed(self, operation, issue):
        with self._lock:
            recovery = self._states[operation]["phase"] == "recovery_required"
        self._update(
            operation,
            state="recovery_required"
            if recovery
            else ("cancelled" if issue == "cancelled" else "failed"),
            issues=("recovery_required",) if recovery else (issue,),
        )

    def _workspace(self, operation):
        work = ensure_storage(self.control_root) / ("inspection-" + operation)
        create_private_directory(work)
        info = work.stat(follow_symlinks=False)
        with self._lock:
            self._workspaces[operation] = work, (info.st_dev, info.st_ino)
        return work

    def _inspect(self, operation, cancel, source, password, limits):
        work = self._workspace(operation)
        self._update(operation, phase="inspecting")
        archive = archive_reader.acquire(
            source, work / "acquired", limits, password, cancel
        )
        archive_reader._check(cancel)
        with self._lock:
            self._archives[operation] = archive
        self._update(
            operation,
            phase="archive_verified",
            result={
                "path": str(source),
                "sha256": archive.digest,
                "archive_verified": True,
            },
        )

    def start_inspection(self, source: Path, *, password: bytes | None, limits=None):
        """Acquire and verify an explicit input on the app-owned worker."""
        source = Path(source)
        limits = limits or ArchiveLimits()
        return self._start(
            "inspect",
            lambda operation, cancel: self._inspect(
                operation,
                cancel,
                source,
                password,
                limits,
            ),
        )

    def start_copy_inspection(self, operation_id: str, *, password: bytes, limits=None):
        """Keep the native ciphertext hold through acquisition and cancellation."""
        limits = limits or ArchiveLimits()

        def inspect(operation, cancel):
            with hold_recovery_copy(self.control_root, operation_id) as entry:
                self._inspect(operation, cancel, entry.path, password, limits)

        return self._start("inspect_recovery_copy", inspect)

    def start_rollback_preview(
        self, operation_id, *, old_password, target, acknowledged_credential_issues=()
    ):
        """Authenticate a retained copy and expose its current replacement plan."""

        def preview(operation, cancel):
            from .later_rollback import preview_rollback

            self._update(operation, phase="reviewing_rollback")
            plan = preview_rollback(
                operation_id,
                control_root=self.control_root,
                old_password=old_password,
                target=target,
                cancel=cancel,
                acknowledged_credential_issues=acknowledged_credential_issues,
            )
            self._update(
                operation, phase="rollback_review_ready", result={"plan": plan}
            )

        return self._start("preview_rollback", preview)

    def preview_rollback(self, operation_id, **choices):
        """Blocking convenience for CLI/view workers; the service owns native work."""
        operation = self.start_rollback_preview(operation_id, **choices)
        state = self.wait(operation)
        if state["state"] != "succeeded":
            raise ValueError(state["issues"][0])
        return state["result"]["plan"]

    def start_rollback(self, operation_id, plan, *, old_password, new_password):
        """Make a new verified safety copy before applying the selected old snapshot."""

        def rollback(operation, cancel):
            from .recovery_copies import rollback as rollback_copy

            self._update(operation, phase="replacing")
            try:
                journal_operation = rollback_copy(
                    operation_id,
                    control_root=self.control_root,
                    old_password=old_password,
                    new_password=new_password,
                    cancel=cancel,
                    approved_plan=plan,
                )
            except Exception:
                self._link_pending_candidate(operation, None, plan=plan)
                raise
            self._update(
                operation,
                phase="restoration_validated",
                result={
                    "journal_operation_id": journal_operation,
                    "restoration_validated": True,
                },
            )

        return self._start("later_rollback", rollback)

    def inspection(self, operation_id):
        """Return the immutable acquired artifact for a subsequent local plan."""
        with self._lock:
            if self._closed or self._states[operation_id]["state"] != "succeeded":
                raise ValueError("archive_inspection_required")
            return self._archives[operation_id]

    def summary(self, operation_id):
        """Present inert verified metadata without exposing secret payloads/locators."""
        doc = archive_reader.verify_sealed(self.inspection(operation_id))
        root_owners = {}
        for item in doc.files:
            root_owners.setdefault(item.root_id, set()).add(item.owner_id)
        roots = tuple(
            MappingProxyType(
                {
                    "logical_id": row.logical_id,
                    "synthetic": row.synthetic,
                    "owners": tuple(sorted(root_owners.get(row.logical_id, ()))),
                }
            )
            for row in doc.directories
            if row.parent_id is None
        )
        config_profiles = {
            row.logical_id.split(":")[1]
            for row in doc.files
            if row.owner_id == "config"
            and len(row.logical_id.split(":")) == 3
            and row.logical_id.startswith("profile:")
        }
        return MappingProxyType(
            {
                "format_version": doc.format_version,
                "producer_version": doc.producer_version,
                "captured_at": doc.captured_at,
                "profile_ids": doc.profile_ids,
                "consistency": doc.consistency,
                "credential_policy": doc.credential_policy,
                "file_count": len(doc.files),
                "payload_bytes": sum(row.size for row in doc.files),
                "owners": tuple(row.owner_id for row in doc.owners),
                "dependency_groups": tuple(
                    MappingProxyType(
                        {
                            "group_id": row.group_id,
                            "members": row.members,
                            "complete": row.complete,
                        }
                    )
                    for row in doc.dependency_groups
                ),
                "required_capabilities": doc.required_capabilities,
                "exclusions": tuple(
                    (row.logical_id, row.reason) for row in doc.exclusions
                ),
                "report": doc.report.lines,
                "roots": roots,
                "destination_slots": tuple(
                    MappingProxyType(
                        {
                            "logical_id": row["logical_id"],
                            "kind": "archive_root",
                            "owners": row["owners"],
                        }
                    )
                    for row in roots
                )
                + tuple(
                    MappingProxyType(
                        {
                            "logical_id": f"profile:{profile}:paths.data_dir",
                            "kind": "data_root",
                            "owners": ("config",),
                        }
                    )
                    for profile in sorted(config_profiles)
                ),
                "archive_verified": True,
            }
        )

    def preview_restore(self, inspection_id, **choices):
        from .restore_plan import plan_restore

        return plan_restore(self.inspection(inspection_id), **choices)

    def start_extraction_preview(
        self, inspection_id, *, group_ids, destination, limits=None
    ):
        """Review manual byte extraction on the retained worker before publication."""
        from .inert_extraction import preview_inert_extraction
        from .service_storage import work_root

        archive = self.inspection(inspection_id)

        def preview(operation, cancel):
            self._update(operation, phase="reviewing_extraction")
            plan = preview_inert_extraction(
                archive,
                group_ids=group_ids,
                destination=Path(destination),
                limits=limits or ArchiveLimits(),
                cancel=cancel,
                protected_roots=(self.control_root, work_root(self.control_root)),
            )
            self._update(
                operation, phase="extraction_review_ready", result={"plan": plan}
            )

        return self._start("preview_extraction", preview)

    def start_extraction(self, inspection_id, plan):
        """Publish reviewed opaque files without installing or activating a profile."""
        from .inert_extraction import InertExtractionPlan, extract_inert
        from .service_storage import work_root

        archive = self.inspection(inspection_id)
        if type(plan) is not InertExtractionPlan or not {
            self.control_root,
            work_root(self.control_root),
        }.issubset(plan.protected_roots):
            raise ValueError("extraction_preview_required")

        def extract(operation, cancel):
            self._update(operation, phase="extracting_inert_files")
            result = extract_inert(archive, plan, cancel=cancel)
            self._update(
                operation,
                phase="inert_extracted",
                result={
                    "path": str(result.destination),
                    "report_path": str(result.report_path),
                    "archive_digest": result.archive_digest,
                    "group_ids": result.group_ids,
                    "inert_extracted": True,
                },
            )

        return self._start("extract_inert", extract)

    def start_restore(self, inspection_id, plan, *, rollback_password=None):
        from .restore_plan import RestorePlan

        archive = self.inspection(inspection_id)
        if type(plan) is not RestorePlan or plan.archive_digest != archive.digest:
            raise ValueError("archive_plan_mismatch")
        if plan.mode == "replace":
            from .replacement import require_rollback_password

            require_rollback_password(rollback_password)
        elif plan.mode != "isolated":
            raise ValueError("invalid_restore_mode")

        def restore(operation, cancel):
            if plan.mode == "isolated":
                from .isolated_restore import restore_isolated

                self._update(operation, phase="restoring")
                try:
                    profile = restore_isolated(archive, plan, self.control_root, cancel)
                except Exception:
                    self._link_pending_candidate(operation, None, plan=plan)
                    raise
                result = {"profile_id": profile, "restoration_validated": True}
            else:
                from . import bootstrap
                from .control_records import UNBOUND_NAMESPACE, admission_authority
                from .replacement import replace
                from .staging import stage_restore

                self._update(operation, phase="staging")
                # Candidate and publication evidence outlive the view and worker.
                with self._lock:
                    work, _ = self._workspaces[inspection_id]
                    self._retained_workspaces.add(inspection_id)
                authority = admission_authority(bootstrap.default_bootstrap_root())
                with authority.maintenance(
                    (UNBOUND_NAMESPACE,), 30, cancel=cancel
                ) as session:
                    candidate = stage_restore(
                        archive,
                        plan,
                        work / ("restore-" + operation),
                        cancel,
                        session=session,
                    )
                self._update(operation, phase="replacing")
                try:
                    journal_operation = replace(
                        plan,
                        candidate,
                        control_root=self.control_root,
                        rollback_password=rollback_password,
                        cancel=cancel,
                    )
                except Exception:
                    self._link_pending_candidate(operation, candidate)
                    raise
                result = {
                    "journal_operation_id": journal_operation,
                    "restoration_validated": True,
                }
            self._update(operation, phase="restoration_validated", result=result)

        return self._start("restore", restore)

    def _link_pending_candidate(self, operation, candidate, *, plan=None):
        from .publication import _plan_digest
        from .recovery_copies import _journal

        for pending in self.pending_operations():
            journal = _journal(self.control_root, pending["operation_id"])
            with journal._locked(exclusive=False) as parent:
                records = journal._records(parent)
            matched = records and (
                records[0].evidence.get("stage", {}).get("path") == str(candidate)
                if candidate is not None
                else records[0].evidence.get("plan_digest") == _plan_digest(plan)
            )
            if matched:
                self._update(
                    operation,
                    phase="recovery_required",
                    result={"journal_operation_id": journal.operation_id},
                )
                return

    def pending_operations(self):
        """Discover strict fixed pending records even during an activation pair write.

        This only offers recovery. Ordinary startup remains fenced, and the
        selected executor independently validates the exact operation and pair.
        """
        from . import bootstrap
        from .control_records import _Pending

        root = bootstrap.default_bootstrap_root()
        try:
            root.lstat()
        except FileNotFoundError:
            return ()
        pending = []
        with pinned_directory(root) as parent:
            info = os.fstat(parent)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError("bootstrap_not_private")
            with os.scandir(parent) as entries:
                for index, entry in enumerate(entries):
                    if index >= bootstrap.MAX_RECORDS:
                        raise ValueError("too_many_records")
                    if not entry.name.startswith("pending-"):
                        continue
                    raw = bootstrap._read(parent, entry.name)
                    row = _Pending.model_validate(raw).model_dump()
                    if (
                        set(raw) != set(row)
                        or entry.name
                        != "pending-" + bootstrap._key(row["operation_id"]) + ".json"
                        or not bootstrap._strings(row["namespaces"])
                        or not bootstrap._paths(row["selectors"])
                        or not bootstrap._paths([row["control_root"]])
                    ):
                        raise ValueError("invalid_pending")
                    pending.append(row)
        return tuple(
            MappingProxyType(
                {
                    "operation_id": row["operation_id"],
                    "config_paths": tuple(row["selectors"]),
                }
            )
            for row in sorted(pending, key=lambda row: row["operation_id"])
            if Path(row["control_root"]) == self.control_root
        )

    def start_recovery(self, operation_id, *, action, rollback_password=None):
        """Resume an explicit durable operation through its existing executor."""
        state = self.status(operation_id)
        if action not in state["actions"]:
            raise ValueError("recovery_action_unavailable")

        def recover(operation, cancel):
            from .replacement import recover_replacement

            self._update(
                operation,
                phase="recovery_required",
                result={"journal_operation_id": operation_id},
            )
            if state["mode"] == "isolated":
                from . import bootstrap
                from .isolated_restore import _finish_isolated
                from .plan_records import load_plan
                from .recovery_copies import _journal

                journal = _journal(self.control_root, operation_id)
                plan = load_plan(journal)
                with journal._locked(exclusive=False) as parent:
                    records = journal._records(parent)
                prepared = next(
                    (row for row in records if row.event == "prepared"), None
                )
                if prepared is not None:
                    from .control_records import _recovery_pending
                    from .journal import _Prepared

                    association = _recovery_pending(
                        bootstrap.default_bootstrap_root(),
                        journal,
                        _Prepared.model_validate(prepared.evidence),
                    )
                    names = tuple(association.namespaces)
                else:
                    # Pair publication cannot precede prepared/installed evidence.
                    pending, _ = bootstrap._records(bootstrap.default_bootstrap_root())
                    association = next(
                        row for row in pending if row["operation_id"] == operation_id
                    )
                    if Path(association["control_root"]) != self.control_root:
                        raise ValueError("recovery_control_mismatch")
                    names = tuple(association["namespaces"])
                _finish_isolated(
                    Path(records[0].evidence["stage"]["path"]),
                    plan,
                    journal,
                    names,
                    operation_id,
                    cancel,
                )
                outcome = "committed"
            else:
                outcome = recover_replacement(
                    operation_id,
                    control_root=self.control_root,
                    action=action,
                    rollback_password=rollback_password,
                    cancel=cancel,
                )
            self._update(
                operation,
                phase=outcome,
                result={
                    "journal_operation_id": operation_id,
                    "restoration_validated": outcome == "committed",
                    "rolled_back": outcome == "rolled_back",
                    "aborted": outcome == "aborted",
                },
            )

        return self._start("recover", recover)

    def recovery_copies(self):
        return list_recovery_copies(self.control_root)

    def delete_copy(self, operation_id, *, user_selected):
        from .recovery_copies import delete_recovery_copy

        return delete_recovery_copy(
            self.control_root, operation_id, user_selected=user_selected
        )

    def start_delete_copy(self, operation_id, *, user_selected):
        """Explicit deletion stays on the app-owned worker through native completion."""

        def delete(operation, cancel):
            archive_reader._check(cancel)
            self._update(operation, phase="deleting_recovery_copy")
            self.delete_copy(operation_id, user_selected=user_selected)
            self._update(
                operation,
                phase="recovery_copy_deleted",
                result={"deleted_operation_id": operation_id},
            )

        return self._start("delete_recovery_copy", delete)

    def profiles(self):
        """Enumerate locally associated restored profiles and recheck launch evidence."""
        from . import bootstrap
        from .isolated_restore import profile_requirements
        from .journal import _Prepared
        from .recovery_copies import _journal

        _, bindings = bootstrap._records(bootstrap.default_bootstrap_root())
        operations = {
            row["activation"]["operation_id"]
            for row in bindings
            if row.get("activation", {}).get("store_root")
            == str(self.control_root / "activation")
        }
        profiles = {}
        for operation in sorted(operations):
            journal = _journal(self.control_root, operation)
            with journal._locked(exclusive=False) as parent:
                records = journal._records(parent)
            prepared = _Prepared.model_validate(
                next(row.evidence for row in records if row.event == "prepared")
            )
            for entry in prepared.isolated_profiles:
                try:
                    requirements = profile_requirements(entry.profile_id, self.control_root)
                except (OSError, ValueError, RuntimeError):
                    status = "recovery_required"
                    requirements = {
                        "generation": None,
                        "required_owners": None,
                        "pending_owners": None,
                        "requirements_checked": False,
                        "needs_setup": None,
                    }
                else:
                    status = "restoration_validated"
                profiles[entry.profile_id] = MappingProxyType(
                    {
                        "profile_id": entry.profile_id,
                        "config": entry.config,
                        "data": entry.data,
                        "status": status,
                        **requirements,
                    }
                )
        return tuple(profiles[key] for key in sorted(profiles))

    def start_open_profile(self, profile_id):
        """The view suspends its terminal while this fresh app owns it."""

        def launch(operation, cancel):
            from .isolated_restore import launch_profile

            archive_reader._check(cancel)
            self._update(operation, phase="opening_profile")
            code = launch_profile(profile_id, self.control_root)
            if code:
                raise ValueError("profile_process_failed")
            # Process exit alone is not a mounted/opened validation receipt.
            self._update(
                operation,
                phase="profile_process_exited",
                result={"profile_id": profile_id, "exit_code": code},
            )

        return self._start("open_profile", launch)

    def preview_backup(self, config_paths, *, options, include_known_profiles=False):
        from .capture import _capture_options
        from .capture_service import preview_capture

        settings, _, _, _ = _capture_options(options)
        if settings["credential_mode"] == "rollback":
            raise ValueError("invalid_backup_credential_mode")
        return preview_capture(
            config_paths, options=settings, include_known_profiles=include_known_profiles
        )

    def preview_backup_details(
        self, config_paths, *, options, destination, include_known_profiles=False
    ):
        """Show current per-volume capture estimates; execution rechecks actual space."""
        from .capture import _capture_options
        from .profile_paths import lexical_path
        from .space import _MARGIN, _volume

        settings, _, _, _ = _capture_options(options)
        inventory = self.preview_backup(
            config_paths, options=settings, include_known_profiles=include_known_profiles
        )
        estimate = sum(
            item.path.stat().st_size
            for item in inventory.items
            if item.path is not None and item.status == "included"
        )
        staging = settings.get("staging_parent", Path(tempfile.gettempdir()).resolve())
        volumes = {}
        for path, required in (
            (staging, estimate * 2),
            (lexical_path(destination), estimate * (5 if settings["encrypted"] else 3)),
        ):
            device, ancestor = _volume(path)
            prior, _ = volumes.get(device, (0, ancestor))
            volumes[device] = prior + required, ancestor
        capacity = []
        for required, ancestor in volumes.values():
            free = shutil.disk_usage(ancestor).free
            capacity.append(
                MappingProxyType(
                    {
                        "path": str(ancestor),
                        "required_bytes": required + _MARGIN,
                        "available_bytes": free,
                        "sufficient": free >= required + _MARGIN,
                    }
                )
            )
        return MappingProxyType(
            {
                "inventory": inventory,
                "estimated_source_bytes": estimate,
                "capacity": tuple(capacity),
                "credential_mode": settings["credential_mode"],
                "credential_coverage": "checked_during_capture",
                "maintenance": "Writers pause while data is copied and resume before archive packaging.",
            }
        )

    def start_backup(
        self, config_paths, approved_scope, destination, *, options, password,
        include_known_profiles=False,
    ):
        """Capture coherently, resume writers, then verify and publish an archive."""
        from .capture import _capture_options
        from .profile_paths import lexical_path

        settings, _, _, _ = _capture_options(options)
        if settings["credential_mode"] == "rollback":
            raise ValueError("invalid_backup_credential_mode")
        if settings["encrypted"] != (password is not None) or password == b"":
            raise ValueError("encryption_password_required")
        config_paths, destination = tuple(config_paths), lexical_path(destination)

        def backup(operation, cancel):
            from . import archive_writer, capture_service

            self._update(operation, phase="capturing")
            captured = capture_service.capture(
                config_paths,
                approved_scope,
                destination,
                options=settings,
                cancel=cancel,
                include_known_profiles=include_known_profiles,
            )
            info = captured.root.stat(follow_symlinks=False)
            identity = info.st_dev, info.st_ino
            try:
                self._update(operation, phase="packaging")
                archive = archive_writer.write_archive(
                    captured, destination, password=password, cancel=cancel
                )
                self._update(
                    operation,
                    phase="archive_verified",
                    result={
                        "path": str(archive.path),
                        "sha256": archive.digest,
                        "archive_verified": True,
                        "complete": captured.inventory.complete,
                    },
                )
            finally:
                try:
                    self._discard_workspace(captured.root, identity)
                except OSError:
                    # A published archive remains visible if temporary cleanup fails.
                    self._update(
                        operation, issues=("private_staging_cleanup_required",)
                    )

        return self._start("backup", backup)

    @staticmethod
    def _discard_workspace(work, identity):
        with pinned_directory(work) as parent:
            info = os.fstat(parent)
            if (info.st_dev, info.st_ino) != identity:
                raise OSError("inspection_workspace_changed")
        shutil.rmtree(work)

    def current(self):
        """Return the latest operation snapshot when a view is reopened."""
        with self._lock:
            return next(reversed(self._states.values()), None)

    def status(self, operation_id):
        """An immutable snapshot remains valid after subsequent worker updates."""
        with self._lock:
            if operation_id in self._states:
                return self._states[operation_id]
        from .recovery_copies import _journal

        journal = _journal(self.control_root, operation_id)
        with journal._locked(exclusive=False) as parent:
            records = journal._records(parent)
        if not records:
            raise ValueError("recovery_evidence_missing")
        phase = records[-1].event
        pending = any(
            row["operation_id"] == operation_id for row in self.pending_operations()
        )
        terminal = (
            phase in {"committed", "rolled_back", "prepublication_aborted"}
            and not pending
        )
        events = {row.event for row in records}
        mode = next(
            (row.evidence["mode"] for row in records if row.event == "prepared"), None
        )
        if mode is None:
            from .plan_records import load_plan

            mode = load_plan(journal).mode
        actions = ()
        if pending and mode == "isolated":
            actions = ("finish",)
        elif (
            pending
            and mode == "replace"
            and events <= {"candidate_staged", "prepared", "prepublication_aborted"}
        ):
            # The executor re-proves untouched originals under native maintenance
            # before any terminal record or own-fence cleanup can occur.
            actions = ("abort",)
        elif pending and mode == "replace" and "rollback_verified" in events:
            actions = (
                ("finish",)
                if phase == "committed"
                else (
                    ("rollback",)
                    if {"rollback_started", "rollback_credentials_planned"} & events
                    else ("finish", "rollback")
                )
            )
        return MappingProxyType(
            {
                "operation_id": operation_id,
                "kind": "recovery",
                "mode": mode,
                "state": "succeeded" if terminal else "recovery_required",
                "phase": phase,
                "cancellation_requested": False,
                "actions": actions,
                "result": MappingProxyType(
                    {
                        "journal_operation_id": operation_id,
                        "restoration_validated": terminal and phase == "committed",
                        "rolled_back": terminal and phase == "rolled_back",
                        "aborted": terminal and phase == "prepublication_aborted",
                    }
                ),
                "issues": () if terminal else ("recovery_required",),
            }
        )

    def wait(self, operation_id, *, timeout=None):
        """For worker/CLI callers; views normally observe status without waiting."""
        with self._lock:
            future = self._futures[operation_id]
        future.result(timeout=timeout)
        return self.status(operation_id)

    def cancel(self, operation_id):
        """Request cancellation; native work owns when it is safe to stop."""
        with self._lock:
            if not self._futures[operation_id].done():
                self._cancellations[operation_id].set()
                self._update(operation_id, cancellation_requested=True)

    def close(self):
        """App shutdown waits for native work before discarding private inspection."""
        with self._lock:
            self._closed = True
            for operation, future in self._futures.items():
                if not future.done():
                    self.cancel(operation)
        self._executor.shutdown(wait=True)
        with self._lock:
            for operation, (work, identity) in self._workspaces.items():
                if operation not in self._retained_workspaces:
                    self._discard_workspace(work, identity)
            self._workspaces.clear()
            self._archives.clear()
