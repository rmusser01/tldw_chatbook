"""Private native mobile preparation; no importer, completion or cutover authority."""

import stat
from contextlib import contextmanager
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap, control_records
from .admission import Admission, fcntl
from .journal import _mobile_state, _MobileSourceClaim, observe_artifact
from .mobile_consolidation import (
    MobileConsolidationRefused,  # noqa: F401 -- Preserve the existing import/patch compatibility slot.
    _path_witnesses,
    _source_claim_locations,
)
from .native_files import flush_directory
from .recovery_copies import _journal
from .storage_admission import (
    MaintenanceSession,
    _CaptureFileDescriptors,
    _contains_capture_path,
)


class _SourceClaim:
    """One actual retained native lock; copied fields never establish custody."""

    def __init__(self):
        raise TypeError("mobile_source_claim_is_native_issued")

    @staticmethod
    def _issue(session, journal, path, lock_path):
        if type(session) is not MaintenanceSession:
            raise ValueError("native_maintenance_required")
        session._check()
        claim = object.__new__(_SourceClaim)
        claim.session, claim.path, claim.lock_path = session, path, lock_path
        claim.operation_id = journal.operation_id
        claim.original_path = path
        claim.custody = _CaptureFileDescriptors(None)
        session._scopes.append(claim)
        return claim

    def _lock_existing(self, parent):
        self.descriptor = Admission._open(parent, self.lock_path.name, os.O_RDWR)
        self.custody.fds.append(self.descriptor)
        info = os.fstat(self.descriptor)
        self.lock_identity = info.st_dev, info.st_ino
        fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _SourceClaim._check_binding(self)

    def check(self, *, _budget=None, _custody=None):
        with self._checked_records(_budget=_budget, _custody=_custody):
            pass

    @contextmanager
    def _checked_records(self, *, _budget=None, _custody=None):
        """Keep this claim's actual journal locked through one validation use."""
        _SourceClaim._check_binding(self, _budget=_budget, _custody=_custody)
        from .storage_admission import _mobile_admission

        public = _mobile_admission(self.session)[4]
        if _budget is None and public is not None:
            _budget, _custody = public, self.custody
        if getattr(self, "journal", None) is not None:
            with self.journal._locked(exclusive=False, _custody=_custody) as fd:
                records = self.journal._records(fd, _budget=_budget, _custody=_custody)
                if _budget is not None:
                    _budget.value(records[0].evidence, text_copies=4)
                prepared, _ = _mobile_state(records)
                profiles, registry = control_records._mobile_pending(
                    self.session,
                    self.journal,
                    prepared,
                    _budget=_budget,
                    _custody=_custody,
                )
                yield records, profiles, registry
        else:
            yield None

    def _check_binding(self, *, _budget=None, _custody=None):
        _SourceClaim._check_native_binding(self, _budget=_budget, _custody=_custody)
        closure = getattr(self, "_ancestors", None)
        if closure is not None:
            closure.verify()

    def _check_native_binding(self, *, _budget=None, _custody=None):
        if (
            type(self) is not _SourceClaim
            or type(self.session) is not MaintenanceSession
            or self not in self.session._scopes
        ):
            raise ValueError("mobile_source_claim_inactive")
        self.session._check()
        from .storage_admission import _mobile_admission

        public = _mobile_admission(self.session)[4]
        if _budget is None and public is not None:
            _budget, _custody = public, self.custody
        if (
            getattr(self, "journal", None) is not None
            and self.session._mobile_journal is not self.journal
        ):
            raise ValueError("mobile_native_preparation_required")
        if self.descriptor not in self.custody.fds:
            raise ValueError("mobile_source_claim_inactive")
        info = os.fstat(self.descriptor)
        named = self.lock_path.lstat()
        if any(
            not stat.S_ISREG(row.st_mode)
            or stat.S_IMODE(row.st_mode) != 0o600
            or row.st_nlink != 1
            or row.st_uid != os.getuid()
            for row in (info, named)
        ):
            raise ValueError("mobile_source_lock_changed")
        if getattr(self, "receipt", None) is not None:
            if _budget is None:
                receipt = observe_artifact(self.path)
            else:
                from .journal import _ArtifactObservationLimits

                size = self.path.lstat().st_size
                if not 0 <= size <= bootstrap.MAX_RECORD or not stat.S_ISREG(
                    self.path.lstat().st_mode
                ):
                    raise ValueError("mobile_source_claim_changed")
                _budget.reserve(147456 + 256 * len(str(self.path)))
                receipt = observe_artifact(
                    self.path,
                    _custody=_custody,
                    _limits=_ArtifactObservationLimits(1, size, 0, 0, 0, 255),
                )
            if receipt != self.receipt:
                raise ValueError("mobile_source_claim_changed")
        if (info.st_dev, info.st_ino) != self.lock_identity or (
            named.st_dev,
            named.st_ino,
        ) != self.lock_identity:
            raise ValueError("mobile_source_lock_changed")

    def retire(self):
        self.custody.retire()


def _existing_claim(path, source_dir):
    with bootstrap.pinned_directory(path.parent) as parent:
        try:
            record = _MobileSourceClaim.model_validate(
                bootstrap._read(parent, path.name)
            )
        except FileNotFoundError:
            return None
    if record.claim_path != str(path) or record.source_dir != str(source_dir):
        raise ValueError("mobile_source_claim_changed")
    # The established recovery reader refuses missing, linked or nonprivate
    # operation evidence; a claim's control locator is never creation authority.
    journal = _journal(Path(record.control_root), record.operation_id)
    with journal._locked(exclusive=False) as parent:
        records = journal._records(parent)
    if (
        not records
        or records[0].event != "mobile_prepared"
        or not any(
            row.event == "mobile_claim_intended" and row.evidence == record.model_dump()
            for row in records
        )
    ):
        raise ValueError("mobile_source_claim_changed")
    return record


@contextmanager
def _claim_source(session, journal, preview, *, approved_scope=None):
    """Claim one checked source once; retain the actual lock through the caller."""
    import json

    if type(session) is not MaintenanceSession:
        raise ValueError("native_maintenance_required")
    session._check()
    if not preview.selector_provenance or not preview.source_witnesses:
        raise ValueError("mobile_selector_review_required")
    if _path_witnesses((preview.source_dir,))[0][1] != preview.source_witnesses:
        raise ValueError("mobile_source_changed")
    locations = _source_claim_locations(preview.source_dir)
    found = [(lock, path) for lock, path in locations if path.exists() or lock.exists()]
    if len(found) > 1:
        raise ValueError("mobile_source_claim_ambiguous")
    lock_path, path = (
        found[0]
        if found
        else next(pair for pair in locations if pair[0].parent.is_dir())
    )
    _, profiles = bootstrap._records(session._control.parent)
    registry = bootstrap._registry(session._control.parent)
    binding = bootstrap._binding(preview.source_selector, profiles, registry)
    if binding is None or not set(binding["namespaces"]) <= set(session._names):
        raise bootstrap.RecoveryRequired(
            "mobile_close_old_instance_enrollment_required"
        )
    roots = bootstrap.effective_roots(
        binding["roots"], (registry[name] for name in binding["namespaces"])
    )
    if not _contains_capture_path(roots, lock_path.parent):
        raise bootstrap.RecoveryRequired("mobile_source_claim_outside_scope")
    claim = _SourceClaim._issue(session, journal, path, lock_path)
    try:
        with claim.custody.pinned_directory(lock_path.parent) as parent:
            try:
                Admission._create_lock(parent, lock_path.name)
                flush_directory(parent)
            except FileExistsError:
                pass
            claim._lock_existing(parent)
            from .mobile_lineage import _reserve_successor, _selection

            existing, _ = _selection(claim, journal, preview)
            path = claim.path
            if approved_scope is not None:
                from .models import DiscoverySelections

                session.prepare_mobile_consolidation(
                    journal,
                    source_config=preview.source_selector,
                    target_config=preview.target_selector,
                    selections=DiscoverySelections(),
                    approved_scope=approved_scope,
                    source_fingerprint=preview.source_fingerprint,
                    destination_witnesses=preview.destination_witnesses,
                    destination_sidecars=preview.destination_sidecars,
                    reviewed_preview=preview,
                )
            if journal is not getattr(session, "_mobile_journal", None):
                raise ValueError("mobile_native_preparation_required")
            _reserve_successor(claim, journal, preview)
            with (
                claim.custody.pinned_directory(path.parent) as claim_parent,
                journal._locked(exclusive=True) as journal_fd,
            ):
                records = journal._records(journal_fd)
                prepared, _ = _mobile_state(records)
                control_records._mobile_pending(session, journal, prepared)
                expected = _MobileSourceClaim(
                    operation_id=journal.operation_id,
                    control_root=str(journal.root.parent),
                    claim_path=str(path),
                    source_dir=str(preview.source_dir),
                    source_fingerprint=preview.source_fingerprint,
                    source_witnesses=[
                        dict(zip(("path", "device", "inode", "mode"), row, strict=True))
                        for row in preview.source_witnesses
                    ],
                    destination_identity=preview.destination.identity,
                    targets=prepared.targets,
                )
                intents = [
                    row for row in records if row.event == "mobile_claim_intended"
                ]
                if intents:
                    if (
                        len(intents) != 1
                        or intents[0].evidence != expected.model_dump()
                    ):
                        raise ValueError("mobile_source_claim_changed")
                else:
                    journal._append(
                        journal_fd, "mobile_claim_intended", expected.model_dump()
                    )
                if existing is not None and existing != expected:
                    raise ValueError("mobile_source_changed_review_required")
                if existing is None:
                    encoded = json.dumps(
                        expected.model_dump(), sort_keys=True, separators=(",", ":")
                    ).encode()
                    Admission._write_new_record(claim_parent, path.name, encoded)
                    flush_directory(claim_parent)
                records = journal._records(journal_fd)
                observed = observe_artifact(path)
                receipts = [
                    row for row in records if row.event == "mobile_claim_observed"
                ]
                if receipts:
                    if len(receipts) != 1 or receipts[0].evidence != observed:
                        raise ValueError("mobile_source_claim_changed")
                else:
                    journal._flush_records(journal_fd)
                    journal._append(journal_fd, "mobile_claim_observed", observed)
                claim.receipt = observed
                claim.journal = journal
            from .mobile_lineage_ancestors import _issue_ancestors

            _issue_ancestors(claim, journal, preview.source_dir)
            claim.check()
            yield claim
    finally:
        claim.retire()


class _MobileCapturePartition:
    """Installed purpose projection; no caller source paths or owner factories."""

    def __init__(self):
        raise TypeError("mobile_capture_partition_is_native_issued")

    def check(self, *, _budget=None, _custody=None):
        self._check_binding()
        if self.review_only:
            if bootstrap._records(self.session._control.parent)[0]:
                raise bootstrap.RecoveryRequired("recovery_pending")
        else:
            self.claim.check(_budget=_budget, _custody=_custody)

    def _check_binding(self):
        if (
            type(self) is not _MobileCapturePartition
            or type(self.session) is not MaintenanceSession
            or self not in self.session._scopes
            or not self.active
        ):
            raise ValueError("mobile_native_partition_required")
        self.session._check()
        if not self.review_only and self.session._mobile_journal is not self.journal:
            raise ValueError("mobile_native_partition_required")

    def service_output_item(self, inventory, destination):
        """Authenticate only this live intention's exact default work overlap."""
        import hashlib

        from .inventory import _service_control_exclusion
        from .service_storage import (
            default_control_root,
            is_private_service_work,
            verify_default_storage,
            work_root,
        )
        from .storage_admission import _mobile_coordinator_budget

        if type(self) is not _MobileCapturePartition:
            raise ValueError("mobile_native_partition_required")
        if self.review_only:
            return None
        control = default_control_root()
        if self.journal.root.parent != control:
            return None  # Custom captures retain every ordinary overlap check.
        budget = _mobile_coordinator_budget(self.session)
        if budget is None:
            raise ValueError("mobile_native_budget_required")
        custody = self.claim.custody
        self._check_binding()
        budget.reserve(8192 + 64 * len(str(destination)))
        expected = work_root(control) / (
            hashlib.sha256(self.journal.operation_id.encode()).hexdigest()
            + "-"
            + self.purpose
            + ".tldw-backup.zip.age"
        )
        if (
            self.purpose not in {"legacy", "canonical"}
            or destination != expected
            or self.destination != expected
        ):
            raise ValueError("mobile_capture_destination_changed")
        with self.claim._checked_records(_budget=budget, _custody=custody) as (
            records,
            profiles,
            registry,
        ):
            if (
                records[-1].event != "mobile_capture_intended"
                or records[-1].evidence != self.intent
                or self.intent["destination"] != str(destination)
            ):
                raise ValueError("mobile_capture_state_changed")
            items = _service_control_exclusion(_budget=budget, _custody=custody)
            if len(items) != 1 or not is_private_service_work(
                destination, items[0], _budget=budget, _custody=custody
            ):
                raise ValueError("mobile_service_output_unverified")
            item = items[0]
            if (
                sum(candidate == item for candidate in inventory.items) != 1
                or registry is None
            ):
                raise ValueError("mobile_service_output_unverified")
            # These are the current objects returned by native pending validation
            # inside this same held journal scope, never caller-supplied records.
            for rows in (profiles, registry.values()):
                for row in rows:
                    budget.value(row["roots"], text_copies=8)
                    for raw in row["roots"]:
                        path = Path(raw).resolve()
                        if (
                            path == control.parent
                            or path in control.parent.parents
                            or control.parent in path.parents
                        ):
                            raise ValueError("mobile_service_output_profile_overlap")
            identity = verify_default_storage(_budget=budget, _custody=custody)
            previous = getattr(self, "_service_output_identity", None)
            if previous is not None and identity != previous:
                raise ValueError("output_parent_changed")
            self._service_output_identity = identity
            return item

    def retire(self):
        self.active = False

    def project(self, session, inventory):
        import hashlib
        from dataclasses import replace

        from .inventory import classify_entries, read_profile_config
        from .mobile_consolidation import mobile_inventory
        from .models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
        from .owner_registry import registered

        self.check()
        if (
            not self.active
            or session is not self.session
            or inventory.scope_digest != self.scope_digest
        ):
            raise ValueError("mobile_capture_scope_changed")
        selector = (
            self.preview.source_selector
            if self.purpose == "legacy"
            else self.preview.target_selector
        )
        config = read_profile_config(selector)
        context = DiscoveryContext(
            selector, hashlib.sha256(str(selector).encode()).hexdigest()[:24]
        )
        config[DISCOVERY_CONTEXT_KEY] = context
        if self.purpose == "legacy":
            declarations = mobile_inventory(config)
        else:
            declarations = tuple(
                item
                for adapter in registered()
                for item in adapter.discover(dict(config))
            )
        # Match installed declarations against the already admitted inventory.
        # Cohort merging may change a logical ID; owner plus exact path survives it.
        keys = {
            (item.owner, item.path) for item in declarations if item.path is not None
        }
        ids = {item.logical_id for item in declarations if item.path is None}
        selected = {
            item.logical_id
            for item in inventory.items
            if (item.owner, item.path) in keys
            or item.logical_id in ids
            or item.owner == "config"
            and item.path == selector
        }
        by_id = {item.logical_id: item for item in inventory.items}
        pending = list(selected)
        while pending:
            for dependency in by_id[pending.pop()].dependencies:
                if dependency not in selected:
                    if dependency not in by_id:
                        raise ValueError("mobile_capture_dependency_missing")
                    selected.add(dependency)
                    pending.append(dependency)
        # Preserve every original path as output-alias protection and explicitly
        # describe omitted producers. No omitted path receives capture authority.
        projected = classify_entries(
            tuple(
                item
                if item.logical_id in selected
                else replace(item, status="intentionally_excluded")
                for item in inventory.items
            )
        )
        if not projected.complete:
            raise ValueError("mobile_capture_partition_incomplete")
        return projected

    def validate_credentials(self, stage, issues, acknowledged):
        from .capture import CaptureReviewRequired
        from .credentials import _material

        self.check()
        if self.review_only:
            raise _MobileCredentialIssues(tuple(sorted(issues)))
        expected = tuple(self.intent["credential_issues"])
        if (
            tuple(sorted(issues)) != expected
            or acknowledged != expected
            or any(
                not issue.startswith("credential_manual_recovery_required:")
                for issue in issues
            )
        ):
            raise CaptureReviewRequired(issues or ("credential_coverage_changed",))
        records = _material(stage)
        if any(record["status"] != "captured" for record in records):
            raise CaptureReviewRequired(("credential_coverage_changed",))
        captured = {
            record["id"] for record in records if record["status"] == "captured"
        }
        if any(issue.split(":", 1)[1] not in captured for issue in issues):
            raise CaptureReviewRequired(("credential_coverage_changed",))

    def stage_publication(self, temporary, destination, verified, inventory):
        import hashlib

        from .storage_admission import _mobile_witness

        self.check()
        if destination != self.destination:
            raise ValueError("mobile_capture_destination_changed")
        with self.journal._locked(exclusive=True) as fd:
            records = self.journal._records(fd)
            if (
                records[-1].event != "mobile_capture_intended"
                or records[-1].evidence != self.intent
            ):
                raise ValueError("mobile_capture_state_changed")
            self.journal._append(
                fd,
                "mobile_capture_staged",
                {
                    "purpose": self.purpose,
                    "temporary": observe_artifact(temporary),
                    "temporary_parent": _mobile_witness(temporary.parent),
                    "destination_parent": _mobile_witness(destination.parent),
                    "manifest_digest": hashlib.sha256(
                        verified.manifest_bytes
                    ).hexdigest(),
                    "partition_digest": inventory.scope_digest,
                },
            )

    def observe_publication(self, destination):
        self.check()
        with self.journal._locked(exclusive=True) as fd:
            records = self.journal._records(fd)
            staged = records[-1]
            if (
                staged.event != "mobile_capture_staged"
                or destination != self.destination
            ):
                raise ValueError("mobile_capture_state_changed")
            self.journal._append(
                fd,
                "mobile_capture_verified",
                {
                    "purpose": self.purpose,
                    "archive": observe_artifact(destination),
                    "manifest_digest": staged.evidence["manifest_digest"],
                    "partition_digest": staged.evidence["partition_digest"],
                },
            )

    def retire_publication_staging(self, records):
        """Retire only the native writer directory whose inode was witnessed."""
        import re
        import shutil

        from .native_files import pinned_directory
        from .storage_admission import _mobile_witness

        self.check()
        staged = [
            row
            for row in records
            if row.event == "mobile_capture_staged"
            and row.evidence["purpose"] == self.purpose
        ]
        if len(staged) != 1:
            raise ValueError("mobile_capture_state_changed")
        evidence = staged[0].evidence
        expected = evidence["temporary_parent"]
        operation = Path(expected["path"])
        if operation.parent != self.destination.parent or not re.fullmatch(
            r"\.backup-write-[0-9a-f]{32}", operation.name
        ):
            raise ValueError("mobile_capture_parent_changed")
        try:
            operation.lstat()
        except FileNotFoundError:
            return  # The native writer already retired its own staging.
        with (
            pinned_directory(operation.parent),
            pinned_directory(operation) as descriptor,
        ):
            info = os.fstat(descriptor)
            if (
                _mobile_witness(operation) != expected
                or _mobile_witness(operation.parent) != evidence["destination_parent"]
                or (info.st_dev, info.st_ino, info.st_mode)
                != (expected["device"], expected["inode"], expected["mode"])
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise ValueError("mobile_capture_parent_changed")
            shutil.rmtree(operation)

    def recover_publication(self, password, cancel):
        import hashlib

        from . import native_files
        from .storage_admission import _mobile_witness

        self.check()
        with self.journal._locked(exclusive=True) as fd:
            records = self.journal._records(fd)
            staged = records[-1]
            if (
                staged.event != "mobile_capture_staged"
                or staged.evidence["purpose"] != self.purpose
            ):
                raise ValueError("mobile_capture_state_changed")
            evidence = staged.evidence
            temporary = Path(evidence["temporary"]["path"])
            for key in ("temporary_parent", "destination_parent"):
                row = evidence[key]
                if _mobile_witness(Path(row["path"])) != row:
                    raise ValueError("mobile_capture_parent_changed")
            expected = {**evidence["temporary"], "path": str(self.destination)}
            if self.destination.exists():
                if temporary.exists() or observe_artifact(self.destination) != expected:
                    raise ValueError("mobile_capture_publication_ambiguous")
                source = self.destination
            else:
                if observe_artifact(temporary) != evidence["temporary"]:
                    raise ValueError("mobile_capture_publication_ambiguous")
                source = temporary
            # This is the actual authenticated archive reader, not a journal claim.
            with _read_mobile_archive(source, password, cancel) as verified:
                if (
                    hashlib.sha256(verified.manifest_bytes).hexdigest()
                    != evidence["manifest_digest"]
                ):
                    raise ValueError("mobile_capture_changed")
            self.journal._flush_records(fd)
            if source == temporary:
                parents = tuple(
                    (evidence[key]["device"], evidence[key]["inode"])
                    for key in ("temporary_parent", "destination_parent")
                )
                native_files.publish_new(
                    temporary, self.destination, parent_identities=parents
                )
        self.observe_publication(self.destination)


def _verify_capture_archives(records):
    for row in records:
        if row.event == "mobile_capture_verified":
            artifact = row.evidence["archive"]
            if observe_artifact(Path(artifact["path"])) != artifact:
                raise ValueError("mobile_capture_changed")


class _CapturedMobile:
    """Private, session-bound verified archives and native owner access for C2."""

    def __init__(self):
        raise TypeError("mobile_lifecycle_is_native_issued")

    def check(self):
        _CapturedMobile._check_binding(self)
        _SourceClaim._check_binding(self.claim)
        with self.journal._locked(exclusive=False) as fd:
            records = self.journal._records(fd)
        prepared, _ = _mobile_state(records)
        control_records._mobile_pending(self.session, self.journal, prepared)
        _verify_capture_archives(records)

    def _check_binding(self):
        if (
            type(self) is not _CapturedMobile
            or type(self.session) is not MaintenanceSession
            or self not in self.session._scopes
            or self.journal is not getattr(self.session, "_mobile_journal", None)
            or any(
                actual is not expected
                for actual, expected in zip(
                    (
                        self.session,
                        self.journal,
                        self.claim,
                        self.preview,
                        self.source,
                        self.canonical,
                        self.cancel,
                        self.session._scopes,
                    ),
                    self.binding,
                    strict=True,
                )
            )
        ):
            raise ValueError("mobile_native_capture_cohort_changed")
        if not self.active:
            raise ValueError("mobile_lifecycle_retired")
        from .archive_reader import _check

        _check(self.cancel)

    @contextmanager
    def owner(self, owner_id):
        from .mobile_native_operation import _native_preopen

        _native_preopen(self, self.cancel)
        with self.session.mobile_consolidation_owner(self.journal, owner_id) as owner:
            yield owner
        self.check()

    def consolidate(self, *, password, cancel):
        """Prepare this actual operation; unresolved native duties refuse effects."""
        from .mobile_native_operation import _consolidate

        return _consolidate(self, password=password, cancel=cancel)

    def sqlite_reads(self, *, password, cancel):
        """Issue the fixed read-only captured SQLite scope under live maintenance."""
        from .mobile_captured_sqlite import _sqlite_reads

        return _sqlite_reads(self, password=password, cancel=cancel)

    def retire(self):
        self.active = False


def _verified_capture(partition, password, cancel):
    import hashlib

    from . import archive_reader
    from .archive_models import SealedArchive
    from .archive_writer import write_archive
    from .capture import _capture_under_maintenance
    from .models import DiscoverySelections

    partition.check()
    with partition.journal._locked(exclusive=True) as fd:
        records = partition.journal._records(fd)
        intents = [
            row
            for row in records
            if row.event == "mobile_capture_intended"
            and row.evidence["purpose"] == partition.purpose
        ]
        if intents:
            if len(intents) != 1 or intents[0].evidence != partition.intent:
                raise ValueError("mobile_capture_review_changed")
        else:
            partition.journal._append(fd, "mobile_capture_intended", partition.intent)
        records = partition.journal._records(fd)
    if (
        records[-1].event == "mobile_capture_intended"
        and records[-1].evidence["purpose"] == partition.purpose
    ):
        # No publication witness means an existing output is never adopted.
        partition.session._discover_capture_inventory(
            partition.selectors,
            DiscoverySelections(),
            partition.scope_digest,
            _mobile_partition=partition,
        )
        staged = _capture_under_maintenance(
            partition.session,
            partition.selectors,
            partition.scope_digest,
            partition.destination,
            options={
                "credential_mode": "rollback",
                "encrypted": True,
                "acknowledged_credential_issues": tuple(
                    partition.intent["credential_issues"]
                ),
            },
            cancel=cancel,
            _mobile_partition=partition,
        )
        try:
            write_archive(
                staged,
                partition.destination,
                password=password,
                cancel=cancel,
                _mobile_capture=partition,
            )
        finally:
            import shutil

            shutil.rmtree(staged.root)
    elif (
        records[-1].event == "mobile_capture_staged"
        and records[-1].evidence["purpose"] == partition.purpose
    ):
        partition.recover_publication(password, cancel)
    with partition.journal._locked(exclusive=False) as fd:
        records = partition.journal._records(fd)
        receipts = [
            row
            for row in records
            if row.event == "mobile_capture_verified"
            and row.evidence["purpose"] == partition.purpose
        ]
    if (
        len(receipts) != 1
        or observe_artifact(partition.destination) != receipts[0].evidence["archive"]
    ):
        raise ValueError("mobile_capture_changed")
    with _read_mobile_archive(partition.destination, password, cancel) as verified:
        if (
            hashlib.sha256(verified.manifest_bytes).hexdigest()
            != receipts[0].evidence["manifest_digest"]
        ):
            raise ValueError("mobile_capture_changed")
        manifest = verified.manifest_bytes
    partition.check()
    partition.retire_publication_staging(records)
    return SealedArchive(
        partition.destination,
        archive_reader._hash(partition.destination, cancel),
        manifest,
    )


@contextmanager
def _captured_mobile(
    session, journal, preview, approved_scope, *, destination, password, cancel
):
    """Prepare/recover one actual operation; retain exclusion through both archives.

    Destination is only a native archive output locator. The fixed encrypted
    rollback credential policy is complete-or-refuse and never caller selectable.
    C2 must read/verify purpose-specific rows and bytes before importing anything.
    No completion, cutover or recovery-terminal event is exposed here.
    """
    import hashlib

    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    from .profile_paths import lexical_path
    from .replacement import require_rollback_password
    from .storage_admission import _mobile_admission

    session._check()
    if session._mobile_attempted_journal is not None:
        raise ValueError("mobile_native_operation_already_attempted")
    require_rollback_password(password)
    admission = _mobile_admission(session)
    if any(type(scope) is _CapturedMobile for scope in session._scopes):
        raise ValueError("mobile_native_operation_already_attempted")
    if admission[2] is None:
        admission[2] = _TraceReadBudget()
    elif type(admission[2]) is not _TraceReadBudget:
        raise ValueError("mobile_native_budget_required")
    destination = lexical_path(destination)
    with _claim_source(
        session, journal, preview, approved_scope=approved_scope
    ) as claim:
        lifecycle = object.__new__(_CapturedMobile)
        lifecycle.session, lifecycle.journal, lifecycle.claim = session, journal, claim
        lifecycle.active = True
        lifecycle.source = None
        lifecycle.preview = preview
        lifecycle.credential_issues = preview.credential_acknowledgements
        lifecycle.manual_recovery_required = any(
            issues for _, issues in lifecycle.credential_issues
        )
        session._scopes.append(lifecycle)
        try:
            if not preview.has_data:
                with journal._locked(exclusive=True) as fd:
                    if not any(
                        row.event == "mobile_source_empty"
                        for row in journal._records(fd)
                    ):
                        journal._append(
                            fd,
                            "mobile_source_empty",
                            {"source_fingerprint": preview.source_fingerprint},
                        )
            for purpose in (
                ("legacy", "canonical") if preview.has_data else ("canonical",)
            ):
                partition = object.__new__(_MobileCapturePartition)
                partition.session, partition.journal, partition.claim = (
                    session,
                    journal,
                    claim,
                )
                partition.preview, partition.scope_digest = preview, approved_scope
                partition.purpose, partition.active = purpose, True
                partition.review_only = False
                partition.selectors = tuple(
                    dict.fromkeys((preview.source_selector, preview.target_selector))
                )
                name = (
                    hashlib.sha256(journal.operation_id.encode()).hexdigest()
                    + "-"
                    + purpose
                    + ".tldw-backup.zip.age"
                )
                partition.destination = destination / name
                partition.intent = {
                    "purpose": purpose,
                    "destination": str(partition.destination),
                    "credential_issues": list(
                        dict(preview.credential_acknowledgements).get(purpose, ())
                    ),
                    "source_fingerprint": preview.source_fingerprint,
                    "inventory_digest": approved_scope,
                    "destination_identity": preview.destination.identity,
                }
                session._scopes.append(partition)
                archive = _verified_capture(partition, password, cancel)
                if purpose == "legacy":
                    lifecycle.source = archive
                else:
                    lifecycle.canonical = archive
            lifecycle.cancel = cancel
            lifecycle.binding = (
                session,
                journal,
                claim,
                preview,
                lifecycle.source,
                lifecycle.canonical,
                cancel,
                session._scopes,
            )
            lifecycle.check()
            yield lifecycle
        finally:
            lifecycle.retire()


def _review_mobile_operation(session, journal):
    """Reconstruct reviewed locators from typed control; fresh preparation rechecks them."""
    from dataclasses import replace

    from .mobile_consolidation import (
        CanonicalConsoleStores,
        MobileSelectorProvenance,
        _preview_selected_mobile,
    )

    if type(session) is not MaintenanceSession:
        raise ValueError("native_maintenance_required")
    session._check()
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    from .storage_admission import _mobile_admission

    # This new coordinator review belongs to the public aggregate only. Private
    # component recovery retains its original ordinary discovery semantics; its
    # native proof allowance is issued later by the installed captured path.
    budget = _mobile_admission(session)[4]
    if budget is not None and type(budget) is not _TraceReadBudget:
        raise ValueError("mobile_native_budget_required")
    custody = _CaptureFileDescriptors(None)
    session._scopes.append(custody)
    try:
        with journal._locked(exclusive=False, _custody=custody) as fd:
            records = journal._records(fd, _budget=budget, _custody=custody)
    finally:
        custody.retire()
    if any(
        row.event
        in {"mobile_committed", "mobile_rollback_started", "mobile_rolled_back"}
        for row in records
    ):
        raise ValueError("mobile_finish_unavailable")
    prepared, _ = _mobile_state(records)
    if not prepared.selector_review:
        raise ValueError("mobile_selector_review_required")
    control_records._mobile_pending(session, journal, prepared)
    with session._discovery_reads(private_sqlite=True, _read_budget=budget):
        fresh = _preview_selected_mobile(
            Path(prepared.source_selector),
            Path(prepared.target_selector),
            review_destination=False,
            _read_budget=budget,
        )

    def chain(rows):
        return tuple((row.path, row.device, row.inode, row.mode) for row in rows)

    def targets(rows):
        return tuple((row.witnesses[-1].path, chain(row.witnesses)) for row in rows)

    def sidecars(rows):
        return tuple((row.witnesses[-1].path, chain(row.sidecars)) for row in rows)

    provenance = tuple(
        MobileSelectorProvenance(
            Path(row.selector),
            row.fingerprint,
            chain(row.witnesses),
            CanonicalConsoleStores(
                *(Path(target.witnesses[-1].path) for target in row.targets)
            )
            if row.targets
            else None,
            targets(row.targets),
            sidecars(row.targets),
        )
        for row in prepared.selector_review
    )
    claims = [row for row in records if row.event == "mobile_claim_intended"]
    source_witnesses = (
        tuple(
            (row["path"], row["device"], row["inode"], row["mode"])
            for row in claims[0].evidence["source_witnesses"]
        )
        if claims
        else fresh.source_witnesses
    )
    return replace(
        fresh,
        source_fingerprint=prepared.source_fingerprint,
        selector_provenance=provenance,
        source_witnesses=source_witnesses,
        credential_acknowledgements=tuple(
            (row.purpose, tuple(row.issues)) for row in prepared.credential_review
        ),
        destination_witnesses=targets(prepared.targets),
        destination_sidecars=sidecars(prepared.targets),
    )


@contextmanager
def _resume_captured_mobile(session, journal, *, destination, password, cancel):
    """Cold recovery of the same operation under its actual native maintenance."""
    preview = _review_mobile_operation(session, journal)
    from .storage_admission import _mobile_admission

    budget = _mobile_admission(session)[4]
    custody = _CaptureFileDescriptors(None)
    session._scopes.append(custody)
    try:
        with journal._locked(exclusive=False, _custody=custody) as fd:
            prepared, _ = _mobile_state(
                journal._records(fd, _budget=budget, _custody=custody)
            )
    finally:
        custody.retire()
    with _captured_mobile(
        session,
        journal,
        preview,
        prepared.inventory_digest,
        destination=destination,
        password=password,
        cancel=cancel,
    ) as lifecycle:
        yield lifecycle


class _MobileCredentialIssues(Exception):
    """Private completed native credential enumeration; contains no secret bodies."""

    def __init__(self, issues):
        self.issues = issues


def _preview_mobile_credential_issues(session, preview, approved_scope, *, cancel):
    """Read fixed native credential coverage before separate explicit consent.

    Uses disposable native captures, never publishes archives or prepares apply.
    The returned issue codes are observations, not acknowledgements.
    """
    import tempfile
    from uuid import uuid4

    from .capture import _capture_under_maintenance
    from .mobile_consolidation import (
        MobileConsolidationPreview,
        _preview_selected_mobile,
    )
    from .models import DiscoverySelections

    if (
        type(session) is not MaintenanceSession
        or type(preview) is not MobileConsolidationPreview
    ):
        raise ValueError("native_mobile_review_required")
    session._check()
    if not preview.selector_provenance:
        raise ValueError("mobile_selector_review_required")
    root = session._control.parent
    pending, profiles = bootstrap._records(root)
    registry = bootstrap._registry(root)
    if pending:
        raise bootstrap.RecoveryRequired("recovery_pending")
    selectors = tuple(dict.fromkeys((preview.source_selector, preview.target_selector)))
    bindings = [bootstrap._binding(path, profiles, registry) for path in selectors]
    if any(
        row is None or not set(row["namespaces"]) <= set(session._names)
        for row in bindings
    ):
        raise bootstrap.RecoveryRequired(
            "mobile_close_old_instance_enrollment_required"
        )
    from .storage_admission import _mobile_coordinator_budget

    budget = _mobile_coordinator_budget(session)
    with session._discovery_reads(private_sqlite=True, _read_budget=budget):
        current = _preview_selected_mobile(
            preview.source_selector, preview.target_selector, _read_budget=budget
        )
    from dataclasses import replace

    if replace(preview, credential_acknowledgements=()) != current:
        raise ValueError("mobile_selector_review_changed")
    session._discover_capture_inventory(
        selectors, DiscoverySelections(), approved_scope, private_sqlite=True
    )
    result = []
    for purpose in ("legacy", "canonical") if preview.has_data else ("canonical",):
        partition = object.__new__(_MobileCapturePartition)
        partition.session, partition.preview = session, preview
        partition.scope_digest, partition.selectors = approved_scope, selectors
        partition.purpose, partition.active, partition.review_only = purpose, True, True
        session._scopes.append(partition)
        try:
            # No file is ever published at this native-only output locator.
            output = Path(tempfile.gettempdir()).resolve() / (
                "mobile-credential-review-" + uuid4().hex
            )
            try:
                _capture_under_maintenance(
                    session,
                    selectors,
                    approved_scope,
                    output,
                    options={"credential_mode": "rollback", "encrypted": True},
                    cancel=cancel,
                    _mobile_partition=partition,
                )
            except _MobileCredentialIssues as observation:
                result.append((purpose, observation.issues))
            else:
                raise ValueError("mobile_credential_review_unobserved")
        finally:
            partition.retire()
    return tuple(result)


@contextmanager
def _read_mobile_archive(source, password, cancel):
    """Authenticate native bytes and retire their owned decrypted staging."""
    import shutil
    from uuid import uuid4

    from . import archive_reader
    from .limits import ArchiveLimits

    root = source.parent / (".mobile-verify-" + uuid4().hex)
    verified = archive_reader.acquire(source, root, ArchiveLimits(), password, cancel)
    before = root.lstat()
    try:
        yield verified
    finally:
        with bootstrap.pinned_directory(root) as fd:
            actual = os.fstat(fd)
            if (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError("mobile_verification_staging_changed")
            shutil.rmtree(root)
