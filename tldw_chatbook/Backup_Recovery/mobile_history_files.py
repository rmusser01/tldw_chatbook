"""Private imported-history custody; file evidence is never a DB receipt."""

import hashlib
import stat
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from . import archive_reader, native_files
from .archive_models import ArchiveManifest
from .journal import (
    _evidence_digest,
    _mobile_active_owner,
    observe_artifact,
)
from .mobile_consolidation_apply import _CapturedMobile, _read_mobile_archive
from .profile_paths import user_data_dir
from .storage_admission import (
    MaintenanceSession,
    _CaptureFileDescriptors,
    _contains_capture_path,
    _local,
    _mobile_witness,
)


@dataclass(frozen=True)
class _HistoryMember:
    """Captured identity and byte association for a later real owner receipt."""

    logical_id: str
    root_id: str
    original_root: str
    relative_path: str
    location: str
    size: int
    sha256: str | None


def _root_component(root_id):
    return hashlib.sha256(root_id.encode()).hexdigest()


def _members(captured, document):
    from .inventory import read_profile_config
    from .mobile_consolidation import mobile_inventory
    from .models import DISCOVERY_CONTEXT_KEY, DiscoveryContext

    selector = captured.preview.source_selector
    config = read_profile_config(selector)
    config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(
        selector, hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    )
    with captured.session._discovery_reads(private_sqlite=True):
        declarations = {
            row.logical_id: row
            for row in mobile_inventory(config)
            if row.owner == "agents.history"
            and row.status in {"included", "included_directory"}
        }
    producers = {
        row.logical_id
        for row in document.producer_inventory
        if row.owner_id == "agents.history"
        and row.status in {"included", "included_directory"}
    }
    records = [
        row
        for row in (*document.directories, *document.files)
        if row.logical_id in producers
    ]
    if producers != set(declarations) or producers != {
        row.logical_id for row in records
    }:
        raise ValueError("mobile_history_inventory_changed")
    result = []
    for row in records:
        declared = declarations[row.logical_id]
        root = declarations.get(row.root_id)
        if (
            root is None
            or root.path is None
            or declared.path is None
            or declared.metadata is None
            or declared.metadata.root_id != row.root_id
            or declared.metadata.relative_path != row.relative_path
            or declared.path != root.path / row.relative_path
        ):
            raise ValueError("mobile_history_root_ambiguous")
        relative = Path(_root_component(row.root_id)) / row.relative_path
        result.append(
            _HistoryMember(
                row.logical_id,
                row.root_id,
                str(root.path),
                row.relative_path,
                relative.as_posix(),
                getattr(row, "size", 0),
                getattr(row, "sha256", None),
            )
        )
    return tuple(sorted(result, key=lambda row: row.location))


class _HistoryFiles:
    """An exact live session capability, never reconstructible from copied fields."""

    def __init__(self):
        raise TypeError("mobile_history_is_native_issued")

    def check(self):
        if (
            type(self) is not _HistoryFiles
            or not self.active
            or self not in self.session._scopes
            or getattr(_local, "mobile_history_scope", None) is not self
        ):
            raise ValueError("mobile_history_inactive")
        if self.binding != (
            self.session,
            self.captured,
            self.journal,
            self.data_root,
            self.location,
            self.archive,
            self.members,
        ):
            raise ValueError("mobile_history_binding_changed")
        self.session._check()
        self.captured.check()
        if (
            self.captured not in self.session._scopes
            or self.session._mobile_journal is not self.journal
        ):
            raise ValueError("mobile_history_inactive")
        if any(resource.close_failed for resource in self.resources):
            raise ValueError("mobile_history_retirement_uncertain")

    def retire(self):
        try:
            for resource in tuple(self.resources):
                resource.retire()
        finally:
            self.active = False

    def _records(self):
        self.check()
        with self.journal._locked(exclusive=False) as fd:
            return self.journal._records(fd)

    def _append(self, event, evidence):
        self.check()
        with self.journal._locked(exclusive=True) as fd:
            self.journal._append(fd, event, evidence)

    def _intent(self):
        records = self._records()
        existing = [
            row.evidence for row in records if row.event == "mobile_history_intended"
        ]
        identity = self.captured.preview.destination.identity
        fingerprint = self.captured.preview.source_fingerprint
        digest = _evidence_digest({"members": [row.__dict__ for row in self.members]})
        if existing:
            intent = existing[0]
            if (
                len(existing) != 1
                or intent["location"] != str(self.location)
                or intent["source_fingerprint"] != fingerprint
                or intent["destination_identity"] != identity
                or intent["manifest_digest"]
                != hashlib.sha256(self.archive.manifest_bytes).hexdigest()
                or intent["members_digest"] != digest
            ):
                raise ValueError("mobile_history_review_changed")
            return intent
        publication = self.location
        while publication.parent != self.data_root and not publication.parent.exists():
            publication = publication.parent
        if publication.exists() or publication.is_symlink():
            raise ValueError("mobile_history_destination_occupied")
        parent = _mobile_witness(publication.parent)
        if parent["inode"] is None or not stat.S_ISDIR(parent["mode"]):
            raise ValueError("mobile_history_parent_missing")
        intent = {
            "location": str(self.location),
            "publication": str(publication),
            "temporary": str(
                publication.parent
                / (
                    ".mobile-history-"
                    + hashlib.sha256(self.journal.operation_id.encode()).hexdigest()
                )
            ),
            "parent": parent,
            "source_fingerprint": fingerprint,
            "destination_identity": identity,
            "manifest_digest": hashlib.sha256(self.archive.manifest_bytes).hexdigest(),
            "members_digest": digest,
        }
        self._append("mobile_history_intended", intent)
        return intent

    def _parent(self, intent):
        self.check()
        if _mobile_witness(Path(intent["parent"]["path"])) != intent["parent"]:
            raise ValueError("mobile_history_parent_changed")

    def _observe(self, path):
        custody = _CaptureFileDescriptors(self)
        try:
            return observe_artifact(path, _custody=custody)
        finally:
            custody.retire()

    def _stage(self, intent):
        custody = _CaptureFileDescriptors(self)
        try:
            self._stage_owned(intent, custody)
        finally:
            custody.retire()

    def _stage_owned(self, intent, staging_custody):
        from .limits import ArchiveLimits
        from .space import held_capacity
        from .staging import _mkdirs

        self._parent(intent)
        temporary = Path(intent["temporary"])
        native_files.create_private_directory(temporary, _custody=staging_custody)
        content = temporary / self.location.relative_to(intent["publication"])
        _mkdirs(content, temporary, _custody=staging_custody)
        for row in self.members:
            if row.sha256 is None:
                _mkdirs(content / row.location, temporary, _custody=staging_custody)
        destinations = {
            row.logical_id: content / row.location
            for row in self.members
            if row.sha256 is not None
        }
        document = ArchiveManifest.model_validate_json(self.archive.manifest_bytes)
        limits = ArchiveLimits()
        capacity = held_capacity(temporary, _custody=staging_custody)
        with archive_reader._regular(
            self.archive.path, _custody=staging_custody
        ) as stream:
            central = archive_reader._central_preflight(stream, limits)
            with zipfile.ZipFile(stream) as container:
                offsets = archive_reader._local_ranges(
                    stream, container.infolist(), central
                )
                for payload in document.files:
                    if payload.logical_id not in destinations:
                        continue
                    self.check()
                    target = destinations[payload.logical_id]
                    _mkdirs(target.parent, temporary, _custody=staging_custody)
                    digest, size = hashlib.sha256(), 0
                    custody = _CaptureFileDescriptors(self)
                    try:
                        with native_files.pinned_directory(
                            target.parent, _custody=custody
                        ) as parent:
                            fd = os.open(
                                target.name,
                                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o600,
                                dir_fd=parent,
                            )
                            custody.fds.append(fd)
                            info = container.getinfo(payload.payload)
                            for chunk in archive_reader._member_chunks(
                                stream, info, offsets[info.filename], self.cancel
                            ):
                                size += len(chunk)
                                if size > payload.size:
                                    raise ValueError("mobile_history_payload_changed")
                                capacity(len(chunk))
                                digest.update(chunk)
                                view = memoryview(chunk)
                                while view:
                                    written = os.write(fd, view)
                                    if written <= 0:
                                        raise OSError("mobile_history_write_failed")
                                    view = view[written:]
                            if (
                                size != payload.size
                                or digest.hexdigest() != payload.sha256
                            ):
                                raise ValueError("mobile_history_payload_changed")
                            native_files.flush_file(fd)
                            native_files.flush_directory(parent)
                    finally:
                        custody.retire()
        archive_reader.verify_sealed(
            self.archive, self.cancel, _custody=staging_custody
        )
        custody = _CaptureFileDescriptors(self)
        try:
            with native_files.pinned_directory(temporary, _custody=custody) as fd:
                native_files._flush_private_tree(
                    fd, os.fstat(fd).st_dev, _custody=custody
                )
            self._parent(intent)
            self._append(
                "mobile_history_staged",
                {
                    "intent_digest": _evidence_digest(intent),
                    "artifact": self._observe(temporary),
                },
            )
        finally:
            custody.retire()

    def read_member(
        self, logical_id: str, *, offset: int = 0, limit: int = 65536
    ) -> bytes:
        """Read one exact verified member inside this live maintenance scope.

        This private custody read grants no runtime history access. The run owner
        must separately qualify committed-receipt and conversation/run admission.
        """
        self.check()
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 0 < limit <= 1024**2
        ):
            raise ValueError("mobile_history_range_invalid")
        member = next(
            (
                row
                for row in self.members
                if row.logical_id == logical_id and row.sha256 is not None
            ),
            None,
        )
        if member is None:
            raise ValueError("mobile_history_member_invalid")
        records = self._records()
        verified = [
            row.evidence for row in records if row.event == "mobile_history_verified"
        ]
        if len(verified) != 1:
            raise ValueError("mobile_history_unverified")
        intent = next(
            row.evidence for row in records if row.event == "mobile_history_intended"
        )
        self._parent(intent)
        if self._observe(Path(intent["publication"])) != verified[0]["artifact"]:
            raise ValueError("mobile_history_object_changed")
        target = self.location / member.location
        custody = _CaptureFileDescriptors(self)
        try:
            with native_files.pinned_directory(
                target.parent, _custody=custody
            ) as parent:
                fd = os.open(
                    target.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=parent,
                )
                custody.fds.append(fd)
                before = os.fstat(fd)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or before.st_size != member.size
                    or before.st_uid != os.geteuid()
                ):
                    raise ValueError("mobile_history_object_changed")
                digest = hashlib.sha256()
                while chunk := os.read(fd, 65536):
                    archive_reader._check(self.cancel)
                    digest.update(chunk)
                if digest.hexdigest() != member.sha256:
                    raise ValueError("mobile_history_object_changed")
                os.lseek(fd, offset, os.SEEK_SET)
                result = os.read(fd, limit)
                if before != os.fstat(fd) or before != os.stat(
                    target.name, dir_fd=parent, follow_symlinks=False
                ):
                    raise ValueError("mobile_history_object_changed")
                self.check()
                return result
        finally:
            custody.retire()

    def publish(self):
        """Publish/recover exact witnessed bytes; return file associations, not receipts."""
        self.check()
        if not self.members:
            return ()
        intent = self._intent()
        rows = self._records()
        if rows[-1].event == "mobile_history_intended":
            # Unknown pre-witness objects are retained: exclusive creation refuses.
            self._stage(intent)
            rows = self._records()
        staged = next(
            row.evidence for row in rows if row.event == "mobile_history_staged"
        )
        temporary, publication = Path(intent["temporary"]), Path(intent["publication"])
        self._parent(intent)
        found = []
        for path in (temporary, publication):
            try:
                observed = self._observe(path)
            except FileNotFoundError:
                continue
            if observed != {**staged["artifact"], "path": str(path)}:
                raise ValueError("mobile_history_object_changed")
            found.append(path)
        if len(found) != 1:
            raise ValueError("mobile_history_custody_ambiguous")
        if rows[-1].event == "mobile_history_staged":
            if found == [temporary]:
                parent = intent["parent"]
                identity = (parent["device"], parent["inode"])
                custody = _CaptureFileDescriptors(self)
                try:
                    native_files.publish_new(
                        temporary,
                        publication,
                        parent_identities=(identity, identity),
                        _custody=custody,
                    )
                finally:
                    custody.retire()
            self._parent(intent)
            observed = self._observe(publication)
            if (
                observed != {**staged["artifact"], "path": str(publication)}
                or temporary.exists()
            ):
                raise ValueError("mobile_history_object_changed")
            self._append(
                "mobile_history_observed",
                {"intent_digest": _evidence_digest(intent), "artifact": observed},
            )
            rows = self._records()
        elif found != [publication]:
            raise ValueError("mobile_history_object_changed")
        if rows[-1].event == "mobile_history_observed":
            self._append("mobile_history_verified", rows[-1].evidence)
        return self.members


@contextmanager
def _history_files(session, captured, *, password, cancel):
    """Authenticate the actual captured owner and derive its sole fixed namespace."""
    from .inventory import read_profile_config

    if type(session) is not MaintenanceSession or type(captured) is not _CapturedMobile:
        raise ValueError("mobile_history_native_required")
    session._check()
    if captured.session is not session or captured not in session._scopes:
        raise ValueError("mobile_history_inactive")
    captured.check()
    if any(
        getattr(_local, name, None) is not None
        for name in (
            "mobile_scope",
            "capture_scope",
            "discovery_scope",
            "mobile_history_scope",
            "mobile_sqlite_scope",
        )
    ):
        raise ValueError("mobile_history_scope_busy")
    with captured.journal._locked(exclusive=False) as fd:
        if _mobile_active_owner(captured.journal._records(fd)) is not None:
            raise ValueError("mobile_history_scope_busy")
    scope = object.__new__(_HistoryFiles)
    scope.session, scope.captured, scope.journal = session, captured, captured.journal
    scope.active, scope.resources, scope.cancel = True, [], cancel
    scope.data_root = user_data_dir(
        read_profile_config(captured.preview.target_selector)
    )
    scope.location = (
        scope.data_root
        / "agent_history"
        / "imports"
        / captured.preview.destination.identity
        / captured.preview.source_fingerprint
    )
    if not _contains_capture_path(session._roots, scope.location):
        raise ValueError("mobile_history_outside_scope")
    session._scopes.append(scope)
    _local.mobile_history_scope = scope
    try:
        if captured.source is None:
            scope.members, scope.archive = (), None
            scope.binding = (
                session,
                captured,
                scope.journal,
                scope.data_root,
                scope.location,
                scope.archive,
                scope.members,
            )
            yield scope
        else:
            with _read_mobile_archive(
                captured.source.path, password, cancel
            ) as archive:
                if archive.manifest_bytes != captured.source.manifest_bytes:
                    raise ValueError("mobile_history_capture_changed")
                scope.archive = archive
                scope.members = _members(
                    captured,
                    ArchiveManifest.model_validate_json(archive.manifest_bytes),
                )
                scope.binding = (
                    session,
                    captured,
                    scope.journal,
                    scope.data_root,
                    scope.location,
                    scope.archive,
                    scope.members,
                )
                scope.check()
                yield scope
    finally:
        try:
            scope.retire()
        finally:
            _local.mobile_history_scope = None
