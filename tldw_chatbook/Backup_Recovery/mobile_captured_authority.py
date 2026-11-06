"""Fixed captured reference proof; raw policies never become imported authority."""

import hashlib
import json
import stat
import zipfile
from dataclasses import dataclass

from tldw_chatbook.Utils.platform_files import os

from . import archive_reader, native_files
from .limits import ArchiveLimits
from .models import DISCOVERY_CONTEXT_KEY, DiscoveryContext, DiscoverySelections

_LIMIT = 16 * 1024**2
_OWNERS = {"persona": "personas", "permission_profile": "mcp.permissions"}


class _ReferenceInvalid(ValueError):
    """Body-free failures issued by the fixed native semantic boundary."""


@dataclass(frozen=True)
class _AuthorityReference:
    kind: str
    reference_id: str
    owner_id: str
    logical_id: str
    member_digest: str
    provenance_digest: str


def _checked_bytes(scope, path):
    """Retain actual stream/parent custody through exact native name checks."""
    with (
        native_files.pinned_directory(path.parent, _custody=scope.custody) as parent,
        archive_reader._regular(path, _custody=scope.custody) as stream,
    ):
        info = os.fstat(stream.fileno())
        before = archive_reader._identity(info)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
            or info.st_size > _LIMIT
        ):
            raise ValueError("mobile_authority_source_unsafe")
        scope.read_budget.reserve(64 + info.st_size + 1)
        payload = stream.read(info.st_size + 1)
        if len(payload) != info.st_size or before != archive_reader._identity(
            os.fstat(stream.fileno())
        ):
            raise ValueError("mobile_authority_source_changed")
        named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if before != archive_reader._identity(named):
            raise ValueError("mobile_authority_source_changed")
    with native_files.pinned_directory(path.parent, _custody=scope.custody) as parent:
        named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if before != archive_reader._identity(named):
            raise ValueError("mobile_authority_source_changed")
    scope.check()
    return before, payload


def _selected_source(scope, purpose, owner):
    from .inventory import _parse_profile_config
    from .mobile_captured_sqlite import _captured_discovery_budget, _CapturedSQLite
    from .owner_registry import registered
    from .profile_paths import user_data_dir
    from .storage_admission import _local

    if (
        type(scope) is not _CapturedSQLite
        or getattr(_local, "mobile_sqlite_scope", None) is not scope
    ):
        raise ValueError("mobile_sqlite_inactive")
    child = getattr(_local, "discovery_scope", None)
    if child is None:
        raise ValueError("mobile_native_discovery_child_changed")
    metadata = _captured_discovery_budget(scope.session, child)
    selector = (
        scope.captured.preview.source_selector
        if purpose == "legacy"
        else scope.captured.preview.target_selector
    )
    _, selector_bytes = _checked_bytes(scope, selector)
    config = _parse_profile_config(selector_bytes, _read_budget=scope.read_budget)
    from .mobile_preview_budget import path_cost

    scope.read_budget.reserve(path_cost(selector, copies=2) + 512)
    context = DiscoveryContext(
        selector, hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    )
    config[DISCOVERY_CONTEXT_KEY] = context
    scope.read_budget.container(len(registered()))
    adapters = [row for row in registered() if row.owner_id == owner]
    if len(adapters) != 1:
        raise ValueError("mobile_authority_owner_unavailable")
    if _captured_discovery_budget(scope.session, child) is not metadata:
        raise ValueError("mobile_preview_budget_changed")
    declarations = adapters[0].discover(config)
    if _captured_discovery_budget(scope.session, child) is not metadata:
        raise ValueError("mobile_preview_budget_changed")
    path = user_data_dir(config) / (
        "tldw_chatbook_personas.json" if owner == "personas" else "mcp_permissions.json"
    )
    provenance = ()
    if owner == "mcp.permissions":
        from tldw_chatbook.MCP import recovery_activation as native

        # These unchanged native control readers keep their own inherited custody.
        witnesses = native._inventory_witnesses(context, path)
        if witnesses:
            _, config_bytes = _checked_bytes(scope, selector)
            from .mobile_runs_selector import _charge_toml

            _charge_toml(config_bytes, scope.read_budget)
            workspace = native._workspace(config_bytes)
            for witness in witnesses:
                native._check(
                    native._record(owner, path, witness),
                    owner,
                    path,
                    witnesses,
                    witness,
                    workspace,
                )
                activation = native.ActivationStore(native.Path(witness["store_root"]))
                if any(
                    not activation.allowed(witness["generation"], name)
                    for name in native._OWNERS.intersection(witness["owners"])
                ):
                    raise ValueError("mobile_authority_generation_unreviewed")
            scope.read_budget.value(witnesses, text_copies=8)
            provenance = tuple(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in witnesses
            )
            path = native._root(path, witnesses) / path.name
    scope.read_budget.container(len(declarations))
    selected = [row for row in declarations if row.path == path]
    if len(selected) != 1 or selected[0].status != "included":
        raise ValueError("mobile_authority_source_unavailable")
    if _captured_discovery_budget(scope.session, child) is not metadata:
        raise ValueError("mobile_preview_budget_changed")
    return path, provenance


def _member(scope, partition, document, owner, path):
    from .mobile_captured_sqlite import _captured_discovery_budget, _CapturedSQLite
    from .storage_admission import _local

    if (
        type(scope) is not _CapturedSQLite
        or getattr(_local, "mobile_sqlite_scope", None) is not scope
    ):
        raise ValueError("mobile_sqlite_inactive")
    with scope.session._discovery_reads(private_sqlite=True):
        child = getattr(_local, "discovery_scope", None)
        if child is None:
            raise ValueError("mobile_native_discovery_child_changed")
        metadata = _captured_discovery_budget(scope.session, child)
        inventory = scope.session._discover_capture_inventory(
            partition.selectors,
            DiscoverySelections(),
            partition.scope_digest,
            _mobile_partition=partition,
        )
        # Fresh projection belongs to the actual held-discovery coordinator.
        if _captured_discovery_budget(scope.session, child) is not metadata:
            raise ValueError("mobile_preview_budget_changed")
        if metadata is not None:
            metadata.reserve(2048 + 2048 * len(inventory.items))
        projected = partition.project(scope.session, inventory)
        if _captured_discovery_budget(scope.session, child) is not metadata:
            raise ValueError("mobile_preview_budget_changed")
    scope.read_budget.container(3 * len(projected.items))
    declared = sorted(
        (row for row in projected.items if row.owner == owner and row.path == path),
        key=lambda row: row.logical_id,
    )
    if not declared:
        raise ValueError("mobile_authority_member_missing")
    scope.read_budget.container(
        len(document.producer_inventory) + len(document.files) + len(document.owners)
    )
    producers = {row.logical_id: row for row in document.producer_inventory}
    payloads = {row.logical_id: row for row in document.files}
    versions = {row.owner_id: row.schema_version for row in document.owners}
    scope.read_budget.container(3 * len(declared))
    selected = []
    for row in declared:
        producer, payload = producers.get(row.logical_id), payloads.get(row.logical_id)
        if (
            producer is None
            or payload is None
            or row.status != "included"
            or (
                producer.owner_id,
                producer.status,
                producer.dependencies,
                producer.shared_group,
            )
            != (row.owner, row.status, row.dependencies, row.shared_group)
            or payload.owner_id != owner
            or versions.get(owner) != 1
            or payload.size > _LIMIT
        ):
            raise ValueError("mobile_authority_member_changed")
        selected.append(payload)
    if len({(row.size, row.sha256) for row in selected}) != 1:
        raise ValueError("mobile_authority_alias_changed")
    if len(declared) > 1 and (
        not declared[0].shared_group or len({row.shared_group for row in declared}) != 1
    ):
        raise ValueError("mobile_authority_alias_unproven")
    return selected[0]


def _payload(scope, sealed, member):
    from .archive_models import ArchiveManifest, SealedArchive
    from .mobile_captured_sqlite import _CapturedSQLite
    from .mobile_consolidation_apply import _CapturedMobile, _MobileCapturePartition
    from .storage_admission import _local

    if (
        type(scope) is not _CapturedSQLite
        or not scope.active
        or scope not in scope.session._scopes
        or getattr(_local, "mobile_sqlite_scope", None) is not scope
    ):
        raise ValueError("mobile_sqlite_inactive")
    _CapturedMobile._check_binding(scope.captured)
    capture = scope._canonical_archive
    if type(capture) is not tuple or len(capture) != 3:
        raise ValueError("mobile_authority_capture_changed")
    partition, document, original = capture
    source = scope.captured.canonical
    if (
        type(partition) is not _MobileCapturePartition
        or type(document) is not ArchiveManifest
        or type(sealed) is not SealedArchive
        or sealed is not original
        or not any(row is member for row in document.files)
    ):
        raise ValueError("mobile_authority_capture_changed")

    def verify():
        partition._check_binding()
        encrypted = sealed.encrypted_source
        if (
            scope._canonical_archive is not capture
            or scope.captured.canonical is not source
            or partition.session is not scope.session
            or partition.claim is not scope.captured.claim
            or partition.journal is not scope.journal
            or partition.purpose != "canonical"
            or partition.review_only
            or source is None
            or partition.destination != source.path
            or sealed.manifest_bytes != source.manifest_bytes
            or encrypted is None
            or encrypted.digest != source.digest
            or encrypted.plaintext_digest != sealed.digest
        ):
            raise ValueError("mobile_authority_capture_changed")
        scope.read_budget.reserve(1024)  # Local proof references and digest text.
        if (
            encrypted.manifest_digest
            != hashlib.sha256(sealed.manifest_bytes).hexdigest()
        ):
            raise ValueError("mobile_authority_capture_changed")
        # Same fresh whole-seal passes; retain native immutable manifest data.
        for _ in range(2):
            if (
                archive_reader._hash(
                    sealed.path,
                    scope.cancel,
                    _custody=scope.custody,
                    _read_budget=scope.read_budget,
                )
                != sealed.digest
            ):
                raise ValueError("sealed_changed")

    verify()
    with archive_reader._regular(sealed.path, _custody=scope.custody) as stream:
        central = archive_reader._central_preflight(
            stream, ArchiveLimits(), _read_budget=scope.read_budget
        )
        with zipfile.ZipFile(stream) as archive:
            offsets = archive_reader._local_ranges(stream, archive.infolist(), central)
            info = archive.getinfo(member.payload)
            if info.file_size != member.size:
                raise ValueError("mobile_authority_payload_changed")
            # Bounded chunk workspace, bytearray growth and the final bytes clone.
            scope.read_budget.reserve(131072 + 3 * member.size)
            data = bytearray()
            for chunk in archive_reader._member_chunks(
                stream, info, offsets[info.filename], scope.cancel
            ):
                if len(data) + len(chunk) > min(member.size, _LIMIT):
                    raise ValueError("mobile_authority_payload_changed")
                data.extend(chunk)
    if len(data) != member.size or hashlib.sha256(data).hexdigest() != member.sha256:
        raise ValueError("mobile_authority_payload_changed")
    verify()
    return bytes(data)


def _validate_reference(kind, reference, data, *, _read_budget):
    from tldw_chatbook.MCP import permission_store as permissions

    _read_budget.json(data)
    payload = json.loads(
        data,
        object_pairs_hook=permissions._reject_duplicate_keys,
        parse_constant=permissions._reject_json_constant,
    )
    if kind == "permission_profile":
        _read_budget.value(payload)  # Native strict-validation scratch sets.
        if not permissions._validate_strict_payload(payload):
            raise _ReferenceInvalid("mobile_authority_policy_invalid")
        if reference not in payload["profiles"]:
            raise _ReferenceInvalid("mobile_authority_reference_missing")
        _read_budget.container(8)  # Named/default chain and lifecycle references.
        if permissions._lifecycle_resolution_block(payload, reference) is not None:
            raise _ReferenceInvalid("mobile_authority_lifecycle_blocked")
        # Use the installed chain, including default; the complete bytes prove equality.
        for profile in permissions._profile_chain(payload, reference):
            if permissions.profile_lifecycle_disposition(profile) == "imported":
                _read_budget.value(profile, text_copies=8)
                if profile["tool_pack_lifecycle"][
                    "policy_digest"
                ] != permissions.profile_policy_digest(profile):
                    raise _ReferenceInvalid("mobile_authority_provenance_invalid")
        return
    from tldw_chatbook.Character_Chat.local_character_persona_service import (
        LocalCharacterPersonaService,
    )

    records = (
        payload.get("profiles", payload.get("items", []))
        if isinstance(payload, dict)
        else payload
    )
    if type(records) is not list:
        raise _ReferenceInvalid("mobile_authority_policy_invalid")
    _read_budget.value(records, text_copies=2)  # Native ID conversion/match list.
    matches = [
        row for row in records if type(row) is dict and str(row.get("id")) == reference
    ]
    if len(matches) != 1:
        raise _ReferenceInvalid("mobile_authority_reference_missing")
    # Persona normalization creates its dict and policy-rule model/dump copies.
    _read_budget.value(matches[0], text_copies=4)
    _read_budget.value(matches[0], text_copies=2)
    record = LocalCharacterPersonaService._persona_profile_view(matches[0])
    if (
        record.get("id") != reference
        or record.get("deleted")
        or not record.get("is_active", True)
        or record.get("backend", "local") != "local"
    ):
        raise _ReferenceInvalid("mobile_authority_persona_unavailable")


def _authority_reference(scope, kind, reference):
    from .mobile_captured_sqlite import _CapturedSQLite

    if type(scope) is not _CapturedSQLite:
        raise ValueError("mobile_authority_native_snapshot_required")
    scope.check()
    if (
        kind not in _OWNERS
        or type(reference) is not str
        or not 1 <= len(reference) <= 1024
    ):
        raise ValueError("mobile_authority_reference_invalid")
    if scope._canonical_archive is None:
        raise ValueError("mobile_authority_capture_missing")
    partition, document, sealed = scope._canonical_archive
    partition.check()
    owner = _OWNERS[kind]
    with scope.session._discovery_reads(private_sqlite=True):
        source, source_provenance = _selected_source(scope, "legacy", owner)
        target, target_provenance = _selected_source(scope, "canonical", owner)
    member = _member(scope, partition, document, owner, target)
    source_identity, source_bytes = _checked_bytes(scope, source)
    target_identity, target_bytes = _checked_bytes(scope, target)
    if (
        source_identity[:2] != target_identity[:2]
        or source_provenance != target_provenance
        or hashlib.sha256(source_bytes).hexdigest() != member.sha256
        or hashlib.sha256(target_bytes).hexdigest() != member.sha256
    ):
        raise ValueError("mobile_authority_source_coverage_unproven")
    data = _payload(scope, sealed, member)
    try:
        _validate_reference(kind, reference, data, _read_budget=scope.read_budget)
    except _ReferenceInvalid:
        raise
    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        UnicodeError,
        RecursionError,
    ):
        raise ValueError("mobile_authority_policy_invalid") from None
    scope.check()
    # Requalify after member/semantic reads; no cached owner or activation verdict.
    with scope.session._discovery_reads(private_sqlite=True):
        if (source, source_provenance) != _selected_source(scope, "legacy", owner) or (
            target,
            target_provenance,
        ) != _selected_source(scope, "canonical", owner):
            raise ValueError("mobile_authority_source_changed")
    if _checked_bytes(scope, source) != (
        source_identity,
        source_bytes,
    ) or _checked_bytes(scope, target) != (target_identity, target_bytes):
        raise ValueError("mobile_authority_source_changed")
    scope.read_budget.container(4)
    provenance = (
        source_identity,
        target_identity,
        source_provenance,
        target_provenance,
    )
    scope.read_budget.value(provenance, text_copies=8)
    scope.read_budget.reserve(1024)
    return _AuthorityReference(
        kind,
        reference,
        owner,
        member.logical_id,
        member.sha256,
        hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
    )
