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
        payload = stream.read(_LIMIT + 1)
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
    from .inventory import read_profile_config
    from .owner_registry import registered
    from .profile_paths import user_data_dir

    selector = (
        scope.captured.preview.source_selector
        if purpose == "legacy"
        else scope.captured.preview.target_selector
    )
    config = read_profile_config(selector)
    context = DiscoveryContext(
        selector, hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    )
    config[DISCOVERY_CONTEXT_KEY] = context
    adapters = [row for row in registered() if row.owner_id == owner]
    if len(adapters) != 1:
        raise ValueError("mobile_authority_owner_unavailable")
    declarations = adapters[0].discover(config)
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
            provenance = tuple(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                for row in witnesses
            )
            path = native._root(path, witnesses) / path.name
    selected = [row for row in declarations if row.path == path]
    if len(selected) != 1 or selected[0].status != "included":
        raise ValueError("mobile_authority_source_unavailable")
    return path, provenance


def _member(scope, partition, document, owner, path):
    inventory = scope.session._discover_capture_inventory(
        partition.selectors,
        DiscoverySelections(),
        partition.scope_digest,
        _mobile_partition=partition,
    )
    with scope.session._discovery_reads(private_sqlite=True):
        projected = partition.project(scope.session, inventory)
    declared = sorted(
        (row for row in projected.items if row.owner == owner and row.path == path),
        key=lambda row: row.logical_id,
    )
    if not declared:
        raise ValueError("mobile_authority_member_missing")
    producers = {row.logical_id: row for row in document.producer_inventory}
    payloads = {row.logical_id: row for row in document.files}
    versions = {row.owner_id: row.schema_version for row in document.owners}
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
    archive_reader.verify_sealed(sealed, scope.cancel, _custody=scope.custody)
    with archive_reader._regular(sealed.path, _custody=scope.custody) as stream:
        central = archive_reader._central_preflight(stream, ArchiveLimits())
        with zipfile.ZipFile(stream) as archive:
            offsets = archive_reader._local_ranges(stream, archive.infolist(), central)
            info = archive.getinfo(member.payload)
            data = bytearray()
            for chunk in archive_reader._member_chunks(
                stream, info, offsets[info.filename], scope.cancel
            ):
                data.extend(chunk)
                if len(data) > member.size or len(data) > _LIMIT:
                    raise ValueError("mobile_authority_payload_changed")
    if len(data) != member.size or hashlib.sha256(data).hexdigest() != member.sha256:
        raise ValueError("mobile_authority_payload_changed")
    archive_reader.verify_sealed(sealed, scope.cancel, _custody=scope.custody)
    return bytes(data)


def _validate_reference(kind, reference, data):
    from tldw_chatbook.MCP import permission_store as permissions

    payload = json.loads(
        data,
        object_pairs_hook=permissions._reject_duplicate_keys,
        parse_constant=permissions._reject_json_constant,
    )
    if kind == "permission_profile":
        if not permissions._validate_strict_payload(payload):
            raise _ReferenceInvalid("mobile_authority_policy_invalid")
        if reference not in payload["profiles"]:
            raise _ReferenceInvalid("mobile_authority_reference_missing")
        if permissions._lifecycle_resolution_block(payload, reference) is not None:
            raise _ReferenceInvalid("mobile_authority_lifecycle_blocked")
        # Use the installed chain, including default; the complete bytes prove equality.
        for profile in permissions._profile_chain(payload, reference):
            if permissions.profile_lifecycle_disposition(
                profile
            ) == "imported" and profile["tool_pack_lifecycle"][
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
    matches = [
        row for row in records if type(row) is dict and str(row.get("id")) == reference
    ]
    if len(matches) != 1:
        raise _ReferenceInvalid("mobile_authority_reference_missing")
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
        _validate_reference(kind, reference, data)
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
    provenance = (
        source_identity,
        target_identity,
        source_provenance,
        target_provenance,
    )
    return _AuthorityReference(
        kind,
        reference,
        owner,
        member.logical_id,
        member.sha256,
        hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
    )
