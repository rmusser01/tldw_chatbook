"""Finite credential-aware candidates for an authenticated local reverse operation."""

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from uuid import uuid4

from . import credentials
from .archive_models import SealedArchive
from .journal import (
    _evidence_digest,
    _matches,
    _RollbackCredentialPlan,
    observe_artifact,
)
from .limits import ArchiveLimits
from .native_files import create_private_directory


def latest_plan(records):
    return next(
        (
            row
            for row in reversed(records)
            if row.event == "rollback_credentials_planned"
        ),
        None,
    )


def current_phase(records):
    plan = latest_plan(records)
    return records[records.index(plan) + 1 :] if plan is not None else records


def _applied_scopes(records):
    """Applied ownership survives amendments only for the exact record/purpose."""
    return {
        (row.evidence["record_id"], row.evidence["purpose"])
        for row in records
        if row.event == "rollback_credential_applied"
    }


@dataclass(frozen=True)
class RollbackMaterial:
    archive: SealedArchive
    root: Path
    session: object
    prepared: object

    def records(self):
        from .publication import _finalization_session

        _finalization_session(self.session, self.prepared.publication, self.prepared)
        limits = ArchiveLimits()
        with self.session._capture_bound_sources(
            (), self.root, limits, limits.expanded_bytes
        ):
            return credentials._material(self.root)

    def check(self, plan_record=None):
        scopes = plan_record.evidence["scopes"] if plan_record else {}
        checked = {}
        for record in self.records():
            if record["status"] != "captured" or record["kind"] == "encrypted_config":
                continue
            if record["id"] in scopes:
                checked[record["id"]] = credentials.verify_rollback_credential(
                    record, scopes[record["id"]]
                )
            else:
                try:
                    value = credentials._read_scope(
                        record, credentials._credential_store()
                    )
                except Exception:  # noqa: BLE001 - backend errors may contain secret values
                    raise ValueError("rollback_credential_scope_changed") from None
                if value != record["value"]:
                    raise ValueError("rollback_credential_scope_changed")
                checked[record["id"]] = credentials._fingerprint(value)
        return _evidence_digest(checked)


def _original_location(item):
    for path in (item.retained, item.target):
        if path and _matches(item.previous_metadata, path, metadata=True):
            return Path(path)
    raise ValueError("rollback_originals_unverified")


def _copy_original(source, destination, cancel):
    """Copy an already proven finite raw artifact; do not migrate its stores."""
    from .publication import _installed_metadata
    from .staging import _copy

    before = source.lstat()
    if stat.S_ISDIR(before.st_mode):
        create_private_directory(destination)
        for name in sorted(os.listdir(source)):
            _copy_original(source / name, destination / name, cancel)
    elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
        _copy(source, destination, cancel)
    else:
        raise ValueError("rollback_originals_unverified")
    info = destination.lstat()
    _installed_metadata(
        destination,
        (info.st_dev, info.st_ino),
        {"mode": stat.S_IMODE(before.st_mode), "mtime_ns": before.st_mtime_ns},
    )


def _reference_mapping(material, prepared, plan, records):
    from .archive_reader import verify_sealed

    document = verify_sealed(material.archive)
    originals = {item.logical_id: item for item in plan.target.items}
    references = []
    for record in records:
        matches = [
            row
            for row in document.files
            if row.payload == record["file"]
            and row.owner_id == "mcp.targets"
            and row.logical_id in originals
        ]
        paths = {originals[row.logical_id].path for row in matches}
        if len(paths) != 1:
            raise ValueError("rollback_credential_mapping_invalid")
        payload = matches[0]
        original = originals[payload.logical_id]
        candidates = [
            item
            for item in prepared.artifacts
            if item.previous is not None
            and (
                Path(item.target) == original.path
                or item.previous.kind == "directory"
                and Path(item.target) in original.path.parents
            )
        ]
        if (
            original.owner != "mcp.targets"
            or len(candidates) != 1
            or original.logical_id in plan.safety_scope
        ):
            raise ValueError("rollback_credential_mapping_invalid")
        item = candidates[0]
        source = _original_location(item)
        leaf = (
            source
            if item.previous.kind == "file"
            else source / original.path.relative_to(item.target)
        )
        from .archive_reader import _hash

        if _hash(leaf, Event()) != payload.sha256:
            raise ValueError("rollback_credential_original_changed")
        references.append(
            {
                "record_id": record["id"],
                "logical_id": original.logical_id,
                "target": str(original.path),
                "artifact_id": item.logical_id,
                "owner": original.owner,
            }
        )
    return references


def prepare_credentials(material, journal, parent, prepared, plan, cancel):
    """Persist a candidate/credential amendment before any reverse value or file effect."""
    from .owner_registry import install_adapters
    from .publication import _installed_metadata
    from .space import require_capacity

    rows = journal._records(parent)
    prior = latest_plan(rows)
    values = material.records()
    scopes = dict(prior.evidence["scopes"]) if prior else {}
    if prior:
        selected = [record for record in values if record["id"] in scopes]
        if (
            credentials._fingerprint(values) != prior.evidence["material_digest"]
            or _reference_mapping(material, prepared, plan, selected)
            != prior.evidence["references"]
        ):
            raise ValueError("rollback_credential_material_changed")
        items = {item.logical_id: item for item in prepared.artifacts}
        for artifact in alternate_artifacts(prior).values():
            if not any(
                _matches(artifact.metadata, path, metadata=True)
                for path in (artifact.candidate.path, items[artifact.logical_id].target)
            ):
                raise ValueError("rollback_originals_unverified")
    changed = False
    historically_applied = _applied_scopes(rows)
    for record in values:
        if record["status"] != "captured" or record["kind"] == "encrypted_config":
            continue
        if record["id"] in scopes:
            # An amendment cannot turn a previously applied scope into a new one.
            if (record["id"], scopes[record["id"]]["purpose"]) in historically_applied:
                credentials.verify_rollback_credential(record, scopes[record["id"]])
            continue
        try:
            matches = (
                credentials._read_scope(record, credentials._credential_store())
                == record["value"]
            )
        except Exception:  # noqa: BLE001 - unavailable old scope needs no shared mutation
            matches = False
        if not matches:
            scopes[record["id"]] = credentials.plan_rollback_credential(record)
            changed = True
    if not changed:
        return prior
    selected = [record for record in values if record["id"] in scopes]
    references = _reference_mapping(material, prepared, plan, selected)
    artifact_ids = {row["artifact_id"] for row in references}
    artifacts = []
    owners = {owner.owner_id: owner for owner in install_adapters()}
    for item in prepared.artifacts:
        if item.logical_id not in artifact_ids:
            continue
        source = _original_location(item)
        container = Path(item.retained).parent / ("credential-candidate-" + uuid4().hex)
        require_capacity({container.parent: item.previous.size * 2})
        create_private_directory(container)
        candidate = container / "artifact"
        _copy_original(source, candidate, cancel)
        copied = observe_artifact(candidate, metadata=True)
        if not _matches(item.previous_metadata, str(source), metadata=True) or (
            copied["size"],
            copied["sha256"],
        ) != (item.previous_metadata.size, item.previous_metadata.sha256):
            raise ValueError("rollback_credential_original_changed")
        for reference in references:
            if reference["artifact_id"] != item.logical_id:
                continue
            record = next(
                row for row in selected if row["id"] == reference["record_id"]
            )
            leaf = (
                candidate
                if item.previous.kind == "file"
                else candidate / Path(reference["target"]).relative_to(item.target)
            )
            before = leaf.lstat()
            _installed_metadata(
                leaf,
                (before.st_dev, before.st_ino),
                {"mode": 0o600, "mtime_ns": before.st_mtime_ns},
            )
            with material.session._capture_bound_sources(
                (), container, ArchiveLimits(), ArchiveLimits().expanded_bytes
            ):
                expected = json.loads(credentials._read(leaf))
                matches = [
                    row
                    for row in credentials._targets(expected)
                    if row["server_id"] == record["server_id"]
                ]
                if not matches or any(
                    row.get("auth_reference") != "keyring:" + record["purpose"]
                    for row in matches
                ):
                    raise ValueError("rollback_credential_reference_changed")
                for row in matches:
                    row["auth_reference"] = "keyring:" + scopes[record["id"]]["purpose"]
                credentials._remap_server_reference(
                    leaf, record, scopes[record["id"]]["purpose"]
                )
                if json.loads(credentials._read(leaf)) != expected:
                    raise ValueError("rollback_credential_reference_changed")
                issues = owners[reference["owner"]].validate(leaf)
                if issues:
                    raise ValueError(issues[0])
            current = leaf.lstat()
            _installed_metadata(
                leaf,
                (current.st_dev, current.st_ino),
                {"mode": stat.S_IMODE(before.st_mode), "mtime_ns": before.st_mtime_ns},
            )
        # Rewrites change containing directory timestamps; reset only the copied
        # declared artifact's supported metadata from the unchanged original tree.
        _restore_copy_directory_metadata(source, candidate)
        if not _matches(item.previous_metadata, str(source), metadata=True):
            raise ValueError("rollback_credential_original_changed")
        artifacts.append(
            {
                "logical_id": item.logical_id,
                "candidate": observe_artifact(candidate),
                "metadata": observe_artifact(candidate, metadata=True),
            }
        )
    rollback = next(row for row in rows if row.event == "rollback_verified")
    proof = {
        "rollback_digest": _evidence_digest(rollback.evidence),
        "previous_digest": _evidence_digest(rows[-1].evidence),
        "material_digest": credentials._fingerprint(values),
        "scopes": scopes,
        "references": references,
        "artifacts": artifacts,
    }
    journal._append(parent, "rollback_credentials_planned", proof)
    journal._flush_records(parent)
    return journal._records(parent)[-1]


def _restore_copy_directory_metadata(source, target):
    from .publication import _installed_metadata

    if not stat.S_ISDIR(source.lstat().st_mode):
        return
    for name in sorted(os.listdir(source)):
        if stat.S_ISDIR((source / name).lstat().st_mode):
            _restore_copy_directory_metadata(source / name, target / name)
    info, current = source.lstat(), target.lstat()
    _installed_metadata(
        target,
        (current.st_dev, current.st_ino),
        {"mode": stat.S_IMODE(info.st_mode), "mtime_ns": info.st_mtime_ns},
    )


def apply_credentials(material, journal, parent, plan_record):
    if plan_record is None:
        material.check()
        return
    values = material.records()
    plan = plan_record.evidence
    if credentials._fingerprint(values) != plan["material_digest"]:
        raise ValueError("rollback_credential_material_changed")
    records = journal._records(parent)
    historically_applied = _applied_scopes(records)
    rows = current_phase(records)
    applied = {
        row.evidence["record_id"]
        for row in rows
        if row.event == "rollback_credential_applied"
    }
    for record in values:
        key = record["id"]
        if key not in plan["scopes"]:
            continue
        result = credentials.verify_rollback_credential(
            record,
            plan["scopes"][key],
            apply=(key, plan["scopes"][key]["purpose"]) not in historically_applied,
        )
        if key not in applied:
            journal._append(
                parent,
                "rollback_credential_applied",
                {"plan_digest": _evidence_digest(plan), "record_id": key, **result},
            )
            journal._flush_records(parent)
    material.check(plan_record)


def alternate_artifacts(plan_record):
    return (
        {
            row.logical_id: row
            for row in _RollbackCredentialPlan.model_validate(
                plan_record.evidence
            ).artifacts
        }
        if plan_record
        else {}
    )


def _move(journal, parent, prepared, item, step, source, destination):
    from .journal import _MoveIntent
    from .publication import (
        _complete_move,
        _move_position,
        _move_topology,
        _reverse_native_move,
    )
    from .space import require_capacity

    require_capacity({source.parent: 0, destination.parent: 0})
    proof = {
        "logical_id": item.logical_id,
        "step": step,
        "source": observe_artifact(source),
        "destination": str(destination),
        "parents": _move_topology(source, destination),
    }
    intent = _MoveIntent.model_validate(proof)
    if _move_position(intent) != "before":
        raise ValueError("move_state_uncertain")
    journal._append(parent, "move_intended", proof)
    journal._flush_records(parent)
    _reverse_native_move(intent)
    _complete_move(journal, parent, prepared, intent, moved=True)


def restore_alternate(journal, parent, prepared, item, alternate):
    """Publish the exact local remapped original, retaining its immutable raw source."""
    from .publication import _begin_move, _complete_move, _reverse_native_move

    _original_location(item)
    if not any(
        _matches(alternate.metadata, path, metadata=True)
        for path in (alternate.candidate.path, item.target)
    ):
        raise ValueError("rollback_originals_unverified")
    if _matches(alternate.metadata, item.target, metadata=True):
        if not _matches(item.previous_metadata, item.retained, metadata=True):
            raise ValueError("rollback_originals_unverified")
        return
    rows = journal._records(parent)
    previous = [
        entry
        for row in rows
        if row.event == "rollback_credentials_planned"
        for entry in _RollbackCredentialPlan.model_validate(row.evidence).artifacts
        if entry.logical_id == item.logical_id
    ]
    older = next(
        (
            entry
            for entry in previous
            if _matches(entry.metadata, item.target, metadata=True)
        ),
        None,
    )
    if older:
        _move(
            journal,
            parent,
            prepared,
            item,
            "credential_unpublish",
            Path(item.target),
            Path(older.candidate.path),
        )
    elif item.candidate and _matches(item.candidate, item.target):
        intent = _begin_move(journal, parent, item, "unpublish")
        _reverse_native_move(intent)
        _complete_move(journal, parent, prepared, intent, moved=True)
    elif _matches(item.previous_metadata, item.target, metadata=True):
        _move(
            journal,
            parent,
            prepared,
            item,
            "credential_retire",
            Path(item.target),
            Path(item.retained),
        )
    elif os.path.lexists(item.target):
        raise ValueError("rollback_originals_unverified")
    if not _matches(
        item.previous_metadata, item.retained, metadata=True
    ) or not _matches(alternate.metadata, alternate.candidate.path, metadata=True):
        raise ValueError("rollback_originals_unverified")
    _move(
        journal,
        parent,
        prepared,
        item,
        "credential_publish",
        Path(alternate.candidate.path),
        Path(item.target),
    )


def originals_proof(prepared, plan_record):
    alternatives = alternate_artifacts(plan_record)
    result = []
    for item in prepared.artifacts:
        if item.previous is None:
            continue
        alternative = alternatives.get(item.logical_id)
        expected = alternative.metadata if alternative else item.previous_metadata
        if not _matches(expected, item.target, metadata=True):
            raise ValueError("rollback_originals_unverified")
        if alternative and not _matches(
            item.previous_metadata, item.retained, metadata=True
        ):
            raise ValueError("rollback_originals_unverified")
        result.append(observe_artifact(Path(item.target), metadata=True))
    return result
