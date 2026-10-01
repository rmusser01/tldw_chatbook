"""Closed managed resume restrictions over the existing continuation owner.

This module does not acquire authority or retain execution. V1 codecs never import
it. A verified pin is only a maximum for a new, ordinarily authorized admission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Annotated, Literal

from pydantic import Field, model_serializer, model_validator

from .authority import AuthorityModel, Digest, Generation, Identifier
from .package_files import canonical_json

MAX_ENVELOPE_BYTES = 256 * 1024


class ScopePin(AuthorityModel):
    kind: Literal["installation", "workspace", "global_default"]
    workspace: Annotated[str, Field(max_length=256)]
    generation: Generation


class ComponentPin(AuthorityModel):
    component_id: Identifier
    definition_digest: Digest
    dependencies: Annotated[list[Identifier], Field(max_length=512)]


class RootPin(AuthorityModel):
    root_id: Identifier
    workspace_id: Identifier | None
    generation: Generation
    path_digest: Digest
    deletion_fenced: bool
    authority_digest: Digest | None = None
    live_epoch: Generation = 0

    @model_serializer(mode="wrap")
    def preserve_absent_root_extension(self, handler):
        value = handler(self)
        for field in ("authority_digest", "live_epoch"):
            if field not in self.model_fields_set:
                value.pop(field, None)
        return value


class InstallationPin(AuthorityModel):
    installation_id: Identifier
    revision_digest: Digest
    workspace_id: Identifier | None
    alias: Identifier
    generations: Annotated[list[ScopePin], Field(max_length=3)]
    live_generations: Annotated[list[ScopePin], Field(max_length=3)]
    components: Annotated[list[ComponentPin], Field(max_length=512)]
    mappings_digest: Digest
    data_coverage: Literal["qualified_none", "known", "unknown"]
    data_roots: Annotated[list[RootPin], Field(max_length=256)]

    @model_validator(mode="after")
    def unique_closed_sets(self):
        for values in (
            [(scope.kind, scope.workspace) for scope in self.generations],
            [(scope.kind, scope.workspace) for scope in self.live_generations],
            [component.component_id for component in self.components],
            [root.root_id for root in self.data_roots],
        ):
            if values != sorted(set(values)):
                raise ValueError("managed pin requires sorted unique identities")
        ids = {component.component_id for component in self.components}
        for component in self.components:
            if (
                component.dependencies != sorted(set(component.dependencies))
                or not set(component.dependencies) <= ids
            ):
                raise ValueError("managed dependency closure incomplete")
        if self.data_coverage == "qualified_none" and self.data_roots:
            raise ValueError("inconsistent managed data coverage")
        return self


class ManagedPin(AuthorityModel):
    schema_version: Literal[1]
    namespace_id: Digest
    session_id: Identifier
    run_id: Identifier
    conversation_id: Identifier
    message_id: Identifier
    installations: Annotated[list[InstallationPin], Field(max_length=64)]

    @model_validator(mode="after")
    def aggregate_limits(self):
        identities = [item.installation_id for item in self.installations]
        if (
            identities != sorted(set(identities))
            or sum(len(item.components) for item in self.installations) > 1024
            or sum(len(item.data_roots) for item in self.installations) > 256
        ):
            raise ValueError("managed pin aggregate limit or duplicate identity")
        return self


class ManagedEnvelope(ManagedPin):
    checkpoint_digest: Digest
    mac: Digest


def validate_envelope(value: dict) -> dict:
    """Validate structure only, preserving foreign pins as private history."""
    if len(canonical_json(value).encode()) > MAX_ENVELOPE_BYTES:
        raise ValueError("managed envelope byte limit")
    return ManagedEnvelope.model_validate(value).model_dump(mode="json")


@dataclass(frozen=True, repr=False)
class ResumeConstraint:
    """Immutable validated source checkpoint, reverified at actual fresh admission."""

    checkpoint: object
    conversation_id: str
    message_id: str


def _digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def installation_pin(service, snapshot, authority) -> dict:
    """Capture actual immutable component definitions and current root membership."""
    from .data_cleanup import applicable_roots, open_root

    roots = applicable_roots(authority, snapshot.installation_id, snapshot.workspace_id)
    known = True
    for row in roots:
        if (
            not row.get("custody")
            or row["root_id"] in service._coordinator.root_usage.recovery
        ):
            known = False
        elif row["custody"]["state"] == "present":
            try:
                with open_root(service._coordinator.owner, row):
                    pass
            except (OSError, ValueError):
                known = False
        elif row["custody"]["state"] != "cleaned_absent":
            known = False
    components = []
    for component_id in snapshot.selection:
        component = snapshot.inspection.inventory[component_id]
        components.append(
            {
                "component_id": component_id,
                "definition_digest": _digest(json.loads(component.definition_json)),
                "dependencies": sorted(component.dependencies),
            }
        )
    # F5 executes native package-only skill/read paths. Future data users must
    # supply qualified ownership here; an arbitrary empty root table is not proof.
    package_only = (
        all(
            snapshot.inspection.inventory[key].kind == "skill"
            for key in snapshot.selection
        )
        and snapshot.mappings_json == "[]"
    )
    return InstallationPin.model_validate(
        {
            "installation_id": snapshot.installation_id,
            "revision_digest": snapshot.revision_digest,
            "workspace_id": snapshot.workspace_id,
            "alias": snapshot.alias,
            "generations": [
                {"kind": kind, "workspace": workspace, "generation": generation}
                for kind, workspace, generation in sorted(snapshot.generations)
            ],
            "live_generations": [
                {"kind": kind, "workspace": workspace, "generation": generation}
                for kind, workspace, generation in sorted(snapshot.live_generations)
            ],
            "components": components,
            "mappings_digest": _digest(json.loads(snapshot.mappings_json)),
            "data_coverage": (
                "known"
                if roots and known
                else (
                    "unknown"
                    if roots
                    else "qualified_none"
                    if package_only
                    else "unknown"
                )
            ),
            "data_roots": [
                {
                    "root_id": row["root_id"],
                    "workspace_id": row["workspace_id"],
                    "generation": row["generation"],
                    "path_digest": _digest(row["path"]),
                    "deletion_fenced": row["deletion_fenced"],
                    "authority_digest": _digest(row),
                    "live_epoch": dict(snapshot.root_epochs).get(row["root_id"], 0),
                }
                for row in roots
            ],
        }
    ).model_dump(mode="json")


def capture_pin(
    service, entries, run_id: str, conversation_id: str, message_id: str
) -> str:
    """Capture only a real currently bound host producer, never transcript names."""
    authority = service._coordinator.published_snapshot()
    snapshots = {}
    with service._lock:
        for entry in entries:
            snapshot = service._admitted.get(entry.get("plugin_admission"))
            if snapshot is None:
                raise PermissionError("managed producer admission unavailable")
            record = service._live_runs.get((snapshot.installation_id, run_id))
            if (
                record is None
                or record.pending_id != snapshot.run_id
                or record.completed.is_set()
            ):
                raise PermissionError("managed producer owner unavailable")
            service.fences.check_snapshot(snapshot)
            if record.component_ceiling is not None:
                snapshot = replace(
                    snapshot,
                    selection=tuple(
                        key
                        for key in snapshot.selection
                        if key in record.component_ceiling
                    ),
                )
            snapshots[snapshot.installation_id] = snapshot
    for snapshot in snapshots.values():
        for component_id in snapshot.selection:
            service._admission.check(snapshot, component_id)
    pin = ManagedPin.model_validate(
        {
            "schema_version": 1,
            "namespace_id": service._coordinator.authority.namespace_id(),
            "session_id": service.fences.session_nonce,
            "run_id": run_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "installations": [
                installation_pin(service, snapshots[key], authority)
                for key in sorted(snapshots)
            ],
        }
    ).model_dump(mode="json")
    if len(canonical_json(pin).encode()) > MAX_ENVELOPE_BYTES - 200:
        raise ValueError("managed pin byte limit")
    return canonical_json(pin)


def seal_checkpoint(
    service, pin_json: str, checkpoint, conversation_id: str, message_id: str
):
    """Seal every final store-resolved checkpoint under frozen original custody."""
    from tldw_chatbook.Chat.provider_continuation import _checkpoint_value

    pin = ManagedPin.model_validate_json(pin_json).model_dump(mode="json")
    if (pin["conversation_id"], pin["message_id"]) != (conversation_id, message_id):
        raise PermissionError("managed checkpoint owner changed")
    body = _checkpoint_value(replace(checkpoint, schema_version=1, managed_resume=None))
    payload = dict(pin, checkpoint_digest=_digest(body))
    envelope = dict(
        payload, mac=service._coordinator.authority.sign_archive_pin(payload)
    )
    validate_envelope(envelope)
    return replace(
        checkpoint, schema_version=2, managed_resume=canonical_json(envelope)
    )


def verify_constraint(service, constraint: ResumeConstraint) -> dict:
    """Authenticate owner/body and compare exact pins to current ordinary authority."""
    from tldw_chatbook.Chat.provider_continuation import _checkpoint_value

    checkpoint = constraint.checkpoint
    envelope = validate_envelope(json.loads(checkpoint.managed_resume))
    payload = dict(envelope)
    mac = payload.pop("mac")
    authority = service._coordinator.published_snapshot()
    store = service._coordinator.authority
    store.verify_archive_pin(payload, mac)
    if (envelope["conversation_id"], envelope["message_id"]) != (
        constraint.conversation_id,
        constraint.message_id,
    ):
        raise PermissionError("managed continuation owner changed")
    body = _checkpoint_value(replace(checkpoint, schema_version=1, managed_resume=None))
    if _digest(body) != envelope["checkpoint_digest"]:
        raise PermissionError("managed checkpoint body changed")
    verify_pin_authority(service, envelope, authority)
    return envelope


def verify_pin_authority(service, envelope, authority=None):
    authority = (
        authority
        if authority is not None
        else service._coordinator.published_snapshot()
    )
    if envelope["namespace_id"] != service._coordinator.authority.namespace_id():
        raise PermissionError("managed namespace changed")
    for pinned in envelope["installations"]:
        current = service._admission.capture(
            pinned["installation_id"],
            pinned["workspace_id"],
            "managed-resume-validation",
        )
        selected = {row["component_id"] for row in pinned["components"]}
        if not selected <= set(current.selection):
            raise PermissionError("managed continuation components unavailable")
        current = replace(current, selection=tuple(sorted(selected)))
        observed = installation_pin(service, current, authority)
        old = dict(pinned)
        if envelope["session_id"] != service.fences.session_nonce:
            old.pop("live_generations")
            observed.pop("live_generations")
            old["data_roots"] = [
                {key: value for key, value in root.items() if key != "live_epoch"}
                for root in old["data_roots"]
            ]
            observed["data_roots"] = [
                {key: value for key, value in root.items() if key != "live_epoch"}
                for root in observed["data_roots"]
            ]
        if (
            old != observed
            or pinned["data_coverage"] == "unknown"
            or any(row["deletion_fenced"] for row in pinned["data_roots"])
        ):
            raise PermissionError("managed continuation authority or data changed")
    return envelope


def constrain_maximum(service, maximum: dict, constraint) -> dict:
    """Intersect frozen metadata before assembly and again at fresh admission."""
    result = dict(maximum)
    allowed = None
    if constraint == "zero":
        allowed = {}
    elif isinstance(constraint, ResumeConstraint):
        envelope = verify_constraint(service, constraint)
        allowed = {
            (row["installation_id"], row["workspace_id"]): {
                component["component_id"] for component in row["components"]
            }
            for row in envelope["installations"]
        }
    elif constraint is not None:
        raise PermissionError("managed resume constraint unavailable")
    if allowed is not None:
        rows = []
        for row in maximum.get("available_skills", ()):
            if row.get("plugin_owned"):
                snapshot = service._snapshots.get(row.get("plugin_ceiling"))
                if snapshot is None or row.get(
                    "plugin_component_id"
                ) not in allowed.get(
                    (snapshot.installation_id, row.get("plugin_workspace_id")), set()
                ):
                    continue
            rows.append(row)
        result["available_skills"] = rows
        result["context_text"] = "\n".join(str(row.get("name", "")) for row in rows)
        result["plugin_resume_constraint"] = constraint
    return result


def fleet_ceiling(service, pin_json, run_id, handle_id, entries):
    """Constrain current-parent inheritance using existing host-held fleet custody."""
    if pin_json is None:
        return {}
    pin = ManagedPin.model_validate_json(pin_json).model_dump(mode="json")
    if (
        len(pin_json.encode()) > MAX_ENVELOPE_BYTES
        or pin["session_id"] != service.fences.session_nonce
        or (pin["run_id"], pin["message_id"]) != (run_id, handle_id)
    ):
        raise PermissionError("managed fleet owner changed")
    verify_pin_authority(service, pin)
    parent = {}
    for entry in entries:
        snapshot = service._admitted.get(entry.get("plugin_admission"))
        if snapshot is not None:
            parent.setdefault(snapshot.installation_id, set()).add(
                entry["plugin_component_id"]
            )
    ceiling = {}
    for installation in pin["installations"]:
        selected = {
            component["component_id"] for component in installation["components"]
        }
        if not selected <= parent.get(installation["installation_id"], set()):
            raise PermissionError("managed parent ceiling unavailable")
        ceiling[installation["installation_id"]] = tuple(sorted(selected))
    return ceiling
