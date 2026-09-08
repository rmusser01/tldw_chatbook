"""Shared inert raw-file declarations; exact installed owners select all paths."""

from dataclasses import dataclass, replace
import hashlib
import json
import stat

from pydantic import ConfigDict, JsonValue, RootModel
from pathlib import Path
from threading import Event
from typing import Mapping

from .file_inventory import inventory_tree
from .models import StorageItem, SchemaPolicy, discovery_context, storage_logical_id


def _tree_member_id(context, owner: str, root: Path, path: Path) -> str:
    """Reproduce installed tree IDs without following historical filesystem text."""
    import os

    root_id = owner + ":" + hashlib.sha256(os.fsencode(root)).hexdigest()
    relative = str(path.relative_to(root))
    local_id = (
        root_id
        if relative == "."
        else root_id + ":" + hashlib.sha256(relative.encode()).hexdigest()
    )
    return storage_logical_id(
        context, owner, hashlib.sha256(local_id.encode()).hexdigest()
    )


class _JSONRecord(RootModel[dict[str, JsonValue]]):
    model_config = ConfigDict(strict=True)


@dataclass(frozen=True)
class _RawDeclaration:
    owner_id: str
    format: str = "opaque"
    max_bytes: int = 256 * 1024**3
    activation_required: bool = True

    def _item(self, config, path, local_id=""):
        context = discovery_context(config)
        entry = inventory_tree(path, owner=self.owner_id, external=False)[0]
        logical_id = storage_logical_id(context, self.owner_id, local_id)
        status = "unsupported" if entry.status == "included_directory" else entry.status
        return StorageItem(
            self.owner_id,
            logical_id,
            path,
            status,
            (storage_logical_id(context, "config"),),
            metadata=replace(entry.metadata, root_id=logical_id, parent_id=None)
            if entry.metadata
            else None,
        )

    def _tree(self, config, root, *, external=False, selected_paths=None):
        context = discovery_context(config)
        from .file_inventory import _inventory_tree

        entries = _inventory_tree(
            root, owner=self.owner_id, external=external, selected_paths=selected_paths
        )
        keys = {
            e.logical_id: storage_logical_id(
                context,
                self.owner_id,
                hashlib.sha256(e.logical_id.encode()).hexdigest(),
            )
            for e in entries
        }
        return tuple(
            StorageItem(
                self.owner_id,
                keys[e.logical_id],
                e.path,
                e.status,
                tuple(keys[k] for k in e.dependencies)
                + (storage_logical_id(context, "config"),),
                metadata=replace(
                    e.metadata,
                    root_id=keys[e.metadata.root_id],
                    parent_id=keys[e.metadata.parent_id]
                    if e.metadata.parent_id
                    else None,
                )
                if e.metadata
                else None,
            )
            for e in entries
        )

    def schema_policy(self) -> SchemaPolicy:
        # Installed raw policy revision, not an invented on-disk version stamp.
        return SchemaPolicy(self.owner_id, (1,), (), ())

    def validate(self, candidate: Path) -> tuple[str, ...]:
        from .storage_admission import _read_recovery_file, _check_recovery_file

        try:
            if self.format == "opaque":
                _check_recovery_file(self.owner_id, candidate, max_bytes=self.max_bytes)
                return ()
            data = json.loads(
                _read_recovery_file(self.owner_id, candidate, max_bytes=self.max_bytes)
            )
            _JSONRecord.model_validate(data, strict=True)
            # Plain JSON only. Operational values remain historical evidence;
            # parsing never loads services, commands, permissions or credentials.
            return ()
        except (OSError, ValueError, RuntimeError, RecursionError, UnicodeError):
            return ("operational_validation_unavailable",)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from .storage_admission import copy_capture_file, _check_recovery_file

        if (
            item.owner != self.owner_id
            or item.status != "included"
            or item.path is None
        ):
            raise ValueError("invalid_capture_item")
        copy_capture_file(
            self.owner_id, item.path, destination, cancel, max_bytes=self.max_bytes
        )
        if self.format == "opaque":
            _check_recovery_file(
                self.owner_id, destination, max_bytes=self.max_bytes, cancel=cancel
            )
        else:
            if cancel.is_set():
                raise InterruptedError("cancelled")
            issues = self.validate(destination)
            if issues:
                raise ValueError(issues[0])

    def relocate(self, candidate: Path, mapping: Mapping[str, Path]) -> None:
        """Preserve evidence; paths/permissions/commands never become live claims.

        Task20 durable activation and staged restore allocate/review separate local
        bindings. No imported authoritative binding is admitted by this owner.
        """
        issues = self.validate(candidate)
        if issues:
            raise ValueError(issues[0])
