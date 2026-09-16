"""Inert inventory of the installed managed store and selected dependency closure."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


def managed_artifact_root(data_dir: Path) -> Path:
    """Pure canonical counterpart of managed_model_artifact_root()."""
    return data_dir / "models" / "managed"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_model_manifest_key")
        result[key] = value
    return result


class _Artifacts(_RawDeclaration):
    def validate_restore_dependencies(self, item, candidate, candidates, *, topology):
        """Resolve declared recipe edges in archive topology, never source paths."""
        from tldw_chatbook.Backup_Recovery.storage_admission import (
            _digest_recovery_file,
        )

        meta = item.metadata
        if (
            item.owner != self.owner_id
            or meta is None
            or topology.get(item.logical_id)
            != (meta.root_id, meta.parent_id, meta.relative_path, meta.kind)
        ):
            return ("invalid_dependency_context",)
        layout = Path(meta.relative_path).parts
        if meta.kind != "file" or not layout or layout[-1] != "manifest.json":
            return ()
        if len(layout) != 6 or layout[:2] != ("managed", "artifacts"):
            return ("invalid_dependency_context",)
        try:
            descriptor = self._descriptor(candidate)
            ref = descriptor.reference
            if layout[2:5] != (ref.artifact_id, ref.revision, ref.variant):
                return ("model_identity_mismatch",)
            parts = item.logical_id.split(":")
            config_key = f"profile:{parts[1]}:config" if len(parts) > 2 else None
            edges = set(item.dependencies) - {meta.parent_id, config_key}
            payload_paths = {
                (Path(meta.relative_path).parent / payload.path).as_posix()
                for payload in descriptor.files
            }
            represented_payload = any(
                root == meta.root_id and relative in payload_paths and kind == "file"
                for key, (root, _parent, relative, kind) in topology.items()
                if key in candidates
            )
            if not edges and not represented_payload:
                return ()  # Baseline inert recipe, with model payloads omitted.
            located = {}
            for key in edges & candidates.keys() & topology.keys():
                root, _parent, relative, kind = topology[key]
                if root == meta.root_id and kind == "file":
                    located.setdefault(relative, []).append(key)

            def selected(relative):
                keys = located.get(relative, ())
                if len(keys) != 1:
                    raise ValueError("dependency_unavailable")
                return candidates[keys[0]]

            for payload in descriptor.files:
                path = selected(
                    (Path(meta.relative_path).parent / payload.path).as_posix()
                )
                if _digest_recovery_file(
                    self.owner_id, path, max_bytes=self.max_bytes
                ) != (payload.size_bytes, payload.sha256):
                    return ("model_payload_mismatch",)
            for dependency in descriptor.dependencies:
                relative = f"managed/artifacts/{dependency.artifact_id}/{dependency.revision}/{dependency.variant}/manifest.json"
                if self._descriptor(selected(relative)).reference != dependency:
                    return ("model_dependency_mismatch",)
            return ()
        except (OSError, ValueError, RuntimeError, RecursionError):
            return ("model_dependency_unavailable",)

    def _descriptor(self, candidate):
        from tldw_chatbook.Backup_Recovery.storage_admission import _read_recovery_file

        from .service import ArtifactDescriptor

        raw = json.loads(
            _read_recovery_file(self.owner_id, candidate, max_bytes=16 * 1024**2),
            object_pairs_hook=_unique_object,
        )
        if (
            type(raw) is not dict
            or set(raw) != {"schema_version", "descriptor"}
            or type(raw["schema_version"]) is not int
            or raw["schema_version"] != 1
        ):
            raise ValueError("invalid_model_manifest")
        return ArtifactDescriptor.from_dict(raw["descriptor"])

    def validate_dependencies(self, item, candidate, candidates):
        """Check exact declared staged manifest edges; never expand capture scope."""
        from tldw_chatbook.Backup_Recovery.models import DiscoveryContext
        from tldw_chatbook.Backup_Recovery.recovery_files import _tree_member_id
        from tldw_chatbook.Backup_Recovery.storage_admission import (
            _digest_recovery_file,
        )

        if item.path is None or item.path.name != "manifest.json":
            return ()
        parts = item.logical_id.split(":")
        if (
            item.owner != self.owner_id
            or len(parts) != 4
            or parts[0] != "profile"
            or parts[2] != self.owner_id
            or item.metadata is None
        ):
            return ("invalid_dependency_context",)
        layout = Path(item.metadata.relative_path).parts
        if (
            len(layout) != 6
            or layout[:2] != ("managed", "artifacts")
            or layout[-1] != "manifest.json"
        ):
            return ("invalid_dependency_context",)
        if len(item.path.parents) < 6:
            return ("invalid_dependency_context",)
        root = item.path.parents[5]
        context = DiscoveryContext(Path("/unused-config"), parts[1])
        if (
            item.path.relative_to(root).as_posix() != item.metadata.relative_path
            or _tree_member_id(context, self.owner_id, root, item.path)
            != item.logical_id
            or item.metadata.root_id
            != _tree_member_id(context, self.owner_id, root, root)
            or item.metadata.parent_id
            != _tree_member_id(context, self.owner_id, root, item.path.parent)
        ):
            return ("invalid_dependency_context",)
        try:
            descriptor = self._descriptor(candidate)
            ref = descriptor.reference
            if layout[2:5] != (ref.artifact_id, ref.revision, ref.variant):
                return ("model_identity_mismatch",)
            selected_edges = set(item.dependencies) - {
                item.metadata.parent_id,
                storage_logical_id(context, "config"),
            }
            if not selected_edges:
                return ()  # Inert baseline recipe; model payloads were not selected.
            for payload in descriptor.files:
                key = _tree_member_id(
                    context, self.owner_id, root, item.path.parent / payload.path
                )
                if key not in item.dependencies or key not in candidates:
                    return ("dependency_unavailable",)
                if _digest_recovery_file(
                    self.owner_id, candidates[key], max_bytes=self.max_bytes
                ) != (payload.size_bytes, payload.sha256):
                    return ("model_payload_mismatch",)
            for dep in descriptor.dependencies:
                path = (
                    root
                    / "managed"
                    / "artifacts"
                    / dep.artifact_id
                    / dep.revision
                    / dep.variant
                    / "manifest.json"
                )
                key = _tree_member_id(context, self.owner_id, root, path)
                if key not in item.dependencies or key not in candidates:
                    return ("dependency_unavailable",)
                if self._descriptor(candidates[key]).reference != dep:
                    return ("model_dependency_mismatch",)
            return ()
        except (OSError, ValueError, RuntimeError, RecursionError):
            return ("model_dependency_unavailable",)

    def discover(self, config):
        from tldw_chatbook.Backup_Recovery.storage_admission import (
            _digest_recovery_file,
        )

        context = discovery_context(config)
        root = managed_artifact_root(user_data_dir(config))
        entries = self._tree(config, root.parent)
        selected = set(context.selections.model_ids)
        descriptors = {}
        malformed = set()
        for item in entries:
            if (
                item.path is None
                or item.path.name != "manifest.json"
                or item.status != "included"
            ):
                continue
            if not item.path.is_relative_to(root):
                malformed.add(item.logical_id)
                continue
            relative = item.path.relative_to(root)
            if len(relative.parts) != 5 or relative.parts[0] != "artifacts":
                malformed.add(item.logical_id)
                continue
            try:
                descriptor = self._descriptor(item.path)
                ref = descriptor.reference
                if relative.parts[1:4] != (ref.artifact_id, ref.revision, ref.variant):
                    raise ValueError("model_identity_mismatch")
                descriptors[ref] = (descriptor, item.path.parent)
            except (OSError, ValueError, RuntimeError, RecursionError):
                malformed.add(item.logical_id)
        closure = set()
        missing = []
        todo = [
            ref
            for ref, (descriptor, _) in descriptors.items()
            if descriptor.model_id in selected
        ]
        seen_ids = {descriptors[ref][0].model_id for ref in todo}
        for model_id in sorted(selected - seen_ids):
            missing.append(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(
                        context,
                        self.owner_id,
                        "selected-" + hashlib.sha256(model_id.encode()).hexdigest(),
                    ),
                    None,
                    "unavailable",
                    (),
                )
            )
        while todo:
            ref = todo.pop()
            if ref in closure:
                continue
            closure.add(ref)
            if ref not in descriptors:
                missing.append(
                    StorageItem(
                        self.owner_id,
                        storage_logical_id(
                            context,
                            self.owner_id,
                            "dependency-"
                            + hashlib.sha256(repr(ref).encode()).hexdigest(),
                        ),
                        root
                        / "artifacts"
                        / ref.artifact_id
                        / ref.revision
                        / ref.variant,
                        "missing_required",
                        (),
                    )
                )
                continue
            todo.extend(descriptors[ref][0].dependencies)
        selected_paths = set()
        invalid_payloads = set()
        known_paths = set()
        for ref, (descriptor, directory) in descriptors.items():
            for payload in descriptor.files:
                path = directory / payload.path
                known_paths.add(path)
                if ref in closure:
                    selected_paths.add(path)
                    try:
                        observed = _digest_recovery_file(
                            self.owner_id, path, max_bytes=self.max_bytes
                        )
                        if observed != (payload.size_bytes, payload.sha256):
                            invalid_payloads.add(path)
                    except (OSError, ValueError, RuntimeError):
                        invalid_payloads.add(path)
        actual = {item.path: item for item in entries}
        for path in selected_paths - actual.keys():
            missing.append(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(
                        context,
                        self.owner_id,
                        "missing-" + hashlib.sha256(str(path).encode()).hexdigest(),
                    ),
                    path,
                    "missing_required",
                    (),
                )
            )
        dependencies = {}
        for ref in closure & descriptors.keys():
            descriptor, directory = descriptors[ref]
            required_paths = [directory / payload.path for payload in descriptor.files]
            required_paths.extend(
                descriptors[dep][1] / "manifest.json"
                for dep in descriptor.dependencies
                if dep in descriptors
            )
            dependencies[directory / "manifest.json"] = tuple(
                actual[path].logical_id for path in required_paths if path in actual
            )
        result = []
        for item in entries:
            if item.path is None or item.status in {
                "unused",
                "unavailable",
                "unsupported",
            }:
                result.append(item)
                continue
            if item.path == root.parent:
                result.append(item)
                continue
            if not item.path.is_relative_to(root):
                result.append(replace(item, status="unsupported"))
                continue
            relative = item.path.relative_to(root)
            parts = relative.parts
            status = item.status
            if item.logical_id in malformed or item.path in invalid_payloads:
                status = "unsupported"
            elif not parts:
                pass
            elif parts[0] == "locks":
                status = "intentionally_excluded"
            elif parts[0] == "staging" and len(parts) > 1:
                # Download resume markers/retained candidates are owner state,
                # never presumed disposable solely from their directory name.
                status = "unsupported"
            elif parts[0] not in {"artifacts", "active", "ready", "staging"}:
                status = "unsupported"
            elif parts[0] == "artifacts" and item.status == "included":
                if item.path.name == "manifest.json":
                    pass
                elif item.path in selected_paths:
                    pass
                elif item.path in known_paths:
                    status = "intentionally_excluded"
                else:
                    status = "unsupported"
            result.append(
                replace(
                    item,
                    status=status,
                    dependencies=tuple(
                        dict.fromkeys(
                            (*item.dependencies, *dependencies.get(item.path, ()))
                        )
                    ),
                )
            )
        return tuple(result + missing)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_Artifacts("models.artifacts"),)
