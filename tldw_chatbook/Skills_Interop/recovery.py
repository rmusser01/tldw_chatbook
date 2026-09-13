"""Installed local skill definitions and quarantined trust-byte inventory."""

from dataclasses import replace
from pathlib import Path

from tldw_chatbook.Backup_Recovery.config_adapter import _Definition
from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import (
    lexical_path,
    setting,
    user_data_dir,
)


def default_local_skills_store_dir(user_data_dir: str | Path) -> Path:
    """Canonical app-owned skills root, shared with the runtime selector."""
    return Path(user_data_dir) / "skills"


def default_script_output_root(config) -> Path | None:
    """Select the installed retained sandbox child; custom roots stay unqualified.

    The sandbox physical owner captures this subtree with its same-owner parent
    topology. Selection never constructs services or creates the source root.
    """
    sandbox = user_data_dir(config) / "tool_sandbox"
    output = sandbox / "skill_script_output"
    for section, key, canonical in (
        ("tools", "file_sandbox_root", sandbox),
        ("skills", "script_scratch_root", output),
    ):
        value = setting(config, section, key)
        if (
            value is not None
            and value != ""
            and (not isinstance(value, str) or lexical_path(value) != canonical)
        ):
            return None
    return output


class _Skills(_Definition):
    def discover(self, config):
        entries = super().discover(config)
        if default_script_output_root(config) is None:
            context = discovery_context(config)
            entries += (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(
                        context, self.owner_id, "script_output_unqualified"
                    ),
                    None,
                    "unsupported",
                    (storage_logical_id(context, "config"),),
                ),
            )
        root = next(
            (
                item.path
                for item in entries
                if item.metadata and item.metadata.relative_path == ""
            ),
            None,
        )
        if root is None:
            return entries
        entries = tuple(
            replace(item, status="unsupported")
            if item.path
            and (
                item.path == root
                and item.status == "included"
                or item.path != root
                and item.path.relative_to(root).parts[0]
                not in {"tldw_chatbook_skills.json", "skills", "trust"}
            )
            else item
            for item in entries
        )
        return self._references(config, root, entries)

    def _references(self, config, root, entries):
        """Check exact saved bundle/snapshot references without unlocking trust."""
        import hashlib
        import json

        from tldw_chatbook.Backup_Recovery.storage_admission import _read_recovery_file
        from tldw_chatbook.tldw_api.skills_schemas import _normalize_skill_name

        context = discovery_context(config)
        by_path = {item.path: item for item in entries if item.path is not None}
        issues = []
        for path, kind in (
            (root / "tldw_chatbook_skills.json", "index"),
            (root / "trust" / "skill_trust_manifest.json", "trust"),
        ):
            catalog = by_path.get(path)
            if catalog is None or catalog.status != "included":
                continue
            try:
                document = json.loads(
                    _read_recovery_file(self.owner_id, path, max_bytes=16 * 1024**2)
                )
                records = (
                    document.get("skills", {})
                    if kind == "index"
                    else document["manifest"]["skills"]
                )
                if not isinstance(records, dict):
                    raise TypeError("invalid_skill_records")
                references = set()
                for name, record in records.items():
                    if not isinstance(name, str) or not isinstance(record, dict):
                        raise TypeError("invalid_skill_record")
                    if kind == "index":
                        references.add(
                            root / "skills" / _normalize_skill_name(name) / "SKILL.md"
                        )
                    else:
                        snapshot = record["snapshot_id"]
                        if (
                            not isinstance(snapshot, str)
                            or not snapshot
                            or snapshot in {".", ".."}
                            or "/" in snapshot
                            or "\\" in snapshot
                        ):
                            raise ValueError("invalid_snapshot_reference")
                        references.add(
                            root / "trust" / "snapshots" / (snapshot + ".json")
                        )
                dependencies = set(catalog.dependencies)
                for reference in sorted(references):
                    target = by_path.get(reference)
                    if target is not None and target.status == "included":
                        dependencies.add(target.logical_id)
                        continue
                    issues.append(
                        StorageItem(
                            self.owner_id,
                            storage_logical_id(
                                context,
                                self.owner_id,
                                "missing_reference_"
                                + hashlib.sha256(str(reference).encode()).hexdigest(),
                            ),
                            None,
                            "missing_required" if target is None else "unsupported",
                            (catalog.logical_id,),
                        )
                    )
                by_path[path] = replace(
                    catalog, dependencies=tuple(sorted(dependencies))
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                KeyError,
                TypeError,
                AttributeError,
                RecursionError,
            ):
                issues.append(
                    StorageItem(
                        self.owner_id,
                        storage_logical_id(
                            context, self.owner_id, kind + "_references_unavailable"
                        ),
                        None,
                        "unsupported",
                        (catalog.logical_id,),
                    )
                )
        return tuple(by_path.get(item.path, item) for item in entries) + tuple(issues)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    # Trust manifests/grants/snapshots remain historical bytes: importing a grant
    # never grants execution. Credential processing belongs to task14.
    return (_Skills("skills", leaf="skills", tree=True),)
