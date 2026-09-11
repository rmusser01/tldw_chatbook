"""Installed local evals recovery declarations; never import runtime engines.

Exact schema SQL captured from real installed constructors at task6 base 8ea52cfc0.
"""

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from threading import Event
from typing import Mapping

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    SchemaPolicy,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import database_path
from tldw_chatbook.DB.recovery_sqlite import _checked_capture, _validate_sqlite

_SCHEMA = (
    (
        5,
        (
            "CREATE INDEX idx_ab_test_runs_test ON ab_test_runs (ab_test_id)",
            "CREATE INDEX idx_ab_tests_models ON ab_tests (model_a_id, model_b_id)",
            "CREATE INDEX idx_ab_tests_status ON ab_tests (status)",
            "CREATE INDEX idx_ab_tests_task ON ab_tests (task_id)",
            "CREATE INDEX idx_eval_results_run ON eval_results (run_id)",
            "CREATE INDEX idx_eval_run_metrics_run ON eval_run_metrics (run_id)",
            "CREATE INDEX idx_eval_runs_group ON eval_runs (run_group_id)",
            "CREATE INDEX idx_eval_runs_model ON eval_runs (model_id)",
            "CREATE INDEX idx_eval_runs_status ON eval_runs (status)",
            "CREATE INDEX idx_eval_runs_task ON eval_runs (task_id)",
            "CREATE INDEX idx_eval_tasks_deleted ON eval_tasks (deleted_at)",
            "CREATE INDEX idx_eval_tasks_type ON eval_tasks (task_type)",
            "CREATE INDEX idx_probe_annotations_group ON eval_probe_turn_annotations (run_group_id)",
            "CREATE INDEX idx_probe_review_group ON eval_probe_review_state (run_group_id)",
            "CREATE TABLE ab_test_runs (\n                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),\n                ab_test_id TEXT NOT NULL,\n                run_a_id TEXT NOT NULL,\n                run_b_id TEXT NOT NULL,\n                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                client_id TEXT NOT NULL,\n                FOREIGN KEY (ab_test_id) REFERENCES ab_tests (id),\n                FOREIGN KEY (run_a_id) REFERENCES eval_runs (id),\n                FOREIGN KEY (run_b_id) REFERENCES eval_runs (id)\n            )",
            "CREATE TABLE ab_tests (\n                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),\n                test_id TEXT NOT NULL UNIQUE,\n                name TEXT NOT NULL,\n                description TEXT,\n                task_id TEXT NOT NULL,\n                model_a_id TEXT NOT NULL,\n                model_b_id TEXT NOT NULL,\n                config TEXT NOT NULL, -- JSON configuration\n                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')) DEFAULT 'pending',\n                winner TEXT CHECK (winner IN ('model_a', 'model_b', 'tie', NULL)),\n                result_data TEXT, -- JSON result data\n                started_at TEXT,\n                completed_at TEXT,\n                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                version INTEGER NOT NULL DEFAULT 1,\n                client_id TEXT NOT NULL,\n                deleted_at TEXT,\n                FOREIGN KEY (task_id) REFERENCES eval_tasks (id),\n                FOREIGN KEY (model_a_id) REFERENCES eval_models (id),\n                FOREIGN KEY (model_b_id) REFERENCES eval_models (id)\n            )",
            "CREATE TABLE eval_datasets (\n                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),\n                name TEXT NOT NULL UNIQUE,\n                description TEXT,\n                format TEXT NOT NULL CHECK (format IN ('huggingface', 'json', 'csv', 'custom')),\n                source_path TEXT NOT NULL,\n                metadata TEXT, -- JSON metadata\n                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                version INTEGER NOT NULL DEFAULT 1,\n                client_id TEXT NOT NULL,\n                deleted_at TEXT\n            )",
            "CREATE VIRTUAL TABLE eval_datasets_fts USING fts5(\n                id UNINDEXED,\n                name,\n                description,\n                content='eval_datasets',\n                content_rowid='rowid'\n            )",
            "CREATE TABLE 'eval_datasets_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID",
            "CREATE TABLE 'eval_datasets_fts_data'(id INTEGER PRIMARY KEY, block BLOB)",
            "CREATE TABLE 'eval_datasets_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB)",
            "CREATE TABLE 'eval_datasets_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID",
            "CREATE TABLE eval_models (\n                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),\n                name TEXT NOT NULL,\n                provider TEXT NOT NULL,\n                model_id TEXT NOT NULL,\n                config TEXT, -- JSON configuration for model parameters\n                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                version INTEGER NOT NULL DEFAULT 1,\n                client_id TEXT NOT NULL,\n                deleted_at TEXT,\n                UNIQUE(name, provider, model_id)\n            )",
            "CREATE TABLE eval_probe_review_state (\n                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),\n                run_group_id TEXT NOT NULL,\n                card_id INTEGER NOT NULL,\n                probe_index INTEGER NOT NULL,\n                sample_index INTEGER NOT NULL,\n                target_id TEXT NOT NULL,\n                note TEXT NOT NULL DEFAULT '',\n                reviewed_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                client_id TEXT NOT NULL,\n                UNIQUE(run_group_id, card_id, probe_index, sample_index, target_id)\n            )",
            "CREATE TABLE eval_probe_turn_annotations (\n                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),\n                run_group_id TEXT NOT NULL,\n                card_id INTEGER NOT NULL,\n                probe_index INTEGER NOT NULL,\n                sample_index INTEGER NOT NULL,\n                target_id TEXT NOT NULL,\n                turn_index INTEGER NOT NULL,\n                tags TEXT NOT NULL,          -- JSON list of tag slugs\n                note TEXT NOT NULL DEFAULT '',\n                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                client_id TEXT NOT NULL,\n                UNIQUE(run_group_id, card_id, probe_index, sample_index, target_id, turn_index)\n            )",
            "CREATE TABLE eval_results (\n                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),\n                run_id TEXT NOT NULL,\n                sample_id TEXT NOT NULL,\n                input_data TEXT NOT NULL, -- JSON input data\n                expected_output TEXT,\n                actual_output TEXT,\n                logprobs TEXT, -- JSON log probabilities if available\n                metrics TEXT, -- JSON metrics for this sample\n                metadata TEXT, -- JSON additional metadata\n                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                client_id TEXT NOT NULL,\n                FOREIGN KEY (run_id) REFERENCES eval_runs (id),\n                UNIQUE(run_id, sample_id)\n            )",
            "CREATE TABLE eval_run_metrics (\n                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),\n                run_id TEXT NOT NULL,\n                metric_name TEXT NOT NULL,\n                metric_value REAL NOT NULL,\n                metric_type TEXT NOT NULL CHECK (metric_type IN ('accuracy', 'f1', 'rouge', 'bleu', 'perplexity', 'custom')),\n                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                client_id TEXT NOT NULL,\n                FOREIGN KEY (run_id) REFERENCES eval_runs (id),\n                UNIQUE(run_id, metric_name)\n            )",
            "CREATE TABLE eval_runs (\n                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),\n                name TEXT NOT NULL,\n                task_id TEXT NOT NULL,\n                model_id TEXT NOT NULL,\n                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')) DEFAULT 'pending',\n                start_time TEXT,\n                end_time TEXT,\n                total_samples INTEGER,\n                completed_samples INTEGER DEFAULT 0,\n                config_overrides TEXT, -- JSON overrides for task config\n                run_group_id TEXT,\n                error_message TEXT,\n                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                version INTEGER NOT NULL DEFAULT 1,\n                client_id TEXT NOT NULL,\n                deleted_at TEXT,\n                FOREIGN KEY (task_id) REFERENCES eval_tasks (id),\n                FOREIGN KEY (model_id) REFERENCES eval_models (id)\n            )",
            "CREATE TABLE eval_tasks (\n                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),\n                name TEXT NOT NULL UNIQUE,\n                description TEXT,\n                task_type TEXT NOT NULL CHECK (task_type IN ('question_answer', 'logprob', 'generation', 'classification')),\n                config_format TEXT NOT NULL CHECK (config_format IN ('eleuther', 'custom')),\n                config_data TEXT NOT NULL, -- JSON configuration\n                dataset_id TEXT,\n                created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),\n                version INTEGER NOT NULL DEFAULT 1,\n                client_id TEXT NOT NULL,\n                deleted_at TEXT,\n                FOREIGN KEY (dataset_id) REFERENCES eval_datasets (id)\n            )",
            "CREATE VIRTUAL TABLE eval_tasks_fts USING fts5(\n                id UNINDEXED,\n                name,\n                description,\n                content='eval_tasks',\n                content_rowid='rowid'\n            )",
            "CREATE TABLE 'eval_tasks_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID",
            "CREATE TABLE 'eval_tasks_fts_data'(id INTEGER PRIMARY KEY, block BLOB)",
            "CREATE TABLE 'eval_tasks_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB)",
            "CREATE TABLE 'eval_tasks_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID",
            "CREATE TRIGGER eval_datasets_fts_delete AFTER DELETE ON eval_datasets BEGIN\n                INSERT INTO eval_datasets_fts (eval_datasets_fts, rowid, id, name, description)\n                VALUES ('delete', old.rowid, old.id, old.name, old.description);\n            END",
            "CREATE TRIGGER eval_datasets_fts_insert AFTER INSERT ON eval_datasets BEGIN\n                INSERT INTO eval_datasets_fts (rowid, id, name, description)\n                VALUES (new.rowid, new.id, new.name, new.description);\n            END",
            "CREATE TRIGGER eval_datasets_fts_update AFTER UPDATE ON eval_datasets BEGIN\n                INSERT INTO eval_datasets_fts (eval_datasets_fts, rowid, id, name, description)\n                VALUES ('delete', old.rowid, old.id, old.name, old.description);\n                INSERT INTO eval_datasets_fts (rowid, id, name, description)\n                VALUES (new.rowid, new.id, new.name, new.description);\n            END",
            "CREATE TRIGGER eval_tasks_fts_delete AFTER DELETE ON eval_tasks BEGIN\n                INSERT INTO eval_tasks_fts (eval_tasks_fts, rowid, id, name, description)\n                VALUES ('delete', old.rowid, old.id, old.name, old.description);\n            END",
            "CREATE TRIGGER eval_tasks_fts_insert AFTER INSERT ON eval_tasks BEGIN\n                INSERT INTO eval_tasks_fts (rowid, id, name, description)\n                VALUES (new.rowid, new.id, new.name, new.description);\n            END",
            "CREATE TRIGGER eval_tasks_fts_update AFTER UPDATE ON eval_tasks BEGIN\n                INSERT INTO eval_tasks_fts (eval_tasks_fts, rowid, id, name, description)\n                VALUES ('delete', old.rowid, old.id, old.name, old.description);\n                INSERT INTO eval_tasks_fts (rowid, id, name, description)\n                VALUES (new.rowid, new.id, new.name, new.description);\n            END",
        ),
    ),
)
_VERSIONS = (5,)
_MIGRATIONS = ()


@dataclass(frozen=True)
class _Adapter:
    owner_id: str = "db.evals"
    activation_required: bool = True

    def discover(self, config: Mapping[str, object]) -> tuple[StorageItem, ...]:
        context = discovery_context(config)
        path = database_path(config, "evals_db_path")
        try:
            status = "included" if path.is_file() else "missing_required"
        except OSError:
            status = "unavailable"
        return (
            StorageItem(
                self.owner_id,
                storage_logical_id(context, self.owner_id),
                path,
                status,
                (
                    storage_logical_id(context, "config"),
                    storage_logical_id(context, "db.chachanotes.primary"),
                ),
            ),
        )

    def schema_policy(self) -> SchemaPolicy:
        return SchemaPolicy(self.owner_id, _VERSIONS, _SCHEMA, _MIGRATIONS)

    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.domain.evals", candidate, read_only=True
                )
            ) as conn:
                return _validate_sqlite(conn, _VERSIONS, _SCHEMA)
        except (OSError, ValueError, sqlite3.Error):
            return ("domain_validation_unavailable",)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.domain.evals", item.path, destination, progress_guard=guard
            )

    def validate_dependencies(
        self, item: StorageItem, candidate: Path, candidates: Mapping[str, Path]
    ) -> tuple[str, ...]:
        """Check live local bench references without interpreting historical runs.

        Snapshots in past runs are self-contained history; current character bench
        definitions reference the selected local character store. External dataset
        source paths remain explicit external inputs, never guessed owned files.
        """
        import json
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
        from tldw_chatbook.DB.recovery_core import core_adapters

        parts = item.logical_id.split(":")
        if (
            item.owner != self.owner_id
            or len(parts) != 3
            or parts[0] != "profile"
            or parts[2] != self.owner_id
        ):
            return ("invalid_dependency_context",)
        issues = self.validate(candidate)
        if issues:
            return issues
        try:
            with closing(
                connect_private_sqlite(
                    "recovery.domain.evals", candidate, read_only=True
                )
            ) as conn:
                conn.execute("PRAGMA trusted_schema=OFF")
                character_ids = set()
                for raw in conn.execute("SELECT config_data FROM eval_tasks"):
                    data = json.loads(raw[0] or "{}")
                    if not isinstance(data, dict):
                        return ("invalid_domain_reference",)
                    if data.get("bench_type") not in ("word_bench", "character_probe"):
                        continue
                    targets = data.get("target_ids", [])
                    if not isinstance(targets, list):
                        return ("invalid_domain_reference",)
                    for target in targets:
                        if (
                            not isinstance(target, str)
                            or conn.execute(
                                "SELECT 1 FROM eval_models WHERE id=?", (target,)
                            ).fetchone()
                            is None
                        ):
                            return ("invalid_domain_reference",)
                    if data.get("bench_type") == "character_probe":
                        ids = data.get("character_ids", [])
                        if not isinstance(ids, list) or any(
                            type(value) is not int for value in ids
                        ):
                            return ("invalid_domain_reference",)
                        character_ids.update(ids)
                if not character_ids:
                    return ()
                key = ":".join(parts[:2]) + ":db.chachanotes.primary"
                if key not in item.dependencies or key not in candidates:
                    return ("dependency_unavailable",)
                peer_adapter = next(
                    a for a in core_adapters() if a.owner_id == "db.chachanotes.primary"
                )
                if peer_adapter.validate(candidates[key]):
                    return ("dependency_unavailable",)
                with closing(
                    connect_private_sqlite(
                        "recovery.domain.evals", candidates[key], read_only=True
                    )
                ) as peer:
                    peer.execute("PRAGMA trusted_schema=OFF")
                    for identity in character_ids:
                        if (
                            peer.execute(
                                "SELECT 1 FROM character_cards WHERE id=?", (identity,)
                            ).fetchone()
                            is None
                        ):
                            return ("invalid_domain_reference",)
                return ()
        except (OSError, ValueError, TypeError, OverflowError, sqlite3.Error):
            return ("dependency_unavailable",)

    def relocate(self, candidate: Path, mapping: Mapping[str, Path]) -> None:
        # Content and identifiers are stored inside this database. External user
        # references remain inert; never rewrite arbitrary prose/JSON as paths.
        issues = self.validate(candidate)
        if issues:
            raise ValueError(issues[0])


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_Adapter(), _DefinitionsAdapter())


def _retained_definition_paths(context) -> tuple[tuple[str, Path], ...]:
    """Resolve inactive eval files from this selector's completed local receipt.

    The private plan supplies destinations; the verified manifest supplies owner
    and config dependencies. Candidate trees and eval execution approvals are not
    needed. Rolled-back generations use the original target and rollback proof;
    incoming receipts cannot describe reinstated originals.
    """
    import hashlib
    import os
    import stat

    from tldw_chatbook.Backup_Recovery import archive_reader, bootstrap
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore, _private
    from tldw_chatbook.Backup_Recovery.journal import _CandidateReceipt, _Prepared
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.plan_records import load_plan
    from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
    from tldw_chatbook.Backup_Recovery.restore_plan import _ancestor

    selector = bootstrap.lexical_path(context.config_path)
    root = bootstrap.default_bootstrap_root()
    before = bootstrap._control_records(root)
    _, profiles, associations = before
    profile = next((r for r in profiles if r["selector"] == str(selector)), None)
    association = next(
        (r for r in associations if r["selector"] == str(selector)), None
    )
    witness = profile.get("activation") if profile else None
    if witness is None and association is None:
        return ()
    if witness is None or association is None or association["activation"] != witness:
        raise ValueError("eval_retained_generation_unverified")
    allowed, reason = bootstrap.startup_permission(selector, root)
    if not allowed:
        raise ValueError(reason)
    binding = bootstrap._binding(selector, profiles, bootstrap._registry(root))
    if binding is None:
        raise ValueError("eval_retained_binding_changed")
    control = Path(witness["store_root"]).parent
    if Path(witness["store_root"]) != control / "activation":
        raise ValueError("eval_retained_generation_unverified")
    journal = _journal(control, witness["operation_id"])
    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
    if not rows or rows[-1].event not in {"committed", "rolled_back"}:
        raise ValueError("eval_retained_commit_required")
    prepared = _Prepared.model_validate(
        next(row.evidence for row in rows if row.event == "prepared")
    )
    if rows[-1].event == "rolled_back":
        retained = _rolled_back_definition_paths(
            selector, root, witness, profiles, binding, rows, prepared, journal
        )
        with journal._locked(exclusive=False) as parent:
            if journal._records(parent) != rows:
                raise ValueError("eval_retained_generation_changed")
        if bootstrap._control_records(root) != before:
            raise ValueError("eval_retained_generation_changed")
        return retained
    activation = next(
        row.evidence for row in rows if row.event == "activation_recorded"
    )
    publication = prepared.publication
    if (
        publication is None
        or publication.bootstrap_root != str(root)
        or prepared.generation != witness["generation"]
        or witness["namespaces"] != binding["namespaces"]
        or not set(witness["namespaces"]) <= set(publication.namespaces)
        or str(selector) not in activation["selectors"]
        or activation["owners"] != witness["owners"]
    ):
        raise ValueError("eval_retained_generation_unverified")
    store = ActivationStore(Path(witness["store_root"]))
    with (
        _private(store.root),
        _private(store._generation(prepared.generation)) as parent,
    ):
        if store._required(parent, prepared.generation).owners != witness["owners"]:
            raise ValueError("eval_retained_generation_unverified")
    if prepared.isolated_profiles:
        from tldw_chatbook.Backup_Recovery.isolated_restore import _launch_descriptor

        entry = next(
            (r for r in prepared.isolated_profiles if r.config == str(selector)), None
        )
        if entry is None or _launch_descriptor(entry.profile_id, control) != entry:
            raise ValueError("eval_retained_profile_unverified")
    elif not any(r.config == str(selector) for r in prepared.replacement_profiles):
        raise ValueError("eval_retained_profile_unverified")

    plan = load_plan(journal)
    receipt = _CandidateReceipt.model_validate(rows[0].evidence)
    if (
        plan.archive_digest != receipt.archive_digest
        or publication.plan_digest != receipt.plan_digest
    ):
        raise ValueError("eval_retained_plan_unverified")
    limits = ArchiveLimits()
    with archive_reader._regular(journal.root / "verified-manifest.json") as stream:
        info = os.fstat(stream.fileno())
        if (
            info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("verified_manifest_changed")
        raw = stream.read(limits.manifest_bytes + 1)
        if archive_reader._identity(info) != archive_reader._identity(
            os.fstat(stream.fileno())
        ):
            raise ValueError("verified_manifest_changed")
    if (
        len(raw) > limits.manifest_bytes
        or hashlib.sha256(raw).hexdigest() != receipt.manifest_digest
    ):
        raise ValueError("verified_manifest_changed")
    doc = archive_reader._manifest(raw, limits, encrypted=True)
    if plan.local_snapshot is not None:
        from tldw_chatbook.Backup_Recovery.later_rollback import verify_snapshot_source

        _, snapshot = verify_snapshot_source(plan)
        if (
            snapshot.manifest_digest != receipt.manifest_digest
            or doc.credential_policy != "rollback"
        ):
            raise ValueError("eval_retained_plan_unverified")
    destinations = dict(plan.restore)
    configs = {
        row.logical_id
        for row in doc.files
        if row.owner_id == "config" and destinations.get(row.logical_id) == selector
    }
    if len(configs) != 1:
        raise ValueError("eval_retained_config_unverified")
    producers = {row.logical_id: row for row in doc.producer_inventory}
    retained = []
    for row in doc.files:
        if row.owner_id != "eval.definitions" or row.logical_id not in destinations:
            continue
        item = producers.get(row.logical_id)
        if item is None or item.status != "included":
            raise ValueError("eval_retained_owner_unverified")
        if not configs.intersection(item.dependencies):
            continue
        path = destinations[row.logical_id]
        if not any(
            path == Path(p) or Path(p) in path.parents for p in binding["roots"]
        ):
            raise ValueError("eval_retained_destination_unverified")
        _ancestor(path)
        retained.append((row.logical_id, path))
    if plan.local_snapshot is not None:
        retained.extend(
            _original_definition_paths(
                selector, binding, rows, prepared, plan, preserved_only=True
            )
        )
    with journal._locked(exclusive=False) as parent:
        if journal._records(parent) != rows:
            raise ValueError("eval_retained_generation_changed")
    if bootstrap._control_records(root) != before:
        raise ValueError("eval_retained_generation_changed")
    return tuple(retained)


def _rolled_back_definition_paths(
    selector, root, witness, profiles, binding, rows, prepared, journal
):
    """Use the original target and actual rollback coverage, never incoming YAML."""
    from tldw_chatbook.Backup_Recovery.activation import _rollback_installation_id
    from tldw_chatbook.Backup_Recovery.plan_records import load_plan

    if prepared.publication is None:
        raise ValueError("eval_retained_generation_unverified")
    _rollback_installation_id(selector, root, witness, profiles, rows, prepared)
    return _original_definition_paths(
        selector, binding, rows, prepared, load_plan(journal)
    )


def _original_definition_paths(
    selector, binding, rows, prepared, plan, *, preserved_only=False
):
    """Retain exact original sources covered by this operation's own safety copy."""
    from tldw_chatbook.Backup_Recovery.journal import _Rollback
    from tldw_chatbook.Backup_Recovery.restore_plan import _ancestor

    target = plan.target
    if target is None or not target.complete:
        raise ValueError("eval_retained_originals_unverified")
    configs = {
        item.logical_id
        for item in target.items
        if item.owner == "config"
        and item.path == selector
        and item.status == "included"
    }
    if len(configs) != 1:
        raise ValueError("eval_retained_config_unverified")
    proof = _Rollback.model_validate(
        next(row.evidence for row in rows if row.event == "rollback_verified")
    )
    items = {item.logical_id: item for item in target.items}
    previous = []
    for artifact in prepared.artifacts:
        if artifact.previous is None:
            continue
        item = items.get(proof.coverage.get(artifact.logical_id))
        # SQLite sidecars map to their main DB's semantic snapshot and do not
        # authorize a raw eval path. Only exact original targets cover YAML.
        if item is not None and item.path == Path(artifact.target):
            previous.append(artifact.previous)
    if proof.safety_sources != prepared.safety_sources:
        raise ValueError("eval_retained_originals_unverified")
    safety = {item.logical_id: item for item in proof.safety_sources}
    retained = []
    for item in target.items:
        if item.owner != "eval.definitions" or not configs.intersection(
            item.dependencies
        ):
            continue
        if preserved_only and (item.logical_id, item.path) not in plan.preserve:
            continue
        path = item.path
        if path is None or item.status != "included":
            raise ValueError("eval_retained_owner_unverified")
        saved = safety.get(item.logical_id)
        covered = (
            not preserved_only
            and any(
                path == Path(record.path)
                or (record.kind == "directory" and Path(record.path) in path.parents)
                for record in previous
            )
        ) or (
            item.logical_id in plan.safety_scope
            and saved is not None
            and saved.owner_id == item.owner
            and Path(saved.source.path) == path
        )
        if not covered:
            raise ValueError("eval_retained_originals_unverified")
        if not any(
            path == Path(p) or Path(p) in path.parents for p in binding["roots"]
        ):
            raise ValueError("eval_retained_destination_unverified")
        _ancestor(path)
        if path in {value for _, value in retained}:
            raise ValueError("eval_retained_owner_unverified")
        retained.append((item.logical_id, path))
    return tuple(retained)


@dataclass(frozen=True)
class _DefinitionsAdapter:
    owner_id: str = "eval.definitions"
    activation_required: bool = True

    def discover(self, config: Mapping[str, object]) -> tuple[StorageItem, ...]:
        context = discovery_context(config)
        # Exact EvalConfigLoader default; never inspect arbitrary parents/home or
        # import its YAML/runtime bootstrap during declaration discovery.
        import hashlib

        from . import _default_config_path

        path = _default_config_path()
        paths = [("", path)]
        for logical_id, retained in _retained_definition_paths(context):
            if retained not in {value for _, value in paths}:
                paths.append(
                    (hashlib.sha256(logical_id.encode()).hexdigest(), retained)
                )
        return tuple(
            StorageItem(
                self.owner_id,
                storage_logical_id(context, self.owner_id, local_id),
                selected,
                "included" if selected.is_file() else "missing_required",
                (storage_logical_id(context, "config"),),
            )
            for local_id, selected in paths
        )

    def schema_policy(self) -> SchemaPolicy:
        # Version 1 is this installed portable YAML mapping policy, not a false
        # on-disk version stamp. SQLite schema SQL does not apply to raw YAML.
        return SchemaPolicy(self.owner_id, (1,), (), ())

    def validate(self, candidate: Path) -> tuple[str, ...]:
        import yaml

        try:
            from tldw_chatbook.Backup_Recovery.storage_admission import (
                _read_recovery_file,
            )

            raw = _read_recovery_file(
                "eval.definitions", candidate, max_bytes=16 * 1024**2
            )
            data = yaml.safe_load(raw)
            if not isinstance(data, dict) or any(not isinstance(k, str) for k in data):
                return ("unsupported_definition_schema",)
            # Imported tags, recursive aliases, nonportable scalars and excessive
            # nesting are unsupported; no object constructors or evaluation run.
            active = set()
            count = 0

            def check(value, depth=0):
                nonlocal count
                count += 1
                if count > 100000 or depth > 32:
                    return False
                if type(value) in (str, int, float, bool, type(None)):
                    return True
                if type(value) not in (dict, list) or id(value) in active:
                    return False
                active.add(id(value))
                valid = (
                    (
                        all(type(k) is str for k in value)
                        and all(check(v, depth + 1) for v in value.values())
                    )
                    if isinstance(value, dict)
                    else all(check(v, depth + 1) for v in value)
                )
                active.remove(id(value))
                return valid

            if not check(data):
                return ("unsupported_definition_schema",)
            return ()
        except (OSError, UnicodeError, ValueError, RuntimeError, yaml.YAMLError):
            return ("definition_validation_unavailable",)

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.Backup_Recovery.storage_admission import copy_capture_file

        if (
            item.owner != self.owner_id
            or item.path is None
            or item.status != "included"
        ):
            raise ValueError("invalid_capture_item")
        copy_capture_file(
            "eval.definitions", item.path, destination, cancel, max_bytes=16 * 1024**2
        )
        issues = self.validate(destination)
        if issues:
            raise ValueError(issues[0])

    def relocate(self, candidate: Path, mapping: Mapping[str, Path]) -> None:
        issues = self.validate(candidate)
        if issues:
            raise ValueError(issues[0])
