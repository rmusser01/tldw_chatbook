"""Installed TTS profiles and voice inventory, without runtime constructors."""

import hashlib
import sqlite3
import stat
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from threading import Event

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import (
    database_path,
    lexical_path,
    setting,
)
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration
from tldw_chatbook.DB.recovery_operations import _SQLiteDeclaration
from tldw_chatbook.DB.recovery_sqlite import _checked_capture, _validate_sqlite
from tldw_chatbook.Utils.platform_files import os

_SCHEMA = (
    (
        4,
        (
            "CREATE INDEX idx_character_tts_assignments_profile_id\nON character_tts_assignments(profile_id)\n",
            "CREATE UNIQUE INDEX idx_tts_profile_clone_references_reference_id\nON tts_profile_clone_references(reference_id)\n",
            "CREATE TABLE character_tts_assignments (\n    source TEXT NOT NULL,\n    authority_id TEXT NOT NULL,\n    character_id TEXT NOT NULL,\n    profile_id TEXT NOT NULL,\n    created_at TEXT NOT NULL,\n    updated_at TEXT NOT NULL,\n    PRIMARY KEY(source, authority_id, character_id),\n    FOREIGN KEY(profile_id)\n        REFERENCES tts_generation_profiles(profile_id)\n        ON DELETE RESTRICT\n)",
            "CREATE TABLE tts_generation_profiles (\n    profile_id TEXT PRIMARY KEY,\n    display_name TEXT NOT NULL,\n    normalized_name TEXT NOT NULL UNIQUE,\n    provider_id TEXT NOT NULL,\n    model_id TEXT NOT NULL,\n    voice_id TEXT NULL,\n    response_format TEXT NOT NULL,\n    speed REAL NOT NULL,\n    options_json TEXT NOT NULL,\n    revision INTEGER NOT NULL,\n    created_at TEXT NOT NULL,\n    updated_at TEXT NOT NULL\n)",
            "CREATE TABLE tts_profile_clone_references (\n    profile_id TEXT PRIMARY KEY,\n    reference_id TEXT NOT NULL,\n    wav_bytes BLOB NOT NULL\n        CHECK(typeof(wav_bytes) = 'blob'\n            AND length(wav_bytes) BETWEEN 1 AND 33554432),\n    reference_text TEXT NOT NULL\n        CHECK(typeof(reference_text) = 'text'\n            AND length(reference_text) BETWEEN 1 AND 4096\n            AND length(CAST(reference_text AS BLOB)) <= 16384),\n    sha256 TEXT NOT NULL\n        CHECK(typeof(sha256) = 'text'\n            AND length(sha256) = 64\n            AND sha256 = lower(sha256)\n            AND sha256 NOT GLOB '*[^0-9a-f]*'),\n    byte_length INTEGER NOT NULL\n        CHECK(typeof(byte_length) = 'integer'\n            AND byte_length BETWEEN 1 AND 33554432\n            AND byte_length = length(wav_bytes)),\n    duration_ms INTEGER NOT NULL\n        CHECK(typeof(duration_ms) = 'integer'\n            AND duration_ms BETWEEN 1 AND 60000),\n    sample_rate_hz INTEGER NOT NULL\n        CHECK(typeof(sample_rate_hz) = 'integer'\n            AND sample_rate_hz BETWEEN 8000\n                AND 96000),\n    channels INTEGER NOT NULL\n        CHECK(typeof(channels) = 'integer' AND channels IN (1, 2)),\n    sample_encoding TEXT NOT NULL\n        CHECK(typeof(sample_encoding) = 'text'\n            AND sample_encoding = 'pcm_s16le'),\n    created_at TEXT NOT NULL,\n    updated_at TEXT NOT NULL,\n    recipe_id TEXT NULL,\n    recipe_revision INTEGER NULL,\n    CHECK(\n        (recipe_id IS NULL AND recipe_revision IS NULL)\n        OR (\n            typeof(recipe_id) = 'text'\n            AND length(CAST(recipe_id AS BLOB)) BETWEEN 1 AND 128\n            AND instr(CAST(recipe_id AS BLOB), x'00') = 0\n            AND recipe_id GLOB '[a-z0-9]*'\n            AND recipe_id NOT GLOB '*[^a-z0-9._-]*'\n            AND typeof(recipe_revision) = 'integer'\n            AND recipe_revision BETWEEN 1 AND 2147483647\n        )\n    ),\n    FOREIGN KEY(profile_id)\n        REFERENCES tts_generation_profiles(profile_id)\n        ON DELETE CASCADE\n)",
        ),
    ),
)


class _Profiles(_SQLiteDeclaration):
    def discover(self, config):
        items = super().discover(config)
        context = discovery_context(config)
        items = tuple(
            replace(item, shared_group="shared:tts:profile:" + context.profile_id)
            if item.status == "included"
            else item
            for item in items
        )
        if self.owner_id != "tts.profile_store":
            return items
        # ProfileStoreLease retains this exact empty sibling after closing. Its
        # native admission holds remain responsible for accepted lock operations.
        database = database_path(config, "tts_profiles_db_path")
        lock = _RawDeclaration(self.owner_id)._item(
            config, database.with_name(database.name + ".lock"), "lock"
        )
        if lock.status == "included":
            from tldw_chatbook.Backup_Recovery.bootstrap import pinned_directory

            try:
                with pinned_directory(lock.path.parent) as parent:
                    info = os.stat(lock.path.name, dir_fd=parent, follow_symlinks=False)
                    empty = (
                        stat.S_ISREG(info.st_mode)
                        and info.st_nlink == 1
                        and info.st_size == 0
                    )
                lock = replace(
                    lock, status="intentionally_excluded" if empty else "unsupported"
                )
            except (OSError, ValueError, RuntimeError):
                lock = replace(lock, status="unavailable")
        elif lock.status == "unused":
            lock = replace(lock, status="intentionally_excluded")
        return items + (lock,)

    def validate(self, candidate: Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite("recovery.files.tts", candidate, read_only=True)
            ) as connection:
                return self._validate_connection(connection)
        except (OSError, ValueError, sqlite3.Error, RuntimeError):
            return ("tts_validation_unavailable",)

    def _validate_connection(self, connection):
        """Run existing owned-content checks on an already restricted connection."""
        issues = _validate_sqlite(connection, self.versions, self.schemas)
        if issues:
            return issues
        for payload, count, digest in connection.execute(
            "SELECT wav_bytes,byte_length,sha256 FROM tts_profile_clone_references"
        ):
            if len(payload) != count or hashlib.sha256(payload).hexdigest() != digest:
                return ("tts_reference_digest_mismatch",)
        return ()

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.files.tts", item.path, destination, progress_guard=guard
            )


class _References(_Profiles):
    """Clone reference bytes are BLOBs in the exact profile store, not loose WAVs."""

    def discover(self, config):
        return tuple(
            replace(
                item,
                dependencies=item.dependencies
                + (storage_logical_id(discovery_context(config), "tts.profile_store"),),
            )
            for item in super().discover(config)
        )


class _Voices(_RawDeclaration):
    def discover(self, config):
        context = discovery_context(config)
        for section, key in (
            ("app_tts", "CHATTERBOX_VOICE_DIR"),
            ("app_tts", "KOKORO_VOICE_BLENDS_DIR"),
            ("HiggsSettings", "voice_samples_dir"),
            ("global_tts_settings", "CHATTERBOX_VOICE_DIR"),
            ("global_tts_settings", "KOKORO_VOICE_BLENDS_DIR"),
            ("local_chatterbox_default", "CHATTERBOX_VOICE_DIR"),
            ("local_kokoro_default_onnx", "KOKORO_VOICE_BLENDS_DIR"),
            ("local_kokoro_default_pytorch", "KOKORO_VOICE_BLENDS_DIR"),
            ("local_higgs_default", "HIGGS_VOICE_SAMPLES_DIR"),
            ("local_higgs_v2", "HIGGS_VOICE_SAMPLES_DIR"),
        ):
            value = setting(config, section, key)
            if value is not None and (type(value) is not str or not value):
                raise ValueError("invalid_voice_path")
        backend_higgs = config.get("HIGGS_VOICE_SAMPLES_DIR")
        if backend_higgs is not None and (
            type(backend_higgs) is not str or not backend_higgs
        ):
            raise ValueError("invalid_voice_path")
        # Catalog readers use these fixed shared locations even when the
        # configured manager/backend destinations differ. Location classification
        # does not make saved voice content an optional model/external folder.
        shared_chatterbox = lexical_path("~/.config/tldw_cli/chatterbox_voices")
        shared_higgs = lexical_path("~/.config/tldw_cli/higgs_voices")
        configured_kokoro = setting(config, "app_tts", "KOKORO_VOICE_BLENDS_DIR")
        kokoro_root = (
            lexical_path(configured_kokoro)
            if configured_kokoro is not None
            else context.config_path.parent / "kokoro_voice_blends"
        )
        roots = (
            lexical_path(setting(config, "app_tts", "CHATTERBOX_VOICE_DIR"))
            if setting(config, "app_tts", "CHATTERBOX_VOICE_DIR") is not None
            else shared_chatterbox,
            lexical_path(setting(config, "HiggsSettings", "voice_samples_dir"))
            if setting(config, "HiggsSettings", "voice_samples_dir") is not None
            else shared_higgs,
            lexical_path(backend_higgs) if backend_higgs is not None else shared_higgs,
            shared_chatterbox,
            shared_higgs,
            kokoro_root,
            context.config_path.parent / "kokoro_voice_blends.json",
        )
        # Mirror only the five installed legacy routes. The adapter receives
        # validated raw TOML; normalized runtime wrappers are not another source.
        chatterbox_base = setting(
            config,
            "app_tts",
            "CHATTERBOX_VOICE_DIR",
            setting(config, "global_tts_settings", "CHATTERBOX_VOICE_DIR"),
        )
        kokoro_base = setting(
            config,
            "app_tts",
            "KOKORO_VOICE_BLENDS_DIR",
            setting(config, "global_tts_settings", "KOKORO_VOICE_BLENDS_DIR"),
        )
        higgs_base = (
            backend_higgs
            if backend_higgs is not None
            else setting(config, "HiggsSettings", "voice_samples_dir")
        )
        for route, key, base, fallback in (
            (
                "local_chatterbox_default",
                "CHATTERBOX_VOICE_DIR",
                chatterbox_base,
                shared_chatterbox,
            ),
            (
                "local_kokoro_default_onnx",
                "KOKORO_VOICE_BLENDS_DIR",
                kokoro_base,
                kokoro_root,
            ),
            (
                "local_kokoro_default_pytorch",
                "KOKORO_VOICE_BLENDS_DIR",
                kokoro_base,
                kokoro_root,
            ),
            (
                "local_higgs_default",
                "HIGGS_VOICE_SAMPLES_DIR",
                higgs_base,
                shared_higgs,
            ),
            ("local_higgs_v2", "HIGGS_VOICE_SAMPLES_DIR", higgs_base, shared_higgs),
        ):
            value = setting(config, route, key, base)
            roots += (lexical_path(value) if value is not None else fallback,)
        # Actual admitted constructor selections may differ from a saved
        # configuration. Never expand reviewed roots from process-local state.
        # A loaded finite owner can add refusal evidence only; cold discovery
        # remains constructor-free and follows the installed configuration routes.
        lifetime = sys.modules.get("tldw_chatbook.TTS.loose_voice_lifetime")
        observed = (
            () if lifetime is None else lifetime.observed_sources(context.config_path)
        )
        uncovered = tuple(
            StorageItem(
                self.owner_id,
                storage_logical_id(
                    context,
                    self.owner_id,
                    "uncovered_live_source-"
                    + hashlib.sha256(str(root).encode()).hexdigest(),
                ),
                root,
                "unsupported",
                (),
            )
            for root, resolved, identity in observed
            if not any(
                root == declared or root.is_relative_to(declared) for declared in roots
            )
        )
        # Dedup only identical lexical roots; resolving symlink/hardlink aliases
        # here would hide the inventory's existing physical-alias diagnostics.
        entries = tuple(
            item for root in dict.fromkeys(roots) for item in self._tree(config, root)
        )
        # The same lexical member can be enumerated from overlapping actual
        # roots. Canonicalize its identity and rewrite relationships together;
        # distinct alias spellings and conflicting status remain observable.
        members = {}
        aliases = {}
        observations = {}
        discovery_changed = False
        for item in entries:
            observation = (
                (
                    item.metadata.version,
                    item.metadata.kind,
                    item.metadata.mode,
                    item.metadata.mtime_ns,
                    item.metadata.policy,
                )
                if item.metadata is not None
                else None
            )
            key = (
                item.path,
                item.status,
                item.shared_group,
                item.deletion_validated,
                observation,
            )
            if item.path is not None:
                previous_observation = observations.setdefault(item.path, key[1:])
                discovery_changed |= previous_observation != key[1:]
            previous = members.get(key)
            if previous is None:
                members[key] = item
            else:
                aliases[item.logical_id] = previous.logical_id
        entries = tuple(
            replace(
                item,
                dependencies=tuple(
                    dict.fromkeys(aliases.get(key, key) for key in item.dependencies)
                ),
                metadata=replace(
                    item.metadata,
                    root_id=aliases.get(item.metadata.root_id, item.metadata.root_id),
                    parent_id=aliases.get(
                        item.metadata.parent_id, item.metadata.parent_id
                    ),
                )
                if item.metadata is not None
                else None,
            )
            for item in members.values()
        )
        entries += uncovered
        if discovery_changed:
            entries += (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id, "discovery_changed"),
                    None,
                    "unsupported",
                    (),
                ),
            )
        # Installed loose voice operations retain exact native source admission
        # through their final publication. The live storage drain, not a blanket
        # inventory marker, refuses accepted work that has not actually settled.
        # Migration journal/candidates/rollback slots are independently bounded
        # exact owner names. Their presence requires existing owner reconciliation;
        # discovery never replays or treats them as generic disposable outputs.
        from .profile_migration_journal import (
            PROFILE_MIGRATION_CANDIDATE_LEAVES,
            PROFILE_MIGRATION_JOURNAL_LEAF,
            PROFILE_MIGRATION_ROLLBACK_LEAVES,
        )

        database = database_path(config, "tts_profiles_db_path")
        names = (
            PROFILE_MIGRATION_JOURNAL_LEAF,
            *PROFILE_MIGRATION_CANDIDATE_LEAVES.values(),
            *PROFILE_MIGRATION_ROLLBACK_LEAVES.values(),
            database.name + ".pre-v3.sqlite3",
            database.name + ".pre-v4.sqlite3",
        )
        for name in names:
            item = self._item(
                config,
                database.parent / name,
                "migration-" + hashlib.sha256(name.encode()).hexdigest(),
            )
            entries += (
                replace(item, status="unsupported")
                if item.status == "included"
                else item,
            )
        return entries


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (
        _Voices("tts.voices"),
        _Profiles(
            "tts.profile_store",
            "tts_profiles_db_path",
            "tldw_chatbook_tts_profiles.db",
            (4,),
            _SCHEMA,
            optional_default=True,
        ),
        _References(
            "tts.references",
            "tts_profiles_db_path",
            "tldw_chatbook_tts_profiles.db",
            (4,),
            _SCHEMA,
            optional_default=True,
        ),
    )
