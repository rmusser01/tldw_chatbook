"""Installed TTS profiles and voice inventory, without runtime constructors."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path
import sqlite3
import hashlib
from threading import Event

from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    StorageItem,
    discovery_context,
    storage_logical_id,
)
from tldw_chatbook.Backup_Recovery.profile_paths import (
    database_path,
    setting,
    lexical_path,
)
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration
from tldw_chatbook.DB.recovery_operations import _SQLiteDeclaration
from tldw_chatbook.DB.recovery_sqlite import _checked_capture, _validate_sqlite

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
        return tuple(
            replace(item, shared_group="shared:tts:profile:" + context.profile_id)
            if item.status == "included"
            else item
            for item in items
        )

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
            if (
                len(payload) != count
                or hashlib.sha256(payload).hexdigest() != digest
            ):
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
        ):
            value = setting(config, section, key)
            if value is not None and type(value) is not str:
                raise ValueError("invalid_voice_path")
        roots = (
            lexical_path(
                setting(config, "app_tts", "CHATTERBOX_VOICE_DIR")
                or "~/.config/tldw_cli/chatterbox_voices"
            ),
            lexical_path(
                setting(config, "HiggsSettings", "voice_samples_dir")
                or "~/.config/tldw_cli/higgs_voices"
            ),
            lexical_path(setting(config, "app_tts", "KOKORO_VOICE_BLENDS_DIR"))
            if setting(config, "app_tts", "KOKORO_VOICE_BLENDS_DIR") is not None
            else context.config_path.parent / "kokoro_voice_blends",
            context.config_path.parent / "kokoro_voice_blends.json",
        )
        entries = tuple(item for root in roots for item in self._tree(config, root))
        if any(item.status in {"included", "included_directory"} for item in entries):
            entries += (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id, "participant_pending"),
                    None,
                    "unsupported",
                    (),
                ),
            )
        # Migration journal/candidates/rollback slots are independently bounded
        # exact owner names. Their presence requires existing owner reconciliation;
        # discovery never replays or treats them as generic disposable outputs.
        from .profile_migration_journal import (
            PROFILE_MIGRATION_JOURNAL_LEAF,
            PROFILE_MIGRATION_CANDIDATE_LEAVES,
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
        ),
        _References(
            "tts.references",
            "tts_profiles_db_path",
            "tldw_chatbook_tts_profiles.db",
            (4,),
            _SCHEMA,
        ),
    )
