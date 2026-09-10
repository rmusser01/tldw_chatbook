"""Pure configuration/definition recovery policy; no bootstrap or decryption."""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import re
import tomllib
from typing import Mapping

from .models import OwnerAdapter, StorageItem, discovery_context, storage_logical_id
from .profile_paths import DATABASE_PATHS, user_data_dir, default_config_path, setting
from .recovery_files import _RawDeclaration

_CONFIG_HISTORY = re.compile(r"config_backup_[0-9]{8}_[0-9]{6}\.toml\Z")
_MAX_CONFIG_BYTES = 16 * 1024**2


def managed_secret_locations(
    config: Mapping[str, object],
) -> tuple[tuple[str, ...], ...]:
    """Expose the installed config encryption owner's key locations to task14.

    No values, environment contents or keychain reads escape. Arbitrary prose is
    never scanned or rewritten. Unknown encrypted blobs are not decrypted.
    """
    from tldw_chatbook.Utils.sensitive_config_keys import is_sensitive_config_key

    locations = []

    def walk(value, prefix, depth):
        if depth > 64:
            raise ValueError("config_depth_limit")
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError("invalid_config_key")
            if key.startswith("__chatbook_"):
                continue
            location = prefix + (key,)
            if is_sensitive_config_key(key):
                locations.append(location)
            elif isinstance(child, Mapping):
                walk(child, location, depth + 1)
            elif isinstance(child, list):
                for index, entry in enumerate(child):
                    if isinstance(entry, Mapping):
                        walk(entry, location + (str(index),), depth + 1)

    walk(config, (), 0)
    return tuple(sorted(locations))


# Exact installed managed location keys. Plain strings inside user content are
# not locators, and a mapping never authorizes a target or changes live config.
CONFIG_LOCATION_KEYS = tuple(("database", row[1]) for row in DATABASE_PATHS) + (
    ("paths", "data_dir"),
    ("Paths", "data_dir"),
    ("notes", "sync_directory"),
    ("console", "workspace_root"),
    ("llm_management", "model_download_dir"),
    ("embedding_config", "model_cache_dir"),
    ("app_tts", "CHATTERBOX_VOICE_DIR"),
    ("app_tts", "KOKORO_VOICE_BLENDS_DIR"),
    ("HiggsSettings", "voice_samples_dir"),
)


def remap_config_locations(
    config: Mapping[str, object], mapping: Mapping[str, Path]
) -> dict:
    """Map exact installed ``section.key`` selector names to new local paths.

    Unknown names and conflicting lower/uppercase data-dir aliases refuse.
    Original path strings are values, never mapping keys or substring targets.
    """
    accepted = {
        section + "." + key: (section, key) for section, key in CONFIG_LOCATION_KEYS
    }
    if any(
        key not in accepted or not isinstance(value, Path)
        for key, value in mapping.items()
    ):
        raise ValueError("invalid_relocation_mapping")
    if (
        "paths.data_dir" in mapping
        and "Paths.data_dir" in mapping
        and mapping["paths.data_dir"] != mapping["Paths.data_dir"]
    ):
        raise ValueError("ambiguous_relocation_mapping")
    result = deepcopy(dict(config))
    for selector, target in mapping.items():
        section, key = accepted[selector]
        values = result.setdefault(section, {})
        if type(values) is not dict:
            raise ValueError("invalid_config_selector")
        values[key] = str(target)
    return result


class _Config(_RawDeclaration):
    def discover(self, config):
        context = discovery_context(config)
        item = self._item(config, context.config_path)
        return (
            replace(
                item,
                status="missing_required" if item.status == "unused" else item.status,
                dependencies=(),
            ),
        )

    def validate(self, candidate):
        from .storage_admission import _read_recovery_file

        try:
            data = _read_recovery_file(
                self.owner_id, candidate, max_bytes=_MAX_CONFIG_BYTES
            )
            tomllib.loads(data.decode("utf-8"))
            return ()
        except (OSError, ValueError, RuntimeError, UnicodeError, RecursionError):
            return ("config_validation_unavailable",)

    managed_secret_locations = staticmethod(managed_secret_locations)
    remap_locations = staticmethod(remap_config_locations)


class _History(_RawDeclaration):
    def discover(self, config):
        context = discovery_context(config)
        current = context.config_path
        paths = [current.with_suffix(current.suffix + ".bak")]
        from .bootstrap import pinned_directory
        import os

        try:
            with pinned_directory(current.parent) as parent:
                with os.scandir(parent) as entries:
                    for entry in entries:
                        if _CONFIG_HISTORY.fullmatch(entry.name):
                            paths.append(current.parent / entry.name)
            return tuple(
                self._item(config, path, hashlib.sha256(str(path).encode()).hexdigest())
                for path in sorted(paths)
            )
        except (OSError, ValueError, RuntimeError):
            return (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id),
                    current.parent,
                    "unavailable",
                    (storage_logical_id(context, "config"),),
                ),
            )


@dataclass(frozen=True)
class _Definition(_RawDeclaration):
    leaf: str = ""
    location: str = "data"
    tree: bool = False
    participant_pending: bool = False

    def _definition_path(self, config):
        context = discovery_context(config)
        base = {
            "data": user_data_dir(config),
            "config": context.config_path.parent,
            "package": Path(__file__).parents[1],
            "default_config": default_config_path().parent,
        }[self.location]
        return base / self.leaf

    def discover(self, config):
        context = discovery_context(config)
        path = self._definition_path(config)
        entries = self._tree(config, path) if self.tree else (self._item(config, path),)
        if self.participant_pending and any(
            item.status in {"included", "included_directory"} for item in entries
        ):
            # Explicit maintenance handoff: task10 cannot release startup admission
            # for these ordinary direct writers until their actual tokens drain.
            entries += (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id, "participant_pending"),
                    None,
                    "unsupported",
                    (storage_logical_id(context, "config"),),
                ),
            )
        return entries

    def _tree(self, config, root, **kwargs):
        context = discovery_context(config)
        entries = super()._tree(config, root, **kwargs)
        if entries:
            root_key = storage_logical_id(context, self.owner_id)
            old_key = entries[0].logical_id
            entries = tuple(
                replace(
                    item,
                    logical_id=root_key
                    if item.logical_id == old_key
                    else item.logical_id,
                    dependencies=tuple(
                        root_key if key == old_key else key for key in item.dependencies
                    ),
                    metadata=replace(
                        item.metadata,
                        root_id=root_key,
                        parent_id=root_key
                        if item.metadata.parent_id == old_key
                        else item.metadata.parent_id,
                    )
                    if item.metadata
                    else None,
                )
                for item in entries
            )
        return entries


class _ChatbookRegistry(_Definition):
    def _definition_path(self, config):
        from .profile_paths import database_path

        # TldwCli._build_chatbook_db_paths always supplies Prompts; the installed
        # service selects its sibling first, ahead of ChaChaNotes and Media.
        return database_path(config, "prompts_db_path").with_name(
            "tldw_chatbook_chatbooks.json"
        )

    def discover(self, config):
        context = discovery_context(config)
        return tuple(
            replace(
                item,
                dependencies=item.dependencies
                + (storage_logical_id(context, "db.prompts.primary"),),
            )
            for item in super().discover(config)
        )


class _InstanceLock(_Definition):
    def discover(self, config):
        item = self._item(config, user_data_dir(config) / ".instance.lock")
        # Only the installed PID/portalocker file; a directory or linked alias
        # at that name is not evidence of this process-owned format.
        return (
            replace(item, status="intentionally_excluded")
            if item.status in {"included", "unused"}
            else item,
        )


class _ChatbookScratch(_Definition):
    def discover(self, config):
        root = user_data_dir(config) / "temp"
        result = []
        for item in self._tree(config, root):
            relative = item.path.relative_to(root) if item.path else Path()
            if relative.parts:
                # Creator and Importer each remove their per-run work directory
                # in finally. Neither root holds a recovery journal or catalog.
                if relative.parts[0] in {"chatbooks", "imports"}:
                    if len(relative.parts) == 1 and item.status == "included":
                        item = replace(item, status="unsupported")
                    elif item.status in {"included", "included_directory"}:
                        item = replace(item, status="intentionally_excluded")
                else:
                    item = replace(item, status="unsupported")
            result.append(item)
        return tuple(result)


def _excluded_root(config, owner, path, *, kind, local_id=""):
    """Apply installed exclusion policy only after checked exact-kind evidence."""
    from .file_inventory import _inventory_root

    context = discovery_context(config)
    item = _inventory_root(path, owner=owner, external=False)
    logical_id = storage_logical_id(context, owner, local_id)
    status = item.status
    if status == "unused":
        status = "intentionally_excluded"
    elif status in {"included", "included_directory"}:
        status = (
            "intentionally_excluded"
            if item.metadata is not None and item.metadata.kind == kind
            else "unsupported"
        )
    return replace(
        item,
        logical_id=logical_id,
        status=status,
        metadata=replace(item.metadata, root_id=logical_id, parent_id=None)
        if item.metadata
        else None,
    )


class _Generated(_RawDeclaration):
    def discover(self, config):
        context = discovery_context(config)
        root = user_data_dir(config) / "generated_images"
        entries = self._tree(config, root)
        result = []
        for item in entries:
            relative = item.path.relative_to(root) if item.path else Path()
            if (
                relative.parts
                and relative.parts[0] == "temp"
                and not context.selections.temporary_media
            ):
                if len(relative.parts) == 1 and item.status == "included":
                    item = replace(item, status="unsupported")
                elif item.status in {"included", "included_directory"}:
                    item = replace(item, status="intentionally_excluded")
            elif relative.parts and relative.parts[0] not in {"temp", "saved"}:
                item = replace(item, status="unsupported")
            result.append(item)
        video_root = user_data_dir(config) / "generated_videos"
        if context.selections.temporary_media:
            result.extend(self._tree(config, video_root))
            # No claim that available pathname bytes establish transcript/media
            # reference completeness; task11 owns the retained asset catalogue.
            result.append(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(
                        context, self.owner_id, "temporary_reference_pending"
                    ),
                    None,
                    "unsupported",
                    (),
                )
            )
        else:
            result.append(
                _excluded_root(
                    config,
                    self.owner_id,
                    video_root,
                    kind="directory",
                    local_id="temporary_videos",
                )
            )
        if any(item.status in {"included", "included_directory"} for item in result):
            result.append(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id, "participant_pending"),
                    None,
                    "unsupported",
                    (),
                )
            )
        return tuple(result)


class _Diagnostics(_RawDeclaration):
    def discover(self, config):
        from .bootstrap import pinned_directory
        import os

        context = discovery_context(config)
        root = user_data_dir(config)
        name = setting(config, "logging", "log_filename", "tldw_cli_app.log")
        if (
            type(name) is not str
            or not name.strip()
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("invalid_diagnostic_selector")
        paths = {root / name}
        try:
            with pinned_directory(root) as parent:
                with os.scandir(parent) as entries:
                    for entry in entries:
                        suffix = entry.name.removeprefix(name + ".")
                        if (
                            entry.name.startswith(name + ".")
                            and suffix.isascii()
                            and suffix.isdigit()
                        ):
                            paths.add(root / entry.name)
        except (OSError, ValueError, RuntimeError):
            # Failure to inspect the rotation namespace is not proof that its
            # installed artifacts are absent, even when diagnostics are off.
            return tuple(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(
                        context,
                        self.owner_id,
                        hashlib.sha256(str(path).encode()).hexdigest(),
                    ),
                    path,
                    "unavailable",
                    (),
                )
                for path in sorted(paths)
            )
        if not context.selections.diagnostics:
            return tuple(
                _excluded_root(
                    config,
                    self.owner_id,
                    path,
                    kind="file",
                    local_id=hashlib.sha256(str(path).encode()).hexdigest(),
                )
                for path in sorted(paths)
            )
        return tuple(
            self._item(config, path, hashlib.sha256(str(path).encode()).hexdigest())
            for path in sorted(paths)
        )


class _CatalogCache(_Definition):
    def discover(self, config):
        return (
            _excluded_root(
                config,
                self.owner_id,
                user_data_dir(config) / "model_catalog_cache.json",
                kind="file",
            ),
        )


@dataclass(frozen=True)
class _Attachments:
    """Semantic ownership of message attachment BLOBs in the shared core file."""

    owner_id: str = "chat.attachments"
    activation_required: bool = True

    @staticmethod
    def _core():
        from tldw_chatbook.DB.recovery_core import core_adapters

        return next(
            adapter
            for adapter in core_adapters()
            if adapter.owner_id == "db.chachanotes.primary"
        )

    def discover(self, config):
        context = discovery_context(config)
        from .profile_paths import database_path

        path = database_path(config, "chachanotes_db_path")
        return (
            StorageItem(
                self.owner_id,
                storage_logical_id(context, self.owner_id),
                path,
                "included" if path.is_file() else "missing_required",
                (storage_logical_id(context, "db.chachanotes.primary"),),
                shared_group="shared:chachanotes:profile:" + context.profile_id,
            ),
        )

    def schema_policy(self):
        return replace(self._core().schema_policy(), owner=self.owner_id)

    def validate(self, candidate):
        return self._core().validate(candidate)

    def capture(self, item, destination, cancel):
        if item.owner != self.owner_id:
            raise ValueError("invalid_capture_item")
        return self._core().capture(
            replace(item, owner="db.chachanotes.primary"), destination, cancel
        )

    def relocate(self, candidate, mapping):
        return self._core().relocate(candidate, mapping)


def config_adapter() -> OwnerAdapter:
    return _Config("config", format="toml", max_bytes=_MAX_CONFIG_BYTES)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    """Installed config history and durable definition selectors only."""
    return (
        config_adapter(),
        _Attachments(),
        _Generated("generation.assets"),
        _Diagnostics("diagnostics.logs"),
        _CatalogCache("cache.model_catalog"),
        _InstanceLock("runtime.instance_lock"),
        _ChatbookScratch("runtime.chatbook_scratch"),
        _Definition(
            "chat.prompt_history", leaf="prompt_history.jsonl", participant_pending=True
        ),
        _Definition(
            "ui.state",
            leaf="ui_state.toml",
            location="config",
            participant_pending=True,
        ),
        _Definition(
            "ui.emoji_recents",
            leaf="recent_emojis.json",
            location="config",
            participant_pending=True,
        ),
        _Definition(
            "ui.themes",
            leaf="themes",
            location="config",
            tree=True,
            participant_pending=True,
        ),
        _ChatbookRegistry("chatbooks.registry", participant_pending=True),
        _Definition(
            "chatbooks.archives", leaf="chatbooks", tree=True, participant_pending=True
        ),
        _Definition(
            "tokenizers.custom",
            leaf="tokenizers",
            location="default_config",
            tree=True,
            participant_pending=True,
        ),
        _History("config.history", max_bytes=_MAX_CONFIG_BYTES),
        _Definition(
            "personas", leaf="tldw_chatbook_personas.json", participant_pending=True
        ),
        _Definition(
            "chat.dictionary_history",
            leaf="tldw_chatbook_chat_dictionary_history.json",
            participant_pending=True,
        ),
        _Definition(
            "chat.rag_context",
            leaf="tldw_chatbook_chat_rag_context.json",
            participant_pending=True,
        ),
        _Definition(
            "chat.grammars",
            leaf="tldw_chatbook_chat_grammars.json",
            participant_pending=True,
        ),
        _Definition(
            "feedback", leaf="tldw_chatbook_feedback.json", participant_pending=True
        ),
        _Definition(
            "audio.history",
            leaf="tldw_chatbook_audio_history.json",
            participant_pending=True,
        ),
        _Definition(
            "chat.dictionaries", leaf="chat_dicts", tree=True, participant_pending=True
        ),
        _Definition(
            "chunking.templates",
            leaf="chunking_templates",
            tree=True,
            participant_pending=True,
        ),
        _Definition(
            "notes.templates",
            leaf="note_templates.json",
            location="config",
            participant_pending=True,
        ),
        _Definition(
            "chat.prompts", leaf="Chat/prompt_templates", location="package", tree=True
        ),
        _Definition(
            "generation.styles",
            leaf="image_generation_styles",
            tree=True,
            participant_pending=True,
        ),
    )
