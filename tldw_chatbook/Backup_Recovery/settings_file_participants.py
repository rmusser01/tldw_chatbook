"""Exact settings/definition source selectors; no caller-supplied owner authority."""

import sys
from datetime import datetime
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap
from .profile_paths import lexical_path

ROUTES = {
    "eval_config",
    "note_templates",
    "theme_directory",
    "theme_file",
    "theme_export",
    "pet",
    "runtime_state",
    "runtime_read",
}


def _class(module, name):
    return getattr(sys.modules.get("tldw_chatbook." + module), name, None)


def binding(source):
    """Return a source's actual selected durable root and installation posture."""
    cls = _class("Evals.config_loader", "EvalConfigLoader")
    if cls is not None and isinstance(source, cls):
        from ..Evals import _default_config_path

        selected = lexical_path(source.config_path)
        return (
            "eval.definitions",
            selected,
            type(source) is cls and selected == lexical_path(_default_config_path()),
        )
    module = sys.modules.get("tldw_chatbook.Notes.template_store")
    if module is not None and source is module:
        from ..config import _get_effective_config_path

        return (
            "notes.templates",
            lexical_path(_get_effective_config_path().parent / "note_templates.json"),
            True,
        )
    cls = _class("Widgets.settings_theme_editor", "SettingsThemeEditor")
    if cls is not None and isinstance(source, cls):
        from ..config import _get_effective_config_path

        selected = lexical_path(source.custom_themes_path)
        return (
            "ui.themes",
            selected,
            type(source) is cls
            and selected
            == lexical_path(_get_effective_config_path().parent / "themes"),
        )
    cls = _class("Widgets.Tamagotchi.tamagotchi_storage", "JSONStorage")
    if cls is not None and isinstance(source, cls):
        installed_cls = _class(
            "Widgets.Tamagotchi.tamagotchi_storage", "ConfigFileStorage"
        )
        root = (
            Path(os.environ.get("APPDATA", "~"))
            if os.name == "nt"
            else Path("~/.config")
        )
        selected = lexical_path(source.filepath)
        return (
            "tamagotchi.config",
            selected,
            type(source) is installed_cls
            and selected
            == lexical_path(root / "tldw_chatbook" / "tamagotchi_pets.json"),
        )
    cls = _class("runtime_policy.source_state", "RuntimeSourceStateStore")
    if cls is not None and isinstance(source, cls):
        from ..runtime_policy.bootstrap import default_runtime_policy_path
        from ..Utils import private_paths

        selected = lexical_path(source.path)
        return (
            "runtime.source_state",
            selected,
            type(source) is cls
            and selected == lexical_path(default_runtime_policy_path())
            and private_paths._posix_guards_available()
            and private_paths._atomic_posix_guards_available(),
        )
    return None


def selection(source, route, target):
    from . import raw_participants as raw

    bound = binding(source)
    if bound is None:
        raise bootstrap.RecoveryRequired("raw_source_not_supported")
    owner, selected, installed = bound
    participant = raw._source_participants.get(source)
    if participant is not None:
        previous = raw._participants[participant]
        if not installed or previous.selected != selected:
            raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    expected = {
        "eval_config": "eval.definitions",
        "note_templates": "notes.templates",
        "theme_directory": "ui.themes",
        "theme_file": "ui.themes",
        "theme_export": "ui.themes",
        "pet": "tamagotchi.config",
        "runtime_state": "runtime.source_state",
        "runtime_read": "runtime.source_state",
    }
    if expected.get(route) != owner:
        raise bootstrap.RecoveryRequired("raw_source_not_supported")
    if route == "theme_directory":
        return selected, installed, True
    if route == "theme_file":
        target = lexical_path(target)
        if target.parent != selected or target.suffix != ".toml":
            raise bootstrap.RecoveryRequired("raw_path_outside_scope")
        return target, installed, False
    if route in {"theme_export", "eval_config"} and target is not None:
        target = lexical_path(target)
        return target, installed and target == selected, False
    if target is not None and lexical_path(target) != selected:
        raise bootstrap.RecoveryRequired("raw_source_selection_changed")
    return selected, installed, False


def preflight(state, route, attempt):
    """Freeze only theme TOMLs or exact timestamp pet backups under a held pin."""
    import re

    from . import storage_admission as storage

    directory = state.selected if route == "theme_directory" else state.selected.parent
    if directory.exists():
        fd = state.pins.get(directory)
        names = os.listdir(fd if state.pinned else directory)
        pattern = (
            re.compile(r".+\.toml")
            if route == "theme_directory"
            else re.compile(
                re.escape(state.selected.stem) + r"\.backup_\d{8}_\d{6}\.json"
            )
        )
        members = sorted(name for name in names if pattern.fullmatch(name))
        if len(members) > 10000:
            raise bootstrap.RecoveryRequired("raw_membership_limit")
        for name in members:
            path = directory / name
            info = (
                os.stat(name, dir_fd=fd, follow_symlinks=False)
                if state.pinned
                else os.stat(path, follow_symlinks=False)
            )
            import stat

            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise bootstrap.RecoveryRequired("raw_not_regular")
            state.observed_files[path] = (info.st_dev, info.st_ino)
    members = tuple(state.observed_files)
    if route == "pet" and state.writing:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state.backup = state.selected.with_suffix(f".backup_{stamp}.json")
        state.backup_temporary = state.backup.with_suffix(state.backup.suffix + ".tmp")
        if state.backup not in members:
            members += (state.backup,)
        members += (state.backup_temporary,)
    for path in members:
        attempt.check()
        state.leases.append(storage.acquire_storage(path))
        state.holds.append(storage._holds.get(state.leases[-1]._key))
        attempt.check()
    state.paths += members


def check_members(state):
    for path, identity in state.observed_files.items():
        info = (
            os.stat(path.name, dir_fd=state.pins[path.parent], follow_symlinks=False)
            if state.pinned
            else os.stat(path, follow_symlinks=False)
        )
        if (info.st_dev, info.st_ino) != identity:
            raise bootstrap.RecoveryRequired("raw_entry_identity_changed")


def pet_operation(function):
    """Bind only JSONStorage's actual methods and its inherited recovery wrapper."""
    from functools import wraps

    from . import raw_participants as raw

    @wraps(function)
    def wrapped(source, *args, **kwargs):
        cls = _class("Widgets.Tamagotchi.tamagotchi_storage", "JSONStorage")
        if cls is None or not isinstance(source, cls):
            # StorageAdapter's wrapper is also inherited by memory/SQLite stores.
            return function(source, *args, **kwargs)
        owner_cls = _class(
            "Widgets.Tamagotchi.tamagotchi_storage", function.__qualname__.split(".")[0]
        )
        if (
            owner_cls is None
            or owner_cls.__dict__.get(function.__name__) is not wrapped
        ):
            raise bootstrap.RecoveryRequired("raw_source_not_supported")
        writing = function.__name__ not in {"_read_data", "load", "list_pets"}
        with raw._scope(source, "pet", writing=writing):
            return function(source, *args, **kwargs)

    return wrapped
