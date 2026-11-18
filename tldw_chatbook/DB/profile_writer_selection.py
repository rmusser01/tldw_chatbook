"""Recheck a stock Windows writer destination without reopening config on every write."""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from typing import NamedTuple

from tldw_chatbook.Backup_Recovery import profile_paths
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Utils.definition_records import definitions_current
from tldw_chatbook.Utils.platform_files import os


_NOTES = "db.chachanotes.primary"
_RUNS = "db.agent_runs"
_LEAVES = {
    _NOTES: profile_paths.database_leaf("chachanotes_db_path"),
    _RUNS: "agent_runs.db",
}


class _Selection(NamedTuple):
    source: tuple
    config: dict
    generation: int
    selector: Path
    stamp: tuple
    profile: Path
    notes: Path
    inputs: tuple
    posture: tuple

    def matches(self, other: _Selection) -> bool:
        return (
            self.source is other.source
            and self.config is other.config
            and self[2:] == other[2:]
        )


# Both repositories use the same two original config selectors. This holds
# immutable observations and the last published config object, never a lease.
_last: _Selection | None = None


def _source(config):
    source = config._SENSITIVE_DB_PATH_SOURCE
    module, life, paths, getters, definitions = source
    # Other database/settings modules may have their own injected selectors.
    # Only the selected ChaChaNotes path and its shared config dependencies
    # determine these two writer destinations.
    relevant = (
        entry
        for entry in definitions
        if entry[1] not in getters or entry[1] == "get_chachanotes_db_path"
    )
    if not (
        module is config is sys.modules.get("tldw_chatbook.config")
        and life is config._config_participants is sys.modules.get(life.__name__)
        and paths is config.profile_paths is life.profile_paths is profile_paths
        and paths is sys.modules.get(paths.__name__)
        and definitions_current(relevant)
    ):
        raise RecoveryRequired("profile_writer_config_source_changed")
    return source


def _stamp(selector):
    info = os.stat(selector, follow_symlinks=False)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _observe(config, source, selector, stamp):
    # The same finite desktop metadata probe used by get_user_data_dir:
    # every candidate directory component, fallback and root-selector lock.
    key, paths = config._user_data_dir_inputs()
    data, generation, config_source, folder, base = key
    if data is None or config_source != selector:
        raise RecoveryRequired("profile_writer_config_source_changed")
    posture = config._user_data_dir_stamps(paths)
    if posture is None:
        raise RecoveryRequired("profile_writer_directory_unavailable")
    profile = profile_paths.user_data_dir(data)
    notes = profile_paths.database_path(data, "chachanotes_db_path")
    if (
        config._CONFIG_CACHE is not data
        or config._CONFIG_GENERATION != generation
        or config._CONFIG_CACHE_SOURCE != selector
    ):
        raise RecoveryRequired("profile_writer_selection_changed")
    default_notes = config.DEFAULT_CONFIG_FROM_TOML.get("database", {}).get(
        "chachanotes_db_path"
    )
    return _Selection(
        source,
        data,
        generation,
        selector,
        stamp,
        profile,
        notes,
        (folder, base, default_notes, paths),
        posture,
    )


def _selected(profile: Path, owner: str, *, facts=None) -> list | None:
    global _last
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap

    if owner not in _LEAVES:
        raise ValueError("Unknown writer repository")
    source = _source(config)
    selector = profile_paths.effective_config_path()
    if config._get_effective_config_path() != selector:
        raise RecoveryRequired("profile_writer_selector_changed")
    stamp = _stamp(selector)
    previous = _last
    if previous is None or (previous.selector, previous.stamp) != (selector, stamp):
        # The normal CLI detects external edits on a throttle. Even a replacement
        # preserving size/mtime must invalidate this routing decision immediately.
        if not config._load_cli_config_bootstrap(force_reload=True).succeeded:
            raise RecoveryRequired("profile_writer_configuration_unavailable")
    current = _observe(config, source, selector, stamp)
    if previous is None or not previous.matches(current):
        # Full original selectors on every miss. Reuse only a stable observation
        # made on both sides of that call, under the original config participant.
        with config._config_participants.operation(config):
            resolved = config.get_user_data_dir(), config.get_chachanotes_db_path()
        if (
            resolved != (current.profile, current.notes)
            or _stamp(selector) != stamp
            or not current.matches(_observe(config, _source(config), selector, stamp))
        ):
            raise RecoveryRequired("profile_writer_selection_changed")
        _last = current
    if current.profile != profile or (
        current.notes != profile / _LEAVES[_NOTES]
        if owner == _NOTES
        else current.notes.parent != profile
    ):
        return None
    # The source's Windows posture probe pins and rechecks named associations
    # bottom-up. Apply the transport's private-directory policy to that same
    # observation, including every component leading to the selected leaf.
    chain = dict(zip(current.inputs[-1], current.posture, strict=True))
    uid = os.geteuid()
    for node in (*profile.parents, profile):
        info = chain.get(node)
        if (
            info is None
            or info[2] != stat.S_IFDIR
            or info[4] not in (0, uid)
            or (info[3] & 0o022 and not info[3] & stat.S_ISVTX)
        ):
            raise RecoveryRequired("profile_writer_directory_unsafe")
    device, inode, _, mode, owner_uid = chain[profile]
    if owner_uid != uid or mode & 0o077:
        raise RecoveryRequired("profile_writer_directory_unsafe")
    observed = [
        device,
        inode,
        os.path.normcase(selector),
        os.path.normcase(bootstrap.default_bootstrap_root()),
        list(stamp),
    ]
    if facts is not None and facts != observed:
        raise RecoveryRequired("profile_writer_selection_changed")
    return observed


def writer_repository_profile(path: Path, owner: str) -> Path | None:
    """Enroll the original concrete repository only at its selected stock file."""
    if (
        sys.platform != "win32"
        or Path(path).name.casefold() != _LEAVES[owner].casefold()
    ):
        return None
    from .profile_writer import in_profile_writer

    if in_profile_writer():
        return None
    path = profile_paths.lexical_path(path)
    config = sys.modules.get("tldw_chatbook.config")
    # Opening an explicit file must keep working independently of a foreign
    # config root. This prefilter supplies no authority to the writer.
    if config is None or path.parent != profile_paths.user_data_dir(
        config._CONFIG_CACHE or {}
    ):
        return None
    try:
        _source(config)
    except RecoveryRequired:
        # A custom/injected opener keeps its original direct repository. A
        # previously enrolled owner goes through require_writer_repository,
        # where the same change is a refusal before submission.
        return None
    return path.parent if _selected(path.parent, owner) else None


def require_writer_repository(profile: Path, owner: str, *, facts=None) -> list:
    """Return freshly checked native profile/selector facts within storage admission."""
    observed = _selected(profile, owner, facts=facts)
    if observed is None:
        raise RecoveryRequired("profile_writer_repository_selection_changed")
    return observed


def writer_file_identity(path: Path) -> tuple[int, int] | None:
    """Identify an existing, ordinary single-link SQLite file without following a link."""
    try:
        info = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RecoveryRequired("profile_writer_database_changed")
    return info.st_dev, info.st_ino
