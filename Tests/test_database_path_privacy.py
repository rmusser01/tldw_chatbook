from __future__ import annotations

import inspect
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_chatbook import config
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
)

DB_PATH_HELPERS = {
    "chachanotes_db_path": (
        config.get_chachanotes_db_path,
        "tldw_chatbook_ChaChaNotes.db",
    ),
    "prompts_db_path": (
        config.get_prompts_db_path,
        "tldw_chatbook_prompts.db",
    ),
    "media_db_path": (
        config.get_media_db_path,
        "tldw_chatbook_media_v2.db",
    ),
    "library_collections_db_path": (
        config.get_library_collections_db_path,
        "tldw_chatbook_library_collections.db",
    ),
    "library_ingest_jobs_db_path": (
        config.get_library_ingest_jobs_db_path,
        "tldw_chatbook_library_ingest_jobs.db",
    ),
    "workspaces_db_path": (
        config.get_workspaces_db_path,
        "tldw_chatbook_workspaces.db",
    ),
    "subscriptions_db_path": (
        config.get_subscriptions_db_path,
        "tldw_chatbook_subscriptions.db",
    ),
    "notifications_db_path": (
        config.get_notifications_db_path,
        "tldw_chatbook_notifications.db",
    ),
    "research_db_path": (
        config.get_research_db_path,
        "tldw_chatbook_research.db",
    ),
    "writing_db_path": (
        config.get_writing_db_path,
        "tldw_chatbook_writing.db",
    ),
    "scheduled_tasks_db_path": (
        config.get_scheduled_tasks_db_path,
        "tldw_chatbook_scheduled_tasks.db",
    ),
}


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _patch_data_dir_settings(
    monkeypatch: pytest.MonkeyPatch,
    configured_data_dir: Path | None,
) -> None:
    def fake_get_cli_setting(section: str, key: str, default=None):
        if section.lower() == "paths" and key == "data_dir":
            return (
                str(configured_data_dir) if configured_data_dir is not None else default
            )
        return default

    monkeypatch.setattr(config, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(config, "get_user_folder_name", lambda: "alice")


def test_default_data_base_ignores_xdg_and_uses_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    xdg_data_home = tmp_path / "xdg-data"
    home = tmp_path / "home"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data_home))
    monkeypatch.setenv("HOME", str(home))

    assert config._default_base_data_dir() == home / ".local" / "share" / "tldw_cli"


def test_default_data_base_falls_back_to_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))

    assert config._default_base_data_dir() == home / ".local" / "share" / "tldw_cli"


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_default_data_directory_is_private_under_permissive_umask(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    default_base = home / ".local" / "share" / "tldw_cli"
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, None)

    previous_umask = os.umask(0)
    try:
        selected = config.get_user_data_dir()
    finally:
        os.umask(previous_umask)

    assert selected == default_base / "alice"
    assert _mode(default_base) == 0o700
    assert _mode(selected) == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_existing_default_data_directories_are_hardened(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    default_base = home / ".local" / "share" / "tldw_cli"
    user_dir = default_base / "alice"
    user_dir.mkdir(parents=True)
    default_base.chmod(0o755)
    user_dir.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, None)

    assert config.get_user_data_dir() == user_dir
    assert _mode(default_base) == 0o700
    assert _mode(user_dir) == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
@pytest.mark.parametrize("ancestor", [".local", ".local/share"])
@pytest.mark.parametrize("mode", [0o775, 0o770, 0o777, 0o2775])
def test_fresh_data_root_recovers_without_changing_shared_ancestors(
    tmp_path, monkeypatch, ancestor, mode
):
    home = tmp_path / "home"
    shared = home / ancestor
    shared.mkdir(parents=True)
    shared.chmod(mode)
    original_mode = _mode(shared)  # Some filesystems strip setgid on chmod.
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, None)

    selected = config.get_user_data_dir()

    assert selected == home / ".tldw_cli-data" / "alice"
    assert _mode(selected.parent) == 0o700
    assert _mode(selected) == 0o700
    assert _mode(shared) == original_mode
    assert not (home / ".local/share/tldw_cli").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
@pytest.mark.parametrize("entry_kind", ["directory", "file", "dangling-symlink"])
def test_recovery_never_bypasses_existing_conventional_data(
    tmp_path, monkeypatch, entry_kind
):
    home = tmp_path / "home"
    conventional = home / ".local/share/tldw_cli"
    conventional.parent.mkdir(parents=True)
    if entry_kind == "directory":
        conventional.mkdir()
        (conventional / "saved-conversation").write_text("keep me")
    elif entry_kind == "file":
        conventional.write_text("keep me")
    else:
        conventional.symlink_to(home / "missing")
    conventional.parent.chmod(0o775)
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, None)

    with pytest.raises(PrivatePathError, match="shared_writable_parent"):
        config.get_user_data_dir()

    assert not (home / ".tldw_cli-data").exists()
    if entry_kind == "directory":
        assert (conventional / "saved-conversation").read_text() == "keep me"


def test_two_default_roots_require_explicit_selection(tmp_path, monkeypatch):
    home = tmp_path / "home"
    conventional = home / ".local/share/tldw_cli"
    fallback = home / ".tldw_cli-data"
    conventional.mkdir(parents=True)
    fallback.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, None)

    with pytest.raises(PrivatePathError, match="ambiguous_default_data_roots"):
        config.get_user_data_dir()

    assert not (conventional / "alice").exists()
    assert not (fallback / "alice").exists()
    _patch_data_dir_settings(monkeypatch, conventional)
    assert config.get_user_data_dir() == conventional / "alice"
    _patch_data_dir_settings(monkeypatch, fallback)
    assert config.get_user_data_dir() == fallback / "alice"


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
def test_recovery_rejects_a_fallback_symlink(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shared = home / ".local/share"
    shared.mkdir(parents=True)
    shared.chmod(0o775)
    outside = tmp_path / "outside"
    outside.mkdir()
    (home / ".tldw_cli-data").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, None)

    with pytest.raises(PrivatePathError):
        config.get_user_data_dir()

    assert list(outside.iterdir()) == []


def test_recovery_does_not_mask_unrelated_storage_errors(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, None)
    attempted = []
    failure = PrivatePathError(
        PrivatePathResult(
            home / ".local/share/tldw_cli",
            PrivatePathStatus.OPERATION_FAILED,
            reason="injected_io_error",
        )
    )

    def refuse(path, **kwargs):
        attempted.append(path)
        raise failure

    monkeypatch.setattr(config, "secure_private_directory", refuse)
    with pytest.raises(PrivatePathError) as caught:
        config.get_user_data_dir()

    assert caught.value is failure
    assert attempted == [home / ".local/share/tldw_cli"]


def test_recovery_does_not_treat_inaccessible_data_as_missing(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shared = home / ".local/share"
    shared.mkdir(parents=True)
    shared.chmod(0o775)
    conventional = shared / "tldw_cli"
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, None)
    original_lstat = Path.lstat

    def denied_lstat(path):
        if path == conventional:
            raise PermissionError("injected inaccessible data root")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", denied_lstat)
    with pytest.raises(PermissionError, match="inaccessible data root"):
        config.get_user_data_dir()

    assert not (home / ".tldw_cli-data").exists()


def test_data_root_probe_validates_before_accessing_the_filesystem(
    tmp_path, monkeypatch
):
    def unexpected_probe(path):
        pytest.fail("invalid environment-derived path reached lstat")

    monkeypatch.setattr(Path, "lstat", unexpected_probe)
    with pytest.raises(ValueError, match="dangerous pattern"):
        config._data_root_entry_exists(tmp_path / "unsafe;home" / "data")


@pytest.mark.skipif(os.name != "posix", reason="POSIX private lock contract")
def test_default_root_lock_is_private_stable_and_does_not_chmod_home(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o755)
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, None)

    first = config.get_user_data_dir()
    lock = home / ".tldw_cli-data-root.lock"
    identity = lock.stat().st_ino
    assert config.get_user_data_dir() == first
    assert lock.stat().st_ino == identity
    assert _mode(lock) == 0o600
    assert _mode(home) == 0o755


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow lock contract")
def test_default_root_selection_rejects_symlinked_lock(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("keep me")
    (home / ".tldw_cli-data-root.lock").symlink_to(outside)
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, None)

    with pytest.raises(PrivatePathError):
        config.get_user_data_dir()

    assert outside.read_text() == "keep me"
    assert not (home / ".local").exists()
    assert not (home / ".tldw_cli-data").exists()


def test_explicit_root_does_not_create_a_default_selection_lock(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    custom = tmp_path / "custom"
    custom.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _patch_data_dir_settings(monkeypatch, custom)

    assert config.get_user_data_dir() == custom / "alice"
    assert list(home.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX fresh-install contract")
def test_fresh_config_import_and_database_survive_restart_and_permission_repair(
    tmp_path,
):
    home = tmp_path / "home"
    shared = home / ".local/share"
    shared.mkdir(parents=True)
    shared.chmod(0o775)
    # Exercise default config bootstrap too, with no inherited test override.
    env = dict(os.environ, HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"))
    env.pop("TLDW_CONFIG_PATH", None)
    env.pop("XDG_DATA_HOME", None)
    script = """
import sys
from tldw_chatbook import config
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
path = config.get_chachanotes_db_path()
assert path.parent.parent.name == '.tldw_cli-data', path
db = connect_private_sqlite('db.chachanotes.primary', path)
try:
    if sys.argv[1] == 'create':
        with db:
            db.execute('CREATE TABLE recovery_probe (value TEXT)')
            db.execute('INSERT INTO recovery_probe VALUES (?)', ('survived restart',))
    assert db.execute('SELECT value FROM recovery_probe').fetchone() == ('survived restart',)
finally:
    db.close()
print('RECOVERY_OK')
"""
    for phase in ("create", "restart", "repaired"):
        if phase == "repaired":
            shared.chmod(0o755)
        result = subprocess.run(
            [sys.executable, "-c", script, phase],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=40,
        )
        assert result.returncode == 0, result.stderr
        assert "RECOVERY_OK" in result.stdout
        assert _mode(shared) == (0o755 if phase == "repaired" else 0o775)

    fallback = home / ".tldw_cli-data"
    assert _mode(fallback) == 0o700
    assert _mode(home / ".config/tldw_cli/config.toml") == 0o600
    databases = list(fallback.glob("*/tldw_chatbook_ChaChaNotes.db"))
    assert len(databases) == 1
    assert _mode(databases[0]) == 0o600
    assert not (shared / "tldw_cli").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_default_data_directory_rejects_intermediate_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HOME", str(alias))
    _patch_data_dir_settings(monkeypatch, None)

    with pytest.raises(PrivatePathError):
        config.get_user_data_dir()

    assert not (target / ".local").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_default_data_directory_rejects_unsafe_intermediate_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    unsafe.chmod(0o777)
    monkeypatch.setenv("HOME", str(unsafe))
    _patch_data_dir_settings(monkeypatch, None)

    with pytest.raises(PrivatePathError):
        config.get_user_data_dir()

    assert not (unsafe / ".local").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_custom_data_base_mode_is_preserved_and_only_user_child_is_secured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custom_base = tmp_path / "shared-base"
    custom_base.mkdir()
    custom_base.chmod(0o751)
    _patch_data_dir_settings(monkeypatch, custom_base)

    selected = config.get_user_data_dir()

    assert selected == custom_base / "alice"
    assert _mode(custom_base) == 0o751
    assert _mode(selected) == 0o700


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
@pytest.mark.parametrize("case", ["missing", "unsafe", "symlink"])
def test_custom_data_base_must_be_existing_trusted_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
) -> None:
    if case == "missing":
        custom_base = tmp_path / "missing"
    elif case == "unsafe":
        custom_base = tmp_path / "unsafe"
        custom_base.mkdir()
        custom_base.chmod(0o777)
    else:
        target = tmp_path / "target"
        target.mkdir()
        custom_base = tmp_path / "alias"
        custom_base.symlink_to(target, target_is_directory=True)
    _patch_data_dir_settings(monkeypatch, custom_base)

    with pytest.raises(PrivatePathError):
        config.get_user_data_dir()

    assert not (custom_base / "alice").exists()


@pytest.mark.skipif(os.name != "posix", reason="symlink identity contract")
@pytest.mark.parametrize("setting_name", DB_PATH_HELPERS)
def test_custom_database_paths_preserve_lexical_symlink_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    setting_name: str,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    helper, filename = DB_PATH_HELPERS[setting_name]
    (target / filename).touch()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    custom_path = alias / filename
    original_exists = Path.exists
    original_resolve = Path.resolve
    exists_calls: list[Path] = []
    resolve_calls: list[Path] = []

    def record_exists(path: Path) -> bool:
        exists_calls.append(path)
        return original_exists(path)

    def record_resolve(path: Path, *args, **kwargs) -> Path:
        resolve_calls.append(path)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            str(custom_path)
            if section == "database" and key == setting_name
            else default
        ),
    )
    monkeypatch.setattr(Path, "exists", record_exists)
    monkeypatch.setattr(Path, "resolve", record_resolve)

    selected = helper()

    assert exists_calls == []
    assert resolve_calls == []
    assert isinstance(selected, Path)
    assert selected == Path(os.path.abspath(custom_path))
    assert selected.parent == alias
    assert selected != target / filename


@pytest.mark.parametrize("setting_name", DB_PATH_HELPERS)
def test_default_database_paths_are_direct_children_of_secured_user_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    setting_name: str,
) -> None:
    helper, filename = DB_PATH_HELPERS[setting_name]
    user_dir = tmp_path / "user-data"
    user_dir.mkdir()
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: default,
    )
    monkeypatch.setattr(config, "get_user_data_dir", lambda: user_dir)

    selected = helper()

    assert isinstance(selected, Path)
    assert selected == user_dir / filename
    assert selected.parent == user_dir


@pytest.mark.parametrize("dangerous", ["/tmp/chatbook.db;touch-pwned", "bad\x00.db"])
def test_custom_database_paths_keep_dangerous_input_validation(
    monkeypatch: pytest.MonkeyPatch,
    dangerous: str,
) -> None:
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            dangerous
            if section == "database" and key == "notifications_db_path"
            else default
        ),
    )

    with pytest.raises(ValueError):
        config.get_notifications_db_path()


def test_scheduled_tasks_path_retains_raw_home_shorthand_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            "~/scheduled.db"
            if section == "database" and key == "scheduled_tasks_db_path"
            else default
        ),
    )

    with pytest.raises(ValueError, match="dangerous pattern"):
        config.get_scheduled_tasks_db_path()


def test_load_settings_does_not_create_unconsumed_server_database_parents() -> None:
    source = inspect.getsource(config.load_settings)

    assert "main_db_file_path_server.parent.mkdir" not in source
    assert "user_data_base_dir_server.mkdir" not in source


def test_prompts_startup_does_not_duplicate_database_parent_creation() -> None:
    app_source = (Path(config.__file__).parent / "app.py").read_text(encoding="utf-8")

    assert "prompts_db_path.parent.mkdir(parents=True, exist_ok=True)" not in app_source
