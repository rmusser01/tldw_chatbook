"""Real config source/cache/private resource lifetimes under maintenance."""

import importlib.util
import sys

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from Tests.Backup_Recovery.test_participant_lifetimes import local_root
from Tests.Backup_Recovery.test_admission import launch
from Tests.private_profile import private_profile_test


@pytest.fixture
def source(tmp_path, monkeypatch, local_root):
    import tldw_chatbook
    from tldw_chatbook import config

    target = tmp_path / "config.toml"
    target.write_text('[general]\nusers_name = "configured"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    spec = importlib.util.spec_from_file_location(config.__name__, config.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setattr(tldw_chatbook, "config", module)
    spec.loader.exec_module(module)
    return module


def test_pause_refuses_cached_bootstrap_without_erasing_cache(source):
    # 5e56aac410 (TASK-32804.1) serves a warm read ahead of the admission
    # handshake, so a pause no longer refuses the public warm hit; the guarded
    # bootstrap behind it still refuses, and neither erases the cache.
    before = source._CONFIG_CACHE
    assert before is not None
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired):
            source._load_cli_config_bootstrap()
        assert source._CONFIG_CACHE is before
        assert source.load_cli_config_and_ensure_existence() is before
        assert source._CONFIG_CACHE is before
    finally:
        pause.resume()


def test_config_serialization_has_actual_source_lifetime(source, monkeypatch):
    original = source.toml.dumps
    observed = []

    def serialize(*args, **kwargs):
        observed.append(bool(storage._raw_operations))
        return original(*args, **kwargs)

    monkeypatch.setattr(source.toml, "dumps", serialize)
    source.replace_cli_config({"general": {"users_name": "saved"}})
    assert observed and all(observed)


def test_refused_force_reload_keeps_last_usable_cache(source):
    before = source._CONFIG_CACHE
    generation = source._CONFIG_GENERATION
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired):
            source.load_cli_config_and_ensure_existence(force_reload=True)
        assert source._CONFIG_CACHE is before
        assert source._CONFIG_GENERATION == generation
    finally:
        pause.resume()


def test_backup_snapshot_revision_and_encryption_share_config_owner(source):
    before = source.get_cli_config_path().read_bytes()
    loaded, backup = source.replace_cli_config_serialized(
        '[general]\nusers_name = "next"\n'
    )
    assert loaded["general"]["users_name"] == "next"
    assert backup.read_bytes() == before
    assert source.read_cli_config_backup_serialized().encode() == before
    snapshot = source.export_cli_config_snapshot(timestamp="20260908_120000")
    assert snapshot.read_bytes() == source.get_cli_config_path().read_bytes()
    first = source.replace_revisioned_settings_section_to_cli_config(
        "speech_studio", {"revision": 1}, expected_revision=0
    )
    conflict = source.replace_revisioned_settings_section_to_cli_config(
        "speech_studio", {"revision": 1}, expected_revision=0
    )
    assert first.caches_reloaded
    assert conflict.conflict
    assert source.enable_config_encryption("private-test-password")
    assert source.change_encryption_password(
        "private-test-password", "other-test-password"
    )
    assert source.disable_config_encryption("other-test-password")
    assert source.persist_cli_config_for_shutdown()


def test_config_failed_write_retains_generation_cache_and_error(source, monkeypatch):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Utils import private_paths

    cache = source._CONFIG_CACHE
    generation = source._CONFIG_GENERATION
    original = private_paths.os.write

    def fail(fd, payload):
        if raw.runtime_operation() is not None:
            raise OSError("write failed")
        return original(fd, payload)

    monkeypatch.setattr(private_paths.os, "write", fail)
    assert not source.save_setting_to_cli_config("general", "users_name", "failed")
    assert source._CONFIG_CACHE is cache
    assert source._CONFIG_GENERATION == generation
    assert source._CONFIG_PERSISTENCE_ERROR is not None
    source.load_cli_config_and_ensure_existence()
    assert source._CONFIG_PERSISTENCE_ERROR is not None


def test_config_selector_change_cannot_demote_installed_source(
    source, tmp_path, monkeypatch
):
    other = tmp_path / "elsewhere.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(other))
    with pytest.raises(bootstrap.RecoveryRequired, match="selection_changed"):
        source.replace_cli_config({"general": {"users_name": "escaped"}})
    assert not other.exists()


def test_pause_during_serialization_refuses_derived_effects_and_publication(
    source, monkeypatch
):
    original = source.toml.dumps
    pauses = []
    cache = source._CONFIG_CACHE
    generation = source._CONFIG_GENERATION

    def serialize(*args, **kwargs):
        result = original(*args, **kwargs)
        if not pauses:
            pauses.append(storage._begin_local_pause())
        return result

    monkeypatch.setattr(source.toml, "dumps", serialize)
    try:
        result = source.apply_settings_mutation_to_cli_config(
            {"general": {"users_name": "new-directory"}}
        )
        assert result.file_replaced and not result.caches_reloaded
        assert source._CONFIG_GENERATION == generation
        assert source._CONFIG_CACHE is cache
        assert "new-directory" in source.get_cli_config_path().read_text()
        assert not (source._default_base_data_dir() / "new-directory").exists()
    finally:
        for pause in reversed(pauses):
            pause.resume()


def test_real_interprocess_lock_wait_cancels_and_keeps_lock_identity(
    source, monkeypatch
):
    import subprocess
    import threading
    import time
    from Tests.Backup_Recovery.test_admission import line

    lock = source.get_cli_config_path().with_suffix(".toml.lock")
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            'import portalocker,sys; f=open(sys.argv[1], "a"); portalocker.lock(f, portalocker.LOCK_EX); print("locked",flush=True); sys.stdin.readline(); f.close()',
            str(lock),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert line(child) == "locked"
    identity = lock.stat().st_ino
    attempting = threading.Event()
    finished = threading.Event()
    original = source.portalocker.lock
    results = []

    def acquire(*args, **kwargs):
        attempting.set()
        return original(*args, **kwargs)

    def save():
        try:
            results.append(
                source.save_setting_to_cli_config("general", "users_name", "waited")
            )
        finally:
            finished.set()

    monkeypatch.setattr(source.portalocker, "lock", acquire)
    worker = threading.Thread(target=save)
    worker.start()
    pause = None
    try:
        assert attempting.wait(3)
        assert not finished.wait(0.05)
        pause = storage._begin_local_pause()
        assert finished.wait(3)
        worker.join()
        assert results == [False]
        assert pause.drain(time.monotonic() + 1)
        assert lock.stat().st_ino == identity
        assert "waited" not in source.get_cli_config_path().read_text()
    finally:
        if pause is not None:
            pause.resume()
        child.stdin.write("exit\n")
        child.stdin.flush()
        child.wait(timeout=5)
        worker.join(5)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()
    assert not worker.is_alive()
    assert source.save_setting_to_cli_config("general", "users_name", "resumed")
    assert lock.stat().st_ino == identity


_CONFIG_NATIVE_CHILD = r"""
import inspect, os, sys, time, tomllib
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw, storage_admission as storage
root, selected, failure = sys.argv[1:]
root, selected = Path(root), Path(selected)
bootstrap.default_bootstrap_root = lambda: root
os.environ['TLDW_CONFIG_PATH'] = str(selected)
selected.write_text('[general]\nusers_name="native"\n')
from tldw_chatbook import config
from tldw_chatbook.Utils import private_paths as private
assert not storage._raw_operations and not storage._pending_acquisitions
# Diagnostic isolation: remove only this child's known successful startup hold.
startup = storage._startups.pop((os.getpid(), str(root)))
startup.close()
cache, generation = config._CONFIG_CACHE, config._CONFIG_GENERATION
original_close, original_open, original_write, original_fdopen = os.close, private._native_open, os.write, os.fdopen
original_load = tomllib.load
fd_target = None
class FailingStream:
    def __init__(self, stream): self.stream = stream
    def __getattr__(self, name): return getattr(self.stream, name)
    def close(self):
        if failure.endswith('after'): self.stream.close()
        raise OSError('native stream close uncertainty')
def opened(*args, **kwargs):
    global fd_target
    fd = original_open(*args, **kwargs)
    direct_body = failure == 'direct_parent' and any(frame.function == '_prepare_config_parent' for frame in inspect.stack())
    if raw.runtime_operation() is not None or direct_body:
        if (failure == 'ancestor' or direct_body) and args[0] == '/': fd_target = fd
        if failure.startswith('lock_') and str(args[0]).endswith('.lock'): fd_target = fd
    return fd
def writing(fd, payload):
    global fd_target
    if failure.startswith('temp_') and raw.runtime_operation() is not None: fd_target = fd
    return original_write(fd, payload)
def loading(stream, *args, **kwargs):
    global fd_target
    if failure == 'read': fd_target = stream.fileno()
    return original_load(stream, *args, **kwargs)
def fdopen(fd, mode, *args, **kwargs):
    global fd_target
    stream = original_fdopen(fd, mode, *args, **kwargs)
    if failure.startswith('stream_') and mode == 'a':
        fd_target = fd
        return FailingStream(stream)
    return stream
def close(fd):
    global fd_target
    if failure.startswith('final_parent') and config._CONFIG_GENERATION > generation:
        if any(fd in state.pins.values() for state in raw._states.values()): fd_target = fd
    if fd == fd_target and not failure.startswith('stream_'):
        if failure.endswith('after'): original_close(fd)
        raise OSError('native fd close uncertainty')
    return original_close(fd)
private._native_open, os.close, os.write, os.fdopen, tomllib.load = opened, close, writing, fdopen, loading
try:
    if failure == 'read': config.load_cli_config_and_ensure_existence(force_reload=True)
    elif failure == 'direct_parent': config._prepare_config_parent(selected)
    else: config.replace_cli_config({'general': {'users_name': 'changed'}})
except Exception: pass
else: raise AssertionError('native uncertainty reported success')
os.close, private._native_open, os.write, os.fdopen, tomllib.load = original_close, original_open, original_write, original_fdopen, original_load
assert config._CONFIG_GENERATION == generation
assert config._CONFIG_CACHE is cache
assert config._CONFIG_PERSISTENCE_ERROR is not None
pause = storage._begin_local_pause()
assert not pause.drain(time.monotonic() + .05)
assert any(fd_target in state.descriptors for state in raw._states.values())
if not failure.endswith('after'): os.fstat(fd_target)
print('held', flush=True)
sys.stdin.readline()
"""


@pytest.mark.parametrize(
    "failure",
    [
        "ancestor",
        "direct_parent",
        "read",
        "temp_before",
        "temp_after",
        "lock_before",
        "lock_after",
        "stream_before",
        "stream_after",
        "final_parent_before",
        "final_parent_after",
    ],
)
def test_config_native_uncertainty_blocks_independent_maintenance(
    tmp_path, local_root, launch, failure
):
    import select
    import subprocess
    from Tests.Backup_Recovery.test_admission import line, release
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        UNBOUND_NAMESPACE,
    )

    authority = admission_authority(local_root)
    log = (tmp_path / "native-child.log").open("w+")
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            _CONFIG_NATIVE_CHILD,
            str(local_root),
            str(tmp_path / "native.toml"),
            failure,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
    )
    try:
        try:
            assert line(child) == "held"
        except AssertionError:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            log.seek(0)
            pytest.fail(log.read())
        observer = launch(authority.control_root, "maintenance", (UNBOUND_NAMESPACE,))
        assert not select.select([observer.stdout], [], [], 0.1)[0]
        child.stdin.write("exit\n")
        child.stdin.flush()
        child.wait(timeout=5)
        log.seek(0)
        assert child.returncode == 0, log.read()
        assert line(observer) == "entered"
        release(observer)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()
        log.close()


def test_default_bootstrap_creates_private_parent_and_pause_preserves_it(
    tmp_path, monkeypatch, local_root
):
    import stat
    from Tests.Backup_Recovery.config_test_support import install_config_source

    home = tmp_path / "first-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    source = install_config_source(monkeypatch)
    target = home / ".config" / "tldw_cli" / "config.toml"
    assert source.first_profile_created_this_session()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    original = target.read_bytes()
    pause = storage._begin_local_pause()
    try:
        assert not source.enable_config_encryption("paused-password")
        with pytest.raises(bootstrap.RecoveryRequired):
            source.export_cli_config_snapshot(timestamp="20260908_130000")
        assert target.read_bytes() == original
        assert not target.with_suffix(".toml.lock").exists()
    finally:
        pause.resume()


def test_custom_missing_parent_is_not_created(tmp_path, monkeypatch, local_root):
    from Tests.Backup_Recovery.config_test_support import install_config_source
    from tldw_chatbook.Utils.private_paths import PrivatePathError

    target = tmp_path / "unowned-parent" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    with pytest.raises(PrivatePathError):
        install_config_source(monkeypatch)
    assert not target.parent.exists()


def test_snapshot_path_validation_precedes_effects(source, tmp_path):
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError):
        source.export_cli_config_snapshot(timestamp="../../escape")
    assert set(tmp_path.rglob("*")) == before


def test_import_into_paused_process_has_no_config_effect(
    tmp_path, monkeypatch, local_root
):
    from Tests.Backup_Recovery.config_test_support import install_config_source

    target = tmp_path / "paused.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired):
            install_config_source(monkeypatch)
        assert not target.exists()
    finally:
        pause.resume()


@pytest.mark.parametrize("windows_fallback", [False, True])
def test_first_portable_config_source_keeps_ordinary_persistence(
    tmp_path, monkeypatch, local_root, windows_fallback
):
    from Tests.Backup_Recovery.config_test_support import install_config_source
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    selected = tmp_path / "portable.toml"
    selected.write_text('[general]\nusers_name="portable"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))
    monkeypatch.setattr(raw, "_pinned_io_available", lambda: False)
    if windows_fallback:
        from tldw_chatbook.Utils import private_paths

        monkeypatch.setattr(private_paths, "_posix_guards_available", lambda: False)
        monkeypatch.setattr(
            private_paths, "_atomic_posix_guards_available", lambda: False
        )
        monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)
    source = install_config_source(monkeypatch)
    assert source.save_setting_to_cli_config("general", "users_name", "ordinary")
    assert "ordinary" in selected.read_text()
    with pytest.raises(bootstrap.RecoveryRequired, match="not_installed"):
        raw._raw_participant(source)


def test_installed_config_cannot_demote_after_native_change(source, monkeypatch):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    original = source.get_cli_config_path().read_bytes()
    monkeypatch.setattr(raw, "_pinned_io_available", lambda: False)
    assert not source.save_setting_to_cli_config("general", "users_name", "demoted")
    assert source.get_cli_config_path().read_bytes() == original


def test_private_directory_verifier_cannot_widen_config_operation(source, tmp_path):
    from tldw_chatbook.Utils.private_paths import verify_trusted_directory

    outside = tmp_path / "unrelated"
    outside.mkdir(mode=0o700)
    with source._config_write_lock(source.get_cli_config_path()):
        with pytest.raises(RuntimeError, match="helper_not_supported"):
            verify_trusted_directory(outside, allow_shared_sticky=False)


def test_two_config_processes_preserve_each_others_keys(source, tmp_path, local_root):
    import select
    import subprocess
    from Tests.Backup_Recovery.test_admission import line

    program = r"""
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
from tldw_chatbook import config
mode = sys.argv[2]
if mode == 'first':
    # A save serializes twice under the lock (the expected round trip, then
    # the write); hold the first only.
    original = config.toml.dumps
    held = []
    def serialize(*args, **kwargs):
        value = original(*args, **kwargs)
        if not held:
            held.append(True)
            print('serializing', flush=True)
            sys.stdin.readline()
        return value
    config.toml.dumps = serialize
else:
    original = config.portalocker.lock
    announced = False
    def locking(*args, **kwargs):
        global announced
        if not announced:
            print('attempting', flush=True)
            announced = True
        return original(*args, **kwargs)
    config.portalocker.lock = locking
assert config.save_setting_to_cli_config('independent', mode, mode)
print('saved', flush=True)
"""
    children = []
    logs = []
    try:
        for mode in ("first", "second"):
            log = (tmp_path / f"{mode}.log").open("w+")
            logs.append(log)
            child = subprocess.Popen(
                [sys.executable, "-u", "-c", program, str(local_root), mode],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
            )
            children.append(child)
            assert line(child) == ("serializing" if mode == "first" else "attempting")
        first, second = children
        assert not select.select([second.stdout], [], [], 0.1)[0]
        first.stdin.write("continue\n")
        first.stdin.flush()
        assert line(first) == "saved"
        assert line(second) == "saved"
        for child in children:
            assert child.wait(timeout=5) == 0
        import tomllib

        assert tomllib.loads(source.get_cli_config_path().read_text())[
            "independent"
        ] == {"first": "first", "second": "second"}
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()
        for log in logs:
            log.close()


def test_reentrant_publication_from_derived_scope_refuses_all_effects(
    source, monkeypatch
):
    from pathlib import Path

    directory = source.get_model_cache_dir()
    directory.rmdir()
    before = source.get_cli_config_path().read_bytes()
    cache = source._CONFIG_CACHE
    generation = source._CONFIG_GENERATION
    original = Path.mkdir

    def reenter(path, *args, **kwargs):
        if path == directory:
            source.replace_cli_config({"general": {"users_name": "reentrant"}})
        return original(path, *args, **kwargs)

    def reenter_secure(path, *args, **kwargs):
        if Path(path) == directory:
            source.replace_cli_config({"general": {"users_name": "reentrant"}})
        return secure(path, *args, **kwargs)

    # 6ae1ddd69f (task-32901) creates the default cache directory through
    # secure_private_directory, which makes it natively on POSIX: re-enter
    # there too, so the publication still fires inside the derived scope.
    secure = source.secure_private_directory
    monkeypatch.setattr(Path, "mkdir", reenter)
    monkeypatch.setattr(source, "secure_private_directory", reenter_secure)
    with pytest.raises(bootstrap.RecoveryRequired, match="raw_path_outside_scope"):
        source.get_model_cache_dir()
    assert not directory.exists()
    assert source.get_cli_config_path().read_bytes() == before
    assert source._CONFIG_CACHE is cache
    assert source._CONFIG_GENERATION == generation


# TASK-42000: a Replace can retire a file the profile enrolled as its own
# registered root (a settings file beside the config). Every absent root must
# hold nothing: it never grants anything beneath it, and it never breaks the
# next start's data-directory check.


@pytest.mark.usefixtures("local_root")
def test_an_absent_registered_root_grants_nothing_beneath_it(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    root = bootstrap.default_bootstrap_root()
    owned = tmp_path / "owned"
    owned.mkdir(mode=0o700)
    selector = tmp_path / "selected.toml"
    selector.write_text('[general]\nusers_name="fixture"\n')
    authority = admission_authority(root)
    authority.register("core", (owned, selector))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    bind_profile(root, selector, ("core",), root / "admission")
    owned.rmdir()
    try:
        contained = storage._contains_capture_path((owned,), owned / "x")
    except OSError as error:
        contained = error
    assert contained is False
    with pytest.raises(bootstrap.RecoveryRequired):
        storage.acquire_storage(owned / "x")


def _registered_file(tmp_path, name="backup.source.file"):
    """A file registered as a root of its own, as a backup registers one."""
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    files = tmp_path / "files"
    files.mkdir(mode=0o700)
    registered = files / "copy.json"
    registered.write_text("{}")
    registered.chmod(0o600)
    authority = admission_authority(bootstrap.default_bootstrap_root())
    authority.register(name, (registered,))
    return authority, files, registered


@pytest.mark.usefixtures("local_root")
@pytest.mark.parametrize(
    "gone",
    [
        "file",
        "folder_root",
        "file_and_its_folder",
        "file_folder_replaced",
        "file_under_mount_point",
        "file_without_evidence",
    ],
)
def test_a_deleted_file_root_holds_nothing_but_a_missing_folder_still_refuses(
    tmp_path, monkeypatch, gone
):
    """TASK-42115: a backup registers single files as roots of their own. Once
    one is deleted, admission treats it as holding nothing while the folder it
    was registered in is still that folder. A missing folder root, a file root
    whose folder is gone or replaced or is a mount point (an unmounted drive),
    or a file registered without its folder's evidence (before TASK-42115)
    still refuses every admission."""
    from tldw_chatbook.Backup_Recovery import effective_roots as proof

    if gone == "file_without_evidence":
        monkeypatch.setattr(proof, "file_root_tokens", lambda roots: set())
    authority, files, registered = _registered_file(tmp_path)
    folder = tmp_path / "folder"
    folder.mkdir(mode=0o700)
    selector = tmp_path / "selected.toml"
    selector.write_text('[general]\nusers_name="fixture"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    authority.register("backup.source.folder", (folder,))
    if gone == "folder_root":
        folder.rmdir()
    else:
        registered.unlink()
    if gone == "file_under_mount_point":
        monkeypatch.setattr(proof, "ismount", lambda path: path == files)
    if gone in {"file_and_its_folder", "file_folder_replaced"}:
        files.rename(tmp_path / "moved")
    if gone == "file_folder_replaced":
        files.mkdir(mode=0o700)
    if gone == "file":
        with storage.acquire_storage(tmp_path / "app.db"):
            pass
        return
    with pytest.raises(
        bootstrap.RecoveryRequired, match="storage_admission_unavailable"
    ):
        with storage.acquire_storage(tmp_path / "app.db"):
            pass


@pytest.mark.usefixtures("local_root")
def test_the_refused_start_names_a_missing_folder_root_but_not_a_deleted_file_root(
    tmp_path, monkeypatch
):
    """TASK-34100.16.1, ported onto TASK-42115 (2026-11-23): a refused start
    lists the registered locations that refuse it. A deleted file root refuses
    nothing, so only the removed folder is named, and once the folder is back
    the start is admitted with the file still gone."""
    authority, _, registered = _registered_file(tmp_path)
    folder = tmp_path / "folder"
    folder.mkdir(mode=0o700)
    selector = tmp_path / "selected.toml"
    selector.write_text('[general]\nusers_name="fixture"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    authority.register("backup.source.folder", (folder,))
    registered.unlink()
    folder.rmdir()

    with pytest.raises(SystemExit) as refused:
        storage.admit_startup()

    text = str(refused.value)
    assert text.startswith("Recovery required: storage_admission_unavailable")
    assert str(folder) in text
    assert str(registered) not in text

    folder.mkdir(mode=0o700)
    storage.admit_startup()


@pytest.mark.usefixtures("local_root")
def test_a_deleted_file_root_stays_strict_while_a_remap_is_pending(tmp_path):
    """TASK-42115: the deleted-file rule applies only to a settled registry;
    while any remap is pending, a missing root still refuses admission."""
    from tldw_chatbook.Backup_Recovery.admission import fcntl

    authority, _, registered = _registered_file(tmp_path)
    folder = tmp_path / "folder"
    folder.mkdir(mode=0o700)
    authority.register("backup.source.folder", (folder,))
    registered.unlink()
    with authority._directory() as parent:
        settled = authority._read(parent)
        assert "backup.source.folder" in authority._groups(
            settled, ("backup.source.folder",)
        )
        with authority._lock(parent, "registry.lock", fcntl.LOCK_EX):
            pending = settled.model_copy(deep=True)
            pending.entries["backup.source.folder"].pending = "remap-in-flight"
            authority._write(parent, pending)
        with pytest.raises(FileNotFoundError):
            authority._groups(authority._read(parent), ("backup.source.folder",))


@pytest.mark.usefixtures("local_root")
@pytest.mark.parametrize("evidence", [True, False])
def test_a_deleted_file_root_scopes_exactly_its_own_path(
    tmp_path, monkeypatch, evidence
):
    """TASK-42115: a deleted file root still contains exactly its own path, as
    a present file root contains its object, so a rollback or the next Replace
    can publish it again. It contains nothing else; without the folder
    evidence it contains nothing."""
    from tldw_chatbook.Backup_Recovery import effective_roots as proof

    if not evidence:
        monkeypatch.setattr(proof, "file_root_tokens", lambda roots: set())
    _, files, registered = _registered_file(tmp_path)
    registered.unlink()
    assert storage._contains_capture_path((registered,), registered) is evidence
    assert storage._contains_capture_path((registered,), files / "other.json") is False
    assert storage._contains_capture_path((registered,), files) is False


@pytest.mark.usefixtures("local_root")
def test_a_bound_profile_with_a_deleted_file_root_admits_and_scopes_its_path(
    tmp_path, monkeypatch
):
    """TASK-42115: a bound profile's binding check and storage scope pass once
    one of its file roots is deleted. The deleted path itself is admitted
    again, so its owner can recreate it, and the folder's other files are
    still outside the profile."""
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile

    authority, files, registered = _registered_file(tmp_path, "core")
    selector = tmp_path / "selected.toml"
    selector.write_text('[general]\nusers_name="fixture"\n')
    selector.chmod(0o600)
    authority.register("selected", (selector,))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    root = bootstrap.default_bootstrap_root()
    bind_profile(root, selector, ("core", "selected"), root / "admission")
    registered.unlink()
    _, profiles = bootstrap._records(root)
    assert bootstrap._binding(selector, profiles, bootstrap._registry(root))
    assert bootstrap.startup_permission(selector, root) == (True, "startup_allowed")
    with storage.acquire_storage(registered):
        pass
    with pytest.raises(bootstrap.RecoveryRequired, match="storage_scope_not_enrolled"):
        storage.acquire_storage(files / "other.json")


@pytest.mark.usefixtures("local_root")
def test_capture_naming_and_maintenance_skip_a_deleted_file_root(tmp_path):
    """TASK-42115: capture naming covers nothing with a deleted file root, and a
    maintenance session's publication roots, containment and staging-overlap
    names keep its exact path without resolving it."""
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.capture_service import _capture_names
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    authority, _, registered = _registered_file(tmp_path)
    data = tmp_path / "data.db"
    data.write_bytes(b"x")
    data.chmod(0o600)
    registered.unlink()
    item = StorageItem("fixture", "profile:fixture:fixture", data, "included", ())
    names = _capture_names(authority, SimpleNamespace(items=(item,)))
    assert "backup.source.file" not in names
    with authority.maintenance(("backup.source.file",), 5) as session:
        assert (registered, False) in session._publication_roots
        assert registered in session._all_roots
        assert storage._contains_capture_path(session._roots, registered)
        staging = tmp_path / "staging"
        staging.mkdir(mode=0o700)
        with session._capture_bound_sources((), staging, ArchiveLimits(), 1):
            pass


def _verified_data_directory(tmp_path, monkeypatch, roots):
    """Run the start-up data-directory check against these enrolled roots."""
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import config_participants
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    selector = tmp_path / "config" / "config.toml"
    source = SimpleNamespace(
        _CONFIG_CACHE={
            "general": {"users_name": "fixture"},
            "paths": {"data_dir": str(tmp_path / "data")},
        },
        _CONFIG_CACHE_SOURCE=selector,
    )
    state = SimpleNamespace(
        source=source,
        companion_guard=object(),
        selected=selector,
        companion_roots=tuple(roots),
    )
    monkeypatch.setattr(raw._local, "operation", object(), raising=False)
    monkeypatch.setattr(raw, "check", lambda operation: state)
    try:
        return config_participants.verified_user_data_directory(source)
    except (OSError, bootstrap.RecoveryRequired) as error:
        return error


def _private_data_directory(tmp_path):
    data = tmp_path / "data"
    (data / "fixture").mkdir(parents=True)
    for path in (data, data / "fixture"):
        path.chmod(0o700)
    return data / "fixture"


def test_a_retired_settings_root_does_not_break_the_data_directory_check(
    tmp_path, monkeypatch
):
    """A note-templates root a Replace retired, enrolled ahead of the data
    directory's root, is skipped and the data directory is still found."""
    selected = _private_data_directory(tmp_path)
    retired = tmp_path / "config" / "note_templates.json"
    result = _verified_data_directory(tmp_path, monkeypatch, (retired, selected))
    assert result == selected


def test_a_data_directory_check_with_every_root_absent_refuses(
    tmp_path, monkeypatch
):
    _private_data_directory(tmp_path)
    roots = (
        tmp_path / "config" / "note_templates.json",
        tmp_path / "data" / "retired",
    )
    result = _verified_data_directory(tmp_path, monkeypatch, roots)
    assert isinstance(result, bootstrap.RecoveryRequired), result
    assert str(result) == "config_directory_selection_changed"


def _missing_registered_folder(tmp_path, monkeypatch):
    authority, _, _ = _registered_file(tmp_path)
    folder = tmp_path / "folder"
    folder.mkdir(mode=0o700)
    selector = tmp_path / "selected.toml"
    selector.write_text('[general]\nusers_name="fixture"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    authority.register("backup.source.folder", (folder,))
    folder.rmdir()
    return authority, selector


def _assert_legacy_entry_refusal(path):
    prior_holds = set(storage._retiring_holds)
    with pytest.raises(bootstrap.RecoveryRequired, match="storage_admission_unavailable"):
        storage.acquire_storage(path)
    [hold] = storage._retiring_holds - prior_holds
    assert isinstance(hold.error, FileNotFoundError)
    assert hold.entry_custody is None
    assert not hold.thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
@pytest.mark.parametrize("fault", ["native_close_error", "replaced_closer"])
@private_profile_test
def test_group_refusal_retains_uncertain_native_close(request, tmp_path, monkeypatch, fault):
    """Retain the hold if actual cleanup cannot finish on verified source."""
    import errno
    import os
    from contextlib import ExitStack

    authority, _ = _missing_registered_folder(tmp_path, monkeypatch)
    if os.name != "posix":
        _assert_legacy_entry_refusal(tmp_path / "app.db")
        return
    group_code = type(authority)._groups.__code__
    admit_code = type(authority)._admit.__wrapped__.__code__
    native_close, native_stat = os.close, os.fstat
    registry_inode = (authority.control_root / "registry.lock").stat().st_ino
    observed, callback_errors, stub_calls = [], [], []

    def observe_unwind(code, _offset, primary):
        if code is not group_code or not isinstance(primary, FileNotFoundError):
            return
        caller = sys._getframe(1).f_back
        if caller is None or caller.f_code is not admit_code:
            return
        try:
            assert caller.f_locals["stock_group_reader"] is True
            custody = caller.f_locals["_custody"]
            fd = custody.fds[-1]
            assert native_stat(fd).st_ino == registry_inode
            observed.append((primary, custody, fd))
            if fault == "native_close_error":
                native_close(fd)
            else:
                monkeypatch.setattr(storage.os, "close", stub_calls.append)
        except BaseException as error:  # Preserve the actual refusal.
            callback_errors.append(error)

    monitoring = sys.monitoring
    event = monitoring.events.PY_UNWIND
    tool = next(i for i in range(6) if monitoring.get_tool(i) is None)
    with ExitStack() as cleanup:
        monitoring.use_tool_id(tool, "test_storage_entry_close")
        cleanup.callback(monitoring.free_tool_id, tool)
        monitoring.register_callback(tool, event, observe_unwind)
        cleanup.callback(monitoring.register_callback, tool, event, None)
        cleanup.callback(monitoring.set_events, tool, 0)
        monitoring.set_events(tool, event)
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_admission_unavailable"):
            storage.acquire_storage(tmp_path / "app.db")

    assert not callback_errors
    [(primary, custody, fd)] = observed
    hold = custody.admission_owner
    assert custody.entry_reader_verified
    assert hold in storage._retiring_holds
    assert not hold.thread.is_alive()
    assert custody.close_failed and fd in custody.fds
    assert hold.error.exceptions[0] is primary
    error = hold.error.exceptions[1]
    chain = []
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    if fault == "native_close_error":
        assert any(isinstance(error, OSError) and error.errno == errno.EBADF for error in chain)
    else:
        assert not stub_calls
        assert any(str(error) == "capture_native_closer_changed" for error in chain)
        assert native_stat(fd).st_ino == registry_inode


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_root")
@private_profile_test
def test_group_refusal_keeps_unverified_admission_in_drain(request, tmp_path, monkeypatch):
    """Replacing admission cannot certify an undisclosed native resource."""
    import os

    authority, selector = _missing_registered_folder(tmp_path, monkeypatch)
    admit = type(authority)._admit
    resources = []

    def admit_with_untracked_file(self, *args, **kwargs):
        assert "_custody" not in kwargs
        resources.append(os.open(selector, os.O_RDONLY))
        return admit(self, *args, **kwargs)

    monkeypatch.setattr(type(authority), "_admit", admit_with_untracked_file)
    try:
        _assert_legacy_entry_refusal(tmp_path / "app.db")
        [fd] = resources
        assert os.fstat(fd).st_ino == selector.stat().st_ino
    finally:
        for fd in resources:
            os.close(fd)
