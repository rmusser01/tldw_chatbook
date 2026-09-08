"""Real config source/cache/private resource lifetimes under maintenance."""

import importlib.util
import sys

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from Tests.Backup_Recovery.test_participant_lifetimes import local_root
from Tests.Backup_Recovery.test_admission import launch


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
    before = source._CONFIG_CACHE
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired):
            source.load_cli_config_and_ensure_existence()
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
        if raw._runtime_operation() is not None:
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
    if raw._runtime_operation() is not None or direct_body:
        if (failure == 'ancestor' or direct_body) and args[0] == '/': fd_target = fd
        if failure.startswith('lock_') and str(args[0]).endswith('.lock'): fd_target = fd
    return fd
def writing(fd, payload):
    global fd_target
    if failure.startswith('temp_') and raw._runtime_operation() is not None: fd_target = fd
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
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        try:
            assert line(child) == "held"
        except AssertionError:
            child.wait(timeout=5)
            pytest.fail(child.stderr.read())
        observer = launch(authority.control_root, "maintenance", (UNBOUND_NAMESPACE,))
        assert not select.select([observer.stdout], [], [], 0.1)[0]
        child.stdin.write("exit\n")
        child.stdin.flush()
        child.wait(timeout=5)
        assert child.returncode == 0, child.stderr.read()
        assert line(observer) == "entered"
        release(observer)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()


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
    original = config.toml.dumps
    def serialize(*args, **kwargs):
        value = original(*args, **kwargs)
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

    monkeypatch.setattr(Path, "mkdir", reenter)
    with pytest.raises(bootstrap.RecoveryRequired, match="raw_path_outside_scope"):
        source.get_model_cache_dir()
    assert not directory.exists()
    assert source.get_cli_config_path().read_bytes() == before
    assert source._CONFIG_CACHE is cache
    assert source._CONFIG_GENERATION == generation
