"""Actual settings/definition sources retain their file and draft lifetimes."""

import json
import time

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from Tests.Backup_Recovery.test_participant_lifetimes import local_root


@pytest.fixture(autouse=True)
def fresh_template_source(monkeypatch):
    # A profile-bound module is process-lived in production. Give each isolated
    # test profile a fresh instance of the actual module, not a registry reset.
    import importlib.util
    import sys
    from tldw_chatbook import Notes
    from tldw_chatbook.Notes import template_store

    spec = importlib.util.spec_from_file_location(
        template_store.__name__, template_store.__file__
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setattr(Notes, "template_store", module)
    spec.loader.exec_module(module)
    yield module


def test_eval_load_refused_during_pause_preserves_memory(tmp_path, local_root):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    path = tmp_path / "eval.yaml"
    path.write_text("budget: {default_limit: 7}\n")
    loader = EvalConfigLoader(str(path))
    loader.update({"budget": {"default_limit": 9}})
    pause = storage._begin_local_pause()
    try:
        loader.reload()
        assert loader.get("budget.default_limit") == 9
        assert loader.persistence_safe_point() == "needs_user_save_or_discard"
    finally:
        pause.resume()


def test_theme_constructor_has_no_directory_effect_during_pause(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook import config
    from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor

    monkeypatch.setattr(
        config, "_get_effective_config_path", lambda: tmp_path / "new" / "config.toml"
    )
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            SettingsThemeEditor()
        assert not (tmp_path / "new").exists()
    finally:
        pause.resume()


def test_tamagotchi_direct_read_refuses_pause(tmp_path, local_root):
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import JSONStorage

    source = JSONStorage(str(tmp_path / "pets.json"), enable_recovery=False)
    assert source.save("one", {"name": "One"})
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            source._read_data()
        assert pause.drain(time.monotonic() + 0.2)
    finally:
        pause.resume()


def test_template_cli_rereads_after_input_without_holding_admission(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Config_Files import create_custom_template as cli

    selected = tmp_path / "profile" / "note_templates.json"
    selected.parent.mkdir()
    selected.write_text('{"templates": {"old": {"title": "Old"}}}')
    from tldw_chatbook import config

    monkeypatch.setattr(
        config, "_get_effective_config_path", lambda: selected.parent / "config.toml"
    )
    monkeypatch.setattr(
        cli, "_get_effective_config_path", lambda: selected.parent / "config.toml"
    )
    answers = iter(["new", "New", "Description", "tag", "Body", "END"])

    def answer(*args):
        assert not storage._raw_operations
        assert not storage._pending_acquisitions
        selected.write_text(
            '{"templates": {"old": {"title": "Old"}, "other": {"title": "Other"}}}'
        )
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)
    cli.create_custom_template()
    assert set(json.loads(selected.read_text())["templates"]) == {"old", "other", "new"}


def test_runtime_serialization_is_inside_admitted_source(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.runtime_policy import source_state
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState

    source = source_state.RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    original = source_state.json.dumps
    seen = []

    def serialize(*args, **kwargs):
        seen.append(bool(storage._raw_operations))
        return original(*args, **kwargs)

    monkeypatch.setattr(source_state.json, "dumps", serialize)
    source.save(RuntimeSourceState())
    assert seen == [True]


def test_pet_save_preserves_unowned_predictable_temp(tmp_path, local_root):
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import JSONStorage

    source = JSONStorage(str(tmp_path / "pets.json"), enable_recovery=False)
    stale = source.filepath.with_suffix(".tmp")
    stale.write_text("not this writer")
    source.save("one", {"name": "One"})
    assert stale.read_text() == "not this writer"


@pytest.mark.asyncio
async def test_template_running_cancel_waits_for_actual_parser(
    tmp_path, local_root, monkeypatch
):
    import asyncio
    import threading
    from Tests.Event_Handlers.test_note_ingest_import_offload import (
        _make_mock_app,
        _write_note_files,
        _dispatch_and_get_worker,
    )
    from tldw_chatbook.Event_Handlers import note_ingest_events

    files = _write_note_files(tmp_path, note_count=2, files=1)
    app = _make_mock_app(files)
    app.query_one("#import-as-templates-radio").value = True
    entered, finish = threading.Event(), threading.Event()
    original = note_ingest_events._parse_single_note_file_for_preview

    def parse(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        note_ingest_events, "_parse_single_note_file_for_preview", parse
    )
    worker = await _dispatch_and_get_worker(app)
    task = asyncio.create_task(worker())
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done(), (
            "cancel retired the awaiter while its actual parser still runs"
        )
        assert storage._pending_acquisitions
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)


def test_eval_mutable_get_export_and_failed_save_report_dirty(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader

    selected = tmp_path / "eval.yaml"
    selected.write_text("budget: {default_limit: 7}\n")
    loader = EvalConfigLoader(str(selected))
    assert loader.persistence_safe_point() == "ready"
    loader.get("budget")["default_limit"] = 9
    loader.save(str(tmp_path / "export" / "eval.yaml"))
    assert loader.persistence_safe_point() == "needs_user_save_or_discard"
    assert "7" in selected.read_text()
    pause = storage._begin_local_pause()
    try:
        loader.save()
        assert loader.persistence_error == "eval_save_failed"
        assert loader.persistence_safe_point() == "needs_user_save_or_discard"
    finally:
        pause.resume()
    loader.save()
    assert loader.persistence_safe_point() == "ready"
    assert "9" in selected.read_text()


def test_template_merge_concurrent_instances_keep_unrelated_records(
    tmp_path, local_root, monkeypatch
):
    import threading
    from tldw_chatbook import config
    from tldw_chatbook.Notes.template_store import merge_templates

    selected = tmp_path / "note_templates.json"
    monkeypatch.setattr(
        config, "_get_effective_config_path", lambda: tmp_path / "config.toml"
    )
    merge_templates([("old", {"title": "Old"})])
    barrier = threading.Barrier(3)
    errors = []

    def write(key):
        try:
            barrier.wait(5)
            merge_templates([(key, {"title": key})])
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=write, args=(key,)) for key in ("a", "b")]
    for thread in threads:
        thread.start()
    barrier.wait(5)
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert not errors
    assert set(json.loads(selected.read_text())["templates"]) == {"old", "a", "b"}


def test_template_failure_leaves_previous_bytes_and_no_success(
    tmp_path, local_root, monkeypatch
):
    import threading
    from tldw_chatbook import config
    from tldw_chatbook.Notes.template_store import merge_templates
    from tldw_chatbook.Event_Handlers.note_ingest_events import _import_template_files
    from Tests.Event_Handlers.test_note_ingest_import_offload import (
        _make_mock_app,
        _write_note_files,
    )
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    selected = tmp_path / "note_templates.json"
    monkeypatch.setattr(
        config, "_get_effective_config_path", lambda: tmp_path / "config.toml"
    )
    merge_templates([("old", {"title": "Old"})])
    before = selected.read_bytes()
    stale = selected.with_suffix(".json.tmp")
    stale.write_text("somebody else's temporary")
    files = _write_note_files(tmp_path, 2, 1)
    results = _import_template_files(
        _make_mock_app(files), files, threading.Event(), selected
    )
    assert len(results) == 2 and all(r["status"] == "failure" for r in results)
    assert all("template_key" not in r for r in results)
    assert selected.read_bytes() == before
    assert stale.read_text() == "somebody else's temporary"
    assert not raw._states


def test_installed_default_selector_changes_cannot_demote_to_custom(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook import Evals
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    selected = tmp_path / "eval.yaml"
    selected.write_text("budget: {default_limit: 7}\n")
    monkeypatch.setattr(Evals, "_default_config_path", lambda: selected)
    loader = EvalConfigLoader()
    participant = raw._raw_participant(loader)
    loader.config_path = tmp_path / "other.yaml"
    loader.save()
    assert loader.persistence_error == "eval_save_failed"
    assert not loader.config_path.exists()
    with pytest.raises(bootstrap.RecoveryRequired, match="selection_changed"):
        participant.close_admission()


def test_pet_pruning_preserves_exact_backup_bytes_and_unknown_sibling(
    tmp_path, local_root
):
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import JSONStorage

    selected = tmp_path / "pets.json"
    selected.write_bytes(b'{\r\n "first": {"name": "First"}\r\n}')
    source = JSONStorage(str(selected), enable_recovery=False, max_backups=2)
    old = selected.with_suffix(".backup_20000101_010101.json")
    newer = selected.with_suffix(".backup_20000102_010101.json")
    unknown = selected.with_suffix(".backup_surprise.json")
    for path in (old, newer, unknown):
        path.write_text("previous")
    before = selected.read_bytes()
    assert source.save("second", {"name": "Second"})
    backups = sorted(
        p for p in selected.parent.glob("pets.backup_*.json") if p != unknown
    )
    assert not old.exists() and newer.exists() and unknown.exists()
    assert len(backups) == 2 and backups[-1].read_bytes() == before
    assert set(json.loads(selected.read_text())) == {"first", "second"}


def test_dynamic_membership_pause_race_has_no_mutation(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import JSONStorage

    source = JSONStorage(str(tmp_path / "pets.json"), enable_recovery=False)
    before = source.filepath.read_bytes()
    original = storage.acquire_storage
    pauses = []

    def admit(path=None):
        lease = original(path)
        if ".backup_" in str(path):
            pauses.append(storage._begin_local_pause())
        return lease

    monkeypatch.setattr(storage, "acquire_storage", admit)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            source.save("no", {"name": "No"})
        assert source.filepath.read_bytes() == before
        assert not list(tmp_path.glob("*.backup*"))
        assert pauses[0].drain(time.monotonic() + 1)
    finally:
        for pause in pauses:
            pause.resume()


@pytest.mark.asyncio
async def test_theme_save_failure_and_pause_preserve_draft_and_tree(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook import config
    from tldw_chatbook.Widgets import settings_theme_editor as themes
    from Tests.UI.test_settings_theme_editor import (
        _isolated_editor_app,
        _user_theme_labels,
    )
    from textual.widgets import Input

    monkeypatch.setattr(
        config, "_get_effective_config_path", lambda: tmp_path / "config.toml"
    )
    editor = themes.SettingsThemeEditor()
    app = _isolated_editor_app(editor)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor.query_one("#settings-theme-name", Input).value = "mine"
        await pilot.pause()
        editor.is_modified = True
        original = themes.toml.dump

        def fail(data, stream):
            stream.write("partial")
            raise OSError("write failed")

        monkeypatch.setattr(themes.toml, "dump", fail)
        editor.on_save_theme()
        assert editor.is_modified
        assert "user:mine" not in _user_theme_labels(editor)
        assert not (editor.custom_themes_path / "mine.toml").exists()
        assert not list(editor.custom_themes_path.glob("*.tmp"))
        monkeypatch.setattr(themes.toml, "dump", original)
        pause = storage._begin_local_pause()
        try:
            editor.on_save_theme()
            assert editor.is_modified
            assert not (editor.custom_themes_path / "mine.toml").exists()
        finally:
            pause.resume()
        editor.on_save_theme()
        assert not editor.is_modified
        assert "user:mine" in _user_theme_labels(editor)
        assert (editor.custom_themes_path / "mine.toml").exists()


import os
import select
import threading
from Tests.Backup_Recovery.test_admission import launch, line, release


@pytest.mark.parametrize("family", ["runtime", "eval", "templates", "pet"])
def test_actual_settings_writer_holds_independent_maintenance_until_close(
    tmp_path, local_root, monkeypatch, launch, family
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook import config, Evals
    from tldw_chatbook.runtime_policy import bootstrap as runtime_bootstrap
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader
    from tldw_chatbook.Notes import template_store
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import ConfigFileStorage
    import yaml

    if family == "runtime":
        selected = tmp_path / "runtime_policy.json"
        monkeypatch.setattr(
            runtime_bootstrap, "default_runtime_policy_path", lambda: selected
        )
        source = RuntimeSourceStateStore(selected)
        work = lambda: source.save(RuntimeSourceState(active_source="local"))
        target, name = json, "dumps"
    elif family == "eval":
        selected = tmp_path / "eval.yaml"
        selected.write_text("budget: {default_limit: 7}\n")
        monkeypatch.setattr(Evals, "_default_config_path", lambda: selected)
        source = EvalConfigLoader()
        source.update({"budget": {"default_limit": 9}})
        work = source.save
        target, name = yaml, "dump"
    elif family == "templates":
        monkeypatch.setattr(
            config, "_get_effective_config_path", lambda: tmp_path / "config.toml"
        )
        source = template_store
        selected = tmp_path / "note_templates.json"
        work = lambda: source.merge_templates([("one", {"title": "One"})])
        target, name = json, "dump"
    else:
        monkeypatch.setenv("HOME", str(tmp_path))
        source = ConfigFileStorage()
        selected = source.filepath
        work = lambda: source.save("one", {"name": "One"})
        target, name = json, "dump"
    entered, finish = threading.Event(), threading.Event()
    original = getattr(target, name)
    errors = []

    def blocked(value, *args, **kwargs):
        # Metadata bootstrap serialization is outside the selected payload.
        if raw._local.operation is not None:
            entered.set()
            assert finish.wait(5)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(target, name, blocked)

    def run():
        try:
            work()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5)
        participant = raw._raw_participant(source)
        participant.close_admission()
        pause = storage._begin_local_pause()
        hold = storage._holds[(os.getpid(), str(local_root))]
        observer = launch(hold.authority.control_root, "maintenance", hold.names)
        try:
            assert not pause.drain(time.monotonic() + 0.03)
            assert not select.select([observer.stdout], [], [], 0.03)[0]
            finish.set()
            thread.join(5)
            assert not thread.is_alive() and not errors
            assert participant.drain(time.monotonic() + 1)
            assert pause.drain(time.monotonic() + 1)
            assert line(observer) == "entered"
            release(observer)
            assert selected.exists()
            if family == "eval":
                assert source.persistence_safe_point() == "ready"
        finally:
            pause.resume()
            participant.resume()
    finally:
        finish.set()
        thread.join(5)


_RUNTIME_UNCERTAIN_CHILD = r"""
import json, os, sys, time
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw, storage_admission as storage
from tldw_chatbook.runtime_policy import bootstrap as runtime_bootstrap
from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.Utils import private_paths as private
root, selected, failure = sys.argv[1:]
selected = Path(selected)
bootstrap.default_bootstrap_root = lambda: Path(root)
runtime_bootstrap.default_runtime_policy_path = lambda: selected
source = RuntimeSourceStateStore(selected)
source.save(RuntimeSourceState())
original_close, original_write, original_open, original_load = os.close, os.write, private._native_open, json.load
fd_target = None
replacement = None
in_pin = False
original_pin = raw._open_verified_parent
original_fdopen = os.fdopen
class FailingStream:
    def __init__(self, stream):
        self.stream = stream
    def __getattr__(self, name):
        return getattr(self.stream, name)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    def close(self):
        if failure == "stream_after":
            self.stream.close()
        raise OSError("stream close uncertainty")
def fdopen(*args, **kwargs):
    stream = original_fdopen(*args, **kwargs)
    return FailingStream(stream) if failure.startswith("stream_") else stream
def pin(*args, **kwargs):
    global in_pin
    in_pin = True
    try:
        return original_pin(*args, **kwargs)
    finally:
        in_pin = False
def opened(*args, **kwargs):
    global fd_target
    fd = original_open(*args, **kwargs)
    if args[0] == '/' and ((failure == 'ancestor' and raw._local.operation is not None) or (failure.startswith('preflight') and in_pin)):
        fd_target = fd
    return fd
def writing(fd, value):
    global fd_target, replacement
    if raw._local.operation is not None:
        fd_target = fd
        if failure == 'temporary_identity':
            operation = raw._local.operation
            replacement = raw._check(operation).temporary
            replacement.rename(replacement.with_suffix('.unpublished'))
            replacement.write_bytes(b'new-owner')
            raise OSError('injected write failure')
    return original_write(fd, value)
def loading(stream):
    global fd_target
    fd_target = stream.fileno()
    return original_load(stream)
def closing(fd):
    if fd == fd_target and failure != 'temporary_identity' and not failure.startswith('stream_'):
        if failure in ('after', 'preflight_after'):
            original_close(fd)
        raise OSError('injected native close uncertainty')
    return original_close(fd)
private._native_open = opened
raw._open_verified_parent = pin
os.write = writing
os.close = closing
json.load = loading
os.fdopen = fdopen
try:
    source.load() if failure == 'reader' or failure.startswith('stream_') else source.save(RuntimeSourceState())
except Exception:
    pass
else:
    raise AssertionError('missing failure')
os.close, os.write, private._native_open, json.load = original_close, original_write, original_open, original_load
assert storage._raw_operations
pause = storage._begin_local_pause()
assert not pause.drain(time.monotonic() + .03)
if failure not in ('after', 'preflight_after', 'temporary_identity'):
    os.fstat(fd_target)
if failure == 'temporary_identity':
    assert replacement.read_bytes() == b'new-owner'
else:
    assert any(fd_target in state.descriptors for state in raw._states.values())
print('held', flush=True)
sys.stdin.readline()
"""


@pytest.mark.parametrize(
    "failure",
    [
        "before",
        "after",
        "ancestor",
        "reader",
        "temporary_identity",
        "preflight",
        "preflight_after",
        "stream_before",
        "stream_after",
    ],
)
def test_runtime_private_native_uncertainty_blocks_independent_maintenance(
    tmp_path, local_root, launch, failure
):
    import subprocess
    import sys
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
            _RUNTIME_UNCERTAIN_CHILD,
            str(local_root),
            str(tmp_path / "runtime_policy.json"),
            failure,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        try:
            assert line(child) == "held"
        except AssertionError:
            child.wait(timeout=5)
            pytest.fail(child.stderr.read())
        observer = launch(authority.control_root, "maintenance", (UNBOUND_NAMESPACE,))
        assert not select.select([observer.stdout], [], [], 0.05)[0]
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


@pytest.mark.asyncio
async def test_template_queued_cancel_does_not_enter_source(tmp_path, local_root):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from Tests.Event_Handlers.test_note_ingest_import_offload import (
        _make_mock_app,
        _write_note_files,
        _dispatch_and_get_worker,
    )

    app = _make_mock_app(_write_note_files(tmp_path, 2, 1))
    app.query_one("#import-as-templates-radio").value = True
    worker = await _dispatch_and_get_worker(app)
    loop = asyncio.get_running_loop()
    previous = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()
    occupied = executor.submit(gate.wait, 5)
    loop.set_default_executor(executor)
    try:
        task = asyncio.create_task(worker())
        await asyncio.sleep(0.03)
        assert storage._pending_acquisitions
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not storage._pending_acquisitions
        gate.set()
        occupied.result(5)
        await asyncio.sleep(0.03)
        assert not storage._raw_operations
    finally:
        gate.set()
        occupied.result(5)
        executor.shutdown(wait=True)
        loop._default_executor = previous


def test_pet_prune_rechecks_observed_identity(tmp_path, local_root, monkeypatch):
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import JSONStorage
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    source = JSONStorage(
        str(tmp_path / "pets.json"), enable_recovery=False, max_backups=1
    )
    old = source.filepath.with_suffix(".backup_20000101_010101.json")
    old.write_text("old")
    original = raw._unlink

    def raced(operation, path):
        path.rename(path.with_suffix(".moved"))
        path.write_text("new owner")
        return original(operation, path)

    monkeypatch.setattr(raw, "_unlink", raced)
    source.save("one", {"name": "One"})
    assert old.read_text() == "new owner"
    assert not storage._raw_operations


def test_runtime_direct_private_helpers_refuse_sibling_and_finish_exact_read(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.runtime_policy import bootstrap as runtime_bootstrap
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState
    from tldw_chatbook.Utils.private_paths import (
        open_private_binary,
        atomic_private_write_text,
    )

    selected = tmp_path / "runtime_policy.json"
    monkeypatch.setattr(
        runtime_bootstrap, "default_runtime_policy_path", lambda: selected
    )
    source = RuntimeSourceStateStore(selected)
    source.save(RuntimeSourceState())
    with raw._scope(source, "runtime_state", writing=True):
        pause = storage._begin_local_pause()
        try:
            with open_private_binary(selected) as opened:
                assert json.load(opened.stream)["active_source"] == "local"
                assert storage._raw_operations
            with pytest.raises(bootstrap.RecoveryRequired, match="outside_scope"):
                atomic_private_write_text(tmp_path / "sibling.json", "no")
            assert not (tmp_path / "sibling.json").exists()
        finally:
            pause.resume()
    assert not storage._raw_operations


@pytest.mark.parametrize("family", ["eval", "pet", "runtime", "templates"])
def test_settings_sources_preserve_portable_ordinary_io(
    tmp_path, local_root, monkeypatch, family
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook import Evals, config
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import JSONStorage
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState
    from tldw_chatbook.Notes import template_store

    monkeypatch.setattr(raw, "_pinned_io_available", lambda: False)
    if family == "eval":
        selected = tmp_path / "eval.yaml"
        monkeypatch.setattr(Evals, "_default_config_path", lambda: selected)
        source = EvalConfigLoader()
        source.save()
        assert selected.exists() and source.persistence_safe_point() == "ready"
    elif family == "pet":
        source = JSONStorage(str(tmp_path / "pets.json"), enable_recovery=False)
        assert (
            source.save("one", {"name": "One"}) and source.load("one")["name"] == "One"
        )
    elif family == "runtime":
        source = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
        source.save(RuntimeSourceState())
        assert source.load() == RuntimeSourceState()
    else:
        monkeypatch.setattr(
            config, "_get_effective_config_path", lambda: tmp_path / "config.toml"
        )
        source = template_store
        source.merge_templates([("one", {"title": "One"})])
        assert source.read_templates()["one"]["title"] == "One"
    with pytest.raises(bootstrap.RecoveryRequired, match="not_installed"):
        raw._raw_participant(source)
    assert not storage._raw_operations


def test_installed_eval_drain_reports_unsaved_mutable_draft(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook import Evals
    from tldw_chatbook.Evals.config_loader import EvalConfigLoader
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    selected = tmp_path / "eval.yaml"
    selected.write_text("budget: {default_limit: 7}\n")
    monkeypatch.setattr(Evals, "_default_config_path", lambda: selected)
    source = EvalConfigLoader()
    source.get("budget")["default_limit"] = 9
    participant = raw._raw_participant(source)
    participant.close_admission()
    try:
        assert not participant.drain(time.monotonic() + 0.1)
        assert source.get("budget.default_limit") == 9
    finally:
        participant.resume()
    source.save()
    participant.close_admission()
    try:
        assert participant.drain(time.monotonic() + 0.1)
    finally:
        participant.resume()


def test_pet_same_timestamp_backup_tracks_previous_save(
    tmp_path, local_root, monkeypatch
):
    from datetime import datetime
    from tldw_chatbook.Backup_Recovery import settings_file_participants as participants
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import JSONStorage

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, *args, **kwargs):
            return cls(2026, 9, 8, 1, 2, 3)

    monkeypatch.setattr(participants, "datetime", FixedDatetime, raising=False)
    source = JSONStorage(str(tmp_path / "pets.json"), enable_recovery=False)
    source.save("first", {"name": "First"})
    previous = source.filepath.read_bytes()
    source.save("second", {"name": "Second"})
    backups = list(tmp_path.glob("pets.backup_*.json"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("action", ["reload", "save"])
def test_eval_native_parent_close_failure_preserves_draft(
    tmp_path, local_root, after, action
):
    import subprocess
    import sys

    program = r"""
import os, stat, sys, time
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw, storage_admission as storage
from tldw_chatbook.Evals.config_loader import EvalConfigLoader
root, path, after, action = sys.argv[1:]
bootstrap.default_bootstrap_root = lambda: Path(root)
path = Path(path)
path.write_text('budget: {default_limit: 7}\n')
source = EvalConfigLoader(str(path))
source.update({'budget': {'default_limit': 9}})
draft = source.get('budget')
original = os.close
failed = None
def close(fd):
    global failed
    if any(fd in state.pins.values() for state in raw._states.values()):
        failed = fd
        if after == 'True':
            original(fd)
        raise OSError('native parent close failed')
    original(fd)
os.close = close
getattr(source, action)()
os.close = original
assert failed is not None
assert source.get('budget') is draft
assert source.get('budget.default_limit') == 9
assert source.persistence_error == ('eval_load_failed' if action == 'reload' else 'eval_save_failed')
assert source.persistence_safe_point() == 'needs_user_save_or_discard'
assert storage._raw_operations
pause = storage._begin_local_pause()
assert not pause.drain(time.monotonic() + .02)
if after == 'False':
    os.fstat(failed)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(local_root),
            str(tmp_path / "eval.yaml"),
            str(after),
            action,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_runtime_pause_during_initial_pin_refuses_without_mutation(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.runtime_policy import bootstrap as runtime_bootstrap
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState

    selected = tmp_path / "runtime.json"
    monkeypatch.setattr(
        runtime_bootstrap, "default_runtime_policy_path", lambda: selected
    )
    source = RuntimeSourceStateStore(selected)
    source.save(RuntimeSourceState())
    previous = selected.read_bytes()
    pauses = []
    original = raw._open_verified_parent

    def pin(*args, **kwargs):
        value = original(*args, **kwargs)
        assert raw._states and storage._raw_operations and storage._pending_acquisitions
        pauses.append(storage._begin_local_pause())
        return value

    monkeypatch.setattr(raw, "_open_verified_parent", pin)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            source.save(RuntimeSourceState())
        assert selected.read_bytes() == previous
        assert not list(tmp_path.glob(".runtime.json.*.tmp"))
        assert pauses[0].drain(time.monotonic() + 1)
    finally:
        for pause in pauses:
            pause.resume()


def test_runtime_source_binding_does_not_load_unrelated_source_types(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.runtime_policy import bootstrap as runtime_bootstrap
    from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState

    selected = tmp_path / "runtime.json"
    monkeypatch.setattr(
        runtime_bootstrap, "default_runtime_policy_path", lambda: selected
    )

    def unrelated_types():
        raise AssertionError(
            "runtime source binding loaded unrelated legacy source modules"
        )

    monkeypatch.setattr(raw, "_types", unrelated_types)
    source = RuntimeSourceStateStore(selected)
    source.save(RuntimeSourceState())
    assert selected.exists()
