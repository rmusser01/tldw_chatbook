"""Source-bound raw mutation and actual native retirement evidence."""

import copy
import json
import time

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Feedback_Interop.local_feedback_service import LocalFeedbackService
from tldw_chatbook.Chat_Grammars_Interop.local_chat_grammars_service import (
    LocalChatGrammarsService,
)
from tldw_chatbook.Chunking.chunking_templates import (
    ChunkingTemplate,
    ChunkingTemplateManager,
)
from Tests.Backup_Recovery.test_participant_lifetimes import local_root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type", [LocalFeedbackService, LocalChatGrammarsService]
)
@pytest.mark.parametrize("action", ["create", "update", "delete", "persist", "load"])
async def test_service_pause_refuses_before_cached_mutation(
    tmp_path, local_root, service_type, action
):
    path = tmp_path / "nested" / "store.json"
    service = service_type(store_path=path)
    if service_type is LocalFeedbackService:
        await service.submit_feedback(
            feedback_type="helpful", helpful=True, query="before"
        )
        create = lambda: service.submit_feedback(
            feedback_type="helpful", helpful=True, query="after"
        )
        update = lambda: service.update_feedback("local-fb-1", user_notes="after")
        delete = lambda: service.delete_feedback("local-fb-1")
    else:
        await service.create_grammar(name="before", grammar_text="root ::= 'x'")
        create = lambda: service.create_grammar(
            name="after", grammar_text="root ::= 'y'"
        )
        update = lambda: service.update_grammar("local-grammar-1", name="after")
        delete = lambda: service.delete_grammar("local-grammar-1", hard_delete=True)
    before = copy.deepcopy((service._records, service._next_id))
    original = path.read_bytes()
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            if action in {"persist", "load"}:
                getattr(service, "_" + action)()
            else:
                await {"create": create, "update": update, "delete": delete}[action]()
        assert (service._records, service._next_id) == before
        assert path.read_bytes() == original
        assert pause.drain(time.monotonic() + 0.2)
    finally:
        pause.resume()
    await create()
    assert len(json.loads(path.read_text())["items"]) == 2


@pytest.mark.parametrize(
    "service_type", [LocalFeedbackService, LocalChatGrammarsService]
)
def test_constructor_refuses_before_storage_read(tmp_path, local_root, service_type):
    path = tmp_path / "store.json"
    path.write_text('{"items": []}')
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            service_type(store_path=path)
    finally:
        pause.resume()


def test_template_default_directory_not_created_while_paused(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook import config

    monkeypatch.setattr(config, "get_cli_data_dir", lambda: tmp_path / "new-data")
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            ChunkingTemplateManager()
        assert not (tmp_path / "new-data").exists()
    finally:
        pause.resume()


@pytest.mark.parametrize("user_template", [True, False])
def test_template_selected_destination_refuses_while_paused(
    tmp_path, local_root, user_template
):
    manager = ChunkingTemplateManager(
        templates_dir=tmp_path, user_templates_dir=tmp_path
    )
    template = ChunkingTemplate(name="new", pipeline=[])
    pause = storage._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            manager.save_template(template, user_template=user_template)
        assert not (tmp_path / "new.json").exists()
    finally:
        pause.resume()


def test_emoji_best_effort_refusal_has_no_directory_side_effect(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Widgets import emoji_picker

    selected = tmp_path / "new-profile" / "recent_emojis.json"
    monkeypatch.setattr(emoji_picker, "_recent_emojis_path", lambda: selected)
    pause = storage._begin_local_pause()
    try:
        assert emoji_picker.save_recent_emoji("😀") is None
        assert not selected.parent.exists()
    finally:
        pause.resume()


import asyncio
import os
import select
import threading
from contextlib import contextmanager

from Tests.Backup_Recovery.test_admission import launch, line, release
from Tests.Backup_Recovery.test_bootstrap import local_scope


@pytest.mark.parametrize(
    "service_type", [LocalFeedbackService, LocalChatGrammarsService]
)
def test_actual_service_worker_finishes_publication_across_pause(
    tmp_path, local_root, monkeypatch, launch, service_type
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    service = service_type(store_path=tmp_path / "nested" / "store.json")
    entered, finish = threading.Event(), threading.Event()
    errors = []
    original = json.dump

    def blocked_dump(value, stream, *args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return original(value, stream, *args, **kwargs)

    monkeypatch.setattr(json, "dump", blocked_dump)

    def work():
        try:
            if service_type is LocalFeedbackService:
                asyncio.run(
                    service.submit_feedback(
                        feedback_type="helpful", helpful=True, query="committed"
                    )
                )
            else:
                asyncio.run(
                    service.create_grammar(
                        name="committed", grammar_text="root ::= 'x'"
                    )
                )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    assert entered.wait(5)
    participant = raw._raw_participant(service)
    participant.close_admission()
    pause = storage._begin_local_pause()
    hold = storage._holds[(os.getpid(), str(local_root))]
    observer = launch(hold.authority.control_root, "maintenance", hold.names)
    try:
        assert not participant.drain(time.monotonic() + 0.03)
        assert not pause.drain(time.monotonic() + 0.03)
        assert not select.select([observer.stdout], [], [], 0.05)[0]
        finish.set()
        thread.join(5)
        assert not thread.is_alive() and not errors
        assert participant.drain(time.monotonic() + 1)
        assert pause.drain(time.monotonic() + 1)
        assert line(observer) == "entered"
        release(observer)
        assert len(json.loads(service.store_path.read_text())["items"]) == 1
        assert not service.store_path.with_suffix(".json.tmp").exists()
    finally:
        finish.set()
        thread.join(5)
        pause.resume()
        participant.resume()


@pytest.mark.asyncio
async def test_exact_file_binding_refuses_unowned_publication_sidecar(local_scope):
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile

    root, config, data, authority = local_scope
    path = data / "feedback.json"
    path.write_text('{"items": []}')
    authority.register("file-only", (path, config))
    bind_profile(root, config, ("file-only",), root / "admission")
    service = LocalFeedbackService(store_path=path)
    with pytest.raises(bootstrap.RecoveryRequired):
        await service.submit_feedback(feedback_type="helpful", helpful=True, query="no")
    assert service._records == [] and service._next_id == 1
    assert path.read_text() == '{"items": []}'
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.asyncio
async def test_directory_binding_admits_all_missing_directories_and_sidecar(
    local_scope,
):
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile

    root, config, data, authority = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    service = LocalFeedbackService(store_path=data / "new" / "deep" / "feedback.json")
    await service.submit_feedback(feedback_type="helpful", helpful=True, query="yes")
    assert len(json.loads(service.store_path.read_text())["items"]) == 1


@pytest.mark.asyncio
async def test_parent_replaced_during_admission_never_receives_mutation(
    tmp_path, local_root, monkeypatch
):
    parent = tmp_path / "original"
    parent.mkdir()
    service = LocalFeedbackService(store_path=parent / "feedback.json")
    original = storage.acquire_storage
    replaced = False

    def replace_parent(path=None):
        nonlocal replaced
        if not replaced:
            replaced = True
            parent.rename(tmp_path / "moved")
            parent.mkdir()
        return original(path)

    monkeypatch.setattr(storage, "acquire_storage", replace_parent)
    with pytest.raises(bootstrap.RecoveryRequired, match="raw_parent_identity_changed"):
        await service.submit_feedback(feedback_type="helpful", helpful=True, query="no")
    assert not (parent / "feedback.json").exists()
    assert not (tmp_path / "moved" / "feedback.json").exists()
    assert service._records == [] and service._next_id == 1


@pytest.mark.asyncio
async def test_write_failure_restores_cached_state_and_cleans_sidecar(
    tmp_path, local_root, monkeypatch
):
    service = LocalFeedbackService(store_path=tmp_path / "feedback.json")
    original = json.dump

    def fail_write(value, stream, *args, **kwargs):
        stream.write("partial")
        raise OSError("injected write failure")

    monkeypatch.setattr(json, "dump", fail_write)
    with pytest.raises(OSError, match="injected"):
        await service.submit_feedback(feedback_type="helpful", helpful=True, query="no")
    assert service._records == [] and service._next_id == 1
    assert not service.store_path.exists()
    assert not service.store_path.with_suffix(".json.tmp").exists()
    pause = storage._begin_local_pause()
    try:
        assert pause.drain(time.monotonic() + 1)
    finally:
        pause.resume()
    monkeypatch.setattr(json, "dump", original)
    await service.submit_feedback(feedback_type="helpful", helpful=True, query="yes")


def test_raw_token_copy_foreign_thread_and_sibling_cannot_write(tmp_path, local_root):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    service = LocalFeedbackService(store_path=tmp_path / "feedback.json")
    with raw._scope(service, "service", writing=True) as operation:
        with pytest.raises(bootstrap.RecoveryRequired, match="provenance"):
            raw._selected(copy.copy(operation))
        with pytest.raises(bootstrap.RecoveryRequired, match="outside_scope"):
            with raw._file(operation, tmp_path / "sibling.json", "w"):
                pytest.fail("sibling admitted")
        errors = []

        def foreign():
            try:
                raw._selected(operation)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join(3)
        assert len(errors) == 1 and isinstance(errors[0], bootstrap.RecoveryRequired)
    with pytest.raises(bootstrap.RecoveryRequired, match="provenance"):
        raw._selected(operation)


@pytest.mark.asyncio
async def test_raw_child_task_does_not_inherit_authority(tmp_path, local_root):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    service = LocalFeedbackService(store_path=tmp_path / "feedback.json")
    with raw._scope(service, "service", writing=True) as operation:

        async def child():
            with pytest.raises(bootstrap.RecoveryRequired, match="provenance"):
                raw._selected(operation)

        await asyncio.create_task(child())


@pytest.mark.asyncio
async def test_multitarget_pause_race_unwinds_before_memory_and_mkdir(
    tmp_path, local_root, monkeypatch
):
    service = LocalFeedbackService(store_path=tmp_path / "new" / "store.json")
    original = storage.acquire_storage
    pauses = []

    def raced(path=None):
        lease = original(path)
        if path.suffix == ".tmp":
            pauses.append(storage._begin_local_pause())
        return lease

    monkeypatch.setattr(storage, "acquire_storage", raced)
    try:
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            await service.submit_feedback(
                feedback_type="helpful", helpful=True, query="no"
            )
        assert service._records == [] and service._next_id == 1
        assert not service.store_path.parent.exists()
        assert pauses[0].drain(time.monotonic() + 1)
    finally:
        for pause in pauses:
            pause.resume()


@pytest.mark.asyncio
async def test_actual_emoji_app_worker_cancellation_does_not_retire_native_write(
    tmp_path, local_root, monkeypatch, launch
):
    from textual.app import App
    from tldw_chatbook.Widgets import emoji_picker

    selected = tmp_path / "profile" / "recent_emojis.json"
    monkeypatch.setattr(emoji_picker, "_recent_emojis_path", lambda: selected)
    entered, finish = threading.Event(), threading.Event()
    original = json.dump

    def blocked(value, stream, *args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return original(value, stream, *args, **kwargs)

    monkeypatch.setattr(json, "dump", blocked)

    class SaveApp(App):
        def on_mount(self):
            self.picker = emoji_picker.EmojiPickerScreen()
            self.picker._save_recent_emoji_off_loop("😀")

    app = SaveApp()
    async with app.run_test():
        assert await asyncio.to_thread(entered.wait, 5)
        worker = next(
            worker for worker in app.workers if worker.group == "emoji-recents-save"
        )
        worker.cancel()
        await asyncio.sleep(0.03)
        pause = storage._begin_local_pause()
        hold = storage._holds[(os.getpid(), str(local_root))]
        observer = launch(hold.authority.control_root, "maintenance", hold.names)
        try:
            assert not pause.drain(time.monotonic() + 0.03)
            assert not select.select([observer.stdout], [], [], 0.03)[0]
            finish.set()
            # Future cancellation may precede native thread completion.
            for _ in range(100):
                if pause.drain(time.monotonic()):
                    break
                await asyncio.sleep(0.01)
            assert pause.drain(time.monotonic() + 0.1)
            assert json.loads(selected.read_text()) == {"recent": ["😀"]}
            assert line(observer) == "entered"
            release(observer)
        finally:
            finish.set()
            pause.resume()


def test_same_source_waiter_is_counted_and_refused_without_cached_changes(
    tmp_path, local_root, monkeypatch
):
    service = LocalFeedbackService(store_path=tmp_path / "feedback.json")
    entered, finish = threading.Event(), threading.Event()
    original = json.dump
    errors = []

    def blocked(value, stream, *args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return original(value, stream, *args, **kwargs)

    monkeypatch.setattr(json, "dump", blocked)

    def work(query):
        try:
            asyncio.run(
                service.submit_feedback(
                    feedback_type="helpful", helpful=True, query=query
                )
            )
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=work, args=("first",))
    second = threading.Thread(target=work, args=("second",))
    first.start()
    assert entered.wait(5)
    second.start()
    for _ in range(100):
        if len(storage._pending_acquisitions) >= 2:
            break
        time.sleep(0.01)
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.03)
        finish.set()
        first.join(5)
        second.join(5)
        assert not first.is_alive() and not second.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], bootstrap.RecoveryRequired)
        assert service._next_id == 2
        assert [item["query"] for item in service._records] == ["first"]
        assert pause.drain(time.monotonic() + 1)
    finally:
        finish.set()
        first.join(5)
        second.join(5)
        pause.resume()


@pytest.mark.asyncio
async def test_uninstalled_subclass_has_ordinary_crud_but_no_participant(
    tmp_path, local_root
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    class CustomFeedback(LocalFeedbackService):
        pass

    service = CustomFeedback(store_path=tmp_path / "feedback.json")
    with pytest.raises(bootstrap.RecoveryRequired, match="not_installed"):
        raw._raw_participant(service)
    await service.submit_feedback(
        feedback_type="helpful", helpful=True, query="ordinary"
    )
    assert len(json.loads(service.store_path.read_text())["items"]) == 1


def test_generic_decorated_callback_cannot_borrow_installed_source_authority(
    tmp_path, local_root
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    service = LocalFeedbackService(store_path=tmp_path / "feedback.json")

    @raw._service_mutation
    def callback(source):
        operation = raw._service_file(source, "w")
        with raw._file(operation, source.store_path, "w") as stream:
            stream.write("forged")

    with pytest.raises(bootstrap.RecoveryRequired, match="raw_source_not_supported"):
        callback(service)
    assert not service.store_path.exists()


import subprocess
import sys
from pathlib import Path


_FAILED_CLOSE_CHILD = r"""
import asyncio, gc, io, json, os, sys, time
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw, storage_admission as storage
from tldw_chatbook.Feedback_Interop.local_feedback_service import LocalFeedbackService
root, selected, failure = sys.argv[1:]
bootstrap.default_bootstrap_root = lambda: Path(root)
portable = failure.startswith('portable-')
# Resolve real source imports before this fixture changes host native capabilities.
raw._types()
if portable:
    raw.os.supports_dir_fd = set()
    failure = failure.removeprefix('portable-')
service = LocalFeedbackService(store_path=selected)
original_dump = json.dump
original_close = os.close
original_text = io.TextIOWrapper
fd_target = None
retained = []
def observed(value, stream, *args, **kwargs):
    global fd_target
    fd_target = stream.fileno()
    return original_dump(value, stream, *args, **kwargs)
def failed_close(fd):
    if fd == fd_target:
        if failure == 'after':
            original_close(fd)
        raise OSError('injected native close uncertainty')
    return original_close(fd)
def failed_wrapper(native, *args, **kwargs):
    global fd_target
    fd_target = native.fileno()
    wrapper = original_text(native, *args, **kwargs)
    retained.append(wrapper)
    raise OSError('injected partial wrapper construction')
json.dump = observed
os.close = failed_close
if failure == 'construction':
    io.TextIOWrapper = failed_wrapper
try:
    asyncio.run(service.submit_feedback(feedback_type='helpful', helpful=True, query='held'))
except Exception:
    pass
else:
    raise AssertionError('failure missing')
os.close = original_close
io.TextIOWrapper = original_text
gc.collect()
assert storage._raw_operations
if portable:
    assert all(state.participant is None for state in raw._states.values())
pause = storage._begin_local_pause()
assert not pause.drain(time.monotonic() + .03)
assert service._records == [] and service._next_id == 1
if failure != 'after':
    os.fstat(fd_target)
    os.write(fd_target, b'actual native handle remains open')
assert any(fd_target in state.descriptors for state in raw._states.values())
print('held', flush=True)
sys.stdin.readline()
"""


@pytest.mark.parametrize(
    "failure", ["before", "after", "construction", "portable-before"]
)
def test_native_uncertainty_retains_resources_and_excludes_independent_maintenance(
    tmp_path, local_root, launch, failure
):
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
            _FAILED_CLOSE_CHILD,
            str(local_root),
            str(tmp_path / "store.json"),
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


def test_custom_template_destinations_are_ordinary_not_installed_coverage(
    tmp_path, local_root, monkeypatch
):
    from Tests.Backup_Recovery.config_test_support import install_config_source

    install_config_source(monkeypatch)
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    user = tmp_path / "user-custom"
    builtin = tmp_path / "builtin-custom"
    user.mkdir()
    builtin.mkdir()
    manager = ChunkingTemplateManager(templates_dir=builtin, user_templates_dir=user)
    with pytest.raises(bootstrap.RecoveryRequired, match="not_installed"):
        raw._raw_participant(manager)
    template = ChunkingTemplate(name="ordinary", pipeline=[])
    manager.save_template(template)
    manager.save_template(template, user_template=False)
    assert json.loads((user / "ordinary.json").read_text())["name"] == "ordinary"
    assert json.loads((builtin / "ordinary.json").read_text())["name"] == "ordinary"


def test_parent_replaced_after_pin_refuses_file_and_preserves_both_directories(
    tmp_path, local_root
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    parent = tmp_path / "parent"
    parent.mkdir()
    service = LocalFeedbackService(store_path=parent / "store.json")
    with raw._scope(service, "service", writing=True) as operation:
        parent.rename(tmp_path / "moved")
        parent.mkdir()
        with pytest.raises(
            bootstrap.RecoveryRequired, match="raw_parent_identity_changed"
        ):
            with raw._file(operation, service.store_path, "w"):
                pytest.fail("replacement received write")
    assert list(parent.iterdir()) == []
    assert list((tmp_path / "moved").iterdir()) == []


_PUBLICATION_CHILD = r"""
import asyncio, json, os, sys, time
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw, storage_admission as storage
from tldw_chatbook.Feedback_Interop.local_feedback_service import LocalFeedbackService
root, selected, failure = sys.argv[1:]
bootstrap.default_bootstrap_root = lambda: Path(root)
service = LocalFeedbackService(store_path=selected)
original = os.replace
def ambiguous(*args, **kwargs):
    if failure == 'after':
        original(*args, **kwargs)
    raise OSError('injected publication uncertainty')
os.replace = ambiguous
original_dump, original_unlink = json.dump, os.unlink
if failure == 'cleanup':
    os.replace = original
    def partial(value, stream, *args, **kwargs):
        stream.write('partial')
        raise OSError('injected body write failure')
    def failed_unlink(*args, **kwargs):
        raise OSError('injected cleanup failure')
    json.dump = partial
    os.unlink = failed_unlink
try:
    asyncio.run(service.submit_feedback(feedback_type='helpful', helpful=True, query='uncertain'))
except Exception:
    pass
else:
    raise AssertionError('missing failure')
os.replace = original
json.dump, os.unlink = original_dump, original_unlink
assert service._records == [] and service._next_id == 1
pause = storage._begin_local_pause()
assert not pause.drain(time.monotonic() + .03), 'publication uncertainty falsely drained'
pause.resume()
old_count = len(storage._raw_operations)
try:
    asyncio.run(service.submit_feedback(feedback_type='helpful', helpful=True, query='must refuse'))
except bootstrap.RecoveryRequired:
    pass
else:
    raise AssertionError('new operation overwrote unresolved publication')
assert service._records == [] and service._next_id == 1
assert len(storage._raw_operations) == old_count
assert Path(selected).exists() is (failure == 'after')
assert Path(selected + '.tmp').exists() is (failure in {'before', 'cleanup'})
print('held', flush=True)
sys.stdin.readline()
"""


@pytest.mark.parametrize("failure", ["before", "after", "cleanup"])
def test_ambiguous_publication_keeps_evidence_and_native_exclusion(
    tmp_path, local_root, launch, failure
):
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
            _PUBLICATION_CHILD,
            str(local_root),
            str(tmp_path / "store.json"),
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
async def test_existing_sidecar_is_preserved_and_not_claimed_by_new_mutation(
    tmp_path, local_root
):
    service = LocalFeedbackService(store_path=tmp_path / "store.json")
    sidecar = service.store_path.with_suffix(".json.tmp")
    sidecar.write_text("previous unresolved bytes")
    with pytest.raises(FileExistsError):
        await service.submit_feedback(feedback_type="helpful", helpful=True, query="no")
    assert sidecar.read_text() == "previous unresolved bytes"
    assert not service.store_path.exists()
    assert service._records == [] and service._next_id == 1


def test_hardlinked_selected_template_is_not_truncated_before_validation(
    tmp_path, local_root, monkeypatch
):
    from Tests.Backup_Recovery.config_test_support import install_config_source

    install_config_source(monkeypatch)
    other = tmp_path / "unrelated"
    other.write_text("preserve unrelated bytes")
    target = tmp_path / "template.json"
    target.hardlink_to(other)
    manager = ChunkingTemplateManager(
        templates_dir=tmp_path, user_templates_dir=tmp_path
    )
    with pytest.raises(ValueError, match="raw_not_regular"):
        manager.save_template(ChunkingTemplate(name="template", pipeline=[]))
    assert other.read_text() == target.read_text() == "preserve unrelated bytes"


def test_selector_lookup_pending_before_pause_is_counted_and_cannot_mkdir(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Widgets import emoji_picker
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    # A profile/session change needs its own installed source association.
    raw._source_participants.pop(emoji_picker, None)
    entered, finish = threading.Event(), threading.Event()
    selected = tmp_path / "new-profile" / "recent_emojis.json"

    def selector():
        entered.set()
        assert finish.wait(5)
        return selected

    monkeypatch.setattr(emoji_picker, "_recent_emojis_path", selector)
    thread = threading.Thread(target=emoji_picker.save_recent_emoji, args=("😀",))
    thread.start()
    assert entered.wait(5)
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.03)
        finish.set()
        thread.join(5)
        assert not thread.is_alive()
        assert not selected.parent.exists()
        assert pause.drain(time.monotonic() + 1)
    finally:
        finish.set()
        thread.join(5)
        pause.resume()


def test_emoji_selection_change_during_binding_refuses_both_paths(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Widgets import emoji_picker
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    raw._source_participants.pop(emoji_picker, None)
    first = tmp_path / "first" / "recent_emojis.json"
    second = tmp_path / "second" / "recent_emojis.json"
    paths = iter((first, second))
    monkeypatch.setattr(emoji_picker, "_recent_emojis_path", lambda: next(paths))
    emoji_picker.save_recent_emoji("😀")
    assert not first.parent.exists() and not second.parent.exists()


@pytest.mark.asyncio
async def test_native_pause_gate_refuses_new_raw_mutation_before_local_responder(
    tmp_path, local_root, launch
):
    from Tests.Backup_Recovery.test_participants import _eventually

    service = LocalFeedbackService(store_path=tmp_path / "store.json")
    retained = storage.acquire_storage(service.store_path)
    hold = storage._holds[retained._key]
    observer = launch(hold.authority.control_root, "maintenance", hold.names)
    try:
        _eventually(storage._local_pause_requested)
        with pytest.raises(bootstrap.RecoveryRequired, match="storage_locally_paused"):
            await service.submit_feedback(
                feedback_type="helpful", helpful=True, query="no"
            )
        assert service._records == [] and service._next_id == 1
        assert not service.store_path.exists()
    finally:
        retained.close()
    assert line(observer) == "entered"
    release(observer)


def test_pending_raw_native_acquisition_cancels_without_directory_creation(
    tmp_path, local_root, launch
):
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        UNBOUND_NAMESPACE,
    )

    authority = admission_authority(local_root)
    observer = launch(authority.control_root, "maintenance", (UNBOUND_NAMESPACE,))
    assert line(observer) == "entered"
    errors = []

    def work():
        try:
            LocalFeedbackService(store_path=tmp_path / "new" / "store.json")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    for _ in range(100):
        if storage._pending_acquisitions:
            break
        time.sleep(0.01)
    pause = storage._begin_local_pause()
    try:
        thread.join(5)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], bootstrap.RecoveryRequired)
        assert pause.drain(time.monotonic() + 1)
        assert not (tmp_path / "new").exists()
    finally:
        release(observer)
        thread.join(5)
        pause.resume()


def test_spoofed_source_function_metadata_cannot_grant_callback_authority(
    tmp_path, local_root
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    service = LocalFeedbackService(store_path=tmp_path / "store.json")

    def forged(source):
        operation = raw._service_file(source, "w")
        with raw._file(operation, source.store_path, "w") as stream:
            stream.write("forged")

    forged.__name__ = "_persist"
    forged.__qualname__ = "LocalFeedbackService._persist"
    forged.__module__ = LocalFeedbackService.__module__
    with pytest.raises(bootstrap.RecoveryRequired, match="raw_source_not_supported"):
        raw._service_mutation(forged)(service)
    assert not service.store_path.exists()


@pytest.mark.asyncio
async def test_successful_publication_cleanup_does_not_delete_next_process_sidecar(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    service = LocalFeedbackService(store_path=tmp_path / "store.json")
    original = os.replace
    children = []
    child_source = r"""
import asyncio, json, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Feedback_Interop.local_feedback_service import LocalFeedbackService
bootstrap.default_bootstrap_root = lambda: Path(sys.argv[1])
service = LocalFeedbackService(store_path=sys.argv[2])
original = json.dump
def staged(value, stream, *args, **kwargs):
    original(value, stream, *args, **kwargs)
    stream.flush()
    print('staged', flush=True)
    sys.stdin.readline()
json.dump = staged
asyncio.run(service.submit_feedback(feedback_type='helpful', helpful=True, query='second'))
print('committed', flush=True)
"""

    def publish_then_next_writer(*args, **kwargs):
        original(*args, **kwargs)
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                child_source,
                str(local_root),
                str(service.store_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        children.append(child)
        assert line(child) == "staged"

    monkeypatch.setattr(raw.os, "replace", publish_then_next_writer)
    try:
        await service.submit_feedback(
            feedback_type="helpful", helpful=True, query="first"
        )
        assert service.store_path.with_suffix(".json.tmp").exists(), (
            "first writer removed next writer's live sidecar"
        )
        child = children[0]
        child.stdin.write("continue\n")
        child.stdin.flush()
        assert line(child) == "committed"
        child.wait(timeout=5)
        assert child.returncode == 0, child.stderr.read()
        assert [
            item["query"]
            for item in json.loads(service.store_path.read_text())["items"]
        ] == ["first", "second"]
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()
            child.stderr.close()


@pytest.fixture
def portable_raw(monkeypatch):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    monkeypatch.setattr(raw.os, "supports_dir_fd", set())
    monkeypatch.setattr(
        storage, "qualified_for", lambda *args: (False, "test_native_unavailable")
    )

    def unavailable_pin(*args, **kwargs):
        raise AssertionError("portable route attempted unavailable raw pin primitive")

    monkeypatch.setattr(raw, "_open_verified_parent", unavailable_pin)
    return raw


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_type", [LocalFeedbackService, LocalChatGrammarsService]
)
async def test_portable_ordinary_service_reads_writes_and_retains_unqualified_leases(
    tmp_path, local_root, portable_raw, service_type, monkeypatch
):
    raw = portable_raw
    original = json.dump
    observed = []

    def observe(value, stream, *args, **kwargs):
        state = raw._states[raw._local.operation]
        assert state.participant is None and all(
            lease._key is None for lease in state.leases
        )
        assert os.fstat(stream.fileno()).st_size >= 0
        observed.append(True)
        return original(value, stream, *args, **kwargs)

    monkeypatch.setattr(json, "dump", observe)
    path = tmp_path / "portable" / "store.json"
    service = service_type(store_path=path)
    if service_type is LocalFeedbackService:
        await service.submit_feedback(
            feedback_type="helpful", helpful=True, query="portable"
        )
    else:
        await service.create_grammar(name="portable", grammar_text="root ::= 'x'")
    reloaded = service_type(store_path=path)
    assert reloaded._records == service._records and reloaded._next_id == 2
    assert observed and not path.with_suffix(".json.tmp").exists()
    with pytest.raises(bootstrap.RecoveryRequired, match="not_installed"):
        raw._raw_participant(service)


def test_portable_template_default_mkdir_and_both_destinations_work(
    tmp_path, local_root, portable_raw, monkeypatch
):
    from tldw_chatbook import config

    monkeypatch.setattr(config, "get_cli_data_dir", lambda: tmp_path / "portable-data")
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    manager = ChunkingTemplateManager(templates_dir=builtin)
    template = ChunkingTemplate(name="portable", pipeline=[])
    manager.save_template(template)
    manager.save_template(template, user_template=False)
    assert (
        manager._load_template_from_file(
            manager.user_templates_dir / "portable.json"
        ).name
        == "portable"
    )
    assert (
        manager._load_template_from_file(builtin / "portable.json").name == "portable"
    )


def test_portable_emoji_reads_and_writes_recents(
    tmp_path, local_root, portable_raw, monkeypatch
):
    from tldw_chatbook.Widgets import emoji_picker

    path = tmp_path / "portable-config" / "recent_emojis.json"
    monkeypatch.setattr(emoji_picker, "_recent_emojis_path", lambda: path)
    emoji_picker.save_recent_emoji("😀")
    emoji_picker.save_recent_emoji("🙂")
    assert emoji_picker.load_recent_emojis() == ["🙂", "😀"]


@pytest.mark.parametrize("evidence", ["pending", "uncertain", "outside", "disjoint"])
def test_portable_raw_preserves_existing_recovery_scope_checks(
    local_scope, portable_raw, evidence
):
    from tldw_chatbook.Backup_Recovery.control_records import (
        bind_profile,
        register_pending,
    )

    # The fixture creates real native records before primitive unavailability is simulated.
    root, config, data, authority = local_scope
    bind_profile(root, config, ("profile",), root / "admission")
    selected = data / "portable.json"
    if evidence == "pending":
        register_pending(
            root, "blocked", ("profile",), root.parent / "recovery", (config,)
        )
    elif evidence == "uncertain":
        unknown = root / "unknown.json"
        unknown.write_text("broken")
        unknown.chmod(0o600)
    elif evidence == "outside":
        selected = data.parent / "outside.json"
    else:
        other = root.parent / "other"
        other.mkdir()
        authority.register("other", (other,))
        (root.parent / "other-config").write_text("other")
        register_pending(
            root,
            "unrelated",
            ("other",),
            root.parent / "recovery",
            (root.parent / "other-config",),
        )
    if evidence == "disjoint":
        service = LocalFeedbackService(store_path=selected)
        service._persist()
        assert json.loads(selected.read_text()) == {"items": []}
    else:
        with pytest.raises(bootstrap.RecoveryRequired):
            LocalFeedbackService(store_path=selected)
        assert not selected.exists()


def test_portable_raw_retains_real_normal_lease_without_pause_authority(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    monkeypatch.setattr(raw.os, "supports_dir_fd", set())
    service = LocalFeedbackService(store_path=tmp_path / "store.json")
    pause = None
    try:
        with raw._scope(service, "service", writing=True) as operation:
            state = raw._states[operation]
            assert state.participant is None
            assert all(lease._key is not None for lease in state.leases)
            pause = storage._begin_local_pause()
            assert not pause.drain(time.monotonic() + 0.03)
            with pytest.raises(
                bootstrap.RecoveryRequired, match="storage_locally_paused"
            ):
                with raw._file(operation, service.store_path, "w"):
                    pytest.fail("portable descendant entered during pause")
        assert pause.drain(time.monotonic() + 1)
        with pytest.raises(
            bootstrap.RecoveryRequired, match="participant_runtime_coverage_incomplete"
        ):
            pause.require_runtime_coverage()
    finally:
        if pause is not None:
            pause.resume()


@pytest.mark.asyncio
async def test_supported_primitive_safety_failure_never_retries_portably(
    tmp_path, local_root, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    service = LocalFeedbackService(store_path=tmp_path / "store.json")

    def unsafe(*args, **kwargs):
        raise OSError("injected unsafe native parent")

    monkeypatch.setattr(raw, "_open_verified_parent", unsafe)
    with pytest.raises(OSError, match="unsafe native parent"):
        await service.submit_feedback(feedback_type="helpful", helpful=True, query="no")
    assert not service.store_path.exists() and service._records == []


@pytest.mark.asyncio
@pytest.mark.parametrize("portable", [False, True])
async def test_ordinary_selected_parent_alias_preserves_same_physical_source(
    tmp_path, local_root, monkeypatch, portable
):
    from tldw_chatbook.Backup_Recovery import raw_participants as raw

    if portable:
        monkeypatch.setattr(raw.os, "supports_dir_fd", set())
    target = tmp_path / "physical"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    if not portable:
        from tldw_chatbook.Utils import private_paths

        # Caller-owned aliases remain refused. Model the platform root-owned
        # trust classification for this exact real link; all actual opens,
        # directory identities and source bytes remain native. The actual
        # Darwin /tmp alias cannot be the immediate parent: it is shared-writable.
        with pytest.raises(private_paths.PrivatePathError):
            LocalFeedbackService(store_path=alias / "store.json")
        original_trust = private_paths._trusted_symlink
        identity = alias.lstat()
        monkeypatch.setattr(
            private_paths,
            "_trusted_symlink",
            lambda info: (
                (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino)
                or original_trust(info)
            ),
        )
    service = LocalFeedbackService(store_path=alias / "store.json")
    await service.submit_feedback(
        feedback_type="helpful", helpful=True, query="aliased"
    )
    assert (
        json.loads((target / "store.json").read_text())["items"][0]["query"]
        == "aliased"
    )
