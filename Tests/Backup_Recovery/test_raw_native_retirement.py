"""Original native retirement uses issued ownership without repeated path scans."""

import copy
import errno
import os
import sys
import threading
import time
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery import test_participant_lifetimes as participant_cases
from Tests.Backup_Recovery.config_test_support import install_config_source
from Tests.private_profile import private_profile_test
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import private_paths

local_root = participant_cases.local_root


@pytest.fixture
def configured_source(tmp_path, monkeypatch, local_root):
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    selector = tmp_path / "config.toml"
    selector.write_text(
        '[general]\nusers_name="retirement"\n[paths]\ndata_dir="'
        + data.as_posix()
        + '"\n',
        encoding="utf-8",
    )
    selector.chmod(0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    source = install_config_source(monkeypatch)
    assert source.get_user_data_dir() == data / "retirement"
    return source


@contextmanager
def _original_tracked_closes(source):
    close_code = raw._close_descriptor.__code__
    native_close_code = private_paths._native_close.__code__
    finite_close_code = next(
        code
        for code in private_paths._prepared_parent_walk.__wrapped__.__code__.co_consts
        if isinstance(code, type(close_code)) and code.co_name == "owned_close"
    )
    check_code = raw._check.__code__
    parent_code = private_paths._open_verified_parent.__code__
    active, native_active, finite_active, operations = {}, set(), set(), set()
    counts = dict(closes=0, checks=0, parent_walk_closes=0)
    closed, removed = [], []
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if frame.f_code is native_close_code:
            if event == "call":
                operation = getattr(raw._local, "operation", None)
                state = raw._states.get(operation)
                if (
                    state is not None
                    and state.source is source
                    and frame.f_locals["fd"] in state.descriptors
                ):
                    # Preserve the task17 regression: a full proof before the
                    # actual native close still counts against retirement.
                    native_active.add(id(frame))
            elif event == "return":
                native_active.discard(id(frame))
        elif frame.f_code is finite_close_code:
            if event == "call":
                state, operation = frame.f_locals["state"], frame.f_locals["operation"]
                if (
                    state.source is source
                    and raw._states.get(operation) is state
                    and frame.f_locals["fd"] in state.descriptors
                ):
                    finite_active.add(id(frame))
            elif event == "return":
                finite_active.discard(id(frame))
        elif frame.f_code is close_code:
            if event == "call":
                state, fd = frame.f_locals["state"], frame.f_locals["fd"]
                if state.source is not source or fd not in state.descriptors:
                    return
                operation = next(
                    (
                        operation
                        for operation, issued in tuple(raw._states.items())
                        if issued is state
                    ),
                    None,
                )
                assert operation is not None, "close must belong to an issued source"
                active[id(frame)] = state, fd
                operations.add(operation)
                counts["closes"] += 1
                parent = frame.f_back
                while parent is not None:
                    if parent.f_code is parent_code:
                        counts["parent_walk_closes"] += 1
                        break
                    parent = parent.f_back
            elif event == "return" and id(frame) in active:
                state, fd = active.pop(id(frame))
                # Check this exact descriptor incarnation before any caller can
                # open another file and reuse its integer descriptor number.
                try:
                    os.fstat(fd)
                except OSError as error:
                    closed.append(error.errno == errno.EBADF)
                else:
                    closed.append(False)
                removed.append(fd not in state.descriptors)
        elif frame.f_code is check_code and event == "call":
            parent = frame.f_back
            while parent is not None:
                if (
                    parent.f_code is close_code
                    and id(parent) in active
                    or parent.f_code is native_close_code
                    and id(parent) in native_active
                    or parent.f_code is finite_close_code
                    and id(parent) in finite_active
                ):
                    counts["checks"] += 1
                    break
                parent = parent.f_back

    sys.setprofile(observe)
    try:
        yield counts, closed, removed, active, operations
    finally:
        sys.setprofile(previous)
        assert not native_active and not finite_active


def test_original_config_parent_retirement_does_not_repeat_path_checks(
    configured_source, request
):
    with _original_tracked_closes(configured_source) as observed:
        with configured_source.locked_hooks_config_snapshot() as snapshot:
            assert (
                snapshot.config_path == configured_source._get_effective_config_path()
            )
    counts, closed, removed, active, operations = observed
    assert counts["closes"] > 0 and counts["parent_walk_closes"] > 0, counts
    assert not active
    assert len(closed) == counts["closes"] and all(closed), (counts, closed)
    assert all(removed), "successfully closed descriptors remained registered"
    assert operations
    assert all(operation not in raw._states for operation in operations)
    assert all(operation not in storage._raw_operations for operation in operations)
    request.node.user_properties.extend(
        ("original_native_retirement_" + name, value) for name, value in counts.items()
    )
    assert counts["checks"] == 0, counts


@contextmanager
def _owned_config_descriptor(source):
    with raw._scope(source, "config", writing=True) as operation:
        state = raw._states[operation]
        fd = private_paths._native_open(state.selected, private_paths.os.O_RDONLY)
        assert fd in state.descriptors
        assert state.pins and state.leases
        assert all(lease in storage._live_leases for lease in state.leases)
        os.fstat(fd)
        try:
            yield operation, state, fd
        finally:
            if fd in state.descriptors and not state.uncertain:
                private_paths._native_close(fd)


@contextmanager
def _original_close_checks():
    check_code = raw._check.__code__
    previous = sys.getprofile()
    calls = []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if frame.f_code is check_code and event == "call":
            calls.append(1)

    sys.setprofile(observe)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def _assert_closed(fd):
    with pytest.raises(OSError) as refused:
        os.fstat(fd)
    assert refused.value.errno == errno.EBADF


@pytest.mark.parametrize("revocation", ["participant", "selector"])
def test_creator_retires_owned_descriptor_after_source_revocation(
    configured_source, monkeypatch, tmp_path, revocation
):
    alternate = tmp_path / "alternate.toml"
    alternate.write_text('[general]\nusers_name="alternate"\n', encoding="utf-8")
    alternate.chmod(0o600)
    with _owned_config_descriptor(configured_source) as (operation, state, fd):
        outcome = private_paths._NativeOpenOutcome()
        try:
            with monkeypatch.context() as changed:
                if revocation == "participant":
                    changed.delitem(raw._source_participants, configured_source)
                else:
                    changed.setenv("TLDW_CONFIG_PATH", str(alternate))
                with _original_close_checks() as checks:
                    private_paths._native_close(fd)
                _assert_closed(fd)
                assert fd not in state.descriptors
                assert not checks
                # Cleanup authority cannot authorize the next effect.
                with pytest.raises(bootstrap.RecoveryRequired):
                    private_paths._native_open(
                        state.selected, private_paths.os.O_RDONLY, _outcome=outcome
                    )
                with pytest.raises(bootstrap.RecoveryRequired):
                    with raw._file(operation, state.selected, "r") as stream:
                        stream.read()
        finally:
            # A failed refusal assertion must still retire a positively issued fd.
            if outcome.descriptor is not None:
                private_paths._native_close(outcome.descriptor)
    assert operation not in raw._states
    assert operation not in storage._raw_operations


def test_owned_cleanup_finishes_after_gate_closes_without_admitting_new_work(
    configured_source,
):
    participant = raw._source_participants[configured_source]
    try:
        with _owned_config_descriptor(configured_source) as (operation, state, fd):
            participant.close_admission()
            assert not participant.drain(time.monotonic() + 0.03)
            with _original_close_checks() as checks:
                private_paths._native_close(fd)
            _assert_closed(fd)
            assert fd not in state.descriptors
            assert not checks
        assert operation not in raw._states
        assert participant.drain(time.monotonic() + 1)
        with pytest.raises(bootstrap.RecoveryRequired):
            with raw._scope(configured_source, "config", writing=True):
                pytest.fail("closed source admitted a new operation")
    finally:
        participant.resume()


@pytest.mark.parametrize("actor", ["foreign_thread", "inactive"])
def test_foreign_or_inactive_actor_keeps_original_close_refusal(
    configured_source, actor
):
    with _owned_config_descriptor(configured_source) as (operation, state, fd):
        errors = []

        def attempt_close():
            previous = getattr(raw._local, "operation", None)
            raw._local.operation = operation
            try:
                try:
                    private_paths._native_close(fd)
                except BaseException as error:
                    errors.append(error)
            finally:
                raw._local.operation = previous

        if actor == "foreign_thread":
            thread = threading.Thread(target=attempt_close)
            thread.start()
            thread.join(3)
            assert not thread.is_alive()
        else:
            state.active = False
            try:
                attempt_close()
            finally:
                state.active = True
        assert len(errors) == 1 and isinstance(errors[0], bootstrap.RecoveryRequired)
        os.fstat(fd)
        assert fd in state.descriptors
        assert not state.uncertain
    _assert_closed(fd)
    assert operation not in raw._states


def test_copied_operation_cannot_borrow_descriptor_retirement_ownership(
    configured_source,
):
    with _owned_config_descriptor(configured_source) as (operation, state, fd):
        copied = copy.copy(operation)
        assert copied not in raw._states
        raw._local.operation = copied
        try:
            assert raw._owned_descriptor_retirement_state(fd) is None
            # The ordinary unowned route remains separate; do not make a new
            # blanket-close refusal policy for copied or absent operations.
            assert raw._runtime_operation() is None
            assert fd in state.descriptors
            os.fstat(fd)
        finally:
            raw._local.operation = operation
    _assert_closed(fd)
    assert operation not in raw._states


def test_untracked_descriptor_preserves_original_checked_close(configured_source):
    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        fd = os.open(state.selected, os.O_RDONLY)
        closed = False
        try:
            assert fd not in state.descriptors
            assert raw._owned_descriptor_retirement_state(fd) is None
            with _original_close_checks() as checks:
                private_paths._native_close(fd)
            closed = True
            _assert_closed(fd)
            assert checks
            assert fd not in state.descriptors
        finally:
            if not closed:
                private_paths._native_close(fd)
    assert operation not in raw._states


@pytest.mark.asyncio
@private_profile_test
async def test_uncertain_owned_close_retains_native_resources_and_admission(
    configured_source, monkeypatch, request
):
    # This existing private-profile wrapper owns interpreter termination. The
    # test must not turn uncertain native custody back into a successful close.
    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        fd = private_paths._native_open(state.selected, private_paths.os.O_RDONLY)
        assert fd in state.descriptors
        pins, leases = dict(state.pins), tuple(state.leases)
        assert pins and leases
        original_close = raw.os.close
        attempts = []

        def fail_exact_close(descriptor):
            if descriptor == fd:
                attempts.append(descriptor)
                raise OSError("native close outcome is uncertain")
            return original_close(descriptor)

        with monkeypatch.context() as failure:
            failure.setattr(raw.os, "close", fail_exact_close)
            with pytest.raises(bootstrap.RecoveryRequired):
                private_paths._native_close(fd)
            assert attempts == [fd]
            assert state.uncertain
            with pytest.raises(bootstrap.RecoveryRequired):
                private_paths._native_close(fd)
            assert attempts == [fd], "uncertain close retried the native effect"
        os.fstat(fd)
        assert fd in state.descriptors
        assert state.pins == pins and tuple(state.leases) == leases
        for pinned in pins.values():
            os.fstat(pinned)
    assert not state.active
    assert raw._states[operation] is state
    assert operation in storage._raw_operations
    assert fd in state.descriptors
    assert all(lease in storage._live_leases for lease in leases)
    os.fstat(fd)
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.05)
    finally:
        pause.resume()
