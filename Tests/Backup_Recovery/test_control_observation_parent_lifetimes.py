"""Initial control readers borrow finite parents and preserve native refusals."""

import errno
import inspect
import stat
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_generation_witness_observation as witness_cases
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Utils import private_paths

witness_case = witness_cases.witness_case


def _observation_frame(frame, code, root):
    while frame is not None:
        if frame.f_code is code and frame.f_locals.get("root") == root:
            return frame
        frame = frame.f_back
    return None


@contextmanager
def _initial_parent_work(root, *, after_records=None):
    """Observe the actual initial pin and close bodies, including FD reuse."""
    observation_code = inspect.unwrap(bootstrap._control_observation).__code__
    pin_code = inspect.unwrap(bootstrap.pinned_directory).__code__
    child_code = inspect.unwrap(bootstrap._pinned_control_child).__code__
    component_code = private_paths._open_directory_component.__code__
    walk_code = private_paths._open_verified_parent.__code__
    close_code = bootstrap._native_close.__code__
    records_code = bootstrap._control_records.__code__
    registry_contents_code = bootstrap._registry_contents.__code__
    previous = sys.getprofile()
    observed = SimpleNamespace(
        opened=[],
        closed=[],
        active={},
        closing={},
        records=[],
        registry_reads=0,
        yields=0,
    )

    def opened(fd, path):
        assert fd not in observed.active
        assert stat.S_ISDIR(bootstrap.os.fstat(fd).st_mode)
        observed.opened.append((fd, path))
        observed.active[fd] = path

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        scope = _observation_frame(frame, observation_code, root)
        if frame.f_code is close_code:
            fd = frame.f_locals["fd"]
            if event == "call" and fd in observed.active:
                observed.closing[id(frame)] = fd
            elif event == "return" and id(frame) in observed.closing:
                fd = observed.closing.pop(id(frame))
                with pytest.raises(OSError) as closed:
                    bootstrap.os.fstat(fd)
                assert closed.value.errno == errno.EBADF
                observed.closed.append((fd, observed.active.pop(fd)))
        elif scope is not None:
            if (
                frame.f_code is walk_code
                and event == "return"
                and type(result) is tuple
                and frame.f_back.f_code is observation_code
                and frame.f_locals["selected"] == root
            ):
                ancestor, leaf = result
                assert leaf == root.name
                opened(ancestor, root.parent)
            elif frame.f_code is pin_code and event == "return" and type(result) is int:  # noqa: E721 -- actual native FD only.
                if frame.f_locals["root"] in (root, root / "admission"):
                    opened(result, frame.f_locals["root"])
            elif (
                frame.f_code is component_code
                and event == "return"
                and type(result) is int  # noqa: E721 -- actual native FD only.
                and frame.f_back.f_code is child_code
            ):
                parent = frame.f_locals["parent_fd"]
                assert parent in observed.active
                path = observed.active[parent] / frame.f_locals["component"]
                assert path in (root, root / "admission")
                opened(result, path)
            elif (
                frame.f_code is records_code
                and event == "return"
                and type(result) is tuple
            ):
                assert frame.f_locals["root"] == root
                observed.records.append(frame.f_locals.get("_parent"))
                if after_records is not None and len(observed.records) == 1:
                    after_records()
            elif frame.f_code is registry_contents_code and event == "call":
                observed.registry_reads += 1
            elif (
                frame.f_code is observation_code
                and event == "return"
                and type(result) is tuple
            ):
                assert not observed.active and not observed.closing
                observed.yields += 1

    sys.setprofile(observe)
    try:
        yield observed
    finally:
        sys.setprofile(previous)


def _assert_parents_retired(observed):
    assert observed.opened
    assert not observed.active and not observed.closing
    assert len(observed.closed) == len(observed.opened)
    assert sorted(observed.closed) == sorted(observed.opened)


def _make_shared(path):
    """Apply actual platform permissions and return their restoration."""
    mode = stat.S_IMODE(bootstrap.os.stat(path).st_mode)
    bootstrap.os.chmod(path, 0o777)
    assert bootstrap.os.stat(path).st_mode & 0o077
    return lambda: bootstrap.os.chmod(path, mode)


@pytest.mark.skipif(
    bootstrap.os.name == "nt",
    reason="POSIX native mode mutation; Windows uses original native DACL controls.",
)
@pytest.mark.parametrize("changed", ["root", "authority"])
def test_shared_initial_parent_rechecks_real_privacy_between_readers(
    witness_case, changed
):
    root, _control, _selector = witness_case
    target = root if changed == "root" else root / "admission"
    restorers = []

    def mutate():
        restorers.append(_make_shared(target))

    try:
        with _initial_parent_work(root, after_records=mutate) as observed:
            with pytest.raises((ValueError, OSError)):
                with bootstrap._control_observation(root):
                    pytest.fail("unsafe initial controls reached their consumer")
    finally:
        for restore in restorers:
            restore()
    assert len(restorers) == 1 and observed.records and observed.records[0] is not None
    assert observed.registry_reads == 0 and observed.yields == 0
    _assert_parents_retired(observed)


@pytest.mark.parametrize("missing", ["root", "authority"])
def test_shared_initial_parent_keeps_original_missing_control_results(
    witness_case, missing
):
    root, _control, _selector = witness_case
    selected = root.parent / ("absent-root" if missing == "root" else "bare-root")
    if missing == "authority":
        bootstrap.os.mkdir(selected, 0o700)
    try:
        expected = bootstrap._control_records(selected), bootstrap._registry(selected)
        assert expected == (([], [], []), None)
        with _initial_parent_work(selected) as observed:
            with bootstrap._control_observation(selected) as actual:
                assert actual == expected
                assert not observed.active and not observed.closing
        assert observed.yields == 1
        if missing == "authority":
            assert observed.records[0] is not None
            _assert_parents_retired(observed)
        else:
            assert observed.records == [None]
            assert all(path == selected.parent for _fd, path in observed.opened)
            _assert_parents_retired(observed)
    finally:
        if missing == "authority":
            bootstrap.os.rmdir(selected)


def test_shared_initial_parents_retire_after_actual_registry_parse_failure(
    witness_case,
):
    root, _control, _selector = witness_case
    registry = root / "admission" / "registry.json"
    before = registry.read_bytes()
    changed = []

    def mutate():
        registry.write_bytes(b"{")
        changed.append(True)

    try:
        with _initial_parent_work(root, after_records=mutate) as observed:
            with pytest.raises(ValueError):
                with bootstrap._control_observation(root):
                    pytest.fail("corrupt native registry reached its consumer")
    finally:
        registry.write_bytes(before)
    assert changed == [True] and observed.registry_reads == 1
    assert observed.records[0] is not None and observed.yields == 0
    assert {path for _fd, path in observed.opened} == {
        root.parent,
        root,
        root / "admission",
    }
    _assert_parents_retired(observed)


def test_custom_initial_reader_keeps_original_keyword_shape(witness_case, monkeypatch):
    root, _control, _selector = witness_case
    original = bootstrap._control_records
    expected = original(root), bootstrap._registry(root)
    calls = []

    def custom(path, *, _observed=None):
        calls.append((path, _observed))
        return original(path, _observed=_observed)

    with monkeypatch.context() as patch:
        patch.setattr(bootstrap, "_control_records", custom)
        with bootstrap._control_observation(root) as actual:
            assert actual == expected
    assert len(calls) == 1 and calls[0][0] == root
    assert isinstance(calls[0][1], dict)
    assert bootstrap._control_records is original


def test_late_initial_reader_change_refuses_and_retires_borrowed_root(
    witness_case, monkeypatch
):
    root, _control, _selector = witness_case
    original = bootstrap._registry
    calls = []

    def custom(path, *, _observed=None):
        calls.append(path)
        return original(path, _observed=_observed)

    with monkeypatch.context() as patch:

        def mutate():
            patch.setattr(bootstrap, "_registry", custom)

        with _initial_parent_work(root, after_records=mutate) as observed:
            with pytest.raises((ValueError, OSError, RuntimeError)):
                with bootstrap._control_observation(root):
                    pytest.fail("changed reader received issued borrowed root")
    assert not calls and observed.records[0] is not None
    assert observed.registry_reads == 0 and observed.yields == 0
    _assert_parents_retired(observed)
    assert bootstrap._registry is original


@pytest.mark.skipif(bootstrap.os.name == "nt", reason="Actual POSIX sticky /tmp parent")
def test_private_control_root_directly_below_sticky_parent_and_absence(witness_case):
    assert stat.S_ISVTX & bootstrap.os.stat("/tmp").st_mode
    with tempfile.TemporaryDirectory(prefix="control-root-", dir="/tmp") as directory:
        root = Path(directory)
        with _initial_parent_work(root) as observed:
            with bootstrap._control_observation(root) as payload:
                assert payload == (([], [], []), None)
                assert not observed.active
        assert observed.yields == 1 and observed.records[0] is not None
        _assert_parents_retired(observed)
        absent = root.with_name(root.name + "-absent")
        assert not absent.exists()
        with _initial_parent_work(absent) as missing:
            with bootstrap._control_observation(absent) as payload:
                assert payload == (([], [], []), None)
                assert not missing.active
        assert missing.yields == 1 and missing.records == [None]
        _assert_parents_retired(missing)
