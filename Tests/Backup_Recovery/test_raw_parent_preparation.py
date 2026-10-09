"""Original private parent walks retain finite source and native ownership."""

import copy
import errno
import stat
import sys
import threading
import time
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_raw_native_retirement as retirement_cases
from Tests.private_profile import private_profile_test
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Utils import private_paths as private

configured_source = retirement_cases.configured_source
local_root = retirement_cases.local_root


def _ancestor(frame, code):
    while frame is not None:
        if frame.f_code is code:
            return frame
        frame = frame.f_back
    return None


@contextmanager
def _original_parent_work(operation, *, selected=None, after_component=None):
    """Observe original walk/check/component/close bodies, never replace them."""
    state = raw._states[operation]
    walker_code = private._open_verified_parent.__code__
    body_code = getattr(
        private, "_walk_verified_parent", private._open_verified_parent
    ).__code__
    component_code = private._open_directory_component.__code__
    check_code = raw._check.__code__
    close_code = raw._close_descriptor.__code__
    previous = sys.getprofile()
    selected = state.selected if selected is None else selected
    observed = SimpleNamespace(
        phase="walk",
        full_checks=[],
        components=[],
        closes=[],
        active_closes={},
        prepared_openers=[],
    )

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        walker = _ancestor(frame.f_back, walker_code)
        if frame.f_code is check_code and event == "call":
            if frame.f_locals["operation"] is operation:
                observed.full_checks.append((observed.phase, walker is not None))
        elif (
            frame.f_code is component_code
            and event == "return"
            and type(result) is int  # noqa: E721 -- actual native descriptor only.
            and walker is not None
            and walker.f_locals["selected"] == selected
        ):
            assert result in state.descriptors
            assert stat.S_ISDIR(raw.os.fstat(result).st_mode)
            observed.components.append(result)
            opener = frame.f_locals.get("_open")
            if opener is not None:
                observed.prepared_openers.append(opener)
            if after_component is not None and len(observed.components) == 1:
                body = _ancestor(frame.f_back, body_code)
                assert body is not None
                after_component(observed, state, body, frame)
        elif frame.f_code is close_code and frame.f_locals["state"] is state:
            if event == "call":
                fd = frame.f_locals["fd"]
                raw.os.fstat(fd)
                observed.active_closes[id(frame)] = fd
            elif event == "return" and id(frame) in observed.active_closes:
                fd = observed.active_closes.pop(id(frame))
                try:
                    raw.os.fstat(fd)
                except OSError as error:
                    observed.closes.append(
                        (error.errno == errno.EBADF, fd not in state.descriptors)
                    )
                else:
                    observed.closes.append((False, fd not in state.descriptors))

    sys.setprofile(observe)
    try:
        yield observed
    finally:
        sys.setprofile(previous)


def test_original_parent_walk_checks_source_at_two_boundaries(
    configured_source, request
):
    selected = configured_source._get_effective_config_path()
    expected = selected.read_bytes()
    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        baseline_descriptors = set(state.descriptors)
        leases = tuple(state.leases)
        assert state.active and state.participant is not None and state.pins
        assert all(lease in storage._live_leases for lease in leases)
        parent_fd = leaf_fd = None
        with _original_parent_work(operation) as observed:
            try:
                parent_fd, leaf = private._open_verified_parent(
                    selected, missing_leaf_allowed=False
                )
                assert parent_fd in state.descriptors
                assert stat.S_ISDIR(raw.os.fstat(parent_fd).st_mode)
                assert leaf == selected.name
                assert len(observed.components) >= 2
                observed.phase = "leaf"
                leaf_fd = private._native_open(
                    leaf, private._PRIVATE_FILE_OPEN_FLAGS, dir_fd=parent_fd
                )
                assert leaf_fd in state.descriptors
                assert raw.os.read(leaf_fd, len(expected) + 1) == expected
            finally:
                observed.phase = "retirement"
                if leaf_fd is not None:
                    private._native_close(leaf_fd)
                if parent_fd is not None:
                    private._native_close(parent_fd)
        assert state.descriptors == baseline_descriptors
        assert not observed.active_closes
        assert observed.closes and all(
            closed and removed for closed, removed in observed.closes
        )
        assert all(lease in storage._live_leases for lease in leases)
    assert not state.active and not state.uncertain
    assert not state.pins and not state.files and not state.descriptors
    assert operation not in raw._states and operation not in storage._raw_operations
    assert all(lease not in storage._live_leases for lease in leases)
    assert selected.read_bytes() == expected
    walk_checks = [row for row in observed.full_checks if row == ("walk", True)]
    leaf_checks = [row for row in observed.full_checks if row == ("leaf", False)]
    assert leaf_checks, "the actual leaf allocation lost its fresh source gate"
    request.node.user_properties.extend(
        [
            ("original_parent_full_checks", len(walk_checks)),
            ("original_directory_allocations", len(observed.components)),
            ("original_leaf_full_checks", len(leaf_checks)),
            ("positive_native_descriptor_closes", len(observed.closes)),
        ]
    )
    assert len(walk_checks) == 2, observed.full_checks


@pytest.mark.parametrize(
    "change", ["selector", "source", "participant", "lease", "generation"]
)
def test_original_walk_refuses_drift_before_another_directory_or_return(
    configured_source, monkeypatch, change
):
    config_path = configured_source._get_effective_config_path()
    original = config_path.read_bytes()
    if change == "generation":
        directory = configured_source.get_user_data_dir()
        route, target = "config_data", directory
        selected = directory / "unopened-leaf"
    else:
        route, target, selected = "config", None, config_path
    alternate = config_path.with_name("alternate.toml")
    if change == "selector":
        alternate.write_bytes(original)
        alternate.chmod(0o600)
    changed, returned = [], []
    with raw._scope(
        configured_source, route, writing=True, selected_read=target
    ) as operation:
        state = raw._states[operation]
        baseline = set(state.descriptors)
        with monkeypatch.context() as patch:

            def mutate(_observed, actual, _walker, _frame):
                assert actual is state
                changed.append(change)
                if change == "selector":
                    patch.setenv("TLDW_CONFIG_PATH", str(alternate))
                elif change == "source":
                    patch.setitem(
                        sys.modules,
                        "tldw_chatbook.config",
                        ModuleType("displaced-config"),
                    )
                elif change == "participant":
                    patch.delitem(raw._source_participants, configured_source)
                elif change == "lease":
                    actual.leases[0].close()
                else:
                    patch.setattr(
                        configured_source,
                        "_CONFIG_GENERATION",
                        configured_source._CONFIG_GENERATION + 1,
                    )

            with _original_parent_work(
                operation, selected=selected, after_component=mutate
            ) as observed:
                with pytest.raises(bootstrap.RecoveryRequired):
                    parent, _leaf = private._open_verified_parent(
                        selected, missing_leaf_allowed=False
                    )
                    returned.append(parent)
        assert changed == [change] and not returned
        assert (
            len(observed.components) == 1
        ), "revoked custody allocated a later directory"
        assert state.descriptors == baseline
        assert not observed.active_closes
        assert observed.closes and all(
            closed and removed for closed, removed in observed.closes
        )
    assert not state.active and not state.uncertain
    assert operation not in raw._states and operation not in storage._raw_operations
    assert not state.pins and not state.files and not state.descriptors
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("callback", ["open", "close"])
def test_explicit_parent_callbacks_keep_original_walk_gates(
    configured_source, callback
):
    selected = configured_source._get_effective_config_path()
    calls = []

    def open_directory(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        return private._native_open(*args, **kwargs)

    def close_directory(fd):
        calls.append(fd)
        return private._native_close(fd)

    kwargs = (
        {"_open": open_directory} if callback == "open" else {"_close": close_directory}
    )
    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        baseline = set(state.descriptors)
        with _original_parent_work(operation) as observed:
            parent, leaf = private._open_verified_parent(
                selected, missing_leaf_allowed=False, **kwargs
            )
            assert parent in state.descriptors and leaf == selected.name
            private._native_close(parent)
        assert state.descriptors == baseline
        assert observed.components and calls
        assert (
            sum(phase == "walk" and in_walk for phase, in_walk in observed.full_checks)
            > 2
        )
        assert observed.closes and all(
            closed and removed for closed, removed in observed.closes
        )
        if callback == "open":
            assert len(calls) == len(observed.components) + 1
            assert all(len(args) == 2 for args, _kwargs in calls)
        else:
            assert len(calls) == len(observed.components)
    assert operation not in raw._states and operation not in storage._raw_operations


def test_finite_directory_opener_refuses_foreign_copy_and_expired_use(
    configured_source,
):
    selected = configured_source._get_effective_config_path()
    failures, captured = [], []
    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        baseline = set(state.descriptors)

        def attempt(_observed, actual, _walker, frame):
            assert actual is state
            opener = frame.f_locals.get("_open")
            assert opener is not None, "the actual finite opener was not reached"
            closer = _walker.f_locals.get("_close")
            assert closer is not None
            captured.append((opener, closer))
            expected = set(state.descriptors)
            with pytest.raises(bootstrap.RecoveryRequired):
                raw._check_directory_allocation(
                    copy.copy(operation), state, state.source, state.participant
                )

            def foreign():
                try:
                    opener(
                        selected.anchor,
                        private._DIRECTORY_OPEN_FLAGS | private._NOFOLLOW,
                    )
                except BaseException as error:
                    failures.append(error)

            worker = threading.Thread(target=foreign)
            worker.start()
            worker.join(3)
            assert not worker.is_alive()
            assert len(failures) == 1 and isinstance(
                failures[0], bootstrap.RecoveryRequired
            )
            assert state.descriptors == expected

        with _original_parent_work(operation, after_component=attempt):
            parent, _leaf = private._open_verified_parent(
                selected, missing_leaf_allowed=False
            )
        try:
            assert len(captured) == 1
            opener, closer = captured[0]
            with pytest.raises(bootstrap.RecoveryRequired):
                opener(
                    selected.anchor, private._DIRECTORY_OPEN_FLAGS | private._NOFOLLOW
                )
            later = private._native_open(
                selected.name, private._PRIVATE_FILE_OPEN_FLAGS, dir_fd=parent
            )
            try:
                with pytest.raises(bootstrap.RecoveryRequired):
                    closer(later)
                assert later in state.descriptors
                raw.os.fstat(later)
            finally:
                private._native_close(later)
        finally:
            private._native_close(parent)
        assert state.descriptors == baseline
    assert operation not in raw._states and operation not in storage._raw_operations


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("closed_before_error", [False, True])
async def test_walk_uncertain_close_retains_actual_descriptors_and_leases(
    configured_source, monkeypatch, request, closed_before_error
):
    selected = configured_source._get_effective_config_path()
    original_close = raw.os.close
    targets, attempts = [], []
    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        pins, leases = dict(state.pins), tuple(state.leases)

        def fail_close(fd):
            if targets and fd == targets[0]:
                attempts.append(fd)
                if closed_before_error:
                    original_close(fd)
                raise OSError("native directory close outcome is uncertain")
            return original_close(fd)

        with monkeypatch.context() as patch:

            def arm(_observed, _actual, walker, _frame):
                targets.append(walker.f_locals["current_fd"])
                raw.os.fstat(targets[0])
                patch.setattr(raw.os, "close", fail_close)

            with _original_parent_work(operation, after_component=arm):
                with pytest.raises(bootstrap.RecoveryRequired):
                    private._open_verified_parent(selected, missing_leaf_allowed=False)
            assert targets and attempts == targets
        assert state.uncertain and targets[0] in state.descriptors
        assert state.pins == pins and tuple(state.leases) == leases
        if closed_before_error:
            with pytest.raises(OSError):
                raw.os.fstat(targets[0])
        else:
            raw.os.fstat(targets[0])
    assert not state.active and raw._states[operation] is state
    assert operation in storage._raw_operations
    assert state.descriptors and all(lease in storage._live_leases for lease in leases)
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.05)
    finally:
        pause.resume()


def test_final_parent_source_check_refuses_and_closes_completed_walk(
    configured_source, monkeypatch
):
    selected = configured_source._get_effective_config_path()
    before = selected.read_bytes()
    alternate = selected.with_name("final-check-alternate.toml")
    alternate.write_bytes(before)
    alternate.chmod(0o600)
    code = private._walk_verified_parent.__code__
    returned, delivered = [], []
    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        baseline = set(state.descriptors)
        with monkeypatch.context() as patch:
            previous = sys.getprofile()

            def observe(frame, event, result):
                if previous is not None:
                    previous(frame, event, result)
                if (
                    frame.f_code is code
                    and event == "return"
                    and type(result) is tuple
                    and not returned
                ):
                    fd, leaf = result
                    assert fd in state.descriptors and leaf == selected.name
                    assert stat.S_ISDIR(raw.os.fstat(fd).st_mode)
                    returned.append(fd)
                    patch.setenv("TLDW_CONFIG_PATH", str(alternate))

            sys.setprofile(observe)
            try:
                with pytest.raises(bootstrap.RecoveryRequired):
                    delivered.append(
                        private._open_verified_parent(
                            selected, missing_leaf_allowed=False
                        )
                    )
            finally:
                sys.setprofile(previous)
        assert len(returned) == 1 and not delivered
        with pytest.raises(OSError) as closed:
            raw.os.fstat(returned[0])
        assert closed.value.errno == errno.EBADF
        assert state.descriptors == baseline
    assert operation not in raw._states and operation not in storage._raw_operations
    assert not state.active and not state.uncertain
    assert not state.pins and not state.files and not state.descriptors
    assert selected.read_bytes() == before


def test_changed_directory_open_default_preserves_original_callback(
    configured_source, monkeypatch
):
    selected = configured_source._get_effective_config_path()
    function = private._open_directory_component
    code, keywords = function.__code__, function.__kwdefaults__
    assert keywords["_open"] is None
    calls = []

    def directory_open(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        return private._native_open(*args, **kwargs)

    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        baseline = set(state.descriptors)
        with monkeypatch.context() as patch:
            patch.setitem(keywords, "_open", directory_open)
            assert (
                private._open_directory_component is function
                and function.__code__ is code
            )
            assert function.__kwdefaults__ is keywords
            with _original_parent_work(operation) as observed:
                parent, leaf = private._open_verified_parent(
                    selected, missing_leaf_allowed=False
                )
                assert parent in state.descriptors and leaf == selected.name
                private._native_close(parent)
            assert calls and len(calls) == len(observed.components)
            assert all(
                len(args) == 2 and set(kwargs) == {"dir_fd"} for args, kwargs in calls
            )
            assert (
                sum(
                    phase == "walk" and in_walk
                    for phase, in_walk in observed.full_checks
                )
                > 2
            )
            assert observed.closes and all(
                closed and removed for closed, removed in observed.closes
            )
            assert state.descriptors == baseline
        assert function.__kwdefaults__ is keywords and keywords["_open"] is None
    assert operation not in raw._states and operation not in storage._raw_operations


@pytest.mark.skipif(
    raw.os.name == "nt",
    reason="Native parent rename is refused in this installed Windows config-scope fixture.",
)
def test_transient_parent_substitution_refuses_and_retires_returned_descriptor(
    configured_source, request
):
    """A restored lexical parent cannot certify an FD opened on its substitute."""
    selected = configured_source._get_effective_config_path()
    parent = selected.parent
    original = parent.with_name(parent.name + "-retained-parent")
    replacement = parent.with_name(parent.name + "-substituted-parent")
    assert not original.exists() and not replacement.exists()
    before = selected.read_bytes()
    grandparent_identity = raw.os.stat(parent.parent)
    allocation_code = raw._check_directory_allocation.__code__
    component_code = private._open_directory_component.__code__
    walker_code = private._walk_verified_parent.__code__
    check_code = raw._check.__code__
    swapped = restored = False
    returned, final_checks, delivered, refusals = [], [], [], []
    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        baseline = set(state.descriptors)
        leases = tuple(state.leases)
        pin = state.pins[parent]
        pinned_identity = raw.os.fstat(pin)
        assert private._same_identity(raw.os.stat(parent), pinned_identity)
        previous = sys.getprofile()

        def observe(frame, event, result):
            nonlocal swapped, restored
            if previous is not None:
                previous(frame, event, result)
            if event != "return":
                return
            component = _ancestor(frame, component_code)
            walker = _ancestor(frame, walker_code)
            if (
                frame.f_code is allocation_code
                and result is state
                and frame.f_locals["operation"] is operation
                and component is not None
                and walker is not None
                and walker.f_locals["selected"] == selected
                and not walker.f_locals["pending"]
                and component.f_locals["component"] == parent.name
                and not swapped
            ):
                assert private._same_identity(
                    raw.os.fstat(component.f_locals["parent_fd"]),
                    grandparent_identity,
                )
                # The original reduced source check has already completed.
                raw.os.rename(parent, original)
                swapped = True
                raw.os.mkdir(parent, 0o700)
                assert stat.S_IMODE(raw.os.stat(parent).st_mode) == 0o700
                assert not private._same_identity(raw.os.stat(parent), pinned_identity)
            elif (
                frame.f_code is component_code
                and type(result) is int  # noqa: E721 -- actual native FD only.
                and swapped
                and not restored
                and walker is not None
                and walker.f_locals["selected"] == selected
                and not walker.f_locals["pending"]
                and frame.f_locals["component"] == parent.name
            ):
                assert result in state.descriptors
                opened = raw.os.fstat(result)
                assert stat.S_ISDIR(opened.st_mode)
                assert private._same_identity(opened, raw.os.stat(parent))
                assert not private._same_identity(opened, pinned_identity)
                returned.append(result)
                # Restore the original lexical source before the final proof.
                raw.os.rename(parent, replacement)
                raw.os.rename(original, parent)
                restored = True
                assert state.pins[parent] == pin
                assert private._same_identity(raw.os.stat(parent), raw.os.fstat(pin))
                assert not private._same_identity(
                    raw.os.fstat(result), raw.os.fstat(pin)
                )
            elif (
                frame.f_code is check_code
                and result is state
                and frame.f_locals["operation"] is operation
                and restored
            ):
                final_checks.append(result)

        sys.setprofile(observe)
        try:
            try:
                delivered.append(
                    private._open_verified_parent(selected, missing_leaf_allowed=False)
                )
            except bootstrap.RecoveryRequired as error:
                refusals.append(error)
        finally:
            sys.setprofile(previous)
            # Also clean the incorrectly delivered FD on the expected RED.
            for fd, _leaf in delivered:
                private._native_close(fd)
            if original.exists():
                if parent.exists():
                    raw.os.rename(parent, replacement)
                raw.os.rename(original, parent)
            if replacement.exists():
                raw.os.rmdir(replacement)
        assert swapped and restored and len(returned) == 1
        assert final_checks, "the original final full source proof was not reached"
        with pytest.raises(OSError) as closed:
            raw.os.fstat(returned[0])
        assert closed.value.errno == errno.EBADF
        assert state.descriptors == baseline
        assert state.pins[parent] == pin
        assert private._same_identity(raw.os.stat(parent), raw.os.fstat(pin))
        assert all(lease in storage._live_leases for lease in leases)
    assert operation not in raw._states and operation not in storage._raw_operations
    assert not state.active and not state.uncertain
    assert not state.pins and not state.files and not state.descriptors
    assert all(lease not in storage._live_leases for lease in leases)
    assert selected.read_bytes() == before
    request.node.user_properties.extend(
        [
            ("actual_substitute_parent_descriptors", len(returned)),
            ("restored_original_final_source_checks", len(final_checks)),
            ("incorrectly_delivered_parent_descriptors", len(delivered)),
        ]
    )
    assert len(refusals) == 1 and not delivered


def test_unpinned_private_parent_keeps_original_walk_checks(configured_source, request):
    """A safe unretained parent uses the original discovery for every open."""
    selected = configured_source._get_effective_config_path()
    directory = selected.parent / "unretained-private-parent"
    raw.os.mkdir(directory, 0o700)
    auxiliary = directory / "unopened-leaf"
    try:
        assert stat.S_IMODE(raw.os.stat(directory).st_mode) == 0o700
        with raw._scope(configured_source, "config", writing=True) as operation:
            state = raw._states[operation]
            baseline = set(state.descriptors)
            leases = tuple(state.leases)
            assert selected.parent in state.pins and directory not in state.pins
            with _original_parent_work(operation, selected=auxiliary) as observed:
                parent, leaf = private._open_verified_parent(
                    auxiliary, missing_leaf_allowed=True
                )
                try:
                    assert leaf == auxiliary.name and parent in state.descriptors
                    assert private._same_identity(
                        raw.os.fstat(parent), raw.os.stat(directory)
                    )
                    assert stat.S_ISDIR(raw.os.fstat(parent).st_mode)
                finally:
                    observed.phase = "retirement"
                    private._native_close(parent)
            assert state.descriptors == baseline
            assert observed.components and not observed.prepared_openers
            assert observed.closes and all(
                closed and removed for closed, removed in observed.closes
            )
            assert all(lease in storage._live_leases for lease in leases)
        assert operation not in raw._states and operation not in storage._raw_operations
        assert not state.active and not state.uncertain
        assert not state.pins and not state.files and not state.descriptors
        assert all(lease not in storage._live_leases for lease in leases)
        checks = sum(row == ("walk", True) for row in observed.full_checks)
        request.node.user_properties.append(("unpinned_original_full_checks", checks))
        assert checks > 2, observed.full_checks
    finally:
        raw.os.rmdir(directory)


def test_original_returned_parent_descriptor_substitution_refuses_and_retires(
    configured_source, request
):
    """The original final source proof cannot certify a substituted native FD."""
    selected = configured_source._get_effective_config_path()
    before = selected.read_bytes()
    replacement = selected.parent.with_name(
        selected.parent.name + "-descriptor-substitute"
    )
    raw.os.mkdir(replacement, 0o700)
    substitute = None
    replaced = False
    returned, final_checks, delivered, refusals = [], [], [], []
    walker_code = private._walk_verified_parent.__code__
    check_code = raw._check.__code__
    try:
        assert stat.S_IMODE(raw.os.stat(replacement).st_mode) == 0o700
        substitute = raw.os.open(
            replacement, private._DIRECTORY_OPEN_FLAGS | private._NOFOLLOW
        )
        substitute_identity = raw.os.fstat(substitute)
        with raw._scope(configured_source, "config", writing=True) as operation:
            state = raw._states[operation]
            baseline = set(state.descriptors)
            leases = tuple(state.leases)
            pin = state.pins[selected.parent]
            pinned_identity = raw.os.fstat(pin)
            assert not private._same_identity(substitute_identity, pinned_identity)
            previous = sys.getprofile()

            def observe(frame, event, result):
                nonlocal replaced
                if previous is not None:
                    previous(frame, event, result)
                if event != "return":
                    return
                if (
                    frame.f_code is walker_code
                    and frame.f_locals["selected"] == selected
                    and type(result) is tuple
                    and not replaced
                ):
                    fd, leaf = result
                    assert fd in state.descriptors and leaf == selected.name
                    assert private._same_identity(raw.os.fstat(fd), pinned_identity)
                    returned.append(fd)
                    # Replace the actual native descriptor, keeping the original
                    # source, pin and ownership metadata entirely untouched.
                    raw.os.dup2(substitute, fd)
                    replaced = True
                    assert fd in state.descriptors
                    assert private._same_identity(raw.os.fstat(fd), substitute_identity)
                    assert not private._same_identity(
                        raw.os.fstat(fd), raw.os.fstat(pin)
                    )
                    assert private._same_identity(
                        raw.os.stat(selected.parent), raw.os.fstat(pin)
                    )
                elif (
                    frame.f_code is check_code
                    and result is state
                    and frame.f_locals["operation"] is operation
                    and replaced
                ):
                    final_checks.append(result)

            sys.setprofile(observe)
            try:
                try:
                    delivered.append(
                        private._open_verified_parent(
                            selected, missing_leaf_allowed=False
                        )
                    )
                except bootstrap.RecoveryRequired as error:
                    refusals.append(error)
            finally:
                sys.setprofile(previous)
                # Also retire actual resources if the current candidate wrongly
                # delivers the substituted descriptor or the observer raises.
                for fd in returned:
                    if fd in state.descriptors:
                        private._native_close(fd)
            assert replaced and len(returned) == 1
            assert final_checks, "the original final full source proof was not reached"
            with pytest.raises(OSError) as closed:
                raw.os.fstat(returned[0])
            assert closed.value.errno == errno.EBADF
            assert state.descriptors == baseline
            assert state.pins[selected.parent] == pin
            assert private._same_identity(
                raw.os.stat(selected.parent), raw.os.fstat(pin)
            )
            assert all(lease in storage._live_leases for lease in leases)
        assert operation not in raw._states and operation not in storage._raw_operations
        assert not state.active and not state.uncertain
        assert not state.pins and not state.files and not state.descriptors
        assert all(lease not in storage._live_leases for lease in leases)
        assert selected.read_bytes() == before
    finally:
        if substitute is not None:
            raw.os.close(substitute)
            with pytest.raises(OSError) as closed:
                raw.os.fstat(substitute)
            assert closed.value.errno == errno.EBADF
        raw.os.rmdir(replacement)
    request.node.user_properties.extend(
        [
            ("actual_substituted_returned_parent_descriptors", len(returned)),
            ("original_final_source_checks_after_substitution", len(final_checks)),
            ("incorrectly_delivered_parent_descriptors", len(delivered)),
        ]
    )
    assert len(refusals) == 1 and not delivered


def test_final_parent_identity_read_refuses_late_real_lease_revocation(
    configured_source, request
):
    """Custody revoked by the final native read cannot publish the parent FD."""
    selected = configured_source._get_effective_config_path()
    before = selected.read_bytes()
    factory_code = private._prepared_parent_walk.__wrapped__.__code__
    finish_code = next(
        value
        for value in factory_code.co_consts
        if isinstance(value, type(factory_code)) and value.co_name == "finish"
    )
    fstat = raw.os.fstat
    fstat_code = getattr(fstat, "__code__", None)
    identity_reads, returned, revoked, delivered, refusals = [], [], [], [], []
    with raw._scope(configured_source, "config", writing=True) as operation:
        state = raw._states[operation]
        baseline = set(state.descriptors)
        leases = tuple(state.leases)
        pin = state.pins[selected.parent]
        previous = sys.getprofile()

        def observe(frame, event, result):
            if previous is not None:
                previous(frame, event, result)
            # The Windows facade has a Python return; POSIX uses c_return.
            # Both must come immediately from the actual finite finish body.
            if (
                fstat_code is not None
                and event == "return"
                and frame.f_code is fstat_code
                and frame.f_back is not None
                and frame.f_back.f_code is finish_code
            ):
                finish = frame.f_back
            elif (
                fstat_code is None
                and event == "c_return"
                and result is fstat
                and frame.f_code is finish_code
            ):
                finish = frame
            else:
                return
            assert finish.f_locals["operation"] is operation
            assert finish.f_locals["state"] is state
            identity_reads.append(event)
            if len(identity_reads) != 2:
                return
            parent = finish.f_locals["parent"]
            assert parent in state.descriptors and state.pins[selected.parent] == pin
            assert private._same_identity(fstat(parent), fstat(pin))
            assert stat.S_ISDIR(fstat(parent).st_mode)
            returned.append(parent)
            lease = leases[-1]
            assert lease in storage._live_leases
            lease.close()
            assert lease not in storage._live_leases
            revoked.append(lease)

        sys.setprofile(observe)
        try:
            try:
                delivered.append(
                    private._open_verified_parent(selected, missing_leaf_allowed=False)
                )
            except bootstrap.RecoveryRequired as error:
                refusals.append(error)
        finally:
            sys.setprofile(previous)
            for fd in returned:
                if fd in state.descriptors:
                    private._native_close(fd)
        assert len(identity_reads) == 2 and len(returned) == 1
        assert revoked == [leases[-1]]
        with pytest.raises(OSError) as closed:
            fstat(returned[0])
        assert closed.value.errno == errno.EBADF
        assert state.descriptors == baseline
        assert state.pins[selected.parent] == pin
        assert revoked[0] not in storage._live_leases
    assert operation not in raw._states and operation not in storage._raw_operations
    assert not state.active and not state.uncertain
    assert not state.pins and not state.files and not state.descriptors
    assert all(lease not in storage._live_leases for lease in leases)
    assert selected.read_bytes() == before
    request.node.user_properties.extend(
        [
            ("original_finish_identity_reads", len(identity_reads)),
            ("late_real_lease_revocations", len(revoked)),
            ("incorrectly_delivered_parent_descriptors", len(delivered)),
        ]
    )
    assert len(refusals) == 1 and not delivered
