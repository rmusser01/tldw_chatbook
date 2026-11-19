"""ACL fixture cleanup must restore and verify the captured native state."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def _fixture(
    monkeypatch,
    *,
    changed=None,
    restore_error=0,
    control=0x9004,
    initialized=None,
    initialize_error=0,
    read_failure_at=None,
):
    from Tests.Backup_Recovery import windows_acl_fixture as helper

    original = ("owner-sid", control, 1, b"original-dacl")
    state = {"security": original, "released": 0, "restored": [], "reads": 0}

    @contextmanager
    def security(native, handle):
        try:
            state["reads"] += 1
            if state["reads"] == read_failure_at:
                raise RuntimeError("fixture native read failed")
            yield "original-pointer", state["security"]
        finally:
            state["released"] += 1

    def restore(handle, kind, flags, owner, group, dacl, sacl):
        state["restored"].append((handle, kind, flags, owner, group, dacl, sacl))
        initial = len(state["restored"]) == 1
        error = initialize_error if initial else restore_error
        if not error:
            state["security"] = (
                initialized or original
                if initial
                else changed or initialized or original
            )
        return error

    monkeypatch.setattr(helper, "_security", security)
    native = SimpleNamespace(advapi=SimpleNamespace(SetSecurityInfo=restore))
    return helper, native, original, state


def test_original_dacl_restored_after_body_failure(monkeypatch):
    helper, native, original, state = _fixture(monkeypatch)
    error = RuntimeError("fixture body failed")
    with pytest.raises(RuntimeError) as caught, helper._preserve_dacl(native, 7):
        state["security"] = ("owner-sid", 0x9004, 1, b"public-dacl")
        raise error
    assert caught.value is error
    assert state["security"] == original
    assert state["released"] == 3
    assert (
        state["restored"]
        == [(7, 1, 4 | 0x80000000, None, None, "original-pointer", None)] * 2
    )


@pytest.mark.parametrize(
    "changed",
    [
        ("other-owner", 0x9004, 1, b"original-dacl"),
        ("owner-sid", 0x9404, 1, b"original-dacl"),
        ("owner-sid", 0x9004, 2, b"original-dacl"),
        ("owner-sid", 0x9004, 1, b"different-dacl"),
    ],
)
def test_native_restoration_mismatch_is_not_normalized(monkeypatch, changed):
    helper, native, _, state = _fixture(monkeypatch, changed=changed)
    with (
        pytest.raises(AssertionError, match="original native security not restored"),
        helper._preserve_dacl(native, 7),
    ):
        pass
    assert state["released"] == 3


def test_native_restore_error_fails_and_releases_original(monkeypatch):
    helper, native, _, state = _fixture(monkeypatch, restore_error=5)
    with pytest.raises(OSError) as caught, helper._preserve_dacl(native, 7):
        pass
    assert caught.value.errno == 5
    assert state["released"] == 2


@pytest.mark.parametrize("restored_acl", [b"sensitive-restored-acl", None])
def test_restoration_mismatch_reports_only_bounded_metadata(monkeypatch, restored_acl):
    changed = ("sensitive-restored-owner", 0x9404, 2, restored_acl)
    helper, native, original, state = _fixture(monkeypatch, changed=changed)
    with pytest.raises(AssertionError) as caught, helper._preserve_dacl(native, 7):
        pass
    expected = (
        "original native security not restored: owner_equal=False "
        "original_control=36868 restored_control=37892 control_xor=1024 "
        "original_revision=1 restored_revision=2 "
        f"original_acl_length=13 restored_acl_length={len(restored_acl or b'')} "
        "acl_equal=False"
    )
    assert str(caught.value) == expected
    assert original[0] not in expected
    assert changed[0] not in expected
    assert "original-dacl" not in expected
    assert "sensitive-restored-acl" not in expected
    assert state["released"] == 3


def test_unprotected_original_is_not_changed_to_protected(monkeypatch):
    helper, native, original, state = _fixture(monkeypatch, control=0x8004)
    with helper._preserve_dacl(native, 7):
        pass
    assert state["restored"][0][2] == 4 | 0x20000000
    assert state["security"] == original


def test_restore_failure_retains_the_original_body_error(monkeypatch):
    helper, native, _, state = _fixture(monkeypatch, restore_error=5)
    original = RuntimeError("fixture body failed")
    with pytest.raises(OSError) as caught, helper._preserve_dacl(native, 7):
        raise original
    assert caught.value.__context__ is original
    assert state["released"] == 2


def test_windows_inheritance_bookkeeping_is_established_before_body(monkeypatch):
    initialized = ("owner-sid", 0x9404, 1, b"original-dacl")
    helper, native, _, state = _fixture(monkeypatch, initialized=initialized)
    with helper._preserve_dacl(native, 7):
        assert state["security"] == initialized
        state["security"] = ("owner-sid", 0x9404, 1, b"public-dacl")
    assert state["security"] == initialized
    assert state["released"] == 3


@pytest.mark.parametrize(
    "initialized",
    [
        ("other-owner", 0x9404, 1, b"original-dacl"),
        ("owner-sid", 0x8404, 1, b"original-dacl"),
        ("owner-sid", 0x9504, 1, b"original-dacl"),
        ("owner-sid", 0x9404, 2, b"original-dacl"),
        ("owner-sid", 0x9404, 1, b"different-dacl"),
    ],
)
def test_initialization_refuses_security_changes_before_body(monkeypatch, initialized):
    helper, native, _, state = _fixture(monkeypatch, initialized=initialized)
    entered = []
    with (
        pytest.raises(AssertionError, match="fixture security initialization"),
        helper._preserve_dacl(native, 7),
    ):
        entered.append(True)
    assert not entered
    assert state["released"] == 2


def test_initialization_error_prevents_body_and_releases_descriptor(monkeypatch):
    helper, native, _, state = _fixture(monkeypatch, initialize_error=5)
    entered = []
    with pytest.raises(OSError) as caught, helper._preserve_dacl(native, 7):
        entered.append(True)
    assert caught.value.errno == 5
    assert not entered
    assert state["released"] == 1


def test_final_cleanup_does_not_mask_inheritance_bit_loss(monkeypatch):
    helper, native, _, state = _fixture(
        monkeypatch,
        initialized=("owner-sid", 0x9404, 1, b"original-dacl"),
        changed=("owner-sid", 0x9004, 1, b"original-dacl"),
    )
    with (
        pytest.raises(AssertionError, match="original native security not restored"),
        helper._preserve_dacl(native, 7),
    ):
        pass
    assert state["released"] == 3


def test_already_normalized_security_remains_exact(monkeypatch):
    helper, native, original, state = _fixture(monkeypatch, control=0x9404)
    with helper._preserve_dacl(native, 7):
        assert state["security"] == original
    assert state["security"] == original
    assert len(state["restored"]) == 2


def test_initialization_cannot_clear_inheritance_bit(monkeypatch):
    helper, native, _, state = _fixture(
        monkeypatch,
        control=0x9404,
        initialized=("owner-sid", 0x9004, 1, b"original-dacl"),
    )
    entered = []
    with (
        pytest.raises(AssertionError, match="fixture security initialization"),
        helper._preserve_dacl(native, 7),
    ):
        entered.append(True)
    assert not entered
    assert len(state["restored"]) == 2


@pytest.mark.parametrize("cleanup_error", [0, 5])
def test_setup_read_failure_keeps_bounded_cleanup(monkeypatch, cleanup_error):
    helper, native, _, state = _fixture(
        monkeypatch,
        read_failure_at=2,
        restore_error=cleanup_error,
    )
    entered = []
    error = OSError if cleanup_error else RuntimeError
    with pytest.raises(error) as caught, helper._preserve_dacl(native, 7):
        entered.append(True)
    if cleanup_error:
        assert isinstance(caught.value.__context__, RuntimeError)
    assert not entered
    assert len(state["restored"]) == 2
    assert state["released"] == 2


@pytest.fixture
def _native_acl_directory(tmp_path):
    import ctypes as C
    import errno
    import os

    if os.name != "nt":
        pytest.skip("requires actual Windows ACL mutation and retirement")
    from Tests.Backup_Recovery import windows_acl_fixture as helper
    from tldw_chatbook.Backup_Recovery.native_files import pinned_directory
    from tldw_chatbook.Utils import platform_files, windows_files

    native, facade = windows_files._native(), platform_files.os
    assert type(native) is windows_files._Native
    assert type(facade) is windows_files.WindowsOS
    assert facade.open.__self__ is facade
    assert facade.open.__func__ is windows_files.WindowsOS.open
    assert facade.close is os.close
    assert type(native.kernel) is type(native.advapi) is C.WinDLL
    assert native.kernel._name == "kernel32" and native.advapi._name == "advapi32"
    path = tmp_path / "native-acl-helper"
    path.mkdir(mode=0o700)
    facade.chmod(path, 0o700)
    # A fresh typed probe leaves every product DLL function signature unchanged.
    probe = native.kernel["GetHandleInformation"]
    probe.argtypes = [windows_files._HANDLE, C.POINTER(C.c_uint32)]
    probe.restype = C.c_int32

    def handle_closed(handle):
        previous = C.get_last_error()
        try:
            C.set_last_error(0)
            flags = C.c_uint32()
            result = probe(handle, C.byref(flags))
            return result == 0 and C.get_last_error() == 6
        finally:
            C.set_last_error(previous)

    def fd_closed(fd):
        try:
            os.fstat(fd)
        except OSError as error:
            return error.errno == errno.EBADF
        return False

    def identity(handle):
        info = native.info(handle)
        return info.volume, info.index_high, info.index_low

    with (
        pinned_directory(path) as anchor_fd,
        native.reopen(
            native.handle(anchor_fd),
            windows_files._READ_CONTROL | windows_files._WRITE_DAC,
        ) as anchor,
    ):
        original_identity = identity(anchor)
        original_stat = facade.fstat(anchor_fd)
        with helper._preserve_dacl(native, anchor):
            with helper._security(native, anchor) as (_, baseline):
                # The original helper establishes inheritance bookkeeping first.
                baseline = tuple(baseline)

            def security():
                with helper._security(native, anchor) as (_, current):
                    return tuple(current)

            yield SimpleNamespace(
                C=C,
                helper=helper,
                native=native,
                facade=facade,
                path=path,
                identity=identity,
                original_identity=original_identity,
                stat_identity=(original_stat.st_dev, original_stat.st_ino),
                baseline=baseline,
                security=security,
                handle_closed=handle_closed,
                fd_closed=fd_closed,
            )
        with helper._security(native, anchor) as (_, restored):
            exact = tuple(restored) == baseline
            metadata = {
                "owner_equal": restored[0] == baseline[0],
                "control_equal": restored[1] == baseline[1],
                "revision_equal": restored[2] == baseline[2],
                "acl_equal": restored[3] == baseline[3],
            }
        assert exact, metadata
        assert identity(anchor) == original_identity
    assert handle_closed(anchor)
    assert fd_closed(anchor_fd)


@contextmanager
def _observe_native_acl_call(case, monkeypatch, sddl, *, after_set_error=None):
    """Forward one real helper call; preserve its outcomes while noting releases."""
    import threading

    C, native, facade = case.C, case.native, case.facade
    advapi, kernel = native.advapi, native.kernel
    original_open, original_close = facade.open, facade.close
    close_was_local = "close" in vars(facade)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    set_security = advapi.SetSecurityInfo
    open_by_id, close_handle, local_free = (
        kernel.OpenFileById,
        kernel.CloseHandle,
        kernel.LocalFree,
    )
    actor = threading.get_ident()
    receipt = {
        "conversion_calls": 0,
        "conversion_success": None,
        "conversion_error": 0,
        "descriptor": None,
        "descriptor_frees": 0,
        "fd_opens": 0,
        "fd_closes": 0,
        "reopens": 0,
        "handle_closes": 0,
        "set_calls": 0,
        "set_success": False,
        "set_identity": False,
        "set_arguments": False,
        "controlled_error": False,
        "errors": [],
        "overflow": False,
        "restored": False,
    }
    fds, handles = {}, {}
    receipt["live_fds"], receipt["live_handles"] = fds, handles

    def address(value):
        return C.cast(value, C.c_void_p).value

    def observe(label, action):
        previous = C.get_last_error()
        try:
            if threading.get_ident() != actor:
                raise AssertionError("foreign_native_acl_observation")
            action()
        except BaseException as error:
            if len(receipt["errors"]) < 8:
                receipt["errors"].append(label + ":" + type(error).__name__)
            else:
                receipt["overflow"] = True
        finally:
            C.set_last_error(previous)

    def converting(*args):
        result = convert(*args)
        native_error = C.get_last_error()
        if args[0] is sddl:

            def note():
                receipt["conversion_calls"] += 1
                receipt["conversion_success"] = bool(result)
                receipt["conversion_error"] = native_error
                receipt["descriptor"] = C.cast(args[2], C.POINTER(C.c_void_p))[0]

            observe("conversion", note)
        return result

    def freeing(pointer):
        result = local_free(pointer)

        def note():
            actual = address(pointer)
            # NULL input is a no-op; only the actual nonzero conversion counts.
            if receipt["descriptor"] and actual == receipt["descriptor"]:
                assert result is None
                receipt["descriptor_frees"] += 1

        observe("descriptor_free", note)
        return result

    def opening(*args, **kwargs):
        fd = original_open(*args, **kwargs)

        def note():
            assert fd not in fds
            receipt["fd_opens"] += 1
            fds[fd] = receipt["fd_opens"]

        observe("fd_open", note)
        return fd

    def closing(fd):
        result = original_close(fd)

        def note():
            assert fd in fds
            assert case.fd_closed(fd)
            del fds[fd]
            receipt["fd_closes"] += 1

        observe("fd_close", note)
        return result

    def reopening(*args):
        handle = open_by_id(*args)
        if handle not in (None, C.c_void_p(-1).value):

            def note():
                assert handle not in handles
                handles[handle] = address(args[0])
                receipt["reopens"] += 1

            observe("reopen", note)
        return handle

    def closing_handle(handle):
        result = close_handle(handle)

        def note():
            value = address(handle)
            if value in handles:
                assert result != 0
                assert case.handle_closed(value)
                del handles[value]
                receipt["handle_closes"] += 1

        observe("handle_close", note)
        return result

    def setting(*args):
        result = set_security(*args)

        def note():
            value = address(args[0])
            receipt["set_calls"] += 1
            receipt["set_success"] = result == 0
            parent = handles[value]
            receipt["set_identity"] = (
                case.identity(value) == case.original_identity
                and case.identity(parent) == case.original_identity
                and len(fds) == 1
                and parent == native.handle(next(iter(fds)))
            )
            receipt["set_arguments"] = (
                args[1:5] == (1, 4 | 0x80000000, None, None)
                and bool(address(args[5]))
                and args[6] is None
            )

        observe("set", note)
        if (
            after_set_error is not None
            and receipt["set_success"]
            and receipt["set_identity"]
            and receipt["set_arguments"]
            and not receipt["errors"]
        ):
            receipt["controlled_error"] = True
            raise after_set_error
        return result

    class Advapi:
        def __getattr__(self, name):
            if name == "ConvertStringSecurityDescriptorToSecurityDescriptorW":
                return converting
            if name == "SetSecurityInfo":
                return setting
            return getattr(advapi, name)

    class Kernel:
        def __getattr__(self, name):
            if name == "OpenFileById":
                return reopening
            if name == "CloseHandle":
                return closing_handle
            if name == "LocalFree":
                return freeing
            return getattr(kernel, name)

    patch = monkeypatch.context()
    changed = patch.__enter__()
    primary = None
    try:
        changed.setattr(native, "advapi", Advapi())
        changed.setattr(native, "kernel", Kernel())
        changed.setattr(facade, "open", opening)
        changed.setattr(facade, "close", closing)
        yield receipt
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            patch.__exit__(None, None, None)
            # MonkeyPatch preserves a delegated attribute's value, not its absence.
            if not close_was_local and vars(facade).get("close") is original_close:
                delattr(facade, "close")
            receipt["restored"] = (
                native.advapi is advapi
                and native.kernel is kernel
                and facade.open is original_open
                and facade.close is original_close
                and ("close" in vars(facade)) is close_was_local
            )
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note("native_acl_observer_cleanup:" + type(cleanup).__name__)


def _assert_native_acl_retirement(receipt, *, acquired):
    assert not receipt["errors"] and not receipt["overflow"]
    assert receipt["restored"]
    assert receipt["conversion_calls"] == 1
    assert not receipt["live_fds"] and not receipt["live_handles"]
    assert receipt["descriptor_frees"] == int(bool(receipt["descriptor"]))
    assert receipt["fd_opens"] == receipt["fd_closes"]
    assert receipt["reopens"] == receipt["handle_closes"]
    if acquired:
        assert receipt["conversion_success"] and receipt["descriptor"]
        assert receipt["fd_opens"] > 0 and receipt["reopens"] > 0
        assert receipt["set_calls"] == 1 and receipt["set_success"]
        assert receipt["set_identity"] and receipt["set_arguments"]
    else:
        assert receipt["conversion_success"] is False
        assert receipt["conversion_error"] != 0
        assert receipt["fd_opens"] == receipt["reopens"] == receipt["set_calls"] == 0


def _assert_native_acl_public_read(case):
    info = case.facade.stat(case.path, follow_symlinks=False)
    current = case.security()
    checks = {
        "same_object": (info.st_dev, info.st_ino) == case.stat_identity,
        "same_owner": current[0] == case.baseline[0],
        "same_revision": current[2] == case.baseline[2],
        "protected": bool(current[1] & 0x1000),
        "different_acl": current[3] != case.baseline[3],
        "public_read": bool(info.st_mode & 0o044),
        "no_public_write": not info.st_mode & 0o022,
        "no_sticky_bit": not info.st_mode & 0o1000,
    }
    assert all(checks.values()), checks


def _native_acl_sddl(case):
    return f"D:P(A;;FA;;;{case.native.user_sid})(A;;FA;;;SY)(A;;FA;;;BA)(A;;FR;;;WD)"


def _record_native_acl_retirement(request, receipt):
    import json

    names = (
        "conversion_calls",
        "descriptor_frees",
        "fd_opens",
        "fd_closes",
        "reopens",
        "handle_closes",
        "set_calls",
        "set_success",
        "set_identity",
        "set_arguments",
        "controlled_error",
        "restored",
    )
    request.node.user_properties.append(
        ("native_acl_retirement", json.dumps({name: receipt[name] for name in names}))
    )


def test_native_mutator_applies_and_retires_exact_resources(
    _native_acl_directory, monkeypatch, request
):
    case = _native_acl_directory
    sddl = _native_acl_sddl(case)
    with _observe_native_acl_call(case, monkeypatch, sddl) as receipt:
        case.helper._replace_protected_windows_dacl(case.path, sddl)
    _assert_native_acl_retirement(receipt, acquired=True)
    _assert_native_acl_public_read(case)
    assert not receipt["controlled_error"]
    _record_native_acl_retirement(request, receipt)


def test_native_mutator_rejects_malformed_sddl_before_acquisition(
    _native_acl_directory, monkeypatch, request
):
    case = _native_acl_directory
    sddl = "not-a-security-descriptor"
    with pytest.raises(OSError) as caught:
        with _observe_native_acl_call(case, monkeypatch, sddl) as receipt:
            case.helper._replace_protected_windows_dacl(case.path, sddl)
    _assert_native_acl_retirement(receipt, acquired=False)
    assert caught.value.winerror == receipt["conversion_error"]
    unchanged = case.security() == case.baseline
    assert unchanged
    code = type(case.native).check.__code__
    traceback = caught.value.__traceback__
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next
    assert traceback is not None and traceback.tb_frame.f_code is code
    assert not receipt["controlled_error"]
    _record_native_acl_retirement(request, receipt)


def test_native_mutator_retires_after_controlled_post_dispatch_error(
    _native_acl_directory, monkeypatch, request
):
    case = _native_acl_directory
    sddl = _native_acl_sddl(case)
    expected = RuntimeError("controlled_native_acl_post_dispatch")
    with pytest.raises(RuntimeError) as caught:
        with _observe_native_acl_call(
            case, monkeypatch, sddl, after_set_error=expected
        ) as receipt:
            case.helper._replace_protected_windows_dacl(case.path, sddl)
    assert caught.value is expected
    _assert_native_acl_retirement(receipt, acquired=True)
    assert receipt["controlled_error"]
    _assert_native_acl_public_read(case)
    _record_native_acl_retirement(request, receipt)
