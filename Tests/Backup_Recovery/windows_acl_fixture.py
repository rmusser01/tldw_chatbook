"""Preserve a fixture directory's native DACL without an icacls text roundtrip."""

import ctypes as C
from contextlib import contextmanager


@contextmanager
def _security(native, handle):
    """Keep the returned descriptor alive while its original DACL is in use."""
    owner, dacl, descriptor = C.c_void_p(), C.c_void_p(), C.c_void_p()
    result = native.advapi.GetSecurityInfo(
        handle, 1, 5, C.byref(owner), None, C.byref(dacl), None, C.byref(descriptor)
    )
    if result:
        raise OSError(result, "GetSecurityInfo failed")
    try:
        control, revision = C.c_ushort(), C.c_uint32()
        read_control = native.advapi.GetSecurityDescriptorControl
        read_control.argtypes = [
            C.c_void_p,
            C.POINTER(C.c_ushort),
            C.POINTER(C.c_uint32),
        ]
        read_control.restype = C.c_int
        native.check(read_control(descriptor, C.byref(control), C.byref(revision)))
        acl = None
        if dacl.value:
            size = int.from_bytes(C.string_at(dacl, 8)[2:4], "little")
            if size < 8:
                raise ValueError("invalid native ACL size")
            acl = C.string_at(dacl, size)
        yield dacl, (native.sid_string(owner), control.value, revision.value, acl)
    finally:
        native.kernel.LocalFree(descriptor)


@contextmanager
def _preserve_dacl(native, handle):
    """Restore only the changed DACL, then require exact native state equality."""
    with _security(native, handle) as (dacl, original):
        try:
            yield
        finally:
            # Preserve the original protection flag; do not replace owner/SACL.
            protection = 0x80000000 if original[1] & 0x1000 else 0x20000000
            result = native.advapi.SetSecurityInfo(
                handle, 1, 4 | protection, None, None, dacl, None
            )
            if result:
                raise OSError(result, "SetSecurityInfo failed")
            with _security(native, handle) as (_, restored):
                if restored != original:
                    raise AssertionError(
                        "original native security not restored: "
                        f"owner_equal={original[0] == restored[0]} "
                        f"original_control={original[1]} restored_control={restored[1]} "
                        f"control_xor={original[1] ^ restored[1]} "
                        f"original_revision={original[2]} restored_revision={restored[2]} "
                        f"original_acl_length={len(original[3] or b'')} "
                        f"restored_acl_length={len(restored[3] or b'')} "
                        f"acl_equal={original[3] == restored[3]}"
                    )


@contextmanager
def preserve_windows_dacl(path):
    """Pin the exact private fixture object and retain its native cleanup rights."""
    from tldw_chatbook.Backup_Recovery.native_files import pinned_directory
    from tldw_chatbook.Utils.windows_files import _READ_CONTROL, _WRITE_DAC, _native

    native = _native()
    with (
        pinned_directory(path) as fd,
        native.reopen(native.handle(fd), _READ_CONTROL | _WRITE_DAC) as handle,
        _preserve_dacl(native, handle),
    ):
        yield
