"""Handle-relative Windows/NTFS operations for private storage and recovery.

This is a *local* os facade, never a patch to Python's os module. Names beneath
pinned directories use NtCreateFile(RootDirectory); every component refuses
reparse points. Security metadata is a conservative projection of the actual
owner SID and DACL, not Windows' synthetic POSIX permission bits. UID 1000 means
private custody of the process token user (actual owner, or proven private
TokenOwner custody); UID 0 means other SYSTEM/Administrators/TrustedInstaller
ownership. Actual owner SID and DACL are always read, never inferred from UID.

Only local NTFS is supported. All native handles use FILE_WRITE_THROUGH. Directory barriers issue normal
NtFlushBuffersFileEx requests (data, metadata and device synchronization); files
additionally use FlushFileBuffers. Failures are propagated. See:
https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-createfilew
https://learn.microsoft.com/windows/win32/api/winternl/nf-winternl-ntcreatefile
https://learn.microsoft.com/windows/win32/api/winbase/ns-winbase-file_rename_info
"""

from __future__ import annotations

import contextlib
import ctypes as C
import errno
import functools
from pathlib import Path
import ntpath
import os as _os
import platform
import stat as _stat
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

_U32 = C.c_uint32
_I32 = C.c_int32
_U16 = C.c_uint16
_P = C.c_void_p
_HANDLE = _P
_SYSTEM_SIDS = frozenset(
    {
        "S-1-5-18",
        "S-1-5-32-544",
        "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464",
    }
)
_OWNER_RIGHTS_SID = "S-1-3-4"
_READ_CONTROL = 0x20000
_WRITE_DAC = 0x40000
_DELETE = 0x10000
_SYNCHRONIZE = 0x100000
_READ_ATTRIBUTES = 0x80
_WRITE_ATTRIBUTES = 0x100
_WRITE_THROUGH = 2
_REPARSE = 0x400
_DIRECTORY = 0x10
_SHARE_ALL = 7
_EPOCH_100NS = 116444736000000000
# STATUS_BUFFER_TOO_SMALL / STATUS_BUFFER_OVERFLOW as signed NTSTATUS values.
_STATUS_BUFFER_SHORT = frozenset({-1073741789, -2147483643})
# FILE_WRITE_DATA/APPEND/EA/ATTRIBUTES, DELETE, WRITE_DAC/OWNER, GENERIC_WRITE/ALL.
_WRITE_RIGHTS = 0x2 | 0x4 | 0x10 | 0x100 | 0x10000 | 0x40000 | 0x80000 | 0x50000000

_native_mutation_lock = threading.Lock()
_native_mutations = 0


def _bump_native_mutation() -> None:
    global _native_mutations
    with _native_mutation_lock:
        _native_mutations += 1


def native_mutation_generation() -> int:
    """Count of this process's write-capable by-id reopens (entry and exit).

    Changes made through a by-id handle notify no watched directory; evidence
    confirmed under one generation is never reused under another (TASK-34601).
    """
    with _native_mutation_lock:
        return _native_mutations


class _UnicodeString(C.Structure):
    _fields_ = [("Length", _U16), ("MaximumLength", _U16), ("Buffer", _P)]


class _ObjectAttributes(C.Structure):
    _fields_ = [
        ("Length", _U32),
        ("RootDirectory", _HANDLE),
        ("ObjectName", C.POINTER(_UnicodeString)),
        ("Attributes", _U32),
        ("SecurityDescriptor", _P),
        ("SecurityQualityOfService", _P),
    ]


class _IOStatus(C.Structure):
    _fields_ = [("Status", _P), ("Information", C.c_size_t)]


class _FileInfo(C.Structure):
    _fields_ = [
        ("attributes", _U32),
        ("created", _U32 * 2),
        ("accessed", _U32 * 2),
        ("written", _U32 * 2),
        ("volume", _U32),
        ("size_high", _U32),
        ("size_low", _U32),
        ("links", _U32),
        ("index_high", _U32),
        ("index_low", _U32),
    ]


class _BasicInfo(C.Structure):
    _fields_ = [
        ("created", C.c_int64),
        ("accessed", C.c_int64),
        ("written", C.c_int64),
        ("changed", C.c_int64),
        ("attributes", _U32),
    ]


class _RenameInfo(C.Structure):
    _fields_ = [
        ("replace", _U32),
        ("root", _HANDLE),
        ("length", _U32),
        ("name", _U16 * 1),
    ]


class _FileIdDescriptor(C.Structure):
    _fields_ = [("size", _U32), ("kind", _U32), ("identifier", C.c_uint64 * 2)]


class _Overlapped(C.Structure):
    _fields_ = [
        ("Internal", C.c_size_t),
        ("InternalHigh", C.c_size_t),
        ("Offset", _U32),
        ("OffsetHigh", _U32),
        ("hEvent", _HANDLE),
    ]


#: Separators, NT namespace/ADS punctuation, wildcards and every control code.
_COMPONENT_FORBIDDEN = frozenset('/\\:\x00<>|?*"').union(map(chr, range(32)))


def _component(name: str) -> str:
    """Require one unambiguous ordinary filename; reject NT namespaces and ADS."""
    # Kept explicit for Python 3.11/3.12, which lack ntpath.isreserved.
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or not _COMPONENT_FORBIDDEN.isdisjoint(name)
        or name[-1] in " ."
    ):
        raise ValueError("invalid_windows_component")
    base = name.split(".", 1)[0].upper()
    if base in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or (
        len(base) == 4 and base[:3] in {"COM", "LPT"} and base[3] in "123456789¹²³"
    ):
        raise ValueError("reserved_windows_component")
    return name


def _acl_mode(
    aces: list[tuple[int, int, int, str]],
    current_sid: str,
    *,
    is_directory: bool,
    owner_sid: str | None = None,
) -> int:
    """Conservatively project effective public grants; never assume deny order.

    Administrative principals are trusted like POSIX root. Public inherit-only
    grants set privacy exposure bits (044), requiring private leaves to harden
    before ordinary SQLite/pathlib children can inherit those grants. They do
    not set destructive ancestor bits (022), since they do not apply here. Unknown/callback/object ACEs fail privacy closed.
    A directory permitting additions alone is sticky-like: outsiders cannot
    remove existing children, so ancestor traversal may safely pin them.
    """
    mode = 0o700 if is_directory else 0o600
    public_add = False
    public_destructive = False
    for kind, flags, mask, sid in aces:
        # OWNER RIGHTS grants apply to this object's actual owner, not to all
        # users. CPython mkdir(0700) emits OW rather than an explicit user SID.
        # Resolve against the owner read from the same security descriptor; an
        # unavailable/untrusted owner never becomes a trusted principal here.
        if sid == _OWNER_RIGHTS_SID and owner_sid is not None:
            sid = owner_sid
        if flags & 8:  # INHERIT_ONLY_ACE
            if (
                is_directory
                and kind != 1
                and sid != current_sid
                and sid not in _SYSTEM_SIDS
            ):
                mode |= 0o044
            continue
        if kind == 1:  # A deny cannot expand access; ignoring it is conservative.
            continue
        if kind != 0:
            return mode | 0o077
        if sid == current_sid or sid in _SYSTEM_SIDS:
            continue
        if mask & (0x80000000 | 0x10000000 | 0x1 | 0x8):
            mode |= 0o044
        destructive = mask & (
            0x40000000
            | 0x10000000
            | _DELETE
            | _WRITE_DAC
            | 0x80000
            | 0x40
            | 0x10
            | 0x100
        )
        additions = mask & 0x6
        if destructive or additions:
            mode |= 0o022
        public_destructive |= bool(destructive)
        public_add |= bool(additions)
    if is_directory and public_add and not public_destructive:
        mode |= _stat.S_ISVTX
    return mode


def _owner_uid(
    owner_sid: str,
    user_sid: str,
    token_owner_sid: str,
    aces: list[tuple[int, int, int, str]],
    mode: int,
) -> int:
    """Project actual ownership, accepting only exact private TokenOwner custody."""
    if owner_sid == user_sid:
        return 1000
    full_access = 0x1F01FF
    if (
        owner_sid == token_owner_sid
        and owner_sid in _SYSTEM_SIDS
        and not mode & 0o077
        and all(kind == 0 for kind, _, _, _ in aces)
        and any(
            kind == 0
            and not flags & 8
            and sid == user_sid
            and mask & full_access == full_access
            for kind, flags, mask, sid in aces
        )
    ):
        return 1000
    return 0 if owner_sid in _SYSTEM_SIDS else -1


class _Native:
    """Typed Windows ABI declarations, loaded only on native Windows."""

    def __init__(self):
        if _os.name != "nt":
            raise OSError(errno.ENOSYS, "native_windows_required")
        import msvcrt

        self.crt = msvcrt
        self.kernel = C.WinDLL("kernel32", use_last_error=True)
        self.advapi = C.WinDLL("advapi32", use_last_error=True)
        self.nt = C.WinDLL("ntdll", use_last_error=True)
        signatures = [
            (
                self.kernel,
                "OpenFileById",
                [_HANDLE, C.POINTER(_FileIdDescriptor), _U32, _U32, _P, _U32],
                _HANDLE,
            ),
            (
                self.nt,
                "NtSetInformationFile",
                [_HANDLE, C.POINTER(_IOStatus), _P, _U32, _U32],
                _I32,
            ),
            (
                self.nt,
                "NtFlushBuffersFileEx",
                [_HANDLE, _U32, _P, _U32, C.POINTER(_IOStatus)],
                _I32,
            ),
            (self.kernel, "CloseHandle", [_HANDLE], _I32),
            (
                self.kernel,
                "DuplicateHandle",
                [_HANDLE, _HANDLE, _HANDLE, C.POINTER(_HANDLE), _U32, _I32, _U32],
                _I32,
            ),
            (self.kernel, "LocalFree", [_P], _P),
            (self.kernel, "GetCurrentProcess", [], _HANDLE),
            (
                self.kernel,
                "GetFileInformationByHandle",
                [_HANDLE, C.POINTER(_FileInfo)],
                _I32,
            ),
            (
                self.kernel,
                "GetFileInformationByHandleEx",
                [_HANDLE, _I32, _P, _U32],
                _I32,
            ),
            (
                self.kernel,
                "SetFileInformationByHandle",
                [_HANDLE, _I32, _P, _U32],
                _I32,
            ),
            (self.kernel, "FlushFileBuffers", [_HANDLE], _I32),
            (self.kernel, "ReOpenFile", [_HANDLE, _U32, _U32, _U32], _HANDLE),
            (
                self.kernel,
                "GetVolumeInformationByHandleW",
                [_HANDLE, _P, _U32, _P, _P, _P, _P, _U32],
                _I32,
            ),
            (self.kernel, "GetFileType", [_HANDLE], _U32),
            (
                self.kernel,
                "LockFileEx",
                [_HANDLE, _U32, _U32, _U32, _U32, C.POINTER(_Overlapped)],
                _I32,
            ),
            (
                self.kernel,
                "UnlockFileEx",
                [_HANDLE, _U32, _U32, _U32, C.POINTER(_Overlapped)],
                _I32,
            ),
            (
                self.advapi,
                "OpenProcessToken",
                [_HANDLE, _U32, C.POINTER(_HANDLE)],
                _I32,
            ),
            (
                self.advapi,
                "GetTokenInformation",
                [_HANDLE, _I32, _P, _U32, C.POINTER(_U32)],
                _I32,
            ),
            (self.advapi, "ConvertSidToStringSidW", [_P, C.POINTER(_P)], _I32),
            (self.advapi, "GetSecurityDescriptorLength", [_P], _U32),
            (
                self.advapi,
                "GetSecurityDescriptorOwner",
                [_P, C.POINTER(_P), C.POINTER(_I32)],
                _I32,
            ),
            (
                self.advapi,
                "ConvertStringSecurityDescriptorToSecurityDescriptorW",
                [C.c_wchar_p, _U32, C.POINTER(_P), _P],
                _I32,
            ),
            (
                self.advapi,
                "GetSecurityInfo",
                [
                    _HANDLE,
                    _I32,
                    _U32,
                    C.POINTER(_P),
                    _P,
                    C.POINTER(_P),
                    _P,
                    C.POINTER(_P),
                ],
                _U32,
            ),
            (
                self.advapi,
                "SetSecurityInfo",
                [_HANDLE, _I32, _U32, _P, _P, _P, _P],
                _U32,
            ),
            (
                self.advapi,
                "GetSecurityDescriptorDacl",
                [_P, C.POINTER(_I32), C.POINTER(_P), C.POINTER(_I32)],
                _I32,
            ),
            (self.advapi, "GetAce", [_P, _U32, C.POINTER(_P)], _I32),
            (
                self.nt,
                "NtCreateFile",
                [
                    C.POINTER(_HANDLE),
                    _U32,
                    C.POINTER(_ObjectAttributes),
                    C.POINTER(_IOStatus),
                    _P,
                    _U32,
                    _U32,
                    _U32,
                    _U32,
                    _P,
                    _U32,
                ],
                _I32,
            ),
            (
                self.nt,
                "NtQueryInformationFile",
                [_HANDLE, C.POINTER(_IOStatus), _P, _U32, _U32],
                _I32,
            ),
            (
                self.nt,
                "NtQueryVolumeInformationFile",
                [_HANDLE, C.POINTER(_IOStatus), _P, _U32, _U32],
                _I32,
            ),
            (self.nt, "RtlNtStatusToDosError", [_I32], _U32),
            (
                self.nt,
                "NtQuerySecurityObject",
                [_HANDLE, _U32, _P, _U32, C.POINTER(_U32)],
                _I32,
            ),
            (
                self.kernel,
                "ReadDirectoryChangesW",
                [_HANDLE, _P, _U32, _I32, _U32, _P, C.POINTER(_Overlapped), _P],
                _I32,
            ),
            (self.kernel, "CreateEventW", [_P, _I32, _I32, _P], _HANDLE),
            (
                self.kernel,
                "WaitForMultipleObjects",
                [_U32, C.POINTER(_HANDLE), _I32, _U32],
                _U32,
            ),
            (self.kernel, "CancelIoEx", [_HANDLE, C.POINTER(_Overlapped)], _I32),
            (
                self.kernel,
                "GetOverlappedResult",
                [_HANDLE, C.POINTER(_Overlapped), C.POINTER(_U32), _I32],
                _I32,
            ),
        ]
        for dll, name, arguments, result in signatures:
            function = getattr(dll, name)
            function.argtypes, function.restype = arguments, result
        self.user_sid = self._user_sid()

    def check(self, result):
        if not result:
            raise C.WinError(C.get_last_error())
        return result

    def ntcheck(self, result):
        if result < 0:
            raise C.WinError(self.nt.RtlNtStatusToDosError(result))

    def sid_string(self, sid):
        output = _P()
        self.check(self.advapi.ConvertSidToStringSidW(sid, C.byref(output)))
        try:
            return C.wstring_at(output)
        finally:
            self.kernel.LocalFree(output)

    def _user_sid(self):
        return self._token_sid(1)

    def _token_sid(self, information_class):
        """Read the current process token SID; never cache TokenOwner custody."""
        token = _HANDLE()
        self.check(
            self.advapi.OpenProcessToken(
                self.kernel.GetCurrentProcess(), 8, C.byref(token)
            )
        )
        try:
            size = _U32()
            self.advapi.GetTokenInformation(
                token, information_class, None, 0, C.byref(size)
            )
            buffer = C.create_string_buffer(size.value)
            self.check(
                self.advapi.GetTokenInformation(
                    token, information_class, buffer, size, C.byref(size)
                )
            )
            return self.sid_string(C.cast(buffer, C.POINTER(_P))[0])
        finally:
            self.kernel.CloseHandle(token)

    @contextlib.contextmanager
    def private_sd(self):
        descriptor = _P()
        sddl = f"O:{self.user_sid}D:P(A;OICI;FA;;;{self.user_sid})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
        self.check(
            self.advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, 1, C.byref(descriptor), None
            )
        )
        try:
            yield descriptor
        finally:
            self.kernel.LocalFree(descriptor)

    def handle(self, fd):
        return self.crt.get_osfhandle(fd)

    def info(self, handle):
        value = _FileInfo()
        self.check(self.kernel.GetFileInformationByHandle(handle, C.byref(value)))
        if value.attributes & _REPARSE:
            raise OSError(errno.ELOOP, "windows_reparse_point_refused")
        if self.kernel.GetFileType(handle) != 1:
            raise OSError(errno.ENOTSUP, "windows_non_disk_object_refused")
        return value

    def ntfs(self, handle):
        name = C.create_unicode_buffer(32)
        flags = _U32()
        self.check(
            self.kernel.GetVolumeInformationByHandleW(
                handle, None, 0, None, None, C.byref(flags), name, len(name)
            )
        )
        if name.value != "NTFS":
            raise OSError(errno.ENOTSUP, "local_ntfs_required")
        device, status = (_U32 * 2)(), _IOStatus()
        self.ntcheck(
            self.nt.NtQueryVolumeInformationFile(
                handle, C.byref(status), device, C.sizeof(device), 4
            )
        )
        # FILE_FS_DEVICE_INFORMATION: Characteristics has FILE_REMOTE_DEVICE.
        if device[1] & 0x10:
            raise OSError(errno.ENOTSUP, "remote_filesystem_refused")

    def security_descriptor(self, handle):
        """Fresh self-relative OWNER|DACL bytes of the object behind ``handle``.

        One NtQuerySecurityObject reads the object's own stored descriptor into
        a call-local buffer; no native allocation or pointer outlives the call.
        GetSecurityInfo, used before, issued the same query through advapi's
        provider and, for objects without SE_DACL_AUTO_INHERITED (every
        directory this app creates), also re-read the PARENT's descriptor to
        synthesize INHERITED_ACE bits and an unrequested group SID: 5-15x the
        cost, and neither affects the owner/DACL projection below.
        """
        size = 512
        for _ in range(4):
            buffer, needed = C.create_string_buffer(size), _U32()
            status = self.nt.NtQuerySecurityObject(
                handle, 5, buffer, size, C.byref(needed)
            )
            if status in _STATUS_BUFFER_SHORT and size < needed.value <= 0x10000:
                size = needed.value  # the descriptor grew; query again
                continue
            self.ntcheck(status)
            length = self.advapi.GetSecurityDescriptorLength(buffer)
            control = struct.unpack_from("<H", buffer, 2)[0]
            # Only contiguous SE_SELF_RELATIVE bytes are decodable copies; an
            # absolute descriptor would hold process pointers. Fail closed.
            if not control & 0x8000 or not 20 <= length <= size:
                raise OSError(errno.EIO, "windows_security_descriptor_invalid")
            return C.string_at(buffer, length)
        raise OSError(errno.EIO, "windows_security_descriptor_unstable")

    def security(self, handle, is_directory, *, with_descriptor=False):
        """Read fresh security; reuse only decoding of byte-identical descriptors.

        TokenOwner is queried afresh, never cached, for every observation whose
        projection can depend on it: an administrative owner other than the
        token user (``_owner_uid``'s custody branch). Other owners project
        identically for every TokenOwner, so no token read is spent on them.
        """
        data = self.security_descriptor(handle)
        # 128 entries of at most 4096 bytes bound retained descriptor data.
        if len(data) <= 4096:
            owner_sid, aces, mode = self._decoded_security(
                data, is_directory, self.user_sid
            )
        elif with_descriptor:
            raise OSError(errno.ENOTSUP, "windows_security_stamp_unavailable")
        else:
            owner_sid, aces, mode = self._decode_descriptor(
                data, is_directory, self.user_sid
            )
        token_owner_sid = (
            self._token_sid(4)
            if owner_sid != self.user_sid and owner_sid in _SYSTEM_SIDS
            else None
        )
        projected = (
            _owner_uid(owner_sid, self.user_sid, token_owner_sid, aces, mode),
            mode,
        )
        return (*projected, data) if with_descriptor else projected

    @functools.lru_cache(maxsize=128)  # noqa: B019 - _native already retains this process-lifetime singleton.
    def _decoded_security(self, data, is_directory, user_sid):
        """Decode immutable bytes, never a handle, TokenOwner or filesystem decision."""
        return self._decode_descriptor(data, is_directory, user_sid)

    def _decode_descriptor(self, data, is_directory, user_sid):
        """Owner SID, ordered ACEs and conservative mode of descriptor bytes."""
        descriptor = C.create_string_buffer(data)
        owner, dacl = _P(), _P()
        defaulted, present = _I32(), _I32()
        self.check(
            self.advapi.GetSecurityDescriptorOwner(
                descriptor, C.byref(owner), C.byref(defaulted)
            )
        )
        self.check(
            self.advapi.GetSecurityDescriptorDacl(
                descriptor, C.byref(present), C.byref(dacl), C.byref(defaulted)
            )
        )
        return self._decode_security(owner, dacl, is_directory, user_sid)

    def _decode_security(self, owner, dacl, is_directory, user_sid):
        """Conservatively project the native owner and ordered ACL entries."""
        sid = self.sid_string(owner)
        if not dacl.value:
            return sid, (), 0o777
        header = C.string_at(dacl, 8)
        count = struct.unpack_from("<H", header, 4)[0]
        aces = []
        for index in range(count):
            ace = _P()
            self.check(self.advapi.GetAce(dacl, index, C.byref(ace)))
            kind, flags, length = struct.unpack("<BBH", C.string_at(ace, 4))
            if length < 8:
                raise OSError(errno.EACCES, "malformed_windows_acl")
            mask = struct.unpack("<I", C.string_at(ace.value + 4, 4))[0]
            trustee = self.sid_string(ace.value + 8) if kind in {0, 1} else ""
            aces.append((kind, flags, mask, trustee))
        mode = _acl_mode(aces, user_sid, is_directory=is_directory, owner_sid=sid)
        return sid, tuple(aces), mode

    def open_handle(
        self,
        name,
        *,
        parent=None,
        flags=0,
        directory=False,
        metadata=False,
        extra_access=0,
    ):
        access = _READ_CONTROL | _READ_ATTRIBUTES | _SYNCHRONIZE | extra_access
        if directory:
            access |= 1 | 0x20  # FILE_LIST_DIRECTORY | FILE_TRAVERSE
        elif not metadata:
            access |= 1 if flags & 3 != _os.O_WRONLY else 0
            if flags & 3 in {_os.O_WRONLY, _os.O_RDWR}:
                access |= 2 | 4 | _WRITE_ATTRIBUTES
        creating = bool(flags & _os.O_CREAT)
        if creating:
            access |= _WRITE_DAC
        disposition = 1
        if creating:
            disposition = 2 if flags & _os.O_EXCL else (5 if flags & _os.O_TRUNC else 3)
        elif flags & _os.O_TRUNC:
            disposition = 4
        options = 0x20 | _WRITE_THROUGH | 0x200000  # synchronous, no reparse following
        if directory:
            options |= 1
        elif not metadata and (flags & 3 or flags & (_os.O_CREAT | _os.O_TRUNC)):
            # A read-only open may select a file or directory; callers inspect
            # the pinned handle. Mutating opens must still select regular files.
            options |= 0x40
        raw = name.encode("utf-16-le")
        if len(raw) > 65532:
            raise ValueError("windows_path_too_long")
        buffer = C.create_string_buffer(raw + b"\x00\x00")
        unicode_name = _UnicodeString(len(raw), len(raw) + 2, C.addressof(buffer))
        with self.private_sd() if creating else contextlib.nullcontext(None) as sd:
            attributes = _ObjectAttributes(
                C.sizeof(_ObjectAttributes),
                parent,
                C.pointer(unicode_name),
                0x40,
                sd,
                None,
            )
            handle, status = _HANDLE(), _IOStatus()
            self.ntcheck(
                self.nt.NtCreateFile(
                    C.byref(handle),
                    access,
                    C.byref(attributes),
                    C.byref(status),
                    None,
                    0x80,
                    _SHARE_ALL,
                    disposition,
                    options,
                    None,
                    0,
                )
            )
        try:
            self.info(handle)
            self.ntfs(handle)
            return handle.value
        except BaseException:
            record = _AdmissionMetadataHandle(self, handle.value, name)
            try:
                closed = self.kernel.CloseHandle(handle)
            except BaseException as error:
                record.close_error = error
            else:
                if closed:
                    raise
                record.close_error = C.get_last_error()
            raise _AdmissionMetadataCloseError((record,))

    @contextlib.contextmanager
    def reopen(self, handle, access):
        """Reopen the live NTFS object by ID, including native directory handles.

        The original handle prevents file-ID reuse. The returned identity is
        checked before use; no pathname is resolved for rights acquisition.
        ReOpenFile fails on directories opened with NtCreateFile on supported
        Windows runners, while OpenFileById explicitly supports directory hints.
        """
        before = self.info(handle)
        descriptor = _FileIdDescriptor(
            C.sizeof(_FileIdDescriptor),
            0,
            (C.c_uint64 * 2)((before.index_high << 32) | before.index_low, 0),
        )
        # Changes made through a by-id handle raise NO directory change
        # notification; count them so watched evidence never relies on one.
        writes = bool(access & _WRITE_RIGHTS)
        if writes:
            _bump_native_mutation()
        value = self.kernel.OpenFileById(
            handle,
            C.byref(descriptor),
            access | _READ_ATTRIBUTES | _SYNCHRONIZE,
            _SHARE_ALL,
            None,
            0x02000000 | 0x00200000 | 0x80000000,
        )
        if value in {None, C.c_void_p(-1).value}:
            raise C.WinError(C.get_last_error())
        try:
            after = self.info(value)
            if (before.volume, before.index_high, before.index_low) != (
                after.volume,
                after.index_high,
                after.index_low,
            ):
                raise OSError(errno.ESTALE, "windows_reopen_identity_changed")
            yield value
        finally:
            self.kernel.CloseHandle(value)
            if writes:
                _bump_native_mutation()


@functools.lru_cache(maxsize=1)
def _native():
    return _Native()


@contextlib.contextmanager
def _parent(path, dir_fd=None):
    """Yield a pinned parent handle and validated leaf; walk local drive paths."""
    native = _native()
    raw = _os.fspath(path)
    if dir_fd is not None:
        yield native.handle(dir_fd), _component(raw)
        return
    absolute = ntpath.abspath(raw)
    drive, tail = ntpath.splitdrive(absolute)
    if (
        len(drive) != 2
        or drive[1] != ":"
        or not drive[0].isalpha()
        or not tail.startswith("\\")
    ):
        raise ValueError("local_absolute_drive_path_required")
    parts = [_component(part) for part in tail.split("\\") if part]
    handle = native.open_handle("\\??\\" + drive + "\\", directory=True)
    try:
        for part in parts[:-1]:
            successor = native.open_handle(part, parent=handle, directory=True)
            native.kernel.CloseHandle(handle)
            handle = successor
        yield handle, parts[-1] if parts else None
    finally:
        native.kernel.CloseHandle(handle)


class _AdmissionMetadataHandle:
    """One physical snapshot handle incarnation, retained on uncertain close."""

    def __init__(self, native, handle, path):
        self.native = native
        self.handle = handle
        self.path = path
        self.identity = None
        self.close_error = None


class _AdmissionMetadataCloseError(OSError):
    def __init__(self, failed_handles):
        self.failed_handles = tuple(failed_handles)
        super().__init__(errno.EIO, "windows_admission_metadata_not_retired")


class WindowsOS:
    """Bounded os-shaped facade used only by explicitly importing modules."""

    O_DIRECTORY = 0x10000000
    O_NOFOLLOW = 0x20000000
    O_NONBLOCK = 0x40000000
    O_NOCTTY = 0x08000000

    def __init__(self):
        # Private storage captures operation identity once for admission guards.
        # Cache bound methods so repeated lookup preserves that identity.
        for name, member in type(self).__dict__.items():
            if callable(member) and not name.startswith("__"):
                setattr(self, name, member.__get__(self, type(self)))

    def __getattr__(self, name):
        return getattr(_os, name)

    @property
    def supports_dir_fd(self):
        return {
            self.open,
            self.stat,
            self.mkdir,
            self.readlink,
            self.rename,
            self.replace,
            self.unlink,
            self.rmdir,
            self.link,
        }

    @property
    def supports_follow_symlinks(self):
        return {self.stat}

    def geteuid(self):
        """Return the token-user category used by verified SID projections."""
        _native()
        return 1000

    getuid = geteuid

    def open(self, path, flags, mode=0o777, *, dir_fd=None):
        native = _native()
        with _parent(path, dir_fd) as (parent, leaf):
            if leaf is None:
                if not flags & self.O_DIRECTORY or flags & (_os.O_CREAT | _os.O_TRUNC):
                    raise ValueError("invalid_drive_root_open")
                process, duplicate = native.kernel.GetCurrentProcess(), _HANDLE()
                native.check(
                    native.kernel.DuplicateHandle(
                        process, parent, process, C.byref(duplicate), 0, False, 2
                    )
                )
                handle = duplicate.value
            else:
                handle = native.open_handle(
                    leaf,
                    parent=parent,
                    flags=flags,
                    directory=bool(flags & self.O_DIRECTORY),
                )
        try:
            return native.crt.open_osfhandle(
                handle, (flags & (3 | _os.O_APPEND)) | _os.O_BINARY
            )
        except BaseException:
            native.kernel.CloseHandle(handle)
            raise

    def mkdir(self, path, mode=0o777, *, dir_fd=None):
        fd = self.open(
            path, self.O_DIRECTORY | _os.O_CREAT | _os.O_EXCL, mode, dir_fd=dir_fd
        )
        _os.close(fd)

    def fstat(self, fd):
        return self._stat_handle(_native().handle(fd))

    def stat(self, path, *, dir_fd=None, follow_symlinks=True):
        return self._named_stat(path, dir_fd=dir_fd)

    def stat_for_admission(self, path, *, follow_symlinks=False):
        """Read metadata and exact owner/DACL bytes from one fresh named handle."""
        return self._named_stat(path, with_descriptor=True)

    def stat_many_for_admission(self, paths):
        """Read a fresh tree snapshot and positively retire every native handle.

        Named associations are reopened bottom-up. A failed close keeps its
        exact handle incarnation in the defining exception; it is never retried.
        """
        selected = tuple(Path(path) for path in paths)
        # Close the selection under parents once, as plain strings. A node key
        # is its case-folded text (WindowsPath equality); the first spelling
        # seen is kept, as the former Path set did. Every component is some
        # node's own name, so validating each node's name once covers them all.
        spelling, parent_of, name_of, depth_of = {}, {}, {}, {}
        selected_keys = []
        for path in selected:
            if (
                not path.is_absolute()
                or len(path.drive) != 2
                or path.drive[1] != ":"
                or not path.drive[0].isalpha()
            ):
                raise ValueError("local_absolute_drive_path_required")
            parts = path.parts
            texts = [parts[0]]
            for part in parts[1:]:
                texts.append(
                    texts[-1] + part if len(texts) == 1 else texts[-1] + "\\" + part
                )
            # The former set inserted each path's parents nearest-first, then
            # the path itself; keep that first-spelling order exactly.
            for depth in (*range(len(texts) - 2, -1, -1), len(texts) - 1):
                text = texts[depth]
                key = text.lower()
                if key not in spelling:
                    spelling[key] = text
                    parent_of[key] = texts[depth - 1].lower() if depth else None
                    name_of[key] = parts[depth] if depth else None
                    depth_of[key] = depth + 1
            selected_keys.append(texts[-1].lower())
        nodes = sorted(spelling, key=lambda key: (depth_of[key], spelling[key]))
        for key in nodes:
            if name_of[key] is not None:
                _component(name_of[key])
        parents = {parent_of[key] for key in nodes if parent_of[key] is not None}
        native = _native()
        handles, identities, observations = {}, {}, {}
        opened, failed = {}, []

        def named_handle(node):
            try:
                if parent_of[node] is None:
                    handle = native.open_handle(
                        "\\??\\" + spelling[node][:2] + "\\", directory=True
                    )
                else:
                    handle = native.open_handle(
                        name_of[node],
                        parent=handles[parent_of[node]],
                        metadata=True,
                        directory=node in parents,
                    )
            except _AdmissionMetadataCloseError as error:
                failed.extend(error.failed_handles)
                raise
            record = _AdmissionMetadataHandle(native, handle, Path(spelling[node]))
            opened[handle] = record
            info = native.info(handle)
            record.identity = (info.volume, (info.index_high << 32) | info.index_low)
            return handle

        def close_handle(handle):
            record = opened.pop(handle)
            try:
                closed = native.kernel.CloseHandle(handle)
            except BaseException as error:
                record.close_error = error
                failed.append(record)
            else:
                if not closed:
                    record.close_error = C.get_last_error()
                    failed.append(record)

        try:
            for node in nodes:
                if parent_of[node] is not None and handles[parent_of[node]] is None:
                    handles[node] = None
                    continue
                try:
                    handle = named_handle(node)
                except FileNotFoundError:
                    handles[node] = None
                    continue
                handles[node] = handle
                identities[node] = opened[handle].identity
                if node not in parents:
                    close_handle(handle)
                    handles[node] = None
                    if failed:
                        raise _AdmissionMetadataCloseError(failed)
            for node in reversed(nodes):
                if parent_of[node] is not None and handles[parent_of[node]] is None:
                    observations[node] = None
                    continue
                try:
                    current = named_handle(node)
                except FileNotFoundError:
                    current = None
                if current is None:
                    observed = None
                else:
                    try:
                        observed = self._stat_handle(current, with_descriptor=True)
                    finally:
                        close_handle(current)
                    if failed:
                        raise _AdmissionMetadataCloseError(failed)
                identity = (
                    None
                    if observed is None
                    else (
                        observed[0].st_dev,
                        observed[0].st_ino,
                    )
                )
                if identity != identities.get(node):
                    raise OSError(errno.ESTALE, "windows_admission_snapshot_changed")
                observations[node] = observed
            return {
                path: observations[key] for path, key in zip(selected, selected_keys)
            }
        finally:
            for handle in reversed(tuple(opened)):
                close_handle(handle)
            if failed:
                raise _AdmissionMetadataCloseError(failed)

    def _named_stat(self, path, *, dir_fd=None, with_descriptor=False):
        if isinstance(path, int):
            return self._stat_handle(
                _native().handle(path), with_descriptor=with_descriptor
            )
        native = _native()
        with _parent(path, dir_fd) as (parent, leaf):
            if leaf is None:
                with native.reopen(
                    parent, _READ_CONTROL | _READ_ATTRIBUTES | _SYNCHRONIZE
                ) as handle:
                    return self._stat_handle(handle, with_descriptor=with_descriptor)
            handle = native.open_handle(leaf, parent=parent, metadata=True)
        try:
            return self._stat_handle(handle, with_descriptor=with_descriptor)
        finally:
            native.kernel.CloseHandle(handle)

    def _stat_handle(self, handle, *, with_descriptor=False):
        native = _native()
        info = native.info(handle)
        security = native.security(
            handle,
            bool(info.attributes & _DIRECTORY),
            **({"with_descriptor": True} if with_descriptor else {}),
        )
        uid, mode = security[:2]
        basic = _BasicInfo()
        native.check(
            native.kernel.GetFileInformationByHandleEx(
                handle, 0, C.byref(basic), C.sizeof(basic)
            )
        )

        def ns(value):
            return (value - _EPOCH_100NS) * 100

        result = _os.stat_result(
            (
                (_stat.S_IFDIR if info.attributes & _DIRECTORY else _stat.S_IFREG)
                | mode,
                (info.index_high << 32) | info.index_low,
                info.volume,
                info.links,
                uid,
                uid,
                (info.size_high << 32) | info.size_low,
                ns(basic.accessed) // 1_000_000_000,
                ns(basic.written) // 1_000_000_000,
                ns(basic.changed) // 1_000_000_000,
            ),
            {
                "st_atime": ns(basic.accessed) / 1e9,
                "st_mtime": ns(basic.written) / 1e9,
                "st_ctime": ns(basic.changed) / 1e9,
                "st_atime_ns": ns(basic.accessed),
                "st_mtime_ns": ns(basic.written),
                "st_ctime_ns": ns(basic.changed),
                "st_file_attributes": info.attributes,
            },
        )
        return (result, security[2]) if with_descriptor else result

    def fchmod(self, fd, mode):
        if mode not in {0o600, 0o700}:
            raise ValueError("windows_private_modes_only")
        native, handle = _native(), _native().handle(fd)
        if (
            native.security(handle, bool(native.info(handle).attributes & _DIRECTORY))[
                0
            ]
            != 1000
        ):
            raise PermissionError(errno.EPERM, "windows_owner_mismatch")
        with (
            native.private_sd() as sd,
            native.reopen(handle, _READ_CONTROL | _WRITE_DAC) as writable,
        ):
            present, defaulted, dacl = _I32(), _I32(), _P()
            native.check(
                native.advapi.GetSecurityDescriptorDacl(
                    sd, C.byref(present), C.byref(dacl), C.byref(defaulted)
                )
            )
            result = native.advapi.SetSecurityInfo(
                writable, 1, 4 | 0x80000000, None, None, dacl, None
            )
            if result:
                raise C.WinError(result)

    def chmod(self, path, mode, *, dir_fd=None, follow_symlinks=True):
        native = _native()
        with _parent(path, dir_fd) as (parent, leaf):
            if leaf is None:
                raise PermissionError("cannot_harden_drive_root")
            handle = native.open_handle(leaf, parent=parent, metadata=True)
        try:
            fd = native.crt.open_osfhandle(handle, _os.O_RDONLY | _os.O_BINARY)
        except BaseException:
            native.kernel.CloseHandle(handle)
            raise
        try:
            self.fchmod(fd, mode)
        finally:
            _os.close(fd)

    def fsync(self, fd):
        if _native().info(_native().handle(fd)).attributes & _DIRECTORY:
            flush_directory(fd)
        else:
            flush_file(fd)

    def rename(self, src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        self._rename(src, dst, src_dir_fd, dst_dir_fd, False)

    def replace(self, src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        self._rename(src, dst, src_dir_fd, dst_dir_fd, True)

    def _rename(self, src, dst, src_fd, dst_fd, replace, *, link=False):
        native = _native()
        with (
            _parent(src, src_fd) as (source, name),
            _parent(dst, dst_fd) as (target, new_name),
        ):
            if name is None or new_name is None:
                raise ValueError("cannot_rename_drive_root")
            handle = native.open_handle(
                name, parent=source, metadata=True, extra_access=_DELETE
            )
            try:
                encoded = new_name.encode("utf-16-le")
                size = C.sizeof(_RenameInfo) + len(encoded)
                buffer = C.create_string_buffer(size)
                info = _RenameInfo.from_buffer(buffer)
                info.replace, info.root, info.length = (
                    int(replace),
                    target,
                    len(encoded),
                )
                C.memmove(
                    C.addressof(buffer) + _RenameInfo.name.offset, encoded, len(encoded)
                )
                status = _IOStatus()
                native.ntcheck(
                    native.nt.NtSetInformationFile(
                        handle,
                        C.byref(status),
                        buffer,
                        size,
                        11 if link else 10,
                    )
                )

            finally:
                native.kernel.CloseHandle(handle)

    def link(self, src, dst, *, src_dir_fd=None, dst_dir_fd=None, follow_symlinks=True):
        self._rename(src, dst, src_dir_fd, dst_dir_fd, False, link=True)

    def unlink(self, path, *, dir_fd=None):
        self._delete(path, dir_fd, False)

    def rmdir(self, path, *, dir_fd=None):
        self._delete(path, dir_fd, True)

    def _delete(self, path, dir_fd, directory):
        native = _native()
        with _parent(path, dir_fd) as (parent, leaf):
            if leaf is None:
                raise ValueError("cannot_delete_drive_root")
            handle = native.open_handle(
                leaf, parent=parent, metadata=True, extra_access=_DELETE
            )
        try:
            actual = bool(native.info(handle).attributes & _DIRECTORY)
            if actual != directory:
                raise OSError(
                    errno.EISDIR if actual else errno.ENOTDIR,
                    "unexpected_windows_object_type",
                )
            # POSIX semantics unlink now while existing share-delete handles live.
            flags = _U32(1 | 2)  # FILE_DISPOSITION_DELETE | POSIX_SEMANTICS
            native.check(
                native.kernel.SetFileInformationByHandle(
                    handle, 21, C.byref(flags), C.sizeof(flags)
                )
            )
        finally:
            native.kernel.CloseHandle(handle)

    def listdir(self, path="."):
        if not isinstance(path, int):
            fd = self.open(path, self.O_DIRECTORY | _os.O_RDONLY)
            try:
                return self.listdir(fd)
            finally:
                _os.close(fd)
        native, output = _native(), []
        buffer = C.create_string_buffer(65536)
        first = True
        # FILE_FULL_DIR_INFO has FileNameLength at offset 60, filename at 68.
        while True:
            okay = native.kernel.GetFileInformationByHandleEx(
                native.handle(path), 15 if first else 14, buffer, len(buffer)
            )
            first = False
            if not okay:
                if C.get_last_error() == 18:
                    return output
                raise C.WinError(C.get_last_error())
            offset = 0
            while True:
                next_offset = struct.unpack_from("<I", buffer, offset)[0]
                length = struct.unpack_from("<I", buffer, offset + 60)[0]
                if length % 2 or offset + 68 + length > len(buffer):
                    raise OSError(errno.EIO, "invalid_windows_directory_information")
                name = buffer[offset + 68 : offset + 68 + length].decode("utf-16-le")
                if name not in {".", ".."}:
                    output.append(name)
                if not next_offset:
                    break
                if next_offset < 68 or offset + next_offset >= len(buffer):
                    raise OSError(errno.EIO, "invalid_windows_directory_offset")
                offset += next_offset

    def scandir(self, path="."):
        # Recovery callers need names only. Intentionally no misleading path or
        # DirEntry.stat fallback that might follow names outside the pinned root.
        return contextlib.closing(_DirectoryNames(self.listdir(path)))

    def readlink(self, path, *, dir_fd=None):
        raise OSError(errno.ELOOP, "windows_reparse_points_are_never_traversed")

    def utime(self, path, times=None, *, ns=None, dir_fd=None, follow_symlinks=True):
        if not isinstance(path, int):
            native = _native()
            with _parent(path, dir_fd) as (parent, leaf):
                if leaf is None:
                    raise ValueError("cannot_change_drive_root_times")
                handle = native.open_handle(
                    leaf, parent=parent, metadata=True, extra_access=_WRITE_ATTRIBUTES
                )
            descriptor = native.crt.open_osfhandle(handle, _os.O_RDONLY | _os.O_BINARY)
            try:
                return self.utime(descriptor, times, ns=ns)
            finally:
                _os.close(descriptor)
        if ns is None:
            if times is None:
                import time

                ns = (time.time_ns(),) * 2
            else:
                ns = tuple(int(value * 1e9) for value in times)
        native = _native()
        with native.reopen(
            native.handle(path), _WRITE_ATTRIBUTES | _READ_ATTRIBUTES
        ) as handle:
            info = _BasicInfo(
                0, ns[0] // 100 + _EPOCH_100NS, ns[1] // 100 + _EPOCH_100NS, 0, 0
            )
            native.check(
                native.kernel.SetFileInformationByHandle(
                    handle, 0, C.byref(info), C.sizeof(info)
                )
            )

    def listxattr(self, fd):
        """Expose alternate data streams as nonordinary metadata, never omit them."""
        native = _native()
        buffer = C.create_string_buffer(65536)
        okay = native.kernel.GetFileInformationByHandleEx(
            native.handle(fd), 7, buffer, len(buffer)
        )
        # FileStreamInfo reports ERROR_HANDLE_EOF when no streams exist.
        # NTFS directories have no unnamed $DATA stream, unlike regular files.
        if not okay and C.get_last_error() == 38:
            return []
        native.check(okay)
        output, offset = [], 0
        while True:
            next_offset, length = struct.unpack_from("<II", buffer, offset)
            if length % 2 or offset + 24 + length > len(buffer):
                raise OSError(errno.EIO, "invalid_windows_stream_information")
            name = buffer[offset + 24 : offset + 24 + length].decode("utf-16-le")
            if name and name != "::$DATA":
                output.append(name)
            if not next_offset:
                return output
            if next_offset < 24 or offset + next_offset >= len(buffer):
                raise OSError(errno.EIO, "invalid_windows_stream_offset")
            offset += next_offset


class _DirectoryNames:
    def __init__(self, names):
        self._names = iter(names)

    def __iter__(self):
        return self

    def __next__(self):
        return SimpleNamespace(name=next(self._names))

    def close(self):
        pass  # Enumeration copied names while its caller-owned handle was pinned.


class WindowsLocks:
    """Cross-process cooperative shared/exclusive byte locks via LockFileEx."""

    LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8

    def flock(self, fd, operation):
        native, overlap = _native(), _Overlapped()
        if operation == self.LOCK_UN:
            native.check(
                native.kernel.UnlockFileEx(native.handle(fd), 0, 1, 0, C.byref(overlap))
            )
            return
        if operation & ~(
            self.LOCK_SH | self.LOCK_EX | self.LOCK_NB
        ) or operation & 3 not in {1, 2}:
            raise ValueError("invalid_windows_lock_operation")
        flags = (1 if operation & self.LOCK_NB else 0) | (
            2 if operation & self.LOCK_EX else 0
        )
        if not native.kernel.LockFileEx(
            native.handle(fd), flags, 0, 1, 0, C.byref(overlap)
        ):
            error = C.get_last_error()
            if error in {32, 33, 158}:
                raise BlockingIOError(errno.EAGAIN, "windows_lock_busy")
            raise C.WinError(error)


def native_identity(fd: int) -> dict[str, str | int]:
    """Read the actual local-NTFS runtime/volume identity for capability checks."""
    native, handle = _native(), _native().handle(fd)
    native.info(handle)
    native.ntfs(handle)
    filesystem, flags = C.create_unicode_buffer(32), _U32()
    native.check(
        native.kernel.GetVolumeInformationByHandleW(
            handle, None, 0, None, None, C.byref(flags), filesystem, len(filesystem)
        )
    )
    if flags.value & 0x80000:  # FILE_READ_ONLY_VOLUME
        raise OSError(errno.EROFS, "windows_readonly_volume")
    return {
        "os": platform.system(),
        "release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "filesystem": filesystem.value,
        "flags": flags.value,
    }


def rename_noreplace(src_fd: int, src: str, dst_fd: int, dst: str) -> None:
    """Publish through native relative rename with ReplaceIfExists false."""
    WindowsOS().rename(src, dst, src_dir_fd=src_fd, dst_dir_fd=dst_fd)


def flush_directory(fd: int) -> None:
    """Flush directory data/metadata and the device cache with native flags zero.

    NtFlushBuffersFileEx(normal) documents metadata and storage synchronization.
    A same-object directory handle obtains FILE_ADD_FILE/ADD_SUBDIRECTORY (the
    directory meanings of write/append) because read-only handles cannot flush.
    Neither unsupported native calls nor permission errors become success.
    """
    native, handle = _native(), _native().handle(fd)
    if not native.info(handle).attributes & _DIRECTORY:
        raise OSError(errno.ENOTDIR, "directory_barrier_requires_directory")
    native.ntfs(handle)
    with native.reopen(handle, 2 | 4) as writable:
        status = _IOStatus()
        native.ntcheck(
            native.nt.NtFlushBuffersFileEx(writable, 0, None, 0, C.byref(status))
        )
        native.ntcheck(C.c_int32(status.Status or 0).value)


def flush_file(fd: int) -> None:
    """Flush the pinned regular file's buffered data; propagate all failures."""
    native, handle = _native(), _native().handle(fd)
    native.info(handle)
    # Windows FlushFileBuffers requires GENERIC_WRITE. Reopen the same object,
    # never a reconstructed path, when a verifier holds a read-only descriptor.
    with native.reopen(handle, 0x40000000) as writable:
        native.check(native.kernel.FlushFileBuffers(writable))


#: ReadDirectoryChangesW filters (FILE_NOTIFY_CHANGE_*).
WATCH_POSTURE = 0x1 | 0x2 | 0x4 | 0x100  # FILE_NAME, DIR_NAME, ATTRIBUTES, SECURITY
WATCH_CONTENT = WATCH_POSTURE | 0x8 | 0x10 | 0x40  # + SIZE, LAST_WRITE, CREATION
_WAIT_TIMEOUT = 0x102
_WAIT_CHUNK = 64  # MAXIMUM_WAIT_OBJECTS
_WATCH_BUFFER = 1024
_watch_issuer_lock = threading.Lock()
_watch_issuer_executor = None


def _watch_issuer() -> ThreadPoolExecutor:
    """The one process-lifetime thread that issues every notification read.

    Windows cancels a thread's pending overlapped I/O when that thread exits,
    and completes event-based reads through an APC to the issuing thread. Arming
    on the many short-lived Send/worker threads would cancel watches spuriously
    and delay completions behind busy threads; one mostly idle owner avoids both.
    """
    global _watch_issuer_executor
    with _watch_issuer_lock:
        if _watch_issuer_executor is None:
            _watch_issuer_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="chatbook-directory-watch"
            )
        return _watch_issuer_executor


class DirectoryWatch:
    """Pending change notifications on exact, reparse-free directory objects.

    Each directory is opened by the same component-wise walk as
    ``stat_many_for_admission`` (every component refuses reparse points), then
    opened again by NAME under its pinned parent for overlapped I/O (NTFS
    refuses change notification on a handle opened by file id) and verified to
    be the very object the walk proved. One ``ReadDirectoryChangesW`` per
    directory stays pending; ANY completion (a change, a buffer overflow, an
    error, cancellation) signals its event and the watch is no longer
    ``quiet()``. The watch never decides admission: it only tells callers
    whether a full observation may be skipped. Changes made through a handle
    opened by file id notify nothing; callers must account for them (see
    ``native_mutation_generation``).

    While open, these handles prevent renaming any ancestor of a watched
    directory (Windows refuses to rename a directory above an open handle). The
    owner accepted this for storage-admission evidence (TASK-34601 ADR-126
    amendment); close the watch to release it.
    """

    def __init__(self, directories: dict) -> None:
        """Arm notifications for ``{Path: filter}``; raise OSError on any failure."""
        native = _native()
        self._native = native
        self._slots = []  # (handle, event, overlapped, buffer) per directory
        self._closed = False
        # Every pending read must belong to the persistent issuer thread.
        _watch_issuer().submit(self._open_and_arm, directories).result()

    def _open_and_arm(self, directories: dict) -> None:
        native = self._native
        nodes = sorted(
            {Path(d) for d in directories}
            | {p for d in directories for p in Path(d).parents},
            key=lambda node: (len(node.parts), str(node)),
        )
        for node in nodes:
            if (
                not node.is_absolute()
                or len(node.drive) != 2
                or node.drive[1] != ":"
                or not node.drive[0].isalpha()
            ):
                raise ValueError("local_absolute_drive_path_required")
            if node.parent != node:
                _component(node.name)
        handles = {}
        try:
            for node in nodes:
                if node.parent == node:
                    handles[node] = native.open_handle(
                        "\\??\\" + node.drive + "\\", directory=True
                    )
                else:
                    handles[node] = native.open_handle(
                        node.name, parent=handles[node.parent], directory=True
                    )
            for directory, notify in directories.items():
                node = Path(directory)
                if node.parent == node:
                    name, parent = "\\??\\" + node.drive + "\\", None
                else:
                    name, parent = node.name, handles[node.parent]
                self._arm(handles[node], name, parent, notify)
        except BaseException:
            self.close()
            raise
        finally:
            for handle in handles.values():
                native.kernel.CloseHandle(handle)

    def _arm(self, pinned, name, parent, notify):
        """Open ``name`` under the pinned ``parent`` for overlapped I/O and arm it.

        Change notification needs a handle opened by NAME (NTFS refuses one
        opened by file id), so the same single-component, reparse-refusing
        relative open as the walk is repeated without synchronous I/O, and the
        result must be the very object the walk pinned.
        """
        native = self._native
        before = native.info(pinned)
        raw = name.encode("utf-16-le")
        text = C.create_string_buffer(raw + b"\x00\x00")
        unicode_name = _UnicodeString(len(raw), len(raw) + 2, C.addressof(text))
        attributes = _ObjectAttributes(
            C.sizeof(_ObjectAttributes),
            parent,
            C.pointer(unicode_name),
            0x40,
            None,
            None,
        )
        opened, status = _HANDLE(), _IOStatus()
        native.ntcheck(
            native.nt.NtCreateFile(
                C.byref(opened),
                1
                | _READ_ATTRIBUTES
                | _SYNCHRONIZE,  # FILE_LIST_DIRECTORY; async (no 0x20)
                C.byref(attributes),
                C.byref(status),
                None,
                0x80,
                _SHARE_ALL,
                1,  # FILE_OPEN
                1 | 0x200000,  # FILE_DIRECTORY_FILE | FILE_OPEN_REPARSE_POINT
                None,
                0,
            )
        )
        handle = opened.value
        event = None
        try:
            after = native.info(handle)
            if (before.volume, before.index_high, before.index_low) != (
                after.volume,
                after.index_high,
                after.index_low,
            ):
                raise OSError(errno.ESTALE, "windows_watch_identity_changed")
            event = native.kernel.CreateEventW(None, 1, 0, None)
            if not event:
                raise C.WinError(C.get_last_error())
            overlapped = _Overlapped(0, 0, 0, 0, event)
            buffer = C.create_string_buffer(_WATCH_BUFFER)
            if not native.kernel.ReadDirectoryChangesW(
                handle,
                buffer,
                _WATCH_BUFFER,
                0,
                notify,
                None,
                C.byref(overlapped),
                None,
            ):
                raise C.WinError(C.get_last_error())
        except BaseException:
            if event:
                native.kernel.CloseHandle(event)
            native.kernel.CloseHandle(handle)
            raise
        self._slots.append((handle, event, overlapped, buffer))

    def quiet(self) -> bool:
        """True only when no watched directory reported anything since arming."""
        if self._closed or not self._slots:
            return False
        kernel = self._native.kernel
        for start in range(0, len(self._slots), _WAIT_CHUNK):
            chunk = self._slots[start : start + _WAIT_CHUNK]
            events = (_HANDLE * len(chunk))(*(slot[1] for slot in chunk))
            if kernel.WaitForMultipleObjects(len(chunk), events, 0, 0) != _WAIT_TIMEOUT:
                return False  # signaled, abandoned or WAIT_FAILED: not quiet
        return True

    def close(self) -> None:
        """Cancel and retire every pending notification; idempotent."""
        if self._closed:
            return
        self._closed = True
        kernel = self._native.kernel
        slots, self._slots = self._slots, []
        for handle, event, overlapped, _buffer in slots:
            # The kernel may write the OVERLAPPED/buffer until the cancelled
            # read completes: wait for it before releasing either.
            kernel.CancelIoEx(handle, C.byref(overlapped))
            transferred = _U32()
            kernel.GetOverlappedResult(
                handle, C.byref(overlapped), C.byref(transferred), 1
            )
            kernel.CloseHandle(handle)
            kernel.CloseHandle(event)


# Original callable provenance belongs to the defining module, before consumers.
_WINDOWS_METADATA_CLASS_ORIGINAL = WindowsOS
_WINDOWS_METADATA_METHODS_ORIGINAL = tuple(
    (name, vars(WindowsOS)[name])
    for name in ("stat", "_named_stat", "_stat_handle", "stat_many_for_admission")
)
_WINDOWS_METADATA_CLOSE_ERROR_ORIGINAL = _AdmissionMetadataCloseError

# The binding batch admits only this definition-time body on an exact stock
# receiver. Custom metadata readers keep their existing scalar path.
_WINDOWS_BINDING_TREE_ORIGINAL = (
    _WINDOWS_METADATA_CLASS_ORIGINAL,
    WindowsOS.stat_many_for_admission,
    WindowsOS.stat_many_for_admission.__code__,
    WindowsOS.stat_many_for_admission.__globals__,
    WindowsOS.stat_many_for_admission.__defaults__,
    WindowsOS.stat_many_for_admission.__kwdefaults__,
    WindowsOS.stat_many_for_admission.__closure__,
    "stat_many_for_admission",
)
