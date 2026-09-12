# Cross-platform backup correction

TASK-32496. The user requires the existing backup feature to work on macOS,
Linux and Windows, with actual Linux SSH and Windows GitHub Actions testing.
This supersedes earlier decisions to leave native Linux/Windows operations out.

Preserve the public recovery service, archive format, Python encryption, private
staging, exclusive publication, cooperative admission, journals, credential review
and rollback semantics. Platform support belongs at filesystem/locking boundaries,
not as duplicated backup services or a global monkeypatch of Python's os module.

macOS retains its native rename/full-sync operations. Linux uses descriptor-relative
renameat2(RENAME_NOREPLACE), fsync barriers and flock. Windows uses native handles,
reparse-point refusal, ACL privacy, exclusive rename and byte-range locking through
stdlib ctypes. Handle-relative operations must preserve containment and identity;
path-based check-then-replace fallbacks are unacceptable. Existing POSIX-specific
private storage callers needed by normal app startup must use the same boundary.

Replace exact OS-patch/Python-patch release assumptions with declared supported
platform/filesystem operation contracts and real integration evidence. Do not make
a test override the production capability gate to manufacture product success.
Unsupported filesystems or unavailable primitives must still fail safely.

Run real installed application create, inspect, isolated restore/open, replacement
and retained-copy rollback with synthetic records on all three platforms. Verify
restored contents and preservation of unrelated source records. Preserve failures,
exact source identities and installed-file hashes. No component-only completion.

ADR: amend existing ADR-126 for these platform contracts; no separate format or
product architecture is introduced. Work stays within backup/recovery and its
necessary storage/startup paths. No new language or crypto dependency.
