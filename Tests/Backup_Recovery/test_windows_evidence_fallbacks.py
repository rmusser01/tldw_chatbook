"""Optional Windows evidence never widens or restricts the original full route."""

import os

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_windows_evidence_lifetime import _script


pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows evidence")

_FALLBACK = r"""
original_volume = native.kernel.GetVolumeInformationByHandleW
original_security = native.security
registry_path = root / 'admission' / 'registry.json'
original_selector = selector.read_bytes()
original_registry = registry_path.read_bytes()
observed = []
reused = []

def readonly_volume(*args):
    okay = original_volume(*args)
    if okay:
        flags = ctypes.cast(args[5], ctypes.POINTER(wf._U32))
        flags.contents.value |= 0x80000
        observed.append('readonly')
    return okay

def unavailable_descriptor(handle, is_directory, *, with_descriptor=False):
    actual = original_security(handle, is_directory, with_descriptor=with_descriptor)
    if with_descriptor:
        observed.append('descriptor-limit')
        raise PermissionError(errno.EACCES, 'windows_admission_descriptor_unavailable')
    return actual

def reuse_returned(_code, _offset, value):
    reused.append(value)

try:
    if route == 'readonly-volume':
        # Observe the real native API first, then inject its documented readonly
        # capability bit. Original native_identity must independently refuse it.
        native.kernel.GetVolumeInformationByHandleW = readonly_volume
    elif route == 'descriptor-limit':
        # Preserve original fresh security decoding. Only the optional exact-
        # descriptor capability is unavailable; ordinary stat still runs itself.
        native.security = unavailable_descriptor
    else:
        assert route == 'aggregate-content-budget'
        # Two valid originals individually remain below bootstrap.MAX_RECORD.
        # Their aggregate exceeds only the optional observation's total budget.
        padding = 600 * 1024
        selector.write_bytes(original_selector + b'\n#' + b'x' * padding + b'\n')
        registry_path.write_bytes(original_registry + b' ' * padding)
        assert selector.stat().st_size < bootstrap.MAX_RECORD
        assert registry_path.stat().st_size < bootstrap.MAX_RECORD
        assert selector.stat().st_size + registry_path.stat().st_size > 1024 * 1024
        observed.append('real-over-budget-content')
    with observe(storage._reuse_evidence, sys.monitoring.events.PY_RETURN, reuse_returned):
        with storage.acquire_storage(target) as lease:
            actual_root, namespaces = lease.execution_context(target)
            assert actual_root == root
            if route == 'readonly-volume':
                assert lease._key is None and namespaces is None
            else:
                assert lease._key == keeper._key and 'profile' in namespaces
            assert target.read_text(encoding='utf-8') == 'actual native payload'
    assert observed and reused == [None], (observed, reused)
    assert set(storage._live_leases) == before_leases | {keeper}
    assert set(storage._pending_acquisitions) == before_pending
finally:
    native.kernel.GetVolumeInformationByHandleW = original_volume
    native.security = original_security
    try:
        if route == 'aggregate-content-budget':
            selector.write_bytes(original_selector)
            registry_path.write_bytes(original_registry)
    finally:
        keeper.close()
assert set(storage._live_leases) == before_leases
assert set(storage._pending_acquisitions) == before_pending
with authority.maintenance(('profile',), 2):
    pass
with storage.acquire_storage(target) as resumed:
    assert resumed._key is not None and 'profile' in resumed.execution_context(target)[1]
assert not blocked_attempts()
print(json.dumps({'fallback': route, 'original_full_route': True,
                  'readonly_unqualified_only': route == 'readonly-volume',
                  'ordinary_retirement_and_resumption': True}))
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "condition", ["readonly-volume", "descriptor-limit", "aggregate-content-budget"]
)
def test_unavailable_evidence_preserves_original_native_or_unqualified_route(tmp_path, condition):
    _run(tmp_path, condition, "approved", script=_script(_FALLBACK))
