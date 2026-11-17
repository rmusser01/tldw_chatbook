"""Fresh Windows metadata alone does not prove unchanged admission inputs."""

import os
from textwrap import indent

import pytest

from Tests.Backup_Recovery.test_activation_agents import _SCRIPT as _AGENT_SETUP
from Tests.Backup_Recovery.test_home_citation_retirement import _run


pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows evidence")

_SETUP = _AGENT_SETUP.split("effects=[]")[0]
_ADVERSARIAL = _SETUP + r"""
import ctypes
import errno
import json
import re
import stat
from tldw_chatbook.Utils import windows_files as wf

target = data / 'evidence-content-source.txt'
with storage.acquire_storage(target):
    target.write_text('native prerequisite payload', encoding='utf-8')
before_leases = set(storage._live_leases)
before_pending = set(storage._pending_acquisitions)
keeper = storage.acquire_storage(target)
assert keeper._key is not None
hold = storage._holds[keeper._key]
assert 'profile' in keeper.execution_context(target)[1]
registry = bootstrap._registry(root)
entry = storage._selector_evidence(root, selector, hold.names,
                                   tuple(registry['profile']['roots']))
assert entry is not None
paths = tuple(dict.fromkeys((*entry.dependencies()[0], *entry.dependencies()[1])))
assert root in paths and root / 'admission' in paths
assert root / 'admission' / 'registry.json' in paths

def observe():
    result = storage.os.stat_many_for_admission(paths)
    return tuple(
        None if result[path] is None else (
            result[path][0].st_dev, result[path][0].st_ino,
            stat.S_IFMT(result[path][0].st_mode),
            stat.S_IMODE(result[path][0].st_mode), result[path][0].st_uid,
            result[path][0].st_nlink, result[path][0].st_size,
            result[path][0].st_mtime_ns, result[path][0].st_ctime_ns,
            result[path][0].st_file_attributes, result[path][1],
        ) for path in paths
    )

native = wf._native()
retired_descriptors = []

def close_fixture_descriptor(descriptor, handle):
    os.close(descriptor)
    # Check this birth immediately, before subsequent fixture operations can
    # allocate another native handle with the same numeric value.
    try:
        os.fstat(descriptor)
    except OSError as error:
        assert error.errno == errno.EBADF
    else:
        raise AssertionError('fixture CRT descriptor remained open')
    try:
        native.info(handle)
    except OSError as error:
        assert error.winerror == 6
    else:
        raise AssertionError('fixture native handle remained open')
    retired_descriptors.append(descriptor)

writer = None
writer_handle = None
parent_fd = None
parent_handle = None
marker = None
original = None
refusal = None
metadata_equal = False
epoch = bootstrap._admission_epoch
try:
    if route == 'same-size-content':
        document = root / 'admission' / 'registry.json'
        original = document.read_bytes()
        changed, substitutions = re.subn(
            rb'("version"\s*:\s*)1(?=\s*[,}])', rb'\g<1>9', original, count=1
        )
        assert substitutions == 1 and changed != original and len(changed) == len(original)
        writer = storage.os.open(document, os.O_RDWR | storage.os.O_NOFOLLOW)
        writer_handle = native.handle(writer)
        # FILE_BASIC_INFORMATION explicitly permits suppressing updates on this
        # real writer handle. No product clock, stat, epoch or policy is replaced.
        suppress = wf._BasicInfo(0, -1, -1, -1, 0)
        status = wf._IOStatus()
        native.ntcheck(native.nt.NtSetInformationFile(
            writer_handle, ctypes.byref(status), ctypes.byref(suppress),
            ctypes.sizeof(suppress), 4,
        ))
        before = observe()
        assert os.write(writer, changed) == len(changed)
        assert document.read_bytes() == changed
        expected_reason = 'recovery_scope_uncertain'
    else:
        assert route == 'new-pending-membership'
        parent_fd = storage.os.open(root, os.O_RDONLY | storage.os.O_DIRECTORY)
        parent_handle = native.handle(parent_fd)
        basic = wf._BasicInfo()
        native.check(native.kernel.GetFileInformationByHandleEx(
            parent_handle, 0, ctypes.byref(basic), ctypes.sizeof(basic),
        ))
        before = observe()
        original_names = tuple(sorted(storage.os.listdir(parent_fd)))
        marker = root / ('pending-' + bootstrap._key('unannounced') + '.json')
        assert marker.name not in original_names and marker not in paths
        payload = json.dumps({
            'version': 1, 'operation_id': 'unannounced', 'namespaces': ['profile'],
            'control_root': str(control), 'selectors': [str(selector)],
        }, sort_keys=True).encode('utf-8')
        child = storage.os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            assert os.write(child, payload) == len(payload)
        finally:
            os.close(child)
        assert tuple(sorted(storage.os.listdir(parent_fd))) == tuple(
            sorted((*original_names, marker.name))
        )
        # Directory ChangeTime is settable on Windows. Restore the old values
        # on the same actual directory while retaining the newly created entry.
        with native.reopen(parent_handle, wf._WRITE_ATTRIBUTES | wf._READ_ATTRIBUTES) as writable:
            native.check(native.kernel.SetFileInformationByHandle(
                writable, 0, ctypes.byref(basic), ctypes.sizeof(basic),
            ))
        expected_reason = 'recovery_pending'

    after = observe()
    metadata_equal = before == after
    assert bootstrap._admission_epoch == epoch
    # Current full acquisition must see the changed bytes/new entry. A future
    # optimization must preserve this refusal, even with identical fresh stamps.
    try:
        with storage.acquire_storage(target):
            raise AssertionError('changed control state reached protected payload')
    except bootstrap.RecoveryRequired as error:
        refusal = str(error)
    assert refusal == expected_reason, (route, refusal)
finally:
    try:
        if writer is not None:
            try:
                os.lseek(writer, 0, os.SEEK_SET)
                assert os.write(writer, original) == len(original)
            finally:
                close_fixture_descriptor(writer, writer_handle)
    finally:
        try:
            if marker is not None and marker.exists():
                marker.unlink()
        finally:
            try:
                if parent_fd is not None:
                    close_fixture_descriptor(parent_fd, parent_handle)
            finally:
                keeper.close()

assert len(retired_descriptors) == 1
assert set(storage._live_leases) == before_leases
assert set(storage._pending_acquisitions) == before_pending
assert hold.stop.is_set() and not hold.thread.is_alive()
with authority.maintenance(('profile',), 2):
    pass
with storage.acquire_storage(target) as restored:
    assert 'profile' in restored.execution_context(target)[1]
    assert target.read_text(encoding='utf-8') == 'native prerequisite payload'
assert not blocked_attempts()
print(json.dumps({'mutation': route, 'metadata_and_sd_unchanged': metadata_equal,
                  'full_refusal': refusal, 'retired_and_restored': True}))
# The premise is last: refusal, restoration, and physical retirement have all
# succeeded before this claims that metadata cannot distinguish the inputs.
assert metadata_equal, 'native fixture did not preserve the complete old metadata/SD projection'
print('retired and reopened')
"""
_ADVERSARIAL = (
    "from Tests.windows_private_fixture_runner import user_fixture_default_owner\n"
    "with user_fixture_default_owner():\n" + indent(_ADVERSARIAL, "    ")
)


@pytest.mark.parametrize("mutation", ["same-size-content", "new-pending-membership"])
def test_full_admission_refuses_changes_hidden_from_fresh_metadata(tmp_path, mutation):
    _run(tmp_path, mutation, "approved", script=_ADVERSARIAL)
