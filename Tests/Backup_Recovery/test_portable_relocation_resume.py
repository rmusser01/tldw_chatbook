"""Interrupted local publication resumes the existing native operation."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "boundary",
    [
        "staged",
        "committed",
        "selection",
        "marker",
        "marker_failure",
        "before_receipt",
        "home_escape",
    ],
)
def test_resume_uses_same_journal_and_requires_activation(tmp_path, boundary):
    code = r"""
import json,sys
from pathlib import Path
from threading import Event
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery import isolated_restore as executor
from tldw_chatbook.Backup_Recovery.portable_relocation import (
    _directory,_resume,_select,_write,host_environment,read_selection,pending_attempt,
)
from tldw_chatbook.Backup_Recovery.destinations import resolve_destinations
from tldw_chatbook.Backup_Recovery.native_files import create_private_directory
base=Path(sys.argv[1]);base.mkdir()
data=base/'Data';data.mkdir()
work=data/'.portable-relocation';_directory(work)
attempt=work/('attempt-'+'1'*32);_directory(attempt)
for leaf in ('home','profiles','work'):_directory(attempt/leaf)
host_environment(data,attempt.name)
def config_manifest(doc):
    doc['owners'][0]['owner_id']='config'
    doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
    doc['dependency_groups'][0]['members']=['profile:profile:config']
archive=sealed(attempt/'work',mutate=config_manifest,data=b'[general]\nusers_name="fixture"\n'.replace(b'\\n',b'\n'),partial=True)
profile_base=attempt/'profiles/restored'
if sys.argv[2]=='home_escape':
    profile_base=attempt/'home/unrelated/restored';_directory(profile_base.parent)
plan=resolve_destinations(archive,mode='isolated',profile_bases={'profile':profile_base},external_destinations={},target=None,profile_names={'profile':'Recovered'})
boundary=sys.argv[2]
def marker():
    operations=list((attempt/'control').glob('operation-*'))
    assert len(operations)==1
    receipt=operations[0]/'000000.json'
    assert json.loads(receipt.read_text())['event']=='candidate_staged'
    assert not list((attempt/'profiles').rglob('config.toml'))
    assert not list((attempt/'control').glob('pending-*.json'))
    if boundary=='marker_failure':raise OSError('simulated marker failure')
    _write(attempt,'publication-started.json',{'version':1,'location':str(base),'source_digest':'2'*64})
    if boundary=='marker':raise OSError('simulated marker interruption')
if boundary in ('staged','committed','selection','home_escape'):
    _write(attempt,'publication-started.json',{'version':1,'location':str(base),'source_digest':'2'*64})
original_record=None
if boundary=='before_receipt':
    from tldw_chatbook.Backup_Recovery.journal import Journal
    original_record=Journal.record_candidate
    def before_receipt(*args,**kwargs):raise OSError('simulated receipt interruption')
    Journal.record_candidate=before_receipt
original_finish=executor._finish_isolated
def crash(*args,**kwargs):raise OSError('simulated stage interruption')
if sys.argv[2] in ('staged','home_escape'):executor._finish_isolated=crash
try:
    if boundary in ('marker','marker_failure','before_receipt'):
        profile=executor.restore_isolated(archive,plan,attempt/'control',Event(),on_staged=marker)
    else:profile=executor.restore_isolated(archive,plan,attempt/'control',Event())
except OSError as error:
    assert str(error) in ('simulated stage interruption','simulated marker interruption','simulated marker failure','simulated receipt interruption'),str(error)
finally:
    executor._finish_isolated=original_finish
    if original_record is not None:Journal.record_candidate=original_record
if boundary in ('marker_failure','before_receipt'):
    assert pending_attempt(data) is None
    assert not list((attempt/'profiles').rglob('config.toml'))
    assert not list((attempt/'control').glob('pending-*.json'))
    assert not list((attempt/'control').glob('profile-catalog/*.json'))
    assert not list((attempt/'home').rglob('pending-*.json'))
    if boundary=='before_receipt':assert not list((attempt/'control').glob('operation-*/000000.json'))
    else:assert len(list((attempt/'control').glob('operation-*/000000.json')))==1
    print('same-operation-resumed',flush=True)
    sys.exit(0)
if sys.argv[2]=='selection':
    import tldw_chatbook.Backup_Recovery.portable_relocation as portable
    original_write=portable._write
    def before_selection(root,name,value):
        if name.startswith('selection-'):raise OSError('simulated selection interruption')
        original_write(root,name,value)
    portable._write=before_selection
    try:_select(data,attempt,profile,'2'*64)
    except OSError as error:assert str(error)=='simulated selection interruption'
    finally:portable._write=original_write
before={p.name for p in (attempt/'control').glob('operation-*')}
assert len(before)==1
if boundary=='home_escape':
    try:_resume(data,attempt)
    except ValueError as error:assert str(error)=='portable_relocation_recovery_required',str(error)
    else:raise AssertionError('non-pet home destination accepted')
    assert read_selection(data) is None
    print('same-operation-resumed',flush=True)
    sys.exit(0)
selected=_resume(data,attempt)
assert {p.name for p in (attempt/'control').glob('operation-*')}==before
assert read_selection(data)==selected
entry,_=executor._launch_state(selected.profile_id,attempt/'control')
assert entry.profile_id==selected.profile_id
requirements=executor.profile_requirements(selected.profile_id,attempt/'control')
assert requirements['pending_owners']==requirements['required_owners']
assert requirements['required_owners']
print('same-operation-resumed',flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "fixture"), boundary],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-5000:] + result.stdout[-1000:]
    assert "same-operation-resumed" in result.stdout


@pytest.mark.parametrize("damage", ["missing", "empty", "malformed", "ambiguous"])
def test_old_unreceipted_publication_still_refuses(tmp_path, damage):
    code = r"""
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.portable_relocation import _directory,_resume,_write,read_selection
base=Path(sys.argv[1]);base.mkdir()
data=base/'Data';data.mkdir()
attempt=data/'.portable-relocation'/('attempt-'+'1'*32)
_directory(attempt.parent);_directory(attempt)
for name in ('home','profiles','control'):_directory(attempt/name)
_write(attempt,'publication-started.json',{'version':1,'location':str(base),'source_digest':'2'*64})
if sys.argv[2]!='missing':
    operation=attempt/'control'/('operation-'+'3'*64);_directory(operation)
    if sys.argv[2]=='malformed':_write(operation,'000000.json',{'operation_id':'4'*32})
    if sys.argv[2]=='ambiguous':_directory(attempt/'control'/('operation-'+'5'*64))
before={str(p.relative_to(attempt)):p.read_bytes() for p in attempt.rglob('*') if p.is_file()}
try:_resume(data,attempt)
except ValueError as error:assert str(error)=='portable_relocation_recovery_required',str(error)
else:raise AssertionError('unreceipted publication accepted')
assert read_selection(data) is None
assert not list((attempt/'profiles').rglob('config.toml'))
assert all((attempt/p).read_bytes()==value for p,value in before.items())
assert len(list(attempt.parent.glob('attempt-*')))==1
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "fixture"), damage],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-5000:] + result.stdout[-1000:]


@pytest.mark.parametrize("old_order", [False, True])
def test_coordinator_pre_receipt_interruption_retry_and_old_order_control(
    tmp_path, old_order
):
    code = r"""
import json,sys
from pathlib import Path
from Tests.Backup_Recovery.test_archive_reader import sealed
from tldw_chatbook.Backup_Recovery import portable_relocation as portable, isolated_restore as executor
from tldw_chatbook.Backup_Recovery.journal import Journal
base=Path(sys.argv[1]);base.mkdir();data=base/'Data';data.mkdir()
(data/'retained.txt').write_text('unchanged copied source')
def config_manifest(doc):
    doc['owners'][0]['owner_id']='config'
    doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
    doc['dependency_groups'][0]['members']=['profile:profile:config']
def produce(source,work):
    portable._directory(work)
    archive=sealed(work,mutate=config_manifest,data=b'[general]\nusers_name="fixture"\n',partial=True)
    return archive.path,()
portable._produce=produce
original_record=Journal.record_candidate
def interrupt(*args,**kwargs):raise OSError('actual pre-receipt interruption')
Journal.record_candidate=interrupt
original_restore=executor.restore_isolated
if sys.argv[2]=='True':
    def old_restore(archive,plan,control,cancel,**kwargs):
        # Faithful negative control for the original coordinator ordering:
        # complete the outer marker before entering the real executor.
        attempt=control.parent
        portable._write(attempt,'publication-started.json',{'version':1,'location':str(base),'source_digest':'2'*64})
        return original_restore(archive,plan,control,cancel,**kwargs)
    executor.restore_isolated=old_restore
try:portable.relocate(data,lambda *_:True)
except OSError as error:assert str(error)=='actual pre-receipt interruption',str(error)
else:raise AssertionError('interruption not reached')
finally:
    Journal.record_candidate=original_record
    executor.restore_isolated=original_restore
attempts=list((data/portable.WORK_NAME).glob('attempt-*'));assert len(attempts)==1
first=attempts[0]
assert not list(first.glob('control/operation-*/000000.json'))
assert not list(first.glob('control/profile-catalog/*.json'))
assert not list(first.glob('home/**/pending-*.json'))
assert not list(first.glob('profiles/**/config.toml'))
if sys.argv[2]=='True':
    assert portable.pending_attempt(data)==first
    try:portable.relocate(data,lambda *_:True)
    except ValueError as error:assert str(error)=='portable_relocation_recovery_required',str(error)
    else:raise AssertionError('old-order ambiguous retry accepted')
    assert len(list((data/portable.WORK_NAME).glob('attempt-*')))==1
else:
    assert portable.pending_attempt(data) is None
    selected=portable.relocate(data,lambda *_:True)
    assert selected is not None and selected.attempt!=first.name
    entry,_=executor._launch_state(selected.profile_id,data/portable.WORK_NAME/selected.attempt/'control')
    assert entry.profile_id==selected.profile_id
assert (data/'retained.txt').read_text()=='unchanged copied source'
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "fixture"), str(old_order)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-5000:] + result.stdout[-1000:]
