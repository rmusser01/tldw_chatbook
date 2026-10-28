"""Interrupted local publication resumes the existing native operation."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("boundary", ["staged", "committed", "selection"])
def test_resume_uses_same_journal_and_requires_activation(tmp_path, boundary):
    code = r"""
import json,sys
from pathlib import Path
from threading import Event
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery import isolated_restore as executor
from tldw_chatbook.Backup_Recovery.portable_relocation import (
    _directory,_resume,_select,_write,host_environment,read_selection,
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
plan=resolve_destinations(archive,mode='isolated',profile_bases={'profile':attempt/'profiles/restored'},external_destinations={},target=None,profile_names={'profile':'Recovered'})
_write(attempt,'publication-started.json',{'version':1,'location':str(base),'source_digest':'2'*64})
original_finish=executor._finish_isolated
def crash(*args,**kwargs):raise OSError('simulated stage interruption')
if sys.argv[2]=='staged':executor._finish_isolated=crash
try:
    profile=executor.restore_isolated(archive,plan,attempt/'control',Event())
except OSError as error:
    assert sys.argv[2]=='staged' and str(error)=='simulated stage interruption'
finally:executor._finish_isolated=original_finish
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
