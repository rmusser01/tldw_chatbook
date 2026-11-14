"""Native ancestry checks visit every parent once and retire borrowed descriptors."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = r"""
import os,stat,subprocess,sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import control_records as records,native_files

case=sys.argv[1]
home=Path.home()
parents=[home]
for index in range(5):
 child=parents[-1]/f'level-{index}';child.mkdir(mode=0o700);parents.append(child)
root=parents[-1]/'bootstrap'
native=native_files.os
original_open,original_close=native.open,native.close
opened=[];live=set();visits=[]
def tracked_open(*args,**kwargs):
 fd=original_open(*args,**kwargs);opened.append(fd);live.add(fd);return fd
def tracked_close(fd):
 original_close(fd);live.discard(fd)
def visit(fd):
 info=native.fstat(fd)
 assert stat.S_ISDIR(info.st_mode)
 visits.append((info.st_dev,info.st_ino))
native.open=tracked_open;native.close=tracked_close
try:
 if case=='linear':
  records._require_settled_ancestry(root)
  expected=len(root.parent.parts)
  print('native directory opens:',len(opened),'linear bound:',expected,flush=True)
  assert len(opened)==expected,(len(opened),expected)
 elif case=='all-parents':
  with native_files.pinned_directory(root.parent,_visit=visit):pass
  expected=[]
  for parent in reversed((root.parent,*root.parent.parents)):
   info=native.stat(parent);expected.append((info.st_dev,info.st_ino))
  assert visits==expected,(visits,expected)
  for parent in parents:
   marker=parent/records._DIRECTORY_CREATION_INTENT
   marker.write_bytes(b'incomplete');marker.chmod(0o600)
   try:
    try:records._require_settled_ancestry(root)
    except records.RecoveryRequired as error:assert str(error)=='bootstrap_creation_unsettled'
    else:raise AssertionError(f'creation intent ignored at {parent}')
    assert marker.read_bytes()==b'incomplete'
   finally:marker.unlink()
 elif case=='missing':
  missing=root/'missing'/'deeper'
  records._require_settled_ancestry(missing)
  for parent in parents:
   marker=parent/records._DIRECTORY_CREATION_INTENT
   marker.write_bytes(b'incomplete');marker.chmod(0o600)
   try:
    try:records._require_settled_ancestry(missing)
    except records.RecoveryRequired:pass
    else:raise AssertionError('missing descendant hid existing ancestor intent')
   finally:marker.unlink()
 elif case=='junction':
  alias=home/'alias'
  if os.name=='nt':
   subprocess.run(['cmd','/c','mklink','/J',str(alias),str(parents[-1])],check=True,capture_output=True)
  else:alias.symlink_to(parents[-1],target_is_directory=True)
  try:
   try:records._require_settled_ancestry(alias/'bootstrap')
   except OSError:pass
   else:raise AssertionError('directory alias admitted')
   assert alias.exists()
  finally:
   if os.name=='nt':alias.rmdir()
   else:alias.unlink()
 elif case=='mode':
  unsafe=parents[2]
  if os.name=='nt':
   subprocess.run(['icacls',str(unsafe),'/grant','*S-1-1-0:(W)'],check=True,capture_output=True)
  else:unsafe.chmod(0o777)
  assert native.stat(unsafe).st_mode & 0o022
  try:
   try:
    with native_files.pinned_directory(root.parent,_visit=visit):pass
   except OSError:pass
   else:raise AssertionError('shared-writable ancestor admitted')
   info=native.stat(unsafe)
   assert (info.st_dev,info.st_ino) not in visits,'visitor ran before posture validation'
   assert native.stat(unsafe).st_mode & 0o022,'read repaired unsafe mode'
  finally:
   if os.name=='nt':subprocess.run(['icacls',str(unsafe),'/remove:g','*S-1-1-0'],check=True,capture_output=True)
   else:unsafe.chmod(0o700)
 elif case.startswith('visitor-'):
  position=1 if case=='visitor-first' else len(root.parent.parts)
  def fail_visitor(fd):
   visit(fd)
   if len(visits)==position:raise RuntimeError('visitor failed')
  try:
   with native_files.pinned_directory(root.parent,_visit=fail_visitor):pass
  except RuntimeError as error:assert str(error)=='visitor failed'
  else:raise AssertionError('visitor failure disappeared')
  assert len(visits)==position
 elif case=='close':
  selected=[]
  def failed_close(fd):
   tracked_close(fd)
   if fd in selected:raise RuntimeError('native close failed')
  native.close=failed_close
  try:
   with native_files.pinned_directory(root.parent,_visit=visit) as fd:selected.append(fd)
  except RuntimeError as error:assert str(error)=='native close failed'
  else:raise AssertionError('native close failure disappeared')
 else:raise AssertionError(case)
 assert not live,('unretired native descriptors',live)
finally:
 native.open=original_open;native.close=original_close
 for fd in live:original_close(fd)
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "case",
    [
        "linear",
        "all-parents",
        "missing",
        "junction",
        "mode",
        "visitor-first",
        "visitor-last",
        "close",
    ],
)
def test_native_ancestry_walk(tmp_path, case):
    _run(tmp_path, case, "native", script=_SCRIPT, timeout=30)
