"""Actual disposable APFS image tests; no native qualification overrides.

Set TLDW_TEST_APFS_MOUNT to the mounted private test image. The surrounding
fixture owns image attachment/detachment; these tests create only private children.
Raw primitive receipts precede the reviewed installed publication-only row.
"""

import json
import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

import pytest


def _run_image_case(tmp_path, program, receipt):
    supplied = os.environ.get("TLDW_TEST_APFS_MOUNT")
    if not supplied:
        pytest.skip("requires an explicitly mounted disposable APFS test image")
    mount = Path(supplied).resolve(strict=True)
    assert mount.is_dir() and mount.stat().st_dev != tmp_path.stat().st_dev
    image_source = Path(tempfile.mkdtemp(prefix="chatbook-test-", dir=mount))
    for name in ("home", "config", "data", "cache", "host", "logs", "tmp"):
        (tmp_path / name).mkdir(mode=0o700)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "LANG", "LC_ALL", "GOMODCACHE", "GOCACHE", "GOPROXY")
    }
    environment.update(
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        TMPDIR=str(tmp_path / "tmp"),
        TLDW_CONFIG_PATH=str(tmp_path / "config" / "config.toml"),
        VOLUME_FIXTURE=str(tmp_path),
        IMAGE_SOURCE=str(image_source),
        PYTHONPATH=str(Path(__file__).resolve().parents[2]),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    (tmp_path / "fixture.json").write_text(
        json.dumps(
            {
                "image_source": str(image_source),
                "source_device": image_source.stat().st_dev,
                "host_device": tmp_path.stat().st_dev,
            },
            indent=2,
        )
    )
    with (tmp_path / "child-output.log").open("w") as output:
        # The interpreter and child program are fixed, owned fixture code.
        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", program],
            cwd=environment["PYTHONPATH"],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=90,
            check=False,
        )
    assert result.returncode == 0, (tmp_path / "child-output.log").read_text()[-7000:]
    return json.loads((tmp_path / "logs" / receipt).read_text())


def test_actual_image_primitives_and_normal_qualification_boundary(tmp_path):
    evidence = _run_image_case(tmp_path, _RAW, "raw-evidence.json")
    assert evidence["host_authority_image_source"] == "normal_and_maintenance_passed"


def test_actual_image_database_and_files_capture_to_host_archive(tmp_path):
    evidence = _run_image_case(tmp_path, _CAPTURE, "capture-evidence.json")
    assert evidence["source_device"] != evidence["archive_device"]
    assert evidence["empty_directory_retained"]
    assert evidence["blocked_network_attempts"] == 0


def test_actual_image_private_tree_flush_and_unsafe_topology(tmp_path):
    evidence = _run_image_case(tmp_path, _TOPOLOGY, "topology-evidence.json")
    assert evidence["refused"] == [
        "symlink",
        "hardlink",
        "fifo",
        "linked_descendant",
        "foreign_device",
    ]


def test_actual_image_distinct_parent_barriers_and_lost_acknowledgement(tmp_path):
    evidence = _run_image_case(tmp_path, _BARRIERS, "barrier-evidence.json")
    assert len(evidence["cases"]) == 4


def test_actual_image_four_process_file_and_tree_races(tmp_path):
    evidence = _run_image_case(tmp_path, _RACES, "race-evidence.json")
    assert all(row["outcomes"].count("published") == 1 for row in evidence["cases"])


def test_actual_image_normal_publication_wrappers(tmp_path):
    evidence = _run_image_case(tmp_path, _WRAPPERS, "wrapper-evidence.json")
    assert evidence["image_admission"] == "refused_before_creation"
    assert len(evidence["existing_cases"]) == 27


# Raw evidence uses the installed ungated primitives. It never replaces a
# qualification function or runs an unqualified guarded publication wrapper.
_PROTOCOL_SETUP = r"""
from Tests.network_guard import install, blocked_attempts
install()
import errno, fcntl, json, os, stat
from pathlib import Path
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import native_files as native
root=Path(os.environ['VOLUME_FIXTURE'])
image=Path(os.environ['IMAGE_SOURCE'])
with native.pinned_directory(image) as fd:
    identity=native.native_identity(fd)
    device=os.fstat(fd).st_dev
assert identity=={'os':'Darwin','release':'25.5.0','arch':'arm64','python':'3.12.11','filesystem':'apfs','flags':76583448},identity
assert device!=root.stat().st_dev
def flush(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:native._flush_private_tree(fd,device)
    finally:os.close(fd)
def receipt(name,facts):
    with native.pinned_directory(image) as fd:
        assert native.native_identity(fd)==identity and os.fstat(fd).st_dev==device
    assert not blocked_attempts(),blocked_attempts()
    facts.update(identity=identity,device=device,blocked_network_attempts=0)
    (root/'logs'/name).write_text(json.dumps(facts,indent=2)+'\n')
"""


_TOPOLOGY = (
    _PROTOCOL_SETUP
    + r"""
tree=image/'tree';tree.mkdir(mode=0o755)
nested=tree/'nested';nested.mkdir(mode=0o755)
empty=tree/'empty';empty.mkdir(mode=0o755)
payload=nested/'payload';payload.write_bytes(b'private nested bytes');payload.chmod(0o644)
single=image/'single';single.write_bytes(b'private regular bytes');single.chmod(0o644)
paths=(payload,nested,empty,tree,single)
inodes={str(path.relative_to(image)):path.stat().st_ino for path in paths}
events=[]
real_fsync,real_fcntl=os.fsync,fcntl.fcntl
def observe_sync(fd):
    result=real_fsync(fd);events.append(['fsync',os.fstat(fd).st_ino]);return result
def observe_full(fd,command,*args):
    result=real_fcntl(fd,command,*args)
    if command==fcntl.F_FULLFSYNC:events.append(['full',os.fstat(fd).st_ino])
    return result
os.fsync=observe_sync;fcntl.fcntl=observe_full
try:flush(tree);flush(single)
finally:os.fsync=real_fsync;fcntl.fcntl=real_fcntl
for path in paths:
    inode=path.stat().st_ino
    assert events.count(['fsync',inode])==events.count(['full',inode])==1
    assert events.index(['fsync',inode])<events.index(['full',inode])
    assert stat.S_IMODE(path.stat().st_mode)==(0o700 if path.is_dir() else 0o600)
assert events.index(['full',payload.stat().st_ino])<events.index(['full',nested.stat().st_ino])<events.index(['full',tree.stat().st_ino])
assert events.index(['full',empty.stat().st_ino])<events.index(['full',tree.stat().st_ino])
assert payload.read_bytes()==b'private nested bytes' and single.read_bytes()==b'private regular bytes'
refused=[]
for kind in ('symlink','hardlink','fifo','linked_descendant'):
    case=image/kind;case.mkdir(mode=0o700)
    external=case/'original';external.write_bytes(b'unchanged original');external.chmod(0o644)
    candidate=case/'candidate'
    if kind=='symlink':candidate.symlink_to(external)
    elif kind=='hardlink':os.link(external,candidate)
    elif kind=='fifo':os.mkfifo(candidate)
    else:
        candidate.mkdir(mode=0o700);(candidate/'link').symlink_to(external)
    try:flush(candidate)
    except OSError as error:
        if kind in ('symlink','linked_descendant'):assert error.errno==errno.ELOOP,error
        else:assert str(error)==('staged_tree_linked' if kind=='hardlink' else 'staged_tree_not_regular'),error
    else:raise AssertionError('unsafe topology accepted: '+kind)
    assert external.read_bytes()==b'unchanged original' and stat.S_IMODE(external.stat().st_mode)==0o644
    refused.append(kind)
host_file=root/'host'/'different-device';host_file.write_bytes(b'host bytes')
fd=os.open(host_file,os.O_RDONLY|os.O_NOFOLLOW)
try:
    try:native._flush_private_tree(fd,device)
    except OSError as error:assert str(error)=='staged_tree_not_private'
    else:raise AssertionError('foreign device accepted')
finally:os.close(fd)
assert host_file.read_bytes()==b'host bytes'
refused.append('foreign_device')
receipt('topology-evidence.json',{'inodes':inodes,'events':events,'refused':refused})
"""
)


_BARRIERS = (
    _PROTOCOL_SETUP
    + r"""
facts=[]
for kind in ('file','empty_directory','populated_directory','lost_acknowledgement'):
    case=image/kind;case.mkdir(mode=0o700)
    a,b=case/'from',case/'to';a.mkdir(mode=0o700);b.mkdir(mode=0o700)
    source,target=a/'candidate',b/'published'
    if kind in ('file','lost_acknowledgement'):source.write_bytes(b'candidate bytes')
    else:
        source.mkdir(mode=0o755)
        if kind=='populated_directory':(source/'payload').write_bytes(b'candidate bytes')
    os.utime(source,ns=(1600000000000000000,1600000000000000000))
    flush(source)
    before=source.stat();events=[]
    real_fcntl=fcntl.fcntl
    def observe(fd,command,*args):
        result=real_fcntl(fd,command,*args)
        if command==fcntl.F_FULLFSYNC:
            events.append({'inode':os.fstat(fd).st_ino,'source_exists':source.exists(),'target_exists':target.exists()})
            if kind=='lost_acknowledgement' and os.fstat(fd).st_ino==b.stat().st_ino and target.exists():
                raise OSError(errno.EIO,'fixture_lost_barrier_acknowledgement')
        return result
    fcntl.fcntl=observe
    try:
        with native.pinned_directory(a) as left,native.pinned_directory(b) as right:
            native._rename_new(left,source.name,right,target.name)
            try:native.flush_directory(right);native.flush_directory(left)
            except OSError as error:
                assert kind=='lost_acknowledgement' and error.errno==errno.EIO,error
            else:assert kind!='lost_acknowledgement'
    finally:fcntl.fcntl=real_fcntl
    after=target.stat()
    assert not source.exists()
    assert (before.st_dev,before.st_ino,before.st_mode,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_mode,after.st_mtime_ns)
    expected=[b.stat().st_ino] if kind=='lost_acknowledgement' else [b.stat().st_ino,a.stat().st_ino]
    assert [e['inode'] for e in events]==expected
    assert all(not e['source_exists'] and e['target_exists'] for e in events)
    if target.is_file():assert target.read_bytes()==b'candidate bytes'
    if kind=='populated_directory':assert (target/'payload').read_bytes()==b'candidate bytes'
    if kind=='lost_acknowledgement':
        source.write_bytes(b'second candidate')
        with native.pinned_directory(a) as left,native.pinned_directory(b) as right:
            try:native._rename_new(left,source.name,right,target.name)
            except FileExistsError:pass
            else:raise AssertionError('lost acknowledgement overwrote destination')
        assert source.read_bytes()==b'second candidate' and target.read_bytes()==b'candidate bytes'
    facts.append({'kind':kind,'events':events,'preserved_inode':after.st_ino,'preserved_mtime_ns':after.st_mtime_ns,'mode':stat.S_IMODE(after.st_mode)})
for kind in ('regular','symlink','hardlink','directory'):
    case=image/('collision-'+kind);case.mkdir(mode=0o700)
    original=case/'original';original.write_bytes(b'original')
    target=case/'target';source=case/'source';source.write_bytes(b'candidate')
    if kind=='regular':target.write_bytes(b'previous')
    elif kind=='symlink':target.symlink_to(original)
    elif kind=='hardlink':os.link(original,target)
    else:target.mkdir(mode=0o700);(target/'retained').write_bytes(b'previous')
    before=target.lstat()
    with native.pinned_directory(case) as fd:
        try:native._rename_new(fd,source.name,fd,target.name)
        except FileExistsError:pass
        else:raise AssertionError('existing destination overwritten')
    assert target.lstat()==before and source.read_bytes()==b'candidate' and original.read_bytes()==b'original'
    if kind=='regular':assert target.read_bytes()==b'previous'
    if kind=='directory':assert (target/'retained').read_bytes()==b'previous'
receipt('barrier-evidence.json',{'cases':facts,'preserved_target_types':['regular','symlink','hardlink','directory']})
"""
)


_RACES = (
    _PROTOCOL_SETUP
    + r"""
import select,subprocess,sys
child_code=r'''
from Tests.network_guard import install
install()
import os,sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import native_files as native
source,target=Path(sys.argv[1]),Path(sys.argv[2])
fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW)
try:native._flush_private_tree(fd,os.fstat(fd).st_dev)
finally:os.close(fd)
with native.pinned_directory(source.parent) as a,native.pinned_directory(target.parent) as b:
    print('ready',flush=True)
    assert sys.stdin.readline()=='go\n'
    try:
        native._rename_new(a,source.name,b,target.name)
        native.flush_directory(b);native.flush_directory(a)
    except FileExistsError:print('exists',flush=True)
    else:print('published',flush=True)
'''
facts=[]
for kind in ('file','populated_directory'):
    case=image/kind;case.mkdir(mode=0o700)
    a,b=case/'from',case/'to';a.mkdir(mode=0o700);b.mkdir(mode=0o700)
    sources=[a/str(i) for i in range(4)];target=b/'winner'
    for index,source in enumerate(sources):
        if kind=='file':source.write_text(str(index))
        else:source.mkdir(mode=0o700);(source/'payload').write_text(str(index))
    inodes=[source.stat().st_ino for source in sources]
    children=[subprocess.Popen([sys.executable,'-u','-c',child_code,str(source),str(target)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for source in sources]
    try:
        for child in children:
            assert select.select([child.stdout],[],[],10)[0],'child not ready'
            assert child.stdout.readline().strip()=='ready'
        for child in children:child.stdin.write('go\n');child.stdin.flush()
        outputs=[child.communicate(timeout=10) for child in children]
        assert all(child.returncode==0 for child in children),outputs
        outcomes=[out.strip() for out,error in outputs]
        assert outcomes.count('published')==1 and outcomes.count('exists')==3,outcomes
        winner=outcomes.index('published')
        assert target.stat().st_ino==inodes[winner] and not sources[winner].exists()
        assert (target if kind=='file' else target/'payload').read_text()==str(winner)
        for index,source in enumerate(sources):
            if index==winner:continue
            assert source.stat().st_ino==inodes[index]
            assert (source if kind=='file' else source/'payload').read_text()==str(index)
        facts.append({'kind':kind,'outcomes':outcomes,'source_inodes':inodes,'winner_inode':target.stat().st_ino})
    finally:
        for child in children:
            if child.poll() is None:child.kill();child.wait(timeout=3)
receipt('race-evidence.json',{'cases':facts})
"""
)


_WRAPPERS = (
    _PROTOCOL_SETUP
    + r"""
import select,subprocess,sys
import pytest
from Tests.Backup_Recovery import test_native_files as existing
from tldw_chatbook.Backup_Recovery.qualification import qualified_for
from tldw_chatbook.Backup_Recovery.admission import Admission,AdmissionError
assert all(qualified_for(name,image)[0] for name in ('publish_new','publish_file','publish_directory'))
assert qualified_for('admission',image)==(False,'operation_not_qualified')
try:Admission(image/'control')
except AdmissionError as error:assert str(error)=='operation_not_qualified'
else:raise AssertionError('image admission granted')
assert not (image/'control').exists()
checks=[]
def check(function,*args,patched=False):
    path=image/('existing-'+str(len(checks)));path.mkdir(mode=0o700)
    if patched:
        with pytest.MonkeyPatch.context() as patch:function(path,patch,*args)
    else:function(path,*args)
    checks.append([function.__name__,*args])
for kind in ('file','empty_directory','populated_directory'):check(existing.test_qualified_publication,kind)
check(existing.test_unqualified_operations_and_symlink_storage_are_visible)
for kind in ('symlink','hardlink','fifo'):check(existing.test_non_private_staged_objects_are_refused,kind)
check(existing.test_publication_race_has_exactly_one_winner)
check(existing.test_directory_publication_refuses_linked_payload)
for kind in ('symlink','hardlink','directory'):check(existing.test_preexisting_target_of_any_type_is_preserved,kind)
check(existing.test_native_directory_full_flush_supported)
for kind in ('file','directory'):check(existing.test_full_flush_occurs_after_publication,kind,patched=True)
check(existing.test_failed_post_publication_full_flush_preserves_published_evidence,patched=True)
for invalid in ('missing_protocol','unsupported_protocol','previous_protocol','boolean_protocol','string_operations','mixed_operations','missing_operations','unknown_operation','unknown_field','boolean_schema','wrong_rows_type'):
    check(existing.test_malformed_qualification_evidence_never_grants_capability,invalid,patched=True)
assert len(checks)==27
distinct=[]
for kind in ('file','empty_directory','populated_directory'):
    case=image/('distinct-'+kind);case.mkdir(mode=0o700)
    a,b=case/'from',case/'to';a.mkdir(mode=0o700);b.mkdir(mode=0o700)
    source,target=a/'candidate',b/'published'
    if kind=='file':source.write_bytes(b'wrapper bytes')
    else:
        source.mkdir(mode=0o700)
        if kind=='populated_directory':(source/'payload').write_bytes(b'wrapper bytes')
    os.utime(source,ns=(1600000000000000000,1600000000000000000));before=source.stat()
    events=[];real=fcntl.fcntl;parents=[b.stat().st_ino,a.stat().st_ino]
    def observe(fd,command,*args):
        result=real(fd,command,*args)
        if command==fcntl.F_FULLFSYNC:events.append((os.fstat(fd).st_ino,source.exists(),target.exists()))
        return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(native.fcntl,'fcntl',observe)
        native.publish_new(source,target)
    assert not source.exists() and target.stat().st_ino==before.st_ino and target.stat().st_mtime_ns==before.st_mtime_ns
    assert stat.S_IMODE(target.stat().st_mode)==(0o600 if kind=='file' else 0o700)
    parent_events=[event for event in events if event[0] in parents]
    assert parent_events==[(parents[0],False,True),(parents[1],False,True)],events
    assert (before.st_ino,True,False) in events
    if kind=='file':assert target.read_bytes()==b'wrapper bytes'
    if kind=='populated_directory':assert (target/'payload').read_bytes()==b'wrapper bytes'
    distinct.append({'kind':kind,'events':events,'inode':target.stat().st_ino})
cross=root/'host'/'cross-source';cross.write_bytes(b'host candidate')
try:native.publish_new(cross,image/'cross-target')
except OSError as error:assert str(error)=='cross_volume_publication_unqualified',error
else:raise AssertionError('cross-device wrapper accepted')
assert cross.read_bytes()==b'host candidate' and not (image/'cross-target').exists()
changed=[]
for side in ('source','target'):
    case=image/('changed-'+side);case.mkdir(mode=0o700)
    a,b=case/'from',case/'to';a.mkdir(mode=0o700);b.mkdir(mode=0o700)
    source,target=a/'candidate',b/'published';source.write_bytes(b'preserved candidate')
    identities=tuple((path.stat().st_dev,path.stat().st_ino) for path in (a,b))
    moved=a if side=='source' else b;held=case/'held';moved.rename(held);moved.mkdir(mode=0o700)
    try:native.publish_new(source,target,parent_identities=identities)
    except OSError as error:assert str(error)=='publication_parent_changed',error
    else:raise AssertionError('changed pinned parent accepted')
    actual_source=held/'candidate' if side=='source' else source
    assert actual_source.read_bytes()==b'preserved candidate' and not target.exists()
    changed.append(side)
race=image/'tree-race';race.mkdir(mode=0o700)
sources=[race/str(i) for i in range(4)];target=race/'winner'
for i,source in enumerate(sources):source.mkdir(mode=0o700);(source/'payload').write_text(str(i))
inodes=[source.stat().st_ino for source in sources]
code=r'''
from Tests.network_guard import install
install()
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.native_files import publish_new
source,target=map(Path,sys.argv[1:])
print('ready',flush=True)
assert sys.stdin.readline()=='go\n'
try:publish_new(source,target)
except FileExistsError:print('exists',flush=True)
else:print('published',flush=True)
'''
children=[subprocess.Popen([sys.executable,'-u','-c',code,str(source),str(target)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for source in sources]
try:
    for child in children:
        assert select.select([child.stdout],[],[],10)[0]
        assert child.stdout.readline().strip()=='ready'
    for child in children:child.stdin.write('go\n');child.stdin.flush()
    outputs=[child.communicate(timeout=10) for child in children]
    assert all(child.returncode==0 for child in children),outputs
    outcomes=[out.strip() for out,error in outputs]
    assert outcomes.count('published')==1 and outcomes.count('exists')==3,outcomes
    winner=outcomes.index('published');assert target.stat().st_ino==inodes[winner] and not sources[winner].exists()
    assert (target/'payload').read_text()==str(winner)
    for index,source in enumerate(sources):
        if index!=winner:assert source.stat().st_ino==inodes[index] and (source/'payload').read_text()==str(index)
finally:
    for child in children:
        if child.poll() is None:child.kill();child.wait(timeout=3)
receipt('wrapper-evidence.json',{'existing_cases':checks,'distinct_parent_cases':distinct,'changed_parents':changed,'cross_device':'refused_source_intact','tree_race':outcomes,'image_admission':'refused_before_creation'})
"""
)


_RAW = r"""
from Tests.network_guard import install, blocked_attempts
install()
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery.native_files import pinned_directory, native_identity, create_private_directory, create_private_file, flush_directory, _rename_new, publish_new
from tldw_chatbook.Backup_Recovery.qualification import qualified_for
from tldw_chatbook.Backup_Recovery.admission import Admission, AdmissionError

root = Path(os.environ['VOLUME_FIXTURE'])
host = root / 'host'
image = Path(os.environ['IMAGE_SOURCE'])
facts = {'identities': {}, 'raw': [], 'qualification': {}}
for label, path in (('host', host), ('image', image)):
    with pinned_directory(path) as fd:
        facts['identities'][label] = {'identity': native_identity(fd), 'device': os.fstat(fd).st_dev}
    facts['qualification'][label] = {operation: qualified_for(operation, path) for operation in ('publish_new','publish_file','publish_directory','admission')}
assert facts['identities']['host']['device'] != facts['identities']['image']['device']
raw = image / 'raw'
create_private_directory(raw)
for kind in ('file', 'empty_directory', 'populated_directory'):
    case = raw / kind
    create_private_directory(case)
    source, existing = case/'source', case/'existing'
    if kind == 'file':
        source.write_bytes(b'candidate'); existing.write_bytes(b'previous')
    else:
        source.mkdir(mode=0o700); existing.mkdir(mode=0o700)
        if kind == 'populated_directory':
            (source/'payload').write_bytes(b'candidate'); (existing/'payload').write_bytes(b'previous')
    with pinned_directory(case) as fd:
        try: _rename_new(fd, 'source', fd, 'existing')
        except FileExistsError: pass
        else: raise AssertionError('no-replace overwritten')
        assert source.exists() and existing.exists()
        _rename_new(fd, 'source', fd, 'published')
        flush_directory(fd)
    assert not source.exists() and (case/'published').exists()
    if kind == 'file': assert existing.read_bytes()==b'previous' and (case/'published').read_bytes()==b'candidate'
    if kind == 'populated_directory': assert (existing/'payload').read_bytes()==b'previous' and (case/'published/payload').read_bytes()==b'candidate'
    facts['raw'].append(kind)
with create_private_file(raw/'private-file') as fd: os.write(fd,b'private')
assert (raw/'private-file').stat().st_mode & 0o777 == 0o600
facts['raw'].append('private_creation_and_full_flush')
cross = host/'cross-source'
cross.write_bytes(b'cross-device retained')
with pinned_directory(host) as a, pinned_directory(raw) as b:
    try: _rename_new(a,'cross-source',b,'cross-destination')
    except OSError as error: assert error.errno==errno.EXDEV,error
    else: raise AssertionError('cross-device rename unexpectedly succeeded')
assert cross.read_bytes()==b'cross-device retained' and not (raw/'cross-destination').exists()
facts['raw'].append('EXDEV_preserved_source')
death = raw/'death'
create_private_directory(death)
(death/'source').write_bytes(b'abrupt native evidence')
child = "from Tests.network_guard import install; install(); import os,sys; from pathlib import Path; from tldw_chatbook.Backup_Recovery.native_files import pinned_directory,_rename_new;\nwith pinned_directory(Path(sys.argv[1])) as fd:\n _rename_new(fd,'source',fd,'published'); os._exit(23)"
run = subprocess.run([sys.executable,'-c',child,str(death)],timeout=10)
assert run.returncode==23 and not (death/'source').exists() and (death/'published').read_bytes()==b'abrupt native evidence'
facts['raw'].append('process_exit_after_raw_rename')
candidate = raw/'gated-candidate'
candidate.write_bytes(b'unqualified candidate');candidate.chmod(0o600)
assert all(facts['qualification']['image'][name][0] for name in ('publish_new','publish_file','publish_directory'))
publish_new(candidate,raw/'gated-target')
assert not candidate.exists() and (raw/'gated-target').read_bytes()==b'unqualified candidate'
facts['gated_publication']='qualified_publication_passed'
assert not facts['qualification']['image']['admission'][0]
if not facts['qualification']['image']['admission'][0]:
    try: Admission(raw/'control')
    except AdmissionError as error: assert str(error)==facts['qualification']['image']['admission'][1],error
    else: raise AssertionError('unqualified admission accepted')
    assert not (raw/'control').exists()
    facts['image_authority']='refused_before_creation'
authority = Admission(host/'control')
authority.register('image.source',(raw/'private-file',))
with authority.normal(('image.source',)): assert (raw/'private-file').read_bytes()==b'private'
with authority.maintenance(('image.source',),3): assert (raw/'private-file').read_bytes()==b'private'
facts['host_authority_image_source']='normal_and_maintenance_passed'
assert not blocked_attempts(),blocked_attempts()
(root/'logs/raw-evidence.json').write_text(json.dumps(facts,indent=2)+'\n')
print(json.dumps(facts),flush=True)
"""


_CAPTURE = r"""
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import threading
import zipfile
from contextlib import closing
from pathlib import Path

from Tests.network_guard import blocked_attempts, install

install()
for name in ("sounddevice", "pyaudio"):
    sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring

keyring.set_keyring(Keyring())
root = Path(os.environ["VOLUME_FIXTURE"])
selector = Path(os.environ["TLDW_CONFIG_PATH"])
image_source = Path(os.environ["IMAGE_SOURCE"])
external = image_source / "documents"
external.mkdir(mode=0o700)
(external / "empty").mkdir(mode=0o700)
document = external / "retained.txt"
document.write_bytes(b"Actual second-volume document\n")
document.chmod(0o600)
research_path = image_source / "research.db"
(root / "data" / "profile").mkdir(mode=0o700)
selector.write_text(
    '[general]\nusers_name="default_user"\ndefault_tab="settings"\n'
    '[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n'
    '[paths]\ndata_dir=' + json.dumps(str(root / "data" / "profile")) + '\n'
    '[database]\nresearch_db_path=' + json.dumps(str(research_path)) + '\n'
)
selector.chmod(0o600)

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
from tldw_chatbook.Backup_Recovery.archive_reader import acquire, verify_sealed
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.capture_service import capture, preview_capture
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.runtime_maintenance import monitor_app


async def main():
    app = TldwCli()
    monitoring = None
    cancel = threading.Event()
    watchdog = None
    try:
        research = app.local_research_service
        assert research.db_path == research_path
        saved = research.create_session(title="Image research", query="Retained on image")
        assert research.get_session(saved["id"])["query"] == "Retained on image"
        assert research_path.stat().st_dev != selector.stat().st_dev
        options = {"allow_partial": True, "staging_parent": root / "host", "external_roots": (external,)}
        preview = preview_capture((selector,), options=options)
        rows = [item for item in preview.items if item.path is not None and (item.path == research_path or external == item.path or external in item.path.parents)]
        assert any(item.owner == "research.local" and item.path == research_path and item.status == "included" for item in rows)
        assert any(item.path == document and item.status == "included" for item in rows)
        assert any(item.path == external / "empty" and item.status == "included_directory" for item in rows)
        (root / "logs" / "capture-preview.json").write_text(json.dumps({"complete": preview.complete, "issues": preview.issues, "selected_image_rows": [{"id": item.logical_id, "owner": item.owner, "path": str(item.path), "status": item.status} for item in rows]}, indent=2))
        print("PREVIEW", preview.complete, preview.issues, flush=True)
        destination = root / "host" / "image-source.tldw-backup.zip"
        monitoring = asyncio.create_task(monitor_app(app))
        watchdog = asyncio.get_running_loop().call_later(55, cancel.set)
        result = await asyncio.to_thread(capture, (selector,), preview.scope_digest, destination, options=options, cancel=cancel)
        print("CAPTURED", flush=True)
        for _ in range(500):
            if storage_admission._pause is None and app._backup_runtime_maintenance is None:
                break
            await asyncio.sleep(.01)
        assert storage_admission._pause is None and app._backup_runtime_maintenance is None
        resumed = research.create_session(title="After capture", query="Resumed original writer")
        assert research.get_session(resumed["id"])["query"] == "Resumed original writer"
        manifest = json.loads(result.manifest_bytes)
        (root / "logs" / "capture-manifest.json").write_bytes(result.manifest_bytes)
        logical_paths = {item.logical_id: item.path for item in result.inventory.items}
        research_member = next(member for member in manifest["files"] if logical_paths[member["logical_id"]] == research_path)
        document_member = next(member for member in manifest["files"] if logical_paths[member["logical_id"]] == document)
        assert (result.root / document_member["payload"]).read_bytes() == document.read_bytes()
        assert result.root.stat().st_dev == selector.stat().st_dev
        assert bootstrap.default_bootstrap_root().stat().st_dev == selector.stat().st_dev
        sealed_output = await asyncio.to_thread(write_archive, result, destination, password=None, cancel=cancel)
        sealed = await asyncio.to_thread(acquire, destination, root / "host" / "readback", ArchiveLimits(), None, cancel)
        verify_sealed(sealed, cancel)
        final_manifest = json.loads(sealed.manifest_bytes)
        assert final_manifest["consistency"] == "partial"  # Explicit external files.
        assert final_manifest["files"] == manifest["files"]
        with zipfile.ZipFile(sealed.path) as archive:
            assert archive.read(document_member["payload"]) == document.read_bytes()
            readback = root / "host" / "research-readback.db"
            readback.write_bytes(archive.read(research_member["payload"]))
        with closing(sqlite3.connect(readback.as_uri() + "?mode=ro", uri=True)) as connection:
            assert connection.execute("SELECT id,title,query FROM research_sessions").fetchall() == [(saved["id"], "Image research", "Retained on image")]
        assert research.get_session(saved["id"])["query"] == "Retained on image"
        assert not blocked_attempts()
        evidence = {"consistency": final_manifest["consistency"], "inventory_issues": list(preview.issues), "source_device": research_path.stat().st_dev, "control_device": bootstrap.default_bootstrap_root().stat().st_dev, "stage_device": result.root.stat().st_dev, "archive_device": destination.stat().st_dev, "archive": str(destination), "archive_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "research_id": saved["id"], "resumed_id": resumed["id"], "archive_files": len(final_manifest["files"]), "empty_directory_retained": any(logical_paths[entry["logical_id"]] == external / "empty" for entry in final_manifest["directories"]), "blocked_network_attempts": len(blocked_attempts())}
        (root / "logs" / "capture-evidence.json").write_text(json.dumps(evidence, indent=2))
        print("SUCCESS", json.dumps(evidence), flush=True)
    finally:
        if watchdog is not None:
            watchdog.cancel()
        cancel.set()
        if monitoring is not None:
            monitoring.cancel()
            try:
                await monitoring
            except asyncio.CancelledError:
                pass
        await app._shutdown_app_owned_lifecycles()
        await app.tts_service.close()


asyncio.run(main())
"""
