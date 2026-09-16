"""Only the installed default tokenizer root may retain an empty private inode."""

import hashlib
import json
import os
import zipfile
from copy import deepcopy
from threading import Event

import pytest

from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery.archive_reader import acquire
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore, recheck_targets


def _archive(base, *, owner="tokenizers.custom", shared=None, config=False):
    """Seal untrusted declared inputs with real archive authentication."""
    data = b'{"retained":"tokenizer document"}'
    doc = manifest(data)
    doc["owners"][0]["owner_id"] = owner
    doc["files"][0].update(owner_id=owner, relative_path="retained.json")
    doc["directories"][0]["metadata"].update(mode=0o700, mtime_ns=1234567890)
    doc["producer_inventory"] = [
        {
            "logical_id": "root",
            "owner_id": owner,
            "status": "included_directory",
            "dependencies": [],
            "shared_group": None,
        },
        {
            "logical_id": "file",
            "owner_id": owner,
            "status": "included",
            "dependencies": ["root"],
            "shared_group": None,
        },
    ]
    payloads = {"payload/1": data}
    if shared is not None:
        for row in doc["producer_inventory"]:
            row["shared_group"] = "shared-" + row["logical_id"]
        for field in ("directories", "files", "producer_inventory"):
            for original in tuple(doc[field]):
                row = deepcopy(original)
                row["logical_id"] = "other-" + row["logical_id"]
                if field == "producer_inventory":
                    row["dependencies"] = [
                        "other-" + key for key in row["dependencies"]
                    ]
                else:
                    row["root_id"] = "other-root"
                    if row["parent_id"] is not None:
                        row["parent_id"] = "other-root"
                    if field == "files":
                        row["payload"] = "payload/other"
                        payloads[row["payload"]] = data
                doc[field].append(row)
        if shared == "metadata":
            doc["directories"][1]["metadata"]["mtime_ns"] += 1
        elif shared == "group":
            doc["producer_inventory"][2]["shared_group"] = None
        elif shared == "closure":
            doc["files"][1]["relative_path"] = "different.json"
    if config:
        config_data = b'[general]\nusers_name="restored"\n'
        doc["owners"].append(
            {"owner_id": "config", "schema_version": 1, "capabilities": []}
        )
        doc["directories"].append(
            {
                "logical_id": "config-root",
                "root_id": "config-root",
                "parent_id": None,
                "relative_path": "",
                "synthetic": True,
                "metadata": {"version": 1, "mode": 0o700, "mtime_ns": 0},
            }
        )
        doc["files"].append(
            {
                "logical_id": "profile:profile:config",
                "root_id": "config-root",
                "parent_id": "config-root",
                "relative_path": "config.toml",
                "owner_id": "config",
                "payload": "payload/config",
                "size": len(config_data),
                "sha256": hashlib.sha256(config_data).hexdigest(),
            }
        )
        doc["producer_inventory"].extend(
            [
                {
                    "logical_id": "config-root",
                    "owner_id": "config",
                    "status": "included_directory",
                    "dependencies": [],
                    "shared_group": None,
                },
                {
                    "logical_id": "profile:profile:config",
                    "owner_id": "config",
                    "status": "included",
                    "dependencies": ["config-root"],
                    "shared_group": None,
                },
            ]
        )
        payloads["payload/config"] = config_data
    doc["dependency_groups"] = [
        {
            "group_id": "all",
            "members": [row["logical_id"] for row in doc["producer_inventory"]],
            "complete": True,
        }
    ]
    path = base / "tokenizers.zip"
    with zipfile.ZipFile(path, "w") as stream:
        stream.writestr("manifest.json", json.dumps(doc))
        for name, value in payloads.items():
            stream.writestr(name, value)
    return acquire(path, base / "archive-work", ArchiveLimits(), None, Event())


def _destination(base):
    root = base / ".config" / "tldw_cli" / "tokenizers"
    root.mkdir(parents=True, mode=0o700)
    return root


def test_default_empty_concrete_tokenizer_root_keeps_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = _destination(tmp_path)
    before = root.stat()
    plan = plan_restore(
        _archive(tmp_path), mode="isolated", destinations={"root": root}, target=None
    )
    assert dict(plan.restore) == {"root": root, "file": root / "retained.json"}
    assert (
        next(applied for key, _, applied in plan.metadata if key == "root").mtime_ns
        == 1234567890
    )
    assert plan.containers == () and plan.retire == () and plan.target is None
    recheck_targets(plan)
    after = root.stat()
    assert (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "kind",
    [
        "owner",
        "path",
        "occupied",
        "mode",
        "symlink",
        "dangling",
        "file",
        "foreign",
        "producer",
    ],
)
def test_existing_tokenizer_root_does_not_adopt_other_objects(
    tmp_path, monkeypatch, kind
):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = _destination(tmp_path)
    archive = _archive(
        tmp_path, owner="ui.state" if kind == "owner" else "tokenizers.custom"
    )
    if kind == "path":
        root = tmp_path / "other"
        root.mkdir(mode=0o700)
    elif kind == "occupied":
        (root / "unrelated").write_bytes(b"preserve")
    elif kind == "mode":
        root.chmod(0o755)
    elif kind in {"symlink", "dangling", "file"}:
        root.rmdir()
        if kind == "file":
            root.write_bytes(b"preserve")
        else:
            peer = tmp_path / "peer"
            if kind == "symlink":
                peer.mkdir(mode=0o700)
            root.symlink_to(peer, target_is_directory=True)
    elif kind == "foreign":
        uid = os.geteuid()
        monkeypatch.setattr(os, "geteuid", lambda: uid + 1)
    elif kind == "producer":
        from tldw_chatbook.Backup_Recovery import restore_plan
        from tldw_chatbook.Backup_Recovery.config_adapter import _Definition

        actual = restore_plan.install_adapters
        monkeypatch.setattr(
            restore_plan,
            "install_adapters",
            lambda: tuple(
                _Definition("tokenizers.custom", leaf="other", tree=True)
                if a.owner_id == "tokenizers.custom"
                else a
                for a in actual()
            ),
        )
    with pytest.raises((ValueError, OSError)):
        plan_restore(archive, mode="isolated", destinations={"root": root}, target=None)
    if kind == "occupied":
        assert (root / "unrelated").read_bytes() == b"preserve"


@pytest.mark.parametrize("change", ["child", "inode", "mode", "ancestor"])
def test_tokenizer_root_review_detects_changed_state(tmp_path, monkeypatch, change):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = _destination(tmp_path)
    plan = plan_restore(
        _archive(tmp_path), mode="isolated", destinations={"root": root}, target=None
    )
    if change == "child":
        (root / "new").write_bytes(b"keep")
    elif change == "mode":
        root.chmod(0o755)
    elif change == "inode":
        root.rename(root.with_name("original"))
        root.mkdir(mode=0o700)
    else:
        root.parent.rename(root.parent.with_name("original-parent"))
        root.mkdir(parents=True, mode=0o700)
    with pytest.raises(ValueError, match="target_changed"):
        recheck_targets(plan)


@pytest.mark.parametrize("shared", ["equal", "metadata", "group", "closure"])
def test_shared_tokenizer_root_requires_identical_declared_tree(
    tmp_path, monkeypatch, shared
):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = _destination(tmp_path)
    archive = _archive(tmp_path, shared=shared)
    if shared != "equal":
        with pytest.raises(ValueError):
            plan_restore(
                archive,
                mode="isolated",
                destinations={"root": root, "other-root": root},
                target=None,
            )
    else:
        plan = plan_restore(
            archive,
            mode="isolated",
            destinations={"root": root, "other-root": root},
            target=None,
        )
        assert dict(plan.restore)["root"] == dict(plan.restore)["other-root"] == root
        assert len(plan.metadata) == 4


_NATIVE = r"""
import json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from Tests.Backup_Recovery.test_isolated_tokenizer_container import _archive,_destination
from tldw_chatbook.Backup_Recovery import bootstrap,staging
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated,_launch_descriptor
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
base=Path.home();mode=sys.argv[1]
ambient=Path(os.environ['TLDW_CONFIG_PATH']);ambient.write_text('broken = [current config')
archive=_archive(base,config=True,shared='equal' if mode=='shared' else None)
root=_destination(base);before=root.stat()
# Bootstrap's sibling is existing user data, outside the selected root.
(root.parent/'unrelated').write_bytes(b'untouched sibling')
destination=base/'destinations';destination.mkdir(mode=0o700)
paths={'root':root,'config-root':destination/'config','profile:profile:paths.data_dir':destination/'data'}
if mode=='shared':paths['other-root']=root
plan=plan_restore(archive,mode='isolated',destinations=paths,target=None,profile_names={'profile':'Recovered'})
assert dict(plan.restore)['root']==root and any(key=='root' for key,_,_ in plan.metadata)
assert plan.target is None and root not in dict(plan.containers).values()
cancel=threading.Event()
if mode=='cancel':
 original=staging.stage_restore
 def cancel_staged(*args,**kwargs):
  result=original(*args,**kwargs);cancel.set();return result
 staging.stage_restore=cancel_staged
try:profile=restore_isolated(archive,plan,base/'control',cancel)
except InterruptedError:
 assert mode=='cancel'
 assert list(root.iterdir())==[]
 assert (root.stat().st_ino,root.stat().st_mode,root.stat().st_mtime_ns)==(before.st_ino,before.st_mode,before.st_mtime_ns)
 assert not (destination/'config').exists()
else:
 assert mode in ('success','shared')
 entry=_launch_descriptor(profile,base/'control')
 assert Path(entry.config).is_file()
 assert (root/'retained.json').read_bytes()==b'{"retained":"tokenizer document"}'
 assert root.stat().st_ino==before.st_ino
 assert (root.stat().st_mode&0o777,root.stat().st_mtime_ns)==(0o700,1234567890)
 from tldw_chatbook.Backup_Recovery.journal import Journal
 journals=list((base/'control').glob('operation-*'))
 assert len(journals)==1
 record_paths=list(journals[0].glob('*'))
 # Observe the actual durable prepared evidence, including one physical intent.
 operation=next(row for row in bootstrap._control_records(bootstrap.default_bootstrap_root())[2] if row['selector']==entry.config)['activation']['operation_id']
 journal=Journal(base/'control',operation)
 with journal._locked(exclusive=False) as parent:
  records=journal._records(parent)
 prepared=next(row.evidence for row in records if row.event=='prepared')
 intents=[row for row in prepared['directory_metadata'] if row['previous']['path']==str(root)]
 assert len(intents)==1 and intents[0]['owner_id']=='tokenizers.custom'
 assert intents[0]['previous']['inode']==before.st_ino
 assert not any(row['target']==str(root) for row in prepared['artifacts'])
 assert records[-1].event=='committed'
 from tldw_chatbook.Backup_Recovery.control_records import _existing_admission_authority
 from tldw_chatbook.Backup_Recovery.admission import AdmissionError
 authority=_existing_admission_authority(bootstrap.default_bootstrap_root())
 try:authority.register('forbidden-parent',(root.parent,))
 except AdmissionError as error:assert str(error)=='control_root_overlaps_target'
 else:raise AssertionError('Control ancestor admitted')
assert (root.parent/'unrelated').read_bytes()==b'untouched sibling'
assert ambient.read_text()=='broken = [current config'
assert not blocked_attempts(),blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("mode", ["success", "shared", "cancel"])
def test_native_tokenizer_container_publication(tmp_path, mode):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, mode, "tokenizer", script=_NATIVE)


_CRASH = (
    _NATIVE.split("try:profile=restore_isolated", 1)[0]
    + r"""
from tldw_chatbook.Backup_Recovery import publication
(base/'original-root.json').write_text(json.dumps({'device':before.st_dev,'inode':before.st_ino,'mode':before.st_mode,'mtime':before.st_mtime_ns}))
original_publish=publication.publish_new
original_metadata=publication._installed_metadata
def publish(source,target,**kwargs):
 result=original_publish(source,target,**kwargs)
 if Path(target)==root/'retained.json' and mode!='metadata':
  assert not blocked_attempts(),blocked_attempts()
  os._exit(73)
 return result
def metadata(path,*args,**kwargs):
 result=original_metadata(path,*args,**kwargs)
 if Path(path)==root and mode=='metadata':
  assert not blocked_attempts(),blocked_attempts()
  os._exit(73)
 return result
publication.publish_new=publish
publication._installed_metadata=metadata
restore_isolated(archive,plan,base/'control',cancel)
raise AssertionError('actual native crash boundary not reached')
"""
)

_FINISH = r"""
import hashlib,json,os,sys,threading
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.journal import Journal,_Prepared
from tldw_chatbook.Backup_Recovery.plan_records import load_plan
from tldw_chatbook.Backup_Recovery.isolated_restore import _finish_isolated,_launch_descriptor
base=Path.home();mode=sys.argv[1];root=base/'.config'/'tldw_cli'/'tokenizers'
before=json.loads((base/'original-root.json').read_text())
pending=bootstrap._records(bootstrap.default_bootstrap_root())[0]
assert len(pending)==1
operation=pending[0]['operation_id'];journal=Journal(base/'control',operation)
plan=load_plan(journal)
with journal._locked(exclusive=False) as parent:records=journal._records(parent)
prepared=_Prepared.model_validate(next(row.evidence for row in records if row.event=='prepared'))
assert not any(row.event=='committed' for row in records)
assert len([row for row in prepared.directory_metadata if row.previous.path==str(root)])==1
assert (root/'retained.json').read_bytes()==b'{"retained":"tokenizer document"}'
assert (root.stat().st_dev,root.stat().st_ino)==(before['device'],before['inode'])
if mode=='metadata':
 assert any(row.event=='directory_metadata_started' for row in records)
 assert not any(row.event=='directory_metadata_applied' and row.evidence['logical_id']=='root' for row in records)
else:
 assert not any(row.event=='artifact_published' and row.evidence.get('logical_id')=='file' for row in records)
if mode=='changed_child':(root/'new-user-data').write_bytes(b'preserve new data')
elif mode=='changed_inode':
 root.rename(root.with_name('original-tokenizers'))
 root.mkdir(mode=0o700)
 (root/'new-user-data').write_bytes(b'preserve replacement')
elif mode=='changed_mode':root.chmod(0o755)
def observed():
 info=root.stat()
 return (info.st_dev,info.st_ino,info.st_mode,info.st_mtime_ns,{p.name:p.read_bytes() for p in root.iterdir()})
changed=observed()
try:_finish_isolated(Path(records[0].evidence['stage']['path']),plan,journal,tuple(prepared.publication.namespaces),operation,threading.Event())
except (ValueError,OSError,RuntimeError) as error:
 assert mode.startswith('changed_'),repr(error)
 assert observed()==changed
 with journal._locked(exclusive=False) as parent:assert journal._records(parent)[-1].event!='committed'
 assert bootstrap._records(bootstrap.default_bootstrap_root())[0]
else:
 assert mode in ('lost_ack','metadata')
 entry=_launch_descriptor(prepared.isolated_profiles[0].profile_id,base/'control')
 assert Path(entry.config).is_file()
 assert root.stat().st_ino==before['inode']
 assert (root.stat().st_mode&0o777,root.stat().st_mtime_ns)==(0o700,1234567890)
 assert (root/'retained.json').read_bytes()==b'{"retained":"tokenizer document"}'
 assert not bootstrap._records(bootstrap.default_bootstrap_root())[0]
 with journal._locked(exclusive=False) as parent:assert journal._records(parent)[-1].event=='committed'
assert (root.parent/'unrelated').read_bytes()==b'untouched sibling'
assert not blocked_attempts(),blocked_attempts()
print('fresh native recovery checked')
"""


@pytest.mark.parametrize(
    "mode", ["lost_ack", "metadata", "changed_child", "changed_inode", "changed_mode"]
)
def test_fresh_finish_after_native_tokenizer_publication_death(tmp_path, mode):
    import subprocess  # nosec B404 - fixed private native crash children only
    import sys
    from pathlib import Path

    for name in ("home", "config", "data"):
        (tmp_path / name).mkdir(mode=0o700)
    environment = os.environ.copy()
    environment.update(
        HOME=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(tmp_path / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    for phase, script, expected in (("crash", _CRASH, 73), ("finish", _FINISH, 0)):
        result = subprocess.run(  # nosec B603 - fixed Python programs, private fixture paths, no shell
            [sys.executable, "-c", script, mode],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        (tmp_path / (phase + ".log")).write_text(result.stdout + result.stderr)
        assert result.returncode == expected, (
            result.stderr[-6000:] + result.stdout[-1000:]
        )
