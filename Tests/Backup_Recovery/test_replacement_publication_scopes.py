"""Actual replacement publishes mapped owners under exact held source scopes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.Backup_Recovery.conftest import (
    helper_resource_root as helper_resource_root,  # noqa: PLC0414 - native helper fixture
)
from Tests.Backup_Recovery.native_package import (
    native_package as native_package,  # noqa: PLC0414 - installed product fixture
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_recovery_service import _LIVE_BACKUP

_REPLACE = r"""
import builtins,json,os,sys,tomllib
from pathlib import Path
from threading import Event
from Tests.network_guard import install,blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
import tldw_chatbook
installed=Path(os.environ['TLDW_TEST_INSTALLED_PACKAGE'])
assert Path(tldw_chatbook.__file__).resolve()==installed/'tldw_chatbook'/'__init__.py'
from tldw_chatbook.Backup_Recovery import archive_reader,launcher
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService,default_control_root
home=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH'])
manual=home/'selected-manual-parent';manual.mkdir(mode=0o700)
from tldw_chatbook.Backup_Recovery import bootstrap
assert not bootstrap._records(bootstrap.default_bootstrap_root())[1]
service=RecoveryService(default_control_root())
inspection=service.start_inspection(home/'service.tldw-backup.zip',password=None)
assert service.wait(inspection)['state']=='succeeded'
archive=service.inspection(inspection);doc=archive_reader.verify_sealed(archive)
target=service.preview_backup((selector,),options={})
assert target.complete,target.issues
actual={item.logical_id:item for item in target.items}
source_profile=doc.profile_ids[0]
current_config=next(item for item in target.items if item.owner=='config' and item.path==selector)
current_profile=current_config.logical_id.split(':')[1]
def current_key(key):
 prefix='profile:'+source_profile+':'
 return 'profile:'+current_profile+':'+key[len(prefix):] if key.startswith(prefix) else key
roots={row.logical_id:row for row in doc.directories if row.parent_id is None}
mapping={};deferred={'agents.history','eval.definitions','tts.voices','persona.visual_identity_builtin'}
for index,(key,root) in enumerate(roots.items()):
 members=[row for row in doc.files if row.root_id==key]
 owners={row.owner_id for row in members}
 if owners & deferred:
  mapping[key]=manual/('reviewed-inactive-'+str(index));continue
 if current_key(key) in actual and actual[current_key(key)].path is not None:
  mapping[key]=actual[current_key(key)].path;continue
 choices=set()
 for row in members:
  path=actual[current_key(row.logical_id)].path
  for part in Path(row.relative_path).parts:path=path.parent
  choices.add(path)
 assert len(choices)==1,(key,choices)
 mapping[key]=choices.pop()
profile=doc.profile_ids[0]
config=tomllib.loads(selector.read_text())
name=config.get('general',{}).get('users_name','default_user')
from tldw_chatbook.Backup_Recovery.profile_paths import data_base
mapping[f'profile:{profile}:paths.data_dir']=data_base(config)
print('TARGET_MAPPING',len(mapping),flush=True)
service.close()
arguments=['restore',str(home/'service.tldw-backup.zip'),'--replace','--target-config',str(selector),'--profile-name',profile+'='+name]
for key,path in mapping.items():arguments += ['--destination',key+'='+str(path)]
for item in target.items:
 if item.owner=='persona.visual_identity_builtin' and item.status in {'included','included_directory'}:
  arguments += ['--safety-scope',item.logical_id]
builtins.input=lambda prompt:'restore'
launcher.getpass.getpass=lambda prompt:'test-only-new-safety-password'
displayed=[]
show=launcher._show
def observed_show(value):
 displayed.append(value)
 show(value)
launcher._show=observed_show
originals={item.path:item.path.read_bytes() for item in target.items if item.path is not None and item.path.is_file()}
code=launcher.recovery_main(arguments)
if code:
 state=displayed[-1]
 issues=state.get('review_issues',())
 assert state['state']=='recovery_required' and issues,state
 assert all(path.read_bytes()==value for path,value in originals.items())
 service=RecoveryService(default_control_root())
 pending=service.pending_operations()
 assert len(pending)==1
 aborted=service.start_recovery(pending[0]['operation_id'],action='abort')
 assert service.wait(aborted)['state']=='succeeded'
 assert not service.pending_operations()
 service.close()
 assert all(path.read_bytes()==value for path,value in originals.items())
 for issue in issues:arguments += ['--acknowledge-credential-issue',issue]
 print('ACKNOWLEDGED_ACTUAL_ISSUES',len(issues),flush=True)
 code=launcher.recovery_main(arguments)
assert code==0,code
service=RecoveryService(default_control_root())
try:
 bindings=bootstrap._records(bootstrap.default_bootstrap_root())[1]
 selected=next(row for row in bindings if row['selector']==str(selector))
 assert selected['activation']['generation'] and selected['namespaces']
 assert bootstrap.startup_permission(selector,bootstrap.default_bootstrap_root())[0]
 from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage
 runtime=selector.parent/'runtime_policy.json'
 with acquire_storage(runtime) as lease:
  scope_root,names=lease.execution_context(runtime)
  assert scope_root==bootstrap.default_bootstrap_root() and set(names)<=set(selected['namespaces'])
 copies=service.recovery_copies()
 assert len(copies)==1 and copies[0].status=='verified',copies
finally:service.close()
assert not blocked_attempts()
assert 'tldw_chatbook.app' not in sys.modules
print('actual CLI replacement completed')
"""


@pytest.mark.parametrize("different_selector", [False, True])
def test_complete_cli_replacement_admits_mapped_sources(
    tmp_path, native_package, different_selector
):
    _run(
        tmp_path,
        "service",
        "backup",
        script=_LIVE_BACKUP,
        timeout=110,
        installed_package=native_package,
    )
    selector = tmp_path / "config" / "config.toml"
    if different_selector:
        current = tmp_path / "current-config" / "config.toml"
        current.parent.mkdir(mode=0o700)
        current.write_bytes(selector.read_bytes())
        current.chmod(0o600)
        selector = current
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(selector),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join(
            (str(native_package), str(Path(__file__).resolve().parents[2]))
        ),
        TLDW_TEST_INSTALLED_PACKAGE=str(native_package),
    )
    output = tmp_path / "first-replacement-child.log"
    with output.open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-c", _REPLACE],
            cwd=tmp_path,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
            check=False,
        )
    observed = output.read_text()
    assert result.returncode == 0, observed[-7000:]
    assert "actual CLI replacement completed" in observed


def test_preparation_keeps_one_physical_wal_shm_pair_for_declared_aliases(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery.journal import _Prepared
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    def aliases(live):
        return tuple(
            StorageItem(
                "sqlite.transient",
                "same-main-alias" + suffix,
                Path(str(live / "research.db") + suffix),
                "intentionally_excluded",
                ("profile:profile:research.local",),
            )
            for suffix in ("-wal", "-shm")
        )

    with replacement_case(tmp_path, monkeypatch, extras=aliases) as case:
        _, _, journal, _, source, _ = case
        with journal._locked(exclusive=False) as parent:
            records = journal._records(parent)
        prepared = _Prepared.model_validate(
            next(row.evidence for row in records if row.event == "prepared")
        )
        group = next(
            row for row in prepared.rollback_sources if row.source.path == str(source)
        )
        assert {Path(row.path) for row in group.sidecars.values()} == {
            Path(str(source) + "-wal"),
            Path(str(source) + "-shm"),
        }
        assert len(prepared.artifacts) == len(
            {row.target for row in prepared.artifacts}
        )


@pytest.mark.parametrize("boundary", ["complete", "missing_dependency", "split_pair"])
def test_actual_mapped_publication_adds_only_its_receipt_bound_selector_scope(
    tmp_path, monkeypatch, helper_resource_root, boundary
):
    import hashlib
    import json
    import zipfile
    from threading import Event

    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery import (
        archive_reader,
        bootstrap,
        control_records,
        crypto,
        replacement,
    )
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
    from tldw_chatbook.Backup_Recovery.staging import stage_restore
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)

    def data_directory(live):
        (live / "data").mkdir(mode=0o700)
        return ()

    with replacement_case(
        tmp_path, monkeypatch, prepared=False, extras=data_directory
    ) as case:
        _, original_plan, _, _, _, selector = case
        with zipfile.ZipFile(tmp_path / "replacement.zip") as packed:
            payloads = {name: packed.read(name) for name in packed.namelist()}
        doc = json.loads(payloads.pop("manifest.json"))
        root_key = "profile:profile:external-root"
        file_key = "profile:profile:external.files"
        doc["directories"].append(
            {
                "logical_id": root_key,
                "root_id": root_key,
                "parent_id": None,
                "relative_path": "",
                "synthetic": True,
                "metadata": {"version": 1, "mode": 0o700, "mtime_ns": 0},
            }
        )
        data = b"[state]\nselected=17\n"
        payloads["payload/state"] = data
        doc["files"].append(
            {
                "logical_id": file_key,
                "root_id": root_key,
                "parent_id": root_key,
                "relative_path": "state.toml",
                "owner_id": "external.files",
                "payload": "payload/state",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "metadata": {"version": 1, "mode": 0o600, "mtime_ns": 0},
            }
        )
        doc["owners"].append(
            {"owner_id": "external.files", "schema_version": 1, "capabilities": []}
        )
        for key, status in ((root_key, "included_directory"), (file_key, "included")):
            doc["producer_inventory"].append(
                {
                    "logical_id": key,
                    "owner_id": "external.files",
                    "status": status,
                    "dependencies": []
                    if key == root_key or boundary == "missing_dependency"
                    else ["profile:profile:config"],
                    "shared_group": None,
                }
            )
        doc["dependency_groups"][0]["members"].append(file_key)
        incoming = tmp_path / "mapped.zip"
        with zipfile.ZipFile(incoming, "w") as packed:
            packed.writestr("manifest.json", json.dumps(doc))
            for name, value in payloads.items():
                packed.writestr(name, value)
        parent = tmp_path / "selected-private-parent"
        parent.mkdir(mode=0o700)
        archive = archive_reader.acquire(
            incoming, tmp_path / "mapped-input", ArchiveLimits(), None, Event()
        )
        plan = plan_restore(
            archive,
            mode="replace",
            target=original_plan.target,
            destinations={
                **dict(original_plan.destinations),
                root_key: parent / "new-state",
                **dict(original_plan.selectors),
            },
            profile_names={"profile": "Local"},
            acknowledged_credential_issues=("credential_format_unreadable",),
        )
        candidate = stage_restore(archive, plan, tmp_path / "mapped-stage", Event())
        root = bootstrap.default_bootstrap_root()
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir(mode=0o700)
        other = unrelated / "config.toml"
        other.write_text('[general]\nusers_name="other"\n')
        other.chmod(0o600)
        authority = admission_authority(root)
        authority.register("unrelated", (unrelated,))
        bind_profile(root, other, ("unrelated",), root / "admission")
        before = next(
            row for row in bootstrap._records(root)[1] if row["selector"] == str(other)
        )
        if boundary == "missing_dependency":
            with pytest.raises(ValueError, match="replacement_rolled_back"):
                replacement.replace(
                    plan,
                    candidate,
                    control_root=tmp_path / "control",
                    rollback_password=b"test-only",
                    cancel=Event(),
                )
            assert not (parent / "new-state").exists()
            assert (
                next(
                    row
                    for row in bootstrap._records(root)[1]
                    if row["selector"] == str(other)
                )
                == before
            )
            return
        if boundary == "split_pair":
            actual_publish = control_records._publish_activation_record

            def interrupted(*args, **kwargs):
                result = actual_publish(*args, **kwargs)
                if args[2].startswith("activation-"):
                    raise KeyboardInterrupt("actual association published")
                return result

            with monkeypatch.context() as fault:
                fault.setattr(
                    control_records, "_publish_activation_record", interrupted
                )
                with pytest.raises(KeyboardInterrupt):
                    replacement.replace(
                        plan,
                        candidate,
                        control_root=tmp_path / "control",
                        rollback_password=b"test-only",
                        cancel=Event(),
                    )
            from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

            service = RecoveryService(tmp_path / "control")
            try:
                operation = service.pending_operations()[0]["operation_id"]
            finally:
                service.close()
            script = """
import sys,keyring
from pathlib import Path
from threading import Event
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import bootstrap,crypto,replacement
bootstrap.default_bootstrap_root=lambda:Path(sys.argv[1])
crypto._package_resource_root=lambda:Path(sys.argv[4])
assert replacement.recover_replacement(sys.argv[3],control_root=Path(sys.argv[2]),action="finish",rollback_password=b"test-only",cancel=Event())=="committed"
"""
            output = tmp_path / "mapped-recovery-child.log"
            with output.open("w") as stream:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        script,
                        str(root),
                        str(tmp_path / "control"),
                        operation,
                        str(helper_resource_root),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=45,
                    check=False,
                )
            assert result.returncode == 0, output.read_text()[-5000:]
        else:
            operation = replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"test-only",
                cancel=Event(),
            )
        pending, profiles = bootstrap._records(root)
        selected = next(row for row in profiles if row["selector"] == str(selector))
        assert not pending and selected["activation"]["operation_id"] == operation
        assert str(parent) in selected["roots"]
        assert next(row for row in profiles if row["selector"] == str(other)) == before
        restored = parent / "new-state" / "state.toml"
        assert restored.read_bytes() == data
        lease = acquire_storage(restored)
        try:
            scope_root, names = lease.execution_context(restored)
            assert scope_root == root
            assert set(names) <= set(selected["namespaces"])
            assert any(name.startswith("replacement.destination.") for name in names)
        finally:
            lease.close()


@pytest.mark.parametrize(
    "boundary",
    [
        "complete",
        "covered",
        "missing_dependency",
        "other_selector",
        "unrelated_registration",
    ],
)
def test_existing_config_container_requires_exact_selected_closure(
    tmp_path, monkeypatch, boundary
):
    import json
    import zipfile
    from dataclasses import replace
    from threading import Event

    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery import archive_reader, bootstrap, replacement
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
    from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore

    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        _, original, _, _, _, _ = case
        folder = tmp_path / "current-config"
        folder.mkdir(mode=0o700)
        selector = folder / "config.toml"
        selector.write_text('[general]\nusers_name="current"\n')
        selector.chmod(0o600)
        root = bootstrap.default_bootstrap_root()
        authority = admission_authority(root)
        authority.register("current-config", (selector,))
        names = ("current-config",)
        if boundary == "covered":
            research = folder / "research.db"
            research.write_bytes(case[4].read_bytes())
            research.chmod(0o600)
            authority.register("current-research", (research,))
            names += ("current-research",)
        bind_profile(root, selector, names, root / "admission")
        if boundary in {"other_selector", "unrelated_registration"}:
            other = folder / "other.toml"
            other.write_text('[general]\nusers_name="other"\n')
            other.chmod(0o600)
            authority.register("other", (other,))
            if boundary == "other_selector":
                bind_profile(root, other, ("other",), root / "admission")
        with zipfile.ZipFile(tmp_path / "replacement.zip") as packed:
            payloads = {name: packed.read(name) for name in packed.namelist()}
        document = json.loads(payloads.pop("manifest.json"))
        for row in document["producer_inventory"]:
            if row["owner_id"] == "research.local" and boundary != "missing_dependency":
                row["dependencies"] = ["profile:profile:config"]
        incoming = tmp_path / "container.zip"
        with zipfile.ZipFile(incoming, "w") as packed:
            packed.writestr("manifest.json", json.dumps(document))
            for name, data in payloads.items():
                packed.writestr(name, data)
        archive = archive_reader.acquire(
            incoming, tmp_path / "container-input", ArchiveLimits(), None, Event()
        )
        target = replace(
            original.target,
            items=tuple(
                replace(
                    item,
                    logical_id=item.logical_id.replace(
                        "profile:profile:", "profile:current:"
                    ),
                    path=(
                        selector
                        if item.owner == "config"
                        else folder / item.path.name
                        if boundary == "covered"
                        and item.path is not None
                        and item.path
                        in {
                            case[4],
                            case[4].with_name(case[4].name + "-wal"),
                            case[4].with_name(case[4].name + "-shm"),
                        }
                        else item.path
                    ),
                    dependencies=tuple(
                        key.replace("profile:profile:", "profile:current:")
                        for key in item.dependencies
                    ),
                )
                for item in original.target.items
            ),
        )
        plan = plan_restore(
            archive,
            mode="replace",
            target=target,
            destinations={
                "root": folder,
                **{
                    key: path
                    for key, path in original.selectors
                    if key.endswith(":paths.data_dir")
                },
            },
            profile_names=dict(original.profile_names),
        )
        before = selector.read_bytes()
        selected_document = archive_reader.verify_sealed(archive)
        if boundary in {"complete", "covered"}:
            replacement._register_publication_parents(
                plan,
                authority,
                (root, tmp_path / "control"),
                document=selected_document,
            )
            registry = bootstrap._registry(root)
            assert any(row["roots"] == [str(folder)] for row in registry.values())
        else:
            with pytest.raises(
                ValueError, match="replacement_destination_parent_required"
            ):
                replacement._register_publication_parents(
                    plan,
                    authority,
                    (root, tmp_path / "control"),
                    document=selected_document,
                )
            assert not any(
                row["roots"] == [str(folder)]
                for row in bootstrap._registry(root).values()
            )
        assert selector.read_bytes() == before
        assert (folder / "research.db").exists() == (boundary == "covered")
