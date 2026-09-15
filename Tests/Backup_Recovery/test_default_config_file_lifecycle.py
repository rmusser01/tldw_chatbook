"""Native default-layout replacement keeps exact config anchors and owner IO."""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_archive_reader import manifest
from tldw_chatbook.Backup_Recovery import (
    archive_reader,
    bootstrap,
    crypto,
    inventory,
    replacement,
    service_storage,
)
from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
)
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
    Inventory,
)
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore
from tldw_chatbook.runtime_policy.recovery import recovery_adapters as runtime_adapters

_OWNERS = ("ui.state", "ui.emoji_recents", "runtime.source_state")


def _payloads(label):
    return {
        "ui.state": f'[sidebar]\nsearch_query="{label}"\nlast_active_section="notes"\n[sidebar.collapsible_states]\n'.encode(),
        "ui.emoji_recents": json.dumps({"recent": [label]}).encode(),
        "runtime.source_state": json.dumps(
            {"active_source": "local", "last_known_server_label": label}
        ).encode(),
    }


@pytest.fixture(params=[False, True], ids=["absent", "present"])
def default_files(tmp_path, monkeypatch, helper_resource_root, request):
    from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
    from tldw_chatbook.Backup_Recovery import credentials
    from tldw_chatbook.runtime_policy.server_credentials import (
        KeyringServerCredentialStore,
    )

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("TLDW_DISABLE_CONFIG_WATCH", "1")
    selector = service_storage.default_config_path()
    selector.parent.mkdir(parents=True, mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    data = home / "data"
    data.mkdir(mode=0o700)
    (data / "fixture").mkdir(mode=0o700)
    (data / "fixture" / "chat_dicts").mkdir(mode=0o700)
    selector.write_text(
        f'[general]\nusers_name="fixture"\n[paths]\ndata_dir="{data.as_posix()}"\n'
    )
    selector.chmod(0o600)
    root = bootstrap.default_bootstrap_root()
    authority = admission_authority(root)
    control = service_storage.default_control_root()
    service_storage.ensure_storage(control)
    authority.register("profile", (selector, data / "fixture"))
    bind_profile(root, selector, ("profile",), root / "admission")
    store = KeyringServerCredentialStore(keyring_backend=FakeKeyring())
    monkeypatch.setattr(credentials, "_credential_store", lambda: store)
    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    adapters = {
        row.owner_id: row for row in (*recovery_adapters(), *runtime_adapters())
    }
    local_id = hashlib.sha256(str(selector).encode()).hexdigest()[:24]
    context = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, local_id)}
    paths = {owner: adapters[owner].discover(context)[0].path for owner in _OWNERS}
    if request.param:
        for owner, path in paths.items():
            path.write_bytes(_payloads("original")[owner])
            path.chmod(0o600)
    items = tuple(
        adapters[owner].discover(context)[0] for owner in ("config", *_OWNERS)
    )
    target = Inventory(
        (
            *items,
            *inventory._service_control_exclusion(),
            *inventory._fixed_control_exclusion(),
        ),
        True,
        "native-default-config",
        (),
    )
    imported_id = hashlib.sha256(
        str(tmp_path / "imported/config.toml").encode()
    ).hexdigest()[:24]
    document = manifest()
    document["profile_ids"] = [imported_id]
    document["directories"][0]["synthetic"] = True
    document["owners"] = [
        {
            "owner_id": owner,
            "schema_version": max(adapters[owner].schema_policy().versions),
            "capabilities": [],
        }
        for owner in ("config", *_OWNERS)
    ]
    payloads = {"config": selector.read_bytes(), **_payloads("incoming")}
    document["files"] = []
    document["producer_inventory"] = [
        {
            "logical_id": "root",
            "owner_id": "config",
            "status": "included_directory",
            "dependencies": [],
            "shared_group": None,
        }
    ]
    config_key = f"profile:{imported_id}:config"
    for owner, content in payloads.items():
        key = f"profile:{imported_id}:{owner}"
        document["files"].append(
            {
                "logical_id": key,
                "root_id": "root",
                "parent_id": "root",
                "relative_path": selector.name
                if owner == "config"
                else paths[owner].name,
                "owner_id": owner,
                "payload": "payload/" + owner,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        document["producer_inventory"].append(
            {
                "logical_id": key,
                "owner_id": owner,
                "status": "included",
                "dependencies": [] if owner == "config" else [config_key],
                "shared_group": None,
            }
        )
    document["dependency_groups"][0]["members"] = [
        row["logical_id"] for row in document["files"]
    ]
    source = tmp_path / "incoming.tldw-backup.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("manifest.json", json.dumps(document))
        for owner, content in payloads.items():
            archive.writestr("payload/" + owner, content)
    archive = archive_reader.acquire(
        source,
        service_storage.work_root(control) / "input",
        ArchiveLimits(),
        None,
        Event(),
    )
    checked = archive_reader.verify_sealed(archive)
    assert {row.owner_id for row in checked.files} == {"config", *_OWNERS}
    plan = plan_restore(
        archive,
        mode="replace",
        destinations={
            "root": selector.parent,
            f"profile:{imported_id}:paths.data_dir": data,
        },
        target=target,
        profile_names={imported_id: "fixture"},
    )
    candidate = stage_restore(
        archive, plan, service_storage.work_root(control) / "candidate", Event()
    )
    assert root.parent == selector.parent and selector.parent in control.parents
    before = bootstrap._registry(root)
    bindings = bootstrap._records(root)[1]
    assert not any(str(selector.parent) in entry["roots"] for entry in before.values())
    projected = replacement._register_publication_parents(
        plan,
        authority,
        (root, control, candidate),
        document=checked,
    )
    Journal(control, "fixture-candidate").record_candidate(candidate, plan, archive)
    assert projected == {path: selector for path in paths.values()}
    assert (
        bootstrap._registry(root) == before and bootstrap._records(root)[1] == bindings
    )
    yield {
        "home": home,
        "root": root,
        "control": control,
        "selector": selector,
        "paths": paths,
        "plan": plan,
        "candidate": candidate,
        "registry": before,
        "bindings": bindings,
        "original": {
            path: path.read_bytes() if path.exists() else None
            for path in paths.values()
        },
        "helper": helper_resource_root,
        "parent_identity": (
            selector.parent.stat().st_dev,
            selector.parent.stat().st_ino,
            selector.parent.stat().st_mode,
        ),
        "markers": {
            path: (path.stat().st_ino, path.read_bytes())
            for path in (control / "service.json", root / "unbound-owner")
        },
    }


_OWNER_REOPEN = r"""
import asyncio,json,os
from pathlib import Path
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import bootstrap,raw_participants as raw
from tldw_chatbook.Backup_Recovery.async_file_participants import _FileJob
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets import emoji_picker
from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore,RuntimeSourceState
selector=Path(os.environ['TLDW_CONFIG_PATH']);root=bootstrap.default_bootstrap_root()
before=bootstrap._records(root)[1];registry=bootstrap._registry(root)
assert bootstrap.startup_permission(selector,root)[0]
emoji_picker.save_recent_emoji('ordinary-reopen')
assert emoji_picker.load_recent_emojis()[0]=='ordinary-reopen'
store=RuntimeSourceStateStore(selector.parent/'runtime_policy.json',application_owned_directory=selector.parent)
store.save(RuntimeSourceState(last_known_server_label='ordinary-reopen'))
assert store.load().last_known_server_label=='ordinary-reopen'
screen=object.__new__(ChatScreen)
async def save():
 with _FileJob(screen,'sidebar_state') as job:
  outcome=await job.run({'collapsible_states':{},'search_query':'ordinary-reopen','last_active_section':'notes'})
  if outcome.error:raise outcome.error
asyncio.run(save())
import toml
with raw._scope(screen,'sidebar_state') as operation:
 with raw._file(operation,raw._selected(operation),'r') as stream:assert toml.load(stream)['sidebar']['search_query']=='ordinary-reopen'
assert bootstrap._records(root)[1]==before and bootstrap._registry(root)==registry
print('ACTUAL_DEFAULT_OWNER_REOPEN_COMPLETE')
"""


def _reopen(case):
    parent = case["selector"].parent
    assert (parent.stat().st_dev, parent.stat().st_ino, parent.stat().st_mode) == case[
        "parent_identity"
    ]
    assert {
        path: (path.stat().st_ino, path.read_bytes()) for path in case["markers"]
    } == case["markers"]
    assert bootstrap._registry(case["root"]) == case["registry"]
    profile = bootstrap._records(case["root"])[1][0]
    assert profile["namespaces"] == case["bindings"][0]["namespaces"]
    assert profile["roots"] == case["bindings"][0]["roots"]
    environment = dict(
        os.environ,
        HOME=str(case["home"]),
        USERPROFILE=str(case["home"]),
        TLDW_CONFIG_PATH=str(case["selector"]),
    )
    log = case["home"] / "ordinary-owner-reopen.log"
    with log.open("w") as output:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", _OWNER_REOPEN],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=output,
            stderr=output,
            text=True,
            timeout=35,
            check=False,
        )
    assert result.returncode == 0, log.read_text()[-8000:]
    assert "ACTUAL_DEFAULT_OWNER_REOPEN_COMPLETE" in log.read_text()


def test_default_config_files_publish_then_ordinary_owners_reopen(default_files):
    case = default_files
    operation = replacement.replace(
        case["plan"],
        case["candidate"],
        control_root=case["control"],
        rollback_password=b"fixture-safety",
        cancel=Event(),
    )
    assert bootstrap._registry(case["root"]) == case["registry"]
    current = bootstrap._records(case["root"])[1]
    assert current[0]["namespaces"] == case["bindings"][0]["namespaces"]
    assert current[0]["roots"] == case["bindings"][0]["roots"]
    for owner, path in case["paths"].items():
        if owner == "ui.state":
            import toml

            assert toml.loads(path.read_text())["sidebar"]["search_query"] == "incoming"
        elif owner == "ui.emoji_recents":
            assert json.loads(path.read_text())["recent"] == ["incoming"]
        else:
            assert json.loads(path.read_text())["last_known_server_label"] == "incoming"
    journal = Journal(case["control"], operation)
    with journal._locked(exclusive=False) as fd:
        assert journal._records(fd)[-1].event == "committed"
    _reopen(case)


def test_default_config_files_untouched_abort_then_ordinary_owners_reopen(
    default_files, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import control_records

    case = default_files
    register = control_records.register_pending

    def stop_after_durable_pending(*args, **kwargs):
        register(*args, **kwargs)
        raise InterruptedError("fixture after durable pending")

    monkeypatch.setattr(control_records, "register_pending", stop_after_durable_pending)
    with pytest.raises(InterruptedError, match="fixture after durable pending"):
        replacement.replace(
            case["plan"],
            case["candidate"],
            control_root=case["control"],
            rollback_password=b"fixture-safety",
            cancel=Event(),
        )
    monkeypatch.setattr(control_records, "register_pending", register)
    pending = bootstrap._records(case["root"])[0]
    assert len(pending) == 1
    assert (
        replacement.recover_replacement(
            pending[0]["operation_id"],
            control_root=case["control"],
            action="abort",
            rollback_password=None,
            cancel=Event(),
        )
        == "aborted"
    )
    assert not bootstrap._records(case["root"])[0]
    for path, data in case["original"].items():
        assert (path.read_bytes() if path.exists() else None) == data
    assert bootstrap._registry(case["root"]) == case["registry"]
    assert bootstrap._records(case["root"])[1] == case["bindings"]
    _reopen(case)


_CRASH_CHILD = r"""
import os,sys
from pathlib import Path
from threading import Event
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Backup_Recovery import bootstrap,crypto,replacement,credentials,control_records
from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
store=KeyringServerCredentialStore(keyring_backend=FakeKeyring())
credentials._credential_store=lambda:store
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.plan_records import load_plan
control=Path(sys.argv[1]);candidate=Path(sys.argv[2]);helper=Path(sys.argv[3]);mode=sys.argv[4];boundary=sys.argv[5]
crypto._package_resource_root=lambda:helper
if mode=='crash':
 journal=Journal(control,'fixture-candidate')
 plan=load_plan(journal)
 append=Journal._append
 def interrupted(self,parent,event,evidence):
  append(self,parent,event,evidence)
  if boundary=='prepared' and event=='rollback_verified':os._exit(91)
  if boundary=='published' and event=='artifact_published' and evidence['logical_id'].endswith(':ui.state'):os._exit(91)
 Journal._append=interrupted
 if boundary.startswith('activation'):
  native_replace=control_records.os.replace
  def partial_pair(source,target,*args,**kwargs):
   if str(target).startswith('profile-'):
    import json
    root=bootstrap.default_bootstrap_root()
    intents=list(root.glob('activation-update-*.json'))
    assert len(intents)==1
    intent=json.loads(intents[0].read_text())
    profile=root/str(target)
    association=root/str(target).replace('profile-','activation-',1)
    assert json.loads(profile.read_text())==intent['before'][0]
    assert json.loads(association.read_text())==intent['after'][1]
    assert intent['before'][1]!=intent['after'][1]
    if boundary=='activation_staged':os._exit(91)
    native_replace(source,target,*args,**kwargs)
    assert json.loads(profile.read_text())==intent['after'][0]
    os._exit(91)
   return native_replace(source,target,*args,**kwargs)
  control_records.os.replace=partial_pair
 replacement.replace(plan,candidate,control_root=control,rollback_password=b'fixture-safety',cancel=Event())
 raise AssertionError('native crash checkpoint was not reached')
else:
 if boundary.startswith('activation'):
  import json
  root=bootstrap.default_bootstrap_root()
  if boundary=='activation_staged':
   native_listdir=bootstrap.os.listdir
   # Preserve every real entry, but force the crash-stage entry before intent.
   bootstrap.os.listdir=lambda parent:sorted(native_listdir(parent),key=lambda name:(not name.startswith('activation-stage-'),name))
  try:bootstrap._records(root)
  except ValueError as error:assert str(error)==('unknown_activation_record' if boundary=='activation_staged' else 'activation_update_pending')
  else:raise AssertionError('ordinary bootstrap accepted partial activation pair')
  pending=[json.loads(path.read_text()) for path in root.glob('pending-*.json')]
 else:pending=bootstrap._records(bootstrap.default_bootstrap_root())[0]
 assert len(pending)==1
 result=replacement.recover_replacement(pending[0]['operation_id'],control_root=control,action=mode,rollback_password=b'fixture-safety',cancel=Event())
 print('ACTUAL_DEFAULT_RECOVERY_COMPLETE',result)
"""


def _crash_process(case, mode, expected, boundary):
    environment = dict(
        os.environ,
        HOME=str(case["home"]),
        USERPROFILE=str(case["home"]),
        TLDW_CONFIG_PATH=str(case["selector"]),
    )
    log = case["home"] / ("native-" + mode + ".log")
    with log.open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "faulthandler",
                "-c",
                _CRASH_CHILD,
                str(case["control"]),
                str(case["candidate"]),
                str(case["helper"]),
                mode,
                boundary,
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=output,
            stderr=output,
            text=True,
            timeout=35,
            check=False,
        )
    assert result.returncode == expected, log.read_text()[-8000:]
    if mode != "crash" and expected == 0:
        assert "ACTUAL_DEFAULT_RECOVERY_COMPLETE" in log.read_text()
    return log.read_text()


@pytest.mark.parametrize("action", ["finish", "rollback"])
@pytest.mark.parametrize(
    "boundary", ["prepared", "published", "activation", "activation_staged"]
)
def test_default_config_files_fresh_process_recovery_then_owner_reopen(
    default_files, action, boundary
):
    case = default_files
    _crash_process(case, "crash", 91, boundary)
    if boundary.startswith("activation"):
        with pytest.raises(
            ValueError, match="^(activation_update_pending|unknown_activation_record)$"
        ):
            bootstrap._records(case["root"])
        pending = [
            json.loads(path.read_text()) for path in case["root"].glob("pending-*.json")
        ]
    else:
        pending = bootstrap._records(case["root"])[0]
    assert len(pending) == 1
    assert not bootstrap.startup_permission(case["selector"], case["root"])[0]
    _crash_process(case, action, 0, boundary)
    assert not bootstrap._records(case["root"])[0]
    assert bootstrap._registry(case["root"]) == case["registry"]
    if action == "rollback":
        for path, data in case["original"].items():
            assert (path.read_bytes() if path.exists() else None) == data
    _reopen(case)


def test_default_config_files_foreign_activation_intent_refuses_without_moves(
    default_files,
):
    case = default_files
    _crash_process(case, "crash", 91, "activation_staged")
    intents = tuple(case["root"].glob("activation-update-*.json"))
    assert len(intents) == 1
    intent = json.loads(intents[0].read_text())
    intent["operation_id"] = "foreign-operation"
    intents[0].write_text(json.dumps(intent))
    paths = (
        case["selector"],
        *case["paths"].values(),
        *(path for path in case["root"].rglob("*") if path.is_file()),
    )
    before = {path: (path.stat().st_ino, path.read_bytes()) for path in paths}
    output = _crash_process(case, "finish", 1, "activation_staged")
    assert "activation_recovery_context_invalid" in output
    assert {path: (path.stat().st_ino, path.read_bytes()) for path in paths} == before
    assert tuple(case["root"].glob("activation-update-*.json")) == intents
    assert not bootstrap.startup_permission(case["selector"], case["root"])[0]
