"""Reviewed capture preserves ordinary installed config sibling ownership."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SEED = r"""
import os,sys
from pathlib import Path
from Tests.network_guard import install
install()
for name in ('sounddevice','pyaudio'):sys.modules[name]=None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
selector=Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\nusers_name="default_user"\n[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n')
selector.chmod(0o600)
from tldw_chatbook.app import TldwCli
app=TldwCli()
app.chachanotes_db.add_note('Capture sibling fixture','Native seeded content.')
print('NATIVE_SEED_COMPLETE',flush=True)
"""

_CAPTURE = r"""
import asyncio,json,os,sys,threading,zipfile
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery import bootstrap,storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,bind_profile,UNBOUND_NAMESPACE
from tldw_chatbook.Backup_Recovery.capture_service import preview_capture,capture,_capture_names
from tldw_chatbook.Backup_Recovery.archive_writer import write_archive
from tldw_chatbook.Backup_Recovery.archive_reader import acquire,verify_sealed
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook import config
from tldw_chatbook.Widgets import emoji_picker
from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore,RuntimeSourceState
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Backup_Recovery.async_file_participants import _FileJob
from tldw_chatbook.Backup_Recovery import raw_participants as raw
home=Path.home();selector=Path(os.environ['TLDW_CONFIG_PATH']);parent=selector.parent
owner=sys.argv[1]
for leaf in ('ui_state.toml','recent_emojis.json','runtime_policy.json'):
 (parent/leaf).unlink(missing_ok=True)
root=bootstrap.default_bootstrap_root();authority=admission_authority(root)
options={'staging_parent':home}
preview=preview_capture((selector,),options=options)
assert preview.complete,(preview.issues,[(i.owner,i.status) for i in preview.items if i.status in {'unsupported','unavailable','missing_required'}])
storage._shutdown()
assert not storage._live_leases and not storage._raw_operations and not storage._operations
# Pre-enroll existing native sources so only the new sibling is under test.
# The config parent (which contains recovery controls) is never a root.
authority.register('selected',(selector,config.get_user_data_dir()))
names=set(_capture_names(authority,preview))-{UNBOUND_NAMESPACE}
bind_profile(root,selector,tuple(sorted(names)),root/'admission')
assert str(parent) not in bootstrap._records(root)[1][0]['roots']
registry_before=bootstrap._registry(root);bindings_before=bootstrap._records(root)[1]
if owner=='emoji':
 path=parent/'recent_emojis.json';owner_id='ui.emoji_recents'
 def write(value):emoji_picker.save_recent_emoji(value)
 def read():
  values=emoji_picker.load_recent_emojis()
  return values[0] if values else None
elif owner=='runtime':
 path=parent/'runtime_policy.json';owner_id='runtime.source_state'
 store=RuntimeSourceStateStore(path,application_owned_directory=parent)
 def write(value):
  state=RuntimeSourceState(last_known_server_label=value)
  store.save(state)
 def read():return store.load().last_known_server_label
else:
 path=parent/'ui_state.toml';owner_id='ui.state';screen=object.__new__(ChatScreen)
 def write(value):
  async def save():
   with _FileJob(screen,'sidebar_state') as job:
    outcome=await job.run({'collapsible_states':{},'search_query':value,'last_active_section':'notes'})
    if outcome.error:raise outcome.error
  asyncio.run(save())
 def read():
  import toml
  with raw._scope(screen,'sidebar_state') as operation:
   with raw._file(operation,raw._selected(operation),'r') as stream:return toml.load(stream)['sidebar']['search_query']
write('before-capture');first=read()
assert path.is_file() and bootstrap._registry(root)==registry_before
storage._shutdown()
assert not storage._live_leases and not storage._raw_operations and not storage._operations
preview=preview_capture((selector,),options=options)
assert preview.complete,preview.issues
assert any(i.owner==owner_id and i.status=='included' for i in preview.items)
destination=home/'sibling.tldw-backup.zip'
result=capture((selector,),preview.scope_digest,destination,options=options,cancel=threading.Event())
assert result.inventory.complete
write_archive(result,destination,password=None,cancel=threading.Event())
archive=acquire(destination,home/'readback',ArchiveLimits(),None,threading.Event())
document=verify_sealed(archive)
assert document.consistency=='coherent'
member=next(row for row in document.files if row.owner_id==owner_id)
with zipfile.ZipFile(archive.path) as saved:
 payload=saved.read(member.payload)
 if owner=='runtime':assert json.loads(payload)==json.loads(path.read_bytes())
 else:assert payload==path.read_bytes()
print('ACTUAL_COMPLETE_CAPTURE',owner,flush=True)
# The ordinary second source operation must remain authorized after capture.
write('after-capture');second=read()
assert first=='before-capture' and second=='after-capture'
assert bootstrap._registry(root)==registry_before
assert bootstrap._records(root)[1]==bindings_before
assert not blocked_attempts()
print('CAPTURE_SIBLING_CONTINUITY_COMPLETE',owner,flush=True)
"""


def _child(home, script, *arguments, timeout=70, selector=None):
    repository = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ,
        HOME=str(home),
        USERPROFILE=str(home),
        TLDW_CONFIG_PATH=str(selector or home / ".config" / "tldw_cli" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHONPATH=str(repository),
    )
    log = home / ("capture-" + ("-".join(arguments) or "seed") + ".log")
    with log.open("w") as output:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", script, *arguments],
            cwd=repository,
            env=environment,
            stdout=output,
            stderr=output,
            text=True,
            timeout=timeout,
            check=False,
        )
    assert result.returncode == 0, log.read_text()[-10000:]
    return log.read_text()


@pytest.fixture
def native_profile(tmp_path, request):
    home = tmp_path / "home"
    parent = home / (
        "custom-config"
        if getattr(request, "param", None) == "custom"
        else ".config/tldw_cli"
    )
    parent.mkdir(parents=True, mode=0o700)
    selector = parent / "config.toml"
    _child(home, _SEED, selector=selector)
    return home, selector


@pytest.mark.parametrize("owner", ["emoji", "runtime", "sidebar"])
def test_complete_capture_keeps_installed_sibling_writable(native_profile, owner):
    output = _child(native_profile[0], _CAPTURE, owner, selector=native_profile[1])
    assert "CAPTURE_SIBLING_CONTINUITY_COMPLETE" in output


_OTHER_WRITER = r"""
import os,sys
from pathlib import Path
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.runtime_policy import source_state
storage._shutdown()
original=source_state.atomic_private_write_text
def paused(*args,**kwargs):
 print('OTHER_WRITER_NATIVE_SCOPE',flush=True)
 assert sys.stdin.readline().strip()=='release'
 return original(*args,**kwargs)
source_state.atomic_private_write_text=paused
path=Path(os.environ['TLDW_CONFIG_PATH']).parent/'runtime_policy.json'
source_state.RuntimeSourceStateStore(path,application_owned_directory=path.parent).save(source_state.RuntimeSourceState())
print('OTHER_WRITER_RETIRED',flush=True)
"""

_SHARED_CAPTURE = (
    _CAPTURE.split("registry_before=")[0]
    + r"""
import subprocess
from contextlib import contextmanager
from tldw_chatbook.Backup_Recovery.admission import Admission,fcntl
other=parent/'other.toml'
import toml
other_data=home/'other-data';other_data.mkdir(mode=0o700)
(other_data/'default_user').mkdir(mode=0o700)
other_settings=toml.loads(selector.read_text());other_settings.setdefault('paths',{})['data_dir']=str(other_data)
other.write_text(toml.dumps(other_settings));other.chmod(0o600)
authority.register('other-selector',(other,other_data))
bind_profile(root,other,('other-selector',),root/'admission')
registry_before=bootstrap._registry(root);bindings_before=bootstrap._records(root)[1]
preview=preview_capture((selector,),options=options)
assert preview.complete,preview.issues
names=_capture_names(authority,preview)
assert 'other-selector' in names,'same-parent native writer namespace was not fenced'
assert bootstrap._registry(root)==registry_before
writer_env=dict(os.environ,TLDW_CONFIG_PATH=str(other))
with (home/'other-writer.log').open('w') as errors:
 writer=subprocess.Popen([sys.executable,'-u','-c',WRITER],env=writer_env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True)
 attempted=threading.Event();finished=threading.Event();outcome=[]
 original=Admission._lock
 @contextmanager
 def observed(self,parent,name,mode,*args,**kwargs):
  if name==self._key('other-selector','lease') and mode==fcntl.LOCK_EX:attempted.set()
  with original(self,parent,name,mode,*args,**kwargs) as value:yield value
 Admission._lock=observed
 try:
  assert writer.stdout.readline().strip()=='OTHER_WRITER_NATIVE_SCOPE'
  def capturing():
   try:outcome.append(capture((selector,),preview.scope_digest,home/'shared.tldw-backup.zip',options=options,cancel=threading.Event()))
   except BaseException as error:outcome.append(error)
   finally:finished.set()
  thread=threading.Thread(target=capturing);thread.start()
  assert attempted.wait(10),('capture did not attempt the related native lease',outcome)
  assert not finished.is_set(),'capture entered while the actual other writer remained active'
  writer.stdin.write('release\n');writer.stdin.flush()
  assert writer.stdout.readline().strip()=='OTHER_WRITER_RETIRED'
  assert writer.wait(timeout=10)==0
  thread.join(30);assert not thread.is_alive()
  # The actual completed writer changed the reviewed scope; a new review is required.
  from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired
  assert len(outcome)==1,outcome
  assert (outcome[0].issues==('scope_changed',) if isinstance(outcome[0],CaptureReviewRequired) else type(outcome[0]) is ValueError and str(outcome[0])=='scope_changed'),outcome
  assert not (home/'shared.tldw-backup.zip').exists()
 finally:
  Admission._lock=original
  if writer.poll() is None:
   writer.stdin.write('release\n');writer.stdin.flush()
   writer.wait(timeout=10)
reviewed=preview_capture((selector,),options=options)
assert reviewed.complete,reviewed.issues
result=capture((selector,),reviewed.scope_digest,home/'after-review.tldw-backup.zip',options=options,cancel=threading.Event())
assert result.inventory.complete
manifest=json.loads(result.manifest_bytes)
assert len(manifest['profile_ids'])==1
assert all(item.path!=other for item in result.inventory.items)
assert bootstrap._registry(root)==registry_before and bootstrap._records(root)[1]==bindings_before
print('SHARED_PARENT_CAPTURE_EXCLUSION_COMPLETE')
"""
)


@pytest.mark.parametrize("native_profile", ["custom"], indirect=True)
def test_capture_waits_for_other_bound_selector_writer_without_extra_payload(
    native_profile,
):
    script = "WRITER=" + repr(_OTHER_WRITER) + "\n" + _SHARED_CAPTURE
    output = _child(
        native_profile[0], script, "runtime", timeout=80, selector=native_profile[1]
    )
    assert "SHARED_PARENT_CAPTURE_EXCLUSION_COMPLETE" in output


@pytest.fixture
def exact_config_binding(tmp_path):
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        bind_profile,
    )

    parent = tmp_path / "private-config"
    parent.mkdir(mode=0o700)
    selector = parent / "config.toml"
    selector.write_text('[general]\nusers_name="fixture"\n')
    selector.chmod(0o600)
    root = tmp_path / "bootstrap"
    authority = admission_authority(root)
    authority.register("profile", (selector,))
    bind_profile(root, selector, ("profile",), root / "admission")
    return root, authority, selector


@pytest.mark.parametrize("alias", ["current", "historical"])
def test_capture_relation_refuses_foreign_native_file_alias(
    exact_config_binding, tmp_path, alias
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    root, authority, selector = exact_config_binding
    path = selector.parent / "runtime_policy.json"
    path.write_text("{}")
    path.chmod(0o600)
    authority.register("foreign", (path,))
    if alias == "historical":
        other = tmp_path / "foreign.json"
        other.write_text("{}")
        other.chmod(0o600)
        authority.remap("foreign", (other,), 2)
    before = bootstrap._registry(root)
    bindings = storage._config_capture_bindings(root, (selector,))
    with pytest.raises(
        bootstrap.RecoveryRequired, match="capture_config_scope_changed"
    ):
        storage._config_capture_sources(root, (selector,), bindings)
    assert bootstrap._registry(root) == before
    assert path.read_text() == "{}"


@pytest.mark.parametrize("change", ["config", "related_selector", "parent"])
def test_capture_rechecks_native_binding_and_parent_before_discovery(
    exact_config_binding, change
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.control_records import (
        UNBOUND_NAMESPACE,
        bind_profile,
    )
    from tldw_chatbook.Backup_Recovery.models import DiscoverySelections

    root, authority, selector = exact_config_binding
    proof = storage._config_capture_bindings(root, (selector,))
    if change == "config":
        selector.write_text('[general]\nusers_name="changed"\n')
    elif change == "related_selector":
        other = selector.with_name("other.toml")
        other.write_bytes(selector.read_bytes())
        other.chmod(0o600)
        authority.register("other", (other,))
        bind_profile(root, other, ("other",), root / "admission")
    else:
        old = selector.parent.with_name("preserved-parent")
        selector.parent.rename(old)
        selector.parent.mkdir(mode=0o700)
        (old / selector.name).rename(selector)
    with (
        authority.maintenance(("profile", UNBOUND_NAMESPACE), 2) as session,
        pytest.raises(bootstrap.RecoveryRequired, match="capture_config_scope_changed"),
    ):
        session._discover_capture_inventory(
            (selector,),
            DiscoverySelections(),
            "must-not-reach-discovery",
            config_bindings=proof,
        )
    assert not bootstrap._records(root)[0]


@pytest.mark.parametrize("damage", ["public_parent", "symlink", "hardlink"])
def test_capture_relation_never_repairs_unsafe_native_sources(
    exact_config_binding, tmp_path, damage
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Utils.platform_files import os as native_os

    root, _, selector = exact_config_binding
    path = selector.parent / "runtime_policy.json"
    if damage == "public_parent":
        if sys.platform == "win32":
            subprocess.run(
                ["icacls", str(selector.parent), "/grant", "*S-1-1-0:(R)"],
                check=True,
                capture_output=True,
            )
        else:
            selector.parent.chmod(0o755)
        assert native_os.stat(selector.parent).st_mode & 0o044
    else:
        original = tmp_path / "preserved.json"
        original.write_text("{}")
        original.chmod(0o600)
        if damage == "symlink":
            path.symlink_to(original)
        else:
            os.link(original, path)
    before = bootstrap._registry(root)
    with pytest.raises(
        bootstrap.RecoveryRequired, match="capture_config_(parent|source)_unsafe"
    ):
        proof = storage._config_capture_bindings(root, (selector,))
        storage._config_capture_sources(root, (selector,), proof)
    assert bootstrap._registry(root) == before
    if damage == "public_parent":
        assert native_os.stat(selector.parent).st_mode & 0o044
    else:
        assert original.read_text() == "{}"


def test_initial_discovery_cannot_read_sibling_as_another_installed_owner(
    exact_config_binding,
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE

    root, authority, selector = exact_config_binding
    path = selector.parent / "runtime_policy.json"
    path.write_text("{}")
    path.chmod(0o600)
    bindings = storage._config_capture_bindings(root, (selector,))
    with authority.maintenance(("profile", UNBOUND_NAMESPACE), 2) as session:
        session._config_capture_sources = storage._config_capture_sources(
            root, (selector,), bindings
        )
        with session._discovery_reads():
            assert (
                storage._read_recovery_file(
                    "runtime.source_state", path, max_bytes=1024
                )
                == b"{}"
            )
            with pytest.raises(
                bootstrap.RecoveryRequired, match="capture_source_outside_scope"
            ):
                storage._read_recovery_file("ui.emoji_recents", path, max_bytes=1024)
    assert path.read_bytes() == b"{}"


def test_shared_sibling_dependency_matches_its_own_selected_config(
    exact_config_binding,
):
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile
    from tldw_chatbook.Backup_Recovery.models import (
        DISCOVERY_CONTEXT_KEY,
        DiscoveryContext,
        Inventory,
        StorageItem,
    )
    from tldw_chatbook.runtime_policy.recovery import recovery_adapters

    root, authority, selector = exact_config_binding
    other = selector.with_name("other.toml")
    other.write_bytes(selector.read_bytes())
    other.chmod(0o600)
    authority.register("other", (other,))
    bind_profile(root, other, ("other",), root / "admission")
    path = selector.parent / "runtime_policy.json"
    path.write_text("{}")
    path.chmod(0o600)
    rows = tuple(
        recovery_adapters()[0].discover(
            {DISCOVERY_CONTEXT_KEY: DiscoveryContext(source, key)}
        )[0]
        for source, key in ((selector, "first"), (other, "second"))
    )
    inventory = Inventory(
        tuple(
            StorageItem("config", row.dependencies[0], source, "included", ())
            for row, source in zip(rows, (selector, other), strict=True)
        )
        + rows,
        True,
        "installed-shared-relation",
        (),
    )
    bindings = storage._config_capture_bindings(root, (selector, other))
    sources = storage._config_capture_sources(root, (selector, other), bindings)
    assert all(storage._config_capture_item(row, inventory, sources) for row in rows)


def test_related_selector_exclusion_never_grants_its_owned_sibling(
    exact_config_binding,
):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.control_records import bind_profile

    root, authority, selector = exact_config_binding
    other = selector.with_name("other.toml")
    other.write_bytes(selector.read_bytes())
    other.chmod(0o600)
    path = selector.parent / "runtime_policy.json"
    path.write_text("{}")
    path.chmod(0o600)
    authority.register("other", (other, path))
    bind_profile(root, other, ("other",), root / "admission")
    proof = storage._config_capture_bindings(root, (selector,))
    assert len(proof) == 2  # Both native scopes must be held, but ownership differs.
    before = bootstrap._registry(root)
    with pytest.raises(
        bootstrap.RecoveryRequired, match="capture_config_scope_changed"
    ):
        storage._config_capture_sources(root, (selector,), proof)
    assert bootstrap._registry(root) == before
    assert path.read_text() == "{}"
