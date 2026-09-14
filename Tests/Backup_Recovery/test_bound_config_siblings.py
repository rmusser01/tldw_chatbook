"""Installed sibling owners use their bound config without parent authority."""

import sys

import pytest

from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SIBLINGS = (
    _SCRIPT.split("assert config.get_cli_setting")[0]
    + r"""
import asyncio,json,sys
from tldw_chatbook.Backup_Recovery import raw_participants as raw,storage_admission as storage
from tldw_chatbook.Widgets import emoji_picker
from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore,RuntimeSourceState,runtime_source_state_to_dict
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Backup_Recovery.async_file_participants import _FileJob
mode=sys.argv[1]
registry_before=bootstrap._registry(root)
if mode=='emoji':
 path=parent/'recent_emojis.json'
 assert not path.exists()
 assert emoji_picker.load_recent_emojis()==[]
 emoji_picker.save_recent_emoji('test-marker')
 assert json.loads(path.read_text())['recent']==['test-marker']
 assert emoji_picker.load_recent_emojis()==['test-marker']
elif mode=='runtime':
 path=parent/'runtime_policy.json'
 path.unlink(missing_ok=True)
 store=RuntimeSourceStateStore(path,application_owned_directory=parent)
 state=store.load()
 assert isinstance(state,RuntimeSourceState) and not path.exists()
 store.save(state)
 assert json.loads(path.read_text())==json.loads(json.dumps(runtime_source_state_to_dict(store.load())))
else:
 path=parent/'ui_state.toml'
 assert not path.exists()
 screen=object.__new__(ChatScreen)
 snapshot={'collapsible_states':{'notes':True},'search_query':'fixture','last_active_section':'notes'}
 async def save():
  with _FileJob(screen,'sidebar_state') as job:
   outcome=await job.run(snapshot)
   if outcome.error:raise outcome.error
 asyncio.run(save())
 import toml
 with raw._scope(screen,'sidebar_state') as operation:
  with raw._file(operation,raw._selected(operation),'r') as stream:
   assert toml.load(stream)['sidebar']==snapshot
 assert not path.with_suffix(path.suffix+'.tmp').exists()
assert bootstrap._registry(root)==registry_before
assert bootstrap._records(root)[1]==before
assert not storage._raw_operations
assert not any(parent.glob('*.tmp'))
print('retired and reopened')
"""
)


@pytest.mark.parametrize("owner", ["emoji", "runtime", "sidebar"])
def test_installed_config_sibling_absent_read_write(tmp_path, owner):
    _run(tmp_path, owner, "sibling", script=_SIBLINGS, timeout=35)


_GUARDS = (
    _SIBLINGS.split("mode=sys.argv[1]")[0]
    + r"""
owner,case=sys.argv[1:3]
if owner=='emoji':source,route,path=emoji_picker,'emoji',parent/'recent_emojis.json'
elif owner=='runtime':
 path=parent/'runtime_policy.json'
 source,route=RuntimeSourceStateStore(path,application_owned_directory=parent),'runtime_state'
else:source,route,path=object.__new__(ChatScreen),'sidebar_state',parent/'ui_state.toml'
path.unlink(missing_ok=True)
if case=='arbitrary':
 with raw._scope(source,route,writing=True) as operation:
  for forbidden in (parent/'unrelated',parent/'ui_state.toml.other.tmp',parent):
   try:raw._check(operation,forbidden,writing=True)
   except bootstrap.RecoveryRequired:pass
   else:raise AssertionError('unselected sibling or directory admitted')
elif case=='unsafe_parent':
 with raw._scope(source,route,writing=True) as operation:
  parent.chmod(0o755)
  try:
   try:raw._check(operation,path,writing=True)
   except bootstrap.RecoveryRequired:pass
   else:raise AssertionError('unsafe parent admitted')
  finally:parent.chmod(0o700)
elif case=='foreign':
 path.write_bytes(b'foreign');path.chmod(0o600)
 authority.register('foreign',(path,))
 try:
  with raw._scope(source,route,writing=True):raise AssertionError('foreign admitted')
 except bootstrap.RecoveryRequired:pass
 assert path.read_bytes()==b'foreign'
else:
 other=selected.with_name('other.toml');other.write_bytes(selected.read_bytes());other.chmod(0o600)
 with raw._scope(source,route,writing=True) as operation:
  state=raw._states[operation]
  participant=state.participant
  if case=='no_participant':state.participant=None
  try:
   os.environ['TLDW_CONFIG_PATH']=str(other)
   try:
    with raw._file(operation,path,'w') as stream:stream.write('must refuse')
   except bootstrap.RecoveryRequired:pass
   else:raise AssertionError('changed config anchor admitted')
  finally:
   os.environ['TLDW_CONFIG_PATH']=str(selected)
   state.participant=participant
 assert not path.exists()
assert not storage._raw_operations
print('retired and reopened')
"""
)


@pytest.mark.parametrize("owner", ["emoji", "runtime", "sidebar"])
@pytest.mark.parametrize(
    "case",
    ["same_parent_selector", "no_participant", "foreign", "arbitrary", "unsafe_parent"],
)
def test_sibling_guard_rechecks_exact_config_anchor_and_foreign_owner(
    tmp_path, owner, case
):
    if case == "unsafe_parent" and sys.platform == "win32":
        pytest.skip(
            "chmod posture transition is POSIX-only; owner IO remains cross-platform"
        )
    _run(tmp_path, owner, case, script=_GUARDS, timeout=35)


def test_runtime_bound_sibling_retains_wrong_owned_directory_refusal(tmp_path):
    script = (
        _SIBLINGS.split("mode=sys.argv[1]")[0]
        + r"""
path=parent/'runtime_policy.json';path.unlink(missing_ok=True)
other=home/'other-private';other.mkdir(mode=0o700)
store=RuntimeSourceStateStore(path,application_owned_directory=other)
try:store.save(RuntimeSourceState())
except ValueError:pass
else:raise AssertionError('wrong owned directory admitted')
assert not path.exists() and not list(other.iterdir())
assert bootstrap._records(root)[1]==before and not storage._raw_operations
print('retired and reopened')
"""
    )
    _run(tmp_path, "runtime", "wrong-parent", script=script, timeout=35)
