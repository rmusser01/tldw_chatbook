"""Standalone guarded late-budget component proof; not a pytest/App test.

Run only after controller source review with the existing evidence runner:
.venv/bin/python .superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-8-public-cutover-test-runner.py EPOCH --script Tests/Backup_Recovery/test_mobile_late_budget_recovery.py ABSOLUTE_RESULT_JSON

The parent and both children inherit real HOME/USERPROFILE unchanged. A fresh
private native fixture is created once; the second process never reseeds it.
"""

import json
import os
import pwd
import subprocess
import sys
import tempfile
from pathlib import Path

_COMMON = r'''
"""SOURCE-ONLY test fixture prelude; not approved for execution.

For a new private fixture, never the retained b025 operation. The launcher must
supply real HOME and the original USERPROFILE state; this child changes neither.
The proposed test uses native _capture_fixture rather than an App constructor.
"""
import os
from pathlib import Path
import pwd
import stat
import sys

ENTRY_HOME = {key: os.environ.get(key) for key in ("HOME", "USERPROFILE")}
UID_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)
assert ENTRY_HOME == {"HOME": str(UID_HOME), "USERPROFILE": None}
assert os.environ.get("PYTHON_KEYRING_BACKEND") == "keyring.backends.null.Keyring"
assert not any(name == "keyring" or name.startswith("keyring.") for name in sys.modules)
from Tests.real_profile_guard import install as install_profile_guard, protected_roots, take_violations
from Tests.network_guard import install as install_network_guard, blocked_attempts
install_profile_guard()
install_network_guard()
GUARD_ROOTS = protected_roots()
assert Path(os.environ["TLDW_TEST_REAL_HOME"]) == UID_HOME
assert {str(UID_HOME / ".config/tldw_cli").lower(), str(UID_HOME / ".local/share/tldw_cli").lower()} <= set(GUARD_ROOTS)
assert not take_violations() and not blocked_attempts()

# The parent supplies ONE newly created private test directory explicitly. Record
# its resolved absolute path and identity before any native fixture construction;
# both processes receive the same path. No fallback to HOME or existing fixture.
FIXTURE = Path(sys.argv[1])
assert FIXTURE.is_absolute() and FIXTURE.resolve(strict=True) == FIXTURE
info = FIXTURE.lstat()
assert stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077
assert not FIXTURE.is_relative_to(UID_HOME)
assert "mp-v7aje_m_" not in FIXTURE.parts
SELECTOR = FIXTURE / "profile/config.toml"
DEFAULT_CONFIG = FIXTURE / "unused-default-config/config.toml"
DEFAULT_DATA = FIXTURE / ".local/share/tldw_cli"
BOOTSTRAP = FIXTURE / "bootstrap"
CONTROL = FIXTURE / "control"
DEFAULT_CONTROL = DEFAULT_CONFIG.parent / "recovery/control"
SOURCE = FIXTURE / "profile/data/shared/mobile"
TARGETS = tuple(FIXTURE / "profile/new/nested" / leaf for leaf in ("chat.db", "workspaces.db", "agent_runs.db"))

os.environ["TLDW_CONFIG_PATH"] = str(SELECTOR)
os.environ["TLDW_TEST_CONFIG_ROOT"] = str(FIXTURE / ".test-config")
os.environ["TLDW_TEST_MODE"] = "1"
os.environ["TLDW_DISABLE_CONFIG_WATCH"] = "1"
# Do not add XDG overrides, replace Path.home/expanduser or replace global
# environment lookup. Native lexical parsing and source/admission checks remain.
assert "tldw_chatbook.config" not in sys.modules
from tldw_chatbook.Backup_Recovery import profile_paths

def fixture_default_config():
    return DEFAULT_CONFIG

def fixture_default_data():
    return DEFAULT_DATA

profile_paths.default_config_path = fixture_default_config
profile_paths.default_base_data_dir = fixture_default_data
from tldw_chatbook.Backup_Recovery import bootstrap, config_adapter, service_storage, capture_service
for module in (bootstrap, config_adapter, service_storage, capture_service):
    assert "tldw_chatbook.config" not in sys.modules
    module.default_config_path = fixture_default_config
# This is the exact bootstrap selector used by the existing native prepared()
# fixture, not a synthetic profile/binding/registry or scope result.
bootstrap.default_bootstrap_root = lambda: BOOTSTRAP
assert profile_paths.effective_config_path() == SELECTOR
assert profile_paths.default_base_data_dir() == DEFAULT_DATA
assert profile_paths.selected_default_base_data_dir() == DEFAULT_DATA
assert bootstrap.default_bootstrap_root() == BOOTSTRAP
assert service_storage.default_control_root() == DEFAULT_CONTROL
assert service_storage.work_root(CONTROL) == CONTROL.with_name("control-work")

from tldw_chatbook.TTS import recovery as voices
from tldw_chatbook.Widgets.Tamagotchi import recovery as pets
native_voice_lexical = voices.lexical_path
VOICE_PATHS = {
    "~/.config/tldw_cli/chatterbox_voices": FIXTURE / ".config/tldw_cli/chatterbox_voices",
    "~/.config/tldw_cli/higgs_voices": FIXTURE / ".config/tldw_cli/higgs_voices",
}

def fixture_voice_path(value):
    spelling = str(value)
    if spelling in VOICE_PATHS:
        return VOICE_PATHS[spelling]
    if spelling.startswith("~"):
        raise ValueError("unexpected_fixture_home_relative_voice_path")
    return native_voice_lexical(value)

voices.lexical_path = fixture_voice_path
pets._ConfiguredPets._parent = lambda self: FIXTURE / ".config/tldw_chatbook"
assert all(voices.lexical_path(key) == value for key, value in VOICE_PATHS.items())
assert voices.lexical_path(FIXTURE / "profile/kokoro_voice_blends") == FIXTURE / "profile/kokoro_voice_blends"
try:
    voices.lexical_path("~/unexpected")
except ValueError:
    pass
else:
    raise AssertionError("unexpected home-relative voice path accepted")

import keyring
from keyring.backends.null import Keyring
assert type(keyring.get_keyring()) is Keyring
from tldw_chatbook.Backup_Recovery import credentials
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore

class EmptyIsolatedKeyring:
    def get_password(self, service, username):
        return None

fixture_credentials = KeyringServerCredentialStore(keyring_backend=EmptyIsolatedKeyring())
credentials._credential_store = lambda: fixture_credentials

def assert_fixture_locators():
    assert {key: os.environ.get(key) for key in ENTRY_HOME} == ENTRY_HOME
    assert protected_roots() == GUARD_ROOTS
    assert profile_paths.effective_config_path() == SELECTOR
    assert profile_paths.default_config_path() == DEFAULT_CONFIG
    assert profile_paths.default_base_data_dir() == DEFAULT_DATA
    assert bootstrap.default_bootstrap_root() == BOOTSTRAP
    assert service_storage.default_control_root() == DEFAULT_CONTROL
    from tldw_chatbook.Backup_Recovery.inventory import read_profile_config
    values = read_profile_config(SELECTOR)
    assert profile_paths.legacy_mobile_dir(values) == SOURCE
    assert profile_paths.canonical_console_paths(values) == TARGETS


import hashlib,json,traceback
from tldw_chatbook.Backup_Recovery import recovery_service
from tldw_chatbook.Backup_Recovery.mobile_consolidation import CanonicalConsoleStores,require_mobile_cutover
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
from Tests.Backup_Recovery.mobile_consolidation_fixture import fingerprints
source=SOURCE
stores=CanonicalConsoleStores(*TARGETS)
selector=SELECTOR
report_path=Path(sys.argv[2])

def leaf(path):
    info=path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and info.st_nlink==1
    assert not info.st_mode & 0o077 and info.st_size<=1024**2
    return {'path':str(path),'device':info.st_dev,'inode':info.st_ino,'mode':info.st_mode,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def binding_facts(operation=None):
    assert_fixture_locators()
    pending,profiles=bootstrap._records(BOOTSTRAP)
    registry=bootstrap._registry(BOOTSTRAP)
    binding=bootstrap._binding(SELECTOR,profiles,registry)
    assert binding is not None and binding['selector']==str(SELECTOR)
    effective=bootstrap.effective_roots(binding['roots'],(registry[name] for name in binding['namespaces']))
    package=Path(__import__('tldw_chatbook').__file__).resolve().parent/'assets'
    assert all(path.is_relative_to(FIXTURE) or path==package for path in effective)
    selected=[row for row in pending if operation is None or row['operation_id']==operation]
    if operation is not None:
        assert len(selected)==1 and len(pending)==1
        assert selected[0]['control_root']==str(CONTROL)
        assert selected[0]['selectors']==[str(SELECTOR)]
        assert set(binding['namespaces'])<=set(selected[0]['namespaces'])
    return {'binding':binding,'registry':{name:registry[name] for name in binding['namespaces']},'effective_roots':[str(path) for path in effective],'selector':leaf(SELECTOR),'pending':selected}

def claim_facts(operation):
    journal=_journal(CONTROL,operation)
    with journal._locked(exclusive=False) as parent:records=journal._records(parent)
    claims=[row for row in records if row.event=='mobile_claim_observed']
    assert len(claims)==1
    path=Path(claims[0].evidence['path'])
    assert path.is_relative_to(FIXTURE)
    return leaf(path)
errors=[]
native_issue=recovery_service.issue_code
def issue(error,**kwargs):
 errors.append({'type':type(error).__name__,'code':str(error),'stack':traceback.format_tb(error.__traceback__)})
 return native_issue(error,**kwargs)
recovery_service.issue_code=issue
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget,_mobile_budget_pair
budgets=[]
native_init=_TraceReadBudget.__init__
def issued(actual):
 native_init(actual)
 budgets.append(actual)
def pairs():
 return [{'coupled':_mobile_budget_pair(row._parent,row),'aggregate_remaining':row._parent._remaining,'native_remaining':row._remaining} for row in budgets if row._parent is not None]

'''

_START = r"""

import pytest
from Tests.Backup_Recovery import mobile_consolidation_fixture
from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture
from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed
assert not SELECTOR.exists() and not SOURCE.exists() and not CONTROL.exists()
patcher=pytest.MonkeyPatch()
patcher.setattr(mobile_consolidation_fixture,'_seed',_small_native_seed)
inherited_mask=os.umask(0o077)
try:
    voice_parent=FIXTURE / '.config/tldw_cli'
    voice_parent.mkdir(mode=0o700,parents=True)
    parent_info=voice_parent.lstat()
    assert stat.S_ISDIR(parent_info.st_mode) and stat.S_IMODE(parent_info.st_mode)==0o700 and parent_info.st_uid==os.getuid()
    assert all(path.parent==voice_parent and not path.exists() for path in VOICE_PATHS.values())
    with _capture_fixture(FIXTURE,patcher) as values:
        _,fixture_journal,fixture_selector,fixture_stores,preview,_,_=values
finally:
    os.umask(inherited_mask)
assert fixture_selector==SELECTOR and fixture_stores==stores
for directory in (SELECTOR.parent, SELECTOR.parent / 'data'):
    observed=directory.lstat()
    assert stat.S_ISDIR(observed.st_mode) and stat.S_IMODE(observed.st_mode)==0o700 and observed.st_uid==os.getuid()
assert fixture_journal.root.parent==CONTROL
# _capture_fixture installed its own empty backend; restore the exact reviewed
# real credential-store wrapper after native fixture construction has retired.
credentials._credential_store=lambda:fixture_credentials
assert_fixture_locators()
initial_binding=binding_facts()
assert initial_binding['pending']==[]
before=fingerprints(source)
service=RecoveryService(CONTROL)
assert service.preview_mobile_consolidation(selector,selector)==preview
_TraceReadBudget.__init__=issued
from tldw_chatbook.Backup_Recovery.journal import Journal
original_append=Journal._append
original_reserve=_TraceReadBudget.reserve
retired=[]
fault=[]
def append(journal,parent,event,evidence,**kwargs):
 result=original_append(journal,parent,event,evidence,**kwargs)
 if event=='mobile_owner_retired':retired.append(evidence['owner_id'])
 return result
def reserve(actual,amount):
 if retired and not fault:
  child=next((row for row in budgets if row._parent is not None),None)
  if child is not None and actual in (child,child._parent):
   aggregate=child._parent
   fault.append({'retired':list(retired),'aggregate_before':aggregate._remaining,'native_before':child._remaining})
   original_reserve(aggregate,aggregate._remaining)
 return original_reserve(actual,amount)
Journal._append=append
_TraceReadBudget.reserve=reserve
report={'scope':'fixture-created standalone component; not ordinary App startup','entry_home':ENTRY_HOME,'guard_roots':GUARD_ROOTS,'initial_binding':initial_binding}
try:
 operation=service.start_mobile_consolidation(preview,destination_confirmed=True,rollback_password=b'test-private-mobile-rollback')
 state=service.wait(operation)
 report.update(operation=operation,state={key:state[key] for key in ('state','phase','issues')},fault=fault,errors=errors,pairs_after_wait=pairs())
 assert state['state']=='recovery_required',report
 assert fault and fault[0]['retired']==['db.chachanotes.primary'],report
finally:
 Journal._append=original_append
 _TraceReadBudget.reserve=original_reserve
 service.close()
 report.update(source_before=before,source_after=fingerprints(source),pairs_after_close=pairs(),profile_guard=take_violations(),network_guard=blocked_attempts())
 report_path.write_text(json.dumps(report,indent=2))
assert report['source_before']==report['source_after']
assert not report['profile_guard'] and not report['network_guard']
assert len(report['pairs_after_close'])==1 and report['pairs_after_close'][0]['coupled']
assert report['pairs_after_close'][0]['aggregate_remaining']==0
journal=_journal(service.control_root,operation)
with journal._locked(exclusive=False) as parent:records=journal._records(parent)
assert any(row.event=='mobile_owner_retired' and row.evidence['owner_id']=='db.chachanotes.primary' for row in records)
assert all(row.event!='mobile_committed' for row in records)
assert service.pending_operations()
report['pending_binding']=binding_facts(operation)
report['claim']=claim_facts(operation)
try:require_mobile_cutover(source,stores)
except (ValueError,RuntimeError) as error:report['gate_refusal']=type(error).__name__+':'+str(error)
else:raise AssertionError('late budget failure released ordinary gate')
report['events']=[row.event for row in records]
report['profile_guard']=take_violations()
report['network_guard']=blocked_attempts()
report_path.write_text(json.dumps(report,indent=2))
assert not report['profile_guard'] and not report['network_guard']

assert {key:os.environ.get(key) for key in ENTRY_HOME}==ENTRY_HOME
assert protected_roots()==GUARD_ROOTS

"""

_FINISH = r"""

previous=json.loads(report_path.with_name('native-interrupted.json').read_text())
operation=previous['operation']
assert binding_facts(operation)==previous['pending_binding']
assert claim_facts(operation)==previous['claim']
before=fingerprints(source)
assert before==previous['source_before']
service=RecoveryService(CONTROL)
report={'scope':'explicit fresh-process same-op continuation','entry_home':ENTRY_HOME,'guard_roots':GUARD_ROOTS}
try:
 status=service.status(operation)
 assert 'finish' in status['actions'],dict(status)
 _TraceReadBudget.__init__=issued
 resumed=service.start_recovery(operation,action='finish',rollback_password=b'test-private-mobile-rollback')
 state=service.wait(resumed)
 report.update(operation=operation,recovery_call=resumed,state={key:state[key] for key in ('state','phase','issues')},errors=errors,pairs_after_wait=pairs())
 assert state['state']=='succeeded',report
 assert service.pending_operations()==()
finally:
 service.close()
 report.update(source_before=before,source_after=fingerprints(source),pairs_after_close=pairs(),profile_guard=take_violations(),network_guard=blocked_attempts())
 report_path.write_text(json.dumps(report,indent=2))
assert before==report['source_after'] and not report['profile_guard'] and not report['network_guard']
assert len(report['pairs_after_close'])==1 and report['pairs_after_close'][0]['coupled']
assert require_mobile_cutover(source,stores) is None
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
chat=CharactersRAGDB(stores.chat_path,'ordinary-explicit-budget-recovery-reader')
try:assert len(chat.get_all_conversation_ids())==2
finally:chat.close_connection()
assert require_mobile_cutover(source,stores) is None
report['ordinary_same_target']=True
report['source_after']=fingerprints(source)
report['profile_guard']=take_violations()
report['network_guard']=blocked_attempts()
report_path.write_text(json.dumps(report,indent=2))
assert report['source_after']==before and not report['profile_guard'] and not report['network_guard']

assert claim_facts(operation)==previous['claim']
assert binding_facts()['pending']==[]
assert {key:os.environ.get(key) for key in ENTRY_HOME}==ENTRY_HOME
assert protected_roots()==GUARD_ROOTS

"""


def run_standalone(result_path):
    entry = {key: os.environ.get(key) for key in ("HOME", "USERPROFILE")}
    uid_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert entry == {"HOME": str(uid_home), "USERPROFILE": None}
    from Tests.network_guard import blocked_attempts
    from Tests.network_guard import install as install_network_guard
    from Tests.real_profile_guard import install as install_profile_guard
    from Tests.real_profile_guard import protected_roots, take_violations

    install_profile_guard()
    install_network_guard()
    assert Path(os.environ["TLDW_TEST_REAL_HOME"]) == uid_home
    assert str(uid_home / ".config/tldw_cli").lower() in protected_roots()
    scratch = Path(os.environ["TMPDIR"]).resolve(strict=True)
    fixture = Path(tempfile.mkdtemp(prefix="fresh-mobile-late-", dir=scratch)).resolve(
        strict=True
    )
    assert fixture.parent == scratch and not fixture.is_relative_to(uid_home)
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(repository),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    assert {key: environment.get(key) for key in entry} == entry
    assert {
        key: environment.get(key) for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME")
    } == {key: os.environ.get(key) for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME")}
    report = {
        "scope": "standalone native fixture-created test, not pytest or ordinary App startup",
        "fixture": str(fixture),
        "entry_home": entry,
        "children": [],
    }
    try:
        for name, body in (("interrupted", _START), ("finished", _FINISH)):
            output = fixture / f"native-{name}.json"
            log = fixture / f"native-{name}.log"
            argv = [
                sys.executable,
                "-X",
                "faulthandler",
                "-c",
                _COMMON + body,
                str(fixture),
                str(output),
            ]
            with log.open("w") as stream:
                child = subprocess.run(
                    argv,
                    cwd=repository,
                    env=environment,
                    stdout=stream,
                    stderr=stream,
                    timeout=300,
                    check=False,
                )
            report["children"].append(
                {
                    "name": name,
                    "returncode": child.returncode,
                    "log": str(log),
                    "output": str(output),
                }
            )
            result_path.write_text(json.dumps(report, indent=2))
            assert child.returncode == 0, log.read_text()[-16000:]
        interrupted = json.loads((fixture / "native-interrupted.json").read_text())
        finished = json.loads((fixture / "native-finished.json").read_text())
        assert interrupted["operation"] == finished["operation"]
        assert interrupted["state"]["state"] == "recovery_required"
        assert (
            finished["state"]["state"] == "succeeded"
            and finished["ordinary_same_target"]
        )
        report["operation"] = interrupted["operation"]
        report["completed"] = True
    finally:
        report["profile_guard"] = take_violations()
        report["network_guard"] = blocked_attempts()
        report["exit_home"] = {key: os.environ.get(key) for key in entry}
        result_path.write_text(json.dumps(report, indent=2))
    assert report["exit_home"] == entry
    assert not report["profile_guard"] and not report["network_guard"]


if __name__ == "__main__":
    output = Path(sys.argv[1])
    assert output.is_absolute() and output.parent.is_dir() and not output.exists()
    run_standalone(output)
