"""An actual restored and reviewed native generation supplies captured authority."""

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_mcp_recovery_review import _SETUP

# Configure the original recovery input, so native receipt bindings stay intact.
_CONFIGURED_SETUP = _SETUP.replace(
    "user=source/'Local';user.mkdir(mode=0o700)",
    r"""selector.write_text(selector.read_text()+'\n[database]\nchachanotes_db_path='+json.dumps(str(source/'canonical'/'chat.db'))+'\nworkspaces_db_path='+json.dumps(str(source/'canonical'/'workspaces.db'))+'\n')
user=source/'Local';user.mkdir(mode=0o700)""",
).replace(
    "'profile:profile:paths.data_dir':destination/'data'",
    "'profile:profile:paths.data_dir':destination/'data','profile:profile:database.chachanotes_db_path':destination/'data'/'canonical'/'chat.db','profile:profile:database.workspaces_db_path':destination/'data'/'canonical'/'workspaces.db'",
)

_SCRIPT = (
    _CONFIGURED_SETUP
    + r"""
from Tests.RuntimePolicy.test_server_credentials import FakeKeyring
from tldw_chatbook.runtime_policy.server_credentials import KeyringServerCredentialStore
from tldw_chatbook.Backup_Recovery import credentials,profile_paths,config_adapter
credentials._credential_store = lambda: KeyringServerCredentialStore(keyring_backend=keyring)
keyring=FakeKeyring()
from tldw_chatbook.Backup_Recovery.credential_policies import GENERATION_KEYRINGS
for _,key_service,key_names in GENERATION_KEYRINGS:
 for key_name in key_names:keyring.set_password(key_service,key_name,'isolated-fixture-generation-secret')
service=plane()
service.approve_recovery_review(service.capture_recovery_review())
from tldw_chatbook.MCP import recovery_activation as native
entry=__import__("tldw_chatbook.Backup_Recovery.isolated_restore",fromlist=["_launch_descriptor"])._launch_descriptor(profile,control)
active_path=native.selected_path(user/'mcp_permissions.json')
assert active_path != user/'mcp_permissions.json'
assert activation.allowed(witness['generation'],'mcp.permissions')
from tldw_chatbook.Backup_Recovery.inventory import read_profile_config,discover
from tldw_chatbook.Backup_Recovery.mobile_consolidation import CanonicalConsoleStores,_preview_selected_mobile
from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile,seed_canonical_console
selector=destination/'config'/'config.toml'
values=read_profile_config(selector)
seed_legacy_mobile(profile_paths.legacy_mobile_dir(values))
stores=CanonicalConsoleStores(*profile_paths.canonical_console_paths(values))
seed_canonical_console(stores)
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
for key,kind in (('media_db_path',MediaDatabase),('prompts_db_path',PromptsDatabase),('evals_db_path',EvalsDB),('library_collections_db_path',LibraryCollectionsDB),('subscriptions_db_path',SubscriptionsDB),('scheduled_tasks_db_path',ScheduledTasksDB)):
 store=kind(profile_paths.database_path(values,key),'authority-generation-fixture');store.close()
unused=base/'unused';unused.mkdir(mode=0o700)
profile_paths.default_config_path=lambda:unused/'config.toml'
config_adapter.default_config_path=profile_paths.default_config_path
from tldw_chatbook.Backup_Recovery.owner_registry import install_adapters
install_adapters()
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _captured_mobile
from Tests.Backup_Recovery.test_mobile_captured_sqlite import _fingerprints
root=bootstrap.default_bootstrap_root()
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None:startup.close()
authority=admission_authority(root)
names=tuple(sorted(set(bootstrap._registry(root))|{'bootstrap.unbound'}))
mobile_control=base/'mobile-control';mobile_control.mkdir(mode=0o700)
mobile_journal=Journal(mobile_control,'mobile-authority-generation')
archives=base/'mobile-archives';archives.mkdir(mode=0o700)
with authority.maintenance(names,2) as session:
 with session._discovery_reads(private_sqlite=True):
  inventory=discover((selector,));preview=_preview_selected_mobile(selector,selector)
 assert inventory.complete,(inventory.issues,[(item.owner,item.logical_id,item.status,str(item.path)) for item in inventory.items if item.status in {'unsupported','unavailable','missing_required'}])
 from dataclasses import replace
 from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _preview_mobile_credential_issues
 credential_issues=_preview_mobile_credential_issues(session,preview,inventory.scope_digest,cancel=Event())
 assert all(issue.startswith('credential_manual_recovery_required:') for _,issues in credential_issues for issue in issues),credential_issues
 preview=replace(preview,credential_acknowledgements=credential_issues)
 with _captured_mobile(session,mobile_journal,preview,inventory.scope_digest,destination=archives,password=b'private-mobile',cancel=Event()) as captured:
  import pytest
  from tldw_chatbook.Backup_Recovery import isolated_restore
  from tldw_chatbook.Backup_Recovery.models import DiscoveryContext
  from tldw_chatbook.Evals.recovery import _retained_definition_paths
  context=DiscoveryContext(selector,hashlib.sha256(str(selector).encode()).hexdigest()[:24])
  environment_before=dict(os.environ)
  for call in (isolated_restore._launch_state, isolated_restore._launch_descriptor,
               isolated_restore.profile_requirements, isolated_restore.launch_profile):
   with pytest.raises(ValueError,match="^recovery_pending$"):
    call(entry.profile_id, control)
  assert dict(os.environ)==environment_before
  with pytest.raises(ValueError,match="^recovery_pending$"):
   _retained_definition_paths(context)
  with session._discovery_reads(private_sqlite=True):
   assert isolated_restore._eval_metadata_descriptor(entry.profile_id,control)==entry
   _retained_definition_paths(context)
   from copy import copy
   actual_scope=storage._local.discovery_scope
   assert not storage._mobile_discovery_pending(root,('foreign.namespace',),selector)
   assert not storage._mobile_discovery_pending(root,names,base/'foreign-selector.toml')
   storage._local.discovery_scope=copy(actual_scope)
   try:
    with pytest.raises((ValueError,RuntimeError)):
     isolated_restore._eval_metadata_descriptor(entry.profile_id,control)
   finally:
    storage._local.discovery_scope=actual_scope
  storage._local.discovery_scope=actual_scope
  try:
   with pytest.raises((ValueError,RuntimeError),match='discovery_scope_inactive'):
    isolated_restore._eval_metadata_descriptor(entry.profile_id,control)
  finally:
   storage._local.discovery_scope=None
  pending_path=root/('pending-'+bootstrap._key(mobile_journal.operation_id)+'.json')
  original_pending=pending_path.read_bytes()
  pending_path.write_bytes(b'{}')
  try:
   assert bootstrap.startup_permission(selector,root)==(False,'recovery_scope_uncertain')
   with pytest.raises(ValueError):isolated_restore._eval_metadata_descriptor(entry.profile_id,control)
   with pytest.raises(ValueError):_retained_definition_paths(context)
  finally:
   pending_path.write_bytes(original_pending)
  before=_fingerprints(destination)
  with captured.sqlite_reads(password=b'private-mobile',cancel=Event()) as reads:
   proof=reads.authority_reference('permission_profile','default')
   assert proof.owner_id=='mcp.permissions'
   # Native selection must use the reviewed fresh file, never the retained original.
   assert proof.member_digest==hashlib.sha256(active_path.read_bytes()).hexdigest()
   assert proof.member_digest!=hashlib.sha256((user/'mcp_permissions.json').read_bytes()).hexdigest()
   from tldw_chatbook.Backup_Recovery.activation import _key
   approval=activation._generation(witness['generation'])/('approved-'+_key('mcp.permissions')+'.json')
   saved_approval=approval.with_suffix('.saved')
   approval.rename(saved_approval)
   try:
    assert not activation.allowed(witness['generation'],'mcp.permissions')
    with pytest.raises(ValueError,match='generation_unreviewed'):
     reads.authority_reference('permission_profile','default')
   finally:
    saved_approval.rename(approval)
   original_approval=approval.read_bytes()
   approval.write_bytes(b'{}')
   try:
    assert not activation.allowed(witness['generation'],'mcp.permissions')
    with pytest.raises(ValueError,match='generation_unreviewed'):
     reads.authority_reference('permission_profile','default')
   finally:
    approval.write_bytes(original_approval)
   binding_path=activation._generation(witness['generation'])/native._record_name('mcp.permissions',user/'mcp_permissions.json')
   original_binding=binding_path.read_bytes()
   binding_path.write_bytes(b'{}')
   try:
    with pytest.raises(ValueError):reads.authority_reference('permission_profile','default')
   finally:
    binding_path.write_bytes(original_binding)
  assert _fingerprints(destination)==before
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_actual_active_generation_reference_uses_native_reviewed_root(tmp_path):
    _run(tmp_path, "mcp", "captured", script=_SCRIPT, timeout=240)
