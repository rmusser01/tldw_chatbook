from pathlib import Path
import ast,json,subprocess
p=Path('Tests/Architecture/test_persistent_diagnostic_inventory.py');source=p.read_text();tree=ast.parse(source);assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='REVIEWED_METADATA_ONLY_DIAGNOSTICS' for t in n.targets));root=assignment.value
moves={
 'tldw_chatbook/Chat/console_agent_bridge.py': {'fleet drain consumer raised': ('tldw_chatbook/Chat/console_agent_bridge.py','fleet settlement consumer raised',('type(exc).__name__',))},
 'tldw_chatbook/Chat/console_chat_store.py': {'Failed to persist Console roleplay message projection': ('tldw_chatbook/Chat/console_chat_store.py','Failed to persist planned Console roleplay message projection',('type(exc).__name__',))},
 'tldw_chatbook/Video_Generation/adapter_registry.py': {
  'Failed to initialize video adapter':('tldw_chatbook/Media_Generation/adapter_registry.py','Failed to initialize {} adapter for \'{}\' (error_type={})',('self.modality','name','type(exc).__name__')),
  'Failed to resolve video adapter class':('tldw_chatbook/Media_Generation/adapter_registry.py','Failed to resolve {} adapter class for \'{}\' (error_type={})',('self.modality','name','type(exc).__name__'))},
 'tldw_chatbook/Video_Generation/config.py': {
  'unknown-key scan failed':('tldw_chatbook/Media_Generation/config_machinery.py','{} unknown-key scan failed',('tables.section_name','type(e).__name__')),
  'keyring lookup failed':('tldw_chatbook/Media_Generation/config_machinery.py','keyring lookup failed for {}/{}',('tables.keyring_label','backend','type(e).__name__'))},
 'tldw_chatbook/app.py': {
  'Generated CSS is stale during module entry; rebuilding':('tldw_chatbook/app_entry.py','Generated CSS is stale during module entry; rebuilding',()),
  'Generated CSS is stale during CLI entry; rebuilding':('tldw_chatbook/app_entry.py','Generated CSS is stale during CLI entry; rebuilding',())},
}
retired={
 'tldw_chatbook/UI/Screens/chat_screen.py': {'Pending sidebar-state write failed': '_persist_sidebar_state_off_loop'},
 'tldw_chatbook/UI/MCP_Modules/mcp_workbench.py': {'MCP Tools-mode local master save failed':'_observe_local_master_save','MCP Tools-mode workspace root save failed':'_observe_workspace_root_save'},
 'tldw_chatbook/UI/Screens/settings_screen.py': {'Failed to persist render_remote_images':'_persist_console_toggle'},
}
fleet_old=('fleet wake drain intake failed','wake send gate raised; deferring','wake user-priority probe raised; deferring','wake delivery ledger stamp failed (exception_type=','wake delivery ledger stamp failed after dispose','wake mark listing failed','wake ledger read failed')
moves['tldw_chatbook/Chat/console_fleet_wake.py']={label:('tldw_chatbook/Chat/console_fleet_wake.py',new,('type(exc).__name__',)) for label,new in zip(fleet_old,('wake delivery failed','wake delivery failed','wake delivery failed','wake pause could not be saved','wake recovery failed','wake recovery failed','wake recovery failed'))}
lines=source.splitlines(True);edits=[];proof=[];new_entries={}
existing=ast.literal_eval(root)
for path_node,dict_node in zip(root.keys,root.values):
 path=ast.literal_eval(path_node);changes=moves.get(path,{})|retired.get(path,{})
 for key,value in zip(dict_node.keys,dict_node.values):
  label=ast.literal_eval(key)
  if label not in changes:continue
  frozen=subprocess.check_output(['git','show','f0ffcf9e819b577bd38c416f38550969c75fb5a0:'+path],text=True)
  assert label not in frozen, (path,label,'exists on frozen dev; needs separate mapping')
  edits.append((key.lineno-1,value.end_lineno,''))
  mapping=changes[label];proof.append({'old_owner':path,'old_label':label,'absent_on_frozen_dev':True,'current':mapping})
  if path in moves and label in moves[path]:
   owner,newlabel,fields=moves[path][label]
   if newlabel not in existing.get(owner,{}):new_entries.setdefault(owner,{})[newlabel]=fields
for a,b,repl in sorted(edits,reverse=True):lines[a:b]=[repl]
source=''.join(lines)
# Add explicit current-owner pins without altering the metadata/exception assertions.
append='\n# Current dev retired/moved these historical sinks before ADR-211 integration.\n'
for path,values in new_entries.items():append+='REVIEWED_METADATA_ONLY_DIAGNOSTICS.setdefault('+repr(path)+', {}).update('+repr(values)+')\n'
source=source.replace('\n\ndef test_reviewed_diagnostic_changes_are_metadata_only()',append+'\n\ndef test_reviewed_diagnostic_changes_are_metadata_only()',1)
# Sidebar writes now have two metadata-only catches with the same type-only receipt.
anchor='        "wake delivery UI hook raised": (1, ("type(exc).__name__",)),\n';assert source.count(anchor)==1;source=source.replace(anchor,'')
source += '''

# Retired log sites now use retained result/projection channels. Keep their
# absence explicit so a private exception sink cannot silently return.
@pytest.mark.parametrize("owner,function", [
    ("tldw_chatbook/Chat/console_fleet_wake.py", "_notify_ui"),
    ("tldw_chatbook/UI/MCP_Modules/mcp_workbench.py", "_observe_local_master_save"),
    ("tldw_chatbook/UI/MCP_Modules/mcp_workbench.py", "_observe_workspace_root_save"),
    ("tldw_chatbook/UI/Screens/settings_screen.py", "_persist_console_toggle"),
])
def test_retired_projection_failure_owners_have_no_diagnostic_sink(owner, function):
    tree = ast.parse((REPO_ROOT / owner).read_text())
    symbols = diagnostic_inventory._logger_symbols(tree)
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function]
    assert len(functions) == 1
    assert not [node for node in ast.walk(functions[0]) if isinstance(node, ast.Call) and diagnostic_inventory._is_diagnostic_call(node, symbols)]


def test_sidebar_failure_receipts_expose_only_exception_types():
    source = (REPO_ROOT / "tldw_chatbook/UI/Screens/chat_screen.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_persist_sidebar_state_off_loop")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "error"]
    assert len(calls) == 2
    assert all(ast.literal_eval(call.args[0]) == "Sidebar-state write failed: {}" for call in calls)
    assert all(ast.unparse(call.args[1]) == "self._sidebar_state_persistence_error" for call in calls)
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign) and any(ast.unparse(target) == "self._sidebar_state_persistence_error" for target in node.targets)]
    assert [ast.unparse(node.value) for node in assignments] == ["type(outcome.error).__name__ if outcome.error is not None else None", "type(error).__name__"]
'''
p.write_text(source)
Path('.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration/diagnostic-owner-mapping.json').write_text(json.dumps(proof,indent=2));print('Mapped',len(proof),'stale frozen-dev pins; no privacy assertions relaxed.')
