import ast,subprocess,difflib
from pathlib import Path
root=Path.cwd();ruff='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff'
files={
'Tests/Chat/test_openai_streaming_usage.py':{'test_responses_request_preserves_native_headers_and_wire_payload'},
'Tests/UI/test_console_idle_voice_boot.py':{'test_changed_extracted_idle_voice_source_keeps_original_callback'},
'Tests/UI/test_console_first_send_preimport.py':{'test_completion_callback_resolves_current_attachment_on_ui_loop'},
'Tests/UI/test_console_idle_heap_refreeze.py':{'test_mounted_received_custody_blocks_completion_and_idle_timer_reclaims'},
'tldw_chatbook/UI/Console_Modules/composer_run_controls.py':{'sync_dictation_presentation'},
'tldw_chatbook/Widgets/Console/console_speech_controls.py':{'_IDLE_COMPOSER_PRESENTATION_CALLBACKS'},
}
out=['*** Begin Patch']
for p,names in files.items():
 old=(root/p).read_text();new=old;nodes=[n for n in ast.walk(ast.parse(old)) if (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
 for n in sorted(nodes,key=lambda n:n.lineno,reverse=True):
  r=subprocess.run([ruff,'format','--no-cache','--stdin-filename',p,f'--range={n.lineno}-{n.end_lineno+1}','-'],input=new,text=True,capture_output=True,check=True);new=r.stdout
 if new!=old:
  out.append('*** Update File: '+str(root/p));out.extend(list(difflib.unified_diff(old.splitlines(),new.splitlines(),n=3,lineterm=''))[2:])
out.append('*** End Patch');Path('/private/tmp/mixins-extraction-finish-format.patch').write_text('\n'.join(out)+'\n')
