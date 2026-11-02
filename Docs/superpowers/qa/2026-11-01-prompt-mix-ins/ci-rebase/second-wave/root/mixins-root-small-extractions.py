import ast,textwrap,json
from pathlib import Path
root=Path.cwd(); proof=[]
def locate(path,name):
 p=root/path;s=p.read_text();ns=[n for n in ast.walk(ast.parse(s)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name];assert len(ns)==1,(path,name);n=ns[0];lines=s.splitlines(keepends=True);start=min([n.lineno,*[d.lineno for d in n.decorator_list]])-1;return p,s,n,lines,start

def replace(path,name,replacement):
 p,s,n,lines,start=locate(path,name);old=''.join(lines[start:n.end_lineno]);p.write_text(''.join(lines[:start])+replacement.rstrip()+'\n'+''.join(lines[n.end_lineno:]));return textwrap.dedent(old),n

def append(path,content):
 p=root/path;p.write_text(p.read_text().rstrip()+'\n\n\n'+content.rstrip()+'\n')
# SQL splitting belongs to the already-loaded SQL validation module.
p,s,n,lines,start=locate('tldw_chatbook/DB/ChaChaNotes_DB.py','_split_sql_statements');chunk=textwrap.dedent(''.join(lines[start:n.end_lineno]));helper=chunk.replace('def _split_sql_statements(script: str) -> List[str]:','def split_migration_statements(script: str, *, error_type: type[Exception]) -> list[str]:').replace('statements: List[str]', 'statements: list[str]').replace('raise SchemaError(', 'raise error_type(').replace('SchemaError: If', 'Exception: The supplied error type if');helper=helper.replace('    Uses ``sqlite3.complete_statement``','    import sqlite3\n\n    Uses ``sqlite3.complete_statement``') if False else helper
# Import sqlite3 inside this pure helper without changing error identity.
body=ast.parse(helper).body[0];ls=helper.splitlines(keepends=True);doc_end=body.body[0].end_lineno;ls[doc_end:doc_end]=['\n','    import sqlite3\n'];helper=''.join(ls);append('tldw_chatbook/DB/sql_validation.py',helper)
replace('tldw_chatbook/DB/ChaChaNotes_DB.py','_split_sql_statements','''def _split_sql_statements(script: str) -> List[str]:
    from .sql_validation import split_migration_statements

    return split_migration_statements(script, error_type=SchemaError)''')
proof.append({'move':'migration splitter','source':'DB/ChaChaNotes_DB.py','destination':'DB/sql_validation.py','invariant':'Same parser and original SchemaError supplied by caller.'})
# Notes startup helpers retain old lazy entry points, with all rationale moved intact.
notes=[]
for name in ['_task_ended_without_result','_owner_still_starting','_await_notes_sync_start']:
 p,s,n,lines,start=locate('tldw_chatbook/app_service_wiring.py',name);chunk=textwrap.dedent(''.join(lines[start:n.end_lineno]));notes.append(chunk)
 call=('await ' if isinstance(n,ast.AsyncFunctionDef) else '')+name+'('+('pending' if name=='_await_notes_sync_start' else 'owner' if name=='_owner_still_starting' else 'task')+')'
 signature=chunk[:chunk.index(':\n')+2];replacement=signature+'    from .Notes.notes_sync_startup import '+name+' as delegated\n\n    return '+call.replace(name+'(', 'delegated(')+'\n';replace('tldw_chatbook/app_service_wiring.py',name,replacement)
(root/'tldw_chatbook/Notes/notes_sync_startup.py').write_text('"""Small task-outcome helpers for the existing Notes startup owner."""\n\nfrom __future__ import annotations\n\nimport asyncio\nfrom typing import Any\n\n\n'+'\n\n'.join(notes))
# Bindings projection reuses the existing visible-help owner; callbacks stay current.
p,s,n,lines,start=locate('tldw_chatbook/app_command_providers.py','_bindings_to_shortcuts');chunk=''.join(lines[start:n.end_lineno]);helper=chunk.replace('def _bindings_to_shortcuts(bindings: Any)', 'def binding_shortcuts(bindings: Any, *, display_key, binding_type)').replace('isinstance(entry, Binding)', 'isinstance(entry, binding_type)').replace('_display_key(', 'display_key(');help_path=root/'tldw_chatbook/UI/Workbench/help.py';hs=help_path.read_text().replace('from dataclasses import dataclass','from dataclasses import dataclass\nfrom typing import Any');help_path.write_text(hs);append('tldw_chatbook/UI/Workbench/help.py',helper)
replace('tldw_chatbook/app_command_providers.py','_bindings_to_shortcuts','''def _bindings_to_shortcuts(bindings: Any) -> tuple[tuple[str, str], ...]:
    """Project current binding classes and glyphs through the help renderer."""
    from .UI.Workbench.help import binding_shortcuts

    return binding_shortcuts(bindings, display_key=_display_key, binding_type=Binding)''')
# Fallback readiness remains a current config read at invocation.
p,s,n,lines,start=locate('tldw_chatbook/Agents/agent_service.py','_provider_is_ready');chunk=textwrap.dedent(''.join(lines[start:n.end_lineno]));helper=chunk.replace('def _provider_is_ready(self, provider: str)', 'def configured_provider_is_ready(provider: str)').replace('    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness\n','');append('tldw_chatbook/Chat/provider_readiness.py',helper)
replace('tldw_chatbook/Agents/agent_service.py','_provider_is_ready','''    def _provider_is_ready(self, provider: str) -> bool:
        """Read current fallback readiness through the shared provider contract."""
        from tldw_chatbook.Chat.provider_readiness import configured_provider_is_ready

        return configured_provider_is_ready(provider)''')
# Warn only after the original optional-failure gate; healthy boot imports nothing.
warning_helpers=[]
for old,new,arg,field,gate in [('_maybe_warn_config_load_failure','notify_config_load_failure','failure','_config_load_failure','failure is None'),('_maybe_warn_config_schema_conflict','notify_config_schema_conflict','conflict','_config_schema_conflict','not conflict')]:
 p,s,n,lines,start=locate('tldw_chatbook/app.py',old);chunk=textwrap.dedent(''.join(lines[start:n.end_lineno]));tree=ast.parse(chunk).body[0];doc=ast.get_source_segment(chunk,tree.body[0]);notify=ast.get_source_segment(chunk,tree.body[-1]);warning_helpers.append('def '+new+'(self, '+arg+': Any) -> None:\n'+textwrap.indent(doc,'    ')+'\n'+textwrap.indent(textwrap.dedent(notify),'    ')+'\n')
 replace('tldw_chatbook/app.py',old,'    def '+old+'(self) -> None:\n        '+arg+' = getattr(self, "'+field+'", None)\n        if '+gate+':\n            return\n        from .app_config_notifications import '+new+'\n\n        '+new+'(self, '+arg+')')
(root/'tldw_chatbook/app_config_notifications.py').write_text('"""Existing config warning presentation, loaded only for a detected warning."""\n\nfrom __future__ import annotations\n\nfrom typing import Any\n\n\n'+'\n\n'.join(warning_helpers))
for path in ['tldw_chatbook/DB/ChaChaNotes_DB.py','tldw_chatbook/app.py','tldw_chatbook/app_service_wiring.py','tldw_chatbook/app_command_providers.py','tldw_chatbook/Agents/agent_service.py']:
 ast.parse((root/path).read_text());print(path,len((root/path).read_text().splitlines()))
Path('/private/tmp/mixins-root-small-extractions-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
