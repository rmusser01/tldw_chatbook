"""Proposal-only adapter placement and named wiring; no production imports."""
import ast
import json
import subprocess
from pathlib import Path

P=Path(__file__).parent
RUFF='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff'
s=(P/'task-9-scope-proof3-screen-union.py').read_text(); lines=s.splitlines(keepends=True)
c=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='ChatScreen')
ns={n.name:n for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
selected=['_dispatch_console_trace_recovery','_console_trace_recovery_state']
def fmt(text,name):
 out=subprocess.run([RUFF,'format','--stdin-filename',str(P/name),'-'],input=text,capture_output=True,text=True,check=True).stdout
 ast.parse(out);compile(out,name,'exec');(P/name).write_text(out);return out
bodies=[]
for name in selected:
 n=ns[name]; raw=''.join(lines[n.lineno-1:n.end_lineno])
 raw=raw.replace('await dispatch_trace_call_recovery_action(', 'await self._read_trace_recovery_dispatch()(').replace('return trace_call_recovery_state(', 'return self._read_trace_recovery_state()(').replace('on_started=self._start_console_transcript_sync_timer,','on_started=self._read_trace_recovery_started(),').replace('on_finished=self._sync_native_console_chat_ui,','on_finished=self._read_trace_recovery_finished(),')
 bodies.append(raw)
owner=fmt('''from __future__ import annotations
from typing import Any, Callable

class ConsoleSessionController:
    def __init__(
        self,
        *,
        read_trace_recovery_dispatch: Callable[[], Callable[..., Any]],
        read_trace_recovery_state: Callable[[], Callable[..., Any]],
        read_trace_recovery_started: Callable[[], Callable[..., Any]],
        read_trace_recovery_finished: Callable[[], Callable[..., Any]],
    ) -> None:
        """Additional parameters and assignments in the existing constructor."""
        self._read_trace_recovery_dispatch = read_trace_recovery_dispatch
        self._read_trace_recovery_state = read_trace_recovery_state
        self._read_trace_recovery_started = read_trace_recovery_started
        self._read_trace_recovery_finished = read_trace_recovery_finished

'''+ '\n'.join(bodies),'task-9-scope-proof3-screen-session.py')
wiring=fmt('''from typing import Any, Callable

def build_console_controllers(
    screen: Any,
    *,
    read_trace_recovery_dispatch: Callable[[], Callable[..., Any]],
    read_trace_recovery_state: Callable[[], Callable[..., Any]],
) -> None:
    """Add these two parameters and four arguments to the existing wiring."""
    screen._session = ConsoleSessionController(
        screen,
        read_trace_recovery_dispatch=read_trace_recovery_dispatch,
        read_trace_recovery_state=read_trace_recovery_state,
        read_trace_recovery_started=lambda: screen._start_console_transcript_sync_timer,
        read_trace_recovery_finished=lambda: screen._sync_native_console_chat_ui,
    )
''','task-9-scope-proof3-screen-wiring.py')
init=ns['__init__']; call=next(n for n in ast.walk(init) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='build_console_controllers')
old=''.join(lines[call.lineno-1:call.end_lineno]); added='            read_trace_recovery_dispatch=lambda: dispatch_trace_call_recovery_action,\n            read_trace_recovery_state=lambda: trace_call_recovery_state,\n'
new=old.rsplit('        )',1)[0]+added+'        )\n'
# Format only the actual fragment, preserve surrounding source/docstrings.
frag=fmt('from __future__ import annotations\n\nclass ChatScreen:\n    def __init__(self):\n'+new,'task-9-scope-proof3-screen.py')
finit=next(n for n in ast.walk(ast.parse(frag)) if isinstance(n,ast.FunctionDef)); fcall=finit.body[0]; flines=frag.splitlines(keepends=True); replacement=''.join(flines[fcall.lineno-1:fcall.end_lineno])
edits=[(call.lineno-1,call.end_lineno,replacement)]
for name in selected:
 n=ns[name];edits.append((n.lineno-1,n.end_lineno,''))
for a,b,v in sorted(edits,reverse=True):lines[a:b]=[v] if v else []
projection=''.join(lines)
for name in selected:projection=projection.replace('self.'+name,'self._session.'+name)
ast.parse(projection);compile(projection,'screen projection','exec')
(P/'task-9-scope-proof3-screen-projected.py').write_text(projection)
c2=next(n for n in ast.parse(projection).body if isinstance(n,ast.ClassDef) and n.name=='ChatScreen')
ownerclass=next(n for n in ast.parse(owner).body if isinstance(n,ast.ClassDef));ownerdefs={n.name:n for n in ownerclass.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for name in selected:
 assert ast.dump(ownerdefs[name].args)==ast.dump(ns[name].args)
 assert ast.get_docstring(ownerdefs[name],clean=False)==ast.get_docstring(ns[name],clean=False)
result={'selected':[{'name':name,'span':[ns[name].lineno,ns[name].end_lineno],'removed_lines':ns[name].end_lineno-ns[name].lineno+1} for name in selected],'base_lines':len(s.splitlines()),'base_methods':len(ns),'construction_delta':len(replacement.splitlines())-len(old.splitlines()),'projected_lines':len(projection.splitlines()),'projected_methods':len([n for n in c2.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]),'session_body_lines':sum(ownerdefs[n].end_lineno-ownerdefs[n].lineno+1 for n in selected),'session_constructor_parameter_additions':4,'session_constructor_assignment_additions':4,'wiring_new_build_parameters':2,'wiring_new_session_keywords':4,'wiring_fragment_lines':len(wiring.splitlines()),'fixture_adaptations':['Tests/Chat/test_console_video_actions.py:96','Tests/UI/test_console_auto_speak_wiring.py:160'],'callback_identity_change':'The two region callbacks bind to screen._session instead of screen; on_started/on_finished still receive the screen methods read at dispatch time.'}
(P/'task-9-scope-proof3-screen-measure.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
