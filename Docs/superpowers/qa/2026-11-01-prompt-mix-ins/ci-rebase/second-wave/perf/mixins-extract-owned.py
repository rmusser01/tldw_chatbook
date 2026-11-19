import ast,difflib,textwrap,builtins
from pathlib import Path
root=Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook')
paths={'composer':'tldw_chatbook/Widgets/Console/console_composer_bar.py','speech':'tldw_chatbook/Widgets/Console/console_speech_controls.py','api':'tldw_chatbook/LLM_Calls/LLM_API_Calls.py','protocol':'tldw_chatbook/LLM_Calls/openai_protocol.py'}
sources={k:(root/p).read_text() for k,p in paths.items()};originals={}
def locate(source,name):return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
def extract(source,name,newname,globals_,owner_import):
 n=locate(source,name);lines=source.splitlines(keepends=True);raw=textwrap.dedent(''.join(lines[n.lineno-1:n.end_lineno]));originals[name]=raw
 t=ast.parse(raw);f=t.body[0];edits=[];rlines=raw.splitlines(keepends=True);starts=[0]
 for line in rlines:starts.append(starts[-1]+len(line))
 def pos(line,col):return starts[line-1]+len(rlines[line-1].encode()[:col].decode())
 for item in ast.walk(f):
  if isinstance(item,ast.Name) and isinstance(item.ctx,ast.Load) and item.id in globals_:
   edits.append((pos(item.lineno,item.col_offset),pos(item.end_lineno,item.end_col_offset),'_owner.'+item.id))
 for a,b,value in sorted(edits,reverse=True):raw=raw[:a]+value+raw[b:]
 raw=raw.replace('def '+name+'(', 'def '+newname+'(',1)
 f=ast.parse(raw).body[0];insertion=f.body[0].end_lineno if isinstance(f.body[0],ast.Expr) and isinstance(f.body[0].value,ast.Constant) and isinstance(f.body[0].value.value,str) else f.body[0].lineno-1
 newlines=raw.splitlines(keepends=True);newlines.insert(insertion,'    '+owner_import+'\n\n');return ''.join(newlines)
voice_globals={'Horizontal','NoMatches','Static','Button','_load_console_voice_states','_swap_size_class','_clear_static_content','STATE_PREPARING','resolve_glyph','GLYPH_VOICE_RECORDING','resolve_glyph_text','cell_len','Content'}
voice_names={'_voice_chip_width_cap':'composer_voice_chip_width_cap','set_voice_status':'set_composer_voice_status','_sync_full_width_voice_presentation':'sync_full_width_composer_voice_presentation'}
helpers=[]
for name,newname in voice_names.items():helpers.append(extract(sources['composer'],name,newname,voice_globals,'from . import console_composer_bar as _owner'))
helpers=[h.replace('self,', 'self: ConsoleComposerBar,',1).replace('self)', 'self: ConsoleComposerBar)',1) for h in helpers]
sources['speech']=sources['speech'].replace('    from textual.app import RenderResult', '    from textual.app import RenderResult\n\n    from .console_composer_bar import ConsoleComposerBar')
sources['speech']=sources['speech'].replace('"""Console speech switches displayed beside the Workbench status."""','"""Console speech switches and composer voice-status presentation."""')+'\n\n'+'\n\n'.join(helpers)
wrappers={
'_voice_chip_width_cap':'''    def _voice_chip_width_cap(self) -> int:
        """Return the live row's voice-chip budget through its presentation helper."""
        from .console_speech_controls import composer_voice_chip_width_cap

        return composer_voice_chip_width_cap(self)
''',
'set_voice_status':'''    def set_voice_status(
        self,
        state: str,
        *,
        partial: str = "",
        elapsed_seconds: int = 0,
        message: str = "",
        segment_transcribing: bool = False,
    ) -> None:
        """Render voice status while retaining this composer's state and widgets."""
        from .console_speech_controls import set_composer_voice_status

        set_composer_voice_status(
            self, state, partial=partial, elapsed_seconds=elapsed_seconds,
            message=message, segment_transcribing=segment_transcribing,
        )
''',
'_sync_full_width_voice_presentation':'''    def _sync_full_width_voice_presentation(self, active: bool) -> None:
        """Make room for executor-wait copy through the speech presentation helper."""
        from .console_speech_controls import sync_full_width_composer_voice_presentation

        sync_full_width_composer_voice_presentation(self, active)
'''}
# Reverse source offsets, so each replacement preserves the remaining locations.
lines=sources['composer'].splitlines(keepends=True)
for name in sorted(voice_names,key=lambda n:locate(sources['composer'],n).lineno,reverse=True):
 n=locate(sources['composer'],name);lines[n.lineno-1:n.end_lineno]=[wrappers[name]]
sources['composer']=''.join(lines)
api_globals={'time','json','_session_ledger','requests','logger'};api_names=['_responses_stream_to_chat_sse','_openai_stream_line_usage'];helpers=[]
for name in api_names:helpers.append(extract(sources['api'],name,name,api_globals,'from . import LLM_API_Calls as _owner'))
sources['protocol']=sources['protocol'].replace('"""OpenAI function-call recognition and Responses payload projection."""','"""OpenAI function-call recognition, response projection and stream adapters."""')+'\n\n'+'\n\n'.join(helpers)
api_wrappers={
'_responses_stream_to_chat_sse':'''def _responses_stream_to_chat_sse(response, *, model: str):
    """Yield the legacy stream while forwarding close/throw to its adapter."""
    from .openai_protocol import _responses_stream_to_chat_sse

    yield from _responses_stream_to_chat_sse(response, model=model)
''',
'_openai_stream_line_usage':'''def _openai_stream_line_usage(line: str) -> dict | None:
    """Read usage through the existing OpenAI protocol adapter."""
    from .openai_protocol import _openai_stream_line_usage

    return _openai_stream_line_usage(line)
'''}
lines=sources['api'].splitlines(keepends=True)
for name in sorted(api_names,key=lambda n:locate(sources['api'],n).lineno,reverse=True):
 n=locate(sources['api'],name);lines[n.lineno-1:n.end_lineno]=[api_wrappers[name]]
sources['api']=''.join(lines)
output=['*** Begin Patch']
for key,p in paths.items():
 old=(root/p).read_text();new=sources[key];ast.parse(new);print(p,len(old.splitlines()),'->',len(new.splitlines()))
 output.append('*** Update File: '+str(root/p));output.extend(list(difflib.unified_diff(old.splitlines(),new.splitlines(),n=3,lineterm=''))[2:])
output.append('*** End Patch');Path('/private/tmp/mixins-extract-owned.patch').write_text('\n'.join(output)+'\n')
import json
Path('/private/tmp/mixins-extraction-original-functions.json').write_text(json.dumps(originals,indent=2))
