import ast,textwrap
from pathlib import Path
p=Path('tldw_chatbook/Chat/console_runtime.py');s=p.read_text();n=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name=='_schedule_legacy_trace_maintenance');lines=s.splitlines(keepends=True);chunk=textwrap.dedent(''.join(lines[n.lineno-1:n.end_lineno]));t=ast.parse(chunk).body[0];provider=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='provider_active');run=next(x for x in t.body if isinstance(x,ast.AsyncFunctionDef) and x.name=='run');start=next(x for x in run.body if isinstance(x,ast.ImportFrom) and x.module=='tldw_chatbook.Chat.console_trace_maintenance');cl=chunk.splitlines(keepends=True)
provider_src=textwrap.dedent(''.join(cl[provider.lineno-1:provider.end_lineno]));work_src=textwrap.dedent(''.join(cl[start.lineno-1:run.end_lineno]));helper='''async def run_legacy_trace_maintenance(self, database, normalizer_factory, *, _source):
'''+textwrap.indent(provider_src,'    ')+'\n'+textwrap.indent(work_src,'    ')
# Only actual global runtime dependencies are live-qualified; local imports stay local.
owned={'asyncio','time','logger','run_owned_db_call','LEGACY_TRACE_MAINTENANCE_RETRY_DELAY_SECONDS','LEGACY_TRACE_MAINTENANCE_PARK_POLL_SECONDS','TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS','TRACE_PHYSICAL_MAINTENANCE_RETRYABLE_REASONS','Mapping'};hnode=ast.parse(helper).body[0];edits={}
for stmt in hnode.body:
 for x in ast.walk(stmt):
  if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load) and x.id in owned:edits.setdefault(x.lineno-1,[]).append((x.col_offset,x.end_col_offset,'_source.'+x.id))
hl=helper.splitlines(keepends=True)
for row,es in edits.items():
 b=hl[row].encode()
 for a,z,v in sorted(set(es),reverse=True):b=b[:a]+v.encode()+b[z:]
 hl[row]=b.decode()
helper=''.join(hl);Path('tldw_chatbook/Chat/console_trace_maintenance_runtime.py').write_text('"""Post-readiness legacy trace maintenance on its existing runtime owner."""\n\nfrom __future__ import annotations\n\nfrom typing import Any\n\n\n'+helper)
# Keep readiness checks, delay, loop lookup and task registration in the original method.
cl[provider.lineno-1:provider.end_lineno]=['']*(provider.end_lineno-provider.lineno+1)
work_start=start.lineno-1;work_end=run.end_lineno
replacement='''        from .console_trace_maintenance_runtime import run_legacy_trace_maintenance

        await run_legacy_trace_maintenance(
            self, database, normalizer_factory, _source=_RUNTIME_MAINTENANCE_SOURCE
        )
''';cl[work_start:work_end]=[replacement]+['']*(work_end-work_start-1)
new_chunk=''.join(cl);new_chunk=new_chunk.replace('\n\n\n\n','\n\n');new_method=textwrap.indent(new_chunk,'    ')
s=''.join(lines[:n.lineno-1])+new_method+''.join(lines[n.end_lineno:]);s=s.replace('import time\n','import time\nimport sys as _maintenance_source_sys\n',1);ix=s.index('class ConsoleRuntime:');s=s[:ix]+'_RUNTIME_MAINTENANCE_SOURCE = _maintenance_source_sys.modules[__name__]\n\n\n'+s[ix:];p.write_text(s);ast.parse(s);print('Runtime',len(s.splitlines()))
