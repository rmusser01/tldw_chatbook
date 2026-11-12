import ast,textwrap
from pathlib import Path
p=Path('tldw_chatbook/Chat/console_provider_gateway.py');s=p.read_text();names=['_tool_definition_rejection_copy','_provider_error_copy_with_model_recovery','_safe_model_id'];moves=[]
owned={'_tool_definition_rejection_copy','_provider_error_copy_with_model_recovery','_safe_model_id','CredentialSanitizer','PROVIDER_ERROR_MODEL_ID_MAX_CHARS','escape_markup'}
for name in names:
 n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name);lines=s.splitlines(keepends=True);chunk=''.join(lines[n.lineno-1:n.end_lineno]);cn=ast.parse(chunk).body[0];edits={}
 for stmt in cn.body:
  for node in ast.walk(stmt):
   if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id in owned:edits.setdefault(node.lineno-1,[]).append((node.col_offset,node.end_col_offset,'_owner.'+node.id))
 cl=chunk.splitlines(keepends=True)
 for row,es in edits.items():
  data=cl[row].encode()
  for a,b,v in sorted(set(es),reverse=True):data=data[:a]+v.encode()+data[b:]
  cl[row]=data.decode()
 helper=''.join(cl)
 if helper.startswith('def '+name+'(\n'):helper=helper.replace('def '+name+'(\n','def '+name+'(\n    _owner: Any,\n',1)
 else:helper=helper.replace('def '+name+'(','def '+name+'(_owner: Any, ',1)
 moves.append(helper)
 sig=''.join(lines[n.lineno-1:n.body[0].lineno-1]);args=[a.arg for a in n.args.args]+[a.arg+'='+a.arg for a in n.args.kwonlyargs];call=',\n        '.join(['_PROVIDER_COPY_SOURCE',*args]);wrapper=sig+'''    """Delegate recovery copy while retaining the current defining dependencies."""
    from .refused_stream import '''+name+''' as render

    return render(
        '''+call+''',
    )
''';s=''.join(lines[:n.lineno-1])+wrapper+''.join(lines[n.end_lineno:])
s=s.replace('import os\n','import os\nimport sys as _copy_source_sys\n',1);anchor=s.index('def _tool_definition_rejection_copy(');s=s[:anchor]+'_PROVIDER_COPY_SOURCE = _copy_source_sys.modules[__name__]\n\n\n'+s[anchor:];p.write_text(s)
dst=Path('tldw_chatbook/Chat/refused_stream.py');ds=dst.read_text();ds=ds.replace('from typing import Any','from typing import Any\nfrom collections.abc import Mapping, Sequence',1);dst.write_text(ds.rstrip()+'\n\n\n'+'\n\n'.join(moves));ast.parse(s);ast.parse(dst.read_text());print('Gateway',len(s.splitlines()))
