import ast,copy,json,subprocess,tokenize,io
from pathlib import Path
root=Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook')
original=json.load(open('/private/tmp/mixins-extraction-original-functions.json'))
mapping={
 '_voice_chip_width_cap':('tldw_chatbook/Widgets/Console/console_speech_controls.py','composer_voice_chip_width_cap'),
 'set_voice_status':('tldw_chatbook/Widgets/Console/console_speech_controls.py','set_composer_voice_status'),
 '_sync_full_width_voice_presentation':('tldw_chatbook/Widgets/Console/console_speech_controls.py','sync_full_width_composer_voice_presentation'),
 '_responses_stream_to_chat_sse':('tldw_chatbook/LLM_Calls/openai_protocol.py','_responses_stream_to_chat_sse'),
 '_openai_stream_line_usage':('tldw_chatbook/LLM_Calls/openai_protocol.py','_openai_stream_line_usage'),
}
class Normalize(ast.NodeTransformer):
 def visit_Attribute(self,n):
  self.generic_visit(n)
  if isinstance(n.value,ast.Name) and n.value.id=='_owner': return ast.Name(id=n.attr,ctx=n.ctx)
  return n
 def visit_ImportFrom(self,n):
  if any(x.asname=='_owner' for x in n.names): return None
  return n
rows=[]
for name,(p,newname) in mapping.items():
 src=(root/p).read_text();node=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,ast.FunctionDef) and n.name==newname)
 old=ast.parse(original[name]).body[0];new=Normalize().visit(copy.deepcopy(node));new.name=name
 if name in {'_voice_chip_width_cap','set_voice_status','_sync_full_width_voice_presentation'}:new.args.args[0].annotation=None
 equal=ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)
 assert equal,name
 newtext='\n'.join(src.splitlines()[node.lineno-1:node.end_lineno])
 def comments(s):return [t.string for t in tokenize.generate_tokens(io.StringIO(s).readline) if t.type==tokenize.COMMENT]
 assert comments(original[name])==comments(newtext),name+' comments'
 rows.append({'function':name,'body_arguments_return_docstring_ast_equivalent_after_owner_qualification':True,'comments_preserved':True})
for p,names in [('tldw_chatbook/Widgets/Console/console_composer_bar.py',set(list(mapping)[:3])),('tldw_chatbook/LLM_Calls/LLM_API_Calls.py',set(list(mapping)[3:]))]:
 old=subprocess.check_output(['git','show','ba1cafc55dd5ac9b6b1f066c936206e652ed6b65:'+p],cwd=root,text=True);new=(root/p).read_text()
 class RemoveMoved(ast.NodeTransformer):
  def visit_FunctionDef(self,n):
   if n.name in names:return None
   return self.generic_visit(n)
 assert ast.dump(RemoveMoved().visit(ast.parse(old)),include_attributes=False)==ast.dump(RemoveMoved().visit(ast.parse(new)),include_attributes=False),p
 rows.append({'owner':p,'all_other_declarations_and_bodies_ast_identical':True})
Path('/private/tmp/mixins-extraction-ast-proof.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
