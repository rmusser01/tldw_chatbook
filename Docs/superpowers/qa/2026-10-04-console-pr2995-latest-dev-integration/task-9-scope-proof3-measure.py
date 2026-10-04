"""Proposal-only generator; parses source and formats text, never imports production."""
import ast
import builtins
import copy
import json
import subprocess
import symtable
from pathlib import Path

P = Path(__file__).parent
RUFF = '/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff'
source = (P / 'task-9-scope-proof3-controller-union.py').read_text()
tree = ast.parse(source)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ConsoleChatController')
methods = {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
selected = ['compact_context_now', '_apply_conversation_memory_preflight', '_assess_context_compaction', '_assess_request_capacity_only', '_context_overflow_alert']
st = next(t for t in symtable.symtable(source, 'union', 'exec').get_children() if t.get_name() == cls.name)
sts = {t.get_name(): t for t in st.get_children()}
def globals_for(table):
    result = {s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced() and s.get_name() not in vars(builtins)}
    for child in table.get_children():
        result.update(globals_for(child))
    return result

def fmt(text, name):
    out = subprocess.run([RUFF, 'format', '--stdin-filename', str(P / name), '-'], input=text, capture_output=True, text=True, check=True).stdout
    ast.parse(out)
    compile(out, name, 'exec')
    (P / name).write_text(out)
    return out

def start(n):
    return min([n.lineno] + [d.lineno for d in n.decorator_list])

def attrs_for(n):
    return {x.attr for x in ast.walk(n) if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == 'self'}

attrs = sorted(set().union(*(attrs_for(methods[n]) for n in selected)))
globs = sorted(set().union(*(globals_for(sts[n]) for n in selected)))
# Full docs and comments stay in owner bodies. Changes are explicit self/global reads.
lines = source.splitlines(keepends=True)
offsets = [0]
for line in lines:
    offsets.append(offsets[-1] + len(line))
def offset(n, end=False):
    line = n.end_lineno if end else n.lineno
    col = n.end_col_offset if end else n.col_offset
    return offsets[line-1] + len(lines[line-1].encode()[:col].decode())
body_map=[]
owner_methods=[]
for name in selected:
    n=methods[name]
    begin=offsets[n.lineno-1]; end=offsets[n.end_lineno]
    text=source[begin:end]
    edits=[]
    for statement in n.body:
        for node in ast.walk(statement):
            if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='self':
                assert isinstance(node.ctx,ast.Load)
                edits.append((offset(node)-begin,offset(node,True)-begin,'self.read_controller_'+node.attr+'()'))
            elif isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id in globals_for(sts[name]):
                edits.append((offset(node)-begin,offset(node,True)-begin,'self.read_global_'+node.id+'()'))
    for a,b,value in sorted(edits,reverse=True):
        text=text[:a]+value+text[b:]
    owner_methods.append(text)
    body_map.append({'name':name,'span':[start(n),n.end_lineno],'lines':n.end_lineno-start(n)+1,'controller_reads':sorted(attrs_for(n)),'global_reads':sorted(globals_for(sts[name])),'binding_replacements':len(edits)})

constructor=['    def __init__(self):','        """Proposed insertion into the existing controller constructor."""','        from tldw_chatbook.Chat.console_context_compaction import ConsoleCompactionPreflight','','        self._compaction_preflight = ConsoleCompactionPreflight(']
constructor += ['            read_controller_'+a+'=lambda: self.'+a+',' for a in attrs]
constructor += ['            read_global_'+g+'=lambda: '+g+',' for g in globs]
constructor += ['        )']
wrappers=[]
for name in selected:
    original=methods[name]; n=copy.deepcopy(original)
    args=[ast.Name(a.arg,ast.Load()) for a in n.args.posonlyargs+n.args.args if a.arg!='self']
    kws=[ast.keyword(a.arg,ast.Name(a.arg,ast.Load())) for a in n.args.kwonlyargs]
    assert n.args.vararg is None and n.args.kwarg is None
    call=ast.Call(ast.Attribute(ast.Attribute(ast.Name('self',ast.Load()),'_compaction_preflight',ast.Load()),name,ast.Load()),args,kws)
    if isinstance(n,ast.AsyncFunctionDef): call=ast.Await(call)
    n.body=[ast.Expr(ast.Constant('Forward to the documented compaction preflight implementation.')),ast.Return(call)]
    wrappers.append(ast.unparse(ast.fix_missing_locations(n)))
controller=fmt('from __future__ import annotations\n\nclass ConsoleChatController:\n'+'\n'.join(constructor)+'\n\n'+'\n\n'.join('\n'.join('    '+line if line else '' for line in w.splitlines()) for w in wrappers)+'\n','task-9-scope-proof3-compaction-controller.py')
ct=ast.parse(controller).body[1]; init=ct.body[0]
wraplines=sum(n.end_lineno-start(n)+1 for n in ct.body[1:]); constructlines=init.end_lineno-init.lineno-1
owner=['from __future__ import annotations','from typing import Any, Callable','','class ConsoleCompactionPreflight:','    """Proposed policy/preflight owner; every live dependency is named."""','    def __init__(','        self,','        *,']
owner += ['        read_controller_'+a+': Callable[[], Any],' for a in attrs]
owner += ['        read_global_'+g+': Callable[[], Any],' for g in globs]
owner += ['    ) -> None:']
owner += ['        self.read_controller_'+a+' = read_controller_'+a for a in attrs]
owner += ['        self.read_global_'+g+' = read_global_'+g for g in globs]
ownertext=fmt('\n'.join(owner)+'\n\n'+'\n'.join(owner_methods),'task-9-scope-proof3-compaction-owner.py')
# No static classification or default evaluation moves: wrappers retain original arguments.
ownercls=next(n for n in ast.parse(ownertext).body if isinstance(n,ast.ClassDef))
owner_defs={n.name:n for n in ownercls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for n in ct.body[1:]:
    assert ast.dump(n.args)==ast.dump(methods[n.name].args)
    assert ast.get_docstring(owner_defs[n.name],clean=False)==ast.get_docstring(methods[n.name],clean=False)
removed=sum(x['lines'] for x in body_map)
result={'selected':body_map,'named_controller_reads':attrs,'named_global_reads':globs,'removed_ast_lines':removed,'formatted_wrapper_ast_lines':wraplines,'formatted_construction_lines':constructlines,'projected_controller_lines':29983-removed+wraplines+constructlines,'headroom':29367-(29983-removed+wraplines+constructlines),'owner_class_lines':ownercls.end_lineno-ownercls.lineno+1,'owner_snippet_lines':len(ownertext.splitlines()),'global_local_shadow_conflicts':{}}
for name in selected:
    local=set()
    def visit(t):
        local.update(x.get_name() for x in t.get_symbols() if x.is_local())
        for child in t.get_children():visit(child)
    visit(sts[name]); result['global_local_shadow_conflicts'][name]=sorted(local & globals_for(sts[name]))
(P/'task-9-scope-proof3-compaction-measure.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['selected','named_controller_reads','named_global_reads']},indent=2))
