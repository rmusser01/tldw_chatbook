import ast,copy,hashlib,json,subprocess
from pathlib import Path
WT=Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook');SDD=WT/'.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge';EV=SDD/'task-18-safe-evidence';BASE='3238094c92e77248483ac95254b877639c842c9b'
def old(p):return subprocess.check_output(['git','show',f'{BASE}:{p}'],cwd=WT,text=True)
def current(p):return (WT/p).read_text()
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def dump(node):return ast.dump(node,include_attributes=False)
def cls(tree,name):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
def meth(c,name):return next(n for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
notes={}
clock='Tests/Chat/test_console_decision_clock.py';b=old(clock);n=current(clock)
new='from Tests.Chat.console_interrupt_test_bindings import (\n    make_interrupt_host as InterruptRoundHost,\n)\nfrom Tests.Chat.test_console_interrupt_rounds import FakeSeamsFull\nfrom tldw_chatbook.Chat.console_interrupt_rounds import KIND_SETTER_ATTRS'
original='from Tests.Chat.test_console_interrupt_rounds import FakeSeamsFull\nfrom tldw_chatbook.Chat.console_interrupt_rounds import (\n    KIND_SETTER_ATTRS,\n    InterruptRoundHost,\n)'
assert n.count(new)==1 and n.replace(new,original)==b
notes[clock]={'entire_source_import_reversal':True,'original_sha256':sha(b),'current_sha256':sha(n)}
buddy='Tests/UI/test_buddy_speech.py';b=old(buddy);n=current(buddy)
new='from Tests.Chat.console_interrupt_test_bindings import (\n    make_interrupt_host as InterruptRoundHost,\n)';original='from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost'
assert n.count(new)==1 and n.count('pytestmark = pytest.mark.bootstrap_profile\n\n')==1
assert n.replace(new,original).replace('pytestmark = pytest.mark.bootstrap_profile\n\n','')==b
notes[buddy]={'entire_source_import_and_marker_reversal':True,'original_sha256':sha(b),'current_sha256':sha(n)}
stop='Tests/UI/test_console_runtime_ownership.py';b=old(stop);n=current(stop);added='                await _wait_for_selector(\n                    chat, pilot, "#console-stop-generation.console-stop-active"\n                )\n'
assert n.count(added)==1 and n.replace(added,'')==b
notes[stop]={'entire_source_single_wait_reversal':True,'original_sha256':sha(b),'current_sha256':sha(n),'helper_unchanged_bound_seconds':2.0}
start='tldw_chatbook/Chat/console_chat_start.py';b=ast.parse(old(start));n=ast.parse(current(start));bm=meth(cls(b,'ConsoleChatStartCoordinator'),'authorizes');nm=meth(cls(n,'ConsoleChatStartCoordinator'),'authorizes');assert ast.get_docstring(bm) is None;doc=nm.body.pop(0);assert isinstance(doc,ast.Expr) and isinstance(doc.value,ast.Constant) and isinstance(doc.value.value,str);assert dump(b)==dump(n)
notes[start]={'full_module_docstring_reversal_AST_equal':True,'authorizes_executable_AST_sha256':sha(dump(bm))}
host='tldw_chatbook/Chat/console_interrupt_rounds.py';btext=old(host);ntext=current(host);bt=ast.parse(btext);nt=ast.parse(ntext);bc=cls(bt,'InterruptRoundHost');nc=cls(nt,'InterruptRoundHost');bi=meth(bc,'__init__');ni=meth(nc,'__init__');removed={'read_global_Any','read_global_Mapping'}
assert len(bi.args.kwonlyargs)==122 and len(ni.args.kwonlyargs)==120
assert all(x is None for x in ni.args.kw_defaults)
assert [a.arg for a in ni.args.kwonlyargs]==[a.arg for a in bi.args.kwonlyargs if a.arg not in removed]
assert sum(a.arg.startswith('read_controller_') for a in ni.args.kwonlyargs)==86
assert sum(a.arg.startswith('read_global_') for a in ni.args.kwonlyargs)==33
assert sum(a.arg.startswith('write_controller_') for a in ni.args.kwonlyargs)==1
refs=[x for x in ast.walk(bc) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='self' and x.func.attr in removed]
assert len(refs)==17
class StripAnnotations(ast.NodeTransformer):
 def visit_arg(self,node):node.annotation=None;return self.generic_visit(node)
 def visit_FunctionDef(self,node):node.returns=None;return self.generic_visit(node)
 visit_AsyncFunctionDef=visit_FunctionDef
 def visit_AnnAssign(self,node):node.annotation=ast.Constant(value=None);return self.generic_visit(node)
methods={};seen=[]
for bm in bc.body:
 if not isinstance(bm,(ast.FunctionDef,ast.AsyncFunctionDef)) or bm.name=='__init__':continue
 nm=meth(nc,bm.name);bd=dump(StripAnnotations().visit(copy.deepcopy(bm)));nd=dump(StripAnnotations().visit(copy.deepcopy(nm)));assert bd==nd,bm.name
 methods[bm.name]=sha(bd)
# Verify exact selected full-module edits, including conventional annotation reflow.
expected=btext.replace('self.read_global_Any()','Any').replace('self.read_global_Mapping()','Mapping')
for field in ['Any','Mapping']:
 expected=expected.replace(f'        read_global_{field}: Callable[[], Any],\n','').replace(f'        self.read_global_{field} = read_global_{field}\n','')
expected=expected.replace('                        nodes: dict[\n                            str, Mapping[str, Any]\n                        ] = {}','                        nodes: dict[str, Mapping[str, Any]] = {}').replace('                        def _walk(\n                            node: Mapping[\n                                str, Any\n                            ],\n                        ) -> None:','                        def _walk(\n                            node: Mapping[str, Any],\n                        ) -> None:')
assert expected==ntext
# Separate build_tool_review_hook remains exact AST.
bhook=next(x for x in bt.body if isinstance(x,ast.FunctionDef) and x.name=='build_tool_review_hook');nhook=next(x for x in nt.body if isinstance(x,ast.FunctionDef) and x.name=='build_tool_review_hook');assert dump(bhook)==dump(nhook)
notes[host]={'exact_selected_source_equal':True,'original_annotation_calls':17,'required_keyword_only_bindings':120,'controller_readers':86,'runtime_global_readers':33,'write_callback':1,'nonconstructor_executable_methods':methods,'separate_review_hook_AST_unchanged':True,'original_lines':len(btext.splitlines()),'current_lines':len(ntext.splitlines())}
for p in ['tldw_chatbook/Chat/console_chat_controller.py','Tests/Chat/console_interrupt_test_bindings.py']:
 bt=ast.parse(old(p));nt=ast.parse(current(p));calls=[x for x in ast.walk(bt) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='InterruptRoundHost'];assert len(calls)==1;call=calls[0];assert sum(k.arg in removed for k in call.keywords)==2;call.keywords=[k for k in call.keywords if k.arg not in removed];assert dump(bt)==dump(nt)
 notes[p]={'full_module_two_host_keyword_reversal_AST_equal':True,'original_lines':len(old(p).splitlines()),'current_lines':len(current(p).splitlines())}
# Exact downward cap changes only.
p='Tests/Architecture/test_module_size_ratchet.py';b=old(p);n=current(p);expected=b.replace('"tldw_chatbook/Chat/console_chat_controller.py": 29301','"tldw_chatbook/Chat/console_chat_controller.py": 29299').replace('"tldw_chatbook/Chat/console_interrupt_rounds.py": 6479','"tldw_chatbook/Chat/console_interrupt_rounds.py": 6471');assert expected==n
notes[p]={'exact_two_demonstrated_downward_cap_changes':True,'controller_old':29301,'controller_new':29299,'host_old':6479,'host_new':6471,'slack_tolerance_unchanged':50}
(EV/'source-AST-reversals.json').write_text(json.dumps({'base':BASE,'owners':notes},indent=2)+'\n')
print('PASS: all exact source/body/AST reversals;120keywords=86+33+1;17 annotation uses; separate APIs unchanged; caps29299/6471.')
