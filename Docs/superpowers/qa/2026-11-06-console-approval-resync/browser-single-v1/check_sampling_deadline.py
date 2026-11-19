"""Actual sampling/failed-send timeout shells; stdlib, no App/CDP/product imports."""
import argparse, ast, asyncio, copy, json, types
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
output=a.output_root.resolve()
if not output.is_relative_to(a.source_root.resolve()/'.superpowers'):raise RuntimeError('Owned_output_required')
output.mkdir(parents=True,exist_ok=True)
tree=ast.parse((Path(__file__).resolve().parent/'web_smoke.py').read_text(encoding='utf-8'))
def timeout_cap(n,cap):
 return isinstance(n,ast.AsyncWith) and len(n.items)==1 and isinstance(n.items[0].context_expr,ast.Call) and isinstance(n.items[0].context_expr.func,ast.Attribute) and n.items[0].context_expr.func.attr=='timeout' and len(n.items[0].context_expr.args)==1 and isinstance(n.items[0].context_expr.args[0],ast.Constant) and n.items[0].context_expr.args[0].value==cap
sampling=next((n for n in ast.walk(tree) if timeout_cap(n,600)),None)
failed=next((n for n in ast.walk(tree) if timeout_cap(n,30)),None)
final_check=next((n for n in (sampling.body if sampling else []) if isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value=='Actual_sampling_safety600_after_reply' for x in ast.walk(n))),None)
class Asyncio:
 @staticmethod
 def timeout(cap):
  assert cap in (30,600);return asyncio.timeout(.01)
 @staticmethod
 async def wait_for(awaitable,cap):return await asyncio.wait_for(awaitable,.01)
async def exercise(shell,cap,late=False):
 calls=[]
 async def delegate():
  calls.append(1)
  if not late:await asyncio.Future()
  return 'original-result'
 payload=ast.Expr(value=ast.Await(value=ast.Call(func=ast.Name(id='delegate',ctx=ast.Load()),args=[],keywords=[])))
 body=[payload]
 if cap==600 and final_check is not None:body.append(copy.deepcopy(final_check))
 if shell is not None:
  context=copy.deepcopy(shell);context.body=body;body=[context]
 node=ast.AsyncFunctionDef(name='actual',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
 scope={'asyncio':Asyncio,'delegate':delegate,'time':types.SimpleNamespace(monotonic=lambda:601 if late else 0),'sampling_started':0}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual-parent-timeout-shell','exec'),scope)
 task=asyncio.create_task(scope['actual']());done,_=await asyncio.wait({task},timeout=.04)
 if not done:
  task.cancel()
  try:await task
  except asyncio.CancelledError:pass
  return False
 try:task.result()
 except TimeoutError:return calls==[1]
 return False
async def main():
 results={'pending_CDP_sampling_bounded':await exercise(sampling,600),'late_CDP_sampling_refused':await exercise(sampling,600,True),'pending_failed_Tracing_end_bounded':await exercise(failed,30)}
 if sampling is not None:
  assert any(isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='round_index' for n in ast.walk(sampling))
 if failed is not None:
  sends=[n for n in ast.walk(failed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='send' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='Tracing.end']
  assert len(sends)==1 and any(isinstance(n,ast.Await) and isinstance(n.value,ast.Attribute) and n.value.attr=='complete' for n in ast.walk(failed))
 (output/'sampling_deadline_controls.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results));assert all(results.values()),results
asyncio.run(main())
