import asyncio,json,time,threading,sys,traceback,os
from pathlib import Path
import pytest,pytest_asyncio
OUT=Path(os.environ["UI28_EVIDENCE"])/"phases.jsonl"
def record(**v):
 with OUT.open("a") as f:f.write(json.dumps({"time":time.time(),**v})+"\n")
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
 record(node=item.nodeid,phase="setup-start");yield;record(node=item.nodeid,phase="setup-end")
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
 record(node=item.nodeid,phase="call-start");yield;record(node=item.nodeid,phase="call-end")
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
 record(node=item.nodeid,phase="teardown-start");yield;record(node=item.nodeid,phase="teardown-end")
def chain(c):
 out=[];seen=set()
 while c is not None and id(c) not in seen:
  seen.add(id(c));f=getattr(c,"cr_frame",None) or getattr(c,"gi_frame",None)
  if f is not None:out.append([f.f_code.co_filename,f.f_code.co_name,f.f_lineno])
  c=getattr(c,"cr_await",None) or getattr(c,"gi_yieldfrom",None)
 return out
@pytest_asyncio.fixture(autouse=True)
async def phase_observer(request):
 async def observe():
  while True:
   await asyncio.sleep(3)
   record(node=request.node.nodeid,phase="tasks",tasks=[{"name":t.get_name(),"stack":chain(t.get_coro())} for t in asyncio.all_tasks()])
 task=asyncio.create_task(observe(),name="ui28-observer")
 try:yield
 finally:
  task.cancel()
  try:await task
  except asyncio.CancelledError:pass

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items):
 if not os.environ.get("UI28_INJECT_PRECONDITION"):return
 import inspect
 module=next(i.module for i in items if "test_console_response_rules" in i.nodeid)
 original=module.mounted_rules_console
 from contextlib import asynccontextmanager
 @asynccontextmanager
 async def observed(*args,**kwargs):
  async with original(*args,**kwargs) as case:
   def trace(frame,event,arg):
    if event=="line" and frame.f_code.co_name=="test_rule_phase_events_project_only_the_mounted_current_session" and frame.f_lineno==427:
     sys.settrace(None)
     record(phase="injected-precondition",helper_entered=case.helper_entered.is_set(),helper_released=case.helper_release.is_set())
     asyncio.get_running_loop().call_later(8,case.helper_release.set)
     raise AssertionError("UI28 diagnostic injected precondition failure")
    return trace
   sys.settrace(trace)
   try:yield case
   finally:sys.settrace(None)
 module.mounted_rules_console=observed
