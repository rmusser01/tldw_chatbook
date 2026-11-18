import asyncio,gc,json,os,time,sys
from pathlib import Path
import pytest
OUT=Path(os.environ['UI18_EVIDENCE'])/'timeline.jsonl'
def log(**values):
 with OUT.open('a') as f:f.write(json.dumps({'time':time.monotonic(),**values})+'\n')
@pytest.fixture(autouse=True)
def observe_refreezer(monkeypatch):
 from tldw_chatbook.Utils.ui_responsiveness import IdleHeapRefreezer
 settled = asyncio.Event()
 to_thread = asyncio.to_thread
 submissions = 0
 async def ordered_to_thread(function,*args,**kwargs):
  nonlocal submissions
  if getattr(function,"__name__",None)=="call_from_thread" and args and getattr(args[0],"__name__",None)=="_finish_first_send_warmup":
   submissions += 1
   if submissions == 2 and os.environ.get("UI18_ORDERED"):
    log(event="await-existing-run-settled-timer")
    await asyncio.wait_for(settled.wait(),2)
    log(event="submit-warmup-after-real-run-settled")
  return await to_thread(function,*args,**kwargs)
 monkeypatch.setattr(asyncio,"to_thread",ordered_to_thread)
 request,fire,maintain=IdleHeapRefreezer.request,IdleHeapRefreezer._fire,IdleHeapRefreezer.maintain
 def requested(self,reason,owner,busy):
  log(event='request',reason=reason,busy=busy(owner),previous=self._request[0] if self._request else None,actions=list(self.actions))
  return request(self,reason,owner,busy)
 def fired(self,generation):
  log(event='fire',generation=generation,current=self._generation,reason=self._request[0] if self._request else None)
  return fire(self,generation)
 def maintained(self,reason):
  log(event='maintain-start',reason=reason)
  result=maintain(self,reason)
  if reason=='run_settled':settled.set()
  log(event='maintain-end',reason=reason,result=result,actions=list(self.actions))
  return result
 def trace(frame,event,arg):
  if event=='line' and frame.f_code.co_name=='test_mounted_received_custody_blocks_completion_and_idle_timer_reclaims' and frame.f_lineno>=301:
   v=frame.f_locals;ref=v.get('reference');r=v.get('refreezer')
   log(event='test-line',line=frame.f_lineno,cycle_alive=ref() is not None if ref else None,actions=list(r.actions) if r else None,pending=r._request[0] if r and r._request else None)
  return trace if frame.f_code.co_name=="test_mounted_received_custody_blocks_completion_and_idle_timer_reclaims" else None
 monkeypatch.setattr(IdleHeapRefreezer,'request',requested);monkeypatch.setattr(IdleHeapRefreezer,'_fire',fired);monkeypatch.setattr(IdleHeapRefreezer,'maintain',maintained)
 sys.settrace(trace)
 try:yield
 finally:sys.settrace(None)
