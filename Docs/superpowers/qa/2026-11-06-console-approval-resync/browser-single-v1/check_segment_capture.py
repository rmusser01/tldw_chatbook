"""Finite stdlib-only delegation/clock controls; no App/browser/socket/timer launch."""
import asyncio,copy,json,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from browser_trace_capture import BrowserTraceCapture


def control_output():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',required=True,type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    args=parser.parse_args()
    source=args.source_root.resolve();output=args.output_root.resolve()
    if not output.is_relative_to(source/'.superpowers'):raise RuntimeError('Control_outputs_must_stay_inside_selectedsource_superpowers')
    output.mkdir(parents=True,exist_ok=True)
    return output

class Page:
 def __init__(self):self.now=10.0;self.events=[];self.marks=0;self.http=0
 async def evaluate(self,code,*args):
  before=self.now;self.now+=.5
  if args:
   self.marks+=1;self.events.append({'name':args[0],'pid':42,'ts':(before+100.04)*1000})
   return {'before_ms':before,'after_ms':before+.1}
  self.http+=1;return {'before_ms':before,'after_ms':before+.2,'server_ns':round((before+500.04)*1e6)}

class CDP:
 def __init__(self,page):self.page=page;self.callbacks={};self.calls=[]
 def on(self,name,callback):self.callbacks[name]=callback
 async def send(self,name,value=None):
  self.calls.append((name,value))
  if name=='Tracing.start':self.page.events=[]
  if name=='Tracing.end':
   self.callbacks['Tracing.dataCollected']({'value':self.page.events})
   self.callbacks['Tracing.tracingComplete']({'dataLossOccurred':False})
  return {}

async def controls(output):
 page=Page();cdp=CDP(page);capture=BrowserTraceCapture(page,cdp)
 await capture.start();default=await capture.stop(output/'default.json')
 import runpy
 original=runpy.run_path(str(HERE/'browser_trace_capture.before.py'))['BrowserTraceCapture'];oldpage=Page();oldcdp=CDP(oldpage);old=original(oldpage,oldcdp)
 await old.start();oldresult=await old.stop(output/'old-default.json')
 assert default==oldresult
 assert page.marks==page.http==40
 assert default['intervals']['before']['trace_page']['samples']==20
 config=cdp.calls[0][1]
 assert config['screenshotMaxCount']==50 and config['traceConfig']['traceBufferSizeInKb']==8192
 assert config['transferMode']=='ReportEvents' and config['traceConfig']['recordMode']=='recordContinuously'
 assert config['traceConfig']['includedCategories']==['benchmark','cc','viz','blink.user_timing','devtools.timeline','disabled-by-default-devtools.timeline.frame','disabled-by-default-devtools.screenshot']
 print('ACTUAL_OPTIONAL_API_CALL_NEXT',flush=True)
 page=Page();cdp=CDP(page);capture=BrowserTraceCapture(page,cdp)
 await capture.start(calibration_samples=1);single=await capture.stop(output/'single.json',calibration_samples=1)
 assert page.marks==page.http==2 and single['page_pid']==42
 assert single['intervals']['before']['trace_page']['samples']==single['intervals']['after']['server_page']['samples']==1
 for bad in (0,True,False,2,20.0,None):
  page=Page();cdp=CDP(page);capture=BrowserTraceCapture(page,cdp)
  try:await capture.start(calibration_samples=bad)
  except ValueError:pass
  else:raise AssertionError('Invalid_count_released_start')
  assert not cdp.calls and page.marks==page.http==0
  await capture.start()
  baseline=(len(cdp.calls),page.marks,page.http)
  try:await capture.stop(output/'bad.json',calibration_samples=bad)
  except ValueError:pass
  else:raise AssertionError('Invalid_count_released_stop')
  assert (len(cdp.calls),page.marks,page.http)==baseline
  try:await capture.calibrate('before',samples=bad)
  except ValueError:pass
  else:raise AssertionError('Invalid_count_released_calibration')
 from browser_series_method import validate_clock_series
 series=[copy.deepcopy(default),copy.deepcopy(default)]
 for index,row in enumerate(series):row.update(round_index=0,role='actionable' if index==0 else 'feedback')
 series[0]['intervals']['after']['trace_page']['samples']=series[0]['intervals']['after']['server_page']['samples']=1
 series[1]['intervals']['before']['trace_page']['samples']=series[1]['intervals']['before']['server_page']['samples']=1
 report=validate_clock_series(series)
 assert report['renderer_pid']==42 and report['segments']==2
 wrong=copy.deepcopy(series);wrong[1]['role']='actionable'
 try:validate_clock_series(wrong)
 except ValueError:pass
 else:raise AssertionError('Reversed_or_duplicate_segment_role_released')
 wrong=copy.deepcopy(series);wrong[1]['round_index']=True
 try:validate_clock_series(wrong)
 except ValueError:pass
 else:raise AssertionError('Boolean_round_index_released')
 long=[]
 for index in range(82):
  row=copy.deepcopy(default);row.update(round_index=index//2,role='actionable' if index%2==0 else 'feedback')
  for phase in ('before','after'):
   for clock in ('trace_page','server_page'):row['intervals'][phase][clock]['samples']=20 if (index==0 and phase=='before') or (index==81 and phase=='after') else 1
  long.append(row)
 assert validate_clock_series(long)['segments']==82
 try:validate_clock_series(long[:41])
 except ValueError:pass
 else:raise AssertionError('Former41_segment_layout_released')

 for clock in ('trace_page','server_page'):
  altered=copy.deepcopy(series)
  for k in ('lower_ms','upper_ms'):altered[1]['envelope'][clock][k]+=1000
  try:validate_clock_series(altered)
  except ValueError:pass
  else:raise AssertionError('Clock_origin_or_drift_mismatch_released')
 for clock in ('trace_page','server_page'):
  altered=copy.deepcopy(series)
  for k in ('lower_ms','upper_ms'):altered[0]['intervals']['after'][clock][k]+=1000
  altered[0]['envelope'][clock]['upper_ms']+=1000
  try:validate_clock_series(altered)
  except ValueError:pass
  else:raise AssertionError('One_sided_endpoint_drift_straddling_union_released')
 altered=copy.deepcopy(series);altered[1]['page_pid']=43
 try:validate_clock_series(altered)
 except ValueError:pass
 else:raise AssertionError('Changed_renderer_released')
 from browser_series_method import validate_capture_limits,validate_capture_series
 data=json.loads((output/'default.json').read_text());validate_capture_limits(output/'default.json')
 overflow=copy.deepcopy(data);overflow['traceEvents']=[{'name':'Screenshot','args':{'snapshot':'AA=='}} for _ in range(51)]
 for count in (32,33):
  split=copy.deepcopy(data);split['traceEvents']=[{'name':'Screenshot','args':{'snapshot':'AA=='}} for _ in range(count)];(output/f'split-{count}.json').write_text(json.dumps(split))
 assert validate_capture_series([output/'split-32.json',output/'split-32.json'])['actual_screenshots']==64
 try:validate_capture_series([output/'split-32.json',output/'split-33.json'])
 except ValueError:pass
 else:raise AssertionError('Combined_round65_frames_released')
 overflow_path=output/'overflow.json';overflow_path.write_text(json.dumps(overflow))
 try:validate_capture_limits(overflow_path)
 except ValueError:pass
 else:raise AssertionError('50_frame_overflow_released')
 assert overflow_path.is_file() and len(json.loads(overflow_path.read_text())['traceEvents'])==51
 for label,change in [('loss',lambda x:x.update(tracing_complete={'dataLossOccurred':True})),('missing_completion',lambda x:x.pop('tracing_complete'))]:
  bad=copy.deepcopy(data);change(bad);path=output/(label+'.json');path.write_text(json.dumps(bad))
  try:validate_capture_limits(path)
  except ValueError:pass
  else:raise AssertionError('Trace_loss_or_missing_completion_released')
 class MissingCDP(CDP):
  async def send(self,name,value=None):
   if name=='Tracing.end':self.page.events.pop()
   return await super().send(name,value)
 page=Page();cdp=MissingCDP(page);capture=BrowserTraceCapture(page,cdp);await capture.start()
 try:await capture.stop(output/'missing-mark.json')
 except AssertionError:pass
 else:raise AssertionError('Missing_actual_trace_clock_mark_released')
 assert (output/'missing-mark.json').is_file()
 from browser_presented_frames import presented_frames
 assert presented_frames([],page_pid=42)[0]==[]
 return {'status':'PASS','default20_original_protocol_math':True,'exact1_endpoint_pair':True,'invalid_counts_refused_before_original_calls':True,'actual_clock_origin_drift_mismatch_refused':True,'changed_renderer_refused':True,'same8MiB50_limits':True,'overflow_loss_missing_marks_retained_invalid':True,'exact2_or82_roles_rounds_and_combined64_budget':True,'App_browser_socket_model_launches':0,'product_samples':0,'qualified':False}

if __name__=='__main__':
 output=control_output()/'segment-controls';output.mkdir(exist_ok=True)
 try:result=asyncio.run(controls(output))
 except Exception as error:
  (output/'last-control-status.json').write_text(json.dumps({'status':'failed','error_type':type(error).__name__,'App_browser_launches':0}),encoding='utf-8');raise
 print(json.dumps(result),flush=True)
