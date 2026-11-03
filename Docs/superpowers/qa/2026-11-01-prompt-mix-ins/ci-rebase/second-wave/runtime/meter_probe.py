import pathlib,json,difflib

def pytest_runtest_makereport(item,call):
 if call.excinfo is None:return
 for entry in call.excinfo.traceback:
  if entry.name != '_assert_fenced_preflight_matches_send':continue
  local=entry.frame.f_locals;rows=local['rows'];actual=local['gateway'].messages_seen[-1]
  data={'node':item.nodeid,'planned':rows,'actual':actual,'diff':list(difflib.unified_diff(json.dumps(rows,indent=2).splitlines(),json.dumps(actual,indent=2).splitlines()))}
  with pathlib.Path(__file__).with_name('meter-observations.jsonl').open('a') as out:out.write(json.dumps(data)+'\n')
