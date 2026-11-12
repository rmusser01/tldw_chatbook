from pathlib import Path
import sys, subprocess, json
sys.path.insert(0,str(Path.cwd()))
import Tests.conftest
from tldw_chatbook.UI.Console_Modules.console_spend_projection import ConsoleDraftSpendRefresh
class Timer:
 def stop(self): self.stopped=True
callbacks=[];calls=[]
def schedule(delay, callback):
 callbacks.append(callback); return Timer()
owner=ConsoleDraftSpendRefresh(schedule_timer=schedule,sync_settings_summary=lambda:calls.append('settings'),sync_cost_chip=lambda:calls.append('cost'))
owner.route_edit(run_active=False)
assert owner.timer is not None
callbacks[0]()
assert owner.timer is None
assert calls==['settings','cost']
print('NORMAL_CALLBACK_CLEARS_TIMER_AND_RECOMPUTES',calls)
for argv in [
 ['git','log','-4','--format=%H %s','--','Tests/UI/test_console_cost_chip_screen.py','tldw_chatbook/UI/Console_Modules/console_spend_projection.py'],
 ['git','blame','-L','154,160','149adb53f1d10b090e31a5f458e8c2ffcd3ca458','--','tldw_chatbook/UI/Console_Modules/console_spend_projection.py'],
]:
 result=subprocess.run(argv,text=True,capture_output=True)
 print(json.dumps({'argv':argv,'returncode':result.returncode,'output':result.stdout+result.stderr}))
 assert result.returncode==0
