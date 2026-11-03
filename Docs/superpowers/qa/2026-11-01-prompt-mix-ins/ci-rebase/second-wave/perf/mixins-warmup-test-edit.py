from pathlib import Path
import difflib
root=Path.cwd(); updates={}
p=Path('Tests/UI/test_console_first_send_preimport.py');old=p.read_text();new=old.replace('warm.preimport_first_send_modules(app)','warm.preimport_first_send_modules(app, app._finish_first_send_warmup)').replace('(warm._request_idle_heap_refreeze, (app,))','(app._finish_first_send_warmup, ())').replace('warm.preimport_first_send_modules(_PassApp(stop_after=3))','app = _PassApp(stop_after=3)\n    warm.preimport_first_send_modules(app, app._finish_first_send_warmup)')
new=new.replace('    def call_from_thread(self, callback, *args):','    def _finish_first_send_warmup(self):\n        pass\n\n    def call_from_thread(self, callback, *args):')
new=new.replace('    from tldw_chatbook.Chat import console_first_send_modules as warm\n    from tldw_chatbook.UI.Console_Modules import transcript','    from tldw_chatbook.app import TldwCli\n    from tldw_chatbook.UI.Console_Modules import transcript')
new=new.replace('    queued = (app._finish_first_send_warmup, ())','    queued = (TldwCli._finish_first_send_warmup, (app,))')
updates[p]=(old,new)
p=Path('Tests/UI/test_console_idle_heap_refreeze.py');old=p.read_text();new=old.replace('    from tldw_chatbook.Chat import console_first_send_modules as warm\n','').replace('                warm._request_idle_heap_refreeze,\n                case.host.app_instance,','                case.host.app_instance._finish_first_send_warmup,');updates[p]=(old,new)
out=['*** Begin Patch']
for p,(old,new) in updates.items():
 out.append('*** Update File: '+str(root/p));out.extend(list(difflib.unified_diff(old.splitlines(),new.splitlines(),n=3,lineterm=''))[2:])
out.append('*** End Patch');Path('/private/tmp/mixins-warmup-test-edit.patch').write_text('\n'.join(out)+'\n')
