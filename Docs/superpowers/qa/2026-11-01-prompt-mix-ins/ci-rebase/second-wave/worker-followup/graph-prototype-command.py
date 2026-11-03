import ast,importlib.util,json,pathlib
repo=pathlib.Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook');spec=importlib.util.spec_from_file_location('guard',repo/'scripts/check_textual_worker_contract.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
helper='''async def extracted(self, *, request_project_instruction_binding):
    result = await request_project_instruction_binding("session", (), "choose_binding")
    return result
'''
owner='''import asyncio
from .helper import extracted
class Controller:
    def __init__(self, select_project_instruction_binding=None):
        self._select_project_instruction_binding = select_project_instruction_binding
    async def run(self):
        async def request_project_instruction_binding(session_id, selections, recovery_code):
            callback = self._select_project_instruction_binding
            if callback is None:
                return None
            return await callback(session_id, selections, recovery_code)
        return await extracted(self, request_project_instruction_binding=request_project_instruction_binding)
    def start(self):
        asyncio.create_task(self.run())
def make(screen):
    return Controller(select_project_instruction_binding=SELECTOR)
class Screen:
    async def _select_project_instruction_binding(self, session_id, selections, recovery_code):
        return await self.app.push_screen_wait(Modal())
'''
out={}
for name,binding in [('absent_safe','None'),('actual_waiting_selector','screen._select_project_instruction_binding')]:
 modules=[(ast.parse(helper),repo/'tldw_chatbook/helper.py'),(ast.parse(owner.replace('SELECTOR',binding)),repo/'tldw_chatbook/owner.py')];rows=g.collect_w003(modules);out[name]=rows
 assert bool(rows)==(name=='actual_waiting_selector'),out
pathlib.Path('/private/tmp/mixins-worker-late-dependency-prototype.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
