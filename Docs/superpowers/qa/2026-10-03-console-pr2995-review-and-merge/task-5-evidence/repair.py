import ast
from pathlib import Path
p=Path('Tests/UI/test_console_runtime_ownership.py')
s=p.read_text()
tree=ast.parse(s)
manual_names={n.name for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and any(isinstance(x,ast.Attribute) and x.attr=='_initial_screen_pushed' for x in ast.walk(n))}
s=s.replace('from Tests.UI.app_factory import _build_test_app, persist_seeded_config', 'from Tests.UI.app_factory import (\n    _build_test_app as _build_startup_test_app,\n    persist_seeded_config,\n)')
s=s.replace('_build_test_app(', '_build_startup_test_app(')
lines=s.splitlines(keepends=True)
tree=ast.parse(s)
for n in tree.body:
 if isinstance(n,ast.AsyncFunctionDef) and n.name in manual_names:
  for i in range(n.lineno-1,n.end_lineno):
   lines[i]=lines[i].replace('_build_startup_test_app(', '_build_manually_mounted_console_app(')
   if 'app._initial_screen_pushed = True' in lines[i]:lines[i]=''
s=''.join(lines)
s=s.replace('    return _build_startup_test_app(**kwargs)\n', '    # This fixture supplies the content screen itself. Claim startup before\n    # run_test schedules the deferred initial-screen task, not after push_screen.\n    app = _build_startup_test_app(**kwargs)\n    app._initial_screen_pushed = True\n    return app\n',1)
start=s.index('async def test_native_acceptance_consumes_only_open_target_revision(')
old='                await asyncio.wait_for(entered.wait(), 5)\n'
assert old in s[start:]
new='''                readiness = asyncio.create_task(entered.wait())
                try:
                    done, _ = await asyncio.wait(
                        {readiness, start},
                        timeout=5,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    assert start not in done, start.result()
                    assert readiness in done and readiness.result(), {
                        "start_done": start.done(),
                        "start_stack": [
                            (frame.f_code.co_name, frame.f_lineno)
                            for frame in start.get_stack()
                        ],
                        "preparation_tasks": [
                            (task.done(), [
                                (frame.f_code.co_name, frame.f_lineno)
                                for frame in task.get_stack()
                            ])
                            for task in controller._chat_start._tasks
                        ],
                    }
                finally:
                    readiness.cancel()
                    await asyncio.gather(readiness, return_exceptions=True)
'''
s=s[:start]+s[start:].replace(old,new,1)
p.write_text(s)
print('Manual fixture callers:', ', '.join(sorted(manual_names)))
