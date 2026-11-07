"""Whole mounted TldwCli over the user-selected local provider, private profile only."""
import asyncio, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(sys.argv[1]).resolve()
STAGE=sys.argv[2]
assert ROOT.is_dir() and ROOT.name.startswith('task10-live-')
for folder in ('home','config','data'):
    (ROOT/folder).mkdir(mode=0o700,exist_ok=True)
os.environ.update(HOME=str(ROOT/'home'),USERPROFILE=str(ROOT/'home'),XDG_CONFIG_HOME=str(ROOT/'config'),XDG_DATA_HOME=str(ROOT/'data'),TLDW_CONFIG_PATH=str(ROOT/'config/config.toml'),TLDW_DISABLE_CONFIG_WATCH='1',PYTHON_KEYRING_BACKEND='keyring.backends.null.Keyring',NO_PROXY='127.0.0.1,localhost')
sys.path.insert(0,str(Path.cwd()))
from Tests import real_profile_guard
real_profile_guard.install()
def loopback_only(kind,args):
    if kind=='socket.connect' and isinstance(args[1],tuple) and args[1][:2] != ('127.0.0.1',8010):
        raise PermissionError('qualification permits only the configured loopback endpoint')
sys.addaudithook(loopback_only)
selector=Path(os.environ['TLDW_CONFIG_PATH'])
if not selector.exists():
    selector.write_text('''[general]
users_name="default_user"
default_tab="chat"
[first_run]
setup_completed=true
[splash_screen]
enabled=false
[model_catalog]
refresh_consent_recorded=true
auto_refresh_enabled=false
[chat_defaults]
provider="llama_cpp"
model="unsloth/Qwen3.8-27B-GGUF:Q8_0"
reasoning_effort="none"
max_tokens=4096
temperature=0.0
streaming=false
[providers]
Llama_cpp=["unsloth/Qwen3.8-27B-GGUF:Q8_0"]
[agents]
workflow_runs_enabled=true
run_log_enabled=true
run_log_max_record_bytes=8192
[console]
raw_cli_permitted=true
local_tools_enabled=true
agent_runtime=true
native_tool_calls=true
[api_settings.llama_cpp]
api_url="http://127.0.0.1:8010"
model="unsloth/Qwen3.8-27B-GGUF:Q8_0"
''')
    selector.chmod(0o600)
import tldw_chatbook.Widgets.compact_model_bar
from tldw_chatbook import app as source
from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
from tldw_chatbook.Chat.console_project_instructions import ProjectInstructionControlState
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_workflow_status import ConsoleWorkflowStatus
from textual.widgets import Button,TextArea,Input,Select
from tldw_chatbook.Chat.console_goal_runs import goal_provider_ref
def record(name,**fields):
    with (ROOT/'live.jsonl').open('a') as file: print(json.dumps(dict(time=time.time(),pid=os.getpid(),event=name,**fields)),file=file,flush=True)
async def main():
    creators=OriginalPrivateAppCreators(source,ROOT)
    creators.install()
    try: app=source.TldwCli()
    except BaseException: creators.close(); raise
    creators.finish_construction(app)
    runtime=app.console_runtime
    project=ROOT/'project'
    registry=app.workspace_registry_service
    if STAGE in {'plan','probe'}:
        project.mkdir(mode=0o700)
        import subprocess
        for args in [('init','-q')]:
            subprocess.run(['git','-C',str(project),*args],check=True)
        (project/'value.txt').write_text('original\n')
        wrapper=project/'verify.sh'
        wrapper.write_text('#!/bin/sh\ncd "$(dirname "$0")" || exit 2\ngrep -Fxq corrected value.txt || exit 7\nprintf "verified\\n"\n')
        wrapper.chmod(0o700)
        for args in [('add','--','value.txt','verify.sh'),('commit','-qm','private qualification source')]:
            subprocess.run(['git','-C',str(project),'-c','user.name=Qualification','-c','user.email=test@example.invalid',*args],check=True)
        (project/'value.txt').write_text('user edit\noriginal\n')
        registry.create_workspace(workspace_id='qualification',name='Qualification')
        registry.add_folder_binding('qualification',project,allow_write=True)
        registry.set_active_workspace('qualification')
    try:
        import httpx, requests
        monitor=sys.monitoring
        monitor.use_tool_id(5,"workflow-live-provider-shape")
        from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider, LocalToolInvocationResult
        def response_observation(code, offset, result):
            if type(result) is LocalToolInvocationResult:
                frame=sys._getframe(1)
                owner=frame.f_locals["self"]
                tool=frame.f_locals.get("tool_id","")
                args=frame.f_locals.get("args",{})
                path=Path(str(owner._root))
                record("local-tool-return",code_name=code.co_name,tool=tool,args=args,
                       root=str(path) if path.is_relative_to(ROOT) else "outside_private_profile",
                       admitted_aliases=list(owner._admitted_roots or {}),
                       ok=result.result.ok,error=result.result.error,reason=str(result.reason_code),
                       dispatch=result.dispatch_started,terminal=str(result.provider_terminal))
                return
            if not isinstance(result,dict) or not result.get("choices"): return
            choice=result["choices"][0]
            msg=choice.get("message",{})
            content=msg.get("content") or ""
            record("completion", model=result.get("model"),finish=choice.get("finish_reason"),content_length=len(content),begins_json=content.lstrip().startswith("{"),ends_json=content.rstrip().endswith("}"),reasoning_length=len(msg.get("reasoning_content") or ""), tool_names=[row.get("function",{}).get("name") for row in (msg.get("tool_calls") or [])],usage=result.get("usage"))
        monitor.register_callback(5,monitor.events.PY_RETURN,response_observation)
        from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor
        from tldw_chatbook.Agents import workflow_task_scope as scopes
        from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply
        from tldw_chatbook.Agents.legacy_hook_ownership import _EventSource, _NativeLegacyHookSource
        observed_unwinds={
            function.__code__:function.__name__
            for function in (
                WorkspaceToolExecutor.execute,scopes.check_task_provider,
                scopes.check_task_call,scopes._check_task_call,scopes._check_paths,
                scopes._refuse,scopes.WorkflowTaskAuthorization.check,
                scopes.WorkflowTaskAuthorization._check_continuation,
                _PreparedWorkflowReply._check_task_source,_PreparedWorkflowReply.require_current,
                scopes._check_hooks,_EventSource.check,_NativeLegacyHookSource.capture.__wrapped__,
            )
        }
        def worker_unwind(code,offset,error):
            if code not in observed_unwinds: return
            frame=sys._getframe(1)
            local=frame.f_locals
            facts={}
            if code is scopes._refuse.__code__ and frame.f_back.f_code is scopes._check_hooks.__code__:
                parent=frame.f_back
                state=parent.f_locals
                issuer=state.get("issuer")
                engine=state.get("engine")
                configured=state.get("configured")
                runtime=state.get("runtime")
                facts.update(
                    hook_line=parent.f_lineno, hook_reason=local.get("reason"),
                    definitions=len(configured.hooks) if configured else None,
                    v2_handlers=len(configured.v2_handlers) if configured else None,
                    config_enabled=configured.enabled if configured else None,
                    engine_type=type(engine).__name__,issuer_type=type(issuer).__name__,
                    issuer_same_engine=issuer.engine is engine if issuer else None,
                    issuer_same_runtime=issuer.runtime is runtime if issuer else None,
                    engine_attached=engine._native_attached if engine else None,
                    original_admissions=issuer._admissions if issuer else None,
                    original_events=len(issuer._events) if issuer else None,
                    published_targets=len(issuer.permissions._published_targets) if issuer else None,
                )
            if code is scopes.WorkflowTaskAuthorization.check.__code__:
                row=local.get("row")
                facts["original_row"]=list(row) if row is not None else None
                facts["expected_launch"]=local["self"].claim.launch_run_id
            if code is scopes._check_paths.__code__:
                facts["same_binding"]=local.get("authority") is local["authorization"].binding
                facts["tool"]=local.get("name")
                facts["args"]=local.get("args")
            record("original-scope-unwind",function=observed_unwinds[code],
                   kind=type(error).__name__,message=str(error)[:900],**facts)
        monitor.register_callback(5,monitor.events.PY_UNWIND,worker_unwind)
        monitor.set_events(5,monitor.events.PY_UNWIND)
        for function in (httpx.Response.json, requests.Response.json, LocalToolProvider._invoke_detailed):
            monitor.set_local_events(5,function.__code__,monitor.events.PY_RETURN)
        async with app.run_test(size=(160,55)) as pilot:
            async def until(check,label,seconds=40,approve_tools=False):
                try:
                    async with asyncio.timeout(seconds):
                        while not check():
                            if approve_tools and console._console_pending_approval_count():
                                await handle_pending()
                            await pilot.pause(.05)
                except BaseException as primary:
                    record('stage-failed',label=label,kind=type(primary).__name__,message=str(primary)[:500],screen=type(app.screen).__name__)
                    import traceback
                    traceback.print_exception(primary)
                    app.save_screenshot(path=str(ROOT),filename='failed-'+str(os.getpid())+'.svg')
                    if approve_tools:
                        try:
                            if app.screen is console:
                                if console._console_pending_approval_count():
                                    buttons=[b for b in console.query(Button) if b.id and b.id.startswith('approval-fast-deny-') and b.display and b.visible and not b.disabled]
                                    if len(buttons)==1:
                                        await click('#'+buttons[0].id)
                                await pilot.press('escape')
                                await click('#console-workflow-line')
                                await until(lambda:isinstance(app.screen,ConsoleWorkflowStatus),'cleanup detail')
                            await click('#workflow-stop')
                            await until(lambda:app.screen.snapshot.status=='stopped','actual Stop returned',40)
                            record('stop-after-stage-failure',label=label)
                        except BaseException as cleanup:
                            record('stop-refused',kind=type(cleanup).__name__)
                    raise
                record('ready',label=label)
            async def click(selector):
                btn=app.screen.query_one(selector,Button)
                btn.scroll_visible(animate=False)
                def hit():
                    x,y=map(int,btn.region.center)
                    if btn.disabled or not btn.visible or not btn.is_on_screen or not app.screen.region.contains(x,y): return False
                    target,_=app.screen.get_widget_at(x,y)
                    return target is btn or btn in target.ancestors
                try: await until(hit,selector,12)
                except TimeoutError:
                    record('hit-refused',selector=selector,screen=type(app.screen).__name__,stack=[type(scr).__name__ for scr in app.screen_stack],region=list(btn.region),disabled=btn.disabled,display=btn.display,visible=btn.visible)
                    raise
                rect=btn.region
                assert await pilot.click(selector,offset=(rect.width//2,rect.height//2))
                record('clicked',id=selector,region=list(rect))
            async def handle_pending():
                from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

                assert isinstance(app.screen,ConsoleWorkflowStatus)
                workflow=app.screen.snapshot.id
                await click('#workflow-refresh')
                await until(lambda:app.screen.query_one('#workflow-pending',Button).display,'pending decision visible')
                await click('#workflow-pending')
                await until(lambda:app.screen is console and any(card.display and card.is_on_screen and card._batch_calls_snapshot for card in console.query(ChatApprovalCard)),'actual approval card')
                card=next(card for card in console.query(ChatApprovalCard) if card.display and card.is_on_screen and card._batch_calls_snapshot)
                calls=card._batch_calls_snapshot
                assert len(calls)==1
                record('pending-call',rows=calls)
                row=calls[0]
                name=row.get('llm_name',row.get('name',row.get('tool_name')))
                assert name in {'fs_read','fs_edit','shell_exec'}, name
                args=row.get('arguments',row.get('args',{}))
                if isinstance(args,str): args=json.loads(args)
                if name=='shell_exec':
                    assert args.get('command')==str(project/'verify.sh')
                else:
                    assert args.get('path')=='value.txt'
                buttons=[b for b in card.query(Button) if b.id and b.id.startswith('approval-fast-approve-') and not b.disabled]
                assert len(buttons)==1
                await click('#'+buttons[0].id)
                await until(lambda:console._console_pending_approval_count()==0 and (not card.is_mounted or not card.display or not card.is_on_screen),'tool approval accepted')
                from tldw_chatbook.Widgets.Console.console_command_popup import ConsoleCommandPopup
                await pilot.press('escape')
                await until(lambda:not console.query_one(ConsoleCommandPopup).is_open,'dismiss preserved command popup')
                await click('#console-workflow-line')
                await until(lambda:isinstance(app.screen,ConsoleWorkflowStatus) and app.screen.snapshot is not None,'detail after approval')

            await until(lambda:isinstance(app.screen,ChatScreen) and app._ui_ready,'whole-mounted-chat')
            console=app.screen
            store=console._ensure_console_chat_store()
            controller=console._ensure_console_chat_controller()
            runtime=app.console_runtime
            assert console._console_runtime() is runtime and controller is runtime._chat_controller
            if STAGE in {'plan','probe'}:
                chat=store.create_session(title='Private workflow qualification',workspace_id='qualification',
                    settings=ConsoleSessionSettings(provider='llama_cpp',model='unsloth/Qwen3.8-27B-GGUF:Q8_0',base_url='http://127.0.0.1:8010',temperature=0.0,max_tokens=4096,reasoning_effort="none",system_prompt="For planning return exactly the requested WorkflowPlan JSON object, no Markdown, analysis or tool call. For execution actually perform authorized tool calls and report their results.",streaming=False,source='user'),
                    project_instruction_state=ProjectInstructionControlState.legacy_disabled())
                conversation=store.persist_session_if_needed(chat.id)
                record('chat',session=chat.id,conversation=conversation)
            else:
                saved=json.loads((ROOT/'selection.json').read_text())
                assert await console._workspace._resume_console_workspace_conversation(saved["conversation"],preserve_persisted_scope=True)
                chat=store.get_session(store.active_session_id) if hasattr(store,'get_session') else next(s for s in store.sessions() if s.id==store.active_session_id)
                conversation=saved["conversation"]
                assert chat.persisted_conversation_id==conversation and chat.workspace_id=="qualification"
                record('hydrated-original-chat',session=chat.id,conversation=conversation)
            console._workspace.activate_workspace_id('qualification')
            await until(lambda:bool(console.query('#console-session-tab-'+chat.id)),'named-chat-tab')
            if store.active_session_id!=chat.id: await click('#console-session-tab-'+chat.id)
            assert store.active_session_id==chat.id
            gateway=controller.provider_gateway if hasattr(controller,'provider_gateway') else controller._provider_gateway
            resolved=await gateway.resolve_for_send(controller._provider_selection_for_session(chat.id))
            record('native-provider',ready=resolved.ready,model=resolved.model,provider=resolved.provider,base=resolved.base_url)
            assert resolved.ready and resolved.base_url.rstrip('/')=='http://127.0.0.1:8010'
            assert runtime.raw_cli_source.raw is app.raw_cli_runtime
            assert app.raw_cli_runtime.arm().armed
            app.raw_cli_runtime.grant_model_session(chat.id)
            assert app.raw_cli_runtime.model_session_granted(chat.id)
            for name in ('fs_read','fs_edit'):
                app.unified_mcp_service.approve_for_session('local:__local__',name)
            composer=console.query_one('#console-native-composer')
            if STAGE=='restart':
                console._focus_console_composer_if_needed(force=True)
                await pilot.pause(.25)
                await pilot.press('/', 'w', 'o', 'r', 'k', 'f', 'l', 'o', 'w')
                await until(lambda:composer.draft_text()=='/workflow','typed history command')
                await click('#console-send-message')
                await until(lambda:bool(app.screen.query('#workflow-saved')),'history')
                app.screen.query_one('#workflow-saved',Select).value=saved["workflow"]
                await click('#workflow-open')
                await until(lambda:isinstance(app.screen,ConsoleWorkflowStatus) and app.screen.snapshot is not None,'reopened history detail')
                assert app.screen.snapshot.id==saved['workflow']
                record('restart-inert',status=app.screen.snapshot.status,spend=dict(app.screen.snapshot.accounting.used),results=len(app.screen.snapshot.results))
                assert app.screen.snapshot.accounting.used['model_call']==1 and not app.screen.snapshot.results
                await click('#workflow-resume')
                await until(lambda:app.screen.snapshot.status=='awaiting_phase_review','explicit resumed review')
                await click('#workflow-approve')
                await until(lambda:app.screen.snapshot.status=='paused' and len(app.screen.snapshot.results)==1,'real failed check',160,approve_tools=True)
                failed=app.screen.snapshot.results[0]
                checkpoint=await app.screen.coordinator.service.call(app.screen.coordinator.service.read_checkpoint,saved['workflow'],failed.attempt_id)
                assert failed.status=='failed' and failed.checks[0].status=='failed' and failed.checks[0].exit_code==7
                observed=checkpoint.verifiers[0]
                assert observed.complete and observed.cleanup_proven and observed.fresh and observed.exit_code==7 and observed.input_manifest
                record('actual-failed',result=failed.model_dump(),checkpoint=checkpoint.model_dump())
                repair=(
                    'Now revise the initial diagnosis-only plan. Preserve its failed check as history. Replace phases with exactly two phases. '
                    'First phase correct has one task fix-source, role worker, exclusive, allowed_tools fs_read, fs_edit, shell_exec. '
                    'Read value.txt, use fs_edit replacing original with corrected, preserve the user edit line, then execute the existing verify.sh with shell bash and its exact absolute path. '
                    'Declare inputs ["value.txt"], outputs ["value.txt"], write footprint value.txt and read footprint verify.sh. '
                    'Copy the original verifier definition and SHA256 unchanged into this task with expected_checks ["value-check"]. '
                    'Second phase review has task review-result, role worker, scoped, allowed_tools fs_read, dependencies ["fix-source"], inputs ["fix-source:value.txt"], outputs [], read footprint value.txt. '
                    'That task must inspect the corrected value.txt with fs_read and independently check that user edit and corrected lines are present. '
                    'Keep request and root_identity unchanged. Initial diagnosis is finished; do not plan it again. '
                    'No shell commands except the approved verifier. Both tasks use the same exact requested target.'
                )
                app.screen.query_one('#workflow-instruction',Input).value=repair
                await click('#workflow-revise')
                await until(lambda:app.screen.snapshot.status=='awaiting_phase_review' and len(app.screen.snapshot.plans)==2,'actual revised plan',150)
                current=app.screen.snapshot
                (ROOT/'revised-plan.json').write_text(current.plans[-1].canonical_json())
                assert len(current.plans[-1].phases)==2
                record('repair-plan',phase_ids=[p.id for p in current.plans[-1].phases],spend=dict(current.accounting.used))
                await click('#workflow-approve')
                await until(lambda:app.screen.snapshot.status=='awaiting_result_review' and len(app.screen.snapshot.results)==2,'real corrected check',160,approve_tools=True)
                corrected=app.screen.snapshot.results[-1]
                passed=await app.screen.coordinator.service.call(app.screen.coordinator.service.read_checkpoint,saved['workflow'],corrected.attempt_id)
                assert corrected.checks[0].status=='passed' and corrected.checks[0].exit_code==0
                assert passed.verifiers[0].complete and passed.verifiers[0].cleanup_proven and passed.verifiers[0].fresh
                assert passed.verifiers[0].input_manifest!=observed.input_manifest
                assert (project/'value.txt').read_text()=='user edit\ncorrected\n'
                record('actual-passed',result=corrected.model_dump(),checkpoint=passed.model_dump())
                await click('#workflow-accept')
                await until(lambda:app.screen.snapshot.status=='awaiting_phase_review','next phase review')
                await click('#workflow-approve')
                await until(lambda:app.screen.snapshot.status=='awaiting_result_review' and len(app.screen.snapshot.results)==3,'independent real review',100,approve_tools=True)
                record('review-result',result=app.screen.snapshot.results[-1].model_dump())
                await click('#workflow-accept')
                await until(lambda:app.screen.snapshot.status=='completed','accepted final result')
                final=app.screen.snapshot
                record('completed',spend=dict(final.accounting.used),result_count=len(final.results))
                app.save_screenshot(path=str(ROOT),filename='completed.svg')
                await pilot.press('escape')
                await until(lambda:app.screen is console,'final detach')
                return
            console._focus_console_composer_if_needed(force=True)
            await pilot.pause(.25)
            await pilot.press('/', 'w', 'o', 'r', 'k', 'f', 'l', 'o', 'w', ' ', 'n', 'e', 'w')
            await until(lambda:composer.draft_text()=='/workflow new','typed workflow command')
            record('draft',active=store.active_session_id,session=chat.id,disabled=console.query_one('#console-send-message',Button).disabled)
            await click('#console-send-message')
            await until(lambda:bool(app.screen.query('#workflow-objective')),'setup')
            wrapper=project/'verify.sh'
            sha=hashlib.sha256(wrapper.read_bytes()).hexdigest()
            goal=(
                'Diagnose and then correct value.txt without changing its existing user edit line. '
                'The first plan must contain exactly ONE phase diagnose with ONE exclusive task named diagnose-source and tool shell_exec. '
                'That task MUST actually call shell_exec with shell bash and command '+str(wrapper)+' then hand off the observed result even if nonzero; no edits before explicit repair. '
                'Declare input_paths ["value.txt"] and expected_checks ["value-check"]; the verifier object has id value-check, executor_tool_id shell_exec, verifier_path '+str(wrapper)+
                ', verifier_sha256 '+sha+', input_paths ["value.txt"], no arguments, expected_exit_code 0. '
                'Include read footprints for value.txt and verify.sh; inputs ["value.txt"]; outputs []. Use the exact request and root_identity from the native planning message. '
                'Use the exact requested target object. roles ["worker"], task role worker. Deliverable is a fresh verified correction. '
                'A later explicit repair instruction will authorize the correction. Never use shell for file editing.'
            )
            if STAGE=='probe':
                goal='Create exactly one phase inspect with exactly one task read-source, role worker, execution_mode exclusive, allowed_tools ["fs_read","shell_exec"], inputs ["value.txt"], outputs [], footprint read value.txt, expected_checks [], verifiers []. The task must actually fs_read value.txt with the current binding alias, then return the two lines it read. Use shell_exec only as an allowed prospective tool, do not invoke shell. Keep request/root_identity exact. This is a read-only diagnostic. Preserve all files.'
            app.screen.query_one('#workflow-objective',TextArea).load_text(goal)
            await click('#workflow-plan')
            await until(lambda:isinstance(app.screen,ConsoleWorkflowStatus) and app.screen.snapshot is not None,'detail',90)
            await until(lambda:app.screen.snapshot.status not in ('planning','running'),'plan terminal',210)
            current=app.screen.snapshot
            record('plan-disposition',workflow=current.id,status=current.status,plans=len(current.plans),spend=dict(current.accounting.used))
            (ROOT/'selection.json').write_text(json.dumps({'workflow':current.id,'conversation':conversation,'session':chat.id}))
            assert current.status=='awaiting_phase_review'
            plan=current.plans[-1]
            (ROOT/'selected-plan.json').write_text(plan.canonical_json())
            task=plan.phases[0].tasks[0]
            if STAGE=='probe':
                await click('#workflow-approve')
                await until(lambda:app.screen.snapshot.status=='awaiting_result_review' and len(app.screen.snapshot.results)==1,'native diagnostic read',55,approve_tools=True)
                record('read-probe-result',result=app.screen.snapshot.results[-1].model_dump())
                await click('#workflow-accept')
                await until(lambda:app.screen.snapshot.status=='completed','read probe quality')
            else:
                assert len(plan.phases)==len(plan.phases[0].tasks)==len(task.verifiers)==1 and task.allowed_tools==('shell_exec',)
                assert task.verifiers[0].verifier_sha256==sha
            await pilot.press('escape')
            await until(lambda:app.screen is console,'detach clean review')
            record('detached',phase=plan.phases[0].id)
    finally:
        # The whole mounted App owns runtime shutdown and may clear its reference.
        if app.console_runtime is runtime and not runtime._disposed:
            await runtime.dispose()
        primary=sys.exc_info()[1]
        await asyncio.get_running_loop().shutdown_default_executor()
        if 'monitor' in locals() and monitor.get_tool(5) == "workflow-live-provider-shape":
            monitor.set_events(5,0)
            for function in (httpx.Response.json,requests.Response.json,LocalToolProvider._invoke_detailed):
                monitor.set_local_events(5,function.__code__,0)
            assert monitor.register_callback(5,monitor.events.PY_RETURN,None) is response_observation
            assert monitor.register_callback(5,monitor.events.PY_UNWIND,None) is worker_unwind
            monitor.free_tool_id(5)
            record('monitor-released')
        try: creators.retire()
        except BaseException as cleanup:
            record("creator-refused", kind=type(cleanup).__name__, reason=str(cleanup)[:420])
            if primary is None: raise
            primary.add_note("actual creator retirement refused: "+str(cleanup))
        else: record('creators-returned')
if __name__=='__main__':
    asyncio.run(main())
