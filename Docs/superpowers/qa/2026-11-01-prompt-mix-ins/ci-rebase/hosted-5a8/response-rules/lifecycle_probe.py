"""Controlled original resume read, concurrent original close; diagnostic only."""
import asyncio
import json
import os
from pathlib import Path

import pytest

from profile_observer import Observer

ROOT=Path('/private/tmp/mixins-5a8-ui26-diagnosis')


async def probe(tmp_path,size,review_check):
    from Tests.UI.response_rules_fixtures import mounted_rules_console, seed_answer
    from tldw_chatbook.UI.Console_Modules import workspace
    from tldw_chatbook.UI.Console_Modules import composer_run_controls as run_controls
    observations={}
    try:
        async with mounted_rules_console(tmp_path,size) as case:
            async with asyncio.timeout(30):
                while not (case.console._console_attach_sync_complete and case.console._console_attach_reconciled and not case.console._console_attach_reconcile_running and not case.console._console_sync_in_progress and not case.console._console_sync_requested and run_controls.poll_source_is_current(case.console,getattr(case.console,'_console_completed_full_source',None))):
                    await asyncio.sleep(.02)
            await case.pilot.pause()
            answer=seed_answer(case)
            old_id=case.session_id
            session=next(s for s in case.chats.sessions() if s.id==old_id)
            conversation_id=session.persisted_conversation_id
            case.composer.load_draft('temporary authored draft')
            case.composer.load_draft('')
            await case.pilot.pause()
            entered=asyncio.Event();release=asyncio.Event()
            original=workspace.load_console_conversation_tree
            async def held(*args,**kwargs):
                entered.set()
                await release.wait()
                return await original(*args,**kwargs)
            with pytest.MonkeyPatch.context() as patches:
                patches.setattr(workspace,'load_console_conversation_tree',held)
                resume=asyncio.create_task(case.console._workspace._resume_console_workspace_conversation(conversation_id,reuse_existing=False,preserve_persisted_scope=True),name='probe-original-resume')
                await asyncio.wait_for(entered.wait(),30)
                snapshot=case.console._session._console_draft_switch_snapshot
                observations.update(old_session=old_id,snapshot_owner=snapshot[0],known_revision=case.console._session._console_visible_draft_revision,live_draft_length=len(case.composer.draft_text()),stage='resume suspended at original tree read, before hydration; no test field mutation')
                try:
                    await case.console._session._close_console_session_tab(old_id)
                    observations['close_error']=None
                except Exception as error:
                    observations['close_error']=repr(error)
                finally:
                    observations['old_session_gone']=all(s.id!=old_id for s in case.chats.sessions())
                    observations['active_after_close']=case.chats.active_session_id
                    release.set()
                    try:
                        observations['resume_result']=await resume
                    except Exception as error:
                        observations['resume_error']=repr(error)
                assert observations['old_session_gone']
                assert observations['close_error'] is not None and 'Unknown Console chat session' in observations['close_error'],observations
                observations['diagnostic_confirmed_retired_snapshot_read']=True
    finally:
        (ROOT/(os.environ.get('UI26_TAG','lifecycle')+'.observation.json')).write_text(json.dumps(observations,indent=2)+'\n')


class Probe:
    def pytest_collection_modifyitems(self,items):
        assert len(items)==1
        items[0].obj=probe


if __name__=='__main__':
    raise SystemExit(pytest.main(['-q','-x','-p','no:cacheprovider','--tb=short','Tests/UI/test_response_rules_end_to_end.py::test_learn_repair_reopen_and_exclude_through_console[ancestry-size1]'],plugins=[Probe()]))
