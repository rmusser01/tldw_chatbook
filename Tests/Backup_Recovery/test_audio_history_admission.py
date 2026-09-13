"""Local audio history scopes cover accepted generation and publication."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import asyncio, copy, json, sys, time
from pathlib import Path
from tldw_chatbook.Audio_Services_Interop import local_audio_services_service as module
from tldw_chatbook.Backup_Recovery import storage_admission as storage

async def main():
    entered, release = asyncio.Event(), asyncio.Event()
    case = sys.argv[1]
    async def generator(**kwargs):
        entered.set()
        await release.wait()
        if case == 'error': raise ValueError('original generation failure')
        return b'completed audio'
    path = Path.home()/'audio.json'
    owner = module.LocalAudioServicesService(history_store_path=path, tts_audio_generator=generator)
    if case == 'paused':
        module._maintenance_close_admission()
        assert await module._maintenance_drain(time.monotonic()+1)
        for call in (lambda:owner.create_audio_speech({'input':'hello'}),
                     lambda:owner.update_tts_history_favorite(1, {'favorite':True}),
                     lambda:owner.delete_tts_history_entry(1)):
            try: await call()
            except RuntimeError as error: assert str(error)=='audio_history_paused_for_maintenance'
            else: raise AssertionError('new history call admitted')
        try: owner._persist_history()
        except RuntimeError: pass
        else: raise AssertionError('direct history publication admitted')
        assert not entered.is_set() and not path.exists() and owner._history_records==[]
        module._maintenance_resume()
    else:
        task = asyncio.create_task(owner.create_audio_speech({'input':'hello'}))
        await entered.wait()
        module._maintenance_close_admission()
        assert not await module._maintenance_drain(time.monotonic())
        pause = storage._begin_local_pause()
        try:
            assert not pause.drain(time.monotonic()), 'accepted generation lacks ordinary lease'
            if case == 'cancel': task.cancel()
            release.set()
            result = (await asyncio.gather(task, return_exceptions=True))[0]
            assert await module._maintenance_drain(time.monotonic()+1)
            assert pause.drain(time.monotonic()+1)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            pause.resume()
            module._maintenance_resume()
        if case == 'cancel': assert isinstance(result, asyncio.CancelledError) and not path.exists()
        elif case == 'error': assert isinstance(result, ValueError) and not path.exists()
        else:
            assert result['content']==b'completed audio'
            assert json.loads(path.read_text())['items'][0]['text']=='hello'
    owner.tts_audio_generator=lambda **kwargs:b'resumed'
    await owner.create_audio_speech({'input':'after'})
    before=copy.deepcopy(owner._history_records)
    pause=storage._begin_local_pause()
    try:
        try: await owner.update_tts_history_favorite(before[0]['id'], {'favorite':True})
        except RuntimeError: pass
        else: raise AssertionError('global pause allowed favorite write')
        assert owner._history_records==before
    finally: pause.resume()
    print('retired and reopened')
asyncio.run(main())
'''


@pytest.mark.parametrize("case", ["paused", "accepted", "cancel", "error"])
def test_audio_history_admission(tmp_path, case):
    _run(tmp_path, case, "success", script=_SCRIPT)
