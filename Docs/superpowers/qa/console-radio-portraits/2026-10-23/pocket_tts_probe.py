"""Opt-in real Pocket TTS / mounted Console / native-device qualification."""

import runpy
import tempfile
from pathlib import Path

from Tests.Backup_Recovery.test_home_citation_retirement import _run

HERE = Path(__file__).resolve().parent
BASE = runpy.run_path(str(HERE / "imagegen_import_probe.py"))["CHILD"]
CONFIG = """
[app_tts]
default_provider="openai"
default_model_mode="exact"
default_model="pocket-tts"
default_voice_mode="exact"
default_voice="alba"
default_format="wav"
OPENAI_BASE_URL="http://127.0.0.1:18765/tts"
OPENAI_AUTH_MODE="none"
"""
INSTRUMENT = r"""
import time,wave
import sounddevice as sd
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
native_stream=sd.OutputStream
native_synthesis=OpenAITTSBackend.generate_speech_stream
callbacks,events,submissions,synthesis,painted,streams=[],[],[],[],[],[]
def observed_stream(**options):
 stream_id=len(streams);streams.append(options)
 callback=options['callback'];finished=options['finished_callback']
 def observed(outdata,frames,timing,status):
  try:return callback(outdata,frames,timing,status)
  finally:
   if len(callbacks)<5000:
    submissions.append(outdata.tobytes())
    callbacks.append({'stream':stream_id,'at':time.monotonic(),'current':timing.currentTime,'dac':timing.outputBufferDacTime,'frames':frames,'status':str(status)})
 def complete():
  events.append({'type':'NativeFinished','stream':stream_id,'at':time.monotonic()});finished()
 options.update(callback=observed,finished_callback=complete)
 return native_stream(**options)
async def observed_synthesis(self,request):
 chunks=[]
 async for chunk in native_synthesis(self,request):
  chunks.append(chunk);yield chunk
 synthesis.append({'text':request.input,'data':b''.join(chunks),'at':time.monotonic()})
sd.OutputStream=observed_stream
OpenAITTSBackend.generate_speech_stream=observed_synthesis
"""
SPEECH = r"""
  print('checkpoint: actual Pocket speech',flush=True)
  original_notify=app.notify
  def observe_notice(message,**kwargs):
   print('SPEECH_NOTICE',str(message),flush=True);return original_notify(message,**kwargs)
  app.notify=observe_notice
  print('TTS_CONFIG',{k:app.app_config.get('COMPREHENSIVE_CONFIG_RAW',{}).get('app_tts',{}).get(k) for k in ('default_provider','default_model','default_voice','default_format','OPENAI_BASE_URL','OPENAI_AUTH_MODE')},flush=True)
  store=chat._console_chat_store
  message=store.append_message(store.active_session_id,role=ConsoleMessageRole.ASSISTANT,content='Hello. This is a portrait preview. Notice the pauses. The final word is complete.')
  await chat._sync_native_console_chat_ui()
  outcomes=[]
  assert await chat._message.request_console_message_speech(message.id,outcome_callback=lambda ok:outcomes.append({'ok':ok,'at':time.monotonic()}))
  async with asyncio.timeout(120):
   while not outcomes:
    speech=chat._message.current_speech_presentation()
    snapshot=speech.read() if speech else None
    avatars=list(chat.query(CharacterExpressionAvatar))
    avatar=avatars[0] if avatars else None
    if snapshot is not None:
     painted.append({'at':time.monotonic(),'state':snapshot.state,'mouth':snapshot.mouth,'timing_valid':snapshot.timing_valid,'level_dbfs':snapshot.level_dbfs,'pose':avatar._frame_index if avatar else None,'visible':avatar._visible() if avatar else False})
     if snapshot.state=='playing' and avatar is not None and avatar._frame_index%3!=0 and not (output/'speaking.svg').exists():
      app.save_screenshot('speaking.svg',path=str(output))
    await asyncio.sleep(.02)
  assert outcomes[0]['ok'],{'outcomes':outcomes,'observed':painted[-10:]}
  assert any(row['state']=='playing' and row['timing_valid'] for row in painted),painted[-10:]
  assert any(row['visible'] and row['pose'] is not None and row['pose']%3!=0 for row in painted)
  assert callbacks and synthesis
  data=synthesis[0]['data'];(output/'pocket-speech.wav').write_bytes(data)
  with wave.open(str(output/'pocket-speech.wav'),'rb') as wav:
   rate=wav.getframerate();channels=wav.getnchannels();assert wav.getsampwidth()==2
   pcm=wav.readframes(wav.getnframes())
  submitted=b''.join(submissions)
  assert pcm in submitted,'Device submission lost actual synthesized speech'
  with wave.open(str(output/'device-submitted.wav'),'wb') as wav:
   wav.setnchannels(channels);wav.setsampwidth(2);wav.setframerate(rate);wav.writeframes(submitted)
  result.update(live_audio=True,synthesis_text=synthesis[0]['text'],sample_rate=rate,channels=channels,synthesized_pcm_sha256=hashlib.sha256(pcm).hexdigest(),full_speech_submitted=True,callbacks=callbacks,events=events,observations=painted,outcomes=outcomes,output_device=sd.query_devices(kind='output')['name'],physical_loopback=False)
  app.save_screenshot('finished.svg',path=str(output))
  # The HTTP adapter is WAV-only. This separate opt-in probe feeds the same real
  # server's arriving PCM payload into the existing transport, without adding an adapter.
  import httpx,struct
  from tldw_chatbook.Audio.streaming_sink import StreamingPcmSink,pump
  from tldw_chatbook.Audio.speech_presentation import SpeechPresentation,SpeechPresentationOwner
  stream_source=[];http_chunks=[]
  owner=SpeechPresentationOwner('real-pocket-stream',db,store.active_session_id,store.active_session_epoch(),cid,None,None)
  channel=SpeechPresentation(owner)
  current=chat._message.bind_speech_presentation(channel,lambda:store.active_session_id==owner.session_id and store.active_session_epoch()==owner.session_epoch)
  streaming=StreamingPcmSink(presentation=channel,on_event=lambda event:events.append({'type':type(event).__name__,'at':time.monotonic()}))
  streaming.open(24000,1)
  assert streaming.state=='open',streaming.fail_reason
  async def chunks():
   head=bytearray();seen=0
   async with httpx.AsyncClient(timeout=90,trust_env=False) as client:
    async with client.stream('POST','http://127.0.0.1:18765/tts',data={'text':'Hello. Pause for a moment. The final word is complete.','voice_url':'alba'}) as response:
     response.raise_for_status()
     async for data in response.aiter_bytes(4800):
      seen+=len(data);assert seen<8*1024*1024
      http_chunks.append({'at':time.monotonic(),'bytes':len(data)})
      if len(head)<44:
       need=44-len(head);head.extend(data[:need]);data=data[need:]
       if len(head)<44:continue
       assert head[:4]==b'RIFF' and head[8:16]==b'WAVEfmt ' and head[36:40]==b'data'
       assert struct.unpack_from('<IHHIIHH',head,16)==(16,1,1,24000,48000,2,16)
      if data:
       assert len(data)%2==0
       stream_source.append(data);yield data
   events.append({'type':'HttpStreamEnded','at':time.monotonic()})
  stream_task=asyncio.create_task(pump(streaming,chunks()))
  stream_seen=[]
  async with asyncio.timeout(90):
   while not stream_task.done():
    snap=channel.read()
    if snap is not None:stream_seen.append({'at':time.monotonic(),'state':snap.state,'mouth':snap.mouth,'timing_valid':snap.timing_valid})
    await asyncio.sleep(.02)
  stream_result=await stream_task
  await asyncio.to_thread(streaming.settle)
  assert stream_result.outcome=='drained',stream_result
  assert current() and len(http_chunks)>1
  assert any(r['state']=='playing' and r['timing_valid'] and r['mouth']!='rest' for r in stream_seen)
  assert b''.join(stream_source) in b''.join(submissions)
  with wave.open(str(output/'http-stream-source.wav'),'wb') as wav:
   wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(b''.join(stream_source))
  result['http_stream']={'chunks':http_chunks,'observations':stream_seen,'complete_pcm_submitted':True,'app_adapter_streaming':False}

  from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSPlaybackEvent
  async def request(text):
   msg=store.append_message(store.active_session_id,role=ConsoleMessageRole.ASSISTANT,content=text)
   await chat._sync_native_console_chat_ui()
   assert await chat._message.request_console_message_speech(msg.id)
   await ready(lambda:chat._message.current_speech_presentation() is not None and chat._message.current_speech_presentation().read() is not None and chat._message.current_speech_presentation().read().state=='playing')
   return msg,chat._message.current_speech_presentation()
  print('checkpoint: real stop/replacement',flush=True)
  msg,stopped=await request('This speech will stop during the final part of its output, before the last word finishes.')
  with wave.open(__import__('io').BytesIO(synthesis[-1]['data'])) as wav:duration=wav.getnframes()/wav.getframerate()
  sid=len(streams)-1
  await ready(lambda:time.monotonic()-next(row['at'] for row in callbacks if row['stream']==sid)>max(.1,duration-.4))
  stop_ack=[];stop_at=time.monotonic()
  await chat._message._dispatch_console_speech_stop_event(TTSPlaybackEvent(action='stop',message_id=msg.id,playback_lifecycle=chat._message._console_speech_owner,outcome_callback=stop_ack.append))
  assert stop_ack==[True] and stopped.read().state=='stopped'
  first,old=await request('This is the first speaking message. Another message will replace its remaining audio.')
  second,new=await request('The replacement message is speaking now. Switching chats must release this exact voice owner.')
  assert old is not new and old.read().state=='stopped'
  assert chat._message.current_speech_presentation() is new
  store.create_session(title='Actual speech context switch',runtime_backend='local')
  await chat._sync_native_console_chat_ui()
  await ready(lambda:new.read().state=='stopped')
  assert chat._message.current_speech_presentation() is None and chat._character._read_character_speech() is None
  result['controls']={'stop_ack':True,'stop_pending_tail_seconds_approx':.4,'stop_at':stop_at,'replacement_retired_old':True,'conversation_switch_retired_voice':True}
  print('checkpoint: real sentence/listening handoff',flush=True)
  await _start_character_context_chat(chat,key,db,'Samira — artwork qualification',lambda:app.chachanotes_db is db)
  from tldw_chatbook.Chat.console_hands_free import HandsFreeController
  from tldw_chatbook.Chat.reply_sentence_sequencer import SentenceSequencer
  from tldw_chatbook.UI.Console_Modules.hands_free import ConsoleHandsFreeSession
  handoff=[]
  fsm=HandsFreeController(emit=lambda intent:handoff.append({'type':type(intent).__name__,'state':getattr(intent,'state',None),'at':time.monotonic()}),send_delay_seconds=1,acoustic_barge_in=False)
  hands=chat._hands_free
  seq=SentenceSequencer(speak=hands._dispatch_console_hands_free_speak,stop_speech=hands._stop_console_hands_free_speech)
  hand_session=ConsoleHandsFreeSession(controller=fsm,sequencer=seq,reply_id='pocket-handoff')
  hands._console_hands_free=hand_session
  seq.on_drained=hands._on_console_hands_free_sequencer_drained
  # Simulated capture-end input; microphone/STT hardware is deliberately absent.
  fsm.enter(capture_live=True);fsm.on_capture_ended(had_segments=True,limit_hit=True);fsm.on_reply_started()
  first_stream=len(streams)
  seq.feed('The first sentence finishes completely. ')
  seq.feed('The second sentence finishes completely. ')
  seq.reply_completed();fsm.on_reply_finished()
  assert fsm.state=='speaking' and not seq.drained
  async with asyncio.timeout(90):
   while fsm.state!='listening' or not seq.drained:await asyncio.sleep(.02)
  assert hand_session.utterances_dispatched==2 and hand_session.utterances_failed==0
  assert len(streams)==first_stream+2
  def final_dac_end(stream):
   return max(row['at']+row['dac']-row['current']+row['frames']/24000 for row in callbacks if row['stream']==stream)
  next_start=min(row['at'] for row in callbacks if row['stream']==first_stream+1)
  capture_open=next(row['at'] for row in handoff if row['type']=='OpenCapture')
  assert next_start>=final_dac_end(first_stream)-.005
  assert capture_open>=final_dac_end(first_stream+1)-.005
  result['hands_free']={'utterances':2,'failures':0,'next_sentence_after_previous_tail':True,'listening_intent_after_final_tail':True,'microphone_hardware_exercised':False,'events':handoff}
  hands._console_hands_free=None


"""


def main() -> None:
    """Reuse the actual-app artwork journey and the real configured TTS route."""
    repo = HERE.parents[4]
    assets = repo / "output/imagegen/console-radio-samira-2026-10-23"
    source = repo / "tldw_chatbook/assets/characters/samira/expressions/neutral.webp"
    script = BASE.replace(
        "for name in ('sounddevice', 'pyaudio'): sys.modules[name] = None", ""
    )
    script = script.replace(
        "install()",
        "install(); from Tests.network_guard import set_mode,NetworkMode; set_mode(NetworkMode.LOOPBACK_ONLY)",
        1,
    )
    script = script.replace(
        "selector.chmod(0o600)",
        "selector.write_text(selector.read_text()+"
        + repr(CONFIG)
        + "); selector.chmod(0o600)",
    )
    script = script.replace("async def main():", INSTRUMENT + "\nasync def main():", 1)
    script = script.replace("output=root/'preview'", "output=root/'pocket-preview'")
    script = script.replace(
        "  await chat._character._apply_console_character_choice_async(ConsoleCharacterChoice(cid,'Samira — artwork qualification','new'))",
        "  from tldw_chatbook.Character_Chat.character_conversation_navigation import ResolvedLocalCharacterKey\n"
        "  from tldw_chatbook.UI.Console_Modules.wiring import _start_character_context_chat\n"
        "  key=ResolvedLocalCharacterKey(db.get_local_authority_id(),cid)\n"
        "  await _start_character_context_chat(chat,key,db,'Samira — artwork qualification',lambda:app.chachanotes_db is db)\n"
        "  assert chat._console_chat_store.ensure_session().assistant_authority_id==key.data_authority_id",
    )
    # Artwork authoring has its own probe; seed its reviewed set before mounting.
    script = script.replace(
        " async with app.run_test(size=(180,72)) as pilot:",
        " from tldw_chatbook.Character_Chat.portrait_performance import capture_performance_source,normalize_performance_frames\n"
        " source,_=capture_performance_source(db,character_id=cid,requested_state='idle')\n"
        " frames=normalize_performance_frames({role:(root/'selected'/(role+'.png')).read_bytes() for role in FRAME_ROLES})\n"
        " db.apply_character_portrait_performance(source,frames,expected_revision=0)\n"
        " async with app.run_test(size=(180,72)) as pilot:",
    )
    authoring = script.index("  print('checkpoint: editor'")
    console = script.index("  app.post_message(NavigateToScreen('chat'))", authoring)
    script = script[:authoring] + script[console:]
    finish = script.index("  personas,reopened=await edit()")
    done = script.index(" assert not blocked_attempts()", finish)
    script = script[:finish] + script[done:]
    start = script.index("  print('checkpoint: console fitted'")
    end = script.index(" assert not blocked_attempts()", start)
    script = script[:start] + SPEECH + script[end:]
    script = script.replace("__ASSET_ROOT__", repr(str(assets))).replace(
        "__SOURCE__", repr(str(source))
    )
    (assets / "pocket-preview").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="radio-pocket-tts-") as folder:
        try:
            _run(Path(folder), "speech", "pocket", script=script, timeout=240)
        finally:
            log = Path(folder) / "restart-output.log"
            if log.is_file():
                (assets / "pocket-preview/probe.log").write_text(log.read_text())
    print("PASS: actual Pocket TTS speech via mounted Console and native WAV sink")


if __name__ == "__main__":
    main()
