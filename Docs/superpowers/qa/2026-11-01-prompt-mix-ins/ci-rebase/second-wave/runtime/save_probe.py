import pytest,pathlib,json
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
 from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
 old=ChatPersistenceService.save_conversation_mixin_attempt
 def observe(self,*args,**kwargs):
  batch=kwargs['batch'];row={'attempt':batch.attempt_id,'user':batch.user_message_id,'assistant':batch.assistant_message_id,'count':len(batch.snapshot.items),'replaces':batch.replaces_attempt_id,'retry_of':batch.retry_of_attempt_id}
  try:return old(self,*args,**kwargs)
  except Exception as exc:row['exception']=type(exc).__name__;row['message']=str(exc);raise
  finally:
   with pathlib.Path(__file__).with_name('receipt-save-observations.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
 ChatPersistenceService.save_conversation_mixin_attempt=observe
 try:yield
 finally:ChatPersistenceService.save_conversation_mixin_attempt=old
