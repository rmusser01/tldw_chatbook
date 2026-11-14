import pytest,pathlib,json
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
 from tldw_chatbook.UI.Wizards import first_run_setup_state as state
 from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer
 original=[]
 for owner,name in [(state,'build_first_run_provider_commit'),(SetupWizardContainer,'_bind_provider_write_expectation')]:
  old=getattr(owner,name);original.append((owner,name,old))
  def observe(*args,_old=old,_name=name,**kwargs):
   try:return _old(*args,**kwargs)
   except Exception as exc:
    with pathlib.Path(__file__).with_name('wizard-observations.jsonl').open('a') as out:out.write(json.dumps({'function':_name,'exception':type(exc).__name__,'message':str(exc)})+'\n')
    raise
  setattr(owner,name,observe)
 try:yield
 finally:
  for owner,name,old in original:setattr(owner,name,old)
