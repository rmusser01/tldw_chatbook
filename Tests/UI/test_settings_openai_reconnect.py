"""Explicit Settings review consumes the actual restored OpenAI owner APIs."""

import pytest

from Tests.Backup_Recovery.test_activation_openai_reconnect import (
    _ISOLATED,
    _ISOLATED_CHILD,
    _SETUP,
)
from Tests.Backup_Recovery.test_home_citation_retirement import _run


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """The private child owns its actual Settings host; no parent app is needed."""


_UI = r"""
import asyncio
from types import SimpleNamespace
from textual.app import App
from textual.widgets import Button,Input
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
EFFECT_FIXTURE
instance=SimpleNamespace(app_config=config.load_settings())
import threading
entered=threading.Event();release=threading.Event()
if mode=='late':
 original_prepare=recovery.prepare_openai_reconnect
 def delayed_prepare():
  result=original_prepare();entered.set();assert release.wait(10);return result
 recovery.prepare_openai_reconnect=delayed_prepare
if mode=='accepted-cancel':
 original_write=recovery._write
 def delayed_write(*args):
  result=original_write(*args);entered.set();assert release.wait(10);return result
 recovery._write=delayed_write
def denied():
 try:calls.chat_with_openai([{'role':'user','content':'hello'}],model='gpt-4o-mini')
 except recovery.ProviderReconnectRequired:pass
 else:raise AssertionError('unreviewed OpenAI request executed')
 assert not effects,effects
 assert not blocked_attempts(),blocked_attempts()
 unrelated_inactive()
class Harness(App):
 def on_mount(self):
  screen=SettingsScreen(instance)
  screen.active_category=SettingsCategoryId.PROVIDERS_MODELS.value
  screen._navigation_provider='openai'
  self.push_screen(screen)
async def main():
 app=Harness()
 async with app.run_test(size=(120,45)) as pilot:
  screen=app.screen
  await app.workers.wait_for_complete();await pilot.pause()
  button=screen.query_one('#settings-openai-reconnect-review',Button)
  assert button.display
  if mode=='dirty':
   screen.query_one('#settings-provider-api-key',Input).value='unsaved-private-key'
   await pilot.pause()
  button.focus();await pilot.press('enter')
  if mode in ('dirty','late'):
   if mode=='late':
    assert await asyncio.to_thread(entered.wait,5)
    screen.query_one('#settings-provider-endpoint-value',Input).value='https://changed.example/v1'
    await pilot.pause();release.set()
   await app.workers.wait_for_complete();await pilot.pause()
   assert app.screen is screen
   denied()
   return
  async with asyncio.timeout(10):
   while not isinstance(app.screen,ConfirmationDialog):await asyncio.sleep(.02)
  dialog=app.screen
  text=str(dialog.message)
  assert 'https://api.openai.com/v1' in text
  assert 'env:OPENAI_API_KEY' in text
  assert witness['generation'] in text
  assert 'owned-test-key' not in text
  assert not effects
  if mode in ('edit','provider','navigation','provider-navigation','external-endpoint','external-key'):
   if mode=='edit':
    screen.query_one('#settings-provider-api-key',Input).value='unsaved-private-key'
   elif mode=='provider':
    screen._apply_provider_value_change('anthropic')
   elif mode=='provider-navigation':
    screen.apply_navigation_context({'category':SettingsCategoryId.PROVIDERS_MODELS.value,'provider':'anthropic'})
    screen.apply_navigation_context({'category':SettingsCategoryId.PROVIDERS_MODELS.value,'provider':'openai'})
   elif mode=='navigation':
    screen._select_category(SettingsCategoryId.OVERVIEW.value)
    screen._select_category(SettingsCategoryId.PROVIDERS_MODELS.value)
   else:
    path=Path(entry.config)
    text=path.read_text()
    if mode=='external-endpoint':text=text.replace('https://api.openai.com/v1','https://changed.example/v1')
    else:
     text=text.replace('[api_settings.openai]','[api_settings.openai]\napi_key="changed-private-key"')
     import tomllib
     assert tomllib.loads(text)['api_settings']['openai']['api_key']=='changed-private-key'
    path.write_text(text)
   await pilot.pause()
  if mode=='cancel':
   dialog.query_one('#cancel-button',Button).focus();await pilot.press('enter')
  else:
   dialog.query_one('#confirm-button',Button).focus();await pilot.press('enter')
  if mode=='accepted-cancel':
   assert await asyncio.to_thread(entered.wait,5)
   from tldw_chatbook.Backup_Recovery.control_records import admission_authority
   from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
   worker=next(w for w in app.workers if w.group=='settings-openai-reconnect-confirm')
   worker.cancel()
   await app.pop_screen()
   try:
    with admission_authority(bootstrap.default_bootstrap_root()).maintenance(tuple(witness['namespaces']),.03):
     raise AssertionError('cancelled confirmation lost native source hold')
   except AdmissionTimeout:pass
   release.set()
   from textual.worker import WorkerCancelled
   try:await worker.wait()
   except WorkerCancelled:pass
   else:raise AssertionError("cancelled confirmation waiter did not cancel")
  else:
   await app.workers.wait_for_complete()
  await pilot.pause()
  if mode not in ('approve','accepted-cancel'):
   denied()
   return
  answer=calls.chat_with_openai([{'role':'user','content':'hello'}],model='gpt-4o-mini')
  assert answer['choices'][0]['message']['content']=='answer'
  assert len([e for e in effects if isinstance(e,tuple)])==1
  unrelated_inactive()
 assert not blocked_attempts(),blocked_attempts()
asyncio.run(main())
print('verified isolated reconnect')
""".replace("EFFECT_FIXTURE", _SETUP[_SETUP.index("effects=[]") :])


@pytest.mark.parametrize(
    "mode",
    [
        "approve", "cancel", "dirty", "edit", "provider", "navigation",
        "provider-navigation", "external-endpoint", "external-key", "late",
        "accepted-cancel",
    ],
)
def test_actual_settings_confirms_only_selected_restored_openai(tmp_path, mode):
    child = _ISOLATED_CHILD.split("if mode=='approve':", 1)[0] + _UI
    script = _ISOLATED.replace("for mode in ('approve','reopen'):", "for mode in (" + repr(mode) + ",):")
    script = script.replace("timeout=20", "timeout=45")
    _run(tmp_path, "settings", "ui", script="CHILD=" + repr(child) + "\n" + script, timeout=70)
