import pytest, time
@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
 if item.name != 'test_keyboard_send_cancels_idle_refresh_before_run_starts':
  return (yield)
 from tldw_chatbook.UI.Console_Modules.console_spend_projection import ConsoleDraftSpendRefresh
 old_route=ConsoleDraftSpendRefresh.route_edit; old_stop=ConsoleDraftSpendRefresh.stop
 def route(self, *, run_active):
  print('TIMER_TRACE_ROUTE',round(time.monotonic(),3),run_active,repr(self.timer),flush=True)
  return old_route(self,run_active=run_active)
 def stop(self):
  print('TIMER_TRACE_STOP',round(time.monotonic(),3),repr(self.timer),flush=True)
  return old_stop(self)
 ConsoleDraftSpendRefresh.route_edit=route;ConsoleDraftSpendRefresh.stop=stop
 try:
  return (yield)
 finally:
  ConsoleDraftSpendRefresh.route_edit=old_route;ConsoleDraftSpendRefresh.stop=old_stop
