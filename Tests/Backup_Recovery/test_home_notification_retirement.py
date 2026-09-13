"""Home notification refresh retires only its own installed worker cache."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r"""
import asyncio,sqlite3,sys
from pathlib import Path
from types import MethodType
import pytest
from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Notifications.client_notifications_service import ClientNotificationsService
from tldw_chatbook.Home.active_work_adapter import LocalNotificationHomeActiveWorkAdapter
route,outcome=sys.argv[1:]
store_type=type('CustomStore',(ClientNotificationsDB,),{}) if route=='custom_store' else ClientNotificationsDB
store=store_type(':memory:' if route=='memory' else Path.home()/'notifications.db')
service_type=type('CustomService',(ClientNotificationsService,),{}) if route=='custom_service' else ClientNotificationsService
class ProxyStore:
 def list_notifications(self,**kwargs):return store.list_notifications(**kwargs)
service=service_type(store=ProxyStore() if route=='custom_proxy' else store)
if route=='custom_store_method':
 original_list=store.list_notifications
 def custom_list(self,**kwargs):return original_list(**kwargs)
 store.list_notifications=MethodType(custom_list,store)
adapter_type=type('CustomAdapter',(LocalNotificationHomeActiveWorkAdapter,),{}) if route=='custom_adapter' else LocalNotificationHomeActiveWorkAdapter
adapter=adapter_type(notification_service=service)
if route=='custom_method':
 original=service.list_queue
 def custom(self,**kwargs):return original(**kwargs)
 service.list_queue=MethodType(custom,service)
observed=[]
original_open=store_type._open_connection
def opened(self):
 connection=original_open(self)
 observed.append(connection)
 return connection
store_type._open_connection=opened
class Refuse:
 def require_allowed(self,**kwargs):
  store._held_connection()
  raise ValueError('expected read failure')
if outcome=='error':service.policy_enforcer=Refuse()
main_connection=store._held_connection()
observed.clear()
def check():
 if outcome=='borrowed':connection=store._held_connection()
 assert adapter._unread_notification_count()==0
 connection=observed[-1] if observed else main_connection
 if route=='native' and outcome!='borrowed':
  with pytest.raises(sqlite3.ProgrammingError):connection.execute('SELECT 1')
 else:assert connection.execute('SELECT 1').fetchone()[0]==1
 store.close()
if route=='memory':check()
else:
 asyncio.run(asyncio.to_thread(check))
 assert main_connection.execute('SELECT 1').fetchone()[0]==1
 store.close()
store_type._open_connection=original_open
print('retired and reopened')
"""

@pytest.mark.parametrize('route',['native','custom_store','custom_service','custom_adapter','custom_method','custom_proxy','custom_store_method','memory'])
def test_notification_count_preserves_foreign_or_custom_lifetimes(tmp_path,route):
    _run(tmp_path,route,'success',script=_SCRIPT)

@pytest.mark.parametrize('outcome',['borrowed','error'])
def test_notification_count_owns_only_current_callback(tmp_path,outcome):
    _run(tmp_path,'native',outcome,script=_SCRIPT)
