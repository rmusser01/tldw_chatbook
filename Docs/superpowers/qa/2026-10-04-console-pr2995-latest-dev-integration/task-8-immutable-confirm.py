"""Immutable incoming controller and exact grant-race test attribution."""
import importlib.util
import subprocess
import sys
import Tests.conftest
_REVISION = '5e0341d1ec701865e019eb2fd8a5e2028ab2d474'
def _source(path):
    return subprocess.check_output(['git','-c','gc.auto=0','show',f'{_REVISION}:{path}'])
_name='tldw_chatbook.Chat.console_chat_controller'
_spec=importlib.util.spec_from_file_location(_name,'tldw_chatbook/Chat/console_chat_controller.py')
_module=importlib.util.module_from_spec(_spec)
sys.modules[_name]=_module
exec(compile(_source('tldw_chatbook/Chat/console_chat_controller.py'),_module.__file__,'exec'),_module.__dict__)
exec(compile(_source('Tests/Chat/test_console_chat_create_confirm.py'),'immutable-5e0341/Tests/Chat/test_console_chat_create_confirm.py','exec'),globals())
