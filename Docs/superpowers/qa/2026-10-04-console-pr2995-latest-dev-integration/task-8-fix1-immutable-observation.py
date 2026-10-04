"""Run the new observation characterization against immutable 40b9 controller."""
import importlib.util
import subprocess
import sys
from pathlib import Path
import Tests.conftest
_REVISION = '40b9aacaa21f7e8c6f2b1989efbb8e8021a1fe77'
_name = 'tldw_chatbook.Chat.console_chat_controller'
_path = 'tldw_chatbook/Chat/console_chat_controller.py'
_spec = importlib.util.spec_from_file_location(_name, _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _module
_source = subprocess.check_output(['git', '-c', 'gc.auto=0', 'show', f'{_REVISION}:{_path}'])
exec(compile(_source, _path, 'exec'), _module.__dict__)
_test_path = 'Tests/Chat/test_console_chat_create_integration.py'
exec(compile(Path(_test_path).read_bytes(), _test_path, 'exec'), globals())
