"""Attribute only the affected controller cap against immutable 40b9."""
import subprocess
import Tests.conftest
from Tests.Architecture import test_module_size_ratchet as caps
_PATH = 'tldw_chatbook/Chat/console_chat_controller.py'
_REVISION = '40b9aacaa21f7e8c6f2b1989efbb8e8021a1fe77'
def test_immutable_controller_cap():
    data = subprocess.check_output(['git', '-c', 'gc.auto=0', 'show', f'{_REVISION}:{_PATH}'])
    lines = len(data.decode().splitlines())
    assert lines <= caps._BUDGETS[_PATH], (lines, caps._BUDGETS[_PATH])
