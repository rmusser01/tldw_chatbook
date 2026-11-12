import json
import os
from pathlib import Path
import pytest

records = []
@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    start = len(os.listdir('/dev/fd'))
    yield
    end = len(os.listdir('/dev/fd'))
    if 'test_console_prompt_mixin_forks.py' in item.nodeid:
        records.append({'node': item.nodeid, 'before': start, 'after': end, 'delta': end-start})
def pytest_sessionfinish(session, exitstatus):
    Path('/private/tmp/mixins-final-fd-audit.json').write_text(json.dumps({'exitstatus': exitstatus, 'records':records}, indent=2)+'\n')
