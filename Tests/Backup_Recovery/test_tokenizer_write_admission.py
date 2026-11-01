"""Actual tokenizer mapping writers retain ordinary storage until return."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import json, sys, threading, time
from pathlib import Path
from tldw_chatbook.Utils import custom_tokenizers as module
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

owner = module.CustomTokenizerManager(str(Path.home()/'tokenizers'))
case = sys.argv[1]
if case == 'constructor':
    target = Path.home()/'new-tokenizers'
    pause = storage._begin_local_pause()
    try:
        try: module.CustomTokenizerManager(str(target))
        except RecoveryRequired: pass
        else: raise AssertionError('paused constructor created directory')
        assert not target.exists()
    finally: pause.resume()
    created = module.CustomTokenizerManager(str(target))
    assert target.is_dir() and created._model_mappings == {}
elif case == 'paused':
    pause = storage._begin_local_pause()
    try:
        for call in (lambda: owner.add_mapping('blocked', 'value'), owner.save_mappings):
            try: call()
            except RecoveryRequired: pass
            else: raise AssertionError('paused tokenizer write admitted')
        assert owner._model_mappings == {}
        assert not (Path(owner.tokenizers_dir)/'mappings.json').exists()
    finally: pause.resume()
else:
    entered, release = threading.Event(), threading.Event()
    original = module.json.dump
    errors = []
    def dump(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        if case == 'failure': raise OSError('controlled write error')
        return original(*args, **kwargs)
    module.json.dump = dump
    original_save = owner.save_mappings
    if case == 'nested':
        module.json.dump = original
        def accepted_save():
            entered.set()
            assert release.wait(3)
            return original_save()
        owner.save_mappings = accepted_save
    def write():
        try: owner.add_mapping('model', 'tokenizer')
        except BaseException as error: errors.append(error)
    worker = threading.Thread(target=write)
    worker.start()
    pause = None
    try:
        assert entered.wait(3)
        pause = storage._begin_local_pause()
        assert not pause.drain(time.monotonic()), 'writer escaped native census'
    finally:
        release.set()
        worker.join(4)
        module.json.dump = original
        owner.save_mappings = original_save
        if pause is not None: pause.resume()
    assert not worker.is_alive()
    assert not errors, errors
    pause = storage._begin_local_pause()
    try: assert pause.drain(time.monotonic()+1)
    finally: pause.resume()
    assert owner._model_mappings == {'model':'tokenizer'}
owner.add_mapping('after', 'resume')
assert json.loads((Path(owner.tokenizers_dir)/'mappings.json').read_text())['after'] == 'resume'
print('retired and reopened')
'''


@pytest.mark.parametrize("case", ["paused", "writer", "failure", "nested", "constructor"])
def test_tokenizer_write_admission(tmp_path, case):
    _run(tmp_path, case, "success", script=_SCRIPT)


_AVAILABILITY = r"""
import sys, json
from pathlib import Path
from Tests.real_profile_guard import install as guard, take_violations
guard()
from Tests.network_guard import install, blocked_attempts
install()
from tldw_chatbook.Utils import custom_tokenizers as module
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery import native_files
from tldw_chatbook.Backup_Recovery.profile_paths import default_config_path
root = default_config_path().parent / 'tokenizers'
root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
case = sys.argv[1]
module.TOKENIZERS_AVAILABLE = True
module._tokenizer_manager = None
original = module.CustomTokenizerManager
calls=[]
def refused(*args, **kwargs):
    calls.append('constructor')
    raise RecoveryRequired('actual-path-required')
module.CustomTokenizerManager=refused
if case == 'no-library':
    module.TOKENIZERS_AVAILABLE=False
    def unexpected(*args, **kwargs):raise AssertionError('absent library performed I/O')
    native_files.pinned_directory=unexpected
    assert module.custom_tokenizers_available() is False
elif case == 'cached':
    class Cached:
        def has_tokenizers(self):return True
    module._tokenizer_manager=Cached()
    assert module.custom_tokenizers_available() is True
elif case in ('missing-root','missing-map'):
    if case == 'missing-map':root.mkdir(mode=0o700)
    before = sorted(str(p) for p in root.parent.rglob('*'))
    assert module.custom_tokenizers_available() is False
    assert sorted(str(p) for p in root.parent.rglob('*')) == before
    assert module._tokenizer_manager is None
elif case == 'real-mapping':
    root.mkdir(mode=0o700)
    (root/'mappings.json').write_text(json.dumps({'model':'installed'}))
    module.CustomTokenizerManager=original
    assert module.custom_tokenizers_available() is True
    assert module._tokenizer_manager._model_mappings == {'model':'installed'}
else:
    if case == 'mapping':
        root.mkdir(mode=0o700)
        (root/'mappings.json').write_text('{}')
    elif case == 'non-directory':root.write_text('not a directory')
    elif case == 'symlink':root.symlink_to(root.parent/'absent-target')
    elif case == 'uncertain':
        def unavailable(*args, **kwargs):raise PermissionError('controlled uncertain parent')
        native_files.pinned_directory=unavailable
    else:raise AssertionError(case)
    try:module.custom_tokenizers_available()
    except RecoveryRequired as error:assert str(error)=='actual-path-required'
    else:raise AssertionError('native refusal swallowed')
    assert calls==['constructor']
assert not blocked_attempts(),blocked_attempts()
assert not take_violations()
print('availability contract verified; retired and reopened')
"""


@pytest.mark.parametrize(
    "case",
    [
        "missing-root",
        "missing-map",
        "no-library",
        "cached",
        "mapping",
        "real-mapping",
        "non-directory",
        "symlink",
        "uncertain",
    ],
)
def test_tokenizer_availability_never_creates_an_absent_optional_store(tmp_path, case):
    _run(tmp_path, case, "success", script=_AVAILABILITY)
