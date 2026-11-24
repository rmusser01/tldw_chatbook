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
    # TASK-34100.16.1: constructing writes nothing, paused or not; the first
    # admitted write creates the folder, owner-only.
    target = Path.home()/'new-tokenizers'
    pause = storage._begin_local_pause()
    try:
        paused = module.CustomTokenizerManager(str(target))
        try: paused.save_mappings()
        except RecoveryRequired: pass
        else: raise AssertionError('paused tokenizer write created directory')
        assert not target.exists()
    finally: pause.resume()
    created = module.CustomTokenizerManager(str(target))
    assert not target.exists() and created._model_mappings == {}
    created.save_mappings()
    assert target.is_dir() and target.stat().st_mode & 0o077 == 0
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
elif case == 'install':
    # TASK-34100.16.1 review: the tokenizer copy runs under the same lease
    # that creates the folder, so a backup's pause waits for it.
    import shutil
    module.TOKENIZERS_AVAILABLE = False  # the copy is under test, not parsing
    source = Path.home()/'source-tokenizer.json'
    source.write_text('{}')
    entered, release = threading.Event(), threading.Event()
    original_copy = shutil.copy2
    def copy2(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original_copy(*args, **kwargs)
    shutil.copy2 = copy2
    results = []
    worker = threading.Thread(
        target=lambda: results.append(owner.install_tokenizer(str(source), 'held'))
    )
    worker.start()
    pause = None
    try:
        assert entered.wait(3)
        pause = storage._begin_local_pause()
        assert not pause.drain(time.monotonic()), 'tokenizer copy escaped native census'
    finally:
        release.set()
        worker.join(4)
        shutil.copy2 = original_copy
        if pause is not None: pause.resume()
    assert results == [True], results
    assert (Path(owner.tokenizers_dir)/'held.json').read_text() == '{}'
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


@pytest.mark.parametrize(
    "case", ["paused", "writer", "failure", "nested", "constructor", "install"]
)
def test_tokenizer_write_admission(tmp_path, case):
    _run(tmp_path, case, "success", script=_SCRIPT)
