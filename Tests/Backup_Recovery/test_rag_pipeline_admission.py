"""Actual pipeline copy/export writers retain ordinary storage admission."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import os, sys, shutil, threading, time
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook import config
from tldw_chatbook.RAG_Search import pipeline_loader as loader_module
from tldw_chatbook.RAG_Search import pipeline_builder_simple as builder
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired, default_bootstrap_root

route, mode = sys.argv[1:]
root = Path.home()
defaults = root/'defaults'
defaults.mkdir()
payload = '[pipelines.example]\nname="Example"\ndescription="Test"\ntype="functional"\nenabled=true\nsteps=[]\n'
(defaults/'rag_pipelines.toml').write_text(payload)
owner = loader_module.PipelineLoader(defaults)
destination = config._get_effective_config_path().parent/'rag_pipelines.toml'
if route == 'builder':
    # Select private shipped-equivalent defaults without importing an engine.
    builder.__file__ = str(root/'RAG_Search'/'pipeline_builder_simple.py')
    (root/'Config_Files').mkdir()
    (root/'Config_Files'/'rag_pipelines.toml').write_text(payload)
    builder._TOML_PIPELINES = None
    call = builder.load_pipelines_from_toml
elif route == 'loader':
    call = owner.load_pipeline_config
else:
    destination = root/'export.toml'
    destination.write_text('preserved')
    owner.pipelines['example'] = loader_module.PipelineConfig(
        id='example', name='Example', description='Test', type='functional')
    call = lambda: owner.export_pipeline_config('example', destination)
storage._startups.pop((os.getpid(), str(default_bootstrap_root()))).close()

if mode == 'paused':
    before = destination.read_bytes() if destination.exists() else None
    pause = storage._begin_local_pause()
    try:
        try: call()
        except RecoveryRequired: pass
        else: raise AssertionError('paused pipeline writer admitted')
        assert (destination.read_bytes() if destination.exists() else None) == before
        if route == 'builder': assert builder._TOML_PIPELINES is None
        elif route == 'loader': assert owner.pipelines == {}
    finally: pause.resume()
else:
    entered, release = threading.Event(), threading.Event()
    original = loader_module.toml.dump if route == 'export' else shutil.copy2
    errors, results = [], []
    def native(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        if mode == 'failure': raise OSError('controlled native error')
        return original(*args, **kwargs)
    if route == 'export': loader_module.toml.dump = native
    else: shutil.copy2 = native
    def work():
        try: results.append(call())
        except BaseException as error: errors.append(error)
    thread = threading.Thread(target=work)
    thread.start()
    pause = None
    try:
        assert entered.wait(4), errors
        pause = storage._begin_local_pause()
        assert not pause.drain(time.monotonic()), 'native pipeline write escaped census'
    finally:
        release.set()
        thread.join(5)
        if route == 'export': loader_module.toml.dump = original
        else: shutil.copy2 = original
        if pause is not None: pause.resume()
    assert not thread.is_alive() and not errors, errors
    if route == 'export': assert results == [mode != 'failure']
    if mode == 'failure' and route != 'export':
        assert not destination.exists()
        if route == 'builder': assert 'example' in builder._TOML_PIPELINES
        else: assert 'example' in owner.pipelines
    pause = storage._begin_local_pause()
    try: assert pause.drain(time.monotonic()+1)
    finally: pause.resume()
if route == 'builder': builder._TOML_PIPELINES = None
call()
if route == 'export':
    assert loader_module.tomllib.loads(destination.read_text())['pipelines']['example']['name'] == 'Example'
else: assert destination.read_text() == payload
print('retired and reopened')
'''


@pytest.mark.parametrize('route', ['loader', 'builder', 'export'])
@pytest.mark.parametrize('mode', ['paused', 'writer', 'failure'])
def test_pipeline_write_admission(tmp_path, route, mode):
    _run(tmp_path, route, mode, script=_SCRIPT)
