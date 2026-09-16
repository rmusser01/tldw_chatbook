"""Finite real profile writes remain admitted through native completion."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_SCRIPT = r'''
import json, os, sys, threading, time
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.RAG_Search import config_profiles as module
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, VectorStoreConfig
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired, default_bootstrap_root

case = sys.argv[1]
root = Path.home() / 'profiles'
owner = module.ConfigProfileManager(root)
profile = module.ProfileConfig(name='Mine', description='test', profile_type='custom',
    rag_config=RAGConfig(vector_store=VectorStoreConfig(type='memory')))
owner.save_profile(profile)
path = root / (profile.id + '.json')
before = path.read_bytes()
# Source-only fixture: imports have completed; no app/runtime is instantiated.
storage._startups.pop((os.getpid(), str(default_bootstrap_root()))).close()

if case == 'existing':
    import inspect
    import pytest
    from Tests.RAG import test_config_profiles as existing
    names = [
        'test_legacy_blob_migrated_to_per_file',
        'test_legacy_blob_migration_isolates_per_entry_failures',
        'test_rename_keeps_id_and_file', 'test_delete_removes_file_and_entry',
        'test_save_profile_rejects_id_collision_with_readonly_builtin',
        'test_legacy_blob_migration_does_not_shadow_readonly_builtin',
        'test_stray_file_named_custom_profiles_json_is_not_destroyed_as_legacy_blob',
        'test_save_profile_does_not_register_when_the_disk_write_fails',
        'test_save_profile_rejects_hard_invalid_rag_config',
        'test_save_profile_accepts_a_valid_rag_config',
        'test_load_degrades_gracefully_on_hand_corrupted_profile_json',
        'test_load_still_skips_a_structurally_unparseable_profile_file',
    ]
    for name in names:
        target = Path.home()/name
        target.mkdir()
        test = getattr(existing, name)
        with pytest.MonkeyPatch.context() as patch:
            values = {'tmp_path': target, 'monkeypatch': patch}
            test(**{key: values[key] for key in inspect.signature(test).parameters})
elif case == 'paused':
    experiment = module.ExperimentConfig(results_dir=Path.home()/'alternate')
    pause = storage._begin_local_pause()
    try:
        calls = [lambda: owner.rename_profile(profile.id, 'Changed'),
            lambda: owner.delete_profile(profile.id), lambda: owner.save_profile(profile),
            lambda: owner._save_one(profile), owner._load_custom_profiles,
            owner._migrate_legacy_blob, lambda: owner.start_experiment(experiment),
            lambda: module.ConfigProfileManager(Path.home()/'new')]
        for call in calls:
            try: call()
            except RecoveryRequired: pass
            else: raise AssertionError('paused mutation was admitted')
        assert path.read_bytes() == before and owner.get_profile(profile.id) is profile
        assert profile.name == 'Mine' and owner._current_experiment is None
        assert not experiment.results_dir.exists() and not (Path.home()/'new').exists()
        owner._current_experiment = experiment
        try: owner.end_experiment()
        except RecoveryRequired: pass
        else: raise AssertionError('paused experiment completion admitted')
        assert owner._current_experiment is experiment
    finally: pause.resume()
else:
    entered, release = threading.Event(), threading.Event()
    errors = []
    original_dump = module.json.dump
    original_save = owner._save_one
    original_unlink = Path.unlink
    observed_paths = []
    original_acquire = getattr(module, 'acquire_storage', storage.acquire_storage)
    def acquire(selected):
        observed_paths.append(str(selected))
        return original_acquire(selected)
    module.acquire_storage = acquire
    def block():
        entered.set()
        assert release.wait(4)
    def dump(*args, **kwargs):
        block()
        if case == 'failure': raise OSError('controlled write failure')
        return original_dump(*args, **kwargs)
    call = lambda: owner.rename_profile(profile.id, 'Renamed')
    module.json.dump = dump
    if case == 'nested':
        module.json.dump = original_dump
        def save(value):
            block()
            return original_save(value)
        owner._save_one = save
    elif case == 'delete':
        def unlink(selected, *args, **kwargs):
            if selected == path: block()
            return original_unlink(selected, *args, **kwargs)
        Path.unlink = unlink
        call = lambda: owner.delete_profile(profile.id)
    elif case == 'selfheal':
        path.rename(root/'Needs Slug.json')
        call = owner._load_custom_profiles
    elif case == 'migration':
        path.unlink()
        (root/'custom_profiles.json').write_text(json.dumps({'profiles':[profile.to_dict()]}))
        call = owner._migrate_legacy_blob
    elif case in ('experiment_start', 'experiment_end'):
        alternate = Path.home()/'alternate'
        alternate.mkdir()
        experiment = module.ExperimentConfig(results_dir=alternate)
        if case == 'experiment_start':
            call = lambda: owner.start_experiment(experiment)
        else:
            owner._current_experiment = experiment
            owner._experiment_results[experiment.experiment_id] = []
            call = owner.end_experiment
    def work():
        try: call()
        except BaseException as error: errors.append(error)
    thread = threading.Thread(target=work)
    thread.start()
    pause = None
    try:
        assert entered.wait(4), errors
        pause = storage._begin_local_pause()
        assert not pause.drain(time.monotonic()), 'native writer escaped census'
    finally:
        release.set()
        thread.join(5)
        module.json.dump = original_dump
        owner._save_one = original_save
        Path.unlink = original_unlink
        if pause is not None: pause.resume()
    assert not thread.is_alive()
    if case == 'failure':
        assert len(errors) == 1 and isinstance(errors[0], OSError), errors
    elif case == 'experiment_start':
        # Existing asdict(Path) JSON failure is preserved, not rewritten here.
        assert len(errors) == 1 and isinstance(errors[0], TypeError), errors
    else: assert not errors, errors
    if case.startswith('experiment_'):
        assert str(alternate) in observed_paths, observed_paths
    if case == 'selfheal':
        assert not (root/'Needs Slug.json').exists() and (root/'needs_slug.json').exists()
    if case == 'migration': assert (root/'custom_profiles.json.migrated').exists()
    if case == 'delete': assert not path.exists() and owner.get_profile(profile.id) is None
    pause = storage._begin_local_pause()
    try: assert pause.drain(time.monotonic()+1)
    finally: pause.resume()
owner.save_profile(profile)
assert json.loads(path.read_text())['name'] == profile.name
print('retired and reopened')
'''


@pytest.mark.parametrize('case', [
    'paused', 'writer', 'nested', 'failure', 'selfheal', 'migration', 'delete',
    'experiment_start', 'experiment_end', 'existing',
])
def test_profile_write_admission(tmp_path, case):
    _run(tmp_path, case, 'success', script=_SCRIPT)
