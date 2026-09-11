"""Recovery enumerates actual voice producers without constructing runtimes."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


def test_discovery_keeps_configured_backend_and_fixed_catalog_sources(tmp_path):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.models import DiscoveryContext, DISCOVERY_CONTEXT_KEY
from tldw_chatbook.TTS.recovery import recovery_adapters
root = Path(sys.argv[1])
shared = Path.home() / '.config' / 'tldw_cli'
roots = [root / 'configured-chatterbox', root / 'configured-higgs', root / 'backend-higgs', shared / 'chatterbox_voices', shared / 'higgs_voices']
files = []
for index, selected in enumerate(roots):
    selected.mkdir(parents=True)
    data = selected / ('chatterbox_profiles.json' if index in (0, 3) else 'voice_profiles.json')
    data.write_text('{}')
    files.append(data)
config = {
    DISCOVERY_CONTEXT_KEY: DiscoveryContext(root / 'profile' / 'config.toml', 'fixture'),
    'app_tts': {'CHATTERBOX_VOICE_DIR': str(roots[0])},
    'HiggsSettings': {'voice_samples_dir': str(roots[1])},
    'HIGGS_VOICE_SAMPLES_DIR': str(roots[2]),
}
voices = next(adapter for adapter in recovery_adapters() if adapter.owner_id == 'tts.voices')
items = voices.discover(config)
included = {item.path for item in items if item.status == 'included'}
assert set(files) <= included, ('actual saved voice sources omitted', set(files) - included)
assert len([item.path for item in items if item.path]) == len({item.path for item in items if item.path})
assert not any(item.status == 'unsupported' for item in items)
assert all(path.read_text() == '{}' for path in files)
assert 'tldw_chatbook.TTS.backends.higgs' not in sys.modules
assert 'tldw_chatbook.TTS.backends.chatterbox' not in sys.modules
""",
    )


def test_discovery_rejects_invalid_paths_and_preserves_overlapping_source_graph(
    tmp_path,
):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.models import DiscoveryContext, DISCOVERY_CONTEXT_KEY
from tldw_chatbook.TTS.recovery import recovery_adapters
root = Path(sys.argv[1])
outer = root / 'voices'
inner = outer / 'nested'
inner.mkdir(parents=True)
selected = inner / 'voice_profiles.json'
selected.write_text('{}')
config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(root / 'config.toml', 'fixture'), 'app_tts': {'CHATTERBOX_VOICE_DIR': str(outer)}, 'HiggsSettings': {'voice_samples_dir': str(inner)}, 'HIGGS_VOICE_SAMPLES_DIR': str(inner)}
voices = recovery_adapters()[0]
items = voices.discover(config)
assert sum(item.path == selected for item in items) == 1, 'overlapping actual voice roots duplicated a physical spelling'
ids = {item.logical_id for item in items}
for item in items:
    if item.metadata is not None:
        assert item.metadata.root_id in ids
        if item.metadata.parent_id is not None:
            assert item.metadata.parent_id in ids
for bad in (False, 1, [], {}, ''):
    config['HIGGS_VOICE_SAMPLES_DIR'] = bad
    try:
        voices.discover(config)
    except ValueError as error:
        assert str(error) == 'invalid_voice_path'
    else:
        raise AssertionError(('invalid configured source accepted', bad))
assert selected.read_text() == '{}'
""",
    )


@pytest.mark.parametrize("stage", ["global", "backend"])
def test_discovery_covers_actual_final_backend_configuration_sources(tmp_path, stage):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.models import DiscoveryContext, DISCOVERY_CONTEXT_KEY
from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager
from tldw_chatbook.TTS.legacy_bridge import legacy_provider_config, resolve_legacy_route
from tldw_chatbook.TTS.recovery import recovery_adapters
root = Path(sys.argv[1])
manager = object.__new__(TTSBackendManager)
manager._check_cuda_available = lambda: False
voices = recovery_adapters()[0]
for stage in (sys.argv[2],):
    config = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(root / 'config.toml', 'fixture'),
        'global_tts_settings': {
            'CHATTERBOX_VOICE_DIR': str(root / 'global-chatterbox'),
            'KOKORO_VOICE_BLENDS_DIR': str(root / 'global-kokoro'),
        },
        'HIGGS_VOICE_SAMPLES_DIR': str(root / 'top-higgs'),
    }
    routes = (
        ('local_chatterbox_default', 'CHATTERBOX_VOICE_DIR', 'chatterbox_profiles.json'),
        ('local_kokoro_default_onnx', 'KOKORO_VOICE_BLENDS_DIR', 'voice_blends.json'),
        ('local_higgs_default', 'HIGGS_VOICE_SAMPLES_DIR', 'voice_profiles.json'),
        ('local_higgs_v2', 'HIGGS_VOICE_SAMPLES_DIR', 'voice_profiles.json'),
        ('local_kokoro_default_pytorch', 'KOKORO_VOICE_BLENDS_DIR', 'voice_blends.json'),
    )
    if stage == 'backend':
        for backend_id, key, _ in routes:
            config[backend_id] = {key: str(root / backend_id)}
    manager.app_config = config
    selected = []
    for backend_id, key, filename in routes:
        route = resolve_legacy_route(backend_id)
        manager.app_config = legacy_provider_config(route.provider_id, config)['app_config']
        prepared = manager._prepare_backend_config(backend_id)
        directory = Path(prepared[key])
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / filename
        file.write_text('{}')
        selected.append(file)
    items = voices.discover(config)
    included = {item.path for item in items if item.status == 'included'}
    assert set(selected) <= included, (stage, 'actual final backend sources missing', set(selected) - included)
    assert all(file.read_text() == '{}' for file in selected)
assert 'tldw_chatbook.TTS.backends.chatterbox' not in sys.modules
assert 'tldw_chatbook.TTS.backends.higgs' not in sys.modules
assert 'tldw_chatbook.TTS.backends.kokoro' not in sys.modules
""",
        stage,
    )


def test_installed_voice_projection_preserves_raw_contract_and_finite_routes(tmp_path):
    _run_private_child(
        tmp_path,
        """
import copy, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.models import DiscoveryContext, DISCOVERY_CONTEXT_KEY
from tldw_chatbook.TTS.TTS_Backends import TTSBackendManager
from tldw_chatbook.TTS.legacy_bridge import legacy_provider_config, resolve_legacy_route
from tldw_chatbook.TTS.recovery import recovery_adapters
root = Path(sys.argv[1])
app = root / 'app-chatterbox'
ignored = root / 'ignored'
for directory in (app, ignored):
    directory.mkdir()
    (directory / 'chatterbox_profiles.json').write_text('{}')
raw = {
    'app_tts': {'CHATTERBOX_VOICE_DIR': str(app)},
    'global_tts_settings': {'CHATTERBOX_VOICE_DIR': str(ignored)},
    'local_chatterbox_unknown': {'CHATTERBOX_VOICE_DIR': str(ignored)},
    'local_higgs_unknown': {'HIGGS_VOICE_SAMPLES_DIR': str(ignored)},
    'local_kokoro_unknown': {'KOKORO_VOICE_BLENDS_DIR': str(ignored)},
    'COMPREHENSIVE_CONFIG_RAW': {'app_tts': {'CHATTERBOX_VOICE_DIR': str(ignored)}},
}
context = DiscoveryContext(root / 'config.toml', 'fixture')
voices = recovery_adapters()[0]
items = voices.discover({**raw, DISCOVERY_CONTEXT_KEY: context})
assert app / 'chatterbox_profiles.json' in {item.path for item in items}
assert all(item.path is None or ignored not in (item.path, *item.path.parents) for item in items)
# The normalized wrapper is interpreted by the actual bridge, never inventory.
canonical = {key: value for key, value in raw.items() if key != 'COMPREHENSIVE_CONFIG_RAW'}
manager = object.__new__(TTSBackendManager)
manager._check_cuda_available = lambda: False
route = resolve_legacy_route('local_chatterbox_default')
manager.app_config = legacy_provider_config(route.provider_id, {'COMPREHENSIVE_CONFIG_RAW': canonical,
    'APP_TTS_CONFIG': {'CHATTERBOX_VOICE_DIR': str(ignored)}})['app_config']
assert Path(manager._prepare_backend_config(route.internal_model_id)['CHATTERBOX_VOICE_DIR']) == app
manager.app_config = legacy_provider_config(route.provider_id, {'APP_TTS_CONFIG': canonical['app_tts']})['app_config']
assert Path(manager._prepare_backend_config(route.internal_model_id)['CHATTERBOX_VOICE_DIR']) == app
for route, key in [('local_chatterbox_default', 'CHATTERBOX_VOICE_DIR'),
                   ('local_kokoro_default_onnx', 'KOKORO_VOICE_BLENDS_DIR'),
                   ('local_kokoro_default_pytorch', 'KOKORO_VOICE_BLENDS_DIR'),
                   ('local_higgs_default', 'HIGGS_VOICE_SAMPLES_DIR'),
                   ('local_higgs_v2', 'HIGGS_VOICE_SAMPLES_DIR')]:
    for bad in (False, 1, [], {}, ''):
        config = {**canonical, DISCOVERY_CONTEXT_KEY: context, route: {key: bad}}
        try: voices.discover(config)
        except ValueError as error: assert str(error) == 'invalid_voice_path'
        else: raise AssertionError((route, bad))
    try: voices.discover({**canonical, DISCOVERY_CONTEXT_KEY: context, route: []})
    except ValueError as error: assert str(error) == 'invalid_config_shape'
    else: raise AssertionError('invalid installed-route mapping accepted')
assert (ignored / 'chatterbox_profiles.json').read_text() == '{}'
assert 'tldw_chatbook.TTS.backends.chatterbox' not in sys.modules
assert 'tldw_chatbook.TTS.backends.higgs' not in sys.modules
assert 'tldw_chatbook.TTS.backends.kokoro' not in sys.modules
""",
    )


def test_overlapping_voice_scans_preserve_changed_metadata_evidence(tmp_path):
    _run_private_child(
        tmp_path,
        """
import os, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.models import DiscoveryContext, DISCOVERY_CONTEXT_KEY
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration
from tldw_chatbook.TTS.recovery import recovery_adapters
root = Path(sys.argv[1])
outer = root / 'voices'
inner = outer / 'nested'
inner.mkdir(parents=True)
file = inner / 'voice_profiles.json'
file.write_text('{}')
first_stamp = file.stat().st_mtime_ns
original = _RawDeclaration._tree
calls = []
def changed(adapter, config, selected, **kwargs):
    items = original(adapter, config, selected, **kwargs)
    if selected == outer:
        file.write_text('{"changed": {}}')
        os.utime(file, ns=(first_stamp + 1000000000, first_stamp + 1000000000))
        calls.append(selected)
    return items
_RawDeclaration._tree = changed
items = recovery_adapters()[0].discover({DISCOVERY_CONTEXT_KEY: DiscoveryContext(root / 'config.toml', 'fixture'),
    'app_tts': {'CHATTERBOX_VOICE_DIR': str(outer)}, 'HiggsSettings': {'voice_samples_dir': str(inner)}})
observed = [item for item in items if item.path == file]
assert calls == [outer]
assert len(observed) == 2, 'changed overlapping source observation was silently collapsed'
assert len({item.metadata.mtime_ns for item in observed}) == 2
from dataclasses import replace
from tldw_chatbook.Backup_Recovery.inventory import classify_entries
from tldw_chatbook.Backup_Recovery.models import StorageItem, storage_logical_id
context = DiscoveryContext(root / 'config.toml', 'fixture')
context.config_path.write_text('')
classified = tuple(replace(item, shared_group='fixture-shared-observations') if item.path == file else item
                   for item in items if not item.logical_id.endswith(':participant_pending'))
classified += (StorageItem('config', storage_logical_id(context, 'config'), context.config_path, 'included', ()),)
result = classify_entries(classified)
assert not result.complete and 'unsupported' in result.issues, ('metadata conflict had no independent diagnostic', result.complete, result.issues)
assert file.read_text() == '{"changed": {}}'
""",
    )


def test_voice_source_alias_spellings_remain_visible(tmp_path):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.models import DiscoveryContext, DISCOVERY_CONTEXT_KEY
from tldw_chatbook.TTS.recovery import recovery_adapters
root = Path(sys.argv[1])
real = root / 'real'
real.mkdir()
(real / 'voice_profiles.json').write_text('{}')
alias = root / 'alias'
alias.symlink_to(real, target_is_directory=True)
items = recovery_adapters()[0].discover({DISCOVERY_CONTEXT_KEY: DiscoveryContext(root / 'config.toml', 'fixture'),
    'local_chatterbox_default': {'CHATTERBOX_VOICE_DIR': str(alias)},
    'HiggsSettings': {'voice_samples_dir': str(real)}})
assert any(item.path == alias and item.status == 'unsupported' for item in items)
assert any(item.path == real and item.status == 'included_directory' for item in items)
assert any(item.path == real / 'voice_profiles.json' and item.status == 'included' for item in items)
ids = {item.logical_id for item in items}
for item in items:
    if item.metadata is not None:
        assert item.metadata.root_id in ids
        if item.metadata.parent_id is not None: assert item.metadata.parent_id in ids
assert alias.is_symlink() and (real / 'voice_profiles.json').read_text() == '{}'
""",
    )
