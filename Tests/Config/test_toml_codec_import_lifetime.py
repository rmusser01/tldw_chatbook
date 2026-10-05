"""Fresh native config writes defer the special codec until a literal \\x."""

import ast

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = r"""
import datetime as dt
import hashlib, json, os, sys, tomllib
from pathlib import Path
import toml
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, outcome = sys.argv[1:]
assert outcome == 'original-writer'
special = 'tldw_chatbook.Utils.toml_serialization'
assert special not in sys.modules, 'fresh child has already imported the special codec'

def main():
    selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
    data = Path(os.environ['XDG_DATA_HOME']).absolute()
    baseline = {'general': {'users_name': 'codec-import'},
                'paths': {'data_dir': data.as_posix()}}
    selector.write_text(toml.dumps(baseline), encoding='utf-8')
    selector.chmod(0o600)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import config_participants as life
    from tldw_chatbook.Backup_Recovery import raw_participants as raw, storage_admission as storage
    assert special not in sys.modules, 'ordinary real config import loaded the special codec'
    anchors = ((config, 'replace_cli_config'), (config, '_write_raw_cli_config_unlocked'),
               (config, 'read_cli_config_serialized'), (config, '_load_settings_guarded'),
               (config, '_load_settings_uncached'), (life, 'operation'),
               (raw, '_check'), (storage, 'acquire_storage'), (toml, 'dumps'))
    bodies = [(owner, name, getattr(owner, name), getattr(owner, name).__code__)
              for owner, name in anchors]
    files = (Path(config.__file__).absolute(),
             Path(config.__file__).parent / 'Utils' / 'toml_serialization.py')
    repository = Path.cwd().resolve()
    assert all(path.is_relative_to(repository) for path in files)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    command = (r'C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe'
               if route == 'affected' else r'C:\Users\fixture\python.exe')
    value = '--literal=\\x41' if route == 'affected' else 'ordinary Unicode ☃ and quote "'
    document = {**baseline,
                'hooks': {'enabled': True, 'hook': [{'id': 'same', 'event': 'PostToolUse',
                          'timeout_s': 5, 'command': [command, value, '-c', 'pass']}]},
                'quoted ordinary.key': {'value': value},
                'date': dt.date(2026, 10, 5), 'enabled': True}
    ordinary_bytes = toml.dumps(document) if route == 'ordinary' else None
    loaded = config.replace_cli_config(document)
    actual = config.read_cli_config_serialized()
    parsed = tomllib.loads(actual)
    assert parsed == document, 'original writer/read-back changed literal values or keys'
    assert loaded['hooks'] == document['hooks']
    assert parsed['hooks']['hook'][0]['command'][0].encode('utf-8') == command.encode('utf-8')
    assert parsed['quoted ordinary.key']['value'].encode('utf-8') == value.encode('utf-8')
    assert selector.read_bytes() == actual.encode('utf-8')
    if route == 'ordinary':
        assert actual == ordinary_bytes, 'ordinary codec bytes differ from original toml.dumps'
        assert special not in sys.modules, 'ordinary original write/read-back imported special codec'
    else:
        assert special in sys.modules, 'literal backslash-x did not reach the real special codec'
        helper = sys.modules[special]
        assert Path(helper.__file__).absolute() == files[1]
        assert actual == helper.dumps_cli_config(document)
        assert actual.count('"quoted ordinary.key"') == 1
    assert all(getattr(owner, name) is function and function.__code__ is code
               for owner, name, function, code in bodies), 'original callback/body changed'
    assert hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    with storage._lock:
        startup = set(storage._startups.values())
        census = {'ordinary': len(storage._live_leases - startup),
                  'pending': len(storage._pending_acquisitions),
                  'operations': len(storage._operations), 'raw': len(storage._raw_operations),
                  'raw_states': len(raw._states), 'retiring': len(storage._retiring_holds)}
    assert all(count == 0 for count in census.values()), census
    receipt = {'route': route, 'original_write_and_readback': True,
               'special_codec_loaded': special in sys.modules,
               'original_callbacks_and_bodies_unchanged': True,
               'final_census': census, 'source_hashes': hashes}
    (selector.parent.parent / 'codec-import-receipt.json').write_text(
        json.dumps(receipt, sort_keys=True), encoding='utf-8')
    print('retired and reopened original config writer/read-back')

with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize("route", ["ordinary", "affected"])
def test_original_config_writer_defers_special_toml_codec(tmp_path, route):
    _run(tmp_path, route, "original-writer", script=_SCRIPT)


def test_codec_import_child_is_syntactically_valid():
    ast.parse(_SCRIPT)
