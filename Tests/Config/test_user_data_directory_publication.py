"""A real settings publication keeps its derived directory memo current."""

import ast

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = r"""
import copy
import inspect
import os
import sys
import threading
import tomllib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import toml
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()

route, outcome = sys.argv[1:]
assert outcome == "publication"


@contextmanager
def original_calls(config):
    targets = {
        "publish": inspect.unwrap(config._publish_runtime_config_unlocked),
        "rebuild": inspect.unwrap(config._load_settings_uncached),
        "directory": inspect.unwrap(config.get_user_data_dir),
        "resolve": config._resolve_user_data_dir,
    }
    bindings = tuple((name, function, function.__code__) for name, function in targets.items())
    codes = {function.__code__: name for name, function in targets.items()}
    observed = SimpleNamespace(publishing=0, rebuilding=0, phase="publish",
                               resolutions={"publish": 0, "after": 0, "reload": 0},
                               rebuild_memos=[], errors=[])
    thread = threading.get_ident()
    monitor = sys.monitoring
    tool = next(number for number in range(6)
                if monitor.get_tool(number) is None and monitor.get_events(number) == 0
                and all(monitor.get_local_events(number, code) == 0 for code in codes))
    monitor.use_tool_id(tool, "directory-publication-original-calls")

    def started(code, _offset):
        if threading.get_ident() != thread:
            return
        name = codes[code]
        if name == "publish":
            observed.publishing += 1
        elif name == "rebuild":
            observed.rebuilding += 1
        elif name == "resolve":
            observed.resolutions[observed.phase] += 1

    def returned(code, _offset, _value):
        if threading.get_ident() != thread:
            return
        try:
            name = codes[code]
            if name == "directory" and observed.publishing and observed.rebuilding:
                memo = config._USER_DATA_DIR_MEMO
                assert type(memo) is tuple and len(memo) == 3
                observed.rebuild_memos.append((memo, config._CONFIG_CACHE, _value))
            elif name == "publish":
                observed.publishing -= 1
            elif name == "rebuild":
                observed.rebuilding -= 1
        except Exception as error:
            observed.errors.append(type(error).__name__)

    events = monitor.events
    mask = events.PY_START | events.PY_RETURN
    callbacks = ((events.PY_START, started), (events.PY_RETURN, returned))
    prior = [(event, callback, monitor.register_callback(tool, event, callback))
             for event, callback in callbacks]
    try:
        for code in codes:
            monitor.set_local_events(tool, code, mask)
        yield observed
    finally:
        for code in codes:
            monitor.set_local_events(tool, code, 0)
            assert monitor.get_local_events(tool, code) == 0
        for event, callback, previous in prior:
            assert monitor.register_callback(tool, event, previous) is callback
        assert monitor.get_events(tool) == 0
        monitor.free_tool_id(tool)
        assert monitor.get_tool(tool) is None
        assert all(function.__code__ is code for _name, function, code in bindings)
        assert inspect.unwrap(config._publish_runtime_config_unlocked) is targets["publish"]
        assert inspect.unwrap(config._load_settings_uncached) is targets["rebuild"]
        assert inspect.unwrap(config.get_user_data_dir) is targets["directory"]
        assert config._resolve_user_data_dir is targets["resolve"]


def main():
    selector = Path(os.environ["TLDW_CONFIG_PATH"]).absolute()
    baseline = {"general": {"users_name": "directory-publication"}}
    if route == "configured":
        baseline["paths"] = {"data_dir": Path(os.environ["XDG_DATA_HOME"]).as_posix()}
    else:
        assert route == "default"
    selector.write_text(toml.dumps(baseline), encoding="utf-8")
    selector.chmod(0o600)

    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    expected_directory = config.get_user_data_dir()
    for _ in range(3):
        assert config.get_user_data_dir() == expected_directory
    assert config._USER_DATA_DIR_MEMO[0][0] is config._CONFIG_CACHE
    assert config._USER_DATA_DIR_MEMO[0][1] == config._CONFIG_GENERATION
    before = tomllib.loads(selector.read_text(encoding="utf-8"))
    expected = copy.deepcopy(before)
    expected["general"]["default_tab"] = "settings"
    generation = config._CONFIG_GENERATION

    with original_calls(config) as observed:
        result = config.apply_settings_mutation_to_cli_config(
            {"general": {"default_tab": "settings"}}
        )
        assert result.file_replaced and result.caches_reloaded and result.failure_phase is None
        assert config._CONFIG_GENERATION == generation + 1
        assert observed.rebuild_memos, "the original settings rebuild did not derive a directory memo"
        memo, installed_cache, returned_directory = observed.rebuild_memos[-1]
        assert memo[0][0] is installed_cache is config._CONFIG_CACHE
        assert memo is config._USER_DATA_DIR_MEMO
        assert returned_directory == expected_directory
        assert observed.resolutions["publish"] >= 1, "new installed cache did not invalidate the old memo"

        observed.phase = "after"
        assert config.get_user_data_dir() == expected_directory
        assert expected_directory.is_dir()
        assert tomllib.loads(selector.read_text(encoding="utf-8")) == expected
        snapshot = config.get_runtime_config_snapshot()
        assert snapshot.generation == generation + 1
        assert snapshot.values["general"]["default_tab"] == "settings"

        # An actual subsequent reload must still invalidate the cache-object key.
        observed.phase = "reload"
        config.load_settings(force_reload=True)
        assert config.get_user_data_dir() == expected_directory
        assert observed.resolutions["reload"] >= 1

    assert not observed.errors, observed.errors
    assert observed.publishing == observed.rebuilding == 0
    with storage._lock:
        startup = set(storage._startups.values())
        census = {"ordinary": len(storage._live_leases - startup),
                  "pending": len(storage._pending_acquisitions),
                  "operations": len(storage._operations), "raw": len(storage._raw_operations),
                  "raw_states": len(raw._states), "retiring": len(storage._retiring_holds)}
    assert all(count == 0 for count in census.values()), census
    assert observed.resolutions["after"] == 0, {
        "extra_resolution_after_publication": observed.resolutions["after"],
        "memo_generation_in_rebuild": memo[0][1],
        "published_generation": generation + 1,
    }
    assert memo[0][1] == generation + 1


with user_fixture_default_owner():
    main()
print("retired and reopened")
"""


@pytest.mark.parametrize("route", ["default", "configured"])
def test_published_directory_memo_survives_the_next_read(tmp_path, route):
    _run(tmp_path, route, "publication", script=_SCRIPT)


def test_directory_publication_child_is_syntactically_valid():
    ast.parse(_SCRIPT)
