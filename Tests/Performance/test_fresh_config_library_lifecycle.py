"""Creation data, original App stamp work and real same-profile process reopen."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio, collections, copy, inspect, json, os, sys, threading, tomllib
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()
route, stage = sys.argv[1:]
assert route in {'fresh', 'old_missing', 'old_corrupt', 'old_starter', 'old_expanded', 'old_graduated'}
assert stage in {'first', 'reopen'}
selector = Path(os.environ['TLDW_CONFIG_PATH']).absolute()
assert 'tldw_chatbook.config' not in sys.modules and 'tldw_chatbook.app' not in sys.modules
if stage == 'first':
    assert not selector.exists()
    if route != 'fresh':
        value = {'old_corrupt': 'not-a-lifecycle', 'old_starter': 'starter',
                 'old_expanded': 'expanded', 'old_graduated': 'graduated'}.get(route)
        text = '[general]\nusers_name="lifecycle-old"\n[first_run]\nsetup_completed=true\n[splash_screen]\nenabled=false\n'
        if value is not None:
            text += '[library.rail_state]\nlifecycle="' + value + '"\n'
        text += '[library.rail_state.sections]\nbrowse_open=false\ndetails_open=true\n'
        selector.write_text(text, encoding='utf-8')
        selector.chmod(0o600)
else:
    assert route == 'fresh' and selector.is_file()


def rail(mapping):
    library = mapping.get('library')
    selected = library.get('rail_state') if type(library) is dict else None
    return selected if type(selected) is dict else {}


def preserves_defaults(actual, expected):
    if type(expected) is dict:
        return type(actual) is dict and all(key in actual and preserves_defaults(actual[key], value)
                                             for key, value in expected.items())
    return actual == expected


async def run():
    # Import creates the actual fresh file through its original bootstrap.
    from tldw_chatbook import config
    from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
    physical_creation = tomllib.loads(selector.read_text(encoding='utf-8'))
    bootstrap_creation = copy.deepcopy(config.load_cli_config_and_ensure_existence())
    settings_creation = copy.deepcopy(config.load_settings())
    admission_at_creation = config.first_profile_created_this_session()
    default_lifecycle = rail(config.DEFAULT_CONFIG_FROM_TOML).get('lifecycle')
    creation_defaults_preserved = preserves_defaults(physical_creation, config.DEFAULT_CONFIG_FROM_TOML)
    from tldw_chatbook import app as app_module
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.Library.library_rail_state import LibraryLifecycle, coerce_library_lifecycle
    from tldw_chatbook.Chat import console_runtime as runtime_module
    from tldw_chatbook.Library import library_rail_state as rail_module
    from tldw_chatbook.UI.Screens import library_screen as library_module
    repository = Path.cwd().resolve()
    assert Path(real_profile_guard.__file__).resolve() == repository / 'Tests/real_profile_guard.py'
    producers = ((config, 'config.py'), (app_module, 'app.py'),
                 (runtime_module, 'Chat/console_runtime.py'),
                 (rail_module, 'Library/library_rail_state.py'),
                 (library_module, 'UI/Screens/library_screen.py'))
    origins = {}
    for module, relative in producers:
        expected = repository / 'tldw_chatbook' / relative
        assert Path(module.__file__).resolve() == expected
        assert module.__spec__ is not None and Path(module.__spec__.origin).resolve() == expected
        origins[module.__name__] = str(expected)
    counts = collections.Counter()

    class StampCounter(OriginalStorageUnitObserver):
        def __init__(self):
            super().__init__(counts, lambda: True, lambda unit: counts.update((unit,)))
            self.owner = None
            self.ctor_code = self.stamp_code = self.writer_code = None

        def _ancestry(self, frame):
            found_stamp = found_ctor = False
            for _ in range(12):
                if frame is None:
                    break
                if frame.f_code is self.stamp_code:
                    found_stamp = frame.f_globals is vars(app_module) and frame.f_locals.get('self') is self.owner
                if frame.f_code is self.ctor_code:
                    found_ctor = frame.f_globals is vars(app_module) and frame.f_locals.get('self') is self.owner
                frame = frame.f_back
            return found_stamp and found_ctor

        def _selected(self, code):
            if code not in self.codes:
                return False
            if code is self.writer_code:
                return self._ancestry(self._frame(code))
            return True

        def _start(self, code, offset):
            try:
                if not self.active or not self._selected(code):
                    return
                frame = self._frame(code)
                if code is self.ctor_code:
                    assert self.owner is None and type(frame.f_locals.get('self')) is TldwCli
                    self.owner = frame.f_locals['self']
                elif code is self.stamp_code:
                    assert frame.f_locals.get('self') is self.owner
                super()._start(code, offset)
            except BaseException as error:
                self.invalid.append('stamp_start:' + type(error).__name__)

        def _return(self, code, offset, value):
            try:
                if not self.active or not self._selected(code):
                    return
                super()._return(code, offset, value)
            except BaseException as error:
                self.invalid.append('stamp_return:' + type(error).__name__)

        def start(self):
            selected = (
                (TldwCli, '__init__', 'constructor'),
                (TldwCli, '_stamp_new_profile_library_lifecycle', 'stamp'),
                (config, 'save_setting_to_cli_config', 'lifecycle_writer'),
                (LibraryScreen, '__init__', 'library_construction'),
                (LibraryScreen, 'on_mount', 'library_mount'),
            )
            for owner, name, unit in selected:
                function = inspect.getattr_static(owner, name)
                code = self._pin(function)
                self.slots.append((owner, name, function))
                self.codes[code] = unit
                if unit == 'constructor': self.ctor_code = code
                elif unit == 'stamp': self.stamp_code = code
                elif unit == 'lifecycle_writer': self.writer_code = code
            assert app_module.save_setting_to_cli_config is config.save_setting_to_cli_config
            self.slots.append((app_module, 'save_setting_to_cli_config', config.save_setting_to_cli_config))
            self._pin(coerce_library_lifecycle)
            self.slots.append((sys.modules[coerce_library_lifecycle.__module__], 'coerce_library_lifecycle', coerce_library_lifecycle))
            # Pin the actual guarded creation wrapper and its exact body.
            function = config._load_cli_config_bootstrap_unlocked
            body = function.__wrapped__
            self._pin(function)
            self._pin(body)
            cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
            assert cells['function'].cell_contents is body
            self.slots.extend(((config, '_load_cli_config_bootstrap_unlocked', function), (function, '__wrapped__', body)))
            for tool in range(5, 0, -1):
                if tool == self.monitor.DEBUGGER_ID: continue
                try: self.monitor.use_tool_id(tool, 'original-creation-library-stamp-counter')
                except ValueError: continue
                self.tool = tool
                break
            assert self.tool is not None
            for event, callback in ((self.monitor.events.PY_START, self._start), (self.monitor.events.PY_RETURN, self._return)):
                previous = self.monitor.register_callback(self.tool, event, callback)
                self.registered[event] = callback
                if previous is not None:
                    self.monitor.register_callback(self.tool, event, previous)
                    self.registered.pop(event)
                    raise RuntimeError('stamp_counter_callback_borrowed')
            assert self.monitor.get_events(self.tool) == 0
            assert all(self.monitor.get_local_events(self.tool, code) == 0 for code in self.codes)
            self.active = self.installed = True
            for code in self.codes:
                self.monitor.set_local_events(self.tool, code, self.monitor.events.PY_START | self.monitor.events.PY_RETURN)
            assert self.monitor.get_events(self.tool) == 0

    observer = StampCounter()
    app = runtime = None
    snapshot_lifecycle = None
    try:
        observer.start()
        app = TldwCli()
        runtime = app.console_runtime
        assert observer.owner is app
        runtime_class = sys.modules['tldw_chatbook.Chat.console_runtime'].ConsoleRuntime
        assert type(runtime) is runtime_class and runtime._app is app
        original_dispose = inspect.getattr_static(runtime_class, 'dispose')
        observer._pin(original_dispose)
        observer.slots.append((runtime_class, 'dispose', original_dispose))
        snapshot_lifecycle = rail(app.app_config).get('lifecycle')
        async with app.run_test(size=(140, 42)):
            # Existing cold-native prerequisite; the outer240s bound is retained.
            while not getattr(app, '_initial_screen_pushed', False):
                await asyncio.sleep(.01)
            assert not isinstance(app.screen, LibraryScreen)
    finally:
        receipt = observer.close()
        record = dict(route=route, stage=stage, observer=receipt, counts=dict(counts),
                      loaded_origins=origins,
                      actual_creation_admission=admission_at_creation,
                      physical_creation_unknown=rail(physical_creation).get('lifecycle') == 'unknown',
                      bootstrap_creation_unknown=rail(bootstrap_creation).get('lifecycle') == 'unknown',
                      settings_creation_unknown=rail(settings_creation).get('lifecycle') == 'unknown',
                      app_snapshot_unknown=snapshot_lifecycle == 'unknown',
                      default_merge_has_no_lifecycle=default_lifecycle is None,
                      creation_template_defaults_preserved=creation_defaults_preserved,
                      normal_runtime_disposal=bool(runtime is not None and runtime._disposed),
                      network_attempts=len(network_guard.blocked_attempts()),
                      profile_refusals=len(real_profile_guard.take_violations()))
        (selector.parent.parent / ('library-creation-' + stage + '.json')).write_text(json.dumps(record), encoding='utf-8')
    # Qualification comes first; no counts from an invalid observer are accepted.
    assert receipt['complete'] and receipt['original_source_current'] and not receipt['invalid'], record
    assert receipt['global_events'] == 0 and receipt['hooks_retired_before_inactive'], record
    assert not observer.active and sys.monitoring.get_tool(observer.tool) is None
    assert all(sys.monitoring.get_local_events(observer.tool, code) == 0 for code in observer.codes)
    assert record['normal_runtime_disposal'] and record['network_attempts'] == record['profile_refusals'] == 0, record
    assert counts['constructor'] == 1
    assert counts['library_construction'] == counts['library_mount'] == 0
    assert default_lifecycle is None, 'Creation UNKNOWN leaked into the old-profile default merge'
    durable = tomllib.loads(selector.read_text(encoding='utf-8'))
    if route == 'fresh':
        assert admission_at_creation is (stage == 'first')
        assert counts['stamp'] == int(stage == 'first')
        assert record['physical_creation_unknown'] and record['bootstrap_creation_unknown'] and record['settings_creation_unknown'], record
        assert creation_defaults_preserved, 'Creation serialization changed original template defaults'
        assert snapshot_lifecycle == 'unknown' and rail(durable).get('lifecycle') == 'unknown'
        assert coerce_library_lifecycle(snapshot_lifecycle, is_new_profile=admission_at_creation) is LibraryLifecycle.UNKNOWN
        assert counts['lifecycle_writer'] == 0, 'Original creation App stamp performed a second lifecycle writer'
    else:
        expected = {'old_missing': None, 'old_corrupt': 'not-a-lifecycle', 'old_starter': 'starter',
                    'old_expanded': 'expanded', 'old_graduated': 'graduated'}[route]
        assert not admission_at_creation and counts['stamp'] == counts['lifecycle_writer'] == 0
        assert rail(physical_creation).get('lifecycle') == rail(bootstrap_creation).get('lifecycle') == rail(durable).get('lifecycle') == expected
        assert rail(durable).get('sections') == {'browse_open': False, 'details_open': True}
        expected_state = LibraryLifecycle.EXPANDED if expected in {None, 'not-a-lifecycle'} else LibraryLifecycle(expected)
        assert coerce_library_lifecycle(rail(durable).get('lifecycle'), is_new_profile=False) is expected_state
    print('retired and reopened')


with user_fixture_default_owner():
    asyncio.run(asyncio.wait_for(run(), timeout=240))
"""


def _reopen_same_profile(root, route, *, script=_SCRIPT):
    """Second finite child uses the identical original fixture profile paths."""
    root = root.resolve()
    assert (root / "config" / "config.toml").is_file()
    assert all((root / name).is_dir() for name in ("home", "config", "data"))
    environment = os.environ.copy()
    environment.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        TLDW_CONFIG_PATH=str(root / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, route, "reopen"],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-1000:]
    assert "retired and reopened" in result.stdout


@pytest.mark.parametrize(
    "route",
    (
        "fresh",
        "old_missing",
        "old_corrupt",
        "old_starter",
        "old_expanded",
        "old_graduated",
    ),
)
def test_fresh_creation_owns_library_unknown_and_original_app_stamp_does_no_second_write(
    tmp_path, route
):
    # Existing first-child launcher preserves ordinary owner, guards and240 bound.
    _run(tmp_path, route, "first", script=_SCRIPT, timeout=240)
    first = json.loads(
        (tmp_path / "library-creation-first.json").read_text(encoding="utf-8")
    )
    assert (
        first["observer"]["complete"]
        and first["counts"].get("lifecycle_writer", 0) == 0
    )
    if route == "fresh":
        _reopen_same_profile(tmp_path, route)
        second = json.loads(
            (tmp_path / "library-creation-reopen.json").read_text(encoding="utf-8")
        )
        assert (
            second["normal_runtime_disposal"]
            and not second["actual_creation_admission"]
        )
        assert second["physical_creation_unknown"] and second["app_snapshot_unknown"]
