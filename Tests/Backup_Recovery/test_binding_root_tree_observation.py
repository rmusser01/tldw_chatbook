"""Actual original binding-root parent work, in a private child without an App."""

from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = r"""
import ast
import contextlib
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from types import CodeType

from Tests.network_guard import install as network_install, blocked_attempts
from Tests import real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

network_install()
real_profile_guard.install()
for name in ('sounddevice', 'pyaudio'):
    sys.modules[name] = None
route, outcome = sys.argv[1:]
assert route == 'binding_tree' and outcome == 'work_count'

def shape(code):
    return (
        code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
        code.co_flags, code.co_stacksize, code.co_code, code.co_exceptiontable,
        code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
        tuple(shape(value) if type(value) is CodeType else value
              for value in code.co_consts),
    )

def locate(code, qualified_name):
    if code.co_qualname == qualified_name:
        return code
    for value in code.co_consts:
        if type(value) is CodeType:
            result = locate(value, qualified_name)
            if result is not None:
                return result
    return None

def wrapper_current(current, metadata):
    original, body, code, namespace, closure = metadata
    return (
        current is original and current.__code__ is code
        and current.__globals__ is namespace
        and current.__defaults__ is None and current.__kwdefaults__ is None
        and current.__wrapped__ is body and current.__closure__ is closure
        and code.co_freevars == ('func',)
        and type(closure) is tuple and len(closure) == 1
        and closure[0].cell_contents is body
    )

def main():
    from tldw_chatbook.Backup_Recovery import bootstrap, control_records
    from tldw_chatbook.Utils import private_paths, windows_files

    selector = Path(os.environ['TLDW_CONFIG_PATH'])
    selector.write_text('[paths]\n', encoding='utf-8')
    sandbox = Path(os.environ['XDG_DATA_HOME']) / 'binding-root-siblings'
    sandbox.mkdir(mode=0o700)
    siblings = tuple(sandbox / ('root-%02d' % number) for number in range(12))
    for path in siblings:
        path.mkdir(mode=0o700)
    root = bootstrap.default_bootstrap_root()
    authority = control_records.admission_authority(root)
    authority.register('binding.tree', (selector, *siblings))
    control_records.bind_profile(root, selector, ('binding.tree',), root / 'admission')
    pending, profiles = bootstrap._records(root)
    registry = bootstrap._registry(root)
    assert not pending
    record = next(row for row in profiles if row['selector'] == str(selector))
    assert set(record['roots']) == {str(path) for path in (selector, *siblings)}

    functions = (
        (bootstrap, '_binding', bootstrap._binding),
        (bootstrap, '_fingerprint', bootstrap._fingerprint),
        (bootstrap, 'effective_roots', bootstrap.effective_roots),
        (bootstrap, 'pinned_directory', bootstrap.pinned_directory.__wrapped__),
        (private_paths, '_open_verified_parent', private_paths._open_verified_parent),
        (private_paths, '_open_directory_component', private_paths._open_directory_component),
        (private_paths, '_trusted_directory_owner', private_paths._trusted_directory_owner),
        (private_paths, '_native_open', private_paths._native_open),
        (private_paths, '_native_close', private_paths._native_close),
        (contextlib, 'contextmanager', contextlib.contextmanager),
        (windows_files, '_Native.open_handle', windows_files._Native.open_handle),
        (windows_files, 'WindowsOS.stat_many_for_admission', windows_files.WindowsOS.stat_many_for_admission),
    )
    modules = tuple(dict.fromkeys(module for module, _, _ in functions))
    origins = {}
    for module in modules:
        origin = Path(module.__spec__.origin).resolve()
        assert Path(module.__file__).resolve() == origin
        assert module.__loader__ is module.__spec__.loader
        raw = origin.read_bytes()
        origins[module] = (
            module.__spec__, module.__loader__, str(origin),
            hashlib.sha256(raw).hexdigest(),
            compile(raw, str(origin), 'exec', dont_inherit=True),
        )
    pinned = []
    for module, qualified_name, function in functions:
        assert function.__globals__ is vars(module)
        assert function.__closure__ is None
        declared = locate(origins[module][4], qualified_name)
        assert declared is not None and shape(function.__code__) == shape(declared)
        declared_tree = ast.parse(Path(origins[module][2]).read_bytes())
        for part in qualified_name.split('.'):
            declared_tree = next(node for node in declared_tree.body
                                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                                 and node.name == part)
        expected_defaults = tuple(ast.literal_eval(node) for node in declared_tree.args.defaults)
        actual_defaults = function.__defaults__ or ()
        assert len(actual_defaults) == len(expected_defaults)
        assert all(type(actual) is type(expected) and actual == expected
                   for actual, expected in zip(actual_defaults, expected_defaults, strict=True))
        expected_keywords = {
            argument.arg: ast.literal_eval(default)
            for argument, default in zip(declared_tree.args.kwonlyargs,
                                         declared_tree.args.kw_defaults, strict=True)
            if default is not None
        }
        actual_keywords = function.__kwdefaults__ or {}
        assert set(actual_keywords) == set(expected_keywords)
        assert all(type(actual_keywords[key]) is type(value) and actual_keywords[key] == value
                   for key, value in expected_keywords.items())
        pinned.append((module, qualified_name, function, function.__code__,
                       function.__globals__, function.__defaults__,
                       function.__kwdefaults__,
                       tuple((function.__kwdefaults__ or {}).items())))

    def source_current():
        for module, (spec, loader, origin, digest, _) in origins.items():
            assert module.__spec__ is spec and module.__loader__ is loader
            assert spec.loader is loader and spec.origin == origin
            assert Path(module.__file__).resolve() == Path(origin)
            assert hashlib.sha256(Path(origin).read_bytes()).hexdigest() == digest
        for module, name, function, code, namespace, defaults, kwdefaults, items in pinned:
            installed = module
            for part in name.split('.'):
                installed = getattr(installed, part)
            if name == 'pinned_directory':
                installed = installed.__wrapped__
            assert installed is function and function.__code__ is code
            assert function.__globals__ is namespace
            assert function.__defaults__ is defaults and function.__closure__ is None
            assert function.__kwdefaults__ is kwdefaults
            current_items = function.__kwdefaults__ or {}
            assert tuple(current_items) == tuple(key for key, _ in items)
            assert all(current_items[key] is value for key, value in items)
        assert wrapper_current(bootstrap.pinned_directory, wrapper_metadata)
        assert bootstrap._open_verified_parent is private_paths._open_verified_parent
        assert bootstrap._native_close is private_paths._native_close
        assert bootstrap.os is private_paths.os is original_facade
        if os.name == 'nt':
            assert original_facade.stat_many_for_admission is original_many

    original_pinned_alias = bootstrap.pinned_directory
    body = original_pinned_alias.__wrapped__
    wrapper_code = locate(origins[contextlib][4], 'contextmanager.<locals>.helper')
    assert wrapper_code is not None and shape(original_pinned_alias.__code__) == shape(wrapper_code)
    assert original_pinned_alias.__globals__ is vars(contextlib)
    assert original_pinned_alias.__defaults__ is None and original_pinned_alias.__kwdefaults__ is None
    assert original_pinned_alias.__code__.co_freevars == ('func',)
    assert type(original_pinned_alias.__closure__) is tuple and len(original_pinned_alias.__closure__) == 1
    assert original_pinned_alias.__closure__[0].cell_contents is body
    wrapper_metadata = (original_pinned_alias, body, original_pinned_alias.__code__,
                        original_pinned_alias.__globals__, original_pinned_alias.__closure__)
    assert bootstrap._open_verified_parent is private_paths._open_verified_parent
    assert bootstrap._native_close is private_paths._native_close
    original_facade = bootstrap.os
    assert original_facade is private_paths.os
    original_many = getattr(original_facade, 'stat_many_for_admission', None)
    if os.name == 'nt':
        assert type(original_facade) is windows_files.WindowsOS
        assert original_many.__self__ is original_facade
        assert original_many.__func__ is windows_files.WindowsOS.stat_many_for_admission
    binding_code = bootstrap._binding.__code__
    native_code = windows_files._Native.open_handle.__code__
    thread = threading.current_thread()
    count = [0]
    calls = []
    invalid = []
    tool_id = next(number for number in range(5, -1, -1)
                   if sys.monitoring.get_tool(number) is None)
    sys.monitoring.use_tool_id(tool_id, 'binding-root-original-native-count')

    def started(code, offset):
        if code is not native_code or threading.current_thread() is not thread:
            return
        frame = sys._getframe(1)
        for _ in range(32):
            if frame is None:
                break
            if frame.f_code is binding_code:
                count[0] += 1
                return
            frame = frame.f_back
        invalid.append('native_open_outside_selected_binding')

    try:
        source_current()
        assert sys.monitoring.get_events(tool_id) == 0
        sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_START, started)
        sys.monitoring.set_local_events(tool_id, native_code, sys.monitoring.events.PY_START)
        for _ in range(2):
            count[0] = 0
            assert bootstrap._binding(selector, profiles, registry) is record
            calls.append(count[0])
        source_current()
    finally:
        try:
            sys.monitoring.set_local_events(tool_id, native_code, 0)
            sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_START, None)
            final_global = sys.monitoring.get_events(tool_id)
            final_local = sys.monitoring.get_local_events(tool_id, native_code)
        finally:
            sys.monitoring.free_tool_id(tool_id)
    assert final_global == 0 and final_local == 0

    # All original binding/native bodies returned before this scalar assertion.
    selected = tuple(Path(path).resolve(strict=True) for path in record['roots'])
    union = {node for path in selected for node in (*path.parents, path)}
    bound = 2 * len(union) + len(selector.parts) + 5
    receipt = {
        'source_current': True, 'original_binding_calls_returned': 2,
        'original_native_open_counts': calls, 'resolved_union_nodes': len(union),
        'structural_union_open_bound': bound, 'global_events': 0,
        'local_events_retired': True, 'tool_retired': sys.monitoring.get_tool(tool_id) is None,
        'invalid': invalid, 'network_attempts': len(blocked_attempts()),
        'classification': 'Leaf original-body work count; no App/performance attribution',
    }
    (selector.parent.parent / 'binding-root-tree.json').write_text(
        json.dumps(receipt, indent=2), encoding='utf-8')
    assert not invalid and not blocked_attempts()
    if os.name == 'nt':
        assert calls[0] == calls[1] and calls[0] > 0
        assert max(calls) <= bound, receipt
    else:
        assert calls == [0, 0]  # The unchanged POSIX branch never opens through this facade.
    print('retired and reopened')

with user_fixture_default_owner():
    main()
"""


def test_original_enrolled_binding_observes_shared_windows_parent_tree_once(tmp_path):
    _run(tmp_path, "binding_tree", "work_count", script=_SCRIPT)
