"""Private actual binding refusal/custody controls, without an App."""

import pytest

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
assert route == 'binding_tree' and outcome in ('parent_permission', 'source_code', 'tree_close', 'production_drift', 'custom_reader')

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
        (windows_files, '_Native.info', windows_files._Native.info),
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

    source_current()
    assert bootstrap._binding(selector, profiles, registry) is record
    details = {}
    if outcome == 'custom_reader':
        if os.name != 'nt':
            assert bootstrap._binding(selector, profiles, registry) is record
            details = {'posix_original_binding_positive': True,
                       'windows_custom_reader_qualified': False}
        else:
            unknown_calls = []
            def unknown_batch(paths):
                unknown_calls.append(True)
                raise AssertionError('unknown_custom_batch_must_never_be_called')
            namespace = object.__getattribute__(original_facade, '__dict__')
            assert namespace['stat_many_for_admission'] is original_many
            try:
                namespace['stat_many_for_admission'] = unknown_batch
                # The original binding/parent bodies still execute. The new
                # optional unknown batch ABI is deliberately unavailable.
                assert bootstrap._binding(selector, profiles, registry) is record
                assert not unknown_calls
            finally:
                namespace['stat_many_for_admission'] = original_many
            source_current()
            details = {'windows_custom_reader_qualified': True,
                       'unknown_custom_batch_calls': len(unknown_calls),
                       'original_scalar_route_positive': True}
    elif outcome == 'parent_permission':
        if os.name == 'nt':
            from Tests.Utils import test_windows_native_admission as native_test
            helper = native_test._replace_security
            helper_origin = Path(native_test.__spec__.origin).resolve()
            helper_bytes = helper_origin.read_bytes()
            helper_code = locate(compile(helper_bytes, str(helper_origin), 'exec', dont_inherit=True), '_replace_security')
            assert helper.__globals__ is vars(native_test) and helper.__closure__ is None
            assert helper.__defaults__ is None and helper.__kwdefaults__ == {'replace_owner': False}
            assert shape(helper.__code__) == shape(helper_code)
            native = windows_files._native()
            sid = native._token_sid(1)
            private_acl = 'D:P(A;;FA;;;%s)' % sid
            shared_acl = private_acl + '(A;;FA;;;WD)'
            helper(sandbox, private_acl)
            original = original_facade.stat(sandbox, follow_symlinks=False)
            assert original.st_uid == original_facade.geteuid() and not original.st_mode & 0o022
            assert bootstrap._binding(selector, profiles, registry) is record
            restore = lambda: helper(sandbox, private_acl)
            damage = lambda: helper(sandbox, shared_acl)
        else:
            original = os.stat(sandbox, follow_symlinks=False)
            assert original.st_uid == os.geteuid() and not original.st_mode & 0o022
            restore = lambda: os.chmod(sandbox, original.st_mode & 0o7777)
            damage = lambda: os.chmod(sandbox, 0o777)
        try:
            damage()
            changed = original_facade.stat(sandbox, follow_symlinks=False)
            assert changed.st_uid == original.st_uid and changed.st_mode & 0o022
            try:
                bootstrap._binding(selector, profiles, registry)
            except private_paths.PrivatePathError as error:
                assert error.result.reason == 'shared_writable_parent'
                assert error.result.lexical_path == sandbox / '.bootstrap-reader'
                details = {'private_parent_positive': True, 'actual_shared_parent_refused': True,
                           'reason': error.result.reason, 'same_owner': True}
            else:
                raise AssertionError('actual_shared_immediate_parent_was_admitted')
        finally:
            restore()
        assert bootstrap._binding(selector, profiles, registry) is record
        if os.name == 'nt':
            assert native_test._replace_security is helper
            assert helper.__globals__ is vars(native_test) and helper.__closure__ is None
            assert shape(helper.__code__) == shape(helper_code)
            assert helper_origin.read_bytes() == helper_bytes
    elif outcome == 'source_code':
        # Qualification refusal only: the substituted body is never invoked.
        original_code = private_paths._open_verified_parent.__code__
        changed_code = original_code.replace(co_filename=original_code.co_filename + '.controlled-drift')
        try:
            private_paths._open_verified_parent.__code__ = changed_code
            try:
                source_current()
            except AssertionError:
                details = {'exact_stock_body_code_drift_refused': True,
                           'mutated_guard_body_invoked': False}
            else:
                raise AssertionError('exact_guard_body_code_drift_was_qualified')
        finally:
            private_paths._open_verified_parent.__code__ = original_code
        source_current()
        assert bootstrap._binding(selector, profiles, registry) is record
    else:
        if os.name != 'nt':
            assert bootstrap._binding(selector, profiles, registry) is record
            details = {'posix_original_binding_positive': True,
                       'windows_close_custody_qualified': False}
        else:
            import ctypes as C
            native = windows_files._native()
            kernel = C.WinDLL('kernel32', use_last_error=True)
            protect = kernel.SetHandleInformation
            protect.argtypes = [windows_files._HANDLE, windows_files._U32, windows_files._U32]
            protect.restype = windows_files._I32
            native_code = windows_files._Native.open_handle.__code__
            binding_code = bootstrap._binding.__code__
            many_code = windows_files.WindowsOS.stat_many_for_admission.__code__
            batch_function = windows_files.WindowsOS.stat_many_for_admission
            batch_original_code = batch_function.__code__
            named_code = locate(many_code, 'WindowsOS.stat_many_for_admission.<locals>.named_handle')
            assert named_code is not None
            thread = threading.current_thread()
            captured = []
            invalid = []
            selected_path = siblings[0].resolve(strict=True)
            tool_id = next(number for number in range(5, -1, -1)
                           if sys.monitoring.get_tool(number) is None)
            sys.monitoring.use_tool_id(tool_id, 'binding-original-tree-close-custody')

            def returned(code, offset, value):
                if code is not native_code or captured or threading.current_thread() is not thread:
                    return
                frame = sys._getframe(1)
                named = many = binding = None
                for _ in range(32):
                    if frame is None:
                        break
                    if frame.f_code is named_code:
                        named = frame
                    elif frame.f_code is many_code:
                        many = frame
                    elif frame.f_code is binding_code:
                        binding = frame
                    frame = frame.f_back
                if named is None or many is None or binding is None:
                    return
                if named.f_locals.get('node') != selected_path:
                    return
                if (named.f_locals.get('native') is not native
                        or many.f_locals.get('self') is not original_facade
                        or type(value) is not int or value <= 0):
                    invalid.append('exact_tree_open_identity_unqualified')
                    return
                if outcome == 'production_drift':
                    batch_function.__code__ = batch_original_code.replace(
                        co_filename=batch_original_code.co_filename + '.controlled-drift')
                    captured.append(value)
                    return
                if not protect(value, 2, 2):
                    invalid.append('protect_exact_owned_handle_failed')
                    return
                captured.append(value)

            error_seen = None
            try:
                source_current()
                assert sys.monitoring.get_events(tool_id) == 0
                sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_RETURN, returned)
                sys.monitoring.set_local_events(tool_id, native_code, sys.monitoring.events.PY_RETURN)
                try:
                    bootstrap._binding(selector, profiles, registry)
                except windows_files._AdmissionMetadataCloseError as error:
                    error_seen = error
                    assert len(captured) == 1 and len(error.failed_handles) == 1
                    retained = error.failed_handles[0]
                    assert retained.native is native and retained.handle == captured[0]
                    assert retained.path == selected_path
                    actual_info = native.info(captured[0])
                    actual_identity = (actual_info.volume,
                                       (actual_info.index_high << 32) | actual_info.index_low)
                    assert retained.identity == actual_identity
                except ValueError as error:
                    if outcome != 'production_drift':
                        raise
                    error_seen = error
            finally:
                try:
                    sys.monitoring.set_local_events(tool_id, native_code, 0)
                    sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_RETURN, None)
                    final_global = sys.monitoring.get_events(tool_id)
                    final_local = sys.monitoring.get_local_events(tool_id, native_code)
                finally:
                    sys.monitoring.free_tool_id(tool_id)
                # Only this test's exact protected native handle is reclaimed.
                cleanup_ok = True
                if outcome == 'production_drift':
                    batch_function.__code__ = batch_original_code
                else:
                    for handle in captured:
                        cleanup_ok = bool(protect(handle, 2, 0)) and cleanup_ok
                        cleanup_ok = bool(native.kernel.CloseHandle(handle)) and cleanup_ok
            assert final_global == 0 and final_local == 0
            assert sys.monitoring.get_tool(tool_id) is None and cleanup_ok and not invalid
            assert len(captured) == 1, 'actual_binding_tree_close_route_was_not_reached'
            if outcome == 'production_drift':
                assert type(error_seen) is ValueError
                assert error_seen.args == ('binding_tree_reader_changed',)
                handle_flags = windows_files._U32()
                get_flags = kernel.GetHandleInformation
                get_flags.argtypes = [windows_files._HANDLE, C.POINTER(windows_files._U32)]
                get_flags.restype = windows_files._I32
                C.set_last_error(0)
                assert not get_flags(captured[0], C.byref(handle_flags)) and C.get_last_error() == 6
                details = {'actual_production_body_drift_refused': True,
                           'first_exact_original_batch_handle_physically_closed': True,
                           'production_refusal_reason': error_seen.args[0],
                           'mutated_bytecode_semantics': False,
                           'global_events': 0, 'local_events_retired': True, 'tool_retired': True}
            else:
                assert type(error_seen) is windows_files._AdmissionMetadataCloseError
                assert len(error_seen.failed_handles) == 1
                retained = error_seen.failed_handles[0]
                assert retained.native is native and retained.handle == captured[0]
                assert retained.path == selected_path and retained.identity is not None
                assert retained.close_error is not None
                details = {'windows_close_custody_qualified': True,
                           'exact_failed_handle_retained': True,
                           'only_exact_test_handle_cleanup': True,
                           'global_events': 0, 'local_events_retired': True, 'tool_retired': True}
            source_current()
            assert bootstrap._binding(selector, profiles, registry) is record
    source_current()
    assert not blocked_attempts()
    receipt = {'outcome': outcome, 'source_current': True, 'network_attempts': 0,
               'classification': 'No-App original binding refusal/custody or exact observer provenance control',
               **details}
    (selector.parent.parent / ('binding-refusal-' + outcome + '.json')).write_text(
        json.dumps(receipt, indent=2), encoding='utf-8')
    print('retired and reopened')

with user_fixture_default_owner():
    main()
"""


@pytest.mark.parametrize(
    "outcome",
    [
        "parent_permission",
        "source_code",
        "tree_close",
        "production_drift",
        "custom_reader",
    ],
)
def test_original_binding_permission_provenance_and_tree_retirement(tmp_path, outcome):
    _run(tmp_path, "binding_tree", outcome, script=_SCRIPT)
