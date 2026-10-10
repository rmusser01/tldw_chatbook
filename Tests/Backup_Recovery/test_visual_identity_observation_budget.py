"""Real finite Shared Visual metadata cost and source/path boundaries."""

import os

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run


_SCRIPT = r"""
import hashlib, json, os, sys
from io import BytesIO
from pathlib import Path
from Tests.network_guard import install as network_install, blocked_attempts
from Tests import real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_install()
real_profile_guard.install()
for name in ('sounddevice', 'pyaudio'): sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
from loguru import logger
keyring.set_keyring(Keyring())
logger.remove()
route, outcome = sys.argv[1:]
uncertain = outcome.startswith('uncertain_close')

def main():
    if outcome == 'uncertain_close_admission':
        from tldw_chatbook.Backup_Recovery import bootstrap, control_records
        selector = Path(os.environ['TLDW_CONFIG_PATH'])
        enrolled_profile = Path(os.environ['XDG_DATA_HOME']) / 'native-visual-profile'
        enrolled_profile.mkdir(mode=0o700)
        selector.write_text('[paths]\ndata_dir=' + json.dumps(str(enrolled_profile).replace('\\', '/')) + '\n', encoding='utf-8')
        authority = control_records.admission_authority(bootstrap.default_bootstrap_root())
        authority.register('visual.native', (selector, enrolled_profile))
        control_records.bind_profile(bootstrap.default_bootstrap_root(), selector, ('visual.native',), bootstrap.default_bootstrap_root() / 'admission')
    from tldw_chatbook import config
    from tldw_chatbook.Character_Chat import visual_identity as visual
    from tldw_chatbook.Backup_Recovery import visual_identity_participants as life
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Utils import windows_files
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    config.load_settings()
    original_facade = life.os
    original_stat = original_facade.stat
    snapshot_code = windows_files.WindowsOS.stat_many_for_admission.__code__
    info_code = windows_files._Native.info.__code__
    stamps_code = storage._observe_stamps.__code__
    original_selector = os.environ['TLDW_CONFIG_PATH']
    native_code = windows_files._Native.open_handle.__code__
    read_code = visual._read_samira_resource.__wrapped__.__code__
    check_code = life._check_path.__code__
    identity_code = life._identity.__code__
    files_code = life.files.__wrapped__.__code__
    acquire_code = storage.acquire_storage.__code__
    observer = getattr(life, '_observe_identities', None)
    observer_code = getattr(observer, '__code__', None)
    native_calls = 0
    resource_reads = 0
    witnesses = []
    changed = []
    pauses = []
    protected = []
    protected_state = []
    admission_stamps = []
    admission_results = []
    database = None
    profile = config.get_user_data_dir()
    selected = None
    source = None
    custom_calls = []
    if uncertain:
        import ctypes as C
        kernel = C.WinDLL('kernel32', use_last_error=True)
        set_handle = kernel.SetHandleInformation
        set_handle.argtypes = [windows_files._HANDLE, windows_files._U32, windows_files._U32]
        set_handle.restype = windows_files._I32
        get_handle = kernel.GetHandleInformation
        get_handle.argtypes = [windows_files._HANDLE, C.POINTER(windows_files._U32)]
        get_handle.restype = windows_files._I32
        snapshot_code = windows_files.WindowsOS.stat_many_for_admission.__code__
    original_bindings = (life._identity, life._check_path, life.files,
        life.native_open, visual._read_samira_resource, visual.ensure_builtin_samira,
        windows_files.WindowsOS.stat, windows_files.WindowsOS.stat_many_for_admission,
        windows_files._Native.open_handle, windows_files._Native.info)

    def census():
        return {
            'ordinary': sum(lease not in storage._startups.values() for lease in storage._live_leases),
            'pending': len(storage._pending_acquisitions),
            'operations': len(storage._operations),
            'raw': len(storage._raw_operations),
            'retiring': len(storage._retiring_holds),
            'visual_states': len(life._states),
        }

    if route == 'seed':
        database = CharactersRAGDB(config.get_chachanotes_db_path(), 'samira-native-budget')
        source = life.db_source(database)
        assert source is not None and type(database) is CharactersRAGDB and not database.is_memory_db
        assert database.execute_query('SELECT COUNT(*) FROM visual_identity_assets').fetchone()[0] == 0
    else:
        from PIL import Image
        payload_stream = BytesIO()
        Image.new('RGB', (16, 16), (12, 31, 49)).save(payload_stream, format='PNG')
        payload = payload_stream.getvalue()
        relative = 'packs/profile-00000000000000000000000000000000/versions/original/neutral.png'
        selected = profile / 'visual_identities' / relative
        selected.parent.mkdir(parents=True, mode=0o700)
        selected.write_bytes(payload)
        asset = visual.VisualIdentityManifestAsset('neutral', 'Neutral', 'Neutral', relative,
            'image/png', len(payload), 16, 16, hashlib.sha256(payload).hexdigest(), False, 1, None)
        source = life.source_for(profile)
        assert source is not None and source.config is config
        if outcome in ('uncertain_close_opener', 'uncertain_close_custom'):
            import subprocess
            selected.unlink()
            target = selected.with_name('native-junction-target')
            target.mkdir(mode=0o700)
            assert selected.is_relative_to(profile) and target.is_relative_to(profile)
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(selected), str(target)],
                check=True, capture_output=True, timeout=10)
        if outcome in ('large_dacl', 'uncertain_close_body', 'mid_read_binding_unsupported'):
            import ctypes as C
            from Tests.Utils.test_windows_native_admission import _replace_security
            native = windows_files._native()
            sid = native._token_sid(1)
            _replace_security(selected, 'D:P(A;;FA;;;' + sid + ')'
                + ''.join('(A;;FR;;;S-1-5-21-1-2-3-' + str(index) + ')' for index in range(1000, 1200)))
            fd = original_facade.open(selected, original_facade.O_RDONLY)
            descriptor = windows_files._P()
            try:
                status = native.advapi.GetSecurityInfo(native.handle(fd), 1, 5,
                    None, None, None, None, C.byref(descriptor))
                if status: raise C.WinError(status)
                assert native.advapi.GetSecurityDescriptorLength(descriptor) > 4096
            finally:
                if descriptor.value: native.kernel.LocalFree(descriptor)
                original_facade.close(fd)
        if outcome in ('custom_facade', 'uncertain_close_custom'):
            class CustomFacade:
                def __getattr__(self, name): return getattr(original_facade, name)
                def stat(self, *args, **kwargs):
                    custom_calls.append(True)
                    return original_facade.stat(*args, **kwargs)
                def stat_many_for_admission(self, paths):
                    raise AssertionError('custom facade snapshot must retain scalar contract')
            life.os = CustomFacade()

    def revoke():
        assert selected is not None and source is not None
        assert not changed
        changed.append(True)
        state = getattr(life._local, 'state', None)
        # A preflight revocation precedes active publication; an actual body
        # metadata revocation retains the installed source and native lease.
        if outcome in ('source', 'pause'): assert state is not None and state.active
        if state is not None:
            assert state in life._states and state in storage._raw_operations
            assert state.source is source
        if outcome == 'leaf':
            original = selected.with_name('original-leaf.png')
            selected.rename(original)
            selected.write_bytes(payload)
        elif outcome == 'parent':
            parent = selected.parent
            old = parent.with_name('prior-carrying-parent')
            leaf_identity = (selected.stat().st_dev, selected.stat().st_ino)
            parent.rename(old)
            parent.mkdir(mode=0o700)
            (old / selected.name).rename(selected)
            assert (selected.stat().st_dev, selected.stat().st_ino) == leaf_identity
        elif outcome == 'source':
            os.environ['TLDW_CONFIG_PATH'] = str(Path(os.environ['USERPROFILE']) / 'other-selector.toml')
        elif outcome == 'pause':
            pauses.append(storage._begin_local_pause())
        else:
            raise AssertionError('unknown revocation')

    def replacement_stat(*args, **kwargs):
        custom_calls.append(True)
        return original_stat(*args, **kwargs)

    def observe(frame, event, argument):
        nonlocal native_calls, resource_reads
        if outcome == 'uncertain_close_admission' and event == 'call' and frame.f_code is stamps_code:
            admission_stamps.append(any(state.source is source for state in life._states))
        if outcome == 'uncertain_close_admission' and event == 'return' and frame.f_code is acquire_code:
            admission_results.append(None if argument is None else argument._key is not None)
        if event == 'call' and frame.f_code is native_code:
            native_calls += 1
        if outcome in ('mid_read_binding', 'mid_read_binding_unsupported') and event == 'return' and frame.f_code is native_code and not changed:
            parent = frame.f_back
            caller = parent.f_back if parent is not None else None
            enclosing = caller.f_back if caller is not None else None
            if caller is not None and caller.f_code is snapshot_code and enclosing is not None and enclosing.f_code is observer_code:
                issued = enclosing.f_locals['state']
                assert issued in life._states and issued in storage._raw_operations and issued.source is source
                assert storage._pending_acquisitions
                changed.append(True)
                original_facade.stat = replacement_stat
        if uncertain and outcome in ('uncertain_close_opener', 'uncertain_close_custom') and event == 'call' and frame.f_code is info_code and not protected:
            caller = frame.f_back
            issued = next((state for state in life._states if state.source is source), None)
            if caller is not None and caller.f_code is native_code and caller.f_locals['name'] == selected.name and (route == 'snapshot' or issued is not None):
                handle = frame.f_locals['handle']
                handle = handle.value if hasattr(handle, 'value') else handle
                assert set_handle(handle, 2, 2), C.get_last_error()
                actual = windows_files._FileInfo()
                native = windows_files._native()
                native.check(native.kernel.GetFileInformationByHandle(handle, C.byref(actual)))
                assert actual.attributes & windows_files._REPARSE
                protected.append((handle, (actual.volume, actual.index_high, actual.index_low)))
                if issued is not None: protected_state.append(issued)
        if uncertain and outcome not in ('uncertain_close_opener', 'uncertain_close_custom') and event == 'return' and frame.f_code is native_code and argument is not None and not protected:
            parent = frame.f_back
            caller = parent.f_back if parent is not None else None
            issued = next((state for state in life._states if state.source is source), None)
            if parent is not None and parent.f_code.co_name == 'named_handle' and caller is not None and caller.f_code is snapshot_code and (route == 'snapshot' or issued is not None):
                node = parent.f_locals['node']
                enclosing = caller.f_back
                admitted = outcome != 'uncertain_close_admission' or (enclosing is not None and enclosing.f_code is stamps_code)
                chosen = node == str(selected.parent).lower() if outcome == 'uncertain_close_body' else node not in caller.f_locals['parents']
                if chosen and admitted:
                    if issued is not None: assert issued in storage._raw_operations
                    assert set_handle(argument, 2, 2), C.get_last_error()
                    actual = windows_files._native().info(argument)
                    protected.append((argument, (actual.volume, actual.index_high, actual.index_low)))
                    if issued is not None: protected_state.append(issued)
        if event == 'call' and frame.f_code is read_code:
            resource_reads += 1
            state = getattr(life._local, 'state', None)
            witnesses.append(state is not None and state in life._states
                and state in storage._raw_operations and state.source is source
                and state.repository is database and len(state.files) == len(state.leases) == 35)
        if route != 'read' or changed or event != 'return':
            return
        if outcome in ('leaf', 'parent'):
            if frame.f_code is acquire_code and frame.f_back is not None and frame.f_back.f_code is files_code:
                issued = frame.f_back.f_locals['state']
                assert issued in life._states and issued in storage._raw_operations
                assert issued.source is source and selected in issued.expectations
                assert argument in storage._live_leases
                revoke()
        elif outcome in ('source', 'pause') and frame.f_back is not None and frame.f_back.f_code is check_code:
            if observer_code is not None and frame.f_code is observer_code:
                revoke()
            elif observer_code is None and frame.f_code is identity_code and frame.f_locals['path'] == selected.parents[-1]:
                revoke()

    assert sys.getprofile() is None
    error = None
    loaded = None
    sys.setprofile(observe)
    try:
        if route == 'seed':
            visual.ensure_builtin_samira(database)
        elif route == 'snapshot':
            loaded = original_facade.stat_many_for_admission((selected,))
        else:
            loaded = visual.load_visual_identity_asset(asset, source_kind='manual', user_data_dir=profile)
    except Exception as actual:
        error = actual
    finally:
        sys.setprofile(None)
        life.os = original_facade
        original_facade.stat = original_stat
        os.environ['TLDW_CONFIG_PATH'] = original_selector
        for pause in pauses: pause.resume()
    bindings_after = (life._identity, life._check_path, life.files,
        life.native_open, visual._read_samira_resource, visual.ensure_builtin_samira,
        windows_files.WindowsOS.stat, windows_files.WindowsOS.stat_many_for_admission,
        windows_files._Native.open_handle, windows_files._Native.info)
    assert all(before is after for before, after in zip(original_bindings, bindings_after))
    result = {'route': route, 'outcome': outcome, 'native_opens': native_calls,
        'resource_reads': resource_reads, 'all_actual_source_witnesses': all(witnesses),
        'actual_revocation': bool(changed), 'changed_native_callback_calls': len(custom_calls), 'error_type': type(error).__name__ if error is not None else None,
        'error_category': str(error) if error is not None else None, 'guarded_bindings_unchanged': True}
    if uncertain:
        if not protected:
            print('MISSING_NATIVE_BOUNDARY ' + json.dumps({'stamps': admission_stamps, 'acquisitions': admission_results,
                'error_type': type(error).__name__ if error is not None else None, 'error_category': str(error) if error is not None else None}), flush=True)
        assert protected, 'actual issued snapshot handle boundary was not reached'
        handle, identity = protected[0]
        flags = windows_files._U32()
        alive = bool(get_handle(handle, C.byref(flags)))
        assert alive and flags.value & 2, 'protected physical metadata handle unexpectedly disappeared'
        actual = windows_files._FileInfo()
        native = windows_files._native()
        native.check(native.kernel.GetFileInformationByHandle(handle, C.byref(actual)))
        assert (actual.volume, actual.index_high, actual.index_low) == identity
        issued = protected_state[0] if protected_state else None
        result['metadata_handle_live_after_callback'] = True
        result['metadata_source_state_retained'] = issued is not None and issued in life._states and issued in storage._raw_operations
        result['metadata_state_uncertain'] = issued is not None and issued.uncertain
        if issued is not None and issued.uncertain:
            records = [record for record in issued.failed_metadata_handles if record.handle == handle]
            assert len(records) == 1 and records[0].native is native
            assert records[0].identity is None if outcome in ('uncertain_close_opener', 'uncertain_close_custom') else records[0].identity == (actual.volume, (actual.index_high << 32) | actual.index_low)
            result['captured_failed_native_owner_and_incarnation'] = True
        if outcome == 'uncertain_close_body':
            import errno
            pending, visited, body_error = [error], set(), False
            while pending:
                current_error = pending.pop()
                if current_error is None or id(current_error) in visited: continue
                visited.add(id(current_error))
                body_error |= isinstance(current_error, OSError) and current_error.errno == errno.ENOTSUP
                pending.extend((current_error.__cause__, current_error.__context__))
            assert body_error, 'actual large-DACL body error was not retained'
            result['actual_large_dacl_body_error_retained'] = True
        result['callback_returned_bytes'] = loaded is not None
        # Clear only the real flag applied by this fixture, then positively close
        # this same test-owned handle. Never clear a production source registry.
        assert set_handle(handle, 2, 0), C.get_last_error()
        assert windows_files._native().kernel.CloseHandle(handle), C.get_last_error()
        assert not get_handle(handle, C.byref(flags))
        result['physical_fixture_cleanup_verified'] = True
    if route == 'seed':
        assert error is None
        assert resource_reads == 35 and len(witnesses) == 35 and all(witnesses)
        counts = {table: database.execute_query('SELECT COUNT(*) FROM ' + table).fetchone()[0]
            for table in ('visual_identity_packs', 'visual_identity_pack_versions',
                          'visual_identity_assets', 'visual_identity_bindings')}
        assert counts == {'visual_identity_packs': 1, 'visual_identity_pack_versions': 1,
            'visual_identity_assets': 31, 'visual_identity_bindings': 1}
        result['durable_counts'] = counts
        database.close_connection()
    elif outcome in ('positive', 'large_dacl', 'custom_facade'):
        assert error is None, type(error).__name__
        assert loaded.data == payload
        if outcome == 'custom_facade': assert custom_calls
    elif not uncertain:
        result['callback_returned_original_bytes'] = loaded is not None and loaded.data == payload
        assert changed, 'actual native revocation boundary was not reached'
    result['final_census'] = census()
    if not uncertain:
        assert all(value == 0 for value in result['final_census'].values())
    result['network_attempts'] = len(blocked_attempts())
    result['real_profile_refusals'] = len(real_profile_guard.take_violations())
    assert result['network_attempts'] == result['real_profile_refusals'] == 0
    (Path(os.environ['USERPROFILE']).parent / 'visual-native-observation.json').write_text(json.dumps(result), encoding='utf-8')
    print('VISUAL_NATIVE_OBSERVATION ' + json.dumps(result), flush=True)
    if route == 'seed':
        assert native_calls <= 12000, ('registered native seed budget exceeded', native_calls, 12000)
    elif route == 'snapshot':
        assert type(error) is windows_files._WINDOWS_METADATA_CLOSE_ERROR_ORIGINAL and loaded is None and error.failed_handles, ('native metadata snapshot did not report its unretired handle', result)
    elif uncertain:
        assert error is not None and loaded is None and result['metadata_source_state_retained'] and result['metadata_state_uncertain'], ('native metadata close uncertainty was retired', result)
    elif outcome in ('mid_read_binding', 'mid_read_binding_unsupported'):
        assert isinstance(error, ValueError) and loaded is None and not custom_calls, ('changed native binding accepted or called', result)
    elif outcome not in ('positive', 'large_dacl', 'custom_facade'):
        expected = 'visual_identity_asset_unavailable' if outcome in ('source', 'pause') and os.name == 'nt' else 'visual_identity_path_invalid'
        assert isinstance(error, ValueError) and str(error) == expected, ('stale native visual read accepted', result)

with user_fixture_default_owner(): main()
print('retired and reopened')
"""


@pytest.mark.skipif(os.name != "nt", reason="actual Windows native opens")
def test_standard_samira_seed_bounds_native_opens_without_skipping_assets(tmp_path):
    _run(tmp_path, "seed", "budget", script=_SCRIPT)


@pytest.mark.parametrize(
    "outcome",
    (
        "positive",
        "leaf",
        "parent",
        "source",
        pytest.param(
            "pause",
            marks=pytest.mark.skipif(
                os.name != "nt",
                reason="Windows visual scope has no across-pause authority",
            ),
        ),
        pytest.param(
            "custom_facade",
            marks=pytest.mark.skipif(
                os.name != "nt", reason="Windows custom scalar-facade contract"
            ),
        ),
    ),
)
def test_finite_visual_observation_retains_native_source_and_path_checks(
    tmp_path, outcome
):
    _run(tmp_path, "read", outcome, script=_SCRIPT)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows large security descriptor")
def test_large_dacl_visual_read_preserves_complete_scalar_fallback(tmp_path):
    _run(tmp_path, "read", "large_dacl", script=_SCRIPT)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows protected snapshot handle")
def test_snapshot_close_uncertainty_retains_actual_visual_source(tmp_path):
    _run(tmp_path, "read", "uncertain_close", script=_SCRIPT)


@pytest.mark.skipif(os.name != "nt", reason="actual Windows protected snapshot handle")
def test_original_native_snapshot_rejects_unretired_metadata_handle(tmp_path):
    _run(tmp_path, "snapshot", "uncertain_close", script=_SCRIPT)


@pytest.mark.skipif(os.name != "nt", reason="actual held native snapshot")
def test_mid_snapshot_binding_change_refuses_before_scalar_callback(tmp_path):
    _run(tmp_path, "read", "mid_read_binding", script=_SCRIPT)


@pytest.mark.skipif(os.name != "nt", reason="actual admitted Windows metadata handle")
def test_admission_metadata_close_keeps_actual_issued_visual_state(tmp_path):
    _run(tmp_path, "read", "uncertain_close_admission", script=_SCRIPT)


@pytest.mark.skipif(os.name != "nt", reason="actual rejected junction handle")
@pytest.mark.parametrize("route", ["snapshot", "read"])
def test_rejected_native_open_close_keeps_handle_uncertainty(tmp_path, route):
    _run(tmp_path, route, "uncertain_close_opener", script=_SCRIPT)


@pytest.mark.skipif(
    os.name != "nt", reason="actual large-DACL error and protected parent"
)
def test_body_error_close_retains_original_failure_and_native_custody(tmp_path):
    _run(tmp_path, "read", "uncertain_close_body", script=_SCRIPT)


@pytest.mark.skipif(os.name != "nt", reason="actual unsupported held snapshot")
def test_unsupported_snapshot_binding_change_refuses_before_scalar_callback(tmp_path):
    _run(tmp_path, "read", "mid_read_binding_unsupported", script=_SCRIPT)


@pytest.mark.skipif(
    os.name != "nt", reason="actual custom scalar rejected junction handle"
)
def test_custom_scalar_rejected_open_retains_actual_visual_source(tmp_path):
    _run(tmp_path, "read", "uncertain_close_custom", script=_SCRIPT)
