"""Exact startup publication rejects revoked custody after its completed check."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Backup_Recovery.test_startup_pending_readmission import _SCRIPT


_PROBE = r"""
from Tests import real_profile_guard as fence_guard
fence_guard.install()
fence_variant = @VARIANT@
fence_expected = @EXPECTED@
fence_entered, fence_release = threading.Event(), threading.Event()
fence_captured, fence_observed, fence_selected, fence_detached = [], [], [], []
fence_check = storage._StartupReacquisition.check.__code__
fence_initializing = storage._Acquisition.initializing.__wrapped__.__code__
fence_readmission = storage._reacquire_paused_startup.__code__
fence_previous = threading.getprofile()
fence_saved = {}
fence_loop, fence_async_entered = None, None

def fence_observe(frame, event, value):
    if event != 'return':
        return
    if frame.f_code is fence_check and frame.f_back is not None and frame.f_back.f_code is fence_initializing and not fence_captured:
        attempt = frame.f_locals['self']
        assert type(attempt) is storage._StartupReacquisition
        assert not storage._lock._is_owned(), 'the completed initial check moved under the coordinator'
        assert attempt in storage._pending_acquisitions
        assert attempt.thread is threading.current_thread()
        assert attempt.pause._startup_thread is threading.current_thread()
        assert storage._pause is attempt.pause
        assert frame.f_locals['path'] is None and attempt.operation is None
        fence_captured.append(attempt)
        if fence_variant == 'unissued':
            # An exact-type copy carries the actual metadata but was never issued.
            copy = object.__new__(storage._StartupReacquisition)
            copy.__dict__.update(attempt.__dict__)
            try:
                with copy.initializing(bootstrap.default_bootstrap_root(), None):
                    raise AssertionError('unissued exact-type startup copy entered initialization')
            except bootstrap.RecoveryRequired as error:
                assert error.args == ('acquisition_provenance_invalid',)
            try:
                with attempt.initializing(bootstrap.default_bootstrap_root(), Path(os.environ['TLDW_CONFIG_PATH'])):
                    raise AssertionError('startup custody acquired file path authority')
            except bootstrap.RecoveryRequired as error:
                assert error.args == ('startup_reacquisition_invalid',)
        fence_entered.set()
        fence_loop.call_soon_threadsafe(fence_async_entered.set)
        assert fence_release.wait(10), 'native publication observer was abandoned'
    elif frame.f_code is fence_readmission and fence_captured:
        attempt = fence_captured[0]
        pause = frame.f_locals['pause']
        fence_observed.append(pause._startup_error.args)
        # Restore only the injected metadata for original parent/finalizer checks.
        # Never reenroll a retired pending attempt or alter its recorded refusal.
        with storage._changed:
            for name, original in fence_saved.items():
                if name.startswith('pause:'):
                    setattr(pause, name.split(':', 1)[1], original)
                else:
                    setattr(attempt, name, original)


def fence_select(frame, event, value):
    # The original coordinator assigns this exact Thread before starting it.
    # Other newly created workers retain the prior observer immediately.
    pause = storage._pause
    observer = (
        fence_observe
        if pause is not None
        and threading.current_thread() is getattr(pause, '_startup_thread', None)
        else fence_previous
    )
    (fence_selected if observer is fence_observe else fence_detached).append(threading.current_thread())
    sys.setprofile(observer)
    if observer is not None:
        observer(frame, event, value)

async def fence_revoke():
    async with asyncio.timeout(30):
        await fence_async_entered.wait()
    attempt = fence_captured[0]
    with storage._changed:
        if fence_variant == 'cancel':
            attempt.cancel.set()
        elif fence_variant == 'pending':
            storage._pending_acquisitions.remove(attempt)
        elif fence_variant in ('pid', 'thread', 'task', 'operation'):
            original = getattr(attempt, fence_variant)
            fence_saved[fence_variant] = original
            replacement = -1 if fence_variant == 'pid' else threading.Thread() if fence_variant == 'thread' else object()
            setattr(attempt, fence_variant, replacement)
        elif fence_variant == 'stale_pause':
            fence_saved['pause'] = attempt.pause
            attempt.pause = object()
        elif fence_variant == 'startup_worker':
            fence_saved['pause:_startup_thread'] = attempt.pause._startup_thread
            attempt.pause._startup_thread = threading.Thread()
        elif fence_variant == 'source_selection':
            original = attempt.pause._startup_source
            fence_saved['pause:_startup_source'] = original
            attempt.pause._startup_source = (original[0], original[1].with_name('other-config.toml'), *original[2:])
    fence_release.set()

async def fence_main():
    global fence_loop, fence_async_entered
    fence_loop = asyncio.get_running_loop()
    fence_async_entered = asyncio.Event()
    revoking = asyncio.create_task(fence_revoke())
    try:
        await main()
        await revoking
        assert len(fence_captured) == 1 and fence_observed == [(fence_expected,)], fence_observed
        with storage._lock:
            assert not storage._pending_acquisitions
            assert not storage._operations and not storage._raw_operations and not storage._retiring_holds
            assert not (storage._live_leases - set(storage._startups.values()))
        with fence_guard._lock:
            assert not fence_guard._violations
        assert len(fence_selected) == 1 and fence_selected[0] is fence_captured[0].thread
        import json
        receipt = {
            'variant': fence_variant,
            'storage_module': storage.__file__,
            'selected_exact_startup_thread': True,
            'target_checks': len(fence_captured),
            'observed_refusal': fence_observed,
            'other_new_threads_restored': len(fence_detached),
            'guard_refusals': 0,
            'native_retired': True,
        }
        (Path(os.environ['USERPROFILE']).parent / 'startup-publication-owner.json').write_text(json.dumps(receipt), encoding='utf-8')
    finally:
        fence_release.set()
        revoking.cancel()
        await asyncio.gather(revoking, return_exceptions=True)
        threading.setprofile(fence_previous)
"""


@pytest.mark.parametrize(
    ("revocation", "reason"),
    [
        ("unissued", "recovery_scope_uncertain"),
        ("cancel", "storage_locally_paused"),
        ("pending", "acquisition_provenance_invalid"),
        ("pid", "acquisition_provenance_invalid"),
        ("thread", "acquisition_provenance_invalid"),
        ("task", "acquisition_provenance_invalid"),
        ("operation", "acquisition_provenance_invalid"),
        ("stale_pause", "startup_reacquisition_invalid"),
        ("startup_worker", "startup_reacquisition_invalid"),
        ("source_selection", "startup_reacquisition_invalid"),
    ],
)
def test_actual_startup_attempt_rechecks_custody_after_completed_check(
    tmp_path, revocation, reason
):
    probe = _PROBE.replace("@VARIANT@", repr(revocation)).replace(
        "@EXPECTED@", repr(reason)
    )
    expected = "assert pause._startup_error.args==('recovery_scope_uncertain',)"
    assert _SCRIPT.count(expected) == 1 and _SCRIPT.count("asyncio.run(main())") == 1
    script = _SCRIPT.replace(
        expected, f"assert pause._startup_error.args==({reason!r},)"
    )
    # Install only for the native threads that can reach startup publication;
    # the original app constructor and restore setup complete first.
    marker = "    monitoring=asyncio.create_task(monitor_app(app))"
    assert script.count(marker) == 1
    script = script.replace(marker, "    threading.setprofile(fence_select)\n" + marker)
    script = script.replace(
        "asyncio.run(main())", probe + "\nasyncio.run(fence_main())"
    )
    _run(tmp_path, "isolated", "pending_failure", script=script)
