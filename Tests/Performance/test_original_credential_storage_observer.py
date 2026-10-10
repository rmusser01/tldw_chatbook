"""First source-qualified eight-tick control, using original census code."""

import json
from types import CodeType, FunctionType

import pytest

from Tests.private_profile import private_profile_test


def _cell(value):
    return (lambda: value).__closure__[0]


def _original_nested(function, name, values):
    code = next(
        value
        for value in function.__code__.co_consts
        if type(value) is CodeType and value.co_name == name
    )
    assert code.co_qualname == function.__qualname__ + ".<locals>." + name
    return FunctionType(
        code,
        function.__globals__,
        name,
        closure=tuple(_cell(values[key]) for key in code.co_freevars),
    )


@pytest.mark.asyncio
@private_profile_test
async def test_eight_original_credential_ticks_keep_stock_sources(
    monkeypatch, tmp_path, request
):
    from textual import worker_manager
    from Tests.Performance import test_console_keystroke_work_census as census

    before_receipts = len(census._STORAGE_UNIT_OBSERVER_RECEIPTS)
    original_phase_owner = census._census_idle_and_visit
    measured = {}

    async def eight_tick_only(
        pilot, counts, counting, trace_maintenance, fixture, media_cleanup
    ):
        # The original census must select its private config before App imports.
        from tldw_chatbook.Backup_Recovery import config_participants, storage_admission
        from tldw_chatbook.DB.private_sqlite_process import HelperLease
        from tldw_chatbook.Utils import sensitive_paths

        aliases = (
            config_participants.operation,
            storage_admission._acquire_storage,
            HelperLease.__dict__["start"],
        )
        # Real original source selection remains available while monitoring is
        # installed. This does not execute custom readers or bless replacements.
        _, _, key = sensitive_paths._raw_inputs_key()
        assert sensitive_paths._stock_sensitive_config_bundle(key) is not None
        console = pilot.app.screen
        started, phases = [], {}
        real_new_worker = worker_manager.WorkerManager._new_worker
        values = dict(
            pilot=pilot,
            counts=counts,
            counting=counting,
            phases=phases,
            started=started,
            console=console,
            real_new_worker=real_new_worker,
        )
        phase = _original_nested(original_phase_owner, "phase", values)
        ticks = _original_nested(original_phase_owner, "credential_ticks", values)
        recorder = _original_nested(
            original_phase_owner, "recording_new_worker", values
        )
        worker_manager.WorkerManager._new_worker = recorder
        try:
            await phase("idle", ticks)
        finally:
            worker_manager.WorkerManager._new_worker = real_new_worker
        assert census.IDLE_TICKS == 8
        measured.update(phases)
        counts.update(phases)
        assert aliases == (
            config_participants.operation,
            storage_admission._acquire_storage,
            HelperLease.__dict__["start"],
        )

    # Only the test's phase dispatcher is narrowed. The original App setup,
    # twenty-four key actions, eight calls, phase settle/worker limits and App
    # teardown remain original. No production callback or timer is replaced.
    monkeypatch.setattr(census, "_census_idle_and_visit", eight_tick_only)
    await census._census(
        monkeypatch, tmp_path, census.SEEDED_MESSAGES, storage_units=True
    )
    receipts = census._STORAGE_UNIT_OBSERVER_RECEIPTS[before_receipts:]
    assert len(receipts) == 1 and receipts[0]["complete"]
    assert receipts[0]["original_source_current"]
    assert receipts[0]["hooks_retired_before_inactive"]
    assert receipts[0]["global_events"] == 0
    assert measured["idle:config_admissions"] >= 0
    assert measured["idle:storage_admissions"] >= 0
    assert measured["idle:helper_spawns"] >= 0
    assert measured["idle:os_opens"] >= 0
    request.node.user_properties.append(
        ("original_storage_observer", json.dumps(receipts[0], sort_keys=True))
    )
    request.node.user_properties.append(
        ("original_credential_ticks", json.dumps(measured, sort_keys=True))
    )
    # Preserve the original per-tick oracle and ceiling; never infer acceptance
    # from helper identity alone or a diagnostic-only timing observation.
    for unit in census.IO_UNITS:
        ceiling = census._ceiling(
            "credential poll (per tick)",
            unit,
            census.MAX_CREDENTIAL_POLL_STORAGE_UNITS_PER_TICK,
        )
        if ceiling is not None:
            slack = census.OS_OPENS_JITTER_SLACK if unit == "os_opens" else 1
            assert measured[f"idle:{unit}"] / 8 <= ceiling * slack
