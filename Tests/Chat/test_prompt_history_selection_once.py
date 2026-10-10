"""Default history selects at its raw operation, not again in its queued worker."""

from __future__ import annotations

import asyncio
from collections import Counter
import json
import os
import threading

import pytest

from Tests.Backup_Recovery.config_test_support import install_config_source
from Tests.Chat import test_default_prompt_history_lifetime as lifetime_controls
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.async_file_participants import _FileJob
from tldw_chatbook.Chat import prompt_history

pytestmark = pytest.mark.asyncio
local_scope = lifetime_controls.local_scope


@pytest.fixture
def history_profile(local_scope, monkeypatch):
    root, configuration, data, _authority = local_scope
    configuration.write_text(
        '[general]\nusers_name = "data"\n[paths]\ndata_dir = '
        + json.dumps(data.parent.as_posix())
        + "\n",
        encoding="utf-8",
    )
    installed = install_config_source(monkeypatch)
    try:
        yield configuration, data, installed
    finally:
        startup = storage._startups.pop((os.getpid(), str(root)), None)
        if startup is not None:
            startup.close()


def _inputs(path):
    return [
        json.loads(line)["input"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


async def test_warm_default_history_has_no_queued_canonical_preselection(
    history_profile,
):
    _configuration, data, _config = history_profile
    history = prompt_history.PromptHistory()
    assert await history.append("warm seed")
    assert history.path == data / "prompt_history.jsonl"
    assert history._loaded
    resolver = prompt_history.default_prompt_history_path
    selection = raw._async_source_selection
    worker = _FileJob._work
    raw_selection = raw._selection
    original_codes = (
        resolver.__code__,
        selection.__code__,
        worker.__code__,
        raw_selection.__code__,
    )
    counts, actors = Counter(), []

    def observe(frame, event, _argument):
        if event != "call" or frame.f_code is not original_codes[0]:
            return
        parent = frame.f_back
        if parent is None or parent.f_code is not original_codes[1]:
            return
        if parent.f_locals.get("source") is not history:
            return
        caller = parent.f_back
        if caller is None:
            return
        if caller.f_code is original_codes[2]:
            route = "queued_preselection"
        elif caller.f_code is original_codes[3]:
            route = "raw_boundary"
        else:
            # POSIX participant binding can legitimately select separately.
            route = "participant_or_other"
        counts[route] += 1
        actors.append(threading.current_thread())

    with lifetime_controls.observe_history_callback(history, observe):
        assert await history.append("warm next send")

    assert _inputs(history.path) == ["warm seed", "warm next send"]
    assert counts["raw_boundary"] == 1, counts
    assert counts["queued_preselection"] == 0, counts
    assert actors and all(actor is not threading.current_thread() for actor in actors)
    assert prompt_history.default_prompt_history_path is resolver
    assert raw._async_source_selection is selection
    assert _FileJob._work is worker
    assert (
        resolver.__code__,
        selection.__code__,
        worker.__code__,
        raw_selection.__code__,
    ) == original_codes


async def test_unbound_default_path_substitution_cannot_become_custom(history_profile):
    _configuration, data, _config = history_profile
    history = prompt_history.PromptHistory()
    substitution = data / "substituted-history.jsonl"
    history.path = substitution
    assert history not in raw._source_participants

    assert await history.append("must not reach substituted source") is False

    assert not substitution.exists()
    assert not history._loaded
    assert history.size == 0
    assert history.persistence_error == "RecoveryRequired"
    assert not history._append_lock.locked()


async def test_unbound_default_profile_movement_cannot_write_old_source(
    history_profile, monkeypatch
):
    _configuration, data, _config = history_profile
    history = prompt_history.PromptHistory()
    original_path = prompt_history.default_prompt_history_path()
    history.path = original_path
    assert history not in raw._source_participants
    other_configuration = data.parent / "moved-config.toml"
    other_data = data.parent / "moved-data"
    other_configuration.write_text(
        '[general]\nusers_name = "data"\n[paths]\ndata_dir = '
        + json.dumps(other_data.as_posix())
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(other_configuration))
    # Let the original history resolver observe the changed source itself.

    assert await history.append("must not write an old selected profile") is False

    assert not original_path.exists()
    assert not (other_data / "data" / "prompt_history.jsonl").exists()
    assert history.size == 0
    assert history.persistence_error == "RecoveryRequired"


async def test_queued_default_origin_cannot_downgrade_to_explicit_path(history_profile):
    _configuration, data, _config = history_profile
    history = prompt_history.PromptHistory()
    assert await history.append("before queued downgrade")
    original_path = history.path
    original_bytes = original_path.read_bytes()
    release = threading.Event()
    with lifetime_controls.observe_history_callback(history) as (observed, executor):
        occupied = executor.submit(release.wait, 5)
        task = asyncio.create_task(history.append("must retain default origin"))
        try:
            for _ in range(300):
                if observed.jobs and observed.jobs[0]._state == "queued":
                    break
                await asyncio.sleep(0.01)
            assert observed.jobs and observed.jobs[0]._state == "queued"
            job = observed.jobs[0]
            assert job._deferred_history is True
            history._default_path = False
            history.path = data / "downgraded-custom.jsonl"
            release.set()
            assert await task is False
            assert job._state == "closed"
            assert not history._append_lock.locked()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            assert occupied.result(5)

    assert original_path.read_bytes() == original_bytes
    assert not (data / "downgraded-custom.jsonl").exists()
    assert _inputs(original_path) == ["before queued downgrade"]
    assert history.size == 1
    assert history.persistence_error == "RecoveryRequired"


@pytest.mark.parametrize("kind", ["explicit", "subclass"])
async def test_explicit_and_custom_history_keep_original_source_route(
    history_profile, kind
):
    _configuration, data, _config = history_profile
    path = data / f"{kind}-history.jsonl"
    history_type = prompt_history.PromptHistory
    if kind == "subclass":

        class CustomHistory(prompt_history.PromptHistory):
            pass

        history_type = CustomHistory
    history = history_type(path, max_entries=1)
    assert await history.append("first")
    assert await history.append("second")
    assert _inputs(path) == ["second"]
    assert history.size == 1
    assert (await history.get_entry(-1))["input"] == "second"
    assert not history._append_lock.locked()
