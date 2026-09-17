"""Settings counts prompts under one continuously held configuration lifetime."""

import pytest

from Tests.Backup_Recovery.test_config_participant_lifetimes import (
    source as source,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants, storage_admission
from tldw_chatbook.Internal_Prompts import authoring


def test_prompt_count_retains_one_scope_and_reads_changes_next_time(
    source, monkeypatch
):
    original = authoring.override_state
    observed = []

    def state(prompt_id):
        operation = getattr(raw_participants._local, "operation", None)
        assert operation is not None, "prompt count reopened config for every value"
        observed.append(operation)
        return original(prompt_id)

    monkeypatch.setattr(authoring, "override_state", state)
    assert authoring.customized_count() == 0
    assert len(observed) == len(authoring.CATALOG)
    assert len(set(observed)) == 1
    assert observed[0] not in raw_participants._states
    source.save_settings_to_cli_config(
        {"internal_prompts.agents": {"subagent_system": {"text": "CUSTOM"}}}
    )
    previous = observed[0]
    observed.clear()
    assert authoring.customized_count() == 1
    assert len(set(observed)) == 1 and observed[0] is not previous
    assert observed[0] not in raw_participants._states


def test_prompt_count_refuses_maintenance_without_reusing_cached_values(source):
    assert authoring.customized_count() == 0
    pause = storage_admission._begin_local_pause()
    try:
        with pytest.raises(bootstrap.RecoveryRequired):
            authoring.customized_count()
    finally:
        pause.resume()


def test_prompt_count_retires_scope_when_resolution_fails(source, monkeypatch):
    observed = []
    failure = ValueError("synthetic resolution failure")

    def fail(prompt_id):
        observed.append(getattr(raw_participants._local, "operation", None))
        raise failure

    monkeypatch.setattr(authoring, "override_state", fail)
    with pytest.raises(ValueError) as caught:
        authoring.customized_count()
    assert caught.value is failure
    assert observed[0] is not None
    assert observed[0] not in raw_participants._states
    assert getattr(raw_participants._local, "operation", None) is None
