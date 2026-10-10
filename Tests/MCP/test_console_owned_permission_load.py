"""The named Console payload read preserves ordinary permission-store behavior."""

from contextlib import contextmanager
import json

import pytest

from Tests.Backup_Recovery import test_raw_owned_permission_load_counts as count_cases
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.MCP import console_snapshot, permission_store

configured_source = count_cases.configured_source
local_root = count_cases.local_root
installed_source_case = count_cases.installed_source_case
checked_permission_case = count_cases.checked_permission_case
pytestmark = pytest.mark.parametrize(
    "installed_source_case", ["permission"], indirect=True
)


@pytest.fixture
def permission_case(checked_permission_case):
    states, leases = set(raw._states), set(storage._live_leases)
    yield checked_permission_case
    assert set(raw._states) == states
    assert set(storage._live_leases) == leases


def _fresh_payload():
    return {
        "schema_version": 1,
        "kill_switch": False,
        "profiles": {"default": {"global_default": "ask", "servers": {}}},
    }


@pytest.mark.parametrize(
    "profiles, expected_state",
    [
        (None, "ask"),
        (
            {
                "default": {"servers": None},
                "work": {"global_default": "deny", "servers": None},
            },
            "deny",
        ),
    ],
)
def test_owned_payload_matches_public_normalization_and_named_policy(
    permission_case, profiles, expected_state
):
    case = permission_case
    case.path.write_text(
        json.dumps({"schema_version": 1, "kill_switch": False, "profiles": profiles}),
        encoding="utf-8",
    )
    before = case.path.read_bytes()
    expected = case.source.load()
    actual = count_cases.read_checked_payload(case)
    assert actual == expected
    assert actual["profiles"]["default"]["servers"] == {}
    assert (
        permission_store.resolve_effective_state_by_key(
            actual, "local:example", "read", profile_id="work"
        ).state
        == expected_state
    )
    assert case.path.read_bytes() == before


def test_owned_missing_file_returns_fresh_defaults_without_creating_source(
    permission_case,
):
    case = permission_case
    case.path.unlink()
    assert count_cases.read_checked_payload(case) == _fresh_payload()
    assert not case.path.exists()
    assert not case.path.with_suffix(".json.bak").exists()


@pytest.mark.parametrize(
    "contents", [b"{not valid json", b'{"schema_version": 99, "kill_switch": true}']
)
def test_owned_corrupt_payload_keeps_original_backup_and_default_behavior(
    permission_case, contents
):
    case = permission_case
    case.path.write_bytes(contents)
    assert count_cases.read_checked_payload(case) == _fresh_payload()
    assert not case.path.exists()
    assert case.path.with_suffix(".json.bak").read_bytes() == contents
    assert case.source._mcp_persistence_error is None


@pytest.mark.parametrize("name", ["load", "_load_locked"])
def test_custom_load_members_decline_owned_body_and_keep_ordinary_call_shape(
    permission_case, monkeypatch, name
):
    case = permission_case
    expected = case.source.load()
    original = getattr(case.source, name)
    calls = []

    def custom(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(case.source, name, custom)
    assert permission_store._capture_console_owned_load(case.source) is None
    assert calls == []  # Qualification cannot invoke the changed reader.
    if name == "load":
        # Public load substitution already declines the outer Console qualifier.
        # Preserve its preceding checked-read callback ABI on that ordinary route.
        assert not console_snapshot.standard_console_sources(case.service)
        actual, receipt = console_snapshot._checked_read(case.source, case.source.load)
        assert receipt == console_snapshot._source_identity(case.source)
    else:
        actual = count_cases.read_checked_payload(case)
    assert actual == expected
    assert calls == [((), {})]


def test_custom_mutation_fence_is_used_by_ordinary_named_read(
    permission_case, monkeypatch
):
    case = permission_case
    expected = case.source.load()
    original = case.source.mutation_fence
    calls = []

    @contextmanager
    def custom():
        calls.append("enter")
        with original():
            try:
                yield
            finally:
                calls.append("exit")

    monkeypatch.setattr(case.source, "mutation_fence", custom)
    assert permission_store._capture_console_owned_load(case.source) is None
    assert calls == []
    assert count_cases.read_checked_payload(case) == expected
    assert calls == ["enter", "exit"]
