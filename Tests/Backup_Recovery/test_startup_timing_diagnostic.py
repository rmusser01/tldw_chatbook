"""The optional startup observer emits aggregate code metadata only."""

import cProfile
import json
import sys
from types import FunctionType

import pytest

from Tests.Backup_Recovery import startup_timing_diagnostic as diagnostic
from Tests.Backup_Recovery.startup_timing_diagnostic import snapshot


def test_profile_snapshot_is_bounded_and_excludes_call_values(tmp_path):
    def observed(value):
        return value

    profile = cProfile.Profile()
    for index in range(50):
        code = observed.__code__.replace(
            co_name=f"call_{index}",
            co_filename=str(tmp_path / "tldw_chatbook" / "fixture.py"),
        )
        profile.runcall(FunctionType(code, {}), "private-value-must-not-appear")
    record = snapshot(profile, tmp_path, "candidate")
    assert len(record["largest_total"]) == len(record["largest_self"]) == 20
    assert record["source_matches"] in (None, False)
    assert {row["file"] for row in record["largest_total"]} == {"fixture.py"}
    assert all(row["calls"] == 1 for row in record["largest_self"])
    assert "private-value-must-not-appear" not in json.dumps(record)
    assert str(tmp_path) not in json.dumps(record)


@pytest.mark.parametrize("failure", ["snapshot", "write", "thread_start"])
@pytest.mark.parametrize("raises", [False, True])
def test_observer_failure_preserves_test_outcome_and_retires_profile(
    tmp_path, monkeypatch, failure, raises
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("TLDW_TEST_PRIVATE_PROFILE_NODE", "before")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnostic",
            "--source",
            str(tmp_path),
            "--node",
            "fixture-node",
            "--label",
            "candidate",
        ],
    )
    original = RuntimeError("original test failure")

    def run(args):
        assert args == ["fixture-node", "--timeout=60", "-q", "--capture=no"]
        if raises:
            raise original
        return 7

    def observation_failed(*args, **kwargs):
        raise RuntimeError("optional observer failure")

    monkeypatch.setattr(pytest, "main", run)
    if failure == "thread_start":
        monkeypatch.setattr(diagnostic.threading.Thread, "start", observation_failed)
    elif failure == "write":
        monkeypatch.setattr(diagnostic, "print", observation_failed, raising=False)
    else:
        monkeypatch.setattr(diagnostic, "snapshot", observation_failed)
    if raises:
        with pytest.raises(RuntimeError) as caught:
            diagnostic.main()
        assert caught.value is original
    else:
        assert diagnostic.main() == 7
    assert sys.getprofile() is None
