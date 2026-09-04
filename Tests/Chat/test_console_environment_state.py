"""Pure-state tests for the Console Environment panel (no I/O, no Textual app)."""
from datetime import datetime, timedelta, timezone

from tldw_chatbook.Chat.console_environment_state import (
    EnvSourceAvailability,
    ExecTargetKind,
    GitEnvState,
    PrEnvState,
    TasksEnvState,
    ExecTargetState,
    EnvironmentSnapshot,
    branch_task_id,
    compact_count,
    relative_age,
    signed_change_counts,
)


def test_compact_count_small_numbers_keep_thousands_separators():
    assert compact_count(0) == "0"
    assert compact_count(1204) == "1,204"
    assert compact_count(99_999) == "99,999"


def test_compact_count_large_numbers_compress():
    assert compact_count(277_870) == "278k"
    assert compact_count(1_679_102) == "1.7M"


def test_signed_change_counts_pairs_plus_and_minus():
    assert signed_change_counts(1204, 86) == "+1,204 −86"
    assert signed_change_counts(1_679_102, 277_870) == "+1.7M −278k"


def test_branch_task_id_matches_plain_and_subtask_ids():
    assert branch_task_id("feat/task-3401-video-generation-foundation") == "3401"
    assert branch_task_id("fix/task-3401.6-comfyui-adapter") == "3401.6"
    assert branch_task_id("chore/no-task-reference-here") is None
    assert branch_task_id(None) is None


def test_relative_age_buckets():
    now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
    assert relative_age(None, now) == ""
    assert relative_age(now - timedelta(minutes=5), now) == "5m ago"
    assert relative_age(now - timedelta(hours=3), now) == "3h ago"
    assert relative_age(now - timedelta(days=6), now) == "6d ago"


def test_environment_snapshot_defaults_are_not_applicable():
    snapshot = EnvironmentSnapshot()
    assert snapshot.git.availability is EnvSourceAvailability.NOT_APPLICABLE
    assert snapshot.pr.availability is EnvSourceAvailability.NOT_APPLICABLE
    assert snapshot.tasks.availability is EnvSourceAvailability.NOT_APPLICABLE
    assert snapshot.target.kind is ExecTargetKind.LOCAL
