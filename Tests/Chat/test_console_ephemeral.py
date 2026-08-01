"""Temporary (non-persisted) Console conversations: vocabulary and blocked actions."""

import pytest

from tldw_chatbook.Chat.console_ephemeral import (
    EPHEMERAL_BLOCKED_ACTIONS,
    TEMPORARY_LABEL,
    blocked_reason,
)


@pytest.mark.unit
def test_blocked_reason_only_applies_to_temporary_sessions():
    """A normal chat blocks nothing; a temporary one blocks the audited sinks."""
    for action_id in EPHEMERAL_BLOCKED_ACTIONS:
        assert blocked_reason(action_id, ephemeral=False) is None
        reason = blocked_reason(action_id, ephemeral=True)
        assert isinstance(reason, str) and reason.strip()

    assert blocked_reason("send", ephemeral=True) is None


@pytest.mark.unit
def test_blocked_reasons_name_the_artifact_not_the_feature():
    """Each reason says what would hit disk -- 'disabled' alone teaches nothing."""
    for action_id, reason in EPHEMERAL_BLOCKED_ACTIONS.items():
        assert "temporary chat" in reason, action_id
        assert reason == reason.strip()


@pytest.mark.unit
def test_user_facing_copy_never_overstates_the_guarantee():
    """The promise is local durability only -- not privacy, not anonymity."""
    forbidden = ("private", "anonym", "untracked", "incognito", "secure")
    copy = " ".join([TEMPORARY_LABEL, *EPHEMERAL_BLOCKED_ACTIONS.values()]).lower()
    for word in forbidden:
        assert word not in copy, f"copy overstates the guarantee: {word!r}"
