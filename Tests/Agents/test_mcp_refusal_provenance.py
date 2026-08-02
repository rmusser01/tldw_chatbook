"""TASK-294: refusal copy must say WHO refused, and When must read local.

Two small honesty fixes from the PR #675 review backlog.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.mark.unit
def test_an_explicit_user_deny_is_not_blamed_on_permissions():
    """The model (and transcript) must distinguish a user's "Deny" from a
    permanent permissions-Off state.

    `_apply_verdict` returned `DENY_REFUSAL` ("blocked by MCP permissions
    (set to Off)") for an explicit card denial -- misleading provenance: the
    permissions were NOT off, a person said no to this call. A model reading
    the old copy retries never (config problem); reading the new copy it can
    ask the user or move on. The wording matches the builtin gate's and the
    review hook's user-denial copy, so a refusal reads the same at every
    layer.
    """
    from tldw_chatbook.Agents import mcp_tool_provider as mtp

    assert mtp.USER_DENY_REFUSAL != mtp.DENY_REFUSAL
    assert "denied by the user" in mtp.USER_DENY_REFUSAL
    assert "permissions" not in mtp.USER_DENY_REFUSAL.lower()

    # And the verdict path actually uses it: source-level pin, so a revert
    # to the shared constant fails here even without a full provider drive.
    import inspect

    src = inspect.getsource(mtp.MCPToolProvider._apply_verdict)
    assert "USER_DENY_REFUSAL" in src, (
        "_apply_verdict's explicit-deny path no longer uses the "
        "user-denial copy"
    )


@pytest.mark.unit
def test_audit_when_column_renders_local_time_not_raw_utc():
    """A UTC-aware ISO timestamp must display in the VIEWER's timezone.

    `_format_when` parsed the ISO string and strftime'd it unchanged, so an
    aware UTC value rendered as UTC wall-clock with no marker -- a "When"
    column that is wrong by the viewer's UTC offset, silently. Naive values
    (no tz recorded) still render as-is: inventing a zone for them would be
    a different lie.
    """
    from tldw_chatbook.UI.MCP_Modules.mcp_audit_mode import _format_when

    aware = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)
    expected = aware.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    assert _format_when(aware.isoformat()) == expected

    # Naive input: unchanged behavior, no invented zone.
    assert _format_when("2026-08-02T09:30:00") == "2026-08-02 09:30:00"
    # Defensive paths unchanged.
    assert _format_when("") == "—"
    assert _format_when("not-a-date") == "not-a-date"


@pytest.mark.unit
def test_an_unresolved_verdict_is_not_audited_as_an_explicit_denial():
    """The audit log must agree with the transcript (review finding).

    The first version of the provenance split fixed the model-facing STRING
    but still recorded `decision="denied"` for a missing verdict -- so
    Decision-based audit filtering reported an explicit denial nobody made.
    The same principle, one layer deeper: `denied-unresolved` mirrors the
    existing `denied-timeout` vocabulary, and the audit view treats it as
    Blocked (the call never reached the tool).
    """
    import inspect

    from tldw_chatbook.Agents import mcp_tool_provider as mtp
    from tldw_chatbook.UI.MCP_Modules import mcp_audit_mode as audit

    src = inspect.getsource(mtp.MCPToolProvider._apply_verdict)
    assert '"denied-unresolved"' in src, (
        "the unresolved branch records plain 'denied' -- the audit log "
        "reports an explicit denial nobody made"
    )
    assert "denied-unresolved" in audit._BLOCKED_DECISIONS, (
        "the audit Outcome column would route an unresolved refusal through "
        "the attempted-run failure template"
    )
    assert any(
        value == "denied-unresolved" for _label, value in audit._DECISION_OPTIONS
    ), "the Decision filter cannot select unresolved refusals"
