"""Controlled scope-policy ordering; no App/native/source qualification."""

import inspect
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_case", ("deny", "revoke", "permitted"))
async def test_scope_policy_precedes_optional_trust_worker(
    monkeypatch, record_property, policy_case
):
    # Complete the natural shipping import order without constructing an App.
    from tldw_chatbook import app  # noqa: F401
    from tldw_chatbook import app_service_wiring as wiring
    from tldw_chatbook import app_skill_wiring as helper
    from tldw_chatbook.runtime_policy.types import PolicyDeniedError
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    events = []
    state = {"current": True}
    denial = PolicyDeniedError(
        action_id="skills.context.list.local",
        reason_code="authority_denied",
        user_message="controlled denial before optional trust",
        effective_source="local",
        authority_owner="local",
    )
    reached_ensure = RuntimeError("controlled_ensure_boundary")

    class Policy:
        def require_allowed(self, *, action_id):
            events.append(("policy", action_id))
            if policy_case == "deny":
                raise denial
            if policy_case == "revoke":
                state["current"] = False

    scope = SkillsScopeService(local_service=None, policy_enforcer=Policy())
    owner = SimpleNamespace(_skills_scope_service=scope)
    reads = set()

    def current():
        events.append(("current", state["current"]))
        return state["current"]

    def source_current():
        raise AssertionError("No source callback or native producer is invoked")

    def ensure(actual_owner, *, _source_current, _owner_current, _read_observers):
        assert actual_owner is owner
        assert _source_current is source_current and _owner_current is current
        assert _read_observers == (reads,)
        events.append(("ensure",))
        # Raise synchronously, before any coroutine or Worker can be created.
        raise reached_ensure

    captured = (owner, source_current, current, (None, None, None, reads))
    original = inspect.getattr_static(
        wiring.ServiceWiringMixin, "ensure_local_skill_trust_service"
    )
    escaped = None
    with monkeypatch.context() as patch:
        patch.setattr(wiring.ServiceWiringMixin, "ensure_local_skill_trust_service", ensure)
        try:
            await helper._prepare_console_skill_trust_service(captured)
        except (PolicyDeniedError, RuntimeError) as error:
            escaped = error

    # Restore before every causal assertion, including the original-source RED.
    assert inspect.getattr_static(
        wiring.ServiceWiringMixin, "ensure_local_skill_trust_service"
    ) is original
    record_property("controlled_policy_case", policy_case)
    record_property("controlled_events", json.dumps(events))
    record_property("escaped_type", type(escaped).__name__ if escaped is not None else None)
    record_property("same_denial", escaped is denial)
    record_property("same_ensure_sentinel", escaped is reached_ensure)
    record_property("controlled_scope", "policy/ensure sentinel; no native qualification")
    assert [row for row in events if row[0] == "policy"] == [
        ("policy", "skills.context.list.local")
    ]
    if policy_case == "deny":
        assert escaped is denial
        assert events == [("current", True), ("policy", "skills.context.list.local")]
    elif policy_case == "revoke":
        assert type(escaped) is RuntimeError
        assert escaped.args == ("console_skill_trust_owner_changed",)
        assert events == [
            ("current", True), ("policy", "skills.context.list.local"),
            ("current", False),
        ]
    else:
        assert escaped is reached_ensure
        assert events == [
            ("current", True), ("policy", "skills.context.list.local"),
            ("current", True), ("ensure",),
        ]
