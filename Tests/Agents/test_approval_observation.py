"""Owner observations cannot control permissions or invent successful writes."""

import threading
import pytest
from Tests.Agents.test_mcp_tool_provider import running_loop  # noqa: F401

pytestmark = pytest.mark.bootstrap_profile


def test_observer_failure_does_not_change_verdict_or_release():
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        approval_observation_scope,
        publish_grant_application,
    )

    released = threading.Event()

    def broken(_event):
        raise RuntimeError("synthetic observer failure")

    with approval_observation_scope(
        ApprovalObservationIdentity("s", "run", "r", 1, "c"), broken
    ):
        publish_grant_application("applied", actual_scope="approve_session")
        released.set()
    assert released.is_set()


def test_service_session_grant_is_reported_only_after_real_cache_write():
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        approval_observation_scope,
    )
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    service = object.__new__(UnifiedMCPControlPlaneService)
    service._session_approvals = set()
    service._session_approvals_lock = threading.Lock()
    seen = []

    def observe(event):
        assert service.is_session_approved("server", "tool", profile_id="Writer")
        seen.append(event)

    with approval_observation_scope(
        ApprovalObservationIdentity("s", "run", "r", 1, "c"), observe
    ):
        assert (
            service.approve_for_session("server", "tool", profile_id="Writer") is None
        )
    assert [(event.outcome, event.actual_scope) for event in seen] == [
        ("applied", "approve_session")
    ]


def test_grant_event_rejects_content_bearing_error_codes():
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservation,
        ApprovalObservationIdentity,
    )

    with pytest.raises(ValueError):
        ApprovalObservation(
            ApprovalObservationIdentity("s", "run", "r", 1),
            "grant",
            "failed",
            error_code="secret/path",
        )


@pytest.mark.parametrize("armed", [True, False])
def test_raw_owner_reports_disarm_noop_without_inventing_grant(armed):
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        approval_observation_scope,
    )

    runtime = RawCliRuntime(lambda: True)
    runtime.arm()
    if not armed:
        runtime.disarm()
    seen = []
    with approval_observation_scope(
        ApprovalObservationIdentity("s", "run", "r", 1, "c"), seen.append
    ):
        assert runtime.grant_model_session("s") is None
    assert runtime.model_session_granted("s") is armed
    assert seen[0].outcome == ("applied" if armed else "not_applied")
    assert seen[0].actual_scope == ("raw_shell_session" if armed else None)


def test_missing_permission_store_and_failure_are_owner_facts(monkeypatch):
    from types import SimpleNamespace
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        approval_observation_scope,
    )

    service = object.__new__(UnifiedMCPControlPlaneService)
    seen = []
    monkeypatch.setattr(
        UnifiedMCPControlPlaneService, "permission_store", property(lambda self: None)
    )
    with approval_observation_scope(
        ApprovalObservationIdentity("s", "run", "r", 1, "c"), seen.append
    ):
        assert service.set_tool_state("agent:builtin", "read", "allow") is None
    assert seen[-1].outcome == "not_applied"

    def failed(*args, **kwargs):
        raise OSError("private diagnostic must not escape observation")

    monkeypatch.setattr(
        UnifiedMCPControlPlaneService,
        "permission_store",
        property(lambda self: SimpleNamespace(set_tool_state=failed)),
    )
    with approval_observation_scope(
        ApprovalObservationIdentity("s", "run", "r", 1, "c"), seen.append
    ):
        with pytest.raises(OSError):
            service.set_tool_state("agent:builtin", "read", "allow")
    assert seen[-1].outcome == "failed"
    assert seen[-1].error_code == "writer_failed"
    assert "private diagnostic" not in repr(seen)


def test_observation_metadata_never_makes_none_stamp_present():
    from tldw_chatbook.Agents.approval_provenance import (
        ApprovalDecisions,
        approval_stamp,
    )

    answers = ApprovalDecisions({"call": None, "tool": "approve_session"})
    answers.observation_contexts = {"call": object()}
    copied = ApprovalDecisions(answers)
    assert dict(copied) == {"call": None, "tool": "approve_session"}
    assert approval_stamp(copied["call"]).approval_decision is None
    assert copied.observation_contexts == answers.observation_contexts


def test_real_persistent_owner_reports_equivalent_rule_with_actual_scope(tmp_path):
    from Tests.MCP.test_control_plane_permissions import _service, _tool
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        approval_observation_scope,
    )

    service, _ = _service(tmp_path)
    tool = _tool()
    seen = []
    with approval_observation_scope(
        ApprovalObservationIdentity("s", "run", "r", 1, "c"), seen.append
    ):
        service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
        service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
        service.add_tool_arg_rule(
            tool.server_key, tool.name, args={"query": "sample"}, tool=tool
        )
    assert [event.actual_scope for event in seen] == [
        "always_allow",
        "always_allow",
        "allow_matching",
    ]
    assert all(event.outcome == "applied" for event in seen)
    assert service.arg_rule_allows_call(tool, {"query": "sample"})


def test_provider_scope_survives_worker_without_copying_policy_context():
    from types import SimpleNamespace
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        ApprovalObservationContext,
        remember_approval_contexts,
        provider_approval_contexts,
        forget_approval_contexts,
    )
    from tldw_chatbook.Agents.approval_provenance import ApprovalDecisions
    from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id

    owner = SimpleNamespace()
    decisions = ApprovalDecisions({"c": "approve_session"})
    context = ApprovalObservationContext(
        ApprovalObservationIdentity("s", "run", "r", 1, "c"), lambda event: None
    )
    decisions.observation_contexts = {"c": context}
    decisions.observation_aliases = {"tool": ("c",), "ambiguous": ("c", "other")}
    remember_approval_contexts(owner, "run", decisions)
    seen = []

    def worker():
        with use_run_id("run"), use_tool_call_id("c"):
            seen.append(provider_approval_contexts(owner, "tool"))
        with use_run_id("run"):
            seen.append(provider_approval_contexts(owner, "ambiguous"))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen == [(context,), ()]
    forget_approval_contexts(owner, "run")
    assert owner._approval_observation_runs == {}


def test_observation_metadata_preserves_mcp_execution_trace_and_missing_writer_evidence(
    running_loop,  # noqa: F811 - shared pytest fixture
):
    from Tests.Agents.test_mcp_tool_provider import (
        FakeMCPService,
        _catalog_record,
        _tool_dict,
        _compose,
    )
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
    from tldw_chatbook.Agents.approval_provenance import ApprovalDecisions
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        ApprovalObservationContext,
    )
    from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id

    results = []
    seen = []
    for observed in (False, True):
        service = FakeMCPService(
            catalog_records=[_catalog_record("srv", [_tool_dict("read")])]
        )
        provider = MCPToolProvider(service=service, main_loop=running_loop)
        _compose(provider)
        tool_id = provider.list_catalog()[0].id
        decisions = ApprovalDecisions({tool_id: "approve_session"})
        if observed:
            decisions.observation_contexts = {
                "c": ApprovalObservationContext(
                    ApprovalObservationIdentity("s", "run", "r", 1, "c"), seen.append
                )
            }
            decisions.observation_aliases = {"read": ("c",)}
        provider.apply_batch_decisions("run", decisions)
        with use_run_id("run"), use_tool_call_id("c"):
            result = provider.invoke(tool_id, {})
        results.append(
            (
                result,
                service.execute_calls,
                service.record_tool_decision_calls,
                service.session_approvals,
            )
        )
    assert results[0] == results[1]
    assert results[0][0].ok
    # Fake service's void-returning grant does not provide an owner observation.
    assert seen == []


@pytest.mark.parametrize("kind", ["local", "mcp", "virtual"])
def test_nested_stamp_scope_restores_observation_owner_without_altering_stamps(kind):
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
    from tldw_chatbook.Agents.approval_provenance import ApprovalDecisions
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        ApprovalObservationContext,
        remember_approval_contexts,
        provider_approval_contexts,
    )
    from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id

    from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider

    owner = object.__new__(
        {
            "local": LocalToolProvider,
            "mcp": MCPToolProvider,
            "virtual": VirtualCliProvider,
        }[kind]
    )
    owner._stamps_lock = threading.Lock()
    owner._stamps = {}
    owner._decisions_lock = threading.Lock()
    owner._stamped_decisions = {}

    def answers(round_id):
        values = ApprovalDecisions({"c": "approve_once"})
        values.observation_contexts = {
            "c": ApprovalObservationContext(
                ApprovalObservationIdentity("s", "run", round_id, 1, "c"),
                lambda event: None,
            )
        }
        return values

    parent = answers("parent")
    child = answers("child")
    remember_approval_contexts(owner, "run", parent)
    with use_run_id("run"), use_tool_call_id("c"):
        with owner.stamp_scope("run"):
            if kind in {"local", "virtual"}:
                assert provider_approval_contexts(owner, "tool") == ()
            remember_approval_contexts(owner, "run", child)
            assert (
                provider_approval_contexts(owner, "tool")[0].identity.round_id
                == "child"
            )
        assert (
            provider_approval_contexts(owner, "tool")[0].identity.round_id == "parent"
        )


def test_hook_definition_refusal_audit_is_equal_with_observation_enabled(running_loop):  # noqa: F811
    from Tests.Agents.test_mcp_tool_provider import (
        FakeMCPService,
        _catalog_record,
        _tool_dict,
        _compose,
    )
    import tldw_chatbook.Agents.mcp_tool_provider as module
    from tldw_chatbook.Agents.approval_provenance import ApprovalDecisions
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        ApprovalObservationContext,
        remember_approval_contexts,
    )
    from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id

    audits = []
    results = []
    observed = []
    for enabled in (False, True):
        service = FakeMCPService(
            catalog_records=[_catalog_record("srv", [_tool_dict("read")])]
        )
        calls = []
        service.record_tool_decision = lambda *args, **kwargs: calls.append(
            (args, kwargs)
        )
        provider = module.MCPToolProvider(service=service, main_loop=running_loop)
        _compose(provider)

        def stale(tool):
            raise PermissionError("definition changed")

        provider._check_current_definition = stale
        decisions = ApprovalDecisions({"c": "approve_session"})
        if enabled:
            decisions.observation_contexts = {
                "c": ApprovalObservationContext(
                    ApprovalObservationIdentity("s", "run", "r", 1, "c"),
                    observed.append,
                )
            }
        remember_approval_contexts(provider, "run", decisions)
        with pytest.MonkeyPatch.context() as patch:
            import contextlib
            import time

            policy = module.MCPInvocationPolicy(
                current=lambda: True,
                deadline=time.monotonic() + 30,
                cancel_event=threading.Event(),
                allow_approval=True,
                wait_scope=lambda kind: contextlib.nullcontext(),
            )
            patch.setattr(module, "current_mcp_invocation_policies", lambda: (policy,))
            with use_run_id("run"), use_tool_call_id("c"):
                results.append(provider.invoke(provider.list_catalog()[0].id, {}))
        audits.append(calls)
    assert results[0] == results[1] and not results[0].ok
    assert audits[0] == audits[1] and len(audits[0]) == 1
    assert audits[0][0][1]["error_category"] == "definition_changed"
    assert observed == []


def test_unmatched_exact_provider_call_cannot_borrow_unique_other_call():
    from types import SimpleNamespace
    from tldw_chatbook.Agents.approval_provenance import ApprovalDecisions
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        ApprovalObservationContext,
        remember_approval_contexts,
        provider_approval_contexts,
    )
    from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id

    owner = SimpleNamespace()
    answers = ApprovalDecisions({"a": "approve_session"})
    answers.observation_contexts = {
        "a": ApprovalObservationContext(
            ApprovalObservationIdentity("s", "run", "r", 1, "a"), lambda event: None
        )
    }
    answers.observation_aliases = {"read": ("a",)}
    remember_approval_contexts(owner, "run", answers)
    with use_run_id("run"), use_tool_call_id("b"):
        assert provider_approval_contexts(owner, "read") == ()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "choice", ["approve_session", "always_allow", "allow_matching"]
)
@pytest.mark.parametrize("writer", ["owner", "missing", "failed"])
def test_virtual_cli_observes_actual_grant_without_native_dispatch(
    tmp_path, direct, choice, writer
):
    from Tests.Agents.test_virtual_cli_provider import make_provider
    from Tests.MCP.test_control_plane_permissions import _service
    from tldw_chatbook.Agents.approval_provenance import ApprovalDecisions
    from tldw_chatbook.Agents.approval_observation import (
        ApprovalObservationIdentity,
        ApprovalObservationContext,
    )
    from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id

    service, _ = _service(tmp_path)
    seen = []
    args = {"command": "ls", "argv": []}

    def persist(hub, decision):
        if writer == "failed":
            raise OSError("synthetic failure")
        if decision == "approve_session":
            service.approve_for_session(hub.server_key, hub.name)
        else:
            service.set_tool_state(hub.server_key, hub.name, "allow", tool=hub)

    def matching(hub, arguments):
        if writer == "failed":
            raise OSError("synthetic failure")
        service.add_tool_arg_rule(
            hub.server_key, hub.name, args=dict(arguments), tool=hub
        )

    provider = make_provider(
        tmp_path,
        persist_approval=None if writer == "missing" else persist,
        persist_arg_rule=None if writer == "missing" else matching,
    )
    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall

    row = MCPPendingCall(
        "virtual_cli",
        "local:__virtual_cli__",
        "ls",
        "Virtual CLI",
        args,
        "ask",
        call_id="ls" if direct else "c",
    )
    provider.pending_gate_for = lambda call: row
    answers = ApprovalDecisions({("ls" if direct else "c"): choice})
    key = "ls" if direct else "c"
    answers.observation_contexts = {
        key: ApprovalObservationContext(
            ApprovalObservationIdentity("s", "run", "r", 1, key), seen.append
        )
    }
    answers.observation_aliases = {"ls": (key,)}
    answers.observation_legacy_keys = frozenset({key}) if direct else frozenset()
    if direct:
        provider._approval_callback = lambda pending: answers
    else:
        provider.apply_batch_decisions("run", answers, [row])
    with use_run_id("run"), use_tool_call_id("" if direct else "c"):
        verdict = provider._ask_verdict_detail(provider.hub_tool_for("ls"), "ls", args)
    assert verdict.decision == "allow" and verdict.approval_decision == "approved"
    assert (
        seen
        and seen[-1].outcome
        == {"owner": "applied", "missing": "not_applied", "failed": "failed"}[writer]
    )
    if writer == "owner":
        assert seen[-1].actual_scope == choice


def test_virtual_legacy_marker_uses_original_call_presence(tmp_path, monkeypatch):
    from Tests.Agents.test_virtual_cli_provider import make_provider
    from tldw_chatbook.Agents.agent_models import ToolCall

    provider = make_provider(tmp_path)
    # This tests only the captured key marker, without claiming native admission.
    monkeypatch.setattr(provider, "_validated_args", lambda args: ("ls", [], None))
    monkeypatch.setattr(provider, "_authority_is_valid", lambda authority: True)
    named = provider.pending_gate_for(ToolCall("virtual_cli", {"command": "ls"}, ""))
    exact = provider.pending_gate_for(ToolCall("virtual_cli", {"command": "ls"}, "ls"))
    assert named.call_id == exact.call_id == "ls"
    assert named.legacy_observation_key and not exact.legacy_observation_key
    assert named == exact  # The display marker cannot alter authority equality.
