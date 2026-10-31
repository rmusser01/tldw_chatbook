"""Native Console acceptance, real tool dispatch and finite review wake turns."""

import asyncio
import subprocess
from collections import defaultdict
from copy import deepcopy

import pytest

from Tests.Chat.test_console_agent_swap import _controller, _fence
from tldw_chatbook import config
from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Agents.run_context import current_run_actor
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture(autouse=True)
def scripted_review_model_capacity(monkeypatch):
    """The scripted external server advertises its actual finite context size."""
    from tldw_chatbook.Utils import token_counter

    original = token_counter.resolve_context_window
    monkeypatch.setattr(
        token_counter,
        "resolve_context_window",
        lambda provider, model, **kwargs: (
            token_counter.ContextWindowResolution(32768, "scripted review server", True)
            if model == "test-model"
            else original(provider, model, **kwargs)
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reviewers,verify,expected",
    [(2, False, [2]), (2, True, [2, 1]), (3, False, [2, 1]), (3, True, [2, 2])],
)
async def test_default_two_child_schedule(
    tmp_path,
    reviewers,
    verify,
    expected,
    prior_spawns=0,
    permission_decision="approve_session",
    later_acceptance=None,
    revoke_read=False,
    destination_decision="proceed",
    wake_mode=None,
    initial_text=None,
    retained_work=None,
    retained_permission_denied=False,
    after_schedule=None,
    journey=None,
    review_skill=None,
    physical_transport=None,
):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Review Test",
            "-c",
            "user.email=review@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "baseline",
        ],
        check=True,
        capture_output=True,
    )
    (root / "main.py").write_text(journey.source if journey else "value = 1\n")
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspace.db", client_id="review")
    )
    registry.create_workspace(
        workspace_id="review", name="Review", assistant_defaults=None
    )
    binding = registry.add_folder_binding("review", root)
    controller, store, db = (
        _controller(tmp_path, [])
        if physical_transport is None
        else physical_transport.create_console(tmp_path)
    )
    if review_skill is not None:
        from Tests.Skills.test_code_review_skill import _import_review_skill

        skills, trust, _ = await _import_review_skill(tmp_path / "review-skill")
        if review_skill != "untrusted":
            reviewed = trust.capture_review("multi-model-review")
            trust.trust_reviewed_snapshot(reviewed["review_id"])
        from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

        controller._skills_service = SkillsScopeService(local_service=skills)

    from tldw_chatbook.MCP.permission_store import MCPPermissionStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    permissions = MCPPermissionStore(tmp_path / "permissions.json")
    mcp = UnifiedMCPControlPlaneService(
        target_store=None, context_store=None, local_service=None, server_service=None
    )
    mcp._permission_store = permissions
    from Tests.UI.consolidated_css import ConsolidatedCSSApp

    app = ConsolidatedCSSApp()
    app.workspace_registry_service = registry
    if review_skill is not None:
        app.skills_scope_service = controller._skills_service
    app.unified_mcp_service = mcp
    mounted = app.run_test(size=(100, 30))
    await mounted.__aenter__()
    # Stands in for ChatScreen's attach: the app plus the report presenter.
    from tldw_chatbook.UI.Console_Modules import wiring

    wiring.attach_review_presenter(controller, app)
    controller.request_mcp_approvals = lambda calls, **kw: (
        {call.llm_name: permission_decision for call in calls}
        if permission_decision
        else {}
    )
    store.persistence.workspace_registry = registry
    session = store.create_session(workspace_id="review")
    destination_notices = []

    def confirm_destination(notice):
        destination_notices.append(notice.destination_label)
        return (
            destination_decision
            if "ollama" in notice.destination_label.lower()
            else "proceed"
        )

    controller._confirm_project_instruction_dispatch = confirm_destination
    ids = [
        db.create_agent_definition(
            AgentDefinition(
                name=f"review-{i}",
                instructions="Review independently",
                provider=(
                    physical_transport.reviewers[i][0]
                    if physical_transport
                    else "ollama"
                    if destination_decision != "proceed"
                    else "llama_cpp"
                ),
                model=physical_transport.reviewers[i][1]
                if physical_transport
                else "test-model",
                params=(
                    ("max_tokens", 4096),
                    *((("reasoning_effort", "none"),) if i == 0 else ()),
                )
                if physical_transport
                else (),
            )
        )
        for i in range(reviewers)
    ]
    verifier_id = db.create_agent_definition(
        AgentDefinition(
            name="verify",
            instructions="Verify independently",
            provider=physical_transport.verifier[0]
            if physical_transport
            else "llama_cpp",
            model=physical_transport.verifier[1]
            if physical_transport
            else "test-model",
            params=(("max_tokens", 4096),) if physical_transport else (),
        )
    )
    old = config.get_cli_setting("agents", "code_review", {})
    config.save_setting_to_cli_config(
        "agents",
        "code_review",
        {
            "reviewers": [{"preset_id": x} for x in ids],
            "verifier_preset_id": verifier_id,
        },
    )
    configured_team = deepcopy(config.get_cli_setting("agents", "code_review", {}))
    counts = defaultdict(int)
    stages = defaultdict(int)
    seen_children = set()
    launched_slots = []
    permission_refusals = []
    retained_started = False
    preparation = {}
    errors = []
    presented = []
    revoked_results = []
    gateway = controller.provider_gateway
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    prepared_gateway = ConsoleProviderGateway(config_provider=dict, environ={})
    if physical_transport is None:
        gateway.prepare_chat_request = prepared_gateway.prepare_chat_request

        async def scripted_window(resolution):
            from tldw_chatbook.Utils.token_counter import resolve_context_window

            return resolve_context_window(resolution.provider, resolution.model)

        gateway.resolve_context_window = scripted_window
    wake_limits_patch = None
    if wake_mode == "exhausted":
        from dataclasses import replace
        from unittest.mock import patch

        from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

        exhausted = replace(AutomaticWorkLimits.from_settings(), generations=0)
        native_boundary = controller.code_review._on_boundary

        def exhaust_at_boundary(receipt):
            nonlocal wake_limits_patch
            if wake_limits_patch is None:
                wake_limits_patch = patch.object(
                    AutomaticWorkLimits, "from_settings", return_value=exhausted
                )
                wake_limits_patch.start()
            native_boundary(receipt)

        controller.code_review._on_boundary = exhaust_at_boundary
    if wake_mode == "manual_priority":
        native_boundary = controller.code_review._on_boundary
        loop = asyncio.get_running_loop()

        def boundary_twice(receipt):
            controller.wake_user_priority_probe = lambda session_id: True
            native_boundary(receipt)
            native_boundary(receipt)

            def release():
                assert sum(counts.values()) == 2
                controller.wake_user_priority_probe = lambda session_id: False
                controller.fleet_wake.retry_soon()

            loop.call_soon_threadsafe(lambda: loop.call_later(0.1, release))

        controller.code_review._on_boundary = boundary_twice

    async def stream(resolution, messages, **kwargs):
        from tldw_chatbook.Chat.console_prepared_request import (
            PreparedProviderRequest,
            thaw_json,
        )

        if journey is not None and physical_transport is None:
            journey.observe_request(current_run_actor(), messages)
        if isinstance(messages, PreparedProviderRequest):
            messages = [thaw_json(row) for row in messages.messages]
        actor = current_run_actor()
        n = stages[actor.run_id]
        stages[actor.run_id] += 1
        coordinator = getattr(controller, "code_review", None)
        if actor.kind == "subagent":
            if actor.run_id not in seen_children:
                seen_children.add(actor.run_id)
                counts[actor.parent_run_id] += 1
            if (
                not preparation
                or actor.run_id
                not in coordinator._reviews[preparation["id"]].store._tab.runs
            ):
                yield "Ordinary task complete."
                return
            review = coordinator._reviews[preparation["id"]]
            if journey is not None:
                yield journey.child(actor, n, review, preparation["id"], messages)
                return
            with review.store.lease(preparation["id"]) as context:
                child_binding = review.store._tab.runs[actor.run_id]
                if n == 0:
                    launched_slots.append(
                        (child_binding.slot_id, child_binding.attempt_id)
                    )
                if child_binding.role == "verifier":
                    presented.append(
                        all(
                            (s.slot_id, context.active_attempts[s.slot_id].attempt_id)
                            in context.presented_attempts
                            for s in review.slots(context)
                            if s.role == "reviewer"
                        )
                    )
                    yield "Verification stage reached."
                    return
                refs = list(context.sources)
            if revoke_read and n == 0:
                mcp.set_tool_state("local:__local__", "fs_read", "deny")
                yield _fence(
                    "code_review_read",
                    {
                        "review_id": preparation["id"],
                        "view": "source",
                        "document_ref": refs[0],
                    },
                )
                return
            if revoke_read and n == 1:
                revoked_results.append(str(messages[-1]))
            yield (
                _fence(
                    "code_review_submit",
                    {
                        "review_id": preparation["id"],
                        "kind": "reviewer_complete",
                        "records": [
                            {
                                "schema_version": 1,
                                "inspected_refs": refs,
                                "gaps": [],
                                "empty_findings": True,
                            }
                        ],
                        "expected_revision": 0,
                        "submission_id": "complete",
                    },
                )
                if n == int(revoke_read)
                else "Reviewer complete."
            )
            return
        if coordinator is None:
            errors.append("native review owner missing")
            yield "Native review owner missing."
            return
        first_turn = not preparation
        if retained_started and retained_permission_denied:
            service = coordinator._reviews[preparation["id"]]
            if n == 0 and retained_permission_denied == "missing":
                service.source_permission = None
            if n == 1:
                with service.store.lease(preparation["id"]) as context:
                    ref = next(iter(context.sources))
                yield _fence(
                    "code_review_read",
                    {
                        "review_id": preparation["id"],
                        "view": "source",
                        "document_ref": ref,
                    },
                )
                return
            if n in {2, 3}:
                permission_refusals.append(str(messages[-1]))
            if n == 3:
                yield "Current source permission refused retained work."
                return
        ordinary = prior_spawns if first_turn else 0
        if first_turn and n < ordinary:
            yield _fence("spawn_subagent", {"task": "Return ordinary result."})
            return
        if n == ordinary:
            yield _fence(
                "load_tools",
                {
                    "ids": [
                        "code_review:code_review_guide",
                        "code_review:code_review_prepare",
                        "code_review:code_review_read",
                    ]
                },
            )
            return
        if journey is not None:
            if n == ordinary + 1:
                yield _fence(
                    "code_review_guide", {"stage": "review" if first_turn else "verify"}
                )
                return
            n -= 1
        if not preparation:
            if n == ordinary + 1:
                yield _fence(
                    "code_review_prepare",
                    {
                        "target": {
                            "kind": "working_tree",
                            "binding_ref": binding.binding_id,
                            "staged": False,
                            "unstaged": True,
                            "untracked_paths": ["main.py"],
                        }
                    },
                )
                return
            if not coordinator._reviews:
                if (
                    prior_spawns == 2
                    or permission_decision != "approve_session"
                    or destination_decision != "proceed"
                ):
                    errors.append(str(messages[-1]))
                    yield "Preparation refused."
                    return
                errors.append("native prepare failed: " + str(messages[-1]))
                yield "Preparation failed."
                return
            preparation["id"] = next(iter(coordinator._reviews))
        service = coordinator._reviews[preparation["id"]]
        with service.store.lease(preparation["id"]) as context:
            pending = [
                s.slot_id
                for s in service.slots(context)
                if s.slot_id not in context.launch_claims
                and (s.role == "reviewer" or service.verification_grant is not None)
            ]
            live = [
                c.run_id
                for c in context.launch_claims.values()
                if c.run_id
                and not context.streams[
                    (c.slot.slot_id, c.binding.attempt_id)
                ].terminal_status
            ]
        if live:
            fleet = controller._agent_bridge._fleet_services[
                session.persisted_conversation_id
            ]._fleet
            handles = [
                handle.handle_id for handle in fleet.snapshot() if handle.run_id in live
            ]
            assert handles and len(handles) == len(live)
            yield _fence("wait_agents", {"ids": handles})
        elif pending and counts[actor.run_id] < 2:
            yield _fence(
                "spawn_subagent",
                {"review_id": preparation["id"], "review_slot_id": pending[0]},
            )
        else:
            yield "Review turn complete."

    if physical_transport is None:
        gateway.stream_chat = stream
    else:
        await physical_transport.install_script(controller, session, stream)
    try:
        result = await controller.submit_draft(
            initial_text
            or ("Review my changes and verify" if verify else "Review my changes"),
            session_id=session.id,
        )
        if review_skill == "body_only":
            assert result.accepted
            assert not counts and not controller.code_review._reviews
            assert not controller.code_review.issuer._grants
            return
        if review_skill == "untrusted":
            assert not result.accepted
            assert not counts and not controller.code_review._reviews
            assert not controller.code_review.issuer._grants
            assert "trust" in result.visible_copy.lower()
            return
        for _ in range(200):
            if (
                sum(counts.values()) >= sum(expected)
                and not controller.fleet_wake.delivering_session_ids()
            ):
                break
            await asyncio.sleep(0.03)
        assert result.accepted, result
        if (
            prior_spawns == 2
            or permission_decision != "approve_session"
            or destination_decision != "proceed"
        ):
            assert errors
            assert not controller.code_review._reviews
            assert not controller.code_review.issuer._grants
            assert sum(counts.values()) == prior_spawns
            if destination_decision != "proceed":
                assert any("ollama" in label.lower() for label in destination_notices)
                assert "destination_consent_required" in errors[-1]
            return
        assert not errors, (errors, getattr(controller.code_review, "diagnostic", ""))
        assert list(counts.values()) == expected, (
            getattr(controller.code_review, "diagnostic", ""),
            [(m.role, m.content) for m in store.messages_for_session(session.id)],
            [
                dict(r)
                for r in db._held_connection().execute(
                    "SELECT payload FROM agent_run_steps WHERE payload LIKE '%error%'"
                )
            ],
        )
        if later_acceptance is not None:
            assert (
                next(iter(controller.code_review._reviews.values())).verification_grant
                is None
            )
            assert not presented
            assert session.id in controller.code_review._offers
            later_result = await controller.submit_draft(
                later_acceptance, session_id=session.id
            )
            assert later_result.accepted
            assert sum(counts.values()) == sum(expected) + 1, (
                counts,
                getattr(controller.code_review, "diagnostic", ""),
            )
        if retained_work is not None:
            retained = controller.code_review._reviews[preparation["id"]]
            with retained.store.lease(preparation["id"]) as context:
                original_preparation = context.preparation
                original_source = context.source_state
                original_sources = dict(context.sources)
                original_receipts = {
                    key: stream.terminal_receipt
                    for key, stream in context.streams.items()
                }
                if wake_mode in {"disabled", "exhausted"}:
                    assert context.continuation_status.startswith(
                        "continuation_blocked:"
                    )
            old_chain = retained.native_chain_id
            old_budget = db.automatic_work.snapshot(old_chain)
            old_links = [
                tuple(row)
                for row in db._held_connection().execute(
                    "SELECT * FROM automatic_review_boundary_claims"
                )
            ]
            old_boundaries = [
                entry[0] for entry in controller.code_review._boundaries.values()
            ]
            previous_slots = list(launched_slots)
            if retained_permission_denied is True:
                mcp.set_tool_state("local:__local__", "fs_read", "deny")
            retained_started = True
            resumed = await controller.submit_draft(
                retained_work, session_id=session.id
            )
            assert resumed.accepted
            assert sum(counts.values()) == sum(expected) + int(
                not retained_permission_denied
            ), (
                counts,
                getattr(controller.code_review, "diagnostic", ""),
            )
            assert len(controller.code_review._reviews) == 1
            with retained.store.lease(preparation["id"]) as context:
                assert context.preparation is original_preparation
                assert context.source_state is original_source
                assert context.sources == original_sources
                assert all(
                    context.streams[key].terminal_receipt == receipt
                    for key, receipt in original_receipts.items()
                )
                assert context.continuation_status is None
            if retained_permission_denied:
                assert launched_slots == previous_slots
                assert len(permission_refusals) == 2
                assert all(
                    "source_permission_" in refusal for refusal in permission_refusals
                )
            else:
                assert launched_slots[:-1] == previous_slots
                expected_slot = "R1" if retained_work.startswith("Retry") else "R3"
                assert launched_slots[-1][0] == expected_slot
                assert launched_slots[-1] not in previous_slots
            assert not controller.fleet_wake._pending_reviews.get(
                session.persisted_conversation_id
            )
            for receipt in old_boundaries:
                controller.fleet_wake.on_review_boundary(receipt)
                controller.code_review.mark_continuation_blocked(receipt, "stale")
            with retained.store.lease(preparation["id"]) as context:
                assert context.continuation_status is None
            assert all(
                not controller.code_review.validate_boundary(receipt)
                for receipt in old_boundaries
            )
            current_budget = db.automatic_work.snapshot(old_chain)
            assert current_budget.used == old_budget.used
            assert current_budget.reserved == old_budget.reserved
            assert current_budget.status == old_budget.status
            assert [
                tuple(row)
                for row in db._held_connection().execute(
                    "SELECT * FROM automatic_review_boundary_claims"
                )
            ] == old_links
        if revoke_read:
            assert revoked_results and all(
                "source_permission_deny" in result for result in revoked_results
            )
        assert (
            journey is not None
            or all(presented)
            and (bool(presented) == (verify or later_acceptance is not None))
        )
        rows = (
            db._held_connection()
            .execute("SELECT DISTINCT work_chain_id FROM agent_runs")
            .fetchall()
        )
        assert len(rows) == (
            2 if later_acceptance is not None or retained_work is not None else 1
        ) and all(row[0] for row in rows)
        if wake_mode == "manual_priority":
            attempts = (
                db._held_connection()
                .execute("SELECT state FROM automatic_wake_attempts")
                .fetchall()
            )
            assert [row[0] for row in attempts] == ["completed"]
        if wake_mode in {"disabled", "exhausted"} and retained_work is None:
            with next(iter(controller.code_review._reviews.values())).store.lease(
                preparation["id"]
            ) as context:
                assert context.continuation_status.startswith("continuation_blocked:")
            if wake_mode == "exhausted":
                assert (
                    not db._held_connection()
                    .execute("SELECT * FROM automatic_review_boundary_claims")
                    .fetchall()
                )
                assert controller.fleet_wake.pause_reason(
                    session.persisted_conversation_id
                )

        assert config.get_cli_setting("agents", "code_review", {}) == configured_team
        if after_schedule is not None:
            await after_schedule(controller, session, preparation["id"])

        transcript = store.messages_for_session(session.id)
        assert any(
            "Original report" in m.content and "Review reports" in m.content
            for m in transcript
        )
    finally:
        if wake_limits_patch is not None:
            wake_limits_patch.stop()
        config.save_setting_to_cli_config("agents", "code_review", old)
        await controller.shutdown()
        await prepared_gateway.aclose()
        if physical_transport is not None:
            await physical_transport.close()
        await mounted.__aexit__(None, None, None)
        registry.db.close()
        with store.persistence.db.quiesce_connections(timeout_seconds=2.0):
            assert store.persistence.db.registered_connection_count() == 0
        db.close()
        from tldw_chatbook.Backup_Recovery import storage_admission

        with storage_admission._lock:
            assert not registry.db._maintenance_participant.connections
            assert not any(
                lease.resource_path == registry.db.db_path
                for lease in storage_admission._live_leases
            )


@pytest.mark.asyncio
async def test_one_prior_spawn_uses_remaining_native_turn_capacity(tmp_path):
    await test_default_two_child_schedule(tmp_path, 2, False, [2, 1], prior_spawns=1)


@pytest.mark.asyncio
async def test_two_prior_spawns_refuse_before_source_capture(tmp_path):
    await test_default_two_child_schedule(tmp_path, 2, False, [2], prior_spawns=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["deny", None])
async def test_native_source_permission_refuses_without_capture_or_child(
    tmp_path, decision
):
    await test_default_two_child_schedule(
        tmp_path, 2, False, [], permission_decision=decision
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("acceptance", ["Verify this review", "yes"])
async def test_later_native_acceptance_uses_fresh_primary_capacity(
    tmp_path, acceptance
):
    await test_default_two_child_schedule(
        tmp_path, 2, False, [2], later_acceptance=acceptance
    )


@pytest.mark.asyncio
async def test_current_native_source_permission_revocation_blocks_captured_read(
    tmp_path,
):
    await test_default_two_child_schedule(tmp_path, 1, False, [1], revoke_read=True)


@pytest.mark.asyncio
async def test_duplicate_native_boundary_obeys_manual_priority_and_claims_once(
    tmp_path,
):
    await test_default_two_child_schedule(
        tmp_path, 3, False, [2, 1], wake_mode="manual_priority"
    )


@pytest.mark.asyncio
async def test_disabled_wake_retains_review_and_shows_blocked_status(tmp_path):
    from unittest.mock import patch

    with patch(
        "tldw_chatbook.Chat.console_fleet_wake.autowake_enabled", return_value=False
    ):
        await test_default_two_child_schedule(
            tmp_path, 3, False, [2], wake_mode="disabled"
        )


@pytest.mark.asyncio
async def test_exhausted_wake_retains_review_and_spends_no_generation(tmp_path):
    await test_default_two_child_schedule(
        tmp_path, 3, False, [2], wake_mode="exhausted"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["cancel", None])
async def test_native_resolved_reviewer_destination_requires_existing_consent_owner(
    tmp_path, decision
):
    await test_default_two_child_schedule(
        tmp_path, 1, False, [], destination_decision=decision
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "instruction",
    [
        "Review my changes and verify them only after I approve.",
        "Review my changes and verify them, but do not automatically verify.",
    ],
)
async def test_qualified_verification_waits_for_later_native_acceptance(
    tmp_path, instruction
):
    await test_default_two_child_schedule(
        tmp_path,
        2,
        False,
        [2],
        initial_text=instruction,
        later_acceptance="Verify this review",
    )


@pytest.mark.asyncio
async def test_native_reviewer_retry_selects_actual_slot_and_preserves_original(
    tmp_path,
):
    await test_default_two_child_schedule(
        tmp_path, 2, False, [2], retained_work="Retry reviewer R1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", ["disabled", "exhausted"])
async def test_native_manual_resume_only_launches_retained_missing_reviewer(
    tmp_path, blocked
):
    from contextlib import nullcontext
    from unittest.mock import patch

    gate = (
        patch(
            "tldw_chatbook.Chat.console_fleet_wake.autowake_enabled", return_value=False
        )
        if blocked == "disabled"
        else nullcontext()
    )
    with gate:
        await test_default_two_child_schedule(
            tmp_path, 3, False, [2], wake_mode=blocked, retained_work="Resume review"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [True, "missing"], ids=["revoked", "missing"])
async def test_native_manual_resume_respects_current_source_permission(
    tmp_path, failure
):
    from unittest.mock import patch

    with patch(
        "tldw_chatbook.Chat.console_fleet_wake.autowake_enabled", return_value=False
    ):
        await test_default_two_child_schedule(
            tmp_path,
            3,
            False,
            [2],
            wake_mode="disabled",
            retained_work="Resume review",
            retained_permission_denied=failure,
        )


@pytest.mark.asyncio
async def test_retained_source_guard_runs_on_owner_and_worker_threads(
    tmp_path, monkeypatch
):
    import threading

    from tldw_chatbook.Chat import console_chat_controller

    owner_thread = threading.get_ident()
    dispatch_threads = set()
    from textual.app import App

    native_dispatch = App.call_from_thread

    def dispatch(app, callback, *args, **kwargs):
        dispatch_threads.add(threading.get_ident())
        return native_dispatch(app, callback, *args, **kwargs)

    guard_threads = set()
    native_guard = console_chat_controller.project_instruction_authority_is_current

    def guard(**kwargs):
        result = native_guard(**kwargs)
        guard_threads.add((threading.get_ident(), result))
        return result

    monkeypatch.setattr(App, "call_from_thread", dispatch)
    monkeypatch.setattr(
        console_chat_controller, "project_instruction_authority_is_current", guard
    )
    await test_default_two_child_schedule(tmp_path, 2, True, [2, 1])
    assert guard_threads == {(owner_thread, True)}
    assert dispatch_threads and owner_thread not in dispatch_threads


@pytest.mark.asyncio
async def test_mounted_stop_wins_while_native_reader_waits_for_ui_guard(tmp_path):
    import asyncio
    from unittest.mock import patch

    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    async def interleave(controller, session, review_id):
        service = controller.code_review._reviews[review_id]
        assert any(
            entry[0].review_id == review_id
            for entry in controller.code_review._boundaries.values()
        )
        admitted = service.source_authority.admitted
        guard = admitted.guard
        observed = []
        first = True

        def stop_before_guard():
            # A timeout makes the old deadlock deterministic without hanging Pilot.
            acquired = service.store._tab.lock.acquire(timeout=0.2)
            observed.append(acquired)
            if acquired:
                service.store._tab.lock.release()
                controller._signal_stop(session_id=session.id)

        def pending_guard(write):
            nonlocal first
            if first:
                first = False
                controller.app.call_from_thread(stop_before_guard)
            return guard(write)

        object.__setattr__(admitted, "guard", pending_guard)
        try:
            with pytest.raises(ReviewRefused):
                await asyncio.to_thread(service.native_report_page, review_id, "R1")
        finally:
            object.__setattr__(admitted, "guard", guard)
        assert observed == [True], (
            "Stop cannot acquire publication lock while native reader waits on UI"
        )
        assert session.id in controller.code_review._fenced
        with service.store.lease(review_id) as context:
            assert context.continuation_status == "continuation_blocked:session_fenced"

    with patch(
        "tldw_chatbook.Chat.console_fleet_wake.autowake_enabled", return_value=False
    ):
        await test_default_two_child_schedule(
            tmp_path, 3, False, [2], wake_mode="disabled", after_schedule=interleave
        )
