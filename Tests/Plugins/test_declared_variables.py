"""Declared values through installed configuration and actual consuming hooks."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from Tests.hooks_v2_process_support import child_argv
from Tests.MCP.test_credential_bindings import MemoryCredentialBackend
from Tests.Plugins.test_native_components import install_components

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.fixture
async def variable_case(tmp_path, native_package, request):
    from Tests.private_profile import is_private_profile_child

    if getattr(
        request.function, "_private_profile_test", False
    ) and not is_private_profile_child(request):
        yield None
        return
    from Tests.Plugins.platform_support import require_managed_runtime

    require_managed_runtime()
    from tldw_chatbook.MCP.credential_bindings import CredentialBindingService
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService

    backend = MemoryCredentialBackend()
    protected = CredentialBindingService(backend)
    local = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "mcp.json"), credential_service=protected
    )
    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: SimpleNamespace(archived=False),
        marker_store_factory=lambda _: FilePluginMarkerStore(
            tmp_path / "variable-marker"
        ),
        accept_reduced_protection=True,
        mcp_mapping_owner=local,
    )
    await service.bootstrap("private test passphrase")
    package = native_package(
        extension={
            "version": 1,
            "hooks": "hooks.json",
            "variables": {
                "TOKEN": {"type": "string", "required": True, "secret": True},
                "COUNT": {
                    "type": "integer",
                    "required": False,
                    "secret": False,
                    "default": 4,
                },
                "UNUSED": {"type": "string", "required": True, "secret": True},
            },
        }
    )
    marker = tmp_path / "hook-ran"
    hook = {
        "id": "guard",
        "type": "command",
        "event": "PreToolUse",
        "required": True,
        "match": {"operation": ["file.write"]},
        "effects": ["deny"],
        "argv": child_argv(
            f"import os;from pathlib import Path;assert os.environ['COUNT']=='4';Path({str(marker)!r}).write_text(os.environ['TOKEN']);print('{{\"version\":2,\"decision\":\"pass\"}}')"
        ),
        "env": {"TOKEN": {"variable": "TOKEN"}, "COUNT": "${COUNT}"},
    }
    (package / "hooks.json").write_text(json.dumps({"version": 2, "hooks": [hook]}))
    try:
        yield SimpleNamespace(
            service=service,
            package=package,
            marker=marker,
            hook=hook,
            protected=protected,
            backend=backend,
            tmp_path=tmp_path,
        )
    finally:
        await service.aclose()


def hook_engine(case, snapshot):
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Plugins.hooks import NativeHooks

    native = NativeHooks(case.service, (snapshot,))
    return HookEngine(
        native.definitions,
        native.authority,
        HookBudgetOwner(),
        process_owner_for=native.process_owner,
        event_projector=native.project_event,
        host_environment=native.host_environment,
        command_launch=native.command_launch,
    )


async def capture(case, installation):
    return await case.service.capture_mcp_snapshot(
        installation, "workspace-a", "variables-run"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("required", [True, False])
async def test_installed_write_guard_keeps_nonmatch_and_uses_typed_secret_reference(
    variable_case,
    required,
):
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import (
        HookPreparationError,
        ToolHookRun,
    )
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.MCP.permission_store import EffectiveToolState
    from tldw_chatbook.Plugins.variables import handle

    case = variable_case
    case.hook["required"] = required
    (case.package / "hooks.json").write_text(
        json.dumps({"version": 2, "hooks": [case.hook]})
    )
    installed = await install_components(
        case.service, case.package, ("skill:review", "hook:guard")
    )
    snapshot = await capture(case, installed.installation_id)
    assert snapshot.selection == ("skill:review",)
    assert snapshot.unavailable_hooks == ("hook:guard",)
    maximum = case.service.capture_maximum("workspace-a")
    requirement = next(
        row
        for row in maximum["available_skills"]
        if row["plugin_component_id"] == "hook:guard"
    )
    assert (
        requirement["plugin_constraint_only"]
        and requirement["availability"] == "needs_configuration"
    )
    with pytest.raises(PermissionError):
        await case.service._call(
            lambda: case.service._admission.check(snapshot, "hook:guard")
        )
    admitted = await case.service.admit(maximum, "projected-run")
    assert await case.service.component_snapshots(
        dict(admitted, plugin_run_id="projected-run")
    )
    with pytest.raises(PermissionError):
        await case.service._call(
            lambda: case.service._checked(
                requirement["record_id"],
                next(
                    row["plugin_admission"]
                    for row in admitted["available_skills"]
                    if row["plugin_component_id"] == "hook:guard"
                ),
            )
        )
    root = case.tmp_path / "tool-root"
    root.mkdir()
    (root / "read.txt").write_text("read control")
    reviews = []
    provider = LocalToolProvider(
        workspace_root=root,
        resolve_state=lambda _: EffectiveToolState(
            state="ask", origin="global_default"
        ),
        approval_callback=lambda calls: reviews.extend(call.llm_name for call in calls)
        or {call.llm_name: "approve_once" for call in calls},
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)

    async def use(current, name, args):
        engine = hook_engine(case, current)
        run = ToolHookRun(
            engine,
            run_id="variables-run",
            session_id="session",
            turn_id="turn",
            resolve_definition=lambda call: registry.snapshot_for_hook(call.name),
        )

        def dispatch():
            prepared = run.prepare_call(ToolCall(name, args, name))
            run.validate_dispatch(prepared)
            return registry.invoke_by_name(
                name,
                prepared.args,
                expected_definition=registry.snapshot_for_hook(name),
            )

        try:
            return await asyncio.to_thread(dispatch)
        finally:
            await engine.close()

    with pytest.raises(HookPreparationError):
        await use(snapshot, "fs_write", {"path": "written.txt", "content": "written"})
    assert (
        not case.marker.exists() and not (root / "written.txt").exists() and not reviews
    )
    assert (await use(snapshot, "fs_read", {"path": "read.txt"})).ok
    assert reviews == ["fs_read"]
    metadata = await case.service.create_variable_reference(
        installed.installation_id, "TOKEN", "private ${OTHER}"
    )
    assert "private ${OTHER}" not in repr(metadata)
    review = await case.service.review_configuration(
        installed.installation_id, variables={"TOKEN": handle(metadata)}
    )
    assert "private ${OTHER}" not in repr(review)
    assert (await case.service.commit(review, review.operation_id)).committed
    ready = await capture(case, installed.installation_id)
    assert not ready.unavailable_hooks and "hook:guard" in ready.selection
    assert "private ${OTHER}" not in repr(ready)
    assert (
        await use(ready, "fs_write", {"path": "written.txt", "content": "written"})
    ).ok
    assert case.marker.read_text() == "private ${OTHER}"
    assert (root / "written.txt").read_text() == "written"
    assert reviews[-1] == "fs_write"
    await case.service.revoke_variable_reference(
        installed.installation_id, metadata["reference_id"]
    )
    with pytest.raises(PermissionError):
        await case.service._call(
            lambda: case.service._admission.check(ready, "hook:guard")
        )
    revoked = await capture(case, installed.installation_id)
    assert revoked.unavailable_hooks == ("hook:guard",)
    with pytest.raises(HookPreparationError):
        await use(revoked, "fs_write", {"path": "revoked.txt", "content": "never"})
    assert not (root / "revoked.txt").exists()


@pytest.mark.asyncio
async def test_required_initializer_and_dependency_only_unavailable_sibling(
    variable_case,
):
    from Tests.Agents.test_hooks_v2_execution import event

    case = variable_case
    case.hook.update(event="SessionStart", effects=[])
    case.hook.pop("match")
    (case.package / "hooks.json").write_text(
        json.dumps({"version": 2, "hooks": [case.hook]})
    )
    installed = await install_components(
        case.service, case.package, ("skill:review", "hook:guard")
    )
    snapshot = await capture(case, installed.installation_id)
    engine = hook_engine(case, snapshot)
    try:
        from tldw_chatbook.Agents.hooks_v2 import parse_event

        raw = {
            key: value
            for key, value in event().model_dump().items()
            if value is not None
        }
        raw.update(event="SessionStart", data={"reason": "startup"})
        outcome = await engine.fire_async(parse_event(raw))
        assert not outcome.succeeded and not outcome.allowed
        assert not engine.processes.records and not case.service.live_runs()
        assert not case.marker.exists()
    finally:
        await engine.close()
    # A separate actual installation has only a dependency requirement, so the
    # independent sibling remains executable without installing an event gate.
    import shutil

    package = case.tmp_path / "dependency-package"
    shutil.copytree(case.package, package)
    manifest = json.loads((package / "plugin.json").read_text())
    manifest["name"] = "dependency-example"
    manifest["extensions"]["io.github.rmusser01.chatbook"]["requires"] = {
        "skill:review": ["hook:guard"]
    }
    (package / "plugin.json").write_text(json.dumps(manifest))
    independent = package / "skills/independent"
    independent.mkdir()
    (independent / "SKILL.md").write_text(
        "---\nname: independent\ndescription: Independent sibling\n---\nIndependent body."
    )
    case.hook["required"] = False
    (package / "hooks.json").write_text(
        json.dumps({"version": 2, "hooks": [case.hook]})
    )
    other = await install_components(
        case.service, package, ("skill:review", "skill:independent", "hook:guard")
    )
    sibling = await capture(case, other.installation_id)
    assert sibling.selection == ("skill:independent",) and not sibling.unavailable_hooks
    await case.service._call(
        lambda: case.service._admission.check(sibling, "skill:independent")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["missing", "wrong-type", "revoke-before-commit"])
async def test_selected_reference_never_falls_back_and_stale_review_refuses(
    variable_case, change
):
    from tldw_chatbook.Plugins.variables import handle

    case = variable_case
    installed = await install_components(
        case.service, case.package, ("skill:review", "hook:guard")
    )
    from tldw_chatbook.MCP.credential_bindings import CredentialError

    with pytest.raises(CredentialError):
        await case.service.create_variable_reference(
            installed.installation_id, "COUNT", True
        )
    secret = await case.service.create_variable_reference(
        installed.installation_id, "TOKEN", "private-test"
    )
    count = await case.service.create_variable_reference(
        installed.installation_id, "COUNT", 7
    )
    review = await case.service.review_configuration(
        installed.installation_id,
        variables={"TOKEN": handle(secret), "COUNT": handle(count)},
    )
    if change == "revoke-before-commit":
        await case.service.revoke_variable_reference(
            installed.installation_id, count["reference_id"]
        )
        with pytest.raises(CredentialError):
            await case.service.commit(review, review.operation_id)
        return
    assert (await case.service.commit(review, review.operation_id)).committed
    snapshot = await capture(case, installed.installation_id)
    assert "hook:guard" in snapshot.selection
    if change == "missing":
        del case.backend.records[count["reference_id"]]
    else:
        raw = json.loads(case.backend.records[count["reference_id"]])
        raw["value"] = True
        case.backend.records[count["reference_id"]] = json.dumps(raw)
    with pytest.raises(PermissionError):
        await case.service._call(
            lambda: case.service._admission.check(snapshot, "hook:guard")
        )
    fresh = await capture(case, installed.installation_id)
    assert fresh.selection == ("skill:review",) and fresh.unavailable_hooks == (
        "hook:guard",
    )
    await case.service._call(
        lambda: case.service._admission.check(fresh, "skill:review")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("requirement_only", [False, True])
async def test_unavailable_guard_survives_actual_child_and_sealed_resume(
    variable_case, requirement_only
):
    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.hooks_v2.tool_pipeline import (
        HookPreparationError,
        ToolHookRun,
    )
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )
    from tldw_chatbook.MCP.permission_store import EffectiveToolState

    case, service = variable_case, variable_case.service
    selection = ("hook:guard",) if requirement_only else ("skill:review", "hook:guard")
    installed = await install_components(service, case.package, selection)
    maximum = service.capture_maximum("workspace-a")
    admitted = await service.admit(maximum, "original-pending")
    entries = tuple(admitted["available_skills"])
    reviews = []
    registry = ToolCatalogRegistry()
    registry.register_provider(
        LocalToolProvider(
            workspace_root=case.tmp_path,
            resolve_state=lambda _: EffectiveToolState(
                state="ask", origin="global_default"
            ),
            approval_callback=lambda calls: reviews.extend(
                call.llm_name for call in calls
            )
            or {call.llm_name: "approve_once" for call in calls},
        )
    )
    (case.tmp_path / "read-control").write_text("read")
    bound = []

    async def use(rows, actor, captured):
        with use_run_actor(actor):
            await asyncio.to_thread(service.check_entries_from_agent, rows)
            snapshots = await service.component_snapshots(
                dict(
                    captured,
                    available_skills=list(rows),
                    plugin_run_id=rows[0]["plugin_run_id"],
                )
            )
            assert len(snapshots) == 1 and snapshots[0].unavailable_hooks == (
                "hook:guard",
            )
            engine = hook_engine(case, snapshots[0])
            run = ToolHookRun(
                engine,
                run_id=actor.run_id,
                session_id="session",
                turn_id="turn",
                resolve_definition=lambda call: registry.snapshot_for_hook(call.name),
            )

            def dispatch(name, args):
                prepared = run.prepare_call(ToolCall(name, args, name))
                run.validate_dispatch(prepared)
                return registry.invoke_by_name(
                    name,
                    prepared.args,
                    expected_definition=registry.snapshot_for_hook(name),
                )

            try:
                assert (
                    await asyncio.to_thread(
                        dispatch, "fs_read", {"path": "read-control"}
                    )
                ).ok
                before = list(reviews)
                with pytest.raises(HookPreparationError):
                    await asyncio.to_thread(
                        dispatch, "fs_write", {"path": "forbidden", "content": "never"}
                    )
                assert reviews == before and not (case.tmp_path / "forbidden").exists()
                assert not engine.processes.records and not case.marker.exists()
            finally:
                await engine.close()

    try:
        await asyncio.to_thread(service.bind_run, entries, "primary", lambda: None)
        bound.append("primary")
        ceiling = {
            installed.installation_id: () if requirement_only else ("skill:review",)
        }
        child = service.constrain_entries(entries, ceiling)
        assert any(row.get("plugin_constraint_only") for row in child)
        await asyncio.to_thread(
            service.bind_run,
            child,
            "child",
            lambda: None,
            parent_run_id="primary",
            component_ceiling=ceiling,
        )
        bound.append("child")
        await use(child, CurrentRunActor("subagent", "child", "primary"), maximum)
        pin = await asyncio.to_thread(
            service.capture_resume_pin, child, "child", "conversation", "message"
        )
        payload = json.loads(pin)
        assert {
            item["component_id"] for item in payload["installations"][0]["components"]
        } == set(ceiling[installed.installation_id])
        sealed = await asyncio.to_thread(
            service.seal_resume_checkpoint,
            pin,
            parse_provider_continuation_json(json.dumps(_checkpoint())),
            "conversation",
            "message",
        )
        # A fresh same-profile producer is sufficient; the old row token is not authority.
        await service._call(service._refresh)
        fresh = service.capture_maximum("workspace-a")
        resumed = await service.resume_maximum(fresh, sealed, "conversation", "message")
        restored = await service.admit(resumed, "resumed-pending")
        resumed_entries = tuple(restored["available_skills"])
        await asyncio.to_thread(
            service.bind_run, resumed_entries, "resumed", lambda: None
        )
        bound.append("resumed")
        await use(resumed_entries, CurrentRunActor("primary", "resumed", None), resumed)
        with pytest.raises(PermissionError):
            await service.resume_maximum(
                dict(fresh, available_skills=[]),
                sealed,
                "conversation",
                "message",
            )
        changed = dict(
            fresh,
            available_skills=[
                dict(row, plugin_workspace_id="other")
                for row in fresh["available_skills"]
            ],
        )
        with pytest.raises(PermissionError):
            await service.resume_maximum(changed, sealed, "conversation", "message")
        if not requirement_only:
            omitted = await service.admit(
                dict(fresh, available_skills=[]), "omitted-pending"
            )
            assert any(
                row.get("plugin_constraint_only") for row in omitted["available_skills"]
            )
        unknown = dict(
            fresh,
            available_skills=[
                dict(row, plugin_ceiling="unknown") for row in fresh["available_skills"]
            ],
        )
        with pytest.raises(PermissionError):
            await service.admit(unknown, "unknown-pending")
        # Only actual Console actor leases exist, never a failed hook process lease.
        assert {row.run_id for row in service.live_runs()} == set(bound)
    finally:
        for run_id in reversed(bound):
            await asyncio.to_thread(service.complete_run, run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["declaration", "reference", "scope", "selection"])
async def test_requirement_metadata_refuses_actual_authority_transition(
    variable_case, change
):
    from tldw_chatbook.Plugins.variables import handle

    case, service = variable_case, variable_case.service
    installed = await install_components(
        service, case.package, ("skill:review", "hook:guard")
    )
    original = await capture(case, installed.installation_id)
    assert original.unavailable_hooks == ("hook:guard",)
    await service._call(
        lambda: service._admission.check_requirement(original, "hook:guard")
    )
    maximum = service.capture_maximum("workspace-a")
    admitted = await service.admit(maximum, "metadata-before-change")
    if change == "declaration":
        manifest = json.loads((case.package / "plugin.json").read_text())
        manifest["extensions"]["io.github.rmusser01.chatbook"]["variables"]["TOKEN"][
            "type"
        ] = "integer"
        (case.package / "plugin.json").write_text(json.dumps(manifest))
        review = await service.review_revision(installed.installation_id, case.package)
        assert (await service.apply_revision(review, review.operation_id)).committed
        trust = await service.review_trust(installed.installation_id)
        assert (await service.commit(trust, trust.operation_id)).committed
    elif change == "reference":
        saved = await service.create_variable_reference(
            installed.installation_id, "TOKEN", "protected"
        )
        review = await service.review_configuration(
            installed.installation_id, variables={"TOKEN": handle(saved)}
        )
        assert (await service.commit(review, review.operation_id)).committed
    elif change == "scope":
        review = await service.review_activation(
            installed.installation_id, workspace_id="workspace-a", intent="disabled"
        )
        assert (await service.commit(review, review.operation_id)).committed
    else:
        review = await service.review_selection(
            installed.installation_id, selection=("skill:review",)
        )
        assert (await service.apply_revision(review, review.operation_id)).committed
    with pytest.raises(PermissionError):
        await service._call(
            lambda: service._admission.check_requirement(original, "hook:guard")
        )
    with pytest.raises(PermissionError):
        await service.check_entries(admitted["available_skills"])
    with pytest.raises(PermissionError):
        await service.admit(maximum, "metadata-after-change")
    assert not service.live_runs() and not case.marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["body", "cwd"])
async def test_required_occurrence_survives_retained_material_failure(
    variable_case, failure
):
    from pathlib import Path

    from tldw_chatbook.Plugins.admission import PluginUnavailable
    from tldw_chatbook.Plugins.hooks import NativeHooks

    case, service = variable_case, variable_case.service
    if failure == "cwd":
        (case.package / "working").mkdir()
        (case.package / "working/keep").write_text("captured directory")
        case.hook["cwd"] = "working"
        (case.package / "hooks.json").write_text(
            json.dumps({"version": 2, "hooks": [case.hook]})
        )
    installed = await install_components(
        service, case.package, ("skill:review", "hook:guard")
    )
    prior = await capture(case, installed.installation_id)
    maximum = service.capture_maximum("workspace-a")
    retained = Path(prior.inspection.materialized_identity)
    target = retained / ("working" if failure == "cwd" else "skills/review/SKILL.md")
    displaced = target.with_name(target.name + ".held")
    target.rename(displaced)
    try:
        # Occurrence parsing does not dereference the missing body/cwd or mint an owner.
        native = NativeHooks(service, (prior,))
        handler = native.definitions[0]
        assert handler.required and handler.cwd is None
        with pytest.raises(PluginUnavailable):
            native.process_owner(handler, None, lambda: None)
        await service._call(service._refresh)
        with pytest.raises(
            PluginUnavailable, match="^plugin_constraint_verification_failed$"
        ):
            service.capture_maximum("workspace-a")
        with pytest.raises(
            PluginUnavailable, match="^plugin_constraint_verification_failed$"
        ):
            await service.admit(maximum, "failed-material")
        assert not service.capture_maximum("workspace-b")["available_skills"]
        assert not service.live_runs() and not case.marker.exists()
    finally:
        displaced.rename(target)
        await service._call(service._refresh)
    restored = service.capture_maximum("workspace-a")["available_skills"]
    assert any(row.get("plugin_constraint_only") for row in restored)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["no-installed", "disabled", "unselected", "revoked", "unknown"]
)
async def test_catalog_failure_keeps_legitimate_empty_states(
    variable_case, monkeypatch, state
):
    from tldw_chatbook.Plugins.admission import PluginUnavailable
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    case, service = variable_case, variable_case.service
    assert not service.capture_maximum("workspace-a")["available_skills"]
    if state == "no-installed":
        maximum = service.capture_maximum("workspace-a")
        assert not (await service.admit(maximum, "known-empty"))["available_skills"]
        return
    installed = await install_components(
        service, case.package, ("skill:review", "hook:guard")
    )
    if state == "disabled":
        review = await service.review_activation(
            installed.installation_id, workspace_id="workspace-a", intent="disabled"
        )
        assert (await service.commit(review, review.operation_id)).committed
    elif state == "unselected":
        review = await service.review_selection(installed.installation_id, selection=())
        assert (await service.apply_revision(review, review.operation_id)).committed
    elif state == "revoked":
        request = service.begin_disable(
            RevocationTarget(installed.installation_id, "workspace-a", False)
        )
        await service.finish_revocation(request)
    else:

        def unavailable(*args, **kwargs):
            raise PermissionError("authenticated_records_unavailable")

        monkeypatch.setattr(service._coordinator, "recorded_inspections", unavailable)
        await service._call(service._refresh)
        with pytest.raises(
            PluginUnavailable, match="^plugin_constraint_verification_failed$"
        ):
            service.capture_maximum("workspace-a")
        with pytest.raises(
            PluginUnavailable, match="^plugin_constraint_verification_failed$"
        ):
            await service.admit({"available_skills": []}, "unknown-state")
        return
    maximum = service.capture_maximum("workspace-a")
    assert not maximum["available_skills"]
    assert not (await service.admit(maximum, "known-empty"))["available_skills"]
    assert not service.live_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", ["plugin_constraint_verification_failed", "optional_catalog_unavailable"]
)
async def test_console_capture_translates_only_exact_constraint_failure(
    native_console, monkeypatch, code
):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat.console_configuration_capture import (
        capture_skill_context_maximum,
    )
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    rig = native_console

    # Exercise the actual published service boundary and both real Console entry points.
    def unavailable(workspace_id):
        assert workspace_id == "workspace-a"
        raise PluginUnavailable(code)

    monkeypatch.setattr(rig.service, "capture_maximum", unavailable)
    if code == "plugin_constraint_verification_failed":
        with pytest.raises(
            RecoveryRequired, match="^plugin_constraint_verification_failed$"
        ):
            capture_skill_context_maximum(rig.controller.app, "workspace-a")
        with pytest.raises(
            RecoveryRequired, match="^plugin_constraint_verification_failed$"
        ):
            await rig.controller.capture_turn_configuration_snapshot(rig.session.id)
    else:
        assert not capture_skill_context_maximum(rig.controller.app, "workspace-a")[
            "available_skills"
        ]
        snapshot = await rig.controller.capture_turn_configuration_snapshot(
            rig.session.id
        )
        assert not snapshot.skill_context_maximum["available_skills"]
    assert not rig.gateway.payloads and not rig.service.live_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["missing", "mapping", "foreign", "workspace", "replaced"]
)
async def test_captured_requirement_ceiling_refuses_changed_custody(
    variable_case, change
):
    from dataclasses import replace

    from tldw_chatbook.Plugins.admission import PluginUnavailable
    from tldw_chatbook.Plugins.service import PluginService

    case, service = variable_case, variable_case.service
    await install_components(service, case.package, ("skill:review", "hook:guard"))
    maximum = service.capture_maximum("workspace-a")
    prior = None
    if change == "missing":
        maximum.pop("_plugin_requirement_ceiling")
    elif change == "mapping":
        maximum["_plugin_requirement_ceiling"] = {"workspace": "workspace-a"}
    elif change == "foreign":
        other = PluginService(
            case.tmp_path / "foreign", workspace_lookup=lambda _: None
        )
        maximum["_plugin_requirement_ceiling"] = other.capture_maximum("workspace-a")[
            "_plugin_requirement_ceiling"
        ]
    elif change == "workspace":
        maximum["available_skills"] = [
            dict(row, plugin_workspace_id="workspace-b")
            for row in maximum["available_skills"]
        ]
    else:
        token = maximum["available_skills"][0]["plugin_ceiling"]
        prior = service._snapshots[token]
        service._snapshots[token] = replace(prior)
    try:
        with pytest.raises(
            PluginUnavailable, match="^plugin_constraint_verification_failed$"
        ):
            await service.admit(maximum, "changed-custody")
        assert not service.live_runs()
    finally:
        if prior is not None:
            service._snapshots[token] = prior


@pytest.mark.asyncio
async def test_captured_requirement_ceiling_survives_freeze_without_execution(
    variable_case,
):
    from copy import deepcopy

    from tldw_chatbook.Chat.console_turn_context import _freeze
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    case, service = variable_case, variable_case.service
    await install_components(service, case.package, ("skill:review", "hook:guard"))
    maximum = service.capture_maximum("workspace-a")
    proof = maximum["_plugin_requirement_ceiling"]
    for source in (deepcopy(maximum), _freeze(maximum)):
        assert source["_plugin_requirement_ceiling"] is proof
        omitted = dict(source, available_skills=[], plugin_run_id="refusal-only")
        admitted = await service.admit(omitted, "refusal-only")
        assert {row["plugin_component_id"] for row in admitted["available_skills"]} == {
            "hook:guard"
        }
        assert all(
            row["plugin_constraint_only"] for row in admitted["available_skills"]
        )
        snapshots = await service.component_snapshots(admitted)
        assert len(snapshots) == 1 and snapshots[0].selection == ()
        assert snapshots[0].unavailable_hooks == ("hook:guard",)
        token = admitted["available_skills"][0]["plugin_admission"]
        with pytest.raises(PluginUnavailable):
            skill = next(
                row
                for row in maximum["available_skills"]
                if row["plugin_component_id"] == "skill:review"
            )
            await service.execute_skill(skill["name"], admission_token=token)
    assert not service.live_runs()


@pytest.mark.asyncio
async def test_runtime_keeps_refusal_engine_when_all_native_rows_are_removed(
    native_console, variable_case
):
    from dataclasses import replace

    from Tests.Plugins.test_owned_hook_acceptance import _runtime
    from Tests.Plugins.test_vendor_hook_adapters import tool_event

    rig = native_console
    await install_components(
        rig.service, variable_case.package, ("skill:review", "hook:guard")
    )
    configuration = await rig.controller.capture_turn_configuration_snapshot(
        rig.session.id
    )
    maximum = dict(configuration.skill_context_maximum, available_skills=[])
    assert maximum["_plugin_requirement_ceiling"]
    configuration = replace(configuration, skill_context_maximum=maximum)
    async with _runtime(rig) as runtime:
        async with rig.controller.plugin_test_admission(configuration) as lifecycle:
            engine = lifecycle.engine
            assert engine is runtime.get_hooks_v2(rig.session.id)
            assert engine.native_plugins.unavailable
            assert all(
                handler.id in engine.native_plugins.unavailable
                for handler in engine.definitions
            )
            source = tool_event(provider="local")
            matching = lifecycle.event(
                "PreToolUse", data=dict(source.data, operation="file.write")
            )
            nonmatching = lifecycle.event(
                "PreToolUse", data=dict(source.data, operation="file.read")
            )
            assert (await engine.fire_async(nonmatching)).allowed
            assert not (await engine.fire_async(matching)).allowed
            assert not rig.gateway.payloads and not rig.service.live_runs()
        assert asyncio.current_task() not in rig.controller._hooks_v2_submissions


@pytest.mark.asyncio
async def test_bridge_validates_captured_requirements_without_availability_rows(
    native_console, variable_case, tmp_path, monkeypatch
):
    from Tests.Plugins.test_owned_hook_acceptance import _runtime
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    rig = native_console
    await install_components(
        rig.service, variable_case.package, ("skill:review", "hook:guard")
    )
    capture, admit = rig.service.capture_maximum, rig.service.admit
    observed = []

    def narrowed(workspace):
        return dict(capture(workspace), available_skills=[])

    async def admitting(maximum, run_id):
        observed.append(
            (
                bool(maximum.get("_plugin_requirement_ceiling")),
                bool(maximum.get("available_skills")),
            )
        )
        return await admit(maximum, run_id)

    monkeypatch.setattr(rig.service, "capture_maximum", narrowed)
    monkeypatch.setattr(rig.service, "admit", admitting)
    db = AgentRunsDB(tmp_path / "refusal-bridge.sqlite", client_id="refusal-bridge")
    try:
        async with _runtime(rig) as runtime:
            bridge = ConsoleAgentBridge(
                agent_runs_db=db,
                store=rig.store,
                provider_gateway=rig.gateway,
                skills_service=rig.skills,
                get_hooks_v2=runtime.get_hooks_v2,
            )
            rig.controller.update_agent_runtime(enabled=True, bridge=bridge)
            result = await rig.controller.submit_draft(
                "ordinary input", session_id=rig.session.id
            )
            assert result.accepted and rig.gateway.payloads
            assert (True, False) in observed
            engine = runtime.get_hooks_v2(rig.session.id)
            assert engine is not None and engine.native_plugins.unavailable
            assert not any(
                snapshot.selection for snapshot in engine.native_plugins.snapshots
            )
            await rig.controller.shutdown()
    finally:
        db.close()
    assert not rig.service.live_runs()


@pytest.mark.asyncio
async def test_legacy_resume_issues_only_empty_native_requirement_custody(
    variable_case,
):
    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )

    case, service = variable_case, variable_case.service
    checkpoint = parse_provider_continuation_json(json.dumps(_checkpoint()))
    assert checkpoint.schema_version == 1
    local = {"name": "local-only", "description": "local"}
    local_maximum = {"available_skills": [local]}
    zero = await service.resume_maximum(
        local_maximum, checkpoint, "conversation", "message"
    )
    assert zero["plugin_resume_constraint"] == "zero"
    assert not zero["_plugin_requirement_ceiling"]
    assert (await service.admit(zero, "legacy-local"))["available_skills"] == [local]

    await install_components(service, case.package, ("skill:review", "hook:guard"))
    captured = service.capture_maximum("workspace-a")
    assert captured["_plugin_requirement_ceiling"]
    zero = await service.resume_maximum(captured, checkpoint, "conversation", "message")
    assert not zero["available_skills"] and not zero["_plugin_requirement_ceiling"]
    assert zero["_plugin_requirement_ceiling"].workspace == "workspace-a"
    assert not await service.component_snapshots(
        dict(zero, plugin_run_id="legacy-required")
    )
    zero["available_skills"] = captured["available_skills"]
    assert not (await service.admit(zero, "legacy-substituted"))["available_skills"]
    assert not service.live_runs()


@pytest.mark.asyncio
@pytest.mark.parametrize("constraint", [None, "nonzero", "zero"])
async def test_missing_requirement_custody_never_enters_ordinary_admit(
    variable_case, constraint
):
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    maximum = {"available_skills": []}
    if constraint is not None:
        maximum["plugin_resume_constraint"] = constraint
    with pytest.raises(
        PluginUnavailable, match="^plugin_constraint_verification_failed$"
    ):
        await variable_case.service.admit(maximum, "missing-custody")
    assert not variable_case.service.live_runs()


@pytest.mark.asyncio
async def test_legacy_resume_refuses_native_rows_without_captured_custody(
    variable_case,
):
    from Tests.Chat.test_provider_continuation import _checkpoint
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
    )
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    case, service = variable_case, variable_case.service
    await install_components(service, case.package, ("skill:review", "hook:guard"))
    maximum = service.capture_maximum("workspace-a")
    maximum.pop("_plugin_requirement_ceiling")
    checkpoint = parse_provider_continuation_json(json.dumps(_checkpoint()))
    with pytest.raises(
        PluginUnavailable, match="^plugin_constraint_verification_failed$"
    ):
        await service.resume_maximum(maximum, checkpoint, "conversation", "message")
    assert not service.live_runs()
