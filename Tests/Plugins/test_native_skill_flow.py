"""Native package behavior through the production Console skill path."""

import pytest


def test_explicit_workspace_disable_beats_global_default():
    from tldw_chatbook.Plugins.admission import effective_activation

    assert not effective_activation(True, "disabled")
    assert effective_activation(True, "inherit")
    assert effective_activation(False, "enabled")


@pytest.mark.asyncio
async def test_native_plugin_reaches_actual_console_provider_payload(
    tmp_path, native_package
):
    from types import SimpleNamespace

    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.console_provider_doubles import persisted_console_store
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    service = PluginService(
        tmp_path / "profile",
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: FilePluginMarkerStore(tmp_path / "marker"),
        accept_reduced_protection=True,
    )
    try:
        await service.bootstrap("test passphrase")
        review = await service.review_install(
            native_package(), selection=("skill:review",), workspace_id=None
        )
        assert (await service.commit(review, "install")).committed
        assert not service.capture_maximum("global")["available_skills"]
        trust = await service.review_trust(review.installation_id)
        await service.commit(trust, "trust")
        enable = await service.review_activation(
            review.installation_id, workspace_id=None, intent="enabled"
        )
        await service.commit(enable, "enable")
        name = service.capture_maximum("global")["available_skills"][0]["name"]
        local = LocalSkillsService(
            store_dir=tmp_path / "skills", plugin_service=service
        )
        skills = SkillsScopeService(local_service=local)
        gateway = _RecordingGateway()
        store = persisted_console_store()
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            provider="llama_cpp",
            model="m",
            skills_service=skills,
        )
        controller.app = SimpleNamespace(
            local_skills_service=local, skills_scope_service=skills
        )
        raw = f"${name} inspect this change"
        result = await controller.submit_draft(raw)
        assert result.accepted is True
        payload = gateway.payloads[-1][-1]
        assert payload["role"] == "user"
        assert "untrusted" in payload["content"].lower()
        assert review.installation_id in payload["content"]
        assert "Check the change and explain actionable findings." in payload["content"]
        assert [
            row
            for row in store.messages_for_session(store.active_session_id)
            if row.role is ConsoleMessageRole.USER
        ][-1].content == raw
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_fork_runner_uses_plugin_authority_and_empty_tool_ceiling(native_console):
    import asyncio

    from tldw_chatbook.Agents.tool_catalog import ToolResult
    from tldw_chatbook.Chat.console_agent_bridge import _BridgeSkillRunner

    rig = native_console
    await rig.install(metadata='  context: "fork"\n', tools="")
    maximum = rig.service.capture_maximum("workspace-a")
    maximum["plugin_run_id"] = "pending:fork"
    admitted = await rig.service.admit(maximum, "pending:fork")
    entry = admitted["available_skills"][0]
    runner = _BridgeSkillRunner(
        skills_service=rig.skills,
        skill_names=frozenset({entry["tool_name"]}),
        builtin_names=("calculator",),
        definition_digests={entry["tool_name"]: entry["definition_digest"]},
        plugin_entries=[entry],
    )
    spawned = []

    def spawn(prompt, *, allowed_tools):
        spawned.append((prompt, allowed_tools))
        return ToolResult(ok=True, content="done")

    result = await asyncio.to_thread(
        runner.run, entry["tool_name"], "literal $other", spawn
    )
    assert result.ok
    assert spawned[0][1] == ()
    assert "untrusted" in spawned[0][0]
    assert "literal $other" in spawned[0][0]


@pytest.mark.asyncio
async def test_owned_ids_refuse_standalone_mutation_with_standalone_control(
    native_console,
):
    rig = native_console
    await rig.install()
    entry = rig.service.capture_maximum("workspace-a")["available_skills"][0]
    assert any(
        row["record_id"] == entry["record_id"]
        for row in (await rig.skills.list_skills(mode="local"))["skills"]
    )
    assert (await rig.local.get_skill(entry["name"]))["plugin_owned"]
    for name in (entry["name"], entry["record_id"], entry["tool_name"]):
        for operation in (
            lambda name=name: rig.local.update_skill(name, content="changed"),
            lambda name=name: rig.local.delete_skill(name),
            lambda name=name: rig.local.export_skill(name),
            lambda name=name: rig.local.get_library_skill_file(name, "irrelevant"),
            lambda name=name: rig.local.import_skill(
                name=name, content="changed", overwrite=True
            ),
            lambda name=name: rig.local.create_skill(name=name, content="changed"),
            lambda name=name: rig.local.import_skill_directory(
                __import__("pathlib").Path("missing"), name=name, overwrite=True
            ),
        ):
            with pytest.raises(ValueError, match="plugin_owned"):
                await operation()
    await rig.local.create_skill(name="standalone", content="Standalone prompt")
    assert (await rig.local.execute_skill("standalone"))[
        "rendered_prompt"
    ] == "Standalone prompt"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["disable", "tamper", "archive"])
async def test_frozen_console_catalog_rechecks_authority(native_console, change):
    rig = native_console
    review = await rig.install()
    configuration = rig.controller.resolve_runtime_turn_configuration_snapshot(
        rig.session.id
    )
    entry = configuration.skill_context_maximum["available_skills"][0]
    if change == "disable":
        disable = await rig.service.review_activation(
            review.installation_id, workspace_id="workspace-a", intent="disabled"
        )
        await rig.service.commit(disable, "disable")
    elif change == "tamper":
        retained = (
            rig.service.profile_root
            / "plugins/packages"
            / review.installation_id
            / "skills/review/SKILL.md"
        )
        retained.chmod(0o600)
        retained.write_text(retained.read_text() + "Tampered")
    else:
        rig.registry.archive_workspace("workspace-a")
    result = await rig.controller.submit_draft(
        f"${entry['name']} inspect",
        session_id=rig.session.id,
        configuration=configuration,
    )
    assert not result.accepted
    assert not rig.gateway.payloads


@pytest.mark.asyncio
async def test_new_enablement_cannot_widen_frozen_console_catalog(native_console):
    rig = native_console
    configuration = rig.controller.resolve_runtime_turn_configuration_snapshot(
        rig.session.id
    )
    await rig.install()
    entry = rig.service.capture_maximum("workspace-a")["available_skills"][0]
    raw = f"${entry['name']} inspect"
    result = await rig.controller.submit_draft(
        raw, session_id=rig.session.id, configuration=configuration
    )
    assert result.accepted
    assert rig.gateway.payloads[-1][-1]["content"] == raw


@pytest.mark.asyncio
async def test_explicit_workspace_selection_does_not_follow_view(native_console):
    rig = native_console
    await rig.install()
    configuration = rig.controller.resolve_runtime_turn_configuration_snapshot(
        rig.session.id
    )
    entry = configuration.skill_context_maximum["available_skills"][0]
    rig.store.create_session(workspace_id="workspace-b")
    result = await rig.controller.submit_draft(
        f"${entry['name']} inspect",
        session_id=rig.session.id,
        configuration=configuration,
    )
    assert result.accepted
    assert "untrusted-plugin-context" in rig.gateway.payloads[-1][-1]["content"]
    assert not rig.service.capture_maximum("workspace-b")["available_skills"]


@pytest.mark.asyncio
async def test_run_ownership_retains_actual_root_and_child_until_terminal(
    native_console,
):
    import asyncio
    import threading

    rig = native_console
    await rig.install()
    admitted = await rig.service.admit(
        rig.service.capture_maximum("workspace-a"), "pending:one"
    )
    entries = admitted["available_skills"]
    root_cancel, child_cancel = threading.Event(), threading.Event()
    await asyncio.to_thread(rig.service.bind_run, entries, "root-run", root_cancel.set)
    await asyncio.to_thread(
        rig.service.bind_run,
        entries,
        "child-run",
        child_cancel.set,
        "handle-child",
        parent_run_id="root-run",
    )
    rig.service.fences.seal(entries[0]["plugin_installation_id"], "workspace-a")
    records = rig.service.live_runs()
    assert {record.run_id for record in records} == {"root-run", "child-run"}
    assert {record.workspace_id for record in records} == {"workspace-a"}
    await rig.service.retire_pending("pending:one")
    assert len(rig.service.live_runs()) == 2
    await asyncio.to_thread(rig.service.complete_run, "root-run")
    assert [record.run_id for record in rig.service.live_runs()] == ["child-run"]
    next(record for record in records if record.run_id == "child-run").cancel()
    assert child_cancel.is_set() and not root_cancel.is_set()
    await asyncio.to_thread(rig.service.complete_run, "child-run")
    assert not rig.service.live_runs()
    assert all(record.completed.is_set() for record in records)


@pytest.mark.asyncio
async def test_model_calls_native_skill_through_real_agent_spawn(
    native_console, tmp_path
):
    import asyncio
    import threading

    from Tests.Chat.test_console_agent_bridge import (
        _ChunkGateway,
        _fence,
        _join_fleet_threads,
        _run,
    )
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    rig = native_console
    await rig.install(metadata='  context: "fork"\n', tools="")
    maximum = rig.service.capture_maximum("workspace-a")
    maximum["plugin_run_id"] = "pending:real-agent"
    entry = maximum["available_skills"][0]
    gateway = _ChunkGateway(
        [
            [_fence(entry["tool_name"], {"args": "the diff"})],
            ["Skill completed."],
            ["All done."],
        ]
    )
    db = AgentRunsDB(tmp_path / "agent-runs.sqlite", client_id="plugins")
    assistant = rig.store.append_message(
        rig.session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=rig.store,
        provider_gateway=gateway,
        skills_service=rig.skills,
    )
    try:
        outcome = await asyncio.to_thread(
            _run,
            bridge,
            rig.store,
            rig.session,
            assistant.id,
            conversation_id="native-plugin-agent",
            skills_context=maximum,
            workspace_id="workspace-a",
            plugin_cancel_root=threading.Event().set,
        )
        await asyncio.to_thread(_join_fleet_threads)
        assert outcome.status == "done"
        assert db.count_subagent_runs("native-plugin-agent") == 1
        child = next(
            row
            for row in db.list_runs("native-plugin-agent")
            if row["agent_kind"] == "subagent"
        )
        assert child["status"] == "done", (child.get("result"), child.get("steps"))
        assert any(
            "untrusted-plugin-context" in str(message["content"])
            for payload in gateway.messages_seen
            for message in payload
            if message["role"] == "user"
        )
        assert not rig.service.live_runs()
    finally:
        await asyncio.to_thread(_join_fleet_threads)
        db.close()


@pytest.mark.asyncio
async def test_plugin_bundle_read_requires_same_admission_and_preserves_identity(
    native_console,
):
    rig = native_console
    await rig.install()
    maximum = rig.service.capture_maximum("workspace-a")
    admitted = await rig.service.admit(maximum, "pending:read")
    entry = admitted["available_skills"][0]
    read = await rig.skills.read_skill_file(
        entry["name"],
        "SKILL.md",
        mode="local",
        plugin_admission=entry["plugin_admission"],
    )
    assert "untrusted" in read["content"]
    assert entry["plugin_installation_id"] in read["content"]
    for path in ("../plugin.json", "/etc/passwd"):
        with pytest.raises((ValueError, PermissionError)):
            await rig.skills.read_skill_file(
                entry["name"],
                path,
                mode="local",
                plugin_admission=entry["plugin_admission"],
            )
    await rig.service.retire_pending("pending:read")
    with pytest.raises(PermissionError):
        await rig.service.read_skill_file(
            entry["name"], "SKILL.md", admission_token=entry["plugin_admission"]
        )


@pytest.mark.asyncio
async def test_invalid_metadata_blocks_execution_but_retains_owned_projection(
    native_console,
):
    rig = native_console
    review = await rig.install(metadata='  context: "system"\n')
    assert not rig.service.capture_maximum("workspace-a")["available_skills"]
    rows = rig.service.list_skills()
    assert len(rows) == 1
    assert rows[0]["plugin_installation_id"] == review.installation_id
    assert rows[0]["trust_blocked"]


@pytest.mark.asyncio
async def test_manual_only_metadata_and_whole_block_limits(native_console):
    from tldw_chatbook.Chat.console_agent_bridge import (
        _compose_run_registry_and_allowed,
    )
    from tldw_chatbook.Plugins.admission import PluginUnavailable
    from tldw_chatbook.Plugins.context import check_context_budget, instruction_block

    rig = native_console
    await rig.install(metadata='  disable_model_invocation: "true"\n')
    context = rig.service.capture_maximum("workspace-a")
    entry = context["available_skills"][0]
    catalog, allowed, *_ = _compose_run_registry_and_allowed(context)
    assert entry["tool_name"] not in allowed
    assert entry["tool_name"] not in {row.name for row in catalog.list_catalog()}
    result = await rig.controller.submit_draft(
        f"${entry['name']} explicit", session_id=rig.session.id
    )
    assert result.accepted and "untrusted" in rig.gateway.payloads[-1][-1]["content"]
    with pytest.raises(PluginUnavailable):
        instruction_block("i", "skill:s", "r", "x" * 8192)
    valid = instruction_block("i", "skill:s", "r", "x" * 7500)
    check_context_budget([valid] * 4)
    with pytest.raises(PluginUnavailable):
        check_context_budget([valid] * 5)


@pytest.mark.asyncio
async def test_cached_metadata_cannot_remove_native_manual_only_rule(native_console):
    rig = native_console
    await rig.install(metadata='  disable_model_invocation: "true"\n')
    maximum = rig.service.capture_maximum("workspace-a")
    maximum["available_skills"][0]["disable_model_invocation"] = False
    maximum["available_skills"][0]["allowed_tools"] = ["fs_write"]
    admitted = await rig.service.admit(maximum, "pending:metadata")
    assert admitted["available_skills"][0]["disable_model_invocation"] is True
    assert admitted["available_skills"][0]["allowed_tools"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["seal", "terminal"])
async def test_seal_wins_during_storage_before_host_registration(
    native_console, boundary
):
    import asyncio
    import threading

    from tldw_chatbook.Plugins.admission import PluginUnavailable

    rig = native_console
    await rig.install()
    admitted = await rig.service.admit(
        rig.service.capture_maximum("workspace-a"), "pending:blocked"
    )
    entries = admitted["available_skills"]
    entered, release = threading.Event(), threading.Event()

    def block_publication():
        owner = rig.service._coordinator.owner
        original = owner.publish_process

        def publish(*args):
            entered.set()
            assert release.wait(5)
            return original(*args)

        owner.publish_process = publish

    await rig.service._call(block_publication)
    bind = asyncio.create_task(
        asyncio.to_thread(rig.service.bind_run, entries, "unstarted", lambda: None)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if boundary == "seal":
            rig.service.fences.seal(entries[0]["plugin_installation_id"], "workspace-a")
        else:
            await asyncio.to_thread(rig.service.complete_run, "unstarted")
        release.set()
        with pytest.raises(PluginUnavailable, match="sealed|terminal"):
            await bind
        assert not rig.service.live_runs()
        processes = await rig.service._call(
            lambda: rig.service._coordinator.owner.list_processes(limit=50, offset=0)
        )
        assert len(processes) == 1
        assert processes[0]["state"] == "settled"
    finally:
        release.set()
        await asyncio.gather(bind, return_exceptions=True)
        await asyncio.to_thread(rig.service.complete_run, "unstarted")


@pytest.mark.asyncio
async def test_pending_admission_cannot_widen_or_rebind_turn(native_console):
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    rig = native_console
    await rig.install()
    maximum = rig.service.capture_maximum("workspace-a")
    maximum["plugin_turn_id"] = "turn-one"
    await rig.service.admit(maximum, "pending:once")
    with pytest.raises(PluginUnavailable, match="turn_identity"):
        await rig.service.admit(
            dict(maximum, plugin_turn_id="turn-two"), "pending:once"
        )
    await rig.install()
    widened = rig.service.capture_maximum("workspace-a")
    widened["plugin_turn_id"] = "turn-one"
    with pytest.raises(PluginUnavailable, match="ceiling_changed"):
        await rig.service.admit(widened, "pending:once")
    await rig.service.retire_pending("pending:once")
    with pytest.raises(PluginUnavailable, match="retired"):
        await rig.service.admit(maximum, "pending:once")


@pytest.mark.asyncio
async def test_consumed_pending_cannot_bind_an_unrelated_root(native_console):
    import asyncio

    from tldw_chatbook.Plugins.admission import PluginUnavailable

    rig = native_console
    await rig.install()
    admitted = await rig.service.admit(
        rig.service.capture_maximum("workspace-a"), "pending:root"
    )
    entries = admitted["available_skills"]
    await asyncio.to_thread(rig.service.bind_run, entries, "root-first", lambda: None)
    try:
        with pytest.raises(PluginUnavailable, match="root_identity"):
            await asyncio.to_thread(
                rig.service.bind_run, entries, "unrelated-root", lambda: None
            )
    finally:
        for record in rig.service.live_runs():
            await asyncio.to_thread(rig.service.complete_run, record.run_id)


@pytest.mark.asyncio
async def test_real_console_revalidates_cached_invocation_metadata(native_console):
    from dataclasses import replace

    rig = native_console
    await rig.install(metadata='  user_invocable: "false"\n')
    configuration = rig.controller.resolve_runtime_turn_configuration_snapshot(
        rig.session.id
    )
    entry = configuration.skill_context_maximum["available_skills"][0]
    configuration = replace(
        configuration,
        skill_context_maximum={
            **configuration.skill_context_maximum,
            "available_skills": [dict(entry, user_invocable=True)],
        },
    )
    raw = f"${entry['name']} inspect"
    result = await rig.controller.submit_draft(
        raw, session_id=rig.session.id, configuration=configuration
    )
    assert result.accepted
    assert rig.gateway.payloads[-1][-1]["content"] == raw
    assert not rig.service.live_runs()
