"""Native package behavior through the production Console skill path."""

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


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
        if "controller" in locals():
            await controller.shutdown()
            store.persistence.db.close()
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
        from tldw_chatbook.Agents.activation import worker_guard

        outcome = await asyncio.to_thread(
            worker_guard(bridge)(_run),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [False, True])
async def test_current_authority_after_dispatch_wait(
    native_console, monkeypatch, change
):
    rig = native_console
    review = await rig.install()
    name = rig.service.capture_maximum("workspace-a")["available_skills"][0]["name"]
    original_wait = rig.controller._wait_for_trace_maintenance_dispatch

    async def wait():
        assert rig.service.live_runs(), "barrier must occur after plugin bind"
        if change:
            disabled = await rig.service.review_activation(
                review.installation_id, workspace_id="workspace-a", intent="disabled"
            )
            await rig.service.commit(disabled, "disable-at-wait")
        await original_wait()

    monkeypatch.setattr(rig.controller, "_wait_for_trace_maintenance_dispatch", wait)
    result = await rig.controller.submit_draft(
        f"${name} inspect", session_id=rig.session.id
    )
    if change:
        assert not rig.gateway.payloads, (
            "disabled package instructions dispatched after the wait"
        )
    else:
        assert result.accepted and len(rig.gateway.payloads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [False, True])
async def test_current_authority_before_direct_result_acceptance(
    native_console, monkeypatch, change
):
    rig = native_console
    review = await rig.install()
    name = rig.service.capture_maximum("workspace-a")["available_skills"][0]["name"]

    async def stream(resolution, messages, **kwargs):
        rig.gateway.payloads.append(messages)
        assert rig.service.live_runs(), "production dispatch must have retained a run"
        if change:
            disabled = await rig.service.review_activation(
                review.installation_id, workspace_id="workspace-a", intent="disabled"
            )
            await rig.service.commit(disabled, "disable-at-result")
        yield "LATE_PLUGIN_REPLY"

    monkeypatch.setattr(rig.gateway, "stream_chat", stream)
    result = await rig.controller.submit_draft(
        f"${name} inspect", session_id=rig.session.id
    )
    replies = [
        m.content
        for m in rig.store.messages_for_session(rig.session.id)
        if str(m.role.value) == "assistant"
    ]
    if change:
        assert "LATE_PLUGIN_REPLY" not in replies, (
            "reply accepted after authenticated scope disable"
        )
    else:
        assert result.accepted and "LATE_PLUGIN_REPLY" in replies


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", ["workspace-a"])
async def test_sentinel_scope_generation_cannot_revive_old_admission(
    native_console, workspace
):
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    rig = native_console
    review = await rig.install(enable=False)

    async def activate(intent, operation):
        change = await rig.service.review_activation(
            review.installation_id, workspace_id=workspace, intent=intent
        )
        await rig.service.commit(change, operation)

    await activate("enabled", "initial")
    context = await rig.service.admit(
        rig.service.capture_maximum(workspace), "pending:generation"
    )
    entries = context["available_skills"]
    assert len(entries) == 1
    await rig.service.check_entries(entries)
    await activate("disabled", "disable")
    with pytest.raises(PluginUnavailable):
        await rig.service.check_entries(entries)
    await activate("enabled", "reenable")
    with pytest.raises(PluginUnavailable, match="changed"):
        await rig.service.check_entries(entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_consumer", [False, True])
async def test_direct_plugin_retains_real_provider_worker_until_terminal(
    native_console, monkeypatch, cancel_consumer
):
    import asyncio
    import threading

    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderSelection,
    )

    rig = native_console
    await rig.install()
    name = rig.service.capture_maximum("workspace-a")["available_skills"][0]["name"]
    entered, release, terminal = threading.Event(), threading.Event(), threading.Event()

    def transport(**kwargs):
        entered.set()
        try:
            assert release.wait(10)
            return {"choices": [{"message": {"content": "RETAINED_REPLY"}}]}
        finally:
            terminal.set()

    gateway = ConsoleProviderGateway(
        chat_api_call_fn=transport,
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
    )
    original_resolve = gateway.resolve_for_send

    async def resolve(selection):
        return await original_resolve(
            ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1")
        )

    monkeypatch.setattr(gateway, "resolve_for_send", resolve)
    monkeypatch.setattr(rig.controller, "provider_gateway", gateway)
    submit = asyncio.create_task(
        rig.controller.submit_draft(f"${name} inspect", session_id=rig.session.id)
    )
    records = ()
    try:
        assert await asyncio.to_thread(entered.wait, 5), (
            submit.result() if submit.done() else "still running"
        )
        records = rig.service.live_runs()
        assert len(records) == 1
        if cancel_consumer:
            submit.cancel()
            await asyncio.gather(submit, return_exceptions=True)
            assert not terminal.is_set()
            assert rig.service.live_runs() == records
            assert not records[0].completed.is_set()
        release.set()
        await asyncio.gather(submit, return_exceptions=True)
        assert await asyncio.to_thread(terminal.wait, 5)
        for _ in range(200):
            if not rig.service.live_runs():
                break
            await asyncio.sleep(0.01)
        assert not rig.service.live_runs()
        assert records[0].completed.is_set()
    finally:
        release.set()
        await asyncio.gather(submit, return_exceptions=True)
        await asyncio.to_thread(terminal.wait, 5)
        for _ in range(200):
            if not rig.service.live_runs():
                break
            await asyncio.sleep(0.01)
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_count, initial, forged",
    [
        (4, False, False),
        (5, False, False),
        (3, True, False),
        (4, True, False),
        (1, False, True),
    ],
)
async def test_real_agent_final_send_bounds_aggregate_plugin_file_results(
    native_console, native_package, tmp_path, file_count, initial, forged
):
    import asyncio
    import threading

    from Tests.Chat.test_console_agent_bridge import _ChunkGateway, _fence, _run
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    rig = native_console
    package = native_package()
    for index in range(file_count):
        (package / f"skills/review/file-{index}.md").write_text(
            f"FILE_{index}:" + "x" * 6800
        )
    review = await rig.service.review_install(
        package, selection=("skill:review",), workspace_id="workspace-a"
    )
    await rig.service.commit(review, "install-files")
    trust = await rig.service.review_trust(review.installation_id)
    await rig.service.commit(trust, "trust-files")
    activation = await rig.service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(activation, "enable-files")
    maximum = rig.service.capture_maximum("workspace-a")
    maximum["plugin_run_id"] = "pending:aggregate"
    entry = maximum["available_skills"][0]
    from tldw_chatbook.Plugins.context import instruction_block

    initial_text = (
        instruction_block(
            "installation", "skill:review", "revision", "INITIAL:" + "x" * 6800
        )
        if initial
        else "hi"
    )
    bound_name = entry["name"] + "-unadmitted" if forged else entry["name"]
    gateway = _ChunkGateway(
        [
            *[
                [
                    _fence(
                        "skill_file",
                        {"skill_name": bound_name, "path": f"file-{index}.md"},
                    )
                ]
                for index in range(file_count)
            ],
            ["done"],
        ]
    )
    db = AgentRunsDB(tmp_path / "aggregate.sqlite", client_id="plugins")
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
        from tldw_chatbook.Agents.activation import worker_guard

        outcome = await asyncio.to_thread(
            worker_guard(bridge)(_run),
            bridge,
            rig.store,
            rig.session,
            assistant.id,
            conversation_id="plugin-file-send",
            skills_context=maximum,
            turn_skill_bindings=(bound_name,),
            workspace_id="workspace-a",
            agent_messages=[{"role": "user", "content": initial_text}],
            turn_bundle_block="Bundled files: file-0.md" if initial else "",
            plugin_cancel_root=threading.Event().set,
        )
        if forged:
            assert outcome.status == "done", outcome
            assert "FILE_0:" not in str(gateway.messages_seen)
        elif file_count + int(initial) <= 4:
            assert outcome.status == "done", outcome
            assert len(gateway.messages_seen) == file_count + 1
            assert all(
                f"FILE_{index}:" in str(gateway.messages_seen[-1])
                for index in range(file_count)
            )
        else:
            assert len(gateway.messages_seen) == 5 - int(initial), (
                "oversized send reached transport"
            )
            assert outcome.status == "error", outcome
        assert all(
            "plugin_context" not in str(row.keys())
            for payload in gateway.messages_seen
            for row in payload
        )
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", ["global", "workspace-default", ""])
async def test_activation_rejects_reserved_scope_without_global_fallback(
    native_console, workspace
):
    rig = native_console
    review = await rig.install(enable=False)
    with pytest.raises((ValueError, PermissionError)):
        await rig.service.review_activation(
            review.installation_id, workspace_id=workspace, intent="enabled"
        )
    assert not rig.service.capture_maximum(None)["available_skills"]
    explicit = await rig.service.review_activation(
        review.installation_id, workspace_id=None, intent="enabled"
    )
    await rig.service.commit(explicit, "explicit-global")
    assert rig.service.capture_maximum(None)["available_skills"]


@pytest.fixture
async def native_agent_submit(native_console, native_package, tmp_path, monkeypatch):
    from Tests.Chat.test_console_agent_bridge import _fence
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    rig = native_console
    established = await rig.controller.submit_draft(
        "Establish saved conversation", session_id=rig.session.id
    )
    assert established.accepted and rig.session.persisted_conversation_id
    package = native_package()
    initial = package / "skills/review/SKILL.md"
    initial.write_text(initial.read_text() + "\nPLUGIN_INITIAL:" + "x" * 6800)
    for index in range(4):
        (initial.parent / f"file-{index}.md").write_text(
            f"PLUGIN_FILE_{index}:" + "x" * 6800
        )
    review = await rig.service.review_install(
        package, selection=("skill:review",), workspace_id="workspace-a"
    )
    await rig.service.commit(review, "install-submit")
    trust = await rig.service.review_trust(review.installation_id)
    await rig.service.commit(trust, "trust-submit")
    active = await rig.service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(active, "enable-submit")
    name = rig.service.capture_maximum("workspace-a")["available_skills"][0]["name"]

    class Gateway(_RecordingGateway):
        file_count = 0

        async def stream_chat(self, resolution, messages, **kwargs):
            index = len(self.payloads)
            self.payloads.append(messages)
            if index < self.file_count:
                yield _fence(
                    "skill_file", {"skill_name": name, "path": f"file-{index}.md"}
                )
            else:
                yield "Done [S1]."

    gateway = Gateway()
    db = AgentRunsDB(tmp_path / "submit-agent.sqlite", client_id="plugins")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=rig.store,
        provider_gateway=gateway,
        skills_service=rig.skills,
    )
    monkeypatch.setattr(rig.controller, "provider_gateway", gateway)
    rig.controller.update_agent_runtime(enabled=True, bridge=bridge)
    rig.agent_gateway = gateway
    rig.agent_db = db
    rig.skill_name = name
    try:
        yield rig
    finally:
        await rig.controller.shutdown()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("file_count", [3, 4])
@pytest.mark.parametrize("exact_evidence", [False, True])
@pytest.mark.parametrize("world_info", [False, True])
async def test_actual_submit_rag_and_world_info_keep_initial_plugin_budget(
    native_agent_submit, monkeypatch, file_count, exact_evidence, world_info
):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.citation_repair import CitationRepairContract
    from tldw_chatbook.Chat.citation_trace_models import MarkerNamespace

    rig = native_agent_submit
    rig.agent_gateway.file_count = file_count
    evidence = "[S1] MEDIA — Retrieved source\nRETRIEVED_EVIDENCE"
    captures, world_calls = [], []

    async def capture(draft):
        captures.append(draft)
        return SimpleNamespace(
            context=evidence,
            citation_repair_contract=(
                CitationRepairContract(
                    schema_version=1,
                    marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
                    allowed_ordinals=(1,),
                    evidence_context=evidence,
                )
                if exact_evidence
                else None
            ),
        )

    def inject_world(_conversation, text, _history, *_frozen):
        world_calls.append(text)
        return "WORLD_INFO_PREFIX\n" + text + "\nWORLD_INFO_SUFFIX"

    monkeypatch.setattr(rig.controller, "_rag_capture_provider", capture)
    if world_info:
        monkeypatch.setattr(rig.controller, "_world_info_applier", inject_world)
    result = await rig.controller.submit_draft(
        f"${rig.skill_name} inspect", session_id=rig.session.id
    )
    assert result.accepted, result
    assert captures == [f"${rig.skill_name} inspect"]
    assert bool(world_calls) == world_info
    payloads = rig.agent_gateway.payloads
    assert len(payloads) == 4, "oversized initial-plus-files payload reached transport"
    assert "RETRIEVED_EVIDENCE" in str(payloads[0])
    assert "PLUGIN_INITIAL:" + "x" * 6800 in str(payloads[0])
    if world_info:
        assert "WORLD_INFO_PREFIX" in str(payloads[0]) and "WORLD_INFO_SUFFIX" in str(
            payloads[0]
        )
    runs = rig.agent_db.list_runs(rig.session.persisted_conversation_id)
    primary = [row for row in runs if row["agent_kind"] == "primary"]
    assert len(primary) == 1
    assert primary[0]["status"] == ("done" if file_count == 3 else "error")
    if file_count == 3:
        assert all(
            f"PLUGIN_FILE_{index}:" + "x" * 6800 in str(payloads[-1])
            for index in range(3)
        )
    assert not rig.service.live_runs()
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases

    assert not worker_leases(rig.agent_db), "settled Console worker retained its DB"


@pytest.mark.asyncio
@pytest.mark.parametrize("transform", ["dictionary", "world_info"])
@pytest.mark.parametrize("edit", ["wrap", "change", "duplicate"])
async def test_actual_submit_refuses_opaque_edits_to_live_plugin_material(
    native_agent_submit, monkeypatch, transform, edit
):
    from tldw_chatbook.Plugins.admission import PluginUnavailable

    rig = native_agent_submit
    calls = []

    def apply(_conversation, text, *_args):
        calls.append(text)
        if edit == "wrap":
            return "HOST_PREFIX\n" + text + "\nHOST_SUFFIX"
        if edit == "duplicate":
            return text + text
        return text.replace("PLUGIN_INITIAL:", "REWRITTEN_PLUGIN:")

    attribute = (
        "_chat_dictionary_applier"
        if transform == "dictionary"
        else "_world_info_applier"
    )
    monkeypatch.setattr(rig.controller, attribute, apply)
    if edit == "wrap":
        result = await rig.controller.submit_draft(
            f"${rig.skill_name} inspect", session_id=rig.session.id
        )
        assert result.accepted, result
        assert len(rig.agent_gateway.payloads) == 1
        assert "HOST_PREFIX" in str(rig.agent_gateway.payloads[0])
    else:
        with pytest.raises(PluginUnavailable, match="transform"):
            await rig.controller.submit_draft(
                f"${rig.skill_name} inspect", session_id=rig.session.id
            )
        assert not rig.agent_gateway.payloads
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_live_service_authority_uses_protected_skills_directory():
    from Tests.Plugins.test_authority_store import MemoryMarker
    from tldw_chatbook import config as app_config
    from tldw_chatbook.Plugins.authority_store import default_plugin_authority_dir
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Skills_Interop.local_skills_service import (
        default_local_skills_store_dir,
    )
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    profile = app_config.get_user_data_dir()
    service = PluginService(
        profile,
        workspace_lookup=lambda _: None,
        marker_store_factory=lambda _: MemoryMarker(),
    )
    try:
        await service.bootstrap("isolated path regression")
        actual = await service._call(lambda: service._coordinator.authority.store_dir)
        assert actual == default_plugin_authority_dir(
            default_local_skills_store_dir(profile)
        )
        for relative in (
            "metadata.json",
            "snapshots/digest.json",
            "intents/op.json",
            "certificates/op.json",
        ):
            assert is_sensitive_path(actual / relative), relative
        assert is_sensitive_path(actual.with_name("plugins-reset-state.json"))
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_storage_worker_retires_its_workspace_lookup_connection(native_console):
    from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases

    rig = native_console
    await rig.install()
    assert worker_leases(rig.registry.db), (
        "real storage worker never read workspace authority"
    )
    await rig.service.aclose()
    assert not worker_leases(rig.registry.db), (
        "closed plugin worker retained its workspace handle"
    )
