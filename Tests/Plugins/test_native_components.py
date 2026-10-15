"""Native capabilities retain explicit selection and narrowing semantics."""


def test_empty_agent_tool_list_never_inherits():
    from tldw_chatbook.Plugins.agent_presets import resolve_agent_tools

    eligible = frozenset({"fs_read"})
    assert resolve_agent_tools((), eligible) == frozenset()
    assert resolve_agent_tools("inherit", eligible) == eligible


def test_unknown_agent_tool_cannot_grant_authority():
    import pytest

    from tldw_chatbook.Plugins.agent_presets import resolve_agent_tools

    with pytest.raises(ValueError, match="plugin_agent_tools_unavailable"):
        resolve_agent_tools(("fs_write",), frozenset({"fs_read"}))


import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


@pytest.mark.asyncio
async def test_selected_native_capabilities_share_one_immutable_snapshot(
    native_console, native_package
):
    rig = native_console
    namespace = "io.github.rmusser01.chatbook"
    root = native_package(
        extension={
            "version": 1,
            "commands": [namespace + "/commands/check.md"],
            "rules": [namespace + "/rules/style.md"],
            "agents": [namespace + "/agents/reviewer.md"],
        }
    )
    for path, text in {
        "commands/check.md": "---\nname: check\ndescription: Check a change.\narguments: [target]\n---\nCheck the supplied target.",
        "rules/style.md": "---\nname: style\nmode: always\n---\nExplain clearly.",
        "agents/reviewer.md": "---\nname: reviewer\ndescription: Review without tools.\ntools: []\n---\nReview only the supplied text.",
    }.items():
        member = root / namespace / path
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text(text)
    review = await rig.service.review_install(
        root,
        selection=("skill:review", "command:check", "rule:style", "agent:reviewer"),
        workspace_id="workspace-a",
    )
    await rig.service.commit(review, review.operation_id)
    trust = await rig.service.review_trust(review.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    active = await rig.service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(active, active.operation_id)
    maximum = rig.service.capture_maximum("workspace-a")
    rows = maximum["available_skills"]
    assert any(row["plugin_component_id"] == "skill:review" for row in rows)
    assert {row["plugin_component_id"] for row in rows} == {
        "skill:review",
        "command:check",
        "rule:style",
        "agent:reviewer",
    }
    assert len({row["plugin_ceiling"] for row in rows}) == 1
    admitted = await rig.service.admit(maximum, "pending:native-components")
    assert len({row["plugin_admission"] for row in admitted["available_skills"]}) == 1


async def _install_material(rig, native_package, kind, text):
    namespace = "io.github.rmusser01.chatbook"
    field = {"command": "commands", "rule": "rules", "agent": "agents"}[kind]
    path = namespace + "/" + field + "/sample.md"
    root = native_package(extension={"version": 1, field: [path]})
    member = root / path
    member.parent.mkdir(parents=True, exist_ok=True)
    member.write_text(text)
    review = await rig.service.review_install(
        root, selection=(kind + ":sample",), workspace_id="workspace-a"
    )
    await rig.service.commit(review, review.operation_id)
    trust = await rig.service.review_trust(review.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    enabled = await rig.service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(enabled, enabled.operation_id)
    return rig.service.capture_maximum("workspace-a")["available_skills"][0]


@pytest.mark.asyncio
async def test_manual_command_refuses_missing_named_argument(
    native_console, native_package
):
    rig = native_console
    row = await _install_material(
        rig,
        native_package,
        "command",
        "---\nname: sample\ndescription: Manual check.\narguments: [target]\n---\nCommand instructions.",
    )
    result = await rig.controller.submit_draft(
        "$" + row["name"] + " {}", session_id=rig.session.id
    )
    assert not result.accepted
    assert not rig.gateway.payloads


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized", [False, True])
async def test_always_rule_enters_actual_untrusted_user_lane_whole(
    native_console, native_package, oversized
):
    rig = native_console
    body = "RULE_BODY_ONLY" + ("x" * 9000 if oversized else "")
    await _install_material(
        rig, native_package, "rule", "---\nname: sample\nmode: always\n---\n" + body
    )
    result = await rig.controller.submit_draft("hello", session_id=rig.session.id)
    if oversized:
        assert not result.accepted and not rig.gateway.payloads
    else:
        assert result.accepted
        messages = rig.gateway.payloads[-1]
        assert any(
            body in str(message["content"])
            for message in messages
            if message["role"] == "user"
        )
        assert not any(
            body in str(message["content"])
            for message in messages
            if message["role"] == "system"
        )


@pytest.mark.asyncio
async def test_actual_managed_agent_keeps_empty_tools_and_untrusted_instructions(
    native_console, native_package, tmp_path, monkeypatch
):
    import asyncio

    from Tests.Agents.conftest import pin_max_live_subagents
    from Tests.Agents.test_agent_service import (
        SUBAGENT_PROMPT_PREFIX,
        fence,
        make_service,
    )
    from Tests.Agents.test_fleet_runtime import _tool_results
    from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME, AgentConfig
    from tldw_chatbook.Agents.agent_service import (
        FirstRequestSchemaPlan,
        build_spawn_schema,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    rig = native_console
    row = await _install_material(
        rig,
        native_package,
        "agent",
        "---\nname: sample\ndescription: Review text.\ntools: []\n---\nAGENT_PACKAGE_BODY",
    )
    admitted = await rig.service.admit(
        rig.service.capture_maximum("workspace-a"), "pending:agent"
    )
    presets = await rig.service.agent_presets(
        admitted["available_skills"], frozenset({"calculator"})
    )
    pin_max_live_subagents(monkeypatch, 1)
    db = AgentRunsDB(tmp_path / "agent-runs.sqlite", client_id="native-preset-test")
    try:
        service, chat = make_service(
            db,
            [
                fence(SPAWN_TOOL_NAME, {"agent": presets[0].name, "task": "read this"}),
                fence("calculator", {"expression": "2+2"}),
                "child done",
                "parent done",
            ],
        )
        config = AgentConfig(
            model="m",
            system_prompt="system",
            allowed_tools=("calculator", SPAWN_TOOL_NAME),
            native_tools=True,
        )
        plan = FirstRequestSchemaPlan(
            active_schemas=(
                service.registry.load_schema(
                    service.registry.resolve_name("calculator")
                ),
            ),
            runtime_schemas=(build_spawn_schema(presets),),
            offer_find_load=False,
            log_active=False,
            system_prompt="system",
            agent_definitions=presets,
            fleet_max_live=1,
        )
        _run_id, outcome = await asyncio.to_thread(
            service.run_turn,
            conversation_id="native-agent",
            messages=[{"role": "user", "content": "go"}],
            config=config,
            api_endpoint="llama_cpp",
            first_request_schema_plan=plan,
        )
        assert outcome.status == "done"
        child = next(
            call
            for call in chat.calls
            if str(call["messages_payload"][0]["content"]).startswith(
                SUBAGENT_PROMPT_PREFIX
            )
        )
        assert "AGENT_PACKAGE_BODY" not in str(child["messages_payload"][0]["content"])
        assert any(
            "AGENT_PACKAGE_BODY" in str(message["content"])
            for message in child["messages_payload"]
            if message["role"] == "user"
        )
        assert not child.get("tools")
        child_run = next(
            row
            for row in db.list_runs("native-agent")
            if row["agent_kind"] == "subagent"
        )
        assert "not permitted" in str(_tool_results(child_run, "calculator")).lower()
        assert row["plugin_component_id"] == "agent:sample"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_actual_console_registers_selected_managed_agent(
    native_console, native_package, tmp_path, monkeypatch
):
    from Tests.Chat.test_console_agent_bridge import _fence
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    rig = native_console
    assert (
        await rig.controller.submit_draft("establish", session_id=rig.session.id)
    ).accepted
    await _install_material(
        rig,
        native_package,
        "agent",
        "---\nname: sample\ndescription: Review text.\ntools: []\n---\nCONSOLE_AGENT_BODY",
    )
    maximum = rig.service.capture_maximum("workspace-a")
    preview = await rig.service.admit(maximum, "pending:preview-agent")
    preset = (
        await rig.service.agent_presets(preview["available_skills"], frozenset())
    )[0]

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            index = len(self.payloads)
            self.payloads.append(messages)
            yield (
                _fence("spawn_subagent", {"agent": preset.name, "task": "read this"})
                if index == 0
                else "Done."
            )

    gateway = Gateway()
    db = AgentRunsDB(
        tmp_path / "console-agent-runs.sqlite", client_id="native-preset-test"
    )
    try:
        bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=rig.store,
            provider_gateway=gateway,
            skills_service=rig.skills,
        )
        monkeypatch.setattr(rig.controller, "provider_gateway", gateway)
        rig.controller.update_agent_runtime(enabled=True, bridge=bridge)
        result = await rig.controller.submit_draft("review", session_id=rig.session.id)
        assert result.accepted
        assert any(
            "CONSOLE_AGENT_BODY" in str(message["content"])
            for messages in gateway.payloads
            for message in messages
            if message["role"] == "user"
        )
        assert not any(
            "CONSOLE_AGENT_BODY" in str(message["content"])
            for messages in gateway.payloads
            for message in messages
            if message["role"] == "system"
        )
    finally:
        await rig.controller.shutdown()
        db.close()


@pytest.mark.asyncio
async def test_actual_console_owned_hook_reserves_native_custody_before_launch(
    native_console, native_package, monkeypatch
):
    import json

    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    rig = native_console
    namespace = "io.github.rmusser01.chatbook"
    path = namespace + "/hooks.json"
    database = rig.service.profile_root / "plugins/registry.sqlite3"
    code = (
        "import json,os,sqlite3,sys; from pathlib import Path; "
        "event=json.load(sys.stdin); "
        f"db=sqlite3.connect({str(database)!r}); "
        'row=db.execute("SELECT root_coverage,root_grants_json FROM processes '
        "WHERE operation_id=? AND state='published'\", "
        "('hook:'+event['event_id']+':'+event['owner_component_id'],)).fetchone(); "
        "assert row and row[0]=='known' and json.loads(row[1]); "
        "Path(os.environ['PLUGIN_DATA'],'native-hook').write_text('owned'); "
        "print(json.dumps({'version':2,'decision':'pass','context':[{'text':'NATIVE_HOOK_CONTEXT','lifetime':'runtime'}]}))"
    )
    package = native_package(extension={"version": 1, "hooks": path})
    (package / path).parent.mkdir(parents=True, exist_ok=True)
    (package / path).write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "initialize",
                        "event": "SessionStart",
                        "type": "command",
                        "effects": ["context"],
                        "required": True,
                        "argv": child_argv(code),
                    }
                ],
            }
        )
    )
    review = await rig.service.review_install(
        package, selection=("hook:initialize",), workspace_id="workspace-a"
    )
    await rig.service.commit(review, review.operation_id)
    roots_review = await rig.service.review_data_creation(review.installation_id)
    root = await rig.service.create_data(roots_review, roots_review.operation_id)
    trust = await rig.service.review_trust(review.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    active = await rig.service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(active, active.operation_id)
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    observed = []
    deliver = HookEngine._deliver

    async def observe(self, *args):
        outcome = await deliver(self, *args)
        observed.append(tuple(failure.code for failure in outcome.failures))
        return outcome

    monkeypatch.setattr(HookEngine, "_deliver", observe)
    runtime = ConsoleRuntime(app=rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    try:
        result = await rig.controller.submit_draft("hello", session_id=rig.session.id)
        assert result.accepted, observed
        assert (root.path / "native-hook").read_text() == "owned"
        assert any(
            "NATIVE_HOOK_CONTEXT" in str(row["content"])
            for row in rig.gateway.payloads[-1]
            if row["role"] == "user"
        )
        engine = runtime.get_hooks_v2(rig.session.id)
        assert engine is not None and engine.native_plugins is not None
        await runtime.prepare_hooks_v2(rig.session.id, reason="manual")
        assert runtime.get_hooks_v2(rig.session.id) is engine
        processes = await rig.service._call(
            lambda: rig.service._coordinator.owner.list_processes(limit=50, offset=0)
        )
        hook_rows = [
            row for row in processes if row["operation_id"].startswith("hook:")
        ]
        assert hook_rows and all(row["state"] == "settled" for row in hook_rows)
        assert not runtime.hooks_v2_cleanup_pending
    finally:
        await runtime.dispose()


from Tests.Plugins.test_owned_mcp_tools import (
    shared_connection_case as shared_connection_case,  # noqa: PLC0414
)


@pytest.mark.asyncio
async def test_actual_console_composes_owned_mcp_under_captured_selection(
    shared_connection_case, tmp_path
):
    import asyncio
    from types import SimpleNamespace

    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.console_provider_doubles import persisted_console_store
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    case = shared_connection_case
    local = LocalSkillsService(
        store_dir=tmp_path / "skills", plugin_service=case.service
    )
    skills = SkillsScopeService(local_service=local)
    store = persisted_console_store()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(),
        provider="llama_cpp",
        model="m",
        skills_service=skills,
    )
    controller.app = SimpleNamespace(
        unified_mcp_service=case.unified,
        local_skills_service=local,
        skills_scope_service=skills,
    )
    tool, _state = next(iter(case.providers["a"]._entry_by_llm_name.values()))
    case.unified.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    maximum = case.service.capture_maximum("a")
    maximum["plugin_run_id"] = "pending:console-mcp"
    try:
        assert await controller._compose_mcp_provider(publish_counts=False) is None
        provider = await controller._compose_mcp_provider(
            publish_counts=False, plugin_maximum=maximum
        )
        assert provider is not None and provider.list_catalog()
        name = provider.list_catalog()[0].name
        assert name.startswith("plugin_mcp_")
        result = await asyncio.to_thread(provider.invoke, name, {"literal": "$other"})
        assert result.ok
        import json

        assert json.loads(case.calls.read_text().splitlines()[-1]) == {
            "literal": "$other"
        }
    finally:
        await controller.shutdown()
        store.persistence.db.close()


@pytest.mark.asyncio
async def test_missing_selected_guard_does_not_disable_unrelated_command(
    native_console, native_package
):
    import json

    from Tests.hooks_v2_process_support import child_argv

    rig = native_console
    ns = "io.github.rmusser01.chatbook"
    command_path, hook_path = ns + "/commands/check.md", ns + "/hooks.json"
    package = native_package(
        extension={
            "version": 1,
            "commands": [command_path],
            "hooks": hook_path,
            "requires": {"skill:review": ["hook:guard"]},
        }
    )
    (package / command_path).parent.mkdir(parents=True, exist_ok=True)
    (package / command_path).write_text(
        "---\nname: check\ndescription: Unrelated check.\narguments: []\n---\nUNRELATED_COMMAND"
    )
    (package / hook_path).write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "guard",
                        "event": "PreToolUse",
                        "type": "command",
                        "effects": ["deny"],
                        "argv": child_argv("pass"),
                    }
                ],
            }
        )
    )
    review = await rig.service.review_install(
        package, selection=("skill:review", "command:check"), workspace_id="workspace-a"
    )
    await rig.service.commit(review, review.operation_id)
    trust = await rig.service.review_trust(review.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    active = await rig.service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(active, active.operation_id)
    available = rig.service.capture_maximum("workspace-a")["available_skills"]
    assert [row["plugin_component_id"] for row in available] == ["command:check"]
    details = {
        row["plugin_component_id"]: row
        for row in rig.service.list_components("workspace-a")
    }
    assert not details["skill:review"]["plugin_available"]
    assert (
        "dependency_unavailable:hook:guard"
        in details["skill:review"]["plugin_blockers"]
    )
    assert not details["hook:guard"]["plugin_selected"]
    result = await rig.controller.submit_draft(
        "$" + available[0]["name"] + " {}", session_id=rig.session.id
    )
    assert result.accepted
    assert "UNRELATED_COMMAND" in str(rig.gateway.payloads[-1][-1]["content"])


@pytest.mark.asyncio
async def test_valid_rule_blocks_refuse_total_context_overflow_whole(
    native_console, native_package
):
    rig = native_console
    ns = "io.github.rmusser01.chatbook"
    paths = [ns + f"/rules/rule{index}.md" for index in range(6)]
    package = native_package(extension={"version": 1, "rules": paths})
    for index, path in enumerate(paths):
        (package / path).parent.mkdir(parents=True, exist_ok=True)
        (package / path).write_text(
            f"---\nname: rule{index}\nmode: always\n---\n" + "x" * 6000
        )
    review = await rig.service.review_install(
        package,
        selection=tuple(f"rule:rule{i}" for i in range(6)),
        workspace_id="workspace-a",
    )
    await rig.service.commit(review, review.operation_id)
    trust = await rig.service.review_trust(review.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    active = await rig.service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(active, active.operation_id)
    result = await rig.controller.submit_draft("hello", session_id=rig.session.id)
    assert not result.accepted and not rig.gateway.payloads


@pytest.mark.asyncio
async def test_native_agent_requires_exact_host_model_and_tool_mapping(
    native_console, native_package
):
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import LocalLLMProviderCatalogService

    rig = native_console
    namespace = "io.github.rmusser01.chatbook"
    path = namespace + "/agents/mapped.md"
    root = native_package(extension={"version": 1, "agents": [path]})
    member = root / path
    member.parent.mkdir(parents=True, exist_ok=True)
    member.write_text(
        "---\nname: mapped\ndescription: Mapped agent.\n"
        "tools: [arithmetic]\nmodel: author-model\n---\nMAPPED_AGENT_BODY"
    )
    install = await rig.service.review_install(
        root, selection=("agent:mapped",), workspace_id="workspace-a"
    )
    await rig.service.commit(install, install.operation_id)
    trust = await rig.service.review_trust(install.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    enable = await rig.service.review_activation(
        install.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(enable, enable.operation_id)
    assert not rig.service.capture_maximum("workspace-a")["available_skills"]
    provider, models = "llama_cpp", ["mapped"]
    owner = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {provider: models},
        settings_loader=lambda: {"api_settings": {provider: {"api_url": "http://127.0.0.1:8080/v1/chat/completions"}}},
        environ={},
    )
    await rig.service._call(lambda: setattr(rig.service._coordinator, "model_mapping_owner", owner))
    review = await rig.service.review_configuration(
        install.installation_id,
        tool_references={"agent:mapped": {"arithmetic": "builtin:calculator"}},
        models={"agent:mapped": provider + "::" + models[0]},
    )
    await rig.service.commit(review, review.operation_id)
    admitted = await rig.service.admit(
        rig.service.capture_maximum("workspace-a"), "pending:mapped-agent"
    )
    assert len(admitted["available_skills"]) == 1
    (detail,) = [
        row
        for row in rig.service.list_components("workspace-a")
        if row["plugin_component_id"] == "agent:mapped"
    ]
    assert detail["plugin_available"] and not detail["plugin_blockers"]
    (preset,) = await rig.service.agent_presets(
        admitted["available_skills"],
        frozenset({"calculator", "datetime"}),
        parent_provider=provider,
    )
    assert preset.tool_allowlist == ("calculator",)
    assert preset.provider == provider and preset.model == models[0]
    assert "MAPPED_AGENT_BODY" in preset.instructions
    with pytest.raises((ValueError, PermissionError)):
        await rig.service.review_configuration(
            install.installation_id,
            tool_references={"agent:mapped": {"arithmetic": "builtin:nonexistent"}},
            models={"agent:mapped": provider + "::" + models[0]},
        )


@pytest.mark.asyncio
async def test_empty_host_child_ceiling_cannot_inherit_owned_mcp_dispatch(
    shared_connection_case,
):
    import asyncio

    from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor

    case = shared_connection_case
    case.release.touch()
    await case.pending
    admitted = await case.service.admit(
        case.service.capture_maximum("a"), "pending:child-ceiling"
    )
    rows = admitted["available_skills"]
    await asyncio.to_thread(case.service.bind_run, rows, "parent-live", lambda: None)
    await asyncio.to_thread(
        case.service.bind_run,
        (),
        "child-empty",
        lambda: None,
        parent_run_id="parent-live",
        component_ceiling={},
    )
    provider = case.providers["a"]
    # Use the parent's actual native transport and private normal permission store.
    tool, _state = next(iter(provider._entry_by_llm_name.values()))
    case.unified.set_tool_state(tool.server_key, tool.name, "ask", tool=tool)
    approvals = []
    original_approval = provider._approval_callback

    def observe_approval(calls):
        approvals.append(calls)
        return original_approval(calls)

    provider._approval_callback = observe_approval
    before = case.calls.read_text()

    def invoke_as_child():
        with use_run_actor(CurrentRunActor("subagent", "child-empty", "parent-live")):
            name = provider.list_catalog()[0].name
            assert provider.pending_gate_for(name, {"child": "forbidden"}) is None
            return provider.invoke(name, {"child": "forbidden"})

    try:
        result = await asyncio.to_thread(invoke_as_child)
        assert not result.ok and result.dispatch_state == "not_started"
        assert case.calls.read_text() == before
        assert not approvals, "An EMPTY child must not arm an impossible approval"
        assert (await case.invoke("b", {"sibling": "allowed"})).ok
    finally:
        await asyncio.to_thread(case.service.complete_run, "child-empty")
        await asyncio.to_thread(case.service.complete_run, "parent-live")


@pytest.mark.asyncio
async def test_missing_hook_data_is_component_local_unavailability(
    native_console, native_package
):
    import json

    rig = native_console
    namespace = "io.github.rmusser01.chatbook"
    path = namespace + "/commands/plain.md"
    root = native_package(
        extension={"version": 1, "commands": [path], "hooks": namespace + "/hooks.json"}
    )
    member = root / path
    member.parent.mkdir(parents=True, exist_ok=True)
    member.write_text(
        "---\nname: plain\ndescription: Unrelated command.\n---\nUNRELATED_COMMAND"
    )
    (root / namespace / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "data",
                        "event": "SessionStart",
                        "type": "command",
                        "effects": ["context"],
                        "argv": ["python3", "${PLUGIN_DATA}/init.py"],
                    }
                ],
            }
        )
    )
    install = await rig.service.review_install(
        root, selection=("command:plain", "hook:data"), workspace_id="workspace-a"
    )
    await rig.service.commit(install, install.operation_id)
    trust = await rig.service.review_trust(install.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    enabled = await rig.service.review_activation(
        install.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(enabled, enabled.operation_id)
    hook = next(
        row
        for row in rig.service.list_components("workspace-a")
        if row["plugin_component_id"] == "hook:data"
    )
    assert hook["plugin_support"] == "supported" and not hook["plugin_available"]
    assert "plugin_requirements_unavailable" in hook["plugin_blockers"]
    rows = rig.service.capture_maximum("workspace-a")["available_skills"]
    assert [row["plugin_component_id"] for row in rows] == ["command:plain"]
    result = await rig.controller.submit_draft(
        "$" + rows[0]["name"], session_id=rig.session.id
    )
    assert result.accepted and "UNRELATED_COMMAND" in str(rig.gateway.payloads[-1])


@pytest.mark.asyncio
async def test_actual_inline_native_skill_empty_tools_refuses_model_tool(
    native_console, tmp_path, monkeypatch
):
    from Tests.Chat.test_console_agent_bridge import _fence
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    rig = native_console
    assert (
        await rig.controller.submit_draft("establish", session_id=rig.session.id)
    ).accepted
    await rig.install(tools="")
    row = rig.service.capture_maximum("workspace-a")["available_skills"][0]

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            index = len(self.payloads)
            self.payloads.append(messages)
            yield _fence("calculator", {"expression": "2+2"}) if index == 0 else "Done."

    gateway = Gateway()
    db = AgentRunsDB(tmp_path / "inline-tools.sqlite", client_id="native-inline")
    try:
        bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=rig.store,
            provider_gateway=gateway,
            skills_service=rig.skills,
        )
        monkeypatch.setattr(rig.controller, "provider_gateway", gateway)
        rig.controller.update_agent_runtime(enabled=True, bridge=bridge)
        result = await rig.controller.submit_draft(
            "$" + row["name"] + " inspect", session_id=rig.session.id
        )
        assert result.accepted
        results = [
            message
            for payload in gateway.payloads
            for message in payload
            if message["role"] == "tool"
        ]
        assert not results, "EMPTY removes tool dispatch entirely from this request"
        assert "untrusted-plugin-context" in str(gateway.payloads[0])
    finally:
        await rig.controller.shutdown()
        db.close()


@pytest.mark.asyncio
async def test_actual_console_dependency_failure_blocks_owned_skill_only(
    native_console, native_package, tmp_path, monkeypatch
):
    import json

    from Tests.Chat.test_console_agent_bridge import _fence
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    rig = native_console
    assert (
        await rig.controller.submit_draft("establish", session_id=rig.session.id)
    ).accepted
    namespace = "io.github.rmusser01.chatbook"
    path = namespace + "/hooks.json"
    root = native_package(
        extension={
            "version": 1,
            "hooks": path,
            "requires": {"skill:review": ["hook:guard"]},
        }
    )
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "guard",
                        "event": "SessionStart",
                        "type": "command",
                        "effects": ["context"],
                        "argv": child_argv("raise SystemExit(1)"),
                    }
                ],
            }
        )
    )
    install = await rig.service.review_install(
        root, selection=("skill:review", "hook:guard"), workspace_id="workspace-a"
    )
    await rig.service.commit(install, install.operation_id)
    trust = await rig.service.review_trust(install.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    enabled = await rig.service.review_activation(
        install.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(enabled, enabled.operation_id)
    name = next(
        row["tool_name"]
        for row in rig.service.capture_maximum("workspace-a")["available_skills"]
        if row["plugin_kind"] == "skill"
    )

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            index = len(self.payloads)
            self.payloads.append(messages)
            yield (
                _fence("calculator", {"expression": "2+2"})
                if index == 0
                else _fence(name, {"args": "inspect"})
                if index == 1
                else "Done."
            )

    gateway = Gateway()
    db = AgentRunsDB(tmp_path / "dependencies.sqlite", client_id="native-dependencies")
    runtime = ConsoleRuntime(app=rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    try:
        bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=rig.store,
            provider_gateway=gateway,
            skills_service=rig.skills,
            get_hooks_v2=runtime.get_hooks_v2,
        )
        monkeypatch.setattr(rig.controller, "provider_gateway", gateway)
        rig.controller.update_agent_runtime(enabled=True, bridge=bridge)
        result = await rig.controller.submit_draft("inspect", session_id=rig.session.id)
        assert result.accepted
        assert any(
            '"result": 4' in str(message["content"])
            for payload in gateway.payloads
            for message in payload
        )
        assert any(
            "hook: preparation refused" in message.content
            for message in rig.store.messages_for_session(rig.session.id)
        )
        assert not any(
            "Check the change and explain actionable findings."
            in str(message["content"])
            for payload in gateway.payloads
            for message in payload
        )
    finally:
        await runtime.dispose()
        await rig.controller.shutdown()
        db.close()


@pytest.mark.asyncio
async def test_actual_console_owned_mcp_dispatch_uses_host_run_binding(
    shared_connection_case, tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    from Tests.Chat.test_console_agent_bridge import _fence
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.console_provider_doubles import persisted_console_store
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    case = shared_connection_case
    case.release.touch()
    await case.pending
    local = LocalSkillsService(
        store_dir=tmp_path / "console-skills", plugin_service=case.service
    )
    skills = SkillsScopeService(local_service=local)
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "console-workspaces.sqlite", client_id="owned-console")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="a", name="Actual MCP")
    store = persisted_console_store(workspace_registry=registry)
    session = store.create_session(workspace_id="a")
    name = case.providers["a"].list_catalog()[0].name

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            index = len(self.payloads)
            self.payloads.append(messages)
            yield (
                _fence("load_tools", {"ids": [name]})
                if index == 0
                else _fence(name, {"from_console": True})
                if index == 1
                else "Done."
            )

    gateway = Gateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_RecordingGateway(),
        provider="llama_cpp",
        model="m",
        skills_service=skills,
    )
    controller.app = SimpleNamespace(
        unified_mcp_service=case.unified,
        local_skills_service=local,
        skills_scope_service=skills,
        workspace_registry_service=registry,
    )
    db = AgentRunsDB(tmp_path / "owned-console.sqlite", client_id="owned-console")
    tool, _state = next(iter(case.providers["a"]._entry_by_llm_name.values()))
    case.unified.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)
    try:
        assert (
            await controller.submit_draft("establish", session_id=session.id)
        ).accepted
        configuration = controller.resolve_runtime_turn_configuration_snapshot(
            session.id
        )
        assert tool.tool_id in configuration.mcp_tool_maximum
        assert any(
            row.get("plugin_component_id") == "mcp:shared"
            for row in configuration.skill_context_maximum["available_skills"]
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=store,
            provider_gateway=gateway,
            skills_service=skills,
        )
        monkeypatch.setattr(controller, "provider_gateway", gateway)
        controller.update_agent_runtime(enabled=True, bridge=bridge)
        result = await controller.submit_draft(
            "use the selected tool", session_id=session.id
        )
        assert result.accepted
        if not any(
            json.loads(line) == {"from_console": True}
            for line in case.calls.read_text().splitlines()
        ):
            pytest.fail(
                "Synthetic Console outcome: "
                + "\n".join(
                    message.content
                    for message in store.messages_for_session(session.id)
                )[-900:]
            )
        assert not [
            record
            for record in case.service.live_runs()
            if record.handle_id is not None or record.turn_id == "console"
        ]
    finally:
        await controller.shutdown()
        db.close()
        store.persistence.db.close()
        registry.db.close()


@pytest.mark.asyncio
async def test_revoking_actual_owned_hook_kills_and_reaps_before_settling(
    native_console, native_package
):
    import asyncio
    import json
    import time

    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Plugins.revocation import RevocationTarget

    rig = native_console
    namespace = "io.github.rmusser01.chatbook"
    path = namespace + "/hooks.json"
    package = native_package(extension={"version": 1, "hooks": path})
    (package / path).parent.mkdir(parents=True, exist_ok=True)
    code = "import os,time; from pathlib import Path; Path(os.environ['PLUGIN_DATA'],'started').touch(); time.sleep(30)"
    (package / path).write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "blocking",
                        "event": "UserPromptSubmit",
                        "type": "command",
                        "effects": ["context"],
                        "required": True,
                        "timeout_seconds": 30,
                        "argv": child_argv(code),
                    }
                ],
            }
        )
    )
    install = await rig.service.review_install(
        package, selection=("hook:blocking",), workspace_id="workspace-a"
    )
    await rig.service.commit(install, install.operation_id)
    data_review = await rig.service.review_data_creation(install.installation_id)
    data = await rig.service.create_data(data_review, data_review.operation_id)
    trust = await rig.service.review_trust(install.installation_id)
    await rig.service.commit(trust, trust.operation_id)
    enable = await rig.service.review_activation(
        install.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await rig.service.commit(enable, enable.operation_id)
    runtime = ConsoleRuntime(app=rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    send = asyncio.create_task(
        rig.controller.submit_draft("hello", session_id=rig.session.id)
    )
    try:
        deadline = time.monotonic() + 10
        while not (data.path / "started").exists():
            assert time.monotonic() < deadline and not send.done()
            await asyncio.sleep(0.025)
        records = [
            record
            for record in rig.service.live_runs()
            if record.installation_id == install.installation_id
        ]
        assert records and not any(record.completed.is_set() for record in records)
        await rig.service.disable(
            RevocationTarget(install.installation_id, "workspace-a", False)
        )
        assert not (await send).accepted
        assert not rig.gateway.payloads
        assert all(record.completed.is_set() for record in records)
        rows = await rig.service._call(
            lambda: rig.service._coordinator.owner.list_processes(limit=50, offset=0)
        )
        assert all(
            row["state"] == "settled"
            for row in rows
            if row["operation_id"].startswith("hook:")
        )
        assert not runtime.hooks_v2_cleanup_pending
    finally:
        if not send.done():
            send.cancel()
        await asyncio.gather(send, return_exceptions=True)
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["command", "rule"])
async def test_embedded_native_manual_material_keeps_full_namespace_and_literal_body(
    native_console, native_package, kind
):
    rig = native_console
    text = (
        "---\nname: sample\ndescription: Manual command.\n---\n"
        if kind == "command"
        else "---\nname: sample\nmode: manual\n---\n"
    ) + "MANUAL_NATIVE_BODY $other"
    row = await _install_material(rig, native_package, kind, text)
    result = await rig.controller.submit_draft(
        "Please apply $" + row["name"] + " here", session_id=rig.session.id
    )
    assert result.accepted
    payload = rig.gateway.payloads[-1]
    assert any(
        "MANUAL_NATIVE_BODY $other" in str(message["content"])
        for message in payload
        if message["role"] == "user"
    )
    assert not any(
        "MANUAL_NATIVE_BODY" in str(message["content"])
        for message in payload
        if message["role"] == "system"
    )


def test_native_tool_reference_uses_mcp_owner_even_when_other_native_mappings_sort_first():
    import json

    from tldw_chatbook.Plugins.host_references import (
        mapped_tools,
        mapping_id,
        owned_mcp_tool_name,
    )
    from tldw_chatbook.Plugins.models import ComponentRecord

    component = ComponentRecord(
        component_id="agent:a",
        kind="agent",
        local_id="a",
        path="a.md",
        definition_json=json.dumps({"tools": ["read"]}),
    )
    target = "local:profile::echo"
    mappings = [
        {
            "mapping_id": "native-other",
            "kind": "tool",
            "component_id": "agent:b",
            "target_reference": target,
        },
        {
            "mapping_id": mapping_id("agent:a", "read"),
            "kind": "tool",
            "component_id": "agent:a",
            "target_reference": target,
        },
        {
            "mapping_id": "tool-original",
            "kind": "tool",
            "component_id": "mcp:shared",
            "target_reference": target,
        },
    ]
    mappings = [{**row, "installation_id": "installed", "revision_digest": "revision"} for row in mappings]
    assert mapped_tools("installed", component, mappings) == (
        owned_mcp_tool_name("installed", "mcp:shared", "echo"),
    )


async def install_components(service, package, selection):
    review = await service.review_install(
        package, selection=selection, workspace_id="workspace-a"
    )
    await service.commit(review, review.operation_id)
    trust = await service.review_trust(review.installation_id)
    await service.commit(trust, trust.operation_id)
    enable = await service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    await service.commit(enable, enable.operation_id)
    return review


def test_model_mapping_recovers_only_exact_current_owner(plugin_stack, native_package):
    from Tests.Plugins.test_recovery import lose_registry
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.Plugins.inspection import inspect_package

    stack = plugin_stack
    settings = {
        "api_settings": {
            "llama_cpp": {"api_url": "http://127.0.0.1:8080/v1/chat/completions"}
        }
    }
    models = {"llama_cpp": ["m"]}
    owner = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: models,
        settings_loader=lambda: settings,
        environ={},
    )
    stack.call(lambda: setattr(stack.coordinator, "model_mapping_owner", owner))
    package = native_package(extension={"version": 1, "agents": ["agent.md"]})
    (package / "agent.md").write_text(
        "---\nname: analyst\ndescription: Analyze\ntools: inherit\nmodel: preferred\n---\nInspect carefully."
    )
    review = stack.call(
        lambda: stack.coordinator.review(
            inspect_package(package), selection=("agent:analyst",), workspace_id=None
        )
    )
    stack.call(lambda: stack.coordinator.commit(review, review.operation_id))
    with pytest.raises(ValueError, match="canonical"):
        stack.call(
            lambda: stack.coordinator.review_configuration(
                review.installation_id, models={"agent:analyst": "m"}
            )
        )
    mapped = stack.call(
        lambda: stack.coordinator.review_configuration(
            review.installation_id, models={"agent:analyst": "llama_cpp/m"}
        )
    )
    stack.call(lambda: stack.coordinator.commit(mapped, mapped.operation_id))
    expected = stack.call(stack.coordinator.published_snapshot)["mappings"]
    assert expected[0]["kind"] == "model"
    lose_registry(stack)
    stack.call(lambda: setattr(stack.coordinator, "model_mapping_owner", owner))
    stack.call(stack.coordinator.recover)
    assert stack.call(stack.coordinator.published_snapshot)["mappings"] == expected
    models["llama_cpp"] = ["different"]
    lose_registry(stack)
    stack.call(lambda: setattr(stack.coordinator, "model_mapping_owner", owner))
    outcomes = stack.call(stack.coordinator.recover)
    assert any(row.phase == "recovery_required" for row in outcomes)
    with pytest.raises(PermissionError):
        stack.call(stack.coordinator.published_snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_connection_case", ["native_tools"], indirect=True)
async def test_configuration_recovery_scopes_identical_tool_ids_to_installation_revision(
    shared_connection_case, native_package
):
    import asyncio
    import json
    import shutil

    from tldw_chatbook.MCP.credential_bindings import CredentialError
    from tldw_chatbook.Plugins.mcp_provider import PluginMCPProvider
    from tldw_chatbook.Plugins.recovery import retained_inspections

    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok

    async def configure(installation, profile_id):
        review = await case.service.review_configuration(
            installation,
            connections={"mcp:shared": profile_id},
            tools={"mcp:shared": ("echo",)},
            component_tools={"agent:reader": {"search": "mcp:shared::echo"}},
        )
        assert (await case.service.commit(review, review.operation_id)).committed

    await configure(case.installation, case.profile_id)
    assert not any(
        row.phase == "recovery_required"
        for row in await case.service._call(case.service._coordinator.recover)
    )
    sibling = native_package()
    shutil.copytree(case.package, sibling, dirs_exist_ok=True)
    manifest = json.loads((sibling / "plugin.json").read_text())
    manifest["name"] = "sibling-review-helper"
    (sibling / "plugin.json").write_text(json.dumps(manifest))
    review = await case.service.review_install(
        sibling,
        selection=("skill:review", "mcp:shared", "agent:reader"),
        workspace_id="a",
    )
    assert (await case.service.commit(review, review.operation_id)).committed
    second = review.installation_id
    trust = await case.service.review_trust(second)
    assert (await case.service.commit(trust, trust.operation_id)).committed
    active = await case.service.review_activation(
        second, workspace_id="a", intent="enabled"
    )
    assert (await case.service.commit(active, active.operation_id)).committed
    data_review = await case.service.review_data_creation(second, workspace_id=None)
    data_root = await case.service.create_data(data_review, data_review.operation_id)
    snapshot = await case.service.capture_mcp_snapshot(second, "a", "sibling-discovery")
    profile = case.local.save_owned_profile(
        installation_id=second,
        inspection=snapshot.inspection,
        component_id="mcp:shared",
        data_root=data_root,
        session_isolation="request_independent",
        protocol_version="2025-03-26",
    )
    connection = await case.service.review_configuration(
        second, connections={"mcp:shared": profile["profile_id"]}
    )
    assert (await case.service.commit(connection, connection.operation_id)).committed
    provider = PluginMCPProvider(
        plugin_service=case.service,
        ownership=case.ownership,
        snapshot=await case.service.capture_mcp_snapshot(
            second, "a", "sibling-discovery"
        ),
        service=case.unified,
        main_loop=asyncio.get_running_loop(),
    )
    assert (await provider.connect("mcp:shared"))["tools"][0]["name"] == "echo"
    await configure(second, profile["profile_id"])
    authority = await case.service._call(case.service._coordinator.published_snapshot)
    mappings = authority["mappings"]
    targets = [row for row in mappings if row["kind"] == "tool"]
    assert len(targets) == 2 and targets[0]["mapping_id"] == targets[1]["mapping_id"]
    assert len({row["installation_id"] for row in targets}) == 2
    assert len({row["revision_digest"] for row in targets}) == 2
    assert not any(
        row.phase == "recovery_required"
        for row in await case.service._call(case.service._coordinator.recover)
    )
    assert (await case.service._call(case.service._coordinator.published_snapshot))[
        "mappings"
    ] == mappings
    inspections = retained_inspections(authority)
    constraints = {
        row["installation_id"]: row
        for row in mappings
        if row["kind"] == "configuration"
    }

    async def validate(installation, candidates):
        constraint = constraints[installation]
        await case.service._call(
            lambda: case.service._coordinator.validate_mapping(
                constraint,
                inspections[(installation, constraint["revision_digest"])],
                [constraint, *[row for row in candidates if row != constraint]],
            )
        )

    own = next(row for row in targets if row["installation_id"] == case.installation)
    foreign = next(row for row in targets if row["installation_id"] == second)
    with pytest.raises(PermissionError):
        await validate(case.installation, [foreign])
    with pytest.raises(PermissionError):
        await validate(
            case.installation, [{**own, "revision_digest": foreign["revision_digest"]}]
        )
    original = case.local.store.get_discovery_snapshot(case.profile_id)
    changed = json.loads(json.dumps(original))
    changed["tools"][0]["description"] = "changed A only"
    case.local.store.save_discovery_snapshot(case.profile_id, changed)
    try:
        with pytest.raises(CredentialError, match="credential_mapping_changed"):
            await validate(case.installation, mappings)
        await validate(second, mappings)
        assert any(
            row.phase == "recovery_required"
            for row in await case.service._call(case.service._coordinator.recover)
        )
    finally:
        case.local.store.save_discovery_snapshot(case.profile_id, original)
        assert not any(
            row.phase == "recovery_required"
            for row in await case.service._call(case.service._coordinator.recover)
        )


@pytest.mark.asyncio
async def test_native_agent_empty_tools_survives_real_spawn_and_resume(
    native_console, native_package, tmp_path, monkeypatch
):
    import asyncio

    from Tests.Agents.conftest import pin_agent_settings
    from Tests.Agents.test_agent_service import fence
    from Tests.Agents.test_fleet_continuation import (
        RESUME_CFG,
        _await_retained,
        _finished_child,
    )
    from Tests.Agents.test_fleet_runtime import make_fleet_service
    from tldw_chatbook.Agents.agent_models import (
        SEND_TO_AGENT_TOOL_NAME,
        SPAWN_TOOL_NAME,
        WAIT_AGENTS_TOOL_NAME,
    )
    from tldw_chatbook.Agents.agent_service import build_first_request_schema_plan
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Plugins.agent_presets import agent_definition

    pin_agent_settings(monkeypatch, max_live_subagents=3)
    rig = native_console
    package = native_package(extension={"version": 1, "agents": ["agent.md"]})
    (package / "agent.md").write_text(
        "---\nname: reviewer\ndescription: Review\ntools: []\n---\nOnly the attributed native instruction."
    )
    install = await install_components(rig.service, package, ("agent:reviewer",))
    snapshot = await rig.service.capture_mcp_snapshot(
        install.installation_id, "workspace-a", "native-agent-test"
    )
    db = AgentRunsDB(tmp_path / "native-runs.sqlite", client_id="native-agent")
    holder = {}
    service, chat, fleet = make_fleet_service(db, [], {})
    definition = agent_definition(snapshot, "agent:reviewer", frozenset(RESUME_CFG.allowed_tools),
                                  registry=service.registry, parent_provider="llama_cpp")
    task_key = str(definition.instructions)
    # Use the committed fleet transport double and the real retained-child path.
    service, chat, fleet = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "Inspect it.", "agent": definition.name}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "first answer",
            lambda: fence(
                SEND_TO_AGENT_TOOL_NAME, {"id": holder["id"], "message": "Continue."}
            ),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "second answer",
        ],
        {
            task_key: [
                fence("calculator", {"expression": "1+1"}),
                "child answer",
                "resumed answer",
            ]
        },
    )

    def run():
        plan = build_first_request_schema_plan(
            service.registry,
            RESUME_CFG.allowed_tools,
            RESUME_CFG,
            "llama_cpp",
            [{"role": "user", "content": "go"}],
            skill_file_enabled=False,
            install_skill_enabled=False,
            run_skill_script_enabled=False,
            run_log_active=False,
            agent_definitions=(definition,),
            fleet_active=True,
            fleet_max_live=3,
        )
        return service.run_turn(
            conversation_id="native-conversation",
            messages=[{"role": "user", "content": "go"}],
            config=RESUME_CFG,
            api_endpoint="llama_cpp",
            first_request_schema_plan=plan,
        )

    try:
        _, first = await asyncio.to_thread(run)
        assert first.status == "done"
        child = _finished_child(fleet)
        holder["id"] = child.handle_id
        await asyncio.to_thread(_await_retained, fleet, child.handle_id)
        _, second = await asyncio.to_thread(run)
        assert second.status == "done"
        calls = chat.child_calls[task_key]
        assert len(calls) == 3
        assert "not permitted" in str(
            calls[1]["messages_payload"]
        ) or "not allowed" in str(calls[1]["messages_payload"])
        for call in calls:
            rows = call["messages_payload"]
            assert "Only the attributed native instruction." not in rows[0]["content"]
            assert (
                sum(
                    "Only the attributed native instruction." in str(row.get("content"))
                    for row in rows
                    if row["role"] == "user"
                )
                == 1
            )
            assert all(
                tool.get("function", {}).get("name") != "calculator"
                for tool in call.get("tools", ())
            )
        children = [
            row
            for row in db.list_runs("native-conversation", include_superseded=True)
            if row["agent_kind"] == "subagent"
        ]
        assert len(children) == 2 and any(
            row["resumed_from_run_id"] for row in children
        )
        assert all(row["definition_fingerprint"] for row in children)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_owned_hook_registration_failure_settles_unlaunched_reservation(
    native_console, native_package, monkeypatch
):
    import json

    from Tests.Agents.test_hooks_v2_execution import event
    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Plugins.hooks import NativeHooks

    rig = native_console
    package = native_package(extension={"version": 1, "hooks": "hooks.json"})
    (package / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "work",
                        "type": "command",
                        "event": "PreToolUse",
                        "effects": ["deny"],
                        "argv": child_argv(
                            'print(\'{"version":2,"decision":"pass"}\')'
                        ),
                    }
                ],
            }
        )
    )
    install = await install_components(rig.service, package, ("hook:work",))
    data_review = await rig.service.review_data_creation(install.installation_id, workspace_id=None)
    await rig.service.create_data(data_review, data_review.operation_id)
    snapshot = await rig.service.capture_mcp_snapshot(
        install.installation_id, "workspace-a", "registration-failure"
    )
    owner = NativeHooks(rig.service, (snapshot,))
    engine = HookEngine(
        owner.definitions,
        owner.authority,
        HookBudgetOwner(),
        process_owner_for=owner.process_owner,
        event_projector=owner.project_event,
    )
    usage = rig.service._coordinator.root_usage

    def refuse(*_args):
        raise PermissionError("registration failed")

    monkeypatch.setattr(usage, "retain_cancel", refuse)
    try:
        outcome = await engine.fire_async(event())
        assert not outcome.succeeded
        assert not rig.service.live_runs() and not engine.processes.records
        assert not await rig.service._call(lambda: usage.pending)
        assert await rig.service._call(lambda: rig.service._coordinator.owner.list_processes(limit=10, offset=0))
        assert all(
            row["state"] == "settled"
            for row in await rig.service._call(
                lambda: rig.service._coordinator.owner.list_processes(
                    limit=10, offset=0
                )
            )
        )
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_combined_hooks_keep_distinct_installation_revision_process_owners(
    native_console, native_package, monkeypatch
):
    import asyncio
    import json
    import threading

    from Tests.Agents.test_hooks_v2_execution import event
    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Plugins.hooks import _NativeHookProcessOwner, NativeHooks

    service = native_console.service
    snapshots, roots = [], {}
    code = "import json,sys,time;from pathlib import Path;e=json.load(sys.stdin);root=Path('${PLUGIN_DATA}');(root/e['runtime_session_id']).write_text('running');exec(\"while not (root/'release').exists(): time.sleep(.01)\");print('{\"version\":2,\"decision\":\"pass\"}')"
    for name in ("a", "b"):
        package = native_package(extension={"version": 1, "hooks": "hooks.json"})
        manifest = json.loads((package / "plugin.json").read_text())
        manifest["name"] = "owned-hook-" + name
        (package / "plugin.json").write_text(json.dumps(manifest))
        (package / "hooks.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "hooks": [
                        {
                            "id": "work",
                            "type": "command",
                            "event": "PreToolUse",
                            "effects": [],
                            "argv": child_argv(code),
                            "timeout_seconds": 20,
                        }
                    ],
                }
            )
        )
        installed = await install_components(service, package, ("hook:work",))
        review = await service.review_data_creation(
            installed.installation_id, workspace_id=None
        )
        roots[installed.installation_id] = await service.create_data(
            review, review.operation_id
        )
        snapshots.append(
            await service.capture_mcp_snapshot(
                installed.installation_id, "workspace-a", "pending-combined"
            )
        )
    a, b = snapshots
    assert (
        a.installation_id != b.installation_id
        and a.revision_digest != b.revision_digest
    )
    owner = NativeHooks(service, snapshots)
    assert len({handler.id for handler in owner.definitions}) == 2
    engine = HookEngine(
        owner.definitions,
        owner.authority,
        HookBudgetOwner(),
        process_owner_for=owner.process_owner,
        event_projector=owner.project_event,
    )
    entered, release_publish = threading.Event(), threading.Event()
    publish = _NativeHookProcessOwner.publish_process

    def delayed(self, token, provenance):
        if self.snapshot.installation_id == a.installation_id:
            entered.set()
            assert release_publish.wait(10)
        return publish(self, token, provenance)

    monkeypatch.setattr(_NativeHookProcessOwner, "publish_process", delayed)
    shared_event = event(runtime="combined")
    assert engine.notify(shared_event)
    try:
        for _ in range(500):
            if (
                entered.is_set()
                and (roots[b.installation_id].path / "combined").exists()
            ):
                break
            await asyncio.sleep(0.01)
        assert (
            entered.is_set() and (roots[b.installation_id].path / "combined").exists()
        )
        jobs = {
            job.owner.snapshot.installation_id: job
            for job in engine.processes.records.values()
        }
        assert set(jobs) == {a.installation_id, b.installation_id}
        original_owners = {key: job.owner for key, job in jobs.items()}
        assert {
            (row.installation_id, row.revision_digest) for row in service.live_runs()
        } == {
            (snapshot.installation_id, snapshot.revision_digest)
            for snapshot in snapshots
        }
        process_rows = await service._call(
            lambda: service._coordinator.owner.list_processes(limit=20, offset=0)
        )
        active = {
            row["installation_id"]: row
            for row in process_rows
            if row["state"] != "settled"
        }
        assert set(active) == set(jobs)
        assert {row["operation_id"] for row in active.values()} == {
            "hook:" + shared_event.event_id + ":hook:work"
        }
        for snapshot in snapshots:
            row = active[snapshot.installation_id]
            assert row["revision_digest"] == snapshot.revision_digest
            assert "plugin:" + row["token"] == jobs[snapshot.installation_id].token
            blockers = await service._call(
                lambda snapshot=snapshot: service._coordinator.root_usage.blockers(
                    (roots[snapshot.installation_id],)
                )
            )
            assert any(
                blocker.get("owner_token") == row["token"] for blocker in blockers
            )
        assert await service._call(lambda: service._coordinator.root_usage.dirty)
        disabled = await service.review_activation(
            a.installation_id, workspace_id="workspace-a", intent="disabled"
        )
        committed = asyncio.create_task(service.commit(disabled, disabled.operation_id))
        for _ in range(100):
            if jobs[a.installation_id].cancel.is_set():
                break
            await asyncio.sleep(0.01)
        assert jobs[a.installation_id].cancel.is_set()
        assert (
            not jobs[b.installation_id].cancel.is_set()
            and not jobs[b.installation_id].task.done()
        )
        assert not (roots[a.installation_id].path / "combined").exists()
        release_publish.set()
        assert (await asyncio.wait_for(jobs[a.installation_id].task, 10)).failure
        assert (await asyncio.wait_for(committed, 10)).committed
        assert {row.installation_id for row in service.live_runs()} == {
            b.installation_id
        }
        assert not await service._call(
            lambda: service._coordinator.root_usage.blockers(
                (roots[a.installation_id],)
            )
        )
        assert await service._call(
            lambda: service._coordinator.root_usage.blockers(
                (roots[b.installation_id],)
            )
        )
        (roots[b.installation_id].path / "release").touch()
        result = await asyncio.wait_for(jobs[b.installation_id].task, 10)
        assert result.failure is None and result.result is not None
        for _ in range(100):
            if not engine.cleanup_pending:
                break
            await asyncio.sleep(0.01)
        assert not engine.cleanup_pending and not service.live_runs()
        assert all(job.owner is original_owners[key] for key, job in jobs.items())
        assert all(
            row["state"] == "settled"
            for row in await service._call(
                lambda: service._coordinator.owner.list_processes(limit=20, offset=0)
            )
        )
        assert not await service._call(
            lambda: service._coordinator.root_usage.blockers(tuple(roots.values()))
        )
    finally:
        release_publish.set()
        for root in roots.values():
            (root.path / "release").touch()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_connection_case", ["native_mixed"], indirect=True)
async def test_mixed_builtin_and_mcp_aliases_retain_exact_owner_forms(
    shared_connection_case, monkeypatch
):
    """Different labels coexist; the same label cannot compete across forms."""
    import json
    from dataclasses import replace

    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
    from tldw_chatbook.Plugins.agent_presets import agent_definition
    from tldw_chatbook.Plugins.host_references import (
        capture_reference, mapping_id, tool_constraint_id,
    )

    case = shared_connection_case
    case.release.touch()
    assert (await case.pending).ok
    review = await case.service.review_configuration(
        case.installation,
        connections={"mcp:shared": case.profile_id},
        tools={"mcp:shared": ("echo",)},
        tool_references={"agent:reader": {"arithmetic": "builtin:calculator"}},
        component_tools={"agent:reader": {"search": "mcp:shared::echo"}},
    )
    assert (await case.service.commit(review, review.operation_id)).committed
    snapshot = await case.service.capture_mcp_snapshot(case.installation, "a", "mixed-aliases")
    mappings = json.loads(snapshot.mappings_json)
    builtin = next(row for row in mappings if row["mapping_id"] == mapping_id("agent:reader", "arithmetic"))
    canonical = next(row for row in mappings if row["mapping_id"] == tool_constraint_id("agent:reader", "search"))
    assert builtin["kind"] == "tool" and canonical["kind"] == "configuration"
    expected = (await case.service._call(case.service._coordinator.published_snapshot))["mappings"]
    assert not any(row.phase == "recovery_required"
                   for row in await case.service._call(case.service._coordinator.recover))
    assert (await case.service._call(case.service._coordinator.published_snapshot))["mappings"] == expected
    admitted = await case.service.admit(case.service.capture_maximum("a"), "pending:mixed")
    assert any(row["plugin_component_id"] == "agent:reader" for row in admitted["available_skills"])
    eligible = frozenset({"calculator", *(row.name for row in case.registries["a"].list_catalog())})
    definition = agent_definition(snapshot, "agent:reader", eligible, parent_provider="llama_cpp")
    assert set(definition.tool_allowlist) == set(eligible)
    builtin_registry = ToolCatalogRegistry()
    builtin_registry.register_provider(BuiltinToolProvider())
    assert "calculator" in definition.tool_allowlist
    result = builtin_registry.invoke_by_name(
        "calculator", {"expression": "2+2"},
        expected_definition=builtin_registry.snapshot_for_hook("calculator"),
    )
    assert result.ok and "4" in result.content
    with pytest.raises(PermissionError, match="tool_unavailable"):
        agent_definition(snapshot, "agent:reader", eligible - {"calculator"}, parent_provider="llama_cpp")

    async def validate(row, rows):
        await case.service._call(lambda: case.service._coordinator.validate_mapping(row, snapshot.inspection, rows))

    target = next(row for row in mappings if row["mapping_id"] == canonical["target_reference"])
    legacy = capture_reference(case.installation, snapshot.inspection, "agent:reader", "tool",
                               target["target_reference"], label="search", mappings=mappings)
    legacy_rows = [legacy if row == canonical else row for row in mappings]
    await validate(legacy, legacy_rows)
    await validate(canonical, mappings)
    for row, rows in ((legacy, legacy_rows), (canonical, mappings), (builtin, mappings)):
        for field in ("definition_digest", "configuration_digest", "mapping_id", "target_reference", "revision_digest"):
            changed = {**row, field: "unknown-or-tampered"}
            with pytest.raises((PermissionError, ValueError, KeyError)):
                await validate(changed, [changed if candidate == row else candidate for candidate in rows])
    with pytest.raises(PermissionError, match="ambiguous"):
        await validate(canonical, [*mappings, legacy])
    with pytest.raises(PermissionError, match="ambiguous"):
        await validate(legacy, [*mappings, legacy])
    with pytest.raises((PermissionError, ValueError)):
        await case.service.review_configuration(
            case.installation,
            tool_references={"agent:reader": {"arithmetic": "builtin:calculator"}},
            component_tools={"agent:reader": {"arithmetic": "builtin:calculator"}},
        )
    original_schema = BuiltinToolProvider.load_schema
    with monkeypatch.context() as patch:
        patch.setattr(BuiltinToolProvider, "load_schema",
                      lambda self, tool_id: replace(original_schema(self, tool_id), description="changed schema"))
        with pytest.raises(PermissionError, match="reference_changed"):
            await validate(builtin, mappings)
        assert any(row.phase == "recovery_required"
                   for row in await case.service._call(case.service._coordinator.recover))
    assert not any(row.phase == "recovery_required"
                   for row in await case.service._call(case.service._coordinator.recover))
    await validate(builtin, mappings)


def test_historical_and_canonical_model_records_keep_distinct_exact_identities(
    plugin_stack, native_package, monkeypatch
):
    import json
    import tldw_chatbook.config as config
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import LocalLLMProviderCatalogService
    from tldw_chatbook.Plugins.host_references import capture_model_mapping, capture_reference, mapped_model
    from tldw_chatbook.Plugins.inspection import inspect_package

    settings = {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8080/v1/chat/completions"}}}
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(config, "get_cli_providers_and_models", lambda: {"llama_cpp": ["m"]})
    package = native_package(extension={"version": 1, "agents": ["agent.md"]})
    (package / "agent.md").write_text("---\nname: reader\ndescription: Read\ntools: []\nmodel: chosen\n---\nRead.")
    inspection = inspect_package(package)
    owner = LocalLLMProviderCatalogService(provider_catalog_loader=lambda: {"llama_cpp": ["m"]},
                                           settings_loader=lambda: settings, environ={})
    stack = plugin_stack
    stack.call(lambda: setattr(stack.coordinator, "model_mapping_owner", owner))
    legacy = capture_reference("installation", inspection, "agent:reader", "model", "llama_cpp::m")
    canonical = capture_model_mapping(owner, "installation", inspection, "agent:reader", "llama_cpp/m")
    assert legacy["mapping_id"] != canonical["mapping_id"]
    for row in (legacy, canonical):
        before = json.dumps(row, sort_keys=True)
        stack.call(lambda: stack.coordinator.validate_mapping(row, inspection, [row]))
        assert json.dumps(row, sort_keys=True) == before
        assert mapped_model(inspection.inventory["agent:reader"], [row]) == ("llama_cpp", "m")
        for field in ("mapping_id", "definition_digest", "configuration_digest", "target_reference", "revision_digest"):
            changed = {**row, field: "unknown-or-tampered"}
            with pytest.raises((PermissionError, ValueError, KeyError)):
                stack.call(lambda: stack.coordinator.validate_mapping(changed, inspection, [changed]))
        with pytest.raises(PermissionError, match="ambiguous"):
            stack.call(lambda: stack.coordinator.validate_mapping(row, inspection, [legacy, canonical]))


def test_direct_canonical_tool_constraint_uses_actual_registry_and_parent():
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
    from tldw_chatbook.Plugins.host_references import resolve_tool_ceiling

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    assert resolve_tool_ceiling(("builtin:calculator",), registry, frozenset({"calculator"})) == ("calculator",)
    assert resolve_tool_ceiling((), registry, frozenset({"calculator"})) == ()
    assert resolve_tool_ceiling(None, registry, frozenset({"calculator"})) == ("calculator",)
    for wanted, parent in ((("builtin:calculator",), frozenset()),
                           (("builtin:missing",), frozenset({"calculator"})),
                           (("Calculator display title",), frozenset({"calculator"}))):
        with pytest.raises(PermissionError, match="tool_unavailable"):
            resolve_tool_ceiling(wanted, registry, parent)
