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
    from tldw_chatbook.config import get_cli_providers_and_models

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
    provider, models = next(
        (provider, models)
        for provider, models in get_cli_providers_and_models().items()
        if models
    )
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
    assert mapped_tools("installed", component, mappings) == (
        owned_mcp_tool_name("installed", "mcp:shared", "echo"),
    )
