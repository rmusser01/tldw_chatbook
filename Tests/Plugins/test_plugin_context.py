"""Native material reaches the real Console payload as attributed user context."""

import pytest

from Tests.Plugins.test_owned_mcp_tools import (
    shared_connection_case as _shared_connection_case,
)

shared_connection_case = _shared_connection_case
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.Plugins.test_native_components import install_components


@pytest.mark.asyncio
async def test_always_rule_and_manual_command_reach_final_provider(
    native_console, native_package
):
    package = native_package(
        extension={"version": 1, "commands": ["command.md"], "rules": ["rule.md"]}
    )
    (package / "command.md").write_text(
        "---\nname: review\ndescription: Review\narguments: [query]\n---\nReview carefully."
    )
    (package / "rule.md").write_text("---\nname: style\nmode: always\n---\nBe precise.")
    rig = native_console
    await install_components(rig.service, package, ("command:review", "rule:style"))
    maximum = rig.service.capture_maximum("workspace-a")
    command = next(
        row for row in maximum["available_skills"] if row["plugin_kind"] == "command"
    )
    literal = "$other /another @agent"
    result = await rig.controller.submit_draft(
        "/" + command["name"].replace(":command:", ":", 1) + " " + literal, session_id=rig.session.id
    )
    assert result.accepted
    payload = rig.gateway.payloads[-1]
    assert all(
        "Review carefully" not in row.get("content", "")
        for row in payload
        if row["role"] == "system"
    )
    text = "\n".join(row.get("content", "") for row in payload if row["role"] == "user")
    assert text.index("Be precise.") < text.index("Review carefully.")
    assert literal in text
    assert "Literal invocation arguments" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rule_size,expected", [(20, True), (6500, True), (7400, False)]
)
async def test_owned_hook_and_plugin_rule_share_final_input(
    native_console, native_package, monkeypatch, rule_size, expected
):
    import json

    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    deliveries = []
    original = HookEngine._deliver

    async def observe(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        deliveries.append(result)
        return result

    monkeypatch.setattr(HookEngine, "_deliver", observe)
    rig = native_console
    package = native_package(
        extension={
            "version": 1,
            "rules": [f"rule{i}.md" for i in range(4)],
            "hooks": "hooks.json",
        }
    )
    for i in range(4):
        (package / f"rule{i}.md").write_text(
            f"---\nname: style{i}\nmode: always\n---\nNative instruction."
            + "x" * rule_size
        )
    output = json.dumps(
        {
            "version": 2,
            "decision": "pass",
            "context": [
                {"text": "Separate hook instruction." + "y" * 3300, "lifetime": "turn"}
            ],
        }
    )
    (package / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "prepare",
                        "event": "UserPromptSubmit",
                        "type": "command",
                        "effects": ["context"],
                        "argv": child_argv(f"print({output!r})"),
                    }
                ],
            }
        )
    )
    await install_components(
        rig.service,
        package,
        tuple(f"rule:style{i}" for i in range(4)) + ("hook:prepare",),
    )
    runtime = ConsoleRuntime(app=rig.controller.app)
    rig.controller.app.console_runtime = runtime
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    try:
        result = await rig.controller.submit_draft(
            "Explain.", session_id=rig.session.id
        )
        assert result.accepted is expected
        assert deliveries and all(row.succeeded for row in deliveries), deliveries
        if not expected:
            assert not rig.gateway.payloads
            return
        payload = rig.gateway.payloads[-1]
        assert any("Native instruction." in str(row.get("content")) for row in payload)
        assert any(
            "Separate hook instruction." in str(row.get("content")) for row in payload
        )
        assert all(
            "instruction." not in str(row.get("content"))
            for row in payload
            if row["role"] == "system"
        )
        assert not rig.service.live_runs()
    finally:
        await runtime.close_hooks_v2()


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_connection_case", ["native_hook"], indirect=True)
async def test_actual_console_composes_owned_mcp_hook_without_automatic_connection(
    shared_connection_case, native_console, monkeypatch
):
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

    case, rig = shared_connection_case, native_console
    case.release.touch()
    assert (await case.pending).ok
    rig.registry.create_workspace(workspace_id="a", name="MCP A")
    session = rig.store.create_session(workspace_id="a")
    local = LocalSkillsService(
        store_dir=case.package.parent / "skills",
        plugin_service=case.service,
        allow_untrusted_without_trust_service=True,
    )
    rig.controller._skills_service = SkillsScopeService(local_service=local)
    rig.controller.app.local_skills_service = local
    rig.controller.app.skills_scope_service = rig.controller._skills_service
    rig.controller.app.local_mcp_control_service = case.local
    rig.controller.app.unified_mcp_service = case.unified
    monkeypatch.setattr(
        rig.controller,
        "request_mcp_approvals",
        lambda calls, **kw: {row.llm_name: "approve_once" for row in calls},
    )
    runtime = ConsoleRuntime(app=rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    observed, explicit_setup = [], False
    compose = rig.controller._compose_native_mcp_providers

    async def capture(*args, **kwargs):
        providers = await compose(*args, **kwargs)
        observed.extend(providers.providers)
        if explicit_setup:
            for provider in providers.providers:
                if hasattr(provider, "snapshot"):
                    await provider.connect("mcp:shared")
        return providers

    monkeypatch.setattr(rig.controller, "_compose_native_mcp_providers", capture)
    before = len(case.calls.read_text().splitlines())
    try:
        refused = await rig.controller.submit_draft("hello", session_id=session.id)
        assert not refused.accepted and not rig.gateway.payloads
        assert len(case.calls.read_text().splitlines()) == before
        assert any(
            getattr(row, "ownership", None) is case.ownership for row in observed
        )
        explicit_setup = True
        sent = await rig.controller.submit_draft("hello again", session_id=session.id)
        assert sent.accepted
        assert any(
            "Owned native MCP context." in str(row.get("content"))
            for row in rig.gateway.payloads[-1]
        )
        assert len(case.calls.read_text().splitlines()) == before + 1
        assert case.ownership.is_connected(case.profile_id, "run-b:b")
    finally:
        await runtime.close_hooks_v2()


@pytest.mark.asyncio
async def test_failed_native_dependency_blocks_only_using_component(
    native_console, native_package
):
    import json

    from Tests.hooks_v2_process_support import child_argv
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    rig = native_console
    package = native_package(
        extension={
            "version": 1,
            "commands": ["dependent.md", "independent.md"],
            "hooks": "hooks.json",
            "requires": {"command:dependent": ["hook:check"]},
        }
    )
    for name in ("dependent", "independent"):
        (package / f"{name}.md").write_text(
            f"---\nname: {name}\ndescription: Run\n---\n{name} instruction"
        )
    (package / "hooks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "hooks": [
                    {
                        "id": "check",
                        "event": "UserPromptSubmit",
                        "type": "command",
                        "effects": [],
                        "argv": child_argv("raise SystemExit(1)"),
                    }
                ],
            }
        )
    )
    install = await install_components(
        rig.service, package, ("command:dependent", "command:independent", "hook:check")
    )
    runtime = ConsoleRuntime(app=rig.controller.app)
    runtime.set_chat_store(rig.store)
    runtime.set_chat_controller(rig.controller)
    try:
        blocked = await rig.controller.submit_draft(
            f"/{install.alias}:dependent", session_id=rig.session.id
        )
        assert not blocked.accepted and not rig.gateway.payloads
        sent = await rig.controller.submit_draft(
            f"/{install.alias}:independent", session_id=rig.session.id
        )
        assert sent.accepted
        assert "independent instruction" in str(rig.gateway.payloads[-1])
    finally:
        await runtime.close_hooks_v2()


def test_production_app_plugin_factory_uses_same_lazy_owners(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import tldw_chatbook.app as app_module
    import tldw_chatbook.app_service_wiring as wiring_module
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore

    local = LocalMCPControlService(
        store=LocalMCPStore(tmp_path / "unused-mcp.json"), manifest_provider=dict
    )
    model = LocalLLMProviderCatalogService(provider_catalog_loader=dict)
    app = SimpleNamespace(
        _plugin_service=None,
        local_mcp_control_service=local,
        local_llm_provider_catalog_service=model,
        workspace_registry_service=SimpleNamespace(get_workspace=lambda _: None),
    )
    monkeypatch.setattr(wiring_module, "get_user_data_dir", lambda: tmp_path / "unstarted")
    service = app_module.TldwCli._build_plugin_service(app)
    assert service is app_module.TldwCli._build_plugin_service(app)
    assert service._mcp_mapping_owner is local and service._model_mapping_owner is model
    assert not (tmp_path / "unstarted").exists()


@pytest.mark.asyncio
async def test_manual_fork_skill_model_reaches_real_provider(
    native_console, native_package, monkeypatch
):
    from Tests.console_provider_doubles import provider_resolution
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.Plugins.skill_provider import PluginSkillProvider

    rig = native_console
    owner = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {"llama_cpp": ["mapped"]},
        settings_loader=lambda: {
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:8080/v1/chat/completions"}
            }
        },
        environ={},
    )
    await rig.service._call(
        lambda: setattr(rig.service._coordinator, "model_mapping_owner", owner)
    )
    package = native_package()
    (package / "skills/review/SKILL.md").write_text(
        '---\nname: review\ndescription: Review\nallowed-tools: ""\nmetadata:\n  context: "fork"\n  model: "preferred"\n  disable_model_invocation: "true"\n---\nMapped fork instruction.'
    )
    install = await install_components(rig.service, package, ("skill:review",))
    configuration = await rig.service.review_configuration(
        install.installation_id, models={"skill:review": "llama_cpp/mapped"}
    )
    await rig.service.commit(configuration, configuration.operation_id)
    assert not PluginSkillProvider(
        rig.service.capture_maximum("workspace-a")["available_skills"]
    ).list_catalog()
    observed = []

    async def resolve(selection):
        observed.append(selection.explicit_model)
        return provider_resolution(provider="llama_cpp", model=selection.explicit_model)

    monkeypatch.setattr(rig.gateway, "resolve_for_send", resolve)
    from tldw_chatbook.Chat.console_chat_store import ConsoleMessageRole

    rig.store.append_message(
        rig.session.id,
        role=ConsoleMessageRole.USER,
        content="Old history must not ride the fork.",
    )
    result = await rig.controller.submit_draft(
        f"${install.alias}:review literal $other /command", session_id=rig.session.id
    )
    assert result.accepted
    assert observed[-1] == "mapped"
    payload = rig.gateway.payloads[-1]
    assert "Mapped fork instruction." in str(payload)
    assert "literal $other /command" in str(payload)
    assert "Old history" not in str(payload)
    assert all(
        "Mapped fork instruction." not in str(row.get("content"))
        for row in payload
        if row["role"] == "system"
    )


@pytest.mark.asyncio
async def test_embedded_native_skill_survives_reference_expansion(
    native_console, monkeypatch
):
    import tldw_chatbook.Chat.console_references as references

    rig = native_console
    install = await rig.install()
    monkeypatch.setattr(
        references,
        "build_console_reference_resolver",
        lambda: lambda token: ("file", "REFERENCE_BODY", None),
    )
    result = await rig.controller.submit_draft(
        f"Use ${install.alias}:review and @evidence.txt", session_id=rig.session.id
    )
    assert result.accepted
    payload = rig.gateway.payloads[-1]
    assert "REFERENCE_BODY" in str(payload)
    assert "Check the change and explain actionable findings." in str(payload)


@pytest.mark.asyncio
async def test_reference_body_cannot_invoke_native_skill(native_console, monkeypatch):
    import tldw_chatbook.Chat.console_references as references

    rig = native_console
    install = await rig.install()
    generated = (
        f"${install.alias}:review /{install.alias}:command @{install.alias}:agent"
    )
    monkeypatch.setattr(
        references,
        "build_console_reference_resolver",
        lambda: lambda token: ("file", generated, None),
    )
    result = await rig.controller.submit_draft(
        "Read @evidence.txt", session_id=rig.session.id
    )
    assert result.accepted
    payload = str(rig.gateway.payloads[-1])
    assert generated in payload
    assert "Check the change and explain actionable findings." not in payload
    result = await rig.controller.submit_draft(
        f"Read @evidence.txt then ${install.alias}:review", session_id=rig.session.id
    )
    assert result.accepted
    payload = str(rig.gateway.payloads[-1])
    assert generated in payload
    assert payload.count("Check the change and explain actionable findings.") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["staged", "recapture"])
@pytest.mark.parametrize("change", [False, True])
async def test_native_final_provider_rechecks_staged_authority(
    native_console, native_package, monkeypatch, boundary, change
):
    import asyncio
    import threading

    rig = native_console
    package = native_package(extension={"version": 1, "rules": ["rule.md"]})
    (package / "rule.md").write_text(
        "---\nname: style\nmode: always\n---\nRequired native rule."
    )
    install = await install_components(rig.service, package, ("rule:style",))
    snapshot = await rig.service.capture_mcp_snapshot(
        install.installation_id, "workspace-a", "path-control"
    )
    from pathlib import Path

    retained = Path(snapshot.inspection.materialized_identity) / "rule.md"
    original_body = retained.read_text()
    original_mode = retained.stat().st_mode
    retained.chmod(0o600)
    entered, release = threading.Event(), threading.Event()
    original_wait = rig.controller._wait_for_trace_maintenance_dispatch
    original_checked = rig.service._checked
    armed = False

    async def wait():
        nonlocal armed
        assert rig.service.live_runs()
        armed = True
        if boundary == "staged" and change:
            retained.write_text(original_body + "changed after staging")
        await original_wait()

    def checked(*args, **kwargs):
        nonlocal armed
        if armed and boundary == "recapture":
            armed = False
            entered.set()
            assert release.wait(10)
        return original_checked(*args, **kwargs)

    monkeypatch.setattr(rig.controller, "_wait_for_trace_maintenance_dispatch", wait)
    monkeypatch.setattr(rig.service, "_checked", checked)
    pending = asyncio.create_task(
        rig.controller.submit_draft("Explain", session_id=rig.session.id)
    )
    try:
        if boundary == "recapture":
            assert await asyncio.to_thread(entered.wait, 10)
            assert not rig.gateway.payloads
            if change:
                retained.write_text(original_body + "changed during recapture")
            release.set()
        result = await asyncio.wait_for(pending, 15)
        if change:
            assert not rig.gateway.payloads
        else:
            assert result.accepted and "Required native rule." in str(
                rig.gateway.payloads[-1]
            )
    finally:
        release.set()
        retained.write_text(original_body)
        retained.chmod(original_mode)
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("disable", [False, True])
async def test_reference_invocation_source_survives_real_preparation_resume(
    native_console, monkeypatch, disable
):
    from types import SimpleNamespace

    import tldw_chatbook.Chat.console_chat_controller as controller_module
    import tldw_chatbook.Chat.console_references as references
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail

    rig = native_console
    install = await rig.install()
    session = rig.store.create_session(ephemeral=True, workspace_id="workspace-a")
    monkeypatch.setattr(rig.gateway, "supports_durable_capture", True, raising=False)
    monkeypatch.setattr(
        controller_module,
        "runtime_capture_policy",
        lambda: SimpleNamespace(enabled=True, detail=CaptureDetail.SAFE, generation=7),
    )
    policy = rig.controller.capture_policy_snapshot(session.id)
    rig.controller.set_next_trace_privacy(
        session.id,
        capture_enabled=True,
        pii_redaction_enabled=True,
        expected_policy_revision=policy.policy_revision,
    )
    generated = f"REFERENCE ${install.alias}:review"
    reads = []

    def resolve(token):
        reads.append(token)
        return "file", generated, None

    monkeypatch.setattr(references, "build_console_reference_resolver", lambda: resolve)
    result = await rig.controller.submit_draft(
        f"Read @evidence.txt then ${install.alias}:review", session_id=session.id
    )
    preparation = rig.store.preparation_for_session(session.id)
    assert not result.accepted and preparation is not None
    assert not rig.gateway.payloads
    if disable:
        review = await rig.service.review_activation(
            install.installation_id, workspace_id="workspace-a", intent="disabled"
        )
        await rig.service.commit(review, review.operation_id)
    resumed = await rig.controller.send_without_capture(preparation.preparation_id)
    if disable:
        assert not resumed.accepted and not rig.gateway.payloads
        assert reads == ["evidence.txt"]
        return
    assert resumed.accepted
    payload = str(rig.gateway.payloads[-1])
    assert generated in payload
    assert payload.count("Check the change and explain actionable findings.") == 1
    assert reads == ["evidence.txt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("reference", [False, True])
async def test_manual_rule_preserves_expanded_remainder_on_initial_and_resumed_send(
    native_console, native_package, monkeypatch, paused, reference
):
    from types import SimpleNamespace

    import tldw_chatbook.Chat.console_chat_controller as controller_module
    import tldw_chatbook.Chat.console_references as references
    from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail

    rig = native_console
    package = native_package(extension={"version": 1, "rules": ["rule.md", "other.md"]})
    (package / "rule.md").write_text(
        "---\nname: style\nmode: manual\n---\nMANUAL STYLE BODY."
    )
    (package / "other.md").write_text(
        "---\nname: other\nmode: manual\n---\nUNSELECTED RULE BODY."
    )
    install = await install_components(
        rig.service, package, ("rule:style", "rule:other", "skill:review")
    )
    generated = f"EVIDENCE @{install.alias}:other ${install.alias}:review"
    reads = []

    def resolve(token):
        reads.append(token)
        if token != "evidence.txt":
            return None
        return "file", generated, None

    monkeypatch.setattr(references, "build_console_reference_resolver", lambda: resolve)
    session = rig.store.create_session(ephemeral=paused, workspace_id="workspace-a")
    if paused:
        monkeypatch.setattr(
            rig.gateway, "supports_durable_capture", True, raising=False
        )
        monkeypatch.setattr(
            controller_module,
            "runtime_capture_policy",
            lambda: SimpleNamespace(
                enabled=True, detail=CaptureDetail.SAFE, generation=7
            ),
        )
        policy = rig.controller.capture_policy_snapshot(session.id)
        rig.controller.set_next_trace_privacy(
            session.id,
            capture_enabled=True,
            pii_redaction_enabled=True,
            expected_policy_revision=policy.policy_revision,
        )
    remainder = "compare @evidence.txt" if reference else "compare literal evidence"
    sent = await rig.controller.submit_draft(
        f"@{install.alias}:style {remainder}", session_id=session.id
    )
    if paused:
        preparation = rig.store.preparation_for_session(session.id)
        assert not sent.accepted and preparation is not None
        assert not rig.gateway.payloads
        sent = await rig.controller.send_without_capture(preparation.preparation_id)
    assert sent.accepted
    payload = rig.gateway.payloads[-1]
    text = "\n".join(
        str(row.get("content", "")) for row in payload if row["role"] == "user"
    )
    assert text.count("MANUAL STYLE BODY.") == 1
    assert "UNSELECTED RULE BODY." not in text
    assert "Check the change and explain actionable findings." not in text
    assert remainder in text
    if reference:
        assert generated in text
    assert reads == (["evidence.txt"] if reference else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling,arguments,accepted", [
    ("legacy", '  {"literal": "$other @file /other"}', True),
    ("legacy", "", True),
    ("current", '{"query":"strict named value"}', True),
    ("current", 'opaque remainder', False),
    ("current", '{"other":"wrong key"}', False),
    ("unknown", 'opaque remainder', False),
    ("wrong_kind", 'opaque remainder', False),
])
async def test_command_selector_keeps_closed_argument_contract(
    native_console, native_package, monkeypatch, spelling, arguments, accepted
):
    import json
    import tldw_chatbook.Chat.console_references as references

    package = native_package(extension={"version": 1, "commands": ["command.md"]})
    (package / "command.md").write_text(
        "---\nname: review\ndescription: Review\narguments: [query]\n---\nCLOSED COMMAND BODY."
    )
    rig = native_console
    installed = await install_components(rig.service, package, ("command:review",))
    calls, reads = [], []
    execute = rig.service.execute_skill

    async def observe(name, **kwargs):
        calls.append(kwargs.get("args", ""))
        return await execute(name, **kwargs)

    monkeypatch.setattr(rig.service, "execute_skill", observe)
    monkeypatch.setattr(references, "build_console_reference_resolver",
                        lambda: lambda token: reads.append(token))
    selector = {
        "legacy": f"/{installed.alias}:review",
        "current": f"${installed.alias}:command:review",
        "unknown": f"/{installed.alias}:missing",
        "wrong_kind": f"@{installed.alias}:review",
    }[spelling]
    result = await rig.controller.submit_draft(selector + " " + arguments, session_id=rig.session.id)
    assert result.accepted is accepted
    assert not reads
    if not accepted:
        assert not rig.gateway.payloads
        return
    rows = rig.gateway.payloads[-1]
    text = "\n".join(str(row.get("content", "")) for row in rows if row["role"] == "user")
    assert text.count("CLOSED COMMAND BODY.") == 1
    expected = {"query": arguments} if spelling == "legacy" else json.loads(arguments)
    assert json.loads(calls[-1]) == expected


@pytest.mark.parametrize("rows", [[], ["command"], ["agent"], ["rule", "rule"]])
def test_legacy_rule_selector_refuses_unknown_other_kind_and_ambiguous_rows(rows):
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    entries = [{"name": f"alias:{kind}:same", "plugin_owned": True,
                "plugin_kind": kind, "plugin_rule_mode": "manual"} for kind in rows]
    with pytest.raises(PermissionError, match="selector_unavailable|agent_child_only"):
        ConsoleChatController._native_selector("@alias:same untouched", entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("declared,arguments,accepted", [
    ([], "", True),
    (["a", "b"], '{"a":"one","b":"two"}', True),
    (["a", "b"], '{"a":"one"}', False),
    (["a", "b"], '{"a":"one","b":3}', False),
    (["a", "b"], '{"a":"one","a":"duplicate","b":"two"}', False),
])
async def test_zero_and_multiple_command_arguments_share_strict_renderer(
    native_console, native_package, legacy, declared, arguments, accepted
):
    import json

    rig = native_console
    package = native_package(extension={"version": 1, "commands": ["command.md"]})
    (package / "command.md").write_text(
        "---\nname: review\ndescription: Review\narguments: "
        + json.dumps(declared) + "\n---\nSTRICT COMMAND BODY."
    )
    installed = await install_components(rig.service, package, ("command:review",))
    selector = f"/{installed.alias}:review" if legacy else f"${installed.alias}:command:review"
    result = await rig.controller.submit_draft(selector + " " + arguments, session_id=rig.session.id)
    assert result.accepted is accepted
    if accepted:
        assert "STRICT COMMAND BODY." in str(rig.gateway.payloads[-1])
    else:
        assert not rig.gateway.payloads
