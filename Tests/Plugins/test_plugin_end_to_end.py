"""Controlled authored example through reviewed owners and the real Console."""

import hashlib
import json
import os
import shlex
import shutil
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.Plugins.platform_support import package_capture


@pytest.fixture(
    params=(
        "local",
        pytest.param("https_git_catalog", marks=pytest.mark.loopback_network),
    )
)
async def end_to_end_case(native_console, tmp_path, monkeypatch, request):
    from Tests.hooks_v2_process_support import child_argv
    from Tests.Plugins.acquisition_process_support import git, https_git, repository
    from tldw_chatbook.Plugins import acquisition
    from tldw_chatbook.Plugins.git_source import parse_source_locator
    from tldw_chatbook.Plugins.source_service import SourceService

    rig = native_console
    package = tmp_path / "native-review"
    source = Path(__file__).parent / "fixtures" / "native_review"
    shutil.copytree(source, package)
    hashes = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    installed = None
    sources = SourceService(rig.service)
    catalog = None
    catalog_locator = None
    resources = ExitStack()
    request.addfinalizer(resources.close)
    for key in tuple(os.environ):
        if key.startswith(("GIT_", "SSH_")) or "proxy" in key.lower():
            monkeypatch.delenv(key)
    monkeypatch.setattr(acquisition, "HOST_GIT_SYSTEM", tmp_path / "host-git-system")
    monkeypatch.setattr(acquisition, "HOST_SSH_SYSTEM", tmp_path / "host-ssh-system")
    if request.param == "https_git_catalog":
        staging = tmp_path / "catalog-source"
        shutil.copytree(package, staging / "plugins/review")
        manifest = staging / ".agents/plugins/marketplace.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "plugins": [
                        {
                            "name": "review",
                            "source": {"source": "local", "path": "./plugins/review"},
                        }
                    ]
                }
            )
        )
        catalog = repository(staging, tmp_path / "controlled-git-catalog")
        package = catalog / "plugins/review"
        origin, ca, requests = resources.enter_context(https_git(tmp_path))
        helper_code = (
            "import sys;"
            "print('username=fixture\\npassword=private-credential-sentinel') "
            "if sys.argv[-1]=='get' else None"
        )
        helper = "!" + shlex.join(child_argv(helper_code))
        (Path(os.environ["HOME"]) / ".gitconfig").write_text(
            "[credential]\n helper = "
            + json.dumps(helper)
            + "\n useHttpPath = true\n[http]\n sslCAInfo = "
            + json.dumps(str(ca))
            + "\n"
        )
        catalog_locator = parse_source_locator(
            origin + "/controlled-git-catalog/.git#ref=refs/heads/main"
        )

    async def provider(resolution, messages, **kwargs):
        rig.gateway.payloads.append(messages)
        yield "review-complete"

    monkeypatch.setattr(rig.gateway, "stream_chat", provider)

    async def install_and_enable():
        nonlocal installed
        if catalog is None:
            installed = await rig.service.review_install(
                package,
                selection=("skill:review",),
                workspace_id="workspace-a",
                enable_in_workspace=True,
            )
        else:
            source_id = await sources.add(catalog_locator)
            assert not sources.list_sources()[0].cached
            assert not await rig.service.management_snapshot("workspace-a")
            refreshed = await sources.refresh(source_id)
            assert refreshed.refresh_error is None
            assert (
                refreshed.catalog_commit
                == git(catalog, "rev-parse", "HEAD").decode().strip()
            )
            assert requests and any(auth is not None for _, _, auth in requests)
            assert "private-credential-sentinel" not in repr(refreshed)
            assert len(refreshed.entries) == 1
            installed = await sources.review_entry(
                source_id,
                refreshed.entries[0].identity,
                selection=("skill:review",),
                workspace_id="workspace-a",
                enable_in_workspace=True,
            )
        assert set(installed.inspection.inventory) == {"skill:review"}
        assert (await rig.service.commit(installed, installed.operation_id)).committed
        assert not rig.service.capture_maximum("workspace-a")["available_skills"]
        trust = await rig.service.review_trust(installed.installation_id)
        assert (await rig.service.commit(trust, trust.operation_id)).committed
        assert rig.service.capture_maximum("workspace-a")["available_skills"]
        assert not rig.service.capture_maximum("workspace-b")["available_skills"]
        return installed

    async def invoke_review_skill(*, updated=False):
        entries = rig.service.capture_maximum("workspace-a")["available_skills"]
        assert len(entries) == 1
        sent = await rig.controller.submit_draft(
            f"${entries[0]['name']} Review the supplied acceptance criteria.",
            session_id=rig.session.id,
        )
        assert sent.accepted
        payload = rig.gateway.payloads[-1][-1]
        assert payload["role"] == "user"
        assert "Do not invent missing evidence" in payload["content"]
        assert ("Updated controlled checklist." in payload["content"]) is updated
        assert installed.installation_id in payload["content"]
        from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

        replies = [
            row.content
            for row in rig.store.messages_for_session(rig.session.id)
            if row.role is ConsoleMessageRole.ASSISTANT
        ]
        assert replies[-1] == "review-complete"
        return replies[-1]

    async def update_and_rollback():
        original = installed.inspection.effective_digest
        skill = package / "skills/review/SKILL.md"
        skill.write_text(skill.read_text() + "\nUpdated controlled checklist.\n")
        if catalog is None:
            update = await rig.service.review_revision(
                installed.installation_id, package
            )
        else:
            git(catalog, "add", ".")
            git(catalog, "commit", "-m", "controlled catalog update")
            candidate = (await sources.check_updates(installed.installation_id))[0]
            update = await sources.review_update(installed.installation_id, candidate)
        assert update.inspection.effective_digest != original
        assert (await rig.service.apply_revision(update, update.operation_id)).committed
        trust = await rig.service.review_trust(installed.installation_id)
        assert (await rig.service.commit(trust, trust.operation_id)).committed
        assert await invoke_review_skill(updated=True) == "review-complete"
        rollback = await rig.service.review_rollback(
            installed.installation_id, original
        )
        assert (
            await rig.service.apply_revision(rollback, rollback.operation_id)
        ).committed
        trust = await rig.service.review_trust(installed.installation_id)
        assert (await rig.service.commit(trust, trust.operation_id)).committed
        assert await invoke_review_skill() == "review-complete"

    async def disable_here():
        review = await rig.service.review_activation(
            installed.installation_id, workspace_id="workspace-a", intent="disabled"
        )
        assert (await rig.service.commit(review, review.operation_id)).committed

    def skill_advertised():
        return bool(rig.service.capture_maximum("workspace-a")["available_skills"])

    try:
        yield SimpleNamespace(
            rig=rig,
            package=package,
            install_and_enable=install_and_enable,
            invoke_review_skill=invoke_review_skill,
            update_and_rollback=update_and_rollback,
            disable_here=disable_here,
            skill_advertised=skill_advertised,
        )
    finally:
        assert hashes == {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source.rglob("*")
            if path.is_file()
        }


@pytest.mark.asyncio
async def test_install_use_disable_uses_production_paths(end_to_end_case):
    case = end_to_end_case
    installed = await case.install_and_enable()
    assert await case.invoke_review_skill() == "review-complete"
    await case.update_and_rollback()
    await case.disable_here()
    assert not case.skill_advertised()
    current = await case.rig.service.management_details(
        installed.installation_id, workspace_id="workspace-a"
    )
    assert current is not None
    removal = await case.rig.service.review_removal(
        installed.installation_id, uninstall=True
    )
    request = case.rig.service.begin_reviewed_removal(removal)
    assert (await case.rig.service.finish_revocation(request)).committed
    assert not await case.rig.service.management_snapshot("workspace-a")


@package_capture
def test_authored_example_has_independent_inventory_and_tool_ceiling():
    from tldw_chatbook.Plugins.inspection import inspect_package

    root = Path(__file__).parent / "fixtures" / "native_review"
    inspection = inspect_package(root)
    assert not inspection.rejected
    assert set(inspection.inventory) == {"skill:review"}
    assert inspection.inventory["skill:review"].kind == "skill"
    assert (
        json.loads(inspection.inventory["skill:review"].definition_json)[
            "allowed-tools"
        ]
        == ""
    )
    assert json.loads((root / "plugin.json").read_text())["version"] == "1.0.0"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["command", "rule", "mcp"])
@pytest.mark.parametrize("operation", ["execute", "read", "definition"])
async def test_skill_lookup_keeps_component_kind(
    native_console, native_package, tmp_path, kind, operation
):
    """Real admitted same-name components keep skill and native identities distinct."""
    import asyncio

    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import LocalMCPStore
    from tldw_chatbook.Plugins.admission import PluginUnavailable
    from tldw_chatbook.Plugins.recovery import retained_inspections

    service = native_console.service
    package = native_package(
        extension={
            "version": 1,
            **(
                {"commands": ["command.md"]}
                if kind == "command"
                else {"rules": ["rule.md"]} if kind == "rule" else {}
            ),
        }
    )
    (package / "skills/review/reference.txt").write_text("Actual skill reference.")
    if kind == "command":
        (package / "command.md").write_text(
            "---\nname: review\ndescription: Native command\narguments: [query]\n---\nNative command body."
        )
    elif kind == "rule":
        (package / "rule.md").write_text(
            "---\nname: review\nmode: manual\n---\nNative manual rule body."
        )
    else:
        (package / "mcp.json").write_text(
            json.dumps(
                {
                    "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                    "mcpServers": {
                        "review": {
                            "type": "streamable-http",
                            "url": "https://example.invalid/mcp",
                        }
                    },
                }
            )
        )
    component_id = kind + ":review"
    review = await service.review_install(
        package, selection=("skill:review", component_id), workspace_id="workspace-a"
    )
    assert (await service.commit(review, review.operation_id)).committed
    trust = await service.review_trust(review.installation_id)
    assert (await service.commit(trust, trust.operation_id)).committed
    active = await service.review_activation(
        review.installation_id, workspace_id="workspace-a", intent="enabled"
    )
    assert (await service.commit(active, active.operation_id)).committed
    if kind == "mcp":
        unconfigured = service.capture_maximum("workspace-a")
        assert len(unconfigured["available_skills"]) == 1
        assert all(
            row["plugin_kind"] == "skill" for row in unconfigured["available_skills"]
        )
        client = MCPClient()
        local = LocalMCPControlService(
            store=LocalMCPStore(tmp_path / "collision-mcp.json"),
            client=client,
            manifest_provider=dict,
        )
        await service._call(
            lambda: setattr(service._coordinator, "mcp_mapping_owner", local)
        )
        inspection = await service._call(
            lambda: retained_inspections(
                service._coordinator.published_snapshot(),
                coordinator=service._coordinator,
            )[(review.installation_id, review.inspection.effective_digest)]
        )
        profile = local.save_owned_profile(
            installation_id=review.installation_id,
            inspection=inspection,
            component_id=component_id,
        )
        configured = await service.review_configuration(
            review.installation_id, connections={component_id: profile["profile_id"]}
        )
        assert (await service.commit(configured, configured.operation_id)).committed
        assert not client.sessions, "saved ownership is not a connected transport"
    admitted = await service.admit(
        service.capture_maximum("workspace-a"), "pending:collision"
    )
    skill = next(
        row for row in admitted["available_skills"] if row["plugin_kind"] == "skill"
    )
    native = next(
        row for row in admitted["available_skills"] if row["plugin_kind"] == kind
    )
    assert skill["name"] != native["name"]
    assert native["plugin_component_id"] == component_id
    assert skill["record_id"] != native["record_id"]
    assert skill["tool_name"] != native["tool_name"]
    await service.check_entries((skill, native))
    snapshots = await service.component_snapshots(admitted)
    assert component_id in snapshots[0].selection
    if kind == "command":
        rendered = await service.execute_skill(
            native["name"],
            admission_token=native["plugin_admission"],
            args=json.dumps({"query": "controlled"}),
        )
        assert "Native command body." in repr(rendered)

    async def invoke(name):
        token = skill["plugin_admission"]
        if operation == "execute":
            result = await service.execute_skill(name, admission_token=token)
            assert result["record_id"] == skill["record_id"]
            assert result["plugin_component_id"] == "skill:review"
            assert "Native command body." not in result["rendered_prompt"]
            assert (
                "Check the change and explain actionable findings."
                in result["rendered_prompt"]
            )
        elif operation == "read":
            result = await service.read_skill_file(
                name, "reference.txt", admission_token=token
            )
            assert result["record_id"] == skill["record_id"]
            assert "Actual skill reference." in result["content"]
        else:
            assert (
                await asyncio.to_thread(service.current_definition, name, token)
                == skill["definition_digest"]
            )

    for identity in (skill["record_id"], skill["tool_name"], skill["name"]):
        await invoke(identity)
    for identity in (native["record_id"], native["tool_name"], native["name"]):
        if operation == "read":
            with pytest.raises(
                PluginUnavailable, match="plugin_component_not_admitted"
            ):
                await service.read_skill_file(
                    identity,
                    "reference.txt",
                    admission_token=native["plugin_admission"],
                )
        elif operation == "definition":
            assert (
                await asyncio.to_thread(
                    service.current_definition, identity, native["plugin_admission"]
                )
                == native["definition_digest"]
            )
        elif kind == "mcp":
            with pytest.raises(
                PluginUnavailable, match="plugin_manual_component_unavailable"
            ):
                await service.execute_skill(
                    identity, admission_token=native["plugin_admission"]
                )
        else:
            result = await service.execute_skill(
                identity,
                admission_token=native["plugin_admission"],
                args=json.dumps({"query": "controlled"}) if kind == "command" else "",
            )
            assert (
                "Native command body."
                if kind == "command"
                else "Native manual rule body."
            ) in result["rendered_prompt"]
            assert "Actual skill reference." not in result["rendered_prompt"]
    await service.retire_pending("pending:collision")
    with pytest.raises(PluginUnavailable, match="plugin_turn_retired"):
        await invoke(skill["name"])
