"""Research Runs own the run and report across a workspace return."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Research_Workspace.test_local_studio import local_studio as local_studio
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
from tldw_chatbook.Research_Interop.research_scope_service import ResearchScopeService
from tldw_chatbook.Research_Workspace.chat_models import ResearchSourceVersionRef
from tldw_chatbook.Research_Workspace.contracts import QualifiedWorkspaceRef
from tldw_chatbook.Research_Workspace.deep_research_bridge import (
    DeepResearchBridge,
    DeepResearchReportPreview,
    ResearchReturnMismatch,
)
from tldw_chatbook.Research_Workspace.deep_research_context import ResearchLaunchContext
from tldw_chatbook.Research_Workspace.local_studio import LocalStudioAdapter
from tldw_chatbook.Research_Workspace.overlay_store import (
    ResearchPresentationOverlayStore,
)


ORIGIN = QualifiedWorkspaceRef("local", "reading-group")
SOURCE = ResearchSourceVersionRef(ORIGIN, "source-id", "42", "media", 4, 2, ("fts",))


def _context(**changes):
    context = ResearchLaunchContext(
        origin=ORIGIN,
        launch_id="specific-launch",
        source_snapshot=(SOURCE,),
        query="  How\n   does storage work?  ",
        created_at="2026-11-24T20:00:00Z",
        local_conversation_id="local-thread",
        initiating_message_id="message-42",
    )
    return replace(context, **changes)


def test_launch_context_keeps_only_qualified_versioned_references():
    context = _context()
    assert context.query == "How does storage work?"
    assert context.return_route == "research_workspace"
    assert ResearchLaunchContext.from_payload(context.to_payload()) == context
    with pytest.raises(ValueError, match="conversation"):
        _context(server_chat_id="other-authority")
    with pytest.raises(ValueError, match="workspace"):
        _context(
            source_snapshot=(
                replace(SOURCE, ref=QualifiedWorkspaceRef("local", "other")),
            )
        )
    with pytest.raises(ValueError, match="fields"):
        ResearchLaunchContext.from_payload(
            {**context.to_payload(), "report_body": "private"}
        )


@pytest.mark.asyncio
async def test_local_launch_and_preview_use_the_same_run_after_restart(tmp_path):
    db_path = tmp_path / "research.sqlite"
    overlay_path = tmp_path / "private" / "workspace-overlay.json"
    owner = LocalResearchService(db_path)
    scope = ResearchScopeService(local_service=owner, server_service=None)
    bridge = DeepResearchBridge(scope, ResearchPresentationOverlayStore(overlay_path))
    context = _context()

    target = await bridge.launch_local(context)
    assert await bridge.launch_local(context) == target
    assert len(owner.list_runs()) == 1
    run = owner.get_run(target.run_id)
    assert run["chat_handoff"] == {"research_workspace": context.to_payload()}
    assert "conversation_id" not in run["chat_handoff"]
    owner.save_artifact(
        target.run_id,
        artifact_name="bundle.json",
        content_type="application/json",
        content={
            "query": context.query,
            "report_markdown": "# Native report",
            "source_count": 1,
        },
    )
    owner.complete_run(target.run_id)

    restarted_owner = LocalResearchService(db_path)
    restarted_scope = ResearchScopeService(
        local_service=restarted_owner, server_service=None
    )
    overlay = ResearchPresentationOverlayStore(overlay_path)
    restarted = DeepResearchBridge(restarted_scope, overlay)
    assert restarted.target_from_run(restarted_owner.get_run(target.run_id)) == target
    assert overlay.load(ORIGIN).deep_research_launches[target.run_id] == context
    preview = await restarted.preview_return(target)
    assert preview.target == target and preview.report == "# Native report"
    assert preview.source_count == 1
    assert "Native report" not in overlay_path.read_text()


@pytest.mark.asyncio
async def test_return_rejects_a_different_run_origin_or_bundle(tmp_path):
    context = _context()
    target = context.target("native-run")
    overlay = ResearchPresentationOverlayStore(
        tmp_path / "private" / "workspace-overlay.json"
    )
    overlay.remember_research_launch(target, context)
    run = {
        "source": "local",
        "id": target.run_id,
        "chat_handoff": {"research_workspace": context.to_payload()},
    }

    async def get_run(run_id, *, mode):
        assert (run_id, mode) == (target.run_id, "local")
        return run

    async def get_bundle(*, mode, run_id):
        assert (run_id, mode) == (target.run_id, "local")
        return {"backend": "local", "run": {**run, "id": "unrelated-run"}}

    bridge = DeepResearchBridge(
        SimpleNamespace(get_run=get_run, get_bundle=get_bundle), overlay
    )
    with pytest.raises(ResearchReturnMismatch, match="bundle"):
        await bridge.preview_return(target)
    with pytest.raises(ResearchReturnMismatch, match="workspace"):
        await bridge.preview_return(
            replace(target, origin=QualifiedWorkspaceRef("local", "other"))
        )
    with pytest.raises(ResearchReturnMismatch, match="workspace"):
        await bridge.preview_return(replace(target, launch_id="other-launch"))


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
async def test_report_import_requires_explicit_save_and_replay_preserves_draft_owner(
    local_studio,
):
    local_studio.registry.create_workspace(
        workspace_id=ORIGIN.workspace_id, name="Deep Research group"
    )
    context = _context()
    preview = DeepResearchReportPreview(
        context.target("native-run"), context, "# Native report", 1
    )
    request = preview.studio_request(context_revision=1)
    adapter = local_studio.adapter
    port = SimpleNamespace(save_imported_report=adapter.save)
    assert await adapter.get_imported_report(request) is None
    assert await local_studio.artifacts.list_chatbooks() == []

    saved = await preview.save_report(port, request)
    reopened = LocalStudioAdapter(
        local_studio.registry,
        local_studio.artifacts,
        local_studio.study,
        local_studio.quizzes,
    )
    assert await reopened.get_imported_report(request) == saved
    assert (
        await preview.save_report(
            SimpleNamespace(save_imported_report=reopened.save), request
        )
        == saved
    )
    record = await local_studio.artifacts.get_chatbook(saved.owner_id)
    assert record["metadata"]["review_status"] == "draft"
    assert record["metadata"]["research_run_ref"] == preview.target.provenance()

    version = await preview.save_report(port, request, supersedes=saved)
    assert version.owner_id != saved.owner_id
    assert len(await local_studio.artifacts.list_chatbooks()) == 2
