from types import SimpleNamespace

import pytest

from Tests.Research_Workspace.test_studio_models import LOCAL, SOURCE
from tldw_chatbook.Research_Workspace.deep_research_context import ResearchLaunchContext


@pytest.mark.asyncio
async def test_status_uses_only_native_runs_that_confirm_the_workspace_launch():
    from tldw_chatbook.UI.Research_Workspace_Modules.operation_status import (
        recent_workspace_runs,
    )

    launch = ResearchLaunchContext(
        LOCAL, "launch", (SOURCE,), "Where is the source?", "2026-11-24T12:00:00Z"
    )
    records = {"run-a": launch, "run-b": launch, "run-c": launch}

    class Runs:
        async def get_run(self, run_id, *, mode):
            assert mode == "local"
            if run_id == "run-c":
                raise OSError("status unavailable")
            return {
                "source": "local",
                "id": run_id,
                "query": launch.query,
                "status": "completed",
                "chat_handoff": {
                    "research_workspace": launch.to_payload()
                    if run_id == "run-a"
                    else {}
                },
            }

    screen = SimpleNamespace(
        app=SimpleNamespace(research_scope_service=Runs()),
        overlay_store=SimpleNamespace(
            load=lambda ref: SimpleNamespace(deep_research_launches=records)
        ),
    )
    assert await recent_workspace_runs(screen, LOCAL) == (
        "Runs · Completed",
        "Runs · Status unavailable. Open Runs to refresh.",
    )
