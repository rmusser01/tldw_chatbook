"""Large-data smoke test for the current evaluation browser."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import DataTable, TextArea

from tldw_chatbook.UI.Evals.screens import EvaluationBrowserScreen


class LargeScopeService:
    async def list_evaluations(self, *, mode=None, limit=100, offset=0, after=None, eval_type=None):
        return [
            {
                "record_id": f"local:evaluation:eval_{index}",
                "record_type": "evaluation",
                "backend": "local",
                "backing_id": f"eval_{index}",
                "name": f"evaluation_{index}",
                "description": "Bulk browser record",
                "eval_type": "classification",
                "eval_spec": {"metric": "accuracy"},
                "dataset_id": f"dataset_{index}",
                "created_at": "1713571200",
                "updated_at": "1713571260",
            }
            for index in range(150)
        ]

    async def list_datasets(self, *, mode=None, limit=100, offset=0):
        return [
            {
                "record_id": f"local:evaluation_dataset:dataset_{index}",
                "record_type": "evaluation_dataset",
                "backend": "local",
                "backing_id": f"dataset_{index}",
                "name": f"dataset_{index}",
                "description": "Bulk dataset",
                "sample_count": 20,
            }
            for index in range(150)
        ]

    async def list_runs(self, *, mode=None, eval_id, limit=100, offset=0, after=None, status=None):
        return [
            {
                "record_id": f"local:evaluation_run:{eval_id}:run_{index}",
                "record_type": "evaluation_run",
                "backend": "local",
                "backing_id": f"{eval_id}_run_{index}",
                "evaluation_id": eval_id,
                "name": f"run_{index}",
                "status": "completed",
                "target_model": "openai:gpt-4.1-mini",
                "created_at": "1713571400",
                "progress": {"completed_samples": 20, "total_samples": 20, "percent_complete": 100.0},
                "results": {"accuracy": 0.9},
            }
            for index in range(25)
        ]

    async def list_targets(self, *, mode=None, provider=None, limit=100, offset=0):
        return []

    async def get_run_artifacts(self, *, mode=None, run_id):
        return {"run": {"record_id": run_id}, "metrics": {"accuracy": 0.9}}


class EvaluationBrowserHost(App[None]):
    def __init__(self, app_instance):
        super().__init__()
        self._screen = EvaluationBrowserScreen(app_instance=app_instance, view_mode="results")

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


@pytest.mark.asyncio
async def test_evaluation_browser_handles_large_evaluation_lists() -> None:
    app_instance = SimpleNamespace(
        evaluation_scope_service=LargeScopeService(),
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = EvaluationBrowserHost(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause(0.2)

        evaluations_table = app.screen.query_one("#evaluations-table", DataTable)
        runs_table = app.screen.query_one("#runs-table", DataTable)
        detail = app.screen.query_one("#evaluation-detail", TextArea)

        assert evaluations_table.row_count == 150
        assert runs_table.row_count == 25
        assert "evaluation_0" in detail.text
