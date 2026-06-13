"""Integration tests for the current evaluation browser manage flow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import DataTable, Input, Select

from tldw_chatbook.UI.Evals.screens import EvaluationBrowserScreen


class FakeEvaluationScopeService:
    def __init__(self):
        self.calls: list[tuple] = []

    async def list_evaluations(self, *, mode=None, limit=100, offset=0, after=None, eval_type=None):
        self.calls.append(("list_evaluations", mode, limit))
        return [
            {
                "record_id": "local:evaluation:eval_1",
                "record_type": "evaluation",
                "backend": "local",
                "backing_id": "eval_1",
                "name": "article_quality",
                "description": "Checks content quality",
                "eval_type": "classification",
                "eval_spec": {"metric": "accuracy"},
                "dataset_id": "dataset_1",
                "created_at": "1713571200",
                "updated_at": "1713571260",
            }
        ]

    async def list_datasets(self, *, mode=None, limit=100, offset=0):
        self.calls.append(("list_datasets", mode, limit))
        return [
            {
                "record_id": "local:evaluation_dataset:dataset_1",
                "record_type": "evaluation_dataset",
                "backend": "local",
                "backing_id": "dataset_1",
                "name": "demo_dataset",
                "description": "Dataset",
                "sample_count": 3,
            }
        ]

    async def list_runs(self, *, mode=None, eval_id, limit=100, offset=0, after=None, status=None):
        self.calls.append(("list_runs", mode, eval_id, limit))
        return [
            {
                "record_id": "local:evaluation_run:run_1",
                "record_type": "evaluation_run",
                "backend": "local",
                "backing_id": "run_1",
                "evaluation_id": eval_id,
                "name": "run_1",
                "status": "completed",
                "target_model": "openai:gpt-4.1-mini",
                "created_at": "1713571300",
                "progress": {"completed_samples": 3, "total_samples": 3, "percent_complete": 100.0},
                "results": {"accuracy": 0.91},
            }
        ]

    async def list_targets(self, *, mode=None, provider=None, limit=100, offset=0):
        self.calls.append(("list_targets", mode, limit))
        return [
            {
                "record_id": "local:evaluation_target:model_1",
                "record_type": "evaluation_target",
                "backend": "local",
                "backing_id": "model_1",
                "name": "Preferred Local",
                "display_name": "Preferred Local (openai:gpt-4.1-mini)",
                "provider": "openai",
                "model_id": "gpt-4.1-mini",
                "target_model": "openai:gpt-4.1-mini",
                "config": {"temperature": 0.2},
            }
        ]

    async def create_run(self, *, mode=None, eval_id, target_id=None, target_model=None, config=None, run_name=None, dataset_override=None, webhook_url=None):
        self.calls.append(("create_run", mode, eval_id, target_id, target_model, config, run_name))
        return {
            "record_id": "local:evaluation_run:run_2",
            "record_type": "evaluation_run",
            "backend": "local",
            "backing_id": "run_2",
            "evaluation_id": eval_id,
            "name": run_name or "run_2",
            "status": "pending",
            "target_model": target_model or "openai:gpt-4.1-mini",
            "created_at": "1713571400",
            "progress": None,
            "results": None,
        }

    async def get_run_artifacts(self, *, mode=None, run_id):
        self.calls.append(("get_run_artifacts", mode, run_id))
        return {"run": {"record_id": f"local:evaluation_run:{run_id}", "name": run_id}, "metrics": {"accuracy": 0.9}}


class EvaluationBrowserHost(App[None]):
    def __init__(self, app_instance, *, view_mode="manage"):
        super().__init__()
        self._screen = EvaluationBrowserScreen(app_instance=app_instance, view_mode=view_mode)

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


@pytest.mark.asyncio
async def test_evaluation_browser_manage_mode_loads_data_and_targets() -> None:
    scope = FakeEvaluationScopeService()
    app_instance = SimpleNamespace(
        evaluation_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = EvaluationBrowserHost(app_instance, view_mode="manage")

    async with app.run_test() as pilot:
        await pilot.pause(0.2)

        evaluations_table = app.screen.query_one("#evaluations-table", DataTable)
        target_select = app.screen.query_one("#target-select", Select)

        assert evaluations_table.row_count == 1
        assert len(getattr(target_select, "_options", [])) >= 1


@pytest.mark.asyncio
async def test_evaluation_browser_create_run_uses_selected_evaluation() -> None:
    scope = FakeEvaluationScopeService()
    app_instance = SimpleNamespace(
        evaluation_scope_service=scope,
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = EvaluationBrowserHost(app_instance, view_mode="manage")

    async with app.run_test() as pilot:
        await pilot.pause(0.2)

        screen = app.screen
        screen.query_one("#run-name-input", Input).value = "Local Smoke Run"
        screen.query_one("#run-config-input", Input).value = '{"temperature": 0.1}'

        await screen._create_run()
        await pilot.pause()

        create_run_calls = [call for call in scope.calls if call[0] == "create_run"]
        assert len(create_run_calls) == 1
        assert create_run_calls[0][2] == "eval_1"
        assert create_run_calls[0][-1] == "Local Smoke Run"
