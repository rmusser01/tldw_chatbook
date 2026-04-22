from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import DataTable, Input, Select, Static, TextArea

from tldw_chatbook.UI.Evals.evals_window_v3 import EvalsWindowV3
from tldw_chatbook.UI.Evals.screens import EvaluationBrowserScreen


class FakeEvaluationScopeService:
    def __init__(self):
        self.calls = []

    async def list_evaluations(self, *, mode=None, limit=100, offset=0, after=None, eval_type=None):
        self.calls.append(("list_evaluations", mode, limit, offset, after, eval_type))
        return [
            {
                "record_id": "server:evaluation:eval_123",
                "record_type": "evaluation",
                "backend": "server",
                "backing_id": "eval_123",
                "name": "server_eval",
                "description": "Server-side evaluation",
                "eval_type": "classification",
                "eval_spec": {"metrics": ["f1"]},
                "dataset_id": "dataset_123",
                "created_at": "1713571200",
                "updated_at": "1713571260",
                "version": 1,
                "metadata": {"project": "server"},
                "client_id": None,
            }
        ]

    async def list_datasets(self, *, mode=None, limit=100, offset=0):
        self.calls.append(("list_datasets", mode, limit, offset))
        return [
            {
                "record_id": "server:evaluation_dataset:dataset_123",
                "record_type": "evaluation_dataset",
                "backend": "server",
                "backing_id": "dataset_123",
                "name": "demo_dataset",
                "description": "Dataset",
                "sample_count": 2,
                "samples": None,
                "format": None,
                "source_path": None,
                "created_at": "1713571200",
                "updated_at": None,
                "version": None,
                "created_by": "user_1",
                "metadata": {"source": "server"},
                "client_id": None,
            }
        ]

    async def list_runs(self, *, mode=None, eval_id, limit=100, offset=0, after=None, status=None):
        self.calls.append(("list_runs", mode, eval_id, limit, offset, after, status))
        return [
            {
                "record_id": "server:evaluation_run:run_123",
                "record_type": "evaluation_run",
                "backend": "server",
                "backing_id": "run_123",
                "evaluation_id": eval_id,
                "name": "server_run",
                "status": "running",
                "target_model": "gpt-4.1",
                "created_at": "1713571200",
                "started_at": None,
                "completed_at": None,
                "progress": {
                    "completed_samples": 3,
                    "total_samples": 10,
                    "percent_complete": 30.0,
                    "current_sample": None,
                    "estimated_completion": None,
                },
                "error_message": None,
                "results": None,
                "usage": None,
                "config": None,
                "metadata": {},
                "client_id": None,
                "version": None,
            }
        ]

    async def create_run(self, *, mode=None, eval_id, target_id=None, target_model=None, config=None, run_name=None, dataset_override=None, webhook_url=None):
        self.calls.append(("create_run", mode, eval_id, target_id, target_model, config, run_name))
        return {
            "record_id": "server:evaluation_run:run_new",
            "record_type": "evaluation_run",
            "backend": "server",
            "backing_id": "run_new",
            "evaluation_id": eval_id,
            "name": run_name or "run_new",
            "status": "pending",
            "target_model": target_model or "openai:gpt-4.1-mini",
            "created_at": "1713571300",
            "started_at": None,
            "completed_at": None,
            "progress": None,
            "error_message": None,
            "results": None,
            "usage": None,
            "config": config or {},
            "metadata": {},
            "client_id": None,
            "version": None,
        }

    async def get_run_artifacts(self, *, mode=None, run_id):
        self.calls.append(("get_run_artifacts", mode, run_id))
        return {
            "run": {
                "record_id": f"server:evaluation_run:{run_id}",
                "record_type": "evaluation_run",
                "backend": "server",
                "backing_id": run_id,
                "evaluation_id": "eval_123",
                "name": "server_run",
                "status": "completed",
                "target_model": "openai:gpt-4.1-mini",
                "created_at": "1713571200",
                "started_at": None,
                "completed_at": None,
                "progress": None,
                "error_message": None,
                "results": {"accuracy": 0.91},
                "usage": None,
                "config": {"max_workers": 2},
                "metadata": {},
                "client_id": None,
                "version": None,
            },
            "metrics": {"accuracy": 0.91},
            "results": None,
            "detail_available": False,
        }

    async def list_targets(self, *, mode=None, provider=None, limit=100, offset=0):
        self.calls.append(("list_targets", mode, provider, limit, offset))
        return [
            {
                "record_id": "local:evaluation_target:model_123",
                "record_type": "evaluation_target",
                "backend": "local",
                "backing_id": "model_123",
                "name": "Preferred Local",
                "display_name": "Preferred Local (openai:gpt-4.1-mini)",
                "provider": "openai",
                "model_id": "gpt-4.1-mini",
                "target_model": "openai:gpt-4.1-mini",
                "config": {"temperature": 0.2},
            }
        ]


class EvaluationBrowserTestApp(App):
    def __init__(self, app_instance, *, view_mode="manage"):
        super().__init__()
        self._screen = EvaluationBrowserScreen(app_instance=app_instance, view_mode=view_mode)

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


def test_evals_window_v3_maps_tasks_to_evaluation_browser_screen():
    app_instance = SimpleNamespace(notify=lambda *args, **kwargs: None)
    window = EvalsWindowV3(app_instance=app_instance)

    screen = window._create_screen("tasks")

    assert isinstance(screen, EvaluationBrowserScreen)
    assert screen.view_mode == "manage"


def test_evals_window_v3_maps_results_to_results_mode_browser_screen():
    app_instance = SimpleNamespace(notify=lambda *args, **kwargs: None)
    window = EvalsWindowV3(app_instance=app_instance)

    screen = window._create_screen("results")

    assert isinstance(screen, EvaluationBrowserScreen)
    assert screen.view_mode == "results"


@pytest.mark.asyncio
async def test_evaluation_browser_screen_loads_evaluations_and_runs_from_scope_service():
    scope = FakeEvaluationScopeService()
    app_instance = SimpleNamespace(
        evaluation_scope_service=scope,
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = EvaluationBrowserTestApp(app_instance=app_instance, view_mode="manage")

    async with app.run_test() as pilot:
        await pilot.pause(0.2)

        table = app.screen.query_one("#evaluations-table", DataTable)
        runs_table = app.screen.query_one("#runs-table", DataTable)
        detail = app.screen.query_one("#evaluation-detail", TextArea)
        target_input = app.screen.query_one("#target-model-input", Input)

        assert table.row_count == 1
        assert runs_table.row_count == 1
        assert "server_eval" in detail.text
        assert "demo_dataset" in detail.text
        assert target_input.display is True


@pytest.mark.asyncio
async def test_evaluation_browser_screen_shows_local_target_selector_and_results_mode_hides_launcher():
    local_app_instance = SimpleNamespace(
        evaluation_scope_service=FakeEvaluationScopeService(),
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    local_app = EvaluationBrowserTestApp(app_instance=local_app_instance, view_mode="manage")

    async with local_app.run_test() as pilot:
        await pilot.pause(0.2)
        target_select = local_app.screen.query_one("#target-select", Select)
        assert len(target_select._options) >= 1

    results_app_instance = SimpleNamespace(
        evaluation_scope_service=FakeEvaluationScopeService(),
        current_runtime_backend="server",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    results_app = EvaluationBrowserTestApp(app_instance=results_app_instance, view_mode="results")

    async with results_app.run_test() as pilot:
        await pilot.pause(0.2)
        detail = results_app.screen.query_one("#evaluation-detail", TextArea)
        assert "server_eval" in detail.text
        with pytest.raises(Exception):
            results_app.screen.query_one("#create-run-button")


@pytest.mark.asyncio
async def test_evaluation_browser_screen_requires_app_owned_scope_service():
    app_instance = SimpleNamespace(
        current_runtime_backend="local",
        runtime_backend=None,
        app_config={},
        notify=lambda *args, **kwargs: None,
    )
    app = EvaluationBrowserTestApp(app_instance=app_instance, view_mode="manage")

    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        context = app.screen.query_one("#browser-context", Static)

        assert "app-owned evaluation scope service is unavailable" in str(context.render())
