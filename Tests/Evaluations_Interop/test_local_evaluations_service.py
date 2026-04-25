from tldw_chatbook.Evaluations_Interop.local_evaluations_service import LocalEvaluationsService


class FakeEvalsDB:
    def __init__(self):
        self.calls = []
        self.created_task = None
        self.created_run = None
        self.task = {
            "id": "task_123",
            "name": "demo_eval",
            "description": "Demo evaluation",
            "task_type": "question_answer",
            "config_format": "custom",
            "config_data": {
                "metrics": ["accuracy"],
                "__tldw_eval_metadata__": {"project": "offline-parity", "tags": ["compat"]},
            },
            "dataset_id": "dataset_123",
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:05:00Z",
            "version": 2,
            "client_id": "local_client",
        }
        self.run = {
            "id": "run_123",
            "name": "demo_run",
            "task_id": "task_123",
            "task_name": "demo_eval",
            "model_id": "model_123",
            "model_name": "gpt-4.1-mini",
            "status": "completed",
            "created_at": "2026-04-20T00:00:00Z",
            "start_time": "2026-04-20T00:01:00Z",
            "end_time": "2026-04-20T00:02:00Z",
            "completed_samples": 3,
            "total_samples": 3,
            "config_overrides": {"temperature": 0.1},
            "error_message": None,
        }
        self.dataset = {
            "id": "dataset_123",
            "name": "demo_dataset",
            "description": "Demo dataset",
            "format": "json",
            "source_path": "/tmp/demo.json",
            "metadata": {"source": "local"},
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:02:00Z",
            "version": 1,
            "client_id": "local_client",
        }
        self.model = {
            "id": "model_123",
            "name": "Preferred Local",
            "provider": "openai",
            "model_id": "gpt-4.1-mini",
            "config": {"temperature": 0.2},
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:02:00Z",
            "version": 1,
            "client_id": "local_client",
        }
        self.run_results = [
            {
                "sample_id": "sample_1",
                "input_data": {"input": "Q1"},
                "expected_output": "A1",
                "actual_output": "A1",
                "metrics": {"accuracy": 1.0},
                "metadata": {"source": "test"},
                "created_at": "2026-04-20T00:02:00Z",
            }
        ]

    def create_task(self, **kwargs):
        self.calls.append(("create_task", kwargs))
        self.created_task = kwargs
        return "task_999"

    def get_task(self, task_id):
        self.calls.append(("get_task", task_id))
        return dict(self.task)

    def list_tasks(self, task_type=None, limit=100, offset=0):
        self.calls.append(("list_tasks", task_type, limit, offset))
        return [dict(self.task)]

    def update_task(self, task_id, **kwargs):
        self.calls.append(("update_task", task_id, kwargs))
        return True

    def delete_task(self, task_id):
        self.calls.append(("delete_task", task_id))
        return True

    def get_dataset(self, dataset_id):
        self.calls.append(("get_dataset", dataset_id))
        return dict(self.dataset)

    def list_datasets(self, limit=100, offset=0):
        self.calls.append(("list_datasets", limit, offset))
        return [dict(self.dataset)]

    def create_dataset(self, **kwargs):
        self.calls.append(("create_dataset", kwargs))
        return "dataset_999"

    def delete_dataset(self, dataset_id):
        self.calls.append(("delete_dataset", dataset_id))
        return True

    def get_model(self, model_id):
        self.calls.append(("get_model", model_id))
        if model_id == self.model["id"]:
            return dict(self.model)
        return None

    def list_models(self, provider=None, limit=100, offset=0):
        self.calls.append(("list_models", provider, limit, offset))
        return [dict(self.model)]

    def create_run(self, name, task_id, model_id, config_overrides=None):
        self.calls.append(("create_run", name, task_id, model_id, config_overrides or {}))
        self.created_run = {
            **self.run,
            "id": "run_999",
            "name": name,
            "task_id": task_id,
            "model_id": model_id,
            "model_name": self.model["name"],
            "status": "pending",
            "config_overrides": dict(config_overrides or {}),
        }
        return "run_999"

    def get_run(self, run_id):
        self.calls.append(("get_run", run_id))
        if self.created_run and run_id == self.created_run["id"]:
            return dict(self.created_run)
        return dict(self.run)

    def list_runs(self, status=None, task_id=None, model_id=None, limit=100, offset=0):
        self.calls.append(("list_runs", status, task_id, model_id, limit, offset))
        return [dict(self.run)]

    def get_run_metrics(self, run_id):
        self.calls.append(("get_run_metrics", run_id))
        return {"accuracy": {"value": 0.95, "type": "accuracy"}}

    def get_run_results(self, run_id, limit=1000, offset=0):
        self.calls.append(("get_run_results", run_id, limit, offset))
        return list(self.run_results)

    def update_run_status(self, run_id, status, error_message=None):
        self.calls.append(("update_run_status", run_id, status, error_message))


def test_create_evaluation_stores_eval_metadata_in_local_task_payload():
    db = FakeEvalsDB()
    service = LocalEvaluationsService(db=db)

    task_id = service.create_evaluation(
        name="demo_eval",
        description="Demo evaluation",
        eval_type="question_answer",
        eval_spec={"metrics": ["accuracy"]},
        dataset_id="dataset_123",
        metadata={"project": "offline-parity", "tags": ["compat"]},
    )

    assert task_id == "task_999"
    assert db.created_task["config_format"] == "custom"
    assert db.created_task["config_data"]["metrics"] == ["accuracy"]
    assert db.created_task["config_data"]["__tldw_eval_metadata__"]["project"] == "offline-parity"


def test_get_run_enriches_local_run_with_flattened_metrics_summary():
    service = LocalEvaluationsService(db=FakeEvalsDB())

    run = service.get_run("run_123")

    assert run["metrics_summary"]["accuracy"] == 0.95
    assert run["task_id"] == "task_123"
    assert run["model_name"] == "gpt-4.1-mini"


def test_create_run_resolves_provider_model_string_to_local_model_id():
    db = FakeEvalsDB()
    service = LocalEvaluationsService(db=db)

    run = service.create_run(
        "task_123",
        target_model="openai:gpt-4.1-mini",
        run_name="manual_run",
        config={"temperature": 0.4},
    )

    assert run["id"] == "run_999"
    assert db.calls[-2] == ("create_run", "manual_run", "task_123", "model_123", {"temperature": 0.4})
    assert run["model_id"] == "model_123"
    assert run["model_name"] == "Preferred Local"


def test_create_run_persists_dataset_override_and_webhook_url_in_local_run_config():
    db = FakeEvalsDB()
    service = LocalEvaluationsService(db=db)
    dataset_override = {
        "name": "inline_cases",
        "samples": [{"input": "Q1", "expected": "A1", "metadata": {"difficulty": "easy"}}],
        "metadata": {"project": "offline-parity"},
    }

    run = service.create_run(
        "task_123",
        target_id="model_123",
        run_name="override_run",
        config={"temperature": 0.2},
        dataset_override=dataset_override,
        webhook_url="http://127.0.0.1:9000/eval-callback",
    )

    assert run["id"] == "run_999"
    assert db.created_run["config_overrides"]["temperature"] == 0.2
    assert db.created_run["config_overrides"]["dataset_override"] == dataset_override
    assert db.created_run["config_overrides"]["webhook_url"] == "http://127.0.0.1:9000/eval-callback"


def test_get_run_artifacts_returns_local_metrics_and_sample_results():
    service = LocalEvaluationsService(db=FakeEvalsDB())

    artifacts = service.get_run_artifacts("run_123")

    assert artifacts["detail_available"] is True
    assert artifacts["metrics"]["accuracy"] == 0.95
    assert artifacts["results"][0]["sample_id"] == "sample_1"


def test_update_evaluation_merges_eval_spec_and_metadata_into_local_task_payload():
    db = FakeEvalsDB()
    service = LocalEvaluationsService(db=db)

    updated = service.update_evaluation(
        "task_123",
        description="Updated evaluation",
        eval_spec={"metrics": ["f1"]},
        metadata={"project": "server-sync"},
    )

    assert updated is True
    assert db.calls[-1][0] == "update_task"
    assert db.calls[-1][1] == "task_123"
    assert db.calls[-1][2]["description"] == "Updated evaluation"
    assert db.calls[-1][2]["config_data"]["metrics"] == ["f1"]
    assert db.calls[-1][2]["config_data"]["__tldw_eval_metadata__"]["project"] == "server-sync"


def test_cancel_run_marks_local_run_cancelled():
    db = FakeEvalsDB()
    service = LocalEvaluationsService(db=db)

    service.cancel_run("run_123")

    assert db.calls[-1] == ("update_run_status", "run_123", "cancelled", None)


def test_local_dataset_create_and_delete_wrap_local_db():
    db = FakeEvalsDB()
    service = LocalEvaluationsService(db=db)

    dataset_id = service.create_dataset(
        name="offline_dataset",
        format="json",
        source_path="/tmp/offline.json",
        description="Offline samples",
        metadata={"project": "offline-parity"},
    )
    service.delete_dataset("dataset_999")

    assert dataset_id == "dataset_999"
    assert db.calls[-2] == (
        "create_dataset",
        {
            "name": "offline_dataset",
            "format": "json",
            "source_path": "/tmp/offline.json",
            "description": "Offline samples",
            "metadata": {"project": "offline-parity"},
        },
    )
    assert db.calls[-1] == ("delete_dataset", "dataset_999")
