from tldw_chatbook.Evaluations_Interop.evaluation_normalizers import (
    normalize_evaluation_dataset_record,
    normalize_evaluation_record,
    normalize_evaluation_run_record,
    normalize_evaluation_target_record,
)


def test_normalize_local_evaluation_maps_task_shape_to_shared_contract():
    record = normalize_evaluation_record(
        "local",
        {
            "id": "task_123",
            "name": "demo_task",
            "description": "Demo task",
            "task_type": "question_answer",
            "config_format": "custom",
            "config_data": {"metrics": ["accuracy"], "threshold": 0.8},
            "dataset_id": "dataset_123",
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:10:00Z",
            "version": 2,
            "client_id": "local_client",
        },
    )

    assert record["record_id"] == "local:evaluation:task_123"
    assert record["eval_type"] == "question_answer"
    assert record["eval_spec"]["threshold"] == 0.8
    assert record["metadata"]["config_format"] == "custom"


def test_normalize_server_dataset_preserves_sample_count_and_samples():
    record = normalize_evaluation_dataset_record(
        "server",
        {
            "id": "dataset_123",
            "object": "dataset",
            "name": "demo_dataset",
            "description": "Demo dataset",
            "sample_count": 2,
            "samples": [
                {"input": "Q1", "expected": "A1", "metadata": {"difficulty": "easy"}},
                {"input": "Q2", "expected": "A2", "metadata": {}},
            ],
            "created": 1713571200,
            "created_by": "u1",
            "metadata": {"source": "server"},
        },
    )

    assert record["record_id"] == "server:evaluation_dataset:dataset_123"
    assert record["sample_count"] == 2
    assert record["samples"][0]["expected"] == "A1"


def test_normalize_local_run_uses_model_name_and_metrics():
    record = normalize_evaluation_run_record(
        "local",
        {
            "id": "run_123",
            "name": "demo_run",
            "task_id": "task_123",
            "status": "completed",
            "model_id": "model_123",
            "model_name": "gpt-4.1-mini",
            "created_at": "2026-04-20T00:00:00Z",
            "start_time": "2026-04-20T00:01:00Z",
            "end_time": "2026-04-20T00:05:00Z",
            "error_message": None,
            "metrics_summary": {"accuracy": 0.9},
            "config_overrides": {"temperature": 0.2},
        },
    )

    assert record["record_id"] == "local:evaluation_run:run_123"
    assert record["target_model"] == "gpt-4.1-mini"
    assert record["results"]["accuracy"] == 0.9


def test_normalize_local_target_exposes_provider_and_target_model():
    record = normalize_evaluation_target_record(
        "local",
        {
            "id": "model_123",
            "name": "Preferred Local",
            "provider": "openai",
            "model_id": "gpt-4.1-mini",
            "config": {"temperature": 0.2},
            "created_at": "2026-04-20T00:00:00Z",
        },
    )

    assert record["record_id"] == "local:evaluation_target:model_123"
    assert record["provider"] == "openai"
    assert record["model_id"] == "gpt-4.1-mini"
    assert record["target_model"] == "openai:gpt-4.1-mini"
