from tldw_chatbook.tldw_api import (
    CreateEvaluationRequest,
    EvaluationDatasetCreateRequest,
    EvaluationDatasetListResponse,
    EvaluationDatasetResponse,
    EvaluationMetadata,
    EvaluationResponse,
    EvaluationRunCreateRequest,
    EvaluationRunResponse,
    EvaluationSpec,
    UpdateEvaluationRequest,
)


def test_evaluation_dataset_create_request_round_trips_samples_and_metadata():
    payload = EvaluationDatasetCreateRequest(
        name="demo_dataset",
        description="Demo dataset",
        samples=[
            {"input": "What is 2+2?", "expected": "4", "metadata": {"difficulty": "easy"}},
            {"input": "What is 3+3?", "expected": "6"},
        ],
        metadata={"source": "local"},
    )

    dumped = payload.model_dump()
    assert dumped["name"] == "demo_dataset"
    assert dumped["samples"][0]["expected"] == "4"
    assert dumped["metadata"]["source"] == "local"


def test_create_evaluation_request_round_trips_eval_spec():
    payload = CreateEvaluationRequest(
        name="demo_eval",
        description="Demo evaluation",
        eval_type="exact_match",
        eval_spec=EvaluationSpec(metrics=["accuracy"], threshold=0.85, model="gpt-4.1-mini"),
        dataset_id="dataset_123",
        metadata=EvaluationMetadata(project="demo", version="v1", tags=["parity"]),
    )

    dumped = payload.model_dump(exclude_none=True)
    assert dumped["eval_spec"]["threshold"] == 0.85
    assert dumped["metadata"]["project"] == "demo"
    assert dumped["dataset_id"] == "dataset_123"


def test_update_evaluation_request_is_partial():
    payload = UpdateEvaluationRequest(description="Updated evaluation")
    assert payload.model_dump(exclude_none=True) == {"description": "Updated evaluation"}


def test_evaluation_response_parses_timestamps_and_metadata():
    payload = EvaluationResponse.model_validate(
        {
            "id": "eval_123",
            "object": "evaluation",
            "name": "demo_eval",
            "description": "Demo evaluation",
            "eval_type": "exact_match",
            "eval_spec": {"metrics": ["accuracy"], "threshold": 0.8},
            "dataset_id": "dataset_123",
            "created": 1713571200,
            "created_at": 1713571200,
            "created_by": "u1",
            "updated": 1713571300,
            "updated_at": 1713571300,
            "metadata": {"project": "demo"},
        }
    )

    assert payload.created == 1713571200
    assert payload.updated_at == 1713571300
    assert payload.metadata["project"] == "demo"


def test_evaluation_run_response_wraps_progress():
    payload = EvaluationRunResponse.model_validate(
        {
            "id": "run_123",
            "object": "run",
            "eval_id": "eval_123",
            "status": "running",
            "target_model": "gpt-4.1-mini",
            "created": 1713571200,
            "created_at": 1713571200,
            "started_at": 1713571210,
            "completed_at": None,
            "progress": {
                "completed_samples": 5,
                "total_samples": 20,
                "percent_complete": 25.0,
                "current_sample": 6,
            },
            "error_message": None,
            "results": None,
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
    )

    assert payload.progress.total_samples == 20
    assert payload.progress.percent_complete == 25.0
    assert payload.usage["input_tokens"] == 100


def test_evaluation_run_create_request_keeps_free_form_config():
    payload = EvaluationRunCreateRequest(
        target_model="gpt-4.1-mini",
        config={"temperature": 0.2, "max_workers": 2, "custom_flag": True},
    )

    dumped = payload.model_dump(exclude_none=True)
    assert dumped["config"]["custom_flag"] is True


def test_evaluation_dataset_list_response_wraps_records():
    payload = EvaluationDatasetListResponse.model_validate(
        {
            "object": "list",
            "data": [
                {
                    "id": "dataset_123",
                    "object": "dataset",
                    "name": "demo_dataset",
                    "description": "Demo dataset",
                    "sample_count": 2,
                    "created": 1713571200,
                    "created_at": 1713571200,
                    "created_by": "u1",
                    "metadata": {"source": "local"},
                }
            ],
            "has_more": False,
            "first_id": "dataset_123",
            "last_id": "dataset_123",
        }
    )

    assert payload.data[0].name == "demo_dataset"
    assert payload.first_id == "dataset_123"


def test_evaluation_dataset_response_defaults_metadata_to_none():
    payload = EvaluationDatasetResponse(
        id="dataset_123",
        name="demo_dataset",
        sample_count=1,
        created=1713571200,
        created_by="u1",
    )
    assert payload.metadata is None
