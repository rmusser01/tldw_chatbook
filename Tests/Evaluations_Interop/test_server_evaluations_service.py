import pytest

from tldw_chatbook.Evaluations_Interop.server_evaluations_service import ServerEvaluationsService


class FakeEvaluationClient:
    def __init__(self):
        self.calls = []

    async def evaluate_geval(self, request_data):
        self.calls.append(("evaluate_geval", request_data))
        return {
            "metrics": {"fluency": {"score": 0.91}},
            "average_score": 0.91,
            "summary_assessment": "Strong summary",
            "evaluation_time": 1.2,
            "metadata": {"evaluation_id": "geval_1"},
        }

    async def evaluate_rag(self, request_data):
        self.calls.append(("evaluate_rag", request_data))
        return {
            "metrics": {"faithfulness": {"score": 0.86}},
            "overall_score": 0.86,
            "retrieval_quality": 0.8,
            "generation_quality": 0.9,
            "suggestions": ["Add source coverage"],
            "metadata": {"evaluation_id": "rag_1"},
        }

    async def evaluate_response_quality(self, request_data):
        self.calls.append(("evaluate_response_quality", request_data))
        return {
            "metrics": {"relevance": {"score": 0.88}},
            "overall_quality": 0.88,
            "format_compliance": {"json": True},
        }

    async def evaluate_propositions(self, request_data):
        self.calls.append(("evaluate_propositions", request_data))
        return {
            "precision": 0.8,
            "recall": 0.75,
            "f1": 0.77,
            "matched": 3,
            "total_extracted": 4,
            "total_reference": 4,
            "claim_density_per_100_tokens": 2.5,
            "avg_prop_len_tokens": 8.0,
            "dedup_rate": 0.0,
        }

    async def evaluate_batch(self, request_data):
        self.calls.append(("evaluate_batch", request_data))
        return {
            "total_items": 2,
            "successful": 2,
            "failed": 0,
            "results": [{"id": "item_1", "score": 0.9}],
            "aggregate_metrics": {"average_score": 0.9},
            "processing_time": 2.4,
        }

    async def evaluate_ocr(self, request_data):
        self.calls.append(("evaluate_ocr", request_data))
        return {
            "evaluation_id": "ocr_1",
            "results": {"items": [{"id": "doc_1", "cer": 0.02}]},
            "evaluation_time": 0.5,
        }

    async def evaluate_ocr_pdf(self, file_paths, **kwargs):
        self.calls.append(("evaluate_ocr_pdf", file_paths, kwargs))
        return {
            "evaluation_id": "ocr_pdf_1",
            "results": {"items": [{"id": "doc.pdf", "cer": 0.02}]},
            "evaluation_time": 1.5,
        }

    async def get_evaluation_history(self, request_data):
        self.calls.append(("get_evaluation_history", request_data))
        return {
            "total_count": 1,
            "items": [{"evaluation_id": "geval_1", "evaluation_type": "geval"}],
            "aggregations": {"geval": 1},
        }

    async def run_evaluation_benchmark(self, benchmark_name, request_data):
        self.calls.append(("run_evaluation_benchmark", benchmark_name, request_data))
        return {
            "benchmark": benchmark_name,
            "total_samples": 2,
            "results_summary": {"average_score": 0.75},
            "evaluation_id": "eval_bench_1",
        }

    async def create_evaluation_recipe_run(self, recipe_id, request_data):
        self.calls.append(("create_evaluation_recipe_run", recipe_id, request_data))
        return {
            "run_id": "recipe_run_1",
            "recipe_id": recipe_id,
            "recipe_version": "1.0.0",
            "status": "pending",
            "review_state": "not_required",
            "child_run_ids": [],
            "created_at": "2026-04-23T12:00:00Z",
            "metadata": {"job_id": "job-1"},
        }

    async def get_evaluation_recipe_run(self, run_id):
        self.calls.append(("get_evaluation_recipe_run", run_id))
        return {
            "run_id": run_id,
            "recipe_id": "rag_answer_quality",
            "recipe_version": "1.0.0",
            "status": "running",
            "review_state": "not_required",
            "child_run_ids": [],
            "created_at": "2026-04-23T12:00:00Z",
        }

    async def get_evaluation_recipe_run_report(self, run_id):
        self.calls.append(("get_evaluation_recipe_run_report", run_id))
        return {
            "run": {
                "run_id": run_id,
                "recipe_id": "rag_answer_quality",
                "recipe_version": "1.0.0",
                "status": "completed",
                "review_state": "approved",
                "child_run_ids": [],
                "created_at": "2026-04-23T12:00:00Z",
            },
            "confidence_summary": {
                "kind": "aggregate",
                "confidence": 0.82,
                "sample_count": 2,
            },
            "recommendation_slots": {},
        }


@pytest.mark.asyncio
async def test_server_evaluations_service_wraps_unified_immediate_routes():
    client = FakeEvaluationClient()
    service = ServerEvaluationsService(client=client)

    geval = await service.evaluate_geval(
        source_text="Original text long enough",
        summary="Summary text long enough",
    )
    rag = await service.evaluate_rag(
        query="What changed?",
        retrieved_contexts=["Context"],
        generated_response="Answer",
        ground_truth="Expected answer",
    )
    quality = await service.evaluate_response_quality(
        prompt="Explain this",
        response="A complete answer",
    )
    propositions = await service.evaluate_propositions(
        extracted=["Claim A"],
        reference=["Claim A"],
    )
    batch = await service.evaluate_batch(
        evaluation_type="geval",
        items=[{"source_text": "A", "summary": "B"}],
    )
    ocr = await service.evaluate_ocr(
        items=[
            {
                "id": "doc_1",
                "extracted_text": "hello world",
                "ground_truth_text": "hello world",
            }
        ]
    )
    ocr_pdf = await service.evaluate_ocr_pdf(
        file_paths=["/tmp/doc.pdf"],
        ground_truths=["hello world"],
        metrics=["cer"],
        thresholds={"max_cer": 0.1},
    )
    history = await service.get_evaluation_history(evaluation_type="geval", limit=10)

    assert geval["average_score"] == 0.91
    assert rag["suggestions"] == ["Add source coverage"]
    assert quality["format_compliance"] == {"json": True}
    assert propositions["f1"] == 0.77
    assert batch["aggregate_metrics"] == {"average_score": 0.9}
    assert ocr["results"]["items"][0]["cer"] == 0.02
    assert ocr_pdf["evaluation_id"] == "ocr_pdf_1"
    assert history["aggregations"] == {"geval": 1}
    assert client.calls[0][0] == "evaluate_geval"
    assert client.calls[0][1].model_dump(mode="json")["metrics"] == [
        "fluency",
        "consistency",
        "relevance",
        "coherence",
    ]
    assert client.calls[3][1].model_dump(mode="json") == {
        "extracted": ["Claim A"],
        "reference": ["Claim A"],
        "method": "semantic",
        "threshold": 0.7,
    }
    assert client.calls[4][1].model_dump(mode="json")["evaluation_type"] == "geval"
    assert client.calls[5][1].model_dump(mode="json")["items"][0]["id"] == "doc_1"
    assert client.calls[6] == (
        "evaluate_ocr_pdf",
        ["/tmp/doc.pdf"],
        {
            "ground_truths": ["hello world"],
            "metrics": ["cer"],
            "ground_truths_pages": None,
            "thresholds": {"max_cer": 0.1},
            "enable_ocr": True,
            "ocr_backend": None,
            "ocr_lang": "eng",
            "ocr_dpi": 300,
            "ocr_mode": "fallback",
            "ocr_min_page_text_chars": 40,
            "ocr_output_format": None,
            "ocr_prompt_preset": None,
        },
    )
    assert client.calls[7][1].model_dump(exclude_none=True, mode="json") == {
        "evaluation_type": "geval",
        "limit": 10,
        "offset": 0,
    }


@pytest.mark.asyncio
async def test_server_evaluations_service_wraps_benchmark_and_recipe_run_routes():
    client = FakeEvaluationClient()
    service = ServerEvaluationsService(client=client)

    benchmark_run = await service.run_benchmark(
        "truthfulqa",
        limit=2,
        parallel=2,
        filter_categories=["truthful"],
    )
    created_recipe_run = await service.create_recipe_run(
        "rag_answer_quality",
        dataset_id="dataset_1",
        run_config={"evaluation_mode": "fixed_context"},
        force_rerun=True,
    )
    fetched_recipe_run = await service.get_recipe_run("recipe_run_1")
    report = await service.get_recipe_run_report("recipe_run_1")

    assert benchmark_run["evaluation_id"] == "eval_bench_1"
    assert created_recipe_run["metadata"] == {"job_id": "job-1"}
    assert fetched_recipe_run["status"] == "running"
    assert report["confidence_summary"]["confidence"] == 0.82

    benchmark_request = client.calls[0][2]
    recipe_request = client.calls[1][2]
    assert client.calls[0][:2] == ("run_evaluation_benchmark", "truthfulqa")
    assert benchmark_request.model_dump(exclude_none=True, mode="json") == {
        "limit": 2,
        "api_name": "openai",
        "parallel": 2,
        "save_results": True,
        "filter_categories": ["truthful"],
    }
    assert client.calls[1][:2] == ("create_evaluation_recipe_run", "rag_answer_quality")
    assert recipe_request.model_dump(exclude_none=True, mode="json") == {
        "dataset_id": "dataset_1",
        "run_config": {"evaluation_mode": "fixed_context"},
        "force_rerun": True,
    }
    assert client.calls[2] == ("get_evaluation_recipe_run", "recipe_run_1")
    assert client.calls[3] == ("get_evaluation_recipe_run_report", "recipe_run_1")
