from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class DatasetSample(BaseModel):
    input: Any
    expected: Optional[Any] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetOverride(BaseModel):
    samples: list[DatasetSample]
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationMetadata(BaseModel):
    project: Optional[str] = None
    version: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)


class EvaluationSpec(BaseModel):
    sub_type: Optional[str] = None
    metrics: Optional[list[str]] = None
    threshold: Optional[float] = None
    thresholds: Optional[dict[str, float]] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    custom_prompts: Optional[dict[str, str]] = None
    rag_pipeline: Optional[dict[str, Any]] = None
    allowed_labels: Optional[list[str]] = None
    label_mapping: Optional[dict[str, str]] = None
    structured_output: Optional[bool] = None
    generate_predictions: Optional[bool] = None
    prompt_template: Optional[str] = None
    nli_model: Optional[str] = None


class CreateEvaluationRequest(BaseModel):
    name: str
    description: Optional[str] = None
    eval_type: str
    eval_spec: EvaluationSpec
    dataset_id: Optional[str] = None
    dataset: Optional[list[DatasetSample]] = None
    metadata: Optional[EvaluationMetadata] = None


class UpdateEvaluationRequest(BaseModel):
    description: Optional[str] = None
    eval_spec: Optional[EvaluationSpec] = None
    metadata: Optional[EvaluationMetadata] = None


class EvaluationResponse(BaseModel):
    id: str
    object: str = Field(default="evaluation")
    name: str
    description: Optional[str] = None
    eval_type: str
    eval_spec: dict[str, Any]
    dataset_id: Optional[str] = None
    created: int
    created_at: Optional[int] = None
    created_by: str
    updated: Optional[int] = None
    updated_at: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class EvaluationRunProgress(BaseModel):
    completed_samples: int = 0
    total_samples: int = 0
    percent_complete: float = 0.0
    current_sample: Optional[int] = None
    estimated_completion: Optional[int] = None


class EvaluationRunCreateRequest(BaseModel):
    target_model: Optional[str] = None
    dataset_override: Optional[DatasetOverride] = None
    config: dict[str, Any] = Field(default_factory=dict)
    webhook_url: Optional[str] = None


class EvaluationRunResponse(BaseModel):
    id: str
    object: str = Field(default="run")
    eval_id: str
    status: str
    target_model: str
    created: int
    created_at: Optional[int] = None
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    progress: Optional[EvaluationRunProgress] = None
    error_message: Optional[str] = None
    results: Optional[dict[str, Any]] = None
    usage: Optional[dict[str, int]] = None

    model_config = ConfigDict(from_attributes=True)


class EvaluationDatasetCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    samples: list[DatasetSample]
    metadata: Optional[dict[str, Any]] = None


class EvaluationDatasetResponse(BaseModel):
    id: str
    object: str = Field(default="dataset")
    name: str
    description: Optional[str] = None
    sample_count: int
    samples: Optional[list[DatasetSample]] = None
    created: int
    created_at: Optional[int] = None
    created_by: str
    metadata: Optional[dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class EvaluationListResponse(BaseModel):
    object: str = Field(default="list")
    data: list[EvaluationResponse]
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    total: Optional[int] = None


class EvaluationRunListResponse(BaseModel):
    object: str = Field(default="list")
    data: list[EvaluationRunResponse]
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    total: Optional[int] = None


class EvaluationDatasetListResponse(BaseModel):
    object: str = Field(default="list")
    data: list[EvaluationDatasetResponse]
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    total: Optional[int] = None


class SyntheticEvalDraftSampleRecord(BaseModel):
    sample_id: str
    recipe_kind: str
    provenance: str
    review_state: str
    sample_payload: dict[str, Any] = Field(default_factory=dict)
    sample_metadata: dict[str, Any] = Field(default_factory=dict)
    source_kind: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class SyntheticEvalGenerationRequest(BaseModel):
    recipe_kind: str
    corpus_scope: dict[str, Any] | list[str] | None = None
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    context_snapshot_ref: Optional[str] = None
    retrieval_baseline_ref: Optional[str] = None
    reference_answer: Optional[str] = None
    real_examples: list[dict[str, Any]] = Field(default_factory=list)
    seed_examples: list[dict[str, Any]] = Field(default_factory=list)
    target_sample_count: int = Field(default=0, ge=0, le=500)


class SyntheticEvalGenerationResponse(BaseModel):
    generation_batch_id: Optional[str] = None
    samples: list[SyntheticEvalDraftSampleRecord] = Field(default_factory=list)
    source_breakdown: dict[str, int] = Field(default_factory=dict)
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    missing_coverage: dict[str, list[str]] = Field(default_factory=dict)
    corpus_scope: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True, extra="allow")


class SyntheticEvalQueueResponse(BaseModel):
    data: list[SyntheticEvalDraftSampleRecord] = Field(default_factory=list)
    total: int = 0

    model_config = ConfigDict(from_attributes=True, extra="allow")


class SyntheticEvalReviewRequest(BaseModel):
    action: str
    reviewer_id: Optional[str] = None
    notes: Optional[str] = None
    action_payload: dict[str, Any] = Field(default_factory=dict)
    resulting_review_state: Optional[str] = None


class SyntheticEvalReviewActionRecord(BaseModel):
    action_id: str
    sample_id: str
    action: str
    reviewer_id: Optional[str] = None
    notes: Optional[str] = None
    action_payload: dict[str, Any] = Field(default_factory=dict)
    resulting_review_state: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class SyntheticEvalPromotionRequest(BaseModel):
    sample_ids: list[str] = Field(..., min_length=1)
    dataset_name: str
    dataset_description: Optional[str] = None
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)
    promoted_by: Optional[str] = None
    promotion_reason: Optional[str] = None


class SyntheticEvalPromotionResponse(BaseModel):
    dataset_id: str
    dataset_snapshot_ref: str
    promotion_ids: list[str] = Field(default_factory=list)
    sample_count: int = 0

    model_config = ConfigDict(from_attributes=True, extra="allow")


class WebhookRegistrationRequest(BaseModel):
    url: str
    events: list[str] = Field(..., min_length=1)
    secret: Optional[str] = None
    retry_count: Optional[int] = Field(default=3, ge=0, le=10)
    timeout_seconds: Optional[int] = Field(default=30, ge=1, le=300)


class WebhookRegistrationResponse(BaseModel):
    webhook_id: int
    url: str
    events: list[str]
    secret: str
    created_at: datetime
    status: str = "active"
    retry_count: int = 3
    timeout_seconds: int = 30

    model_config = ConfigDict(from_attributes=True, extra="allow")


class WebhookStatusResponse(BaseModel):
    webhook_id: int
    url: str
    events: list[str]
    status: str
    retry_count: Optional[int] = None
    timeout_seconds: Optional[int] = None
    created_at: datetime
    last_triggered: Optional[datetime] = None
    failure_count: int = 0

    model_config = ConfigDict(from_attributes=True, extra="allow")


class WebhookTestRequest(BaseModel):
    url: str


class WebhookTestResponse(BaseModel):
    success: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class RecipeManifest(BaseModel):
    recipe_id: str
    recipe_version: str
    name: str
    description: str
    launchable: bool = True
    supported_modes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    default_run_config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True, extra="allow")


class RecipeLaunchReadiness(BaseModel):
    recipe_id: str
    ready: bool
    can_enqueue_runs: bool
    can_reuse_completed_runs: bool = True
    runtime_checks: dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class RecipeDatasetValidationRequest(BaseModel):
    dataset_id: Optional[str] = None
    dataset: Any = None
    run_config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class RecipeDatasetValidationResponse(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    dataset_mode: Optional[str] = None
    sample_count: int = 0
    dataset_snapshot_ref: Optional[str] = None
    dataset_content_hash: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class RecipeRunCreateRequest(RecipeDatasetValidationRequest):
    force_rerun: bool = False


class RecipeRunRecord(BaseModel):
    run_id: str
    recipe_id: str
    recipe_version: Optional[str] = None
    status: str
    review_state: str = "not_required"
    dataset_snapshot_ref: Optional[str] = None
    dataset_content_hash: Optional[str] = None
    confidence_summary: dict[str, Any] = Field(default_factory=dict)
    recommendation_slots: list[dict[str, Any]] = Field(default_factory=list)
    child_run_ids: list[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True, extra="allow")
