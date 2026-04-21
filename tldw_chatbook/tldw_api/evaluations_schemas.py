from __future__ import annotations

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
