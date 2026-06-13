from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PromptStudioStatus = Literal["draft", "active", "archived"]
PromptStudioPromptFormat = Literal["legacy", "structured"]
PromptStudioImportFormat = Literal["csv", "json"]
PromptStudioExportFormat = Literal["csv", "json"]


class PromptStudioStandardResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool
    data: Any | None = None
    error: str | None = None
    error_code: str | None = None
    metadata: dict[str, Any] | None = None


class PromptStudioListResponse(PromptStudioStandardResponse):
    data: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptStudioDeleteMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str


class PromptStudioProjectCreate(BaseModel):
    name: str
    description: str | None = None
    status: PromptStudioStatus = "draft"
    metadata: dict[str, Any] | None = None


class PromptStudioProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    status: PromptStudioStatus | None = None
    metadata: dict[str, Any] | None = None


class PromptStudioPromptModule(BaseModel):
    type: str
    enabled: bool = True
    config: dict[str, Any] | None = None


class PromptStudioFewShotExample(BaseModel):
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    explanation: str | None = None


class PromptStudioPromptCreate(BaseModel):
    project_id: int
    name: str
    system_prompt: str | None = None
    user_prompt: str | None = None
    prompt_format: PromptStudioPromptFormat = "legacy"
    prompt_schema_version: int | None = None
    prompt_definition: dict[str, Any] | None = None
    few_shot_examples: list[PromptStudioFewShotExample] | None = None
    modules_config: list[PromptStudioPromptModule] | None = None
    change_description: str | None = None
    signature_id: int | None = None
    parent_version_id: int | None = None


class PromptStudioPromptUpdate(BaseModel):
    name: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    prompt_format: PromptStudioPromptFormat | None = None
    prompt_schema_version: int | None = None
    prompt_definition: dict[str, Any] | None = None
    few_shot_examples: list[PromptStudioFewShotExample] | None = None
    modules_config: list[PromptStudioPromptModule] | None = None
    change_description: str


class PromptStudioPromptPreviewRequest(BaseModel):
    project_id: int
    signature_id: int | None = None
    prompt_format: PromptStudioPromptFormat = "legacy"
    system_prompt: str | None = None
    user_prompt: str | None = None
    prompt_schema_version: int | None = None
    prompt_definition: dict[str, Any] | None = None
    few_shot_examples: list[PromptStudioFewShotExample] | None = None
    modules_config: list[PromptStudioPromptModule] | None = None
    variables: dict[str, Any] = Field(default_factory=dict)


class PromptStudioPromptConvertRequest(BaseModel):
    project_id: int
    system_prompt: str | None = None
    user_prompt: str | None = None


class PromptStudioPromptExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: int
    inputs: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = "openai"
    model: str | None = "gpt-3.5-turbo"


class PromptStudioPromptExecutionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    output: str
    tokens_used: int = 0
    execution_time: float = 0.0


class PromptStudioTestCaseBase(BaseModel):
    name: str | None = None
    description: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_outputs: dict[str, Any] | None = None
    tags: list[str] | None = None
    is_golden: bool = False


class PromptStudioTestCaseCreate(PromptStudioTestCaseBase):
    project_id: int
    signature_id: int | None = None


class PromptStudioTestCaseUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    inputs: dict[str, Any] | None = None
    expected_outputs: dict[str, Any] | None = None
    tags: list[str] | None = None
    is_golden: bool | None = None


class PromptStudioTestCaseBulkCreate(BaseModel):
    project_id: int
    signature_id: int | None = None
    test_cases: list[PromptStudioTestCaseBase]


class PromptStudioTestCaseImportRequest(BaseModel):
    project_id: int
    format: PromptStudioImportFormat
    data: str
    signature_id: int | None = None
    auto_generate_names: bool = True


class PromptStudioTestCaseExportRequest(BaseModel):
    project_id: int | None = None
    format: PromptStudioExportFormat = "json"
    include_golden_only: bool = False
    tag_filter: list[str] | None = None


class PromptStudioTestCaseGenerateRequest(BaseModel):
    project_id: int
    prompt_id: int | None = None
    signature_id: int | None = None
    num_cases: int = 5
    generation_strategy: str = "diverse"
    base_on_description: str | None = None


class PromptStudioRunTestCasesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: int
    test_case_ids: list[int] = Field(default_factory=list)
    model: str | None = None
    project_id: int | None = None


class PromptStudioRunTestCasesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: list[dict[str, Any]] = Field(default_factory=list)


class PromptStudioEvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    custom_metrics: dict[str, Any] | None = None


class PromptStudioEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    provider: str | None = None


class PromptStudioEvaluationCreate(BaseModel):
    project_id: int
    prompt_id: int
    test_run_id: int | str | None = None
    name: str | None = None
    description: str | None = None
    metrics: PromptStudioEvaluationMetrics | dict[str, Any] | None = None
    config: PromptStudioEvaluationConfig | dict[str, Any] | None = None
    run_async: bool = False
    test_case_ids: list[int] | None = None
    tags: list[str] | None = None


class PromptStudioEvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    project_id: int
    prompt_id: int
    status: str
    test_run_id: int | None = None
    name: str | None = None
    description: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    tags: list[str] = Field(default_factory=list)
    uuid: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None


class PromptStudioEvaluationListResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    evaluations: list[PromptStudioEvaluationResponse] = Field(default_factory=list)
    total: int = 0
    limit: int = 100
    offset: int = 0


class PromptStudioOptimizationConfig(BaseModel):
    optimizer_type: str
    max_iterations: int = 50
    target_metric: str
    target_value: float | None = None
    early_stopping: bool = True
    early_stopping_patience: int = 5
    temperature_range: list[float] = Field(default_factory=lambda: [0.0, 1.0])
    techniques_to_try: list[str] = Field(default_factory=lambda: ["cot", "few_shot"])
    models_to_test: list[str] | None = None
    budget_limit: float | None = None
    strategy_params: dict[str, Any] = Field(default_factory=dict)


class PromptStudioBootstrapConfig(BaseModel):
    num_samples: int = 50
    selection_method: str = "diverse"
    quality_threshold: float = 0.7
    max_examples_per_prompt: int = 5


class PromptStudioOptimizationCreate(BaseModel):
    project_id: int
    initial_prompt_id: int
    optimization_config: PromptStudioOptimizationConfig
    bootstrap_config: PromptStudioBootstrapConfig | None = None
    test_case_ids: list[int] | None = None
    name: str | None = None
    description: str | None = None


class PromptStudioOptimizationSimpleCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: int | None = None
    initial_prompt_id: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    project_id: int | None = None
    strategy: str | None = None

    @model_validator(mode="after")
    def _require_prompt_identity(self) -> PromptStudioOptimizationSimpleCreateRequest:
        if self.prompt_id is None and self.initial_prompt_id is None:
            raise ValueError("One of prompt_id or initial_prompt_id must be provided")
        return self


class PromptStudioSimpleJobResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    status: str


class PromptStudioOptimizationIterationCreate(BaseModel):
    iteration_number: int
    prompt_variant: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    tokens_used: int | None = None
    cost: float | None = None
    note: str | None = None


class PromptStudioCompareStrategiesRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    prompt_id: int
    project_id: int | None = None
    test_case_ids: list[int]
    strategies: list[str]
    model_configuration: dict[str, Any] = Field(default_factory=dict, alias="config")


class PromptStudioStatusResponse(PromptStudioStandardResponse):
    data: dict[str, Any] = Field(default_factory=dict)
