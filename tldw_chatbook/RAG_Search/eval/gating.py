# gating.py
# Description: CI/CD quality gating for the RAG evaluation harness.
#
"""Ported from tldw_server2 rag_service/quality_gating.py (P1, RAG server-port programme).

CI/CD quality gating for the RAG pipeline.

Provides intelligent gating that distinguishes between:
- **Stable metrics** (retrieval precision, recall, MRR, NDCG, latency):
  Deterministic and reproducible. Failing = hard CI failure (exit code 1).
- **Unstable metrics** (LLM-based faithfulness, relevance, hallucination):
  Non-deterministic due to LLM variance. Failing = warning only (exit code 2).

This prevents LLM variance from blocking deployments while still surfacing
quality concerns.

Ported from RAGnarok-AI's gating system, adapted for tldw_server2.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MetricCategory(str, Enum):
    """Category of metric for gating purposes."""
    STABLE = "stable"
    UNSTABLE = "unstable"


class GatingResult(str, Enum):
    """Result of gating evaluation."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class MetricResult(BaseModel):
    """Result for a single metric evaluation.

    Attributes:
        name: Name of the metric.
        value: Actual value of the metric.
        threshold: Threshold value for the metric.
        category: Whether this is a stable or unstable metric.
        result: Pass, warn, or fail result.
    """

    model_config = {"frozen": True}

    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Actual metric value")
    threshold: float = Field(..., description="Threshold value")
    category: MetricCategory = Field(..., description="Metric category")
    result: GatingResult = Field(..., description="Evaluation result")


class GatingConfig(BaseModel):
    """Configuration for CI gating thresholds.

    Attributes:
        stable: Thresholds for stable metrics (hard fail if below).
        unstable: Thresholds for unstable metrics (warning only if below).
        lower_is_better: Metric names where a lower value is better
            (e.g., hallucination rate, latency).
    """

    model_config = {"frozen": True}

    stable: dict[str, float] = Field(
        default_factory=lambda: {
            "precision": 0.8,
            "recall": 0.8,
            "mrr": 0.8,
            "ndcg": 0.8,
        },
        description="Stable metric thresholds (fail if below)",
    )
    unstable: dict[str, float] = Field(
        default_factory=lambda: {
            "faithfulness": 0.7,
            "relevance": 0.7,
            "hallucination": 0.3,
        },
        description="Unstable metric thresholds (warn if below)",
    )
    lower_is_better: list[str] = Field(
        default_factory=lambda: ["hallucination", "latency_p99_ms"],
        description="Metrics where lower values are better",
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> GatingConfig:
        """Load gating configuration from a YAML file.

        Expected YAML structure::

            gating:
              stable:
                precision: 0.8
                recall: 0.8
              unstable:
                faithfulness: 0.7
              lower_is_better:
                - hallucination

        Args:
            path: Path to the YAML configuration file.

        Returns:
            GatingConfig loaded from the file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config loading: pip install pyyaml") from None

        path = Path(path)
        if not path.exists():
            msg = f"Configuration file not found: {path}"
            raise FileNotFoundError(msg)

        content = path.read_text()
        data = yaml.safe_load(content)

        if data is None:
            return cls()

        gating_data = data.get("gating", data)
        return cls(
            stable=gating_data.get("stable", {}),
            unstable=gating_data.get("unstable", {}),
            lower_is_better=gating_data.get("lower_is_better", ["hallucination", "latency_p99_ms"]),
        )

    def to_yaml(self, path: Path | str) -> None:
        """Save gating configuration to a YAML file.

        Args:
            path: Path to the output YAML file.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config saving: pip install pyyaml") from None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "gating": {
                "stable": dict(self.stable),
                "unstable": dict(self.unstable),
                "lower_is_better": list(self.lower_is_better),
            }
        }

        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


class GatingEvaluationResult(BaseModel):
    """Result of evaluating metrics against gating thresholds.

    Attributes:
        overall_result: The overall gating result (pass/warn/fail).
        exit_code: Exit code for CI (0=pass, 1=fail, 2=warn).
        metrics: Individual metric results.
        summary: Human-readable summary of the evaluation.
    """

    model_config = {"frozen": True}

    overall_result: GatingResult = Field(..., description="Overall result")
    exit_code: int = Field(..., ge=0, le=2, description="Exit code for CI")
    metrics: list[MetricResult] = Field(default_factory=list, description="Metric results")
    summary: str = Field(..., description="Human-readable summary")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "overall_result": self.overall_result.value,
            "exit_code": self.exit_code,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "category": m.category.value,
                    "result": m.result.value,
                }
                for m in self.metrics
            ],
            "summary": self.summary,
        }


class GatingEvaluator:
    """Evaluates metrics against gating thresholds.

    Distinguishes between stable and unstable metrics:
    - Stable metrics (retrieval, latency): Hard fail if below threshold
    - Unstable metrics (LLM-as-judge): Warning only if below threshold

    Exit codes:
    - 0: All metrics pass
    - 1: At least one stable metric failed
    - 2: Only unstable metrics below threshold (warning)

    Example::

        config = GatingConfig()
        evaluator = GatingEvaluator(config)
        result = evaluator.evaluate({"precision": 0.85, "faithfulness": 0.65})
        print(result.exit_code)  # 2 (warning - faithfulness below threshold)
    """

    def __init__(self, config: GatingConfig | None = None) -> None:
        """Initialize GatingEvaluator.

        Args:
            config: Gating configuration. Uses defaults if not provided.
        """
        self.config = config or GatingConfig()

    def evaluate(self, metrics: dict[str, float]) -> GatingEvaluationResult:
        """Evaluate metrics against gating thresholds.

        Args:
            metrics: Dictionary of metric names to values.

        Returns:
            GatingEvaluationResult with overall result and per-metric details.
        """
        results: list[MetricResult] = []
        has_stable_fail = False
        has_unstable_warn = False
        lower_is_better = set(self.config.lower_is_better)

        # Check stable metrics
        for name, threshold in self.config.stable.items():
            if name in metrics:
                value = metrics[name]
                passed = value <= threshold if name in lower_is_better else value >= threshold
                result = GatingResult.PASS if passed else GatingResult.FAIL

                if not passed:
                    has_stable_fail = True

                results.append(
                    MetricResult(
                        name=name,
                        value=value,
                        threshold=threshold,
                        category=MetricCategory.STABLE,
                        result=result,
                    )
                )

        # Check unstable metrics
        for name, threshold in self.config.unstable.items():
            if name in metrics:
                value = metrics[name]
                passed = value <= threshold if name in lower_is_better else value >= threshold
                result = GatingResult.PASS if passed else GatingResult.WARN

                if not passed:
                    has_unstable_warn = True

                results.append(
                    MetricResult(
                        name=name,
                        value=value,
                        threshold=threshold,
                        category=MetricCategory.UNSTABLE,
                        result=result,
                    )
                )

        # Determine overall result and exit code
        if has_stable_fail:
            overall_result = GatingResult.FAIL
            exit_code = 1
            failed = [r for r in results if r.result == GatingResult.FAIL]
            failed_names = ", ".join(r.name for r in failed)
            summary = f"Gating FAILED: Stable metrics below threshold: {failed_names}"
        elif has_unstable_warn:
            overall_result = GatingResult.WARN
            exit_code = 2
            warned = [r for r in results if r.result == GatingResult.WARN]
            warned_names = ", ".join(r.name for r in warned)
            summary = f"Gating WARNING: Unstable metrics below threshold: {warned_names}"
        else:
            overall_result = GatingResult.PASS
            exit_code = 0
            summary = "All metrics passed gating thresholds."

        return GatingEvaluationResult(
            overall_result=overall_result,
            exit_code=exit_code,
            metrics=results,
            summary=summary,
        )
