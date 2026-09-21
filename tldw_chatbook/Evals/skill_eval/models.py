"""Pure data model for the skill eval sub-harness. No I/O, no UI imports."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Tuple

METHODOLOGY_VERSION = "skill-eval/1"


class SkillEvalDepth(str, Enum):
    """Evaluation depth: which layers run.

    QUICK = static only, STANDARD = static + judge, DEEP = all three layers.
    Str-mixin Enum: comparisons must use explicit ranks (see scoring), since
    ``>=`` compares the string values lexicographically.
    """

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


ProgressCallback = Callable[[int, int], None]


class CancelToken:
    """Cooperative cancellation flag (character_probe runner precedent).

    Layers poll ``is_cancelled`` between cells; a cancelled run keeps whatever
    partial results were already recorded.
    """

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation; idempotent."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled


def digest_skill(name: str, description: str, body: str) -> str:
    """Compute the sha256 provenance digest of a skill definition.

    Args:
        name: Skill name.
        description: Front-matter description.
        body: Full SKILL.md content the digest is taken over.

    Returns:
        Hex-encoded sha256 of the NUL-joined fields.
    """
    joined = "\x00".join((name, description, body))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvalTarget:
    id: str            # eval_models row id
    provider: str
    model_id: str
    name: str = ""


@dataclass(frozen=True)
class SkillSubject:
    name: str
    description: str
    body: str
    allowed_tools: Tuple[str, ...] = ()
    script_paths: Tuple[str, ...] = ()
    referenced_files: Tuple[str, ...] = ()
    bundle_paths: Tuple[str, ...] = ()
    source_kind: str = "store"          # "store" | "directory"
    source_path: str = ""
    trust_status: str = "unknown"
    digest: str = ""
    line_count: int = 0

    def to_provenance(self) -> dict[str, Any]:
        """Return the provenance dict embedded in reports and run snapshots."""
        return {
            "name": self.name, "digest": self.digest,
            "trust_status": self.trust_status, "source_kind": self.source_kind,
            "source_path": self.source_path, "line_count": self.line_count,
            "allowed_tools": list(self.allowed_tools),
        }


@dataclass(frozen=True)
class SkillEvalConfig:
    name: str
    subject_ref: str                    # store skill name OR directory path
    subject_kind: str                   # "store" | "directory"
    depth: SkillEvalDepth
    generator_target_id: str
    judge_target_id: str
    bench_id: Optional[str] = None
    concurrency: int = 2
    seed: int = 0
    deep_sim_total: int = 50
    judge_temperature: float = 0.2
    sim_temperature: float = 0.7
    max_tokens: int = 1024

    def to_config_data(self) -> dict[str, Any]:
        """Serialize to the JSON-friendly dict persisted in ``config_data``.

        ``bench_id`` is intentionally excluded: it is the DB row identity and
        is re-attached by the loader.
        """
        return {
            "name": self.name, "subject_ref": self.subject_ref,
            "subject_kind": self.subject_kind, "depth": self.depth.value,
            "generator_target_id": self.generator_target_id,
            "judge_target_id": self.judge_target_id,
            "concurrency": self.concurrency, "seed": self.seed,
            "deep_sim_total": self.deep_sim_total,
            "judge_temperature": self.judge_temperature,
            "sim_temperature": self.sim_temperature, "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_config_data(cls, data: Mapping[str, Any]) -> "SkillEvalConfig":
        """Rebuild a config from persisted ``config_data``.

        Args:
            data: Mapping as produced by ``to_config_data`` (plus optional
                ``bench_id``).

        Returns:
            The reconstructed ``SkillEvalConfig``.

        Raises:
            KeyError: If a required key is missing.
            ValueError: If ``depth`` is not a known ``SkillEvalDepth`` value.
        """
        return cls(
            name=data["name"], subject_ref=data["subject_ref"],
            subject_kind=data["subject_kind"],
            depth=SkillEvalDepth(data["depth"]),
            generator_target_id=data["generator_target_id"],
            judge_target_id=data["judge_target_id"],
            bench_id=data.get("bench_id"),
            concurrency=int(data.get("concurrency", 2)),
            seed=int(data.get("seed", 0)),
            deep_sim_total=int(data.get("deep_sim_total", 50)),
            judge_temperature=float(data.get("judge_temperature", 0.2)),
            sim_temperature=float(data.get("sim_temperature", 0.7)),
            max_tokens=int(data.get("max_tokens", 1024)),
        )


@dataclass(frozen=True)
class StaticFinding:
    code: str
    penalty: float
    remediation: str


@dataclass(frozen=True)
class StaticLayerResult:
    sub_scores: dict[str, float]
    dimension_scores: dict[str, Optional[float]]   # None where static weight is 0
    findings: Tuple[StaticFinding, ...] = ()


@dataclass(frozen=True)
class JudgeLayerResult:
    rubrics: dict[str, float]                      # 0..1 (points/5)
    trigger_f1: Optional[float] = None
    trigger_precision: Optional[float] = None
    trigger_recall: Optional[float] = None
    artifacts: Tuple[dict, ...] = ()
    failed: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SimLayerResult:
    activation: Optional[float] = None
    activation_ci: Optional[Tuple[float, float]] = None
    consistency: Optional[float] = None
    consistency_ci: Optional[Tuple[float, float]] = None
    failure_rate: Optional[float] = None
    failure_ci: Optional[Tuple[float, float]] = None
    cells: Tuple[dict, ...] = ()


@dataclass(frozen=True)
class DimensionScore:
    name: str
    weight: float
    blended: float
    available_layers: Tuple[str, ...]


@dataclass(frozen=True)
class SkillEvalReport:
    provenance: dict[str, Any]
    depth: str
    dimensions: Tuple[DimensionScore, ...]
    composite: float
    grade: str
    confidence: str
    findings: Tuple[StaticFinding, ...] = ()
    warnings: Tuple[str, ...] = ()
    methodology_version: str = METHODOLOGY_VERSION
    layer_summaries: dict[str, Any] = field(default_factory=dict)
