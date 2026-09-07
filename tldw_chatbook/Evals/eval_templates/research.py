# eval_templates/research.py
# Description: Research-report self-eval templates (task-16485)
"""
Research Templates
------------------

Templates for scoring deep-search research reports from stored verification
payloads (citation accuracy, quote grounding, claim support, gate pass-rate).
Deterministic -- no LLM is consulted.
"""

from pathlib import Path
from typing import Dict, Any

from .base import BaseTemplates

_RESEARCH_DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "eval_datasets"
    / "research_report_verification.json"
)


class ResearchTemplates(BaseTemplates):
    """Templates for research-report self-evaluation."""

    def _initialize_templates(self):
        """Initialize research templates."""
        self._templates = {
            "research_report": self._research_report_template(),
        }

    def _create_base_template(
        self,
        name: str,
        description: str,
        task_type: str,
        metric: str,
        category: str,
        subcategory: str,
        dataset_name: str,
        generation_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "description": description,
            "task_type": task_type,
            "dataset_name": dataset_name,
            "metric": metric,
            "generation_kwargs": generation_kwargs,
            "metadata": {
                "category": category,
                "subcategory": subcategory,
            },
        }

    def _research_report_template(self) -> Dict[str, Any]:
        """Score research reports from stored verification payloads.

        The bundled dataset carries synthetic payloads shaped like a
        completed run's ``citation_verification`` block plus gate counts;
        swap in payloads recorded by the live baseline script
        (Helper_Scripts/Benchmarks/record_research_baseline.py) to score
        real runs.
        """
        return self._create_base_template(
            name="Research Report Self-Eval",
            description=(
                "Scores deep-search research reports on citation accuracy, "
                "quote grounding, claim support, cited-sentence ratio, and "
                "gate pass-rate from stored verification payloads"
            ),
            task_type="research_report",
            metric="research_report_metrics",
            category="research",
            subcategory="self_eval",
            dataset_name=str(_RESEARCH_DATASET_PATH),
            generation_kwargs={},
        )
