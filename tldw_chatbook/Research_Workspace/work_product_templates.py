"""Grounded Markdown templates for the text work products Studio can save."""

from dataclasses import dataclass
from types import MappingProxyType

from .studio_models import ResearchWorkProductKind


@dataclass(frozen=True, slots=True)
class TextWorkProductTemplate:
    sections: tuple[str, ...]
    guidance: str

    @property
    def instruction(self) -> str:
        headings = "\n".join(f"## {section}" for section in self.sections)
        return (
            f"{self.guidance} Use these Markdown headings in order and write a nonempty body under each. "
            "Cite relevant supplied [S#] evidence and identify any limits in the evidence.\n"
            f"{headings}"
        )


TEXT_WORK_PRODUCT_TEMPLATES = MappingProxyType(
    {
        ResearchWorkProductKind.EXECUTIVE_BRIEF: TextWorkProductTemplate(
            ("Situation", "Key Findings", "Evidence", "Risks", "Recommended Actions"),
            "Write a decision-oriented executive brief. Identify the most consequential findings and which risks or recommendations need more evidence.",
        ),
        ResearchWorkProductKind.EVIDENCE_BOUND_HYPOTHESES: TextWorkProductTemplate(
            (
                "Hypotheses",
                "Support",
                "Predictions",
                "Methods",
                "Validity",
                "Confidence",
            ),
            "Generate testable hypotheses supported by the excerpts. Give each a prediction and a way to test it, and identify alternative explanations and limits on confidence.",
        ),
        ResearchWorkProductKind.RESEARCH_PROPOSAL_PACK: TextWorkProductTemplate(
            (
                "Research Question",
                "Literature",
                "Gaps",
                "Hypotheses",
                "Methods",
                "Risks",
                "Source Audit",
            ),
            "Draft a research proposal tied to the available literature. Explain how the proposed method addresses the evidence gaps and audit which sources support it.",
        ),
    }
)
