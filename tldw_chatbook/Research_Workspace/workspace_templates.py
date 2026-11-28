"""Confirmed workspace presets and their editable starter notes."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ResearchWorkspaceTemplate:
    name: str
    description: str
    note_title: str
    outline: str
    source_checklist: str
    next_steps: str

    @property
    def note(self) -> str:
        return (
            f"## Outline\n\n{self.outline}\n\n"
            f"## Source checklist\n\n{self.source_checklist}\n\n"
            f"## Next steps\n\n{self.next_steps}\n"
        )


WORKSPACE_TEMPLATES = MappingProxyType(
    {
        "literature-review": ResearchWorkspaceTemplate(
            "Literature Review",
            "Compare published evidence and identify gaps in the literature.",
            "Literature Review Plan",
            "Define the research question, inclusion criteria, and review period.",
            "- [ ] Add primary studies\n- [ ] Add relevant reviews and background sources",
            "Compare study methods and findings. Record disagreements and evidence gaps.",
        ),
        "interview-analysis": ResearchWorkspaceTemplate(
            "Interview Analysis",
            "Trace interview themes and contrast participant experiences.",
            "Interview Analysis Plan",
            "Define the interview questions, participant groups, and coding approach.",
            "- [ ] Add transcripts cleared for this workspace\n- [ ] Add the interview guide",
            "Summarize themes with supporting quotes. Look for cases that challenge each theme.",
        ),
        "product-brief": ResearchWorkspaceTemplate(
            "Product Brief",
            "Connect a product decision to user evidence, constraints, and success measures.",
            "Product Brief Draft",
            "Describe the audience, problem, proposed outcome, and constraints.",
            "- [ ] Add relevant user research\n- [ ] Add product notes and decision records",
            "Compare the options. Cite the evidence for the recommendation and note open questions.",
        ),
    }
)
