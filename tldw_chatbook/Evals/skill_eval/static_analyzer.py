"""Layer 1: deterministic analysis of a SkillSubject snapshot. Pure function."""

from __future__ import annotations

import re
from typing import Optional

from .models import SkillSubject, StaticFinding, StaticLayerResult

ANTIPATTERN_PENALTY = 0.05
PENALTY_FLOOR = 0.5
_TRIGGER_PHRASES = ("use when", "use this when", "use for", "when you")
_DIRECTIVE_RE = re.compile(r"\b(?:MUST|ALWAYS|NEVER)\b")

ANTI_PATTERN_CODES = frozenset({
    "EMPTY_DESCRIPTION", "MISSING_TRIGGER", "OVER_CONSTRAINED", "BLOATED_SKILL",
    "ORPHAN_REFERENCE", "UNKNOWN_TOOLS", "NAME_COLLISION",
})

_REMEDIATION = {
    "EMPTY_DESCRIPTION": "Write a one-to-three sentence description.",
    "MISSING_TRIGGER": 'Add trigger phrasing, e.g. "Use when ...".',
    "OVER_CONSTRAINED": "Reduce MUST/ALWAYS/NEVER directives to <= 15.",
    "BLOATED_SKILL": "Split the body or move detail into references/.",
    "ORPHAN_REFERENCE": "Add the referenced file to the skill package.",
    "UNKNOWN_TOOLS": "allowed_tools entries must be bare builtin/local tool names.",
    "NAME_COLLISION": "Rename; the name collides with a reserved tool or skill.",
}


def analyze_static(subject: SkillSubject, *, builtin_tool_names: frozenset[str],
                   local_tool_names: frozenset[str],
                   reserved_names: frozenset[str]) -> StaticLayerResult:
    """Layer 1: deterministic checks over a frozen skill snapshot.

    Runs the seven anti-pattern detectors (each a multiplicative penalty
    applied later by scoring) and computes per-dimension sub-scores.
    Dimensions a static-only view cannot measure get ``None``.

    Args:
        subject: Immutable skill snapshot.
        builtin_tool_names: Bare names of builtin tools (exact-match surface).
        local_tool_names: Bare names of local tools.
        reserved_names: Names the skill must not collide with.

    Returns:
        ``StaticLayerResult`` with sub-scores, per-dimension scores, findings.
    """
    findings: list[StaticFinding] = []

    desc = subject.description.strip()
    desc_lower = desc.lower()
    has_trigger = any(p in desc_lower for p in _TRIGGER_PHRASES)

    if not desc:
        findings.append(StaticFinding("EMPTY_DESCRIPTION", ANTIPATTERN_PENALTY,
                                      _REMEDIATION["EMPTY_DESCRIPTION"]))
    if not has_trigger:  # empty description lacks trigger phrasing too: both fire
        findings.append(StaticFinding("MISSING_TRIGGER", ANTIPATTERN_PENALTY,
                                      _REMEDIATION["MISSING_TRIGGER"]))
    if len(_DIRECTIVE_RE.findall(subject.body)) > 15:
        findings.append(StaticFinding("OVER_CONSTRAINED", ANTIPATTERN_PENALTY,
                                      _REMEDIATION["OVER_CONSTRAINED"]))
    if subject.line_count > 800 and not any(
            p.startswith(("references/", "assets/")) for p in subject.bundle_paths):
        findings.append(StaticFinding("BLOATED_SKILL", ANTIPATTERN_PENALTY,
                                      _REMEDIATION["BLOATED_SKILL"]))
    orphans = [r for r in subject.referenced_files if r not in subject.bundle_paths]
    if orphans:
        findings.append(StaticFinding("ORPHAN_REFERENCE", ANTIPATTERN_PENALTY,
                                      _REMEDIATION["ORPHAN_REFERENCE"]))
    known = builtin_tool_names | local_tool_names
    unknown = [t for t in subject.allowed_tools if t not in known]
    if unknown:
        findings.append(StaticFinding("UNKNOWN_TOOLS", ANTIPATTERN_PENALTY,
                                      _REMEDIATION["UNKNOWN_TOOLS"]))
    if subject.name in reserved_names:
        findings.append(StaticFinding("NAME_COLLISION", ANTIPATTERN_PENALTY,
                                      _REMEDIATION["NAME_COLLISION"]))

    # --- sub-scores (0..1), deliberately simple and deterministic -----------
    desc_len = len(desc)
    frontmatter_quality = (
        0.0 if not desc else
        1.0 if 40 <= desc_len <= 1000 else
        0.6 if desc_len < 40 else 0.8
    )
    trigger_quality = 1.0 if (has_trigger and desc_len >= 40) else (0.5 if desc else 0.0)
    words = subject.body.split()
    unique = len(set(w.lower() for w in words))
    density = (unique / len(words)) if words else 0.0
    token_efficiency = 1.0 if subject.line_count <= 400 else max(
        0.2, 1.0 - (subject.line_count - 400) / 800.0)
    token_efficiency = round(0.5 * token_efficiency + 0.5 * min(1.0, density * 4), 4)
    completeness = (
        (0.5 if subject.name else 0.0) + (0.5 if desc else 0.0)
    )
    structure_ok = subject.line_count <= 800 or bool(
        any(p.startswith("references/") for p in subject.bundle_paths))
    disclosure = 1.0 if structure_ok else 0.5
    tool_surface = 1.0 if not unknown else max(0.0, 1.0 - 0.5 * len(unknown))
    over_broad = [t for t in subject.allowed_tools
                  if t in known and t not in subject.body]
    tool_surface = round(tool_surface * (1.0 - 0.1 * len(over_broad)), 4)
    trust_surface = 1.0
    if subject.trust_status != "trusted" and (subject.allowed_tools or subject.script_paths):
        trust_surface = 0.7

    sub_scores = {
        "frontmatter_quality": frontmatter_quality,
        "trigger_quality": trigger_quality,
        "token_efficiency": token_efficiency,
        "structural_completeness": completeness,
        "progressive_disclosure": disclosure,
        "tool_surface_sanity": tool_surface,
        "trust_surface": trust_surface,
    }

    dimension_scores: dict[str, Optional[float]] = {
        "triggering_accuracy": round(0.5 * frontmatter_quality + 0.5 * trigger_quality, 4),
        "instruction_fitness": round(0.5 * disclosure + 0.25 * completeness
                                     + 0.25 * trust_surface, 4),
        "output_quality": None,
        "scope_calibration": round(0.5 * trust_surface + 0.5 * tool_surface, 4),
        "progressive_disclosure": disclosure,
        "tool_surface_sanity": tool_surface,
        "token_efficiency": token_efficiency,
        "robustness": None,
        "structural_completeness": completeness,
    }

    return StaticLayerResult(sub_scores=sub_scores,
                             dimension_scores=dimension_scores,
                             findings=tuple(findings))
