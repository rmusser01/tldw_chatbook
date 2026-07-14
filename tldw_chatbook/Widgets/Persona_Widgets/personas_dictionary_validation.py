"""Warn-not-block validation for dictionary entries (Roleplay P1c).

Pure functions over API-named entry dicts. The regex probe goes through the
real parser chain (``_entry_from_payload`` -> ``ChatDictionary``) so the
wrap/slash/flag rules can never drift from the engine's.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...Character_Chat.local_chat_dictionary_service import _entry_from_payload


@dataclass(frozen=True)
class ValidationFinding:
    """One advisory finding about a dictionary entry.

    Args:
        code: Stable machine code (invalid_regex, duplicate_pattern,
            probability_zero, case_flag_on_regex).
        field: The entry field the finding is about.
        message: Human-readable explanation.
        entry_id: The positional entry id, or None when unavailable.
    """

    code: str
    field: str
    message: str
    entry_id: str | None


def validate_entries(entries: list[dict]) -> list[ValidationFinding]:
    """Returns advisory findings for a dictionary's entries.

    Args:
        entries: API-named entry dicts (as ``get_dictionary`` returns).

    Returns:
        Findings in entry order; empty when everything is clean.
    """
    findings: list[ValidationFinding] = []
    seen: dict[tuple[str, str], str | None] = {}
    for entry in entries:
        entry_id = entry.get("id")
        pattern = str(entry.get("pattern") or "")
        etype = str(entry.get("type") or "literal")

        if etype == "regex":
            probe = _entry_from_payload(entry)
            if not probe.is_regex:
                findings.append(ValidationFinding(
                    code="invalid_regex", field="pattern", entry_id=entry_id,
                    message="Pattern does not compile; the engine will treat it as a literal.",
                ))
            if entry.get("case_sensitive"):
                findings.append(ValidationFinding(
                    code="case_flag_on_regex", field="case_sensitive", entry_id=entry_id,
                    message="Case-sensitive is ignored for regex entries; use the /i flag instead.",
                ))

        key = (pattern, etype)
        if key in seen:
            findings.append(ValidationFinding(
                code="duplicate_pattern", field="pattern", entry_id=entry_id,
                message="Same pattern and type as an earlier entry; only one will usually fire.",
            ))
        else:
            seen[key] = entry_id

        probability = entry.get("probability")
        try:
            probability_value = float(probability) if probability is not None else None
        except (TypeError, ValueError):
            probability_value = None  # malformed: not zero, no finding; display layer falls back
        if probability_value is not None and probability_value == 0.0:
            findings.append(ValidationFinding(
                code="probability_zero", field="probability", entry_id=entry_id,
                message="Probability 0 means this entry can never fire.",
            ))
    return findings


__all__ = ["ValidationFinding", "validate_entries"]
