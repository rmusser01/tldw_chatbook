"""Validated, content-safe configuration for custom trace PII rules."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

CUSTOM_PII_RULESET_VERSION = 1
CUSTOM_PII_DETECTOR_VERSION = "custom-pii-v1"
MAX_CUSTOM_PII_RULES = 64
MAX_CUSTOM_PII_PATTERN_CHARS = 2_048
_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*\Z", re.ASCII)
_INLINE_FLAG_PATTERN = re.compile(r"(?<!\\)\(\?[aiLmsux-]+(?::|\))")
_FLAG_BITS = {
    "ascii": re.ASCII,
    "dotall": re.DOTALL,
    "ignorecase": re.IGNORECASE,
    "multiline": re.MULTILINE,
}
_DIAGNOSTIC_MESSAGES = {
    "duplicate_rule_id": "Use a unique rule ID.",
    "empty_match": "Use a pattern that always consumes at least one character.",
    "invalid_pattern": "Correct the regular-expression syntax.",
    "invalid_pattern_length": "Use a pattern between 1 and 2048 characters.",
    "invalid_revision_id": "Set revision_id to a canonical UUIDv4 value.",
    "invalid_rule": "Check the rule ID, label, category, enabled state, and priority.",
    "invalid_ruleset": "Use a versioned ruleset mapping with only supported fields.",
    "rule_count_limit": "Configure no more than 64 rules.",
    "unsupported_construct": "Move inline flags to the rule's flags list.",
    "unsupported_flag": "Use only ascii, dotall, ignorecase, or multiline.",
    "unsupported_version": "Set the custom PII ruleset version to 1.",
}


class _RuleModel(BaseModel):
    """Strict Pydantic boundary for one user-authored rule."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )

    rule_id: str = Field(alias="id", min_length=1, max_length=64, pattern=_TOKEN_PATTERN)
    label: str = Field(min_length=1, max_length=80)
    category: str = Field(min_length=1, max_length=64, pattern=_TOKEN_PATTERN)
    pattern: str = Field(min_length=1, max_length=MAX_CUSTOM_PII_PATTERN_CHARS, repr=False)
    flags: tuple[Literal["ascii", "dotall", "ignorecase", "multiline"], ...] = ()
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=1_000)

    @field_validator("flags", mode="before")
    @classmethod
    def _freeze_flags(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("flags")
    @classmethod
    def _unique_flags(
        cls,
        value: tuple[Literal["ascii", "dotall", "ignorecase", "multiline"], ...],
    ) -> tuple[Literal["ascii", "dotall", "ignorecase", "multiline"], ...]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate flags")
        return tuple(sorted(value))


@dataclass(frozen=True, slots=True)
class CustomPIIRule:
    """One validated rule whose source pattern stays out of representations."""

    rule_id: str
    label: str
    category: str
    pattern: str = field(repr=False)
    flags: tuple[str, ...] = ()
    enabled: bool = True
    priority: int = 100


@dataclass(frozen=True, slots=True)
class CustomPIIRuleset:
    """One validated ruleset revision frozen for future trace captures."""

    version: Literal[1]
    revision_id: str
    rules: tuple[CustomPIIRule, ...]

    @property
    def runnable_rules(self) -> tuple[CustomPIIRule, ...]:
        """Return enabled rules in deterministic evaluation order."""

        return tuple(rule for rule in self.rules if rule.enabled)


@dataclass(frozen=True, slots=True)
class CustomPIIRuleDiagnostic:
    """Actionable validation failure containing no regex or matched content."""

    rule_id: str | None
    code: str
    message: str

    @property
    def display(self) -> str:
        """Return a content-free Settings-safe diagnostic label."""

        return f"{self.rule_id or 'ruleset'}: {self.code}"


@dataclass(frozen=True, slots=True)
class CustomPIIRulesValidation:
    """Validated runnable rules plus content-free diagnostics."""

    ruleset: CustomPIIRuleset | None
    diagnostics: tuple[CustomPIIRuleDiagnostic, ...]


def _diagnostic(code: str, rule_id: str | None = None) -> CustomPIIRuleDiagnostic:
    return CustomPIIRuleDiagnostic(rule_id, code, _DIAGNOSTIC_MESSAGES[code])


def _safe_rule_id(value: object) -> str | None:
    return (
        value
        if type(value) is str
        and len(value) <= 64
        and _TOKEN_PATTERN.fullmatch(value) is not None
        else None
    )


def _canonical_uuid4(value: object) -> str | None:
    if type(value) is not str:
        return None
    try:
        parsed = UUID(value)
    except ValueError:
        return None
    return value if parsed.version == 4 and str(parsed) == value else None


def _pattern_diagnostic(raw: Mapping[object, object], rule_id: str | None) -> str | None:
    flags = raw.get("flags", ())
    if not isinstance(flags, Sequence) or isinstance(flags, (str, bytes, bytearray)):
        return "invalid_rule"
    if any(type(flag) is not str or flag not in _FLAG_BITS for flag in flags):
        return "unsupported_flag"
    pattern = raw.get("pattern")
    if type(pattern) is not str:
        return "invalid_rule"
    if not 0 < len(pattern) <= MAX_CUSTOM_PII_PATTERN_CHARS:
        return "invalid_pattern_length"
    if _INLINE_FLAG_PATTERN.search(pattern) is not None:
        return "unsupported_construct"
    bits = 0
    for flag in flags:
        bits |= _FLAG_BITS[flag]
    try:
        compiled = re.compile(pattern, bits)
    except re.error:
        return "invalid_pattern"
    empty = compiled.search("")
    if empty is not None and empty.start() == empty.end():
        return "empty_match"
    return None


def _validated_rule(
    raw: object,
) -> tuple[CustomPIIRule | None, CustomPIIRuleDiagnostic | None]:
    if not isinstance(raw, Mapping):
        return None, _diagnostic("invalid_rule")
    rule_id = _safe_rule_id(raw.get("id"))
    pattern_problem = _pattern_diagnostic(raw, rule_id)
    if pattern_problem is not None:
        return None, _diagnostic(pattern_problem, rule_id)
    try:
        model = _RuleModel.model_validate(raw)
    except ValidationError:
        return None, _diagnostic("invalid_rule", rule_id)
    return (
        CustomPIIRule(
            rule_id=model.rule_id,
            label=model.label,
            category=model.category,
            pattern=model.pattern,
            flags=tuple(model.flags),
            enabled=model.enabled,
            priority=model.priority,
        ),
        None,
    )


def validate_custom_pii_rules_config(value: object) -> CustomPIIRulesValidation:
    """Validate a custom ruleset without exposing pattern text in failures.

    Args:
        value: Raw ``console.trace_custom_pii_rules`` configuration, or None.

    Returns:
        Runnable validated rules and content-free diagnostics. Invalid rules
        are excluded individually; an invalid envelope disables the batch.
    """

    if value is None:
        return CustomPIIRulesValidation(None, ())
    if not isinstance(value, Mapping) or set(value) != {
        "version",
        "revision_id",
        "rules",
    }:
        return CustomPIIRulesValidation(None, (_diagnostic("invalid_ruleset"),))
    if value.get("version") != CUSTOM_PII_RULESET_VERSION or type(
        value.get("version")
    ) is not int:
        return CustomPIIRulesValidation(None, (_diagnostic("unsupported_version"),))
    revision_id = _canonical_uuid4(value.get("revision_id"))
    if revision_id is None:
        return CustomPIIRulesValidation(None, (_diagnostic("invalid_revision_id"),))
    raw_rules = value.get("rules")
    if not isinstance(raw_rules, Sequence) or isinstance(
        raw_rules, (str, bytes, bytearray)
    ):
        return CustomPIIRulesValidation(None, (_diagnostic("invalid_ruleset"),))
    if len(raw_rules) > MAX_CUSTOM_PII_RULES:
        return CustomPIIRulesValidation(None, (_diagnostic("rule_count_limit"),))

    valid: list[CustomPIIRule] = []
    diagnostics: list[CustomPIIRuleDiagnostic] = []
    for raw in raw_rules:
        rule, diagnostic = _validated_rule(raw)
        if rule is not None:
            valid.append(rule)
        if diagnostic is not None:
            diagnostics.append(diagnostic)

    duplicate_ids = {
        rule_id for rule_id, count in Counter(rule.rule_id for rule in valid).items() if count > 1
    }
    if duplicate_ids:
        valid = [rule for rule in valid if rule.rule_id not in duplicate_ids]
        diagnostics.extend(
            _diagnostic("duplicate_rule_id", rule_id)
            for rule_id in sorted(duplicate_ids)
        )
    ordered = tuple(sorted(valid, key=lambda rule: (rule.priority, rule.rule_id)))
    return CustomPIIRulesValidation(
        CustomPIIRuleset(CUSTOM_PII_RULESET_VERSION, revision_id, ordered),
        tuple(diagnostics),
    )
