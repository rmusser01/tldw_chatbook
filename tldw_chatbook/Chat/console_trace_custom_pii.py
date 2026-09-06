"""Validated, content-safe configuration for custom trace PII rules."""

from __future__ import annotations

import re
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tldw_chatbook.Chat.console_trace_redaction import (
    BUILTIN_PII_RULESET_REVISION_ID,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_trace_redaction import PIIValueRedactionResult
    from tldw_chatbook.Chat.console_trace_regex_worker import CustomPIIWorkerLimits

CUSTOM_PII_RULESET_VERSION = 1
CUSTOM_PII_DETECTOR_VERSION = "custom-pii-v1"
MAX_CUSTOM_PII_RULES = 64
MAX_CUSTOM_PII_PATTERN_CHARS = 2_048
MAX_EPHEMERAL_CUSTOM_PII_RULESETS = 32
CUSTOM_PII_RULESET_UNAVAILABLE = "custom_pii_ruleset_unavailable"
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
    "reserved_revision_id": "Choose a revision ID that is not reserved by built-in rules.",
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

    rule_id: str = Field(
        alias="id", min_length=1, max_length=64, pattern=_TOKEN_PATTERN
    )
    label: str = Field(min_length=1, max_length=80)
    category: str = Field(min_length=1, max_length=64, pattern=_TOKEN_PATTERN)
    pattern: str = Field(
        min_length=1, max_length=MAX_CUSTOM_PII_PATTERN_CHARS, repr=False
    )
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


class _RulesetModel(BaseModel):
    """Strict Pydantic boundary for the versioned ruleset envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1]
    revision_id: str = Field(min_length=36, max_length=36)
    rules: tuple[object, ...] = Field(max_length=MAX_CUSTOM_PII_RULES)

    @field_validator("rules", mode="before")
    @classmethod
    def _freeze_rules(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("revision_id")
    @classmethod
    def _canonical_revision_id(cls, value: str) -> str:
        if _canonical_uuid4(value) is None:
            raise ValueError("invalid revision")
        return value


@dataclass(frozen=True, slots=True)
class CustomPIIRule:
    """One validated rule whose source pattern stays out of representations."""

    rule_id: str
    label: str = field(repr=False)
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


_RULESET_REGISTRY_LOCK = RLock()
_RULESET_REGISTRY: OrderedDict[str, CustomPIIRuleset] = OrderedDict()


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


def _pattern_diagnostic(raw: Mapping[object, object]) -> str | None:
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
        re.compile(pattern, bits)
    except re.error:
        return "invalid_pattern"
    parser = getattr(re, "_parser", None)
    if parser is None:
        return "unsupported_construct"
    try:
        minimum_width = parser.parse(pattern, bits).getwidth()[0]
    except (AttributeError, re.error):
        return "invalid_pattern"
    if minimum_width == 0:
        return "empty_match"
    return None


def _validated_rule(
    raw: object,
) -> tuple[CustomPIIRule | None, CustomPIIRuleDiagnostic | None]:
    if not isinstance(raw, Mapping):
        return None, _diagnostic("invalid_rule")
    rule_id = _safe_rule_id(raw.get("id"))
    pattern_problem = _pattern_diagnostic(raw)
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
    try:
        envelope = _RulesetModel.model_validate(value)
    except ValidationError as error:
        details = error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        if any(
            item["type"] in {"extra_forbidden", "missing", "model_type", "tuple_type"}
            for item in details
        ):
            code = "invalid_ruleset"
        elif any(item["loc"] == ("version",) for item in details):
            code = "unsupported_version"
        elif any(item["loc"] == ("revision_id",) for item in details):
            code = "invalid_revision_id"
        elif any(
            item["loc"] == ("rules",) and item["type"] == "too_long" for item in details
        ):
            code = "rule_count_limit"
        else:
            code = "invalid_ruleset"
        return CustomPIIRulesValidation(None, (_diagnostic(code),))
    revision_id = envelope.revision_id
    if revision_id == BUILTIN_PII_RULESET_REVISION_ID:
        return CustomPIIRulesValidation(None, (_diagnostic("reserved_revision_id"),))
    raw_rules = envelope.rules

    valid: list[CustomPIIRule] = []
    diagnostics: list[CustomPIIRuleDiagnostic] = []
    for raw in raw_rules:
        rule, diagnostic = _validated_rule(raw)
        if rule is not None:
            valid.append(rule)
        if diagnostic is not None:
            diagnostics.append(diagnostic)

    duplicate_ids = {
        rule_id
        for rule_id, count in Counter(rule.rule_id for rule in valid).items()
        if count > 1
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


def register_custom_pii_ruleset(ruleset: CustomPIIRuleset) -> bool:
    """Retain one bounded process-local ruleset without revising its identity.

    Args:
        ruleset: Validated immutable ruleset to make available for new masks.

    Returns:
        True when the revision is registered or already identical. False when
        the revision ID was reused for different rule content.
    """

    if not isinstance(ruleset, CustomPIIRuleset):
        raise TypeError("ruleset")
    with _RULESET_REGISTRY_LOCK:
        existing = _RULESET_REGISTRY.get(ruleset.revision_id)
        if existing is not None and existing != ruleset:
            return False
        _RULESET_REGISTRY[ruleset.revision_id] = ruleset
        _RULESET_REGISTRY.move_to_end(ruleset.revision_id)
        while len(_RULESET_REGISTRY) > MAX_EPHEMERAL_CUSTOM_PII_RULESETS:
            _RULESET_REGISTRY.popitem(last=False)
    return True


def custom_pii_ruleset_for_revision(
    revision_id: str,
) -> CustomPIIRuleset | None:
    """Return one process-local immutable ruleset without exposing its source."""

    if type(revision_id) is not str:
        raise TypeError("revision_id")
    with _RULESET_REGISTRY_LOCK:
        return _RULESET_REGISTRY.get(revision_id)


def redact_pii_value_with_custom_rules(
    value: object,
    ruleset: CustomPIIRuleset,
    *,
    worker_limits: CustomPIIWorkerLimits | None = None,
) -> PIIValueRedactionResult:
    """Mask built-in and custom PII, failing closed as one component.

    Args:
        value: JSON-like capture component to inspect without mutation.
        ruleset: Validated, frozen custom rules for this capture.
        worker_limits: Optional process and structural limits for the custom batch.

    Returns:
        A :class:`PIIValueRedactionResult` containing merged content-free spans,
        or the worker's content-free omission reason when custom detection fails.
    """

    from tldw_chatbook.Chat.console_trace_redaction import (
        PIIValueRedactionResult,
        redact_pii_value,
    )

    if not isinstance(ruleset, CustomPIIRuleset):
        raise TypeError("ruleset")
    if not ruleset.runnable_rules:
        return redact_pii_value(value)
    from tldw_chatbook.Chat.console_trace_regex_worker import run_custom_pii_batch

    custom = run_custom_pii_batch(
        value,
        ruleset.runnable_rules,
        limits=worker_limits,
    )
    if not custom.available:
        return PIIValueRedactionResult(
            available=False,
            value=None,
            field_redactions=(),
            omission_reason_code=custom.omission_reason_code,
        )
    return redact_pii_value(
        value,
        additional_field_redactions=custom.field_redactions,
    )


def redact_pii_value_for_ruleset_revision(
    value: object,
    revision_id: str,
    *,
    worker_limits: CustomPIIWorkerLimits | None = None,
) -> PIIValueRedactionResult:
    """Apply the exact built-in or registered custom frozen policy revision."""

    from tldw_chatbook.Chat.console_trace_redaction import (
        BUILTIN_PII_RULESET_REVISION_ID,
        PIIValueRedactionResult,
        redact_pii_value,
    )

    if revision_id == BUILTIN_PII_RULESET_REVISION_ID:
        return redact_pii_value(value)
    ruleset = custom_pii_ruleset_for_revision(revision_id)
    if ruleset is None or not ruleset.runnable_rules:
        return PIIValueRedactionResult(
            available=False,
            value=None,
            field_redactions=(),
            omission_reason_code=CUSTOM_PII_RULESET_UNAVAILABLE,
        )
    return redact_pii_value_with_custom_rules(
        value,
        ruleset,
        worker_limits=worker_limits,
    )
