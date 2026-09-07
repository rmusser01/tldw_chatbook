"""Typed Console conversation context-policy contracts and resolution.

The policy deliberately keeps provider capability facts separate from user
intent.  Model windows and provider input caps are request-time inputs; only
global defaults and sparse conversation overrides are persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Mapping


DEFAULT_COMPACTION_TRIGGER_RATIO = 0.80
DEFAULT_COMPACTION_TARGET_RATIO = 0.55
DEFAULT_SUMMARY_MAX_TOKENS = 1024
MINIMUM_TRIGGER_TARGET_GAP = 0.15


class ContextPolicyError(ValueError):
    """Raised when persisted or user-supplied context policy is invalid."""


class ContextBudgetMode(str, Enum):
    AUTOMATIC = "automatic"
    CUSTOM = "custom"


class ContextCompactionMode(str, Enum):
    ASK = "ask"
    AUTOMATIC = "automatic"
    OFF = "off"


class ContextCompactionRepresentation(str, Enum):
    TEXT_SUMMARY = "text_summary"
    VISUAL_TRANSCRIPT = "visual_transcript"
    HYBRID = "hybrid"


class CompactionFailureBehavior(str, Enum):
    STOP_AND_ASK = "stop_and_ask"
    OMIT_OLDER_CONTEXT = "omit_older_context"


class ContextCarryForwardMode(str, Enum):
    MEMORY_WITH_RECENT_TURNS = "memory_with_recent_turns"
    MEMORY_WITH_LATEST_EXCHANGE = "memory_with_latest_exchange"


@dataclass(frozen=True)
class ConsoleContextPolicyDefaults:
    """Concrete context-policy values after application/global precedence."""

    budget_mode: ContextBudgetMode = ContextBudgetMode.AUTOMATIC
    custom_budget_tokens: int | None = None
    compaction_mode: ContextCompactionMode = ContextCompactionMode.ASK
    compaction_representation: ContextCompactionRepresentation = (
        ContextCompactionRepresentation.TEXT_SUMMARY
    )
    trigger_ratio: float = DEFAULT_COMPACTION_TRIGGER_RATIO
    target_ratio: float = DEFAULT_COMPACTION_TARGET_RATIO
    summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS
    failure_behavior: CompactionFailureBehavior = CompactionFailureBehavior.STOP_AND_ASK
    carry_forward_mode: ContextCarryForwardMode = (
        ContextCarryForwardMode.MEMORY_WITH_RECENT_TURNS
    )

    def __post_init__(self) -> None:
        _validate_complete_policy(self)


@dataclass(frozen=True)
class ConsoleContextPolicyOverrides:
    """Sparse per-conversation or global overrides.

    ``None`` means inherit that individual field.  A custom token value is
    retained even while Automatic mode is selected so switching models or
    modes never silently destroys the user's previous custom intent.
    """

    budget_mode: ContextBudgetMode | None = None
    custom_budget_tokens: int | None = None
    compaction_mode: ContextCompactionMode | None = None
    compaction_representation: ContextCompactionRepresentation | None = None
    trigger_ratio: float | None = None
    target_ratio: float | None = None
    summary_max_tokens: int | None = None
    failure_behavior: CompactionFailureBehavior | None = None
    carry_forward_mode: ContextCarryForwardMode | None = None

    def __post_init__(self) -> None:
        _validate_optional_enum("budget_mode", self.budget_mode, ContextBudgetMode)
        _validate_optional_enum(
            "compaction_mode", self.compaction_mode, ContextCompactionMode
        )
        _validate_optional_enum(
            "compaction_representation",
            self.compaction_representation,
            ContextCompactionRepresentation,
        )
        _validate_optional_enum(
            "failure_behavior", self.failure_behavior, CompactionFailureBehavior
        )
        _validate_optional_enum(
            "carry_forward_mode",
            self.carry_forward_mode,
            ContextCarryForwardMode,
        )
        _validate_optional_positive_int(
            "custom_budget_tokens", self.custom_budget_tokens
        )
        _validate_optional_positive_int("summary_max_tokens", self.summary_max_tokens)
        _validate_optional_ratio("trigger_ratio", self.trigger_ratio)
        _validate_optional_ratio("target_ratio", self.target_ratio)
        if self.trigger_ratio is not None and self.target_ratio is not None:
            _validate_hysteresis(self.trigger_ratio, self.target_ratio)

    @property
    def is_empty(self) -> bool:
        return all(getattr(self, item.name) is None for item in fields(self))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe sparse representation."""
        payload: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None:
                continue
            payload[item.name] = value.value if isinstance(value, Enum) else value
        return payload

    @classmethod
    def from_mapping(
        cls, source: Mapping[str, object] | None
    ) -> "ConsoleContextPolicyOverrides":
        """Parse a sparse policy mapping with strict value validation."""
        if source is None:
            return cls()
        if not isinstance(source, Mapping):
            raise ContextPolicyError("Context policy overrides must be a mapping.")
        return cls(
            budget_mode=_optional_enum(source, "budget_mode", ContextBudgetMode),
            custom_budget_tokens=_optional_int(source, "custom_budget_tokens"),
            compaction_mode=_optional_enum(
                source, "compaction_mode", ContextCompactionMode
            ),
            compaction_representation=_optional_enum(
                source,
                "compaction_representation",
                ContextCompactionRepresentation,
            ),
            trigger_ratio=_optional_float(source, "trigger_ratio"),
            target_ratio=_optional_float(source, "target_ratio"),
            summary_max_tokens=_optional_int(source, "summary_max_tokens"),
            failure_behavior=_optional_enum(
                source, "failure_behavior", CompactionFailureBehavior
            ),
            carry_forward_mode=_optional_enum(
                source, "carry_forward_mode", ContextCarryForwardMode
            ),
        )


@dataclass(frozen=True)
class ConsoleContextCapacity:
    """Request-time model/provider capacity facts; never persisted as policy."""

    model_context_window_tokens: int | None
    provider_input_cap_tokens: int | None = None
    response_reservation_tokens: int = 0
    safety_margin_tokens: int = 0
    mandatory_input_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "model_context_window_tokens",
            "provider_input_cap_tokens",
        ):
            value = getattr(self, name)
            if value is not None:
                _validate_positive_int(name, value)
        for name in (
            "response_reservation_tokens",
            "safety_margin_tokens",
            "mandatory_input_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ContextPolicyError(f"{name} must be a non-negative integer.")


@dataclass(frozen=True)
class ResolvedConsoleContextPolicy:
    """Effective policy plus explicit safety/validation state for the UI."""

    policy: ConsoleContextPolicyDefaults
    model_context_window_tokens: int | None
    safe_input_ceiling_tokens: int | None
    available_conversation_capacity_tokens: int | None
    effective_conversation_budget_tokens: int | None
    safety_verified: bool
    validation_errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def can_compact(self) -> bool:
        return (
            self.policy.compaction_mode is not ContextCompactionMode.OFF
            and self.effective_conversation_budget_tokens is not None
            and not self.validation_errors
        )


def application_context_policy_defaults() -> ConsoleContextPolicyDefaults:
    """Return immutable application defaults from ADR-052."""
    return ConsoleContextPolicyDefaults()


def context_policy_overrides_from_console_config(
    console_config: Mapping[str, object] | None,
) -> ConsoleContextPolicyOverrides:
    """Read canonical global Console Behavior keys as sparse overrides."""
    source = console_config or {}
    translated = {
        "budget_mode": source.get("conversation_budget_mode"),
        "custom_budget_tokens": source.get("conversation_budget_tokens"),
        "compaction_mode": source.get("compaction_mode"),
        "compaction_representation": source.get("compaction_representation"),
        "trigger_ratio": source.get("compaction_trigger_ratio"),
        "target_ratio": source.get("compaction_target_ratio"),
        "summary_max_tokens": source.get("compaction_summary_max_tokens"),
        "failure_behavior": source.get("compaction_failure_behavior"),
        "carry_forward_mode": source.get("compaction_carry_forward_mode"),
    }
    return ConsoleContextPolicyOverrides.from_mapping(
        {key: value for key, value in translated.items() if value is not None}
    )


def merge_context_policy(
    *,
    application_defaults: ConsoleContextPolicyDefaults | None = None,
    global_overrides: ConsoleContextPolicyOverrides | None = None,
    conversation_overrides: ConsoleContextPolicyOverrides | None = None,
) -> ConsoleContextPolicyDefaults:
    """Resolve field-by-field precedence: conversation > global > app."""
    base = application_defaults or application_context_policy_defaults()
    global_policy = global_overrides or ConsoleContextPolicyOverrides()
    conversation = conversation_overrides or ConsoleContextPolicyOverrides()

    def selected(name: str) -> object:
        local_value = getattr(conversation, name)
        if local_value is not None:
            return local_value
        global_value = getattr(global_policy, name)
        if global_value is not None:
            return global_value
        return getattr(base, name)

    return ConsoleContextPolicyDefaults(
        budget_mode=selected("budget_mode"),  # type: ignore[arg-type]
        custom_budget_tokens=selected("custom_budget_tokens"),  # type: ignore[arg-type]
        compaction_mode=selected("compaction_mode"),  # type: ignore[arg-type]
        compaction_representation=selected("compaction_representation"),  # type: ignore[arg-type]
        trigger_ratio=selected("trigger_ratio"),  # type: ignore[arg-type]
        target_ratio=selected("target_ratio"),  # type: ignore[arg-type]
        summary_max_tokens=selected("summary_max_tokens"),  # type: ignore[arg-type]
        failure_behavior=selected("failure_behavior"),  # type: ignore[arg-type]
        carry_forward_mode=selected("carry_forward_mode"),  # type: ignore[arg-type]
    )


def resolve_context_policy(
    *,
    capacity: ConsoleContextCapacity,
    application_defaults: ConsoleContextPolicyDefaults | None = None,
    global_overrides: ConsoleContextPolicyOverrides | None = None,
    conversation_overrides: ConsoleContextPolicyOverrides | None = None,
) -> ResolvedConsoleContextPolicy:
    """Resolve policy and capacity without rewriting stored custom intent."""
    policy = merge_context_policy(
        application_defaults=application_defaults,
        global_overrides=global_overrides,
        conversation_overrides=conversation_overrides,
    )
    errors: list[str] = []
    warnings: list[str] = []

    model_window = capacity.model_context_window_tokens
    context_derived_ceiling: int | None = None
    if model_window is not None:
        context_derived_ceiling = (
            model_window
            - capacity.response_reservation_tokens
            - capacity.safety_margin_tokens
        )
        if context_derived_ceiling <= 0:
            errors.append(
                "Response reservation and safety margin leave no model input capacity."
            )

    ceiling_candidates = [
        value
        for value in (
            context_derived_ceiling,
            capacity.provider_input_cap_tokens,
        )
        if value is not None and value > 0
    ]
    safe_input_ceiling = min(ceiling_candidates) if ceiling_candidates else None
    available_capacity = (
        safe_input_ceiling - capacity.mandatory_input_tokens
        if safe_input_ceiling is not None
        else None
    )
    if available_capacity is not None and available_capacity <= 0:
        errors.append("Mandatory request material leaves no conversation capacity.")
        available_capacity = 0

    effective_budget: int | None
    safety_verified = model_window is not None and safe_input_ceiling is not None
    if policy.budget_mode is ContextBudgetMode.AUTOMATIC:
        if model_window is None:
            errors.append(
                "Automatic conversation budget requires a known model context window."
            )
            effective_budget = None
        elif available_capacity is None or available_capacity <= 0:
            effective_budget = None
        else:
            effective_budget = available_capacity
    else:
        custom_budget = policy.custom_budget_tokens
        if custom_budget is None:
            errors.append("Custom conversation budget requires a positive token value.")
            effective_budget = None
        elif available_capacity is None:
            effective_budget = custom_budget
            warnings.append(
                "Custom budget can trigger compaction, but provider safety is unverified "
                "because the model context window is unknown."
            )
        else:
            effective_budget = min(custom_budget, available_capacity)
            if custom_budget > available_capacity:
                warnings.append(
                    "The saved custom budget exceeds current model capacity; its effective "
                    "value is lower for this request and saved intent was preserved."
                )

    return ResolvedConsoleContextPolicy(
        policy=policy,
        model_context_window_tokens=model_window,
        safe_input_ceiling_tokens=safe_input_ceiling,
        available_conversation_capacity_tokens=available_capacity,
        effective_conversation_budget_tokens=effective_budget,
        safety_verified=safety_verified,
        validation_errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _validate_complete_policy(policy: ConsoleContextPolicyDefaults) -> None:
    _validate_enum("budget_mode", policy.budget_mode, ContextBudgetMode)
    _validate_enum("compaction_mode", policy.compaction_mode, ContextCompactionMode)
    _validate_enum(
        "compaction_representation",
        policy.compaction_representation,
        ContextCompactionRepresentation,
    )
    _validate_enum(
        "failure_behavior", policy.failure_behavior, CompactionFailureBehavior
    )
    _validate_enum(
        "carry_forward_mode",
        policy.carry_forward_mode,
        ContextCarryForwardMode,
    )
    _validate_optional_positive_int("custom_budget_tokens", policy.custom_budget_tokens)
    _validate_positive_int("summary_max_tokens", policy.summary_max_tokens)
    _validate_ratio("trigger_ratio", policy.trigger_ratio)
    _validate_ratio("target_ratio", policy.target_ratio)
    _validate_hysteresis(policy.trigger_ratio, policy.target_ratio)


def _validate_hysteresis(trigger_ratio: float, target_ratio: float) -> None:
    if trigger_ratio > 0.95:
        raise ContextPolicyError("trigger_ratio must be no greater than 0.95.")
    if target_ratio >= trigger_ratio:
        raise ContextPolicyError("target_ratio must be lower than trigger_ratio.")
    if trigger_ratio - target_ratio < MINIMUM_TRIGGER_TARGET_GAP:
        raise ContextPolicyError(
            "trigger_ratio and target_ratio must differ by at least 0.15."
        )


def _validate_optional_positive_int(name: str, value: int | None) -> None:
    if value is not None:
        _validate_positive_int(name, value)


def _validate_optional_enum(
    name: str, value: Enum | None, enum_type: type[Enum]
) -> None:
    if value is not None:
        _validate_enum(name, value, enum_type)


def _validate_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise ContextPolicyError(f"{name} must be a {enum_type.__name__} value.")


def _validate_positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ContextPolicyError(f"{name} must be a positive integer.")


def _validate_optional_ratio(name: str, value: float | None) -> None:
    if value is not None:
        _validate_ratio(name, value)


def _validate_ratio(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContextPolicyError(f"{name} must be a number between 0 and 1.")
    if not 0 < float(value) < 1:
        raise ContextPolicyError(f"{name} must be a number between 0 and 1.")


def _optional_enum(
    source: Mapping[str, object], key: str, enum_type: type[Enum]
) -> Enum | None:
    value = source.get(key)
    if value is None:
        return None
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ContextPolicyError(f"{key} must be a string.")
    try:
        return enum_type(value.strip().lower())
    except ValueError as exc:
        raise ContextPolicyError(f"Unsupported {key}: {value!r}.") from exc


def _optional_int(source: Mapping[str, object], key: str) -> int | None:
    value = source.get(key)
    if value is None:
        return None
    if type(value) is int:
        return value
    if isinstance(value, str) and value.strip().isdecimal():
        return int(value.strip())
    raise ContextPolicyError(f"{key} must be an integer.")


def _optional_float(source: Mapping[str, object], key: str) -> float | None:
    value = source.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ContextPolicyError(f"{key} must be a number.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ContextPolicyError(f"{key} must be a number.") from exc
