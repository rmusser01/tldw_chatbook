"""Pure, single-pass Prompt variable lexing and rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import math
import re
import time
from typing import Literal, TypeAlias


MAX_VARIABLES = 64
MAX_VARIABLE_NAME_LENGTH = 64
APPLICATION_TTL_SECONDS = 120.0

PromptLane: TypeAlias = Literal["system", "user"]
PromptVariableIssueCode: TypeAlias = Literal["name_too_long", "too_many_variables"]
PromptApplicationDestination: TypeAlias = Literal["replace_snapshot", "append_active"]

_COMPOSER_FINGERPRINT = re.compile(r"[0-9a-f]{64}")
_SYSTEM_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")


class PromptVariableValidationError(ValueError):
    """Report that a Prompt variable plan cannot be rendered safely."""


def validate_prompt_application_guards(
    *,
    destination: PromptApplicationDestination,
    target_session_id: str,
    composer_fingerprint: str | None,
    system_fingerprint: str | None,
    requires_system_fingerprint: bool,
) -> None:
    """Validate application identity without starting an application TTL.

    Args:
        destination: Composer replacement or active-draft append behavior.
        target_session_id: Exact Console session authorized by the caller.
        composer_fingerprint: Captured composer digest for replacement.
        system_fingerprint: Authorized System digest when required.
        requires_system_fingerprint: Whether the selected source has a System
            lane that requires authorization.

    Raises:
        TypeError: If the System-fingerprint requirement is not a bool.
        ValueError: If the destination, target, or fingerprints are invalid.
    """
    if type(requires_system_fingerprint) is not bool:
        raise TypeError("System fingerprint requirement must be a boolean")
    if type(destination) is not str or destination not in (
        "replace_snapshot",
        "append_active",
    ):
        raise ValueError("Prompt application destination is invalid")
    if (
        type(target_session_id) is not str
        or not target_session_id.strip()
        or target_session_id != target_session_id.strip()
    ):
        raise ValueError("Prompt application target session is invalid")

    if destination == "replace_snapshot":
        if not _is_composer_fingerprint(composer_fingerprint):
            raise ValueError("Prompt application composer fingerprint is invalid")
    elif composer_fingerprint is not None:
        raise ValueError("Append application forbids a composer fingerprint")

    if requires_system_fingerprint:
        if not _is_system_fingerprint(system_fingerprint):
            raise ValueError("Prompt application System fingerprint is invalid")
    elif system_fingerprint is not None:
        raise ValueError("Inactive System lane forbids a System fingerprint")


@dataclass(frozen=True, slots=True)
class PromptVariableApplication:
    """Carry one guarded, memory-only Prompt application.

    Attributes:
        system_text: Final selected System payload, or ``None`` when inactive.
        user_text: Final selected User payload, or ``None`` when inactive.
        apply_system: Whether to replace the authorized System lane.
        apply_user: Whether to apply the selected User lane.
        destination: Composer replacement or active-draft append behavior.
        target_session_id: Exact Console session authorized by the caller.
        composer_fingerprint: Captured composer snapshot digest for replacement.
        system_fingerprint: Authorized System text digest when System applies.
        created_monotonic: Monotonic request creation time.
        expires_monotonic: Derived exact application expiry time.
    """

    system_text: str | None = field(repr=False)
    user_text: str | None = field(repr=False)
    apply_system: bool
    apply_user: bool
    destination: PromptApplicationDestination
    target_session_id: str
    composer_fingerprint: str | None = field(repr=False)
    system_fingerprint: str | None = field(repr=False)
    created_monotonic: float = field(default_factory=time.monotonic)
    expires_monotonic: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.apply_system) is not bool or type(self.apply_user) is not bool:
            raise TypeError("Prompt application lane flags must be booleans")
        if not self.apply_system and not self.apply_user:
            raise ValueError("Prompt application requires at least one lane")
        _validate_lane_payload("System", self.system_text, self.apply_system)
        _validate_lane_payload("User", self.user_text, self.apply_user)
        validate_prompt_application_guards(
            destination=self.destination,
            target_session_id=self.target_session_id,
            composer_fingerprint=self.composer_fingerprint,
            system_fingerprint=self.system_fingerprint,
            requires_system_fingerprint=self.apply_system,
        )

        created = _finite_monotonic(
            self.created_monotonic,
            message="Prompt application creation time is invalid",
        )
        expires = created + APPLICATION_TTL_SECONDS
        if not math.isfinite(expires):
            raise ValueError("Prompt application creation time is invalid")
        object.__setattr__(self, "created_monotonic", created)
        object.__setattr__(self, "expires_monotonic", expires)

    def is_expired(self, *, now_monotonic: float | None = None) -> bool:
        """Return whether the application reached its exact expiry boundary.

        Args:
            now_monotonic: Caller-supplied monotonic time, or the current clock.

        Returns:
            ``True`` when current time is at or beyond the stored expiry.

        Raises:
            ValueError: If the supplied/current monotonic time is not finite.
        """
        now = time.monotonic() if now_monotonic is None else now_monotonic
        current = _finite_monotonic(
            now,
            message="Prompt application current monotonic time is invalid",
        )
        return current >= self.expires_monotonic


def fingerprint_system_text(text: str) -> str:
    """Return the existing canonical one-way System text fingerprint.

    Args:
        text: Exact current System text, including blank text.

    Returns:
        ``sha256:`` followed by the full lowercase hexadecimal digest.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("System fingerprint input must be text")
    return f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"


def _validate_lane_payload(
    lane: str,
    payload: str | None,
    applies: bool,
) -> None:
    if applies:
        if payload is None:
            raise ValueError(f"{lane} payload is required for an active lane")
        if not isinstance(payload, str):
            raise TypeError(f"{lane} payload must be text")
    elif payload is not None:
        raise ValueError(f"{lane} payload must be absent for an inactive lane")


def _is_composer_fingerprint(value: object) -> bool:
    return isinstance(value, str) and _COMPOSER_FINGERPRINT.fullmatch(value) is not None


def _is_system_fingerprint(value: object) -> bool:
    return isinstance(value, str) and _SYSTEM_FINGERPRINT.fullmatch(value) is not None


def _finite_monotonic(value: object, *, message: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(message)
    return float(value)


@dataclass(frozen=True, slots=True)
class PromptVariableSpec:
    """Describe one unique variable and its active lanes.

    Attributes:
        name: Exact, case-sensitive variable name.
        lanes: Active lanes containing the variable, in System/User order.
    """

    name: str = field(repr=False)
    lanes: tuple[PromptLane, ...]


@dataclass(frozen=True, slots=True)
class PromptVariableIssue:
    """Describe a bounded validation issue without retaining Prompt content.

    Attributes:
        code: Stable category for the violated limit.
        lane: Active lane containing the first offending placeholder.
        position: Character offset of that placeholder within its lane.
    """

    code: PromptVariableIssueCode
    lane: PromptLane
    position: int


@dataclass(frozen=True, slots=True)
class RenderedPromptLanes:
    """Hold rendered active lanes while hiding their content from representations.

    Attributes:
        system_text: Rendered System lane, or ``None`` when inactive.
        user_text: Rendered User lane, or ``None`` when inactive.
    """

    system_text: str | None = field(repr=False)
    user_text: str | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class _LiteralToken:
    text: str


@dataclass(frozen=True, slots=True)
class _VariableToken:
    name: str


_Token: TypeAlias = _LiteralToken | _VariableToken


@dataclass(frozen=True, slots=True)
class _LanePlan:
    tokens: tuple[_Token, ...]


@dataclass(frozen=True, slots=True)
class PromptVariablePlan:
    """Store an immutable tokenized render plan for active Prompt lanes.

    Attributes:
        variables: Unique variable specifications in first-occurrence order.
        issues: Bounded validation issues that prevent interpolated rendering.
    """

    variables: tuple[PromptVariableSpec, ...]
    issues: tuple[PromptVariableIssue, ...]
    _system: _LanePlan | None = field(repr=False)
    _user: _LanePlan | None = field(repr=False)

    @property
    def is_valid(self) -> bool:
        """Return whether the plan may be rendered."""
        return not self.issues

    def render(self, values: Mapping[str, str]) -> RenderedPromptLanes:
        """Render active lanes once using exact shared variable values.

        Args:
            values: Ephemeral values keyed by exact variable name. Blank values
                are valid; every variable must have an explicit entry.

        Returns:
            The rendered System and User lane text.

        Raises:
            PromptVariableValidationError: If the plan violates a limit, a
                variable value is missing, or a supplied value is not text.
        """
        if self.issues:
            raise PromptVariableValidationError(
                "Prompt variable plan has validation issues"
            )
        if any(variable.name not in values for variable in self.variables):
            raise PromptVariableValidationError(
                "Prompt variable plan has missing values"
            )
        if any(
            not isinstance(values[variable.name], str) for variable in self.variables
        ):
            raise PromptVariableValidationError("Prompt variable values must be text")
        return RenderedPromptLanes(
            system_text=_render_lane(self._system, values),
            user_text=_render_lane(self._user, values),
        )


def compile_prompt_variables(
    *,
    system_text: str | None = None,
    user_text: str | None = None,
) -> PromptVariablePlan:
    """Compile active Prompt lanes into a deterministic single-pass render plan.

    ``None`` marks a lane inactive; an empty string is an active blank lane.
    Variables are ordered by first occurrence in System text, then User text.

    Args:
        system_text: Active System lane text, or ``None``.
        user_text: Active User lane text, or ``None``.

    Returns:
        An immutable plan containing tokens, variable specifications, and any
        explicit limit issue.

    Raises:
        TypeError: If an active lane value is not text.
    """
    if system_text is not None and not isinstance(system_text, str):
        raise TypeError("Active System lane must be text")
    if user_text is not None and not isinstance(user_text, str):
        raise TypeError("Active User lane must be text")

    lane_uses: dict[str, list[PromptLane]] = {}
    issues: list[PromptVariableIssue] = []
    system = (
        _lex_lane(system_text, "system", lane_uses, issues)
        if system_text is not None
        else None
    )
    user = (
        _lex_lane(user_text, "user", lane_uses, issues)
        if user_text is not None
        else None
    )
    variables = tuple(
        PromptVariableSpec(name=name, lanes=tuple(lanes))
        for name, lanes in lane_uses.items()
    )
    return PromptVariablePlan(
        variables=variables,
        issues=tuple(issues),
        _system=system,
        _user=user,
    )


def _lex_lane(
    text: str,
    lane: PromptLane,
    lane_uses: dict[str, list[PromptLane]],
    issues: list[PromptVariableIssue],
) -> _LanePlan:
    tokens: list[_Token] = []
    literal: list[str] = []
    index = 0

    def flush_literal() -> None:
        if literal:
            tokens.append(_LiteralToken("".join(literal)))
            literal.clear()

    while index < len(text):
        if text.startswith("{{", index):
            literal.append("{")
            index += 2
            continue
        if text.startswith("}}", index):
            literal.append("}")
            index += 2
            continue
        if text[index] != "{":
            literal.append(text[index])
            index += 1
            continue

        variable = _variable_at(text, index)
        if variable is None:
            literal.append("{")
            index += 1
            continue

        name, end = variable
        flush_literal()
        tokens.append(_VariableToken(name))
        _register_variable(name, lane, index, lane_uses, issues)
        index = end

    flush_literal()
    return _LanePlan(tuple(tokens))


def _variable_at(text: str, opening: int) -> tuple[str, int] | None:
    name_start = opening + 1
    if name_start >= len(text) or not _is_name_start(text[name_start]):
        return None

    cursor = name_start + 1
    while cursor < len(text) and _is_name_continue(text[cursor]):
        cursor += 1
    if cursor >= len(text) or text[cursor] != "}":
        return None
    return text[name_start:cursor], cursor + 1


def _register_variable(
    name: str,
    lane: PromptLane,
    position: int,
    lane_uses: dict[str, list[PromptLane]],
    issues: list[PromptVariableIssue],
) -> None:
    if len(name) > MAX_VARIABLE_NAME_LENGTH:
        if not issues:
            issues.append(PromptVariableIssue("name_too_long", lane, position))
        return

    lanes = lane_uses.get(name)
    if lanes is not None:
        if lane not in lanes:
            lanes.append(lane)
        return
    if issues:
        return
    if len(lane_uses) == MAX_VARIABLES:
        issues.append(PromptVariableIssue("too_many_variables", lane, position))
        return
    lane_uses[name] = [lane]


def _render_lane(
    lane: _LanePlan | None,
    values: Mapping[str, str],
) -> str | None:
    if lane is None:
        return None
    return "".join(
        token.text if isinstance(token, _LiteralToken) else values[token.name]
        for token in lane.tokens
    )


def _is_name_start(character: str) -> bool:
    return character == "_" or "A" <= character <= "Z" or "a" <= character <= "z"


def _is_name_continue(character: str) -> bool:
    return _is_name_start(character) or "0" <= character <= "9"
