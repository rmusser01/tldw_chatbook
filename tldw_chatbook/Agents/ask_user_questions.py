"""Bounds, validation, and result shapes for the ``ask_user`` tool.

PRD Feature A (A1, A2, A6, A9, A13). Pure: no I/O, no Textual. The payload is
model-controlled text that goes straight to a card, and the answers are
user-typed text that goes straight back to the model, so both boundaries
are constrained Pydantic models (the repo's validation mechanism) and only
their validated output is used downstream. Imported lazily by
``local_tool_provider`` and the controller so this module never rides the
boot path (ADR-097).
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

MAX_QUESTIONS = 4
MIN_OPTIONS = 2
MAX_OPTIONS = 4
MAX_QUESTION_CHARS = 500
MAX_HEADER_CHARS = 12
MAX_LABEL_CHARS = 100
MAX_DESCRIPTION_CHARS = 300
#: The user's free-text "Other" answer is bounded like a question: it is
#: rendered in the transcript marker and handed back to the model.
MAX_OTHER_TEXT_CHARS = 500

#: A9: the second consecutive ``busy`` in one run is refused outright.
MAX_CONSECUTIVE_BUSY = 2

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_NEWLINE_RE = re.compile(r"[\r\n\t]+")

ASK_USER_DESCRIPTION = (
    "Ask the user up to 4 multiple-choice questions and wait for the answers. "
    "Use it ONLY for a decision that is genuinely the user's to make: a "
    "preference, a trade-off between valid designs, or something neither the "
    "code nor the conversation can tell you. Do not ask when a conventional "
    "default exists, when the answer is discoverable by reading the code or "
    "running a tool, when you can proceed and state your assumption, or to "
    "confirm a plan you already have. Batch related questions into ONE call "
    "instead of asking several times. Each question offers 2-4 options; the "
    "user can always type a free-text 'Other' answer instead. The result lists "
    "the selected labels per question; 'unanswered' marks questions the user "
    "skipped, and 'answered': false with a reason means no answer will come. "
    "If the reason is 'busy', another question is already waiting for the "
    "user: proceed without asking again this turn."
)

ASK_USER_REFUSAL_COPY = (
    "ask_user refused: it returned 'busy' twice in a row in this run. A "
    "question is already waiting for the user. Do not call ask_user again "
    "this turn; proceed without the answer."
)

_OPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "minLength": 1, "maxLength": MAX_LABEL_CHARS},
        "description": {"type": "string", "maxLength": MAX_DESCRIPTION_CHARS},
    },
    "required": ["label"],
    "additionalProperties": False,
}

#: The tool-facing schema. Hand-written rather than derived from the models
#: below so it stays flat (no ``$defs``) and never declares "Other" -- the
#: card injects that escape hatch, the model cannot suppress it (A2).
ASK_USER_PARAMETERS = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_QUESTIONS,
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_QUESTION_CHARS,
                    },
                    "header": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_HEADER_CHARS,
                    },
                    "multiSelect": {"type": "boolean"},
                    "options": {
                        "type": "array",
                        "minItems": MIN_OPTIONS,
                        "maxItems": MAX_OPTIONS,
                        "items": _OPTION_SCHEMA,
                    },
                },
                "required": ["question", "header", "options"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


#: task-31420: env override for the registry prompt `agents.ask_user_tool_description`
#: (env -> config -> catalog default, the repo's precedence rule).
ASK_USER_DESCRIPTION_ENV_VAR = "TLDW_INTERNAL_PROMPT_AGENTS_ASK_USER_TOOL_DESCRIPTION"
MAX_TOOL_DESCRIPTION_CHARS = 4000


class ToolDescriptionText(BaseModel):
    """A configurable tool description: non-blank, bounded, control-free text."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1, max_length=MAX_TOOL_DESCRIPTION_CHARS)

    @field_validator("text", mode="before")
    @classmethod
    def _clean(cls, value: object) -> str:
        return _clean_text(value, required=True)


def resolve_tool_description(*candidates: object) -> str:
    """Pick the first candidate that validates as a tool description.

    task-31420: the ask_user description is user-configurable. Each
    candidate (environment value, registry value, shipped constant) is run
    through ``ToolDescriptionText``; the first that validates wins, so an
    empty, blank, over-long, or non-text override is skipped rather than
    shipped to the model.

    Args:
        *candidates: Values in precedence order; ``None`` entries are skipped.

    Returns:
        The validated text of the first acceptable candidate.

    Raises:
        AskUserValidationError: No candidate validated -- unreachable while
            the shipped constant is the last candidate.
    """
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return ToolDescriptionText(text=candidate).text
        except ValidationError:
            continue
    raise AskUserValidationError("no usable tool description")


class AskUserValidationError(ValueError):
    """A rejected ``ask_user`` payload; the message names the field and the rule."""


class AskUserBusyRefusal(ValueError):
    """A9: too many consecutive ``busy`` results in one run -- refused outright."""


def _flatten(value: str) -> str:
    """Collapse newlines/tabs to one space, drop other controls, strip."""
    cleaned = _CONTROL_RE.sub("", _NEWLINE_RE.sub(" ", value))
    return re.sub(r" {2,}", " ", cleaned).strip()


def _clean_text(value: object, *, required: bool) -> str:
    """Pre-validate one text field: type, UTF-8, control flattening, blank.

    Length is left to the model's ``Field`` constraint so the limit lives in
    exactly one place per field.

    Args:
        value: The raw value.
        required: Whether a blank value is an error.

    Returns:
        The flattened string.

    Raises:
        ValueError: Wrong type, invalid UTF-8, or blank when required.
    """
    if not isinstance(value, str):
        # ValueError on purpose: Pydantic wraps ValueError/AssertionError from a
        # validator into a ValidationError; a TypeError would escape as a crash.
        raise ValueError("must be a string")  # noqa: TRY004
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("is not valid UTF-8") from exc
    cleaned = _flatten(value)
    if required and not cleaned:
        raise ValueError("must not be blank")
    return cleaned


class AskUserOption(BaseModel):
    """One option the model offers (A1)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    label: str = Field(max_length=MAX_LABEL_CHARS)
    description: str = Field(default="", max_length=MAX_DESCRIPTION_CHARS)

    @field_validator("label", mode="before")
    @classmethod
    def _clean_label(cls, value: object) -> str:
        return _clean_text(value, required=True)

    @field_validator("description", mode="before")
    @classmethod
    def _clean_description(cls, value: object) -> str:
        return _clean_text(value, required=False)


class AskUserQuestion(BaseModel):
    """One question the model asks (A1)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(max_length=MAX_QUESTION_CHARS)
    header: str = Field(max_length=MAX_HEADER_CHARS)
    multi_select: bool = Field(default=False, alias="multiSelect")
    options: list[AskUserOption] = Field(min_length=MIN_OPTIONS, max_length=MAX_OPTIONS)

    @field_validator("question", "header", mode="before")
    @classmethod
    def _clean_required_text(cls, value: object) -> str:
        return _clean_text(value, required=True)

    @field_validator("options")
    @classmethod
    def _labels_are_distinct(cls, options: list[AskUserOption]) -> list[AskUserOption]:
        seen: set[str] = set()
        for option in options:
            key = option.label.casefold()
            if key in seen:
                raise ValueError(f"repeats option label {option.label!r}")
            seen.add(key)
        return options


class AskUserAnswer(BaseModel):
    """One answer the card returns (A6), validated before the worker trusts it."""

    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(max_length=MAX_QUESTION_CHARS)
    selected: list[Annotated[str, Field(max_length=MAX_LABEL_CHARS)]] = Field(
        default_factory=list, max_length=MAX_OPTIONS
    )
    other_text: str | None = Field(default=None, max_length=MAX_OTHER_TEXT_CHARS)
    unanswered: bool = False


_QUESTIONS = TypeAdapter(
    Annotated[list[AskUserQuestion], Field(min_length=1, max_length=MAX_QUESTIONS)]
)
_ANSWERS = TypeAdapter(
    Annotated[list[AskUserAnswer], Field(min_length=0, max_length=MAX_QUESTIONS)]
)


def _describe(error: ValidationError) -> str:
    """Turn the first Pydantic error into one actionable sentence.

    Args:
        error: The raised ``ValidationError``.

    Returns:
        ``"question 2 option 1 label: String should have at most 100
        characters"``-style text, or the bare message for list-level errors.
    """
    first = error.errors()[0]
    parts: list[str] = []
    loc = list(first.get("loc", ()))
    index = 0
    while index < len(loc):
        part = loc[index]
        if isinstance(part, int):
            parts.append(f"question {part + 1}")
        elif part == "options" and index + 1 < len(loc) and isinstance(loc[index + 1], int):
            parts.append(f"option {loc[index + 1] + 1}")
            index += 1
        else:
            parts.append(str(part))
        index += 1
    message = str(first.get("msg", "invalid"))
    message = message.removeprefix("Value error, ")
    return f"{' '.join(parts)}: {message}" if parts else message


def validate_questions(raw: object) -> list[dict[str, Any]]:
    """Validate an ``ask_user`` call's ``questions`` and return cleaned copies.

    Args:
        raw: The call's ``questions`` value, straight from the model.

    Returns:
        1-4 question dicts, each ``{"question", "header", "multiSelect",
        "options": [{"label", "description"}]}`` with every string cleaned --
        the validated model output, nothing from ``raw`` passes through.

    Raises:
        AskUserValidationError: Any bound in PRD A1 violated. The message
            names the question/option and the rule.
    """
    try:
        questions = _QUESTIONS.validate_python(raw)
    except ValidationError as error:
        raise AskUserValidationError(_describe(error)) from None
    return [question.model_dump(by_alias=True) for question in questions]


def validate_answers(raw: object) -> list[dict[str, Any]]:
    """Validate the answers a card returns before the worker trusts them.

    Args:
        raw: The ``QuestionAnswered`` payload's ``answers`` list.

    Returns:
        The validated answer dicts (``question``, ``selected``,
        ``other_text``, ``unanswered``).

    Raises:
        AskUserValidationError: Shape or bound violated -- a resolve carrying
            such a list is dropped, never partially applied.
    """
    try:
        answers = _ANSWERS.validate_python(raw)
    except ValidationError as error:
        raise AskUserValidationError(_describe(error)) from None
    return [answer.model_dump() for answer in answers]


def clean_other_text(value: object) -> str | None:
    """Bound the user's free-text "Other" answer at the card boundary.

    Newlines and control characters are flattened exactly as the model's
    text is, the result is truncated to ``MAX_OTHER_TEXT_CHARS`` (a user's
    own typing is never rejected, only bounded), and the shared
    ``validate_text_input`` length check is the final gate.

    Args:
        value: The Input's raw ``value``.

    Returns:
        The cleaned text, or None when nothing usable was typed.
    """
    if not isinstance(value, str):
        return None
    cleaned = _flatten(value)[:MAX_OTHER_TEXT_CHARS]
    if not cleaned:
        return None
    from tldw_chatbook.Utils.input_validation import validate_text_input

    if not validate_text_input(cleaned, max_length=MAX_OTHER_TEXT_CHARS, allow_html=True):
        return None
    return cleaned


def busy_result() -> dict[str, Any]:
    """A9: the immediate result when a question is already live in the session.

    Returns:
        ``{"answered": False, "reason": "busy", "instruction": ...}`` -- the
        instruction tells the model not to retry this turn.
    """
    return {
        "answered": False,
        "reason": "busy",
        "instruction": (
            "A question is already waiting for the user in this session. Do not "
            "retry ask_user now: proceed without the answer, or ask again in a "
            "later turn."
        ),
    }


def unanswered_result(reason: str) -> dict[str, Any]:
    """A6: the result for a round that ended without answers.

    Args:
        reason: ``"timeout"`` or ``"cancelled"``.

    Returns:
        ``{"answered": False, "reason": reason}``.
    """
    return {"answered": False, "reason": reason}


def answered_result(answers: list[dict[str, Any]]) -> dict[str, Any]:
    """A6: wrap per-question answers into the tool result.

    Args:
        answers: One answer dict per question, in question order.

    Returns:
        ``{"answered": True, "answers": [...]}`` with defensive copies.
    """
    return {"answered": True, "answers": [dict(answer) for answer in answers]}


def empty_answers(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One ``unanswered`` entry per question -- the shape a blank submit yields.

    Args:
        questions: The validated questions.

    Returns:
        Answer dicts with nothing selected and ``unanswered`` set.
    """
    return [
        {
            "question": str(question.get("question", "")),
            "selected": [],
            "other_text": None,
            "unanswered": True,
        }
        for question in questions
    ]
