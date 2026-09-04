"""Bounds, validation, and result shapes for the ``ask_user`` tool.

PRD Feature A (A1, A2, A6, A9, A13). Pure: no I/O, no Textual. The payload is
model-controlled text that goes straight to a card, so every bound in A1 is
enforced here before anything reaches a widget -- the same posture as
``SessionTodoStore``. Imported lazily by ``local_tool_provider`` so this
module never rides the boot path (ADR-097).
"""

from __future__ import annotations

import re
from typing import Any

MAX_QUESTIONS = 4
MIN_OPTIONS = 2
MAX_OPTIONS = 4
MAX_QUESTION_CHARS = 500
MAX_HEADER_CHARS = 12
MAX_LABEL_CHARS = 100
MAX_DESCRIPTION_CHARS = 300

#: A9: the second consecutive ``busy`` in one run is refused outright.
MAX_CONSECUTIVE_BUSY = 2

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_NEWLINE_RE = re.compile(r"[\r\n\t]+")
_QUESTION_KEYS = frozenset({"question", "header", "multiSelect", "options"})
_OPTION_KEYS = frozenset({"label", "description"})

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


class AskUserValidationError(ValueError):
    """A rejected ``ask_user`` call; the message is the tool error the model sees."""


class AskUserBusyRefusal(ValueError):
    """A9: too many consecutive ``busy`` results in one run -- refused outright."""


def _clean_text(value: object, *, field: str, limit: int, required: bool = True) -> str:
    """Return ``value`` flattened for render, or raise with the field named.

    Args:
        value: The raw model-supplied value.
        field: Human-readable field path for the error message.
        limit: Maximum length AFTER cleaning.
        required: Whether a blank value is an error.

    Returns:
        The cleaned string: newlines/tabs collapsed to one space, other
        control characters removed, surrounding whitespace stripped.

    Raises:
        AskUserValidationError: Wrong type, invalid UTF-8, blank, or over limit.
    """
    if not isinstance(value, str):
        raise AskUserValidationError(f"{field} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise AskUserValidationError(f"{field} is not valid UTF-8") from exc
    cleaned = _CONTROL_RE.sub("", _NEWLINE_RE.sub(" ", value))
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    if required and not cleaned:
        raise AskUserValidationError(f"{field} must not be blank")
    if len(cleaned) > limit:
        raise AskUserValidationError(f"{field} exceeds {limit} characters")
    return cleaned


def validate_questions(raw: object) -> list[dict[str, Any]]:
    """Validate an ``ask_user`` call's ``questions`` and return cleaned copies.

    Args:
        raw: The call's ``questions`` value, straight from the model.

    Returns:
        1-4 question dicts, each ``{"question", "header", "multiSelect",
        "options": [{"label", "description"}]}`` with every string cleaned.

    Raises:
        AskUserValidationError: Any bound in PRD A1 violated. The message
            names the question/option index and the rule.
    """
    if not isinstance(raw, list):
        raise AskUserValidationError("questions must be a list")
    if not 1 <= len(raw) <= MAX_QUESTIONS:
        raise AskUserValidationError(f"questions must hold 1 to {MAX_QUESTIONS} items")
    questions: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise AskUserValidationError(f"question {index} must be an object")
        unknown = set(item) - _QUESTION_KEYS
        if unknown:
            raise AskUserValidationError(
                f"question {index} has unknown keys: {sorted(unknown)}"
            )
        for key in ("question", "header", "options"):
            if key not in item:
                raise AskUserValidationError(f"question {index} is missing {key}")
        multi = item.get("multiSelect", False)
        if not isinstance(multi, bool):
            raise AskUserValidationError(f"question {index}: multiSelect must be a boolean")
        options = item["options"]
        if not isinstance(options, list) or not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
            raise AskUserValidationError(
                f"question {index}: options must hold {MIN_OPTIONS} to {MAX_OPTIONS} items"
            )
        cleaned_options: list[dict[str, str]] = []
        seen: set[str] = set()
        for opt_index, option in enumerate(options, start=1):
            where = f"question {index} option {opt_index}"
            if not isinstance(option, dict):
                raise AskUserValidationError(f"{where} must be an object")
            unknown_option = set(option) - _OPTION_KEYS
            if unknown_option:
                raise AskUserValidationError(
                    f"{where} has unknown keys: {sorted(unknown_option)}"
                )
            label = _clean_text(
                option.get("label"), field=f"{where} label", limit=MAX_LABEL_CHARS
            )
            if label.casefold() in seen:
                raise AskUserValidationError(
                    f"question {index} repeats option label {label!r}"
                )
            seen.add(label.casefold())
            description = _clean_text(
                option.get("description", ""),
                field=f"{where} description",
                limit=MAX_DESCRIPTION_CHARS,
                required=False,
            )
            cleaned_options.append({"label": label, "description": description})
        questions.append(
            {
                "question": _clean_text(
                    item["question"],
                    field=f"question {index} text",
                    limit=MAX_QUESTION_CHARS,
                ),
                "header": _clean_text(
                    item["header"],
                    field=f"question {index} header",
                    limit=MAX_HEADER_CHARS,
                ),
                "multiSelect": multi,
                "options": cleaned_options,
            }
        )
    return questions


def busy_result() -> dict[str, Any]:
    """A9: the immediate result when a question is already live in the session."""
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
