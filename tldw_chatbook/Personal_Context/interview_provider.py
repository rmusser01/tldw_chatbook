"""Question-provider boundaries for bounded Personal Context interviews."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from pydantic import ValidationError
from tldw_profile_core import (
    InterviewPack,
    InterviewQuestion,
    InterviewTurn,
    ProfileRecord,
)


class InterviewProviderError(RuntimeError):
    """Content-safe provider failure retained by the encrypted draft."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class InterviewProviderRequest:
    """One immutable, privacy-filtered question request."""

    pack: InterviewPack
    scope_id: str
    question_attempt: int
    turns: tuple[InterviewTurn, ...] = field(default=(), repr=False)
    existing_records: tuple[ProfileRecord, ...] = field(default=(), repr=False)


class InterviewQuestionProvider(Protocol):
    """Return exactly one Shared Core-valid question."""

    provider_id: str | None
    provider_label: str
    model_id: str | None

    def next_question(self, request: InterviewProviderRequest) -> InterviewQuestion: ...


class FixedQuestionProvider:
    """Versioned local questionnaire that performs no provider call."""

    provider_id = None
    provider_label = "Fixed local questionnaire"
    model_id = None

    def __init__(self, pack: InterviewPack) -> None:
        self.pack = pack

    def next_question(self, request: InterviewProviderRequest) -> InterviewQuestion:
        if request.pack.pack_id != self.pack.pack_id:
            raise InterviewProviderError("question_pack_mismatch")
        index = request.question_attempt - 1
        try:
            return self.pack.questions[index]
        except IndexError:
            raise InterviewProviderError("question_pack_complete") from None


class ConfiguredModelQuestionProvider:
    """Pinned configured-model adapter with tools and streaming disabled."""

    def __init__(
        self,
        *,
        provider_id: str,
        provider_label: str,
        model_id: str,
        call: Callable[..., Any],
    ) -> None:
        if (
            not provider_id.strip()
            or not provider_label.strip()
            or not model_id.strip()
        ):
            raise ValueError("provider identity, label, and model must be disclosed")
        self.provider_id = provider_id
        self.provider_label = provider_label
        self.model_id = model_id
        self._call = call

    def next_question(self, request: InterviewProviderRequest) -> InterviewQuestion:
        """Call the pinned model once and validate one structured question."""

        messages = [
            {
                "role": "system",
                "content": (
                    "Ask exactly one Personal Context interview question. "
                    "Return only the requested structured object. Do not request "
                    "credentials or secrets. The required JSON Schema is: "
                    + json.dumps(
                        InterviewQuestion.model_json_schema(),
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "audience": request.pack.audience.value,
                        "coverage_topics": request.pack.coverage_topics,
                        "question_attempt": request.question_attempt,
                        "prior_turns": [
                            turn.model_dump(mode="json") for turn in request.turns
                        ],
                        "eligible_existing_records": [
                            record.model_dump(mode="json")
                            for record in request.existing_records
                        ],
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]
        try:
            raw = self._call(
                api_endpoint=self.provider_id,
                model=self.model_id,
                messages_payload=messages,
                tools=None,
                streaming=False,
                response_format={"type": "json_object"},
            )
            if isinstance(raw, str):
                raw = json.loads(raw)
            elif isinstance(raw, dict) and "content" in raw:
                content = raw["content"]
                raw = json.loads(content) if isinstance(content, str) else content
            elif isinstance(raw, dict) and "choices" in raw:
                choices = raw["choices"]
                if not isinstance(choices, list) or not choices:
                    raise ValueError("empty provider choices")
                choice = choices[0]
                if not isinstance(choice, dict) or not isinstance(
                    choice.get("message"), dict
                ):
                    raise ValueError("invalid provider choice")
                content = choice["message"].get("content")
                raw = json.loads(content) if isinstance(content, str) else content
            return InterviewQuestion.model_validate(raw)
        except InterviewProviderError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            raise InterviewProviderError("invalid_question") from exc
        except Exception as exc:
            raise InterviewProviderError("provider_unavailable") from exc
