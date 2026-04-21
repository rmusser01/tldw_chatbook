"""Normalization helpers for the study quiz compat seam."""

from __future__ import annotations

from typing import Any, Mapping


def _require_mapping(record: Any) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"Expected mapping-like quiz record, got {type(record)!r}")
    return record


def _normalize_backend(backend: str) -> str:
    normalized = str(backend or "").strip().lower()
    if normalized not in {"local", "server"}:
        raise ValueError(f"Invalid quiz backend: {backend!r}")
    return normalized


def normalize_quiz_record(backend: str, record: Any) -> dict[str, Any]:
    payload = _require_mapping(record)
    normalized_backend = _normalize_backend(backend)
    backing_id = str(payload.get("id"))
    return {
        "record_id": f"{normalized_backend}:quiz:{backing_id}",
        "record_type": "quiz",
        "backend": normalized_backend,
        "backing_id": backing_id,
        "name": payload.get("name"),
        "description": payload.get("description"),
        "workspace_id": payload.get("workspace_id"),
        "total_questions": int(payload.get("total_questions") or 0),
        "time_limit_seconds": payload.get("time_limit_seconds"),
        "passing_score": payload.get("passing_score"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("last_modified") or payload.get("updated_at"),
        "deleted": bool(payload.get("deleted", False)),
        "client_id": payload.get("client_id"),
        "version": int(payload.get("version") or 1),
    }


def normalize_quiz_question_record(backend: str, record: Any) -> dict[str, Any]:
    payload = _require_mapping(record)
    normalized_backend = _normalize_backend(backend)
    backing_id = str(payload.get("id"))
    quiz_id = str(payload.get("quiz_id"))
    correct_answer = payload.get("correct_answer")
    explanation = payload.get("explanation")
    return {
        "record_id": f"{normalized_backend}:quiz_question:{backing_id}",
        "record_type": "quiz_question",
        "backend": normalized_backend,
        "backing_id": backing_id,
        "quiz_record_id": f"{normalized_backend}:quiz:{quiz_id}",
        "quiz_backing_id": quiz_id,
        "question_type": payload.get("question_type"),
        "question_text": payload.get("question_text"),
        "options": list(payload.get("options") or []) if payload.get("options") is not None else None,
        "correct_answer": correct_answer,
        "explanation": explanation,
        "answer_visible": correct_answer is not None or explanation is not None,
        "hint": payload.get("hint"),
        "hint_penalty_points": int(payload.get("hint_penalty_points") or 0),
        "source_citations": list(payload.get("source_citations") or []) if payload.get("source_citations") is not None else None,
        "points": int(payload.get("points") or 0),
        "order_index": int(payload.get("order_index") or 0),
        "tags": list(payload.get("tags") or []) if payload.get("tags") is not None else None,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("last_modified") or payload.get("updated_at"),
        "deleted": bool(payload.get("deleted", False)),
        "client_id": payload.get("client_id"),
        "version": int(payload.get("version") or 1),
    }


def normalize_quiz_attempt_record(backend: str, record: Any) -> dict[str, Any]:
    payload = _require_mapping(record)
    normalized_backend = _normalize_backend(backend)
    backing_id = str(payload.get("id"))
    quiz_id = str(payload.get("quiz_id"))
    questions = [
        normalize_quiz_question_record(normalized_backend, question)
        for question in list(payload.get("questions") or [])
    ]
    answers = []
    for answer in list(payload.get("answers") or []):
        if not isinstance(answer, Mapping):
            continue
        question_id = answer.get("question_id")
        answers.append(
            {
                "question_id": question_id,
                "question_record_id": (
                    f"{normalized_backend}:quiz_question:{question_id}" if question_id is not None else None
                ),
                "user_answer": answer.get("user_answer"),
                "is_correct": bool(answer.get("is_correct", False)),
                "correct_answer": answer.get("correct_answer"),
                "explanation": answer.get("explanation"),
                "hint_used": answer.get("hint_used"),
                "hint_penalty_points": answer.get("hint_penalty_points"),
                "source_citations": answer.get("source_citations"),
                "points_awarded": answer.get("points_awarded"),
                "time_spent_ms": answer.get("time_spent_ms"),
            }
        )

    return {
        "record_id": f"{normalized_backend}:quiz_attempt:{backing_id}",
        "record_type": "quiz_attempt",
        "backend": normalized_backend,
        "backing_id": backing_id,
        "quiz_record_id": f"{normalized_backend}:quiz:{quiz_id}",
        "quiz_backing_id": quiz_id,
        "started_at": payload.get("started_at"),
        "completed_at": payload.get("completed_at"),
        "score": payload.get("score"),
        "total_possible": int(payload.get("total_possible") or 0),
        "time_spent_seconds": payload.get("time_spent_seconds"),
        "answers": answers,
        "questions": questions,
    }
