"""Public service exports, resolved lazily for dependency-light recovery."""

from importlib import import_module

_EXPORTS = {
    "LocalQuizService": "local_quiz_service",
    "LocalStudyService": "local_study_service",
    "normalize_quiz_attempt_record": "quiz_normalizers",
    "normalize_quiz_question_record": "quiz_normalizers",
    "normalize_quiz_record": "quiz_normalizers",
    "QuizBackend": "quiz_scope_service",
    "QuizScopeService": "quiz_scope_service",
    "ServerQuizService": "server_quiz_service",
    "ServerStudyService": "server_study_service",
    "merge_review_outcome_record": "study_normalizers",
    "normalize_study_deck_record": "study_normalizers",
    "normalize_study_flashcard_record": "study_normalizers",
    "normalize_study_review_candidate": "study_normalizers",
    "StudyBackend": "study_scope_service",
    "StudyScopeService": "study_scope_service",
}
__all__ = [
    "LocalQuizService",
    "LocalStudyService",
    "merge_review_outcome_record",
    "normalize_quiz_attempt_record",
    "normalize_quiz_question_record",
    "normalize_quiz_record",
    "normalize_study_deck_record",
    "normalize_study_flashcard_record",
    "normalize_study_review_candidate",
    "QuizBackend",
    "QuizScopeService",
    "ServerQuizService",
    "ServerStudyService",
    "StudyBackend",
    "StudyScopeService",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
