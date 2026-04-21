"""Shared local/server study seam for compat-first flashcards parity."""

from .local_quiz_service import LocalQuizService
from .local_study_service import LocalStudyService
from .quiz_normalizers import (
    normalize_quiz_attempt_record,
    normalize_quiz_question_record,
    normalize_quiz_record,
)
from .quiz_scope_service import QuizBackend, QuizScopeService
from .server_quiz_service import ServerQuizService
from .server_study_service import ServerStudyService
from .study_normalizers import (
    merge_review_outcome_record,
    normalize_study_deck_record,
    normalize_study_flashcard_record,
    normalize_study_review_candidate,
)
from .study_scope_service import StudyBackend, StudyScopeService

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
