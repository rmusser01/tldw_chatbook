"""Encrypted local Personal Context persistence."""

from importlib import import_module
from typing import Any

from .crypto import EncryptedEnvelope, EnvelopeCipher
from .context_service import (
    ProfileContextRequest,
    ProfileContextService,
    ProfileContextSnapshot,
)
from .key_protector import (
    InMemoryProfileKeyProtector,
    InterviewPassphraseKeyProtector,
    KeyringProfileKeyProtector,
    PassphraseProfileKeyProtector,
    ProfileKeyMaterial,
    ProfileKeyProtector,
    ProfileLockedError,
)
from .repository import PersonalContextRepository
from .runtime_policy import AgentAuthority, PersonalContextAuthorityError
from .service import (
    AuthorizedProfileContextView,
    PersonalContextSettingsSnapshot,
    PersonalContextService,
    ProfileConflictError,
    ProfileKeyCollisionError,
    ProfileOperationalStatus,
    RecordMutation,
    SettingsScopeSnapshot,
)

__all__ = [
    "EncryptedEnvelope",
    "EnvelopeCipher",
    "InMemoryProfileKeyProtector",
    "InterviewPassphraseKeyProtector",
    "InterviewCommitOutcomeUnknownError",
    "InterviewCommitReceipt",
    "InterviewDiff",
    "InterviewDiffChange",
    "InterviewDraftConflictError",
    "InterviewDraftExpiredError",
    "InterviewDraftRepository",
    "InterviewProviderError",
    "InterviewProviderRequest",
    "InterviewQuestionProvider",
    "InterviewSession",
    "KeyringProfileKeyProtector",
    "PassphraseProfileKeyProtector",
    "AgentAuthority",
    "AuthorizedProfileContextView",
    "PersonalContextAuthorityError",
    "PersonalContextRepository",
    "PersonalContextSettingsSnapshot",
    "PersonalContextService",
    "ProfileContextRequest",
    "ProfileContextService",
    "ProfileContextSnapshot",
    "ProfileConflictError",
    "ProfileKeyCollisionError",
    "ProfileKeyMaterial",
    "ProfileKeyProtector",
    "ProfileLockedError",
    "ProfileOperationalStatus",
    "ProfileInterviewCoordinator",
    "RecordMutation",
    "SettingsScopeSnapshot",
    "StoredInterviewDraft",
    "ConfiguredModelQuestionProvider",
    "FixedQuestionProvider",
    "build_interview_diff",
]


# Ordinary sends use the profile store without starting an interview.
_INTERVIEW_EXPORTS = {
    "InterviewCommitOutcomeUnknownError": "interview_coordinator",
    "InterviewCommitReceipt": "interview_coordinator",
    "InterviewSession": "interview_coordinator",
    "ProfileInterviewCoordinator": "interview_coordinator",
    "InterviewDiff": "interview_diff",
    "InterviewDiffChange": "interview_diff",
    "build_interview_diff": "interview_diff",
    "InterviewDraftConflictError": "interview_draft_repository",
    "InterviewDraftExpiredError": "interview_draft_repository",
    "InterviewDraftRepository": "interview_draft_repository",
    "StoredInterviewDraft": "interview_draft_repository",
    "ConfiguredModelQuestionProvider": "interview_provider",
    "FixedQuestionProvider": "interview_provider",
    "InterviewProviderError": "interview_provider",
    "InterviewProviderRequest": "interview_provider",
    "InterviewQuestionProvider": "interview_provider",
}
_INTERVIEW_MODULES = (
    "interview_coordinator",
    "interview_diff",
    "interview_draft_repository",
    "interview_provider",
)


def __getattr__(name: str) -> Any:
    module_name = _INTERVIEW_EXPORTS.get(name)
    if module_name is not None:
        value = getattr(import_module("." + module_name, __name__), name)
    elif name in _INTERVIEW_MODULES:
        value = import_module("." + name, __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _INTERVIEW_EXPORTS.keys() | set(_INTERVIEW_MODULES))
