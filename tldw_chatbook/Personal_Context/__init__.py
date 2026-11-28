"""Encrypted local Personal Context persistence with lazy public exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .crypto import EncryptedEnvelope, EnvelopeCipher
    from .context_service import (
        ProfileContextRequest,
        ProfileContextService,
        ProfileContextSnapshot,
    )
    from .interview_coordinator import (
        InterviewCommitOutcomeUnknownError,
        InterviewCommitReceipt,
        InterviewSession,
        ProfileInterviewCoordinator,
    )
    from .interview_diff import InterviewDiff, InterviewDiffChange, build_interview_diff
    from .interview_draft_repository import (
        InterviewDraftConflictError,
        InterviewDraftExpiredError,
        InterviewDraftRepository,
        StoredInterviewDraft,
    )
    from .interview_provider import (
        ConfiguredModelQuestionProvider,
        FixedQuestionProvider,
        InterviewProviderError,
        InterviewProviderRequest,
        InterviewQuestionProvider,
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

_EXPORTS = {
    "EncryptedEnvelope": "crypto",
    "EnvelopeCipher": "crypto",
    "InMemoryProfileKeyProtector": "key_protector",
    "InterviewPassphraseKeyProtector": "key_protector",
    "InterviewCommitOutcomeUnknownError": "interview_coordinator",
    "InterviewCommitReceipt": "interview_coordinator",
    "InterviewDiff": "interview_diff",
    "InterviewDiffChange": "interview_diff",
    "InterviewDraftConflictError": "interview_draft_repository",
    "InterviewDraftExpiredError": "interview_draft_repository",
    "InterviewDraftRepository": "interview_draft_repository",
    "InterviewProviderError": "interview_provider",
    "InterviewProviderRequest": "interview_provider",
    "InterviewQuestionProvider": "interview_provider",
    "InterviewSession": "interview_coordinator",
    "KeyringProfileKeyProtector": "key_protector",
    "PassphraseProfileKeyProtector": "key_protector",
    "AgentAuthority": "runtime_policy",
    "AuthorizedProfileContextView": "service",
    "PersonalContextAuthorityError": "runtime_policy",
    "PersonalContextRepository": "repository",
    "PersonalContextSettingsSnapshot": "service",
    "PersonalContextService": "service",
    "ProfileContextRequest": "context_service",
    "ProfileContextService": "context_service",
    "ProfileContextSnapshot": "context_service",
    "ProfileConflictError": "service",
    "ProfileKeyCollisionError": "service",
    "ProfileKeyMaterial": "key_protector",
    "ProfileKeyProtector": "key_protector",
    "ProfileLockedError": "key_protector",
    "ProfileOperationalStatus": "service",
    "ProfileInterviewCoordinator": "interview_coordinator",
    "RecordMutation": "service",
    "SettingsScopeSnapshot": "service",
    "StoredInterviewDraft": "interview_draft_repository",
    "ConfiguredModelQuestionProvider": "interview_provider",
    "FixedQuestionProvider": "interview_provider",
    "build_interview_diff": "interview_diff",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
