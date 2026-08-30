"""Encrypted local Personal Context persistence."""

from .crypto import EncryptedEnvelope, EnvelopeCipher
from .context_service import (
    ProfileContextRequest,
    ProfileContextService,
    ProfileContextSnapshot,
)
from .key_protector import (
    InMemoryProfileKeyProtector,
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
    "RecordMutation",
    "SettingsScopeSnapshot",
]
