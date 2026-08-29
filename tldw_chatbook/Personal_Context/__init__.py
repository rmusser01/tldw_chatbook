"""Encrypted local Personal Context persistence."""

from .crypto import EncryptedEnvelope, EnvelopeCipher
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
    PersonalContextService,
    ProfileConflictError,
    ProfileKeyCollisionError,
    ProfileOperationalStatus,
    RecordMutation,
)

__all__ = [
    "EncryptedEnvelope",
    "EnvelopeCipher",
    "InMemoryProfileKeyProtector",
    "KeyringProfileKeyProtector",
    "PassphraseProfileKeyProtector",
    "AgentAuthority",
    "PersonalContextAuthorityError",
    "PersonalContextRepository",
    "PersonalContextService",
    "ProfileConflictError",
    "ProfileKeyCollisionError",
    "ProfileKeyMaterial",
    "ProfileKeyProtector",
    "ProfileLockedError",
    "ProfileOperationalStatus",
    "RecordMutation",
]
