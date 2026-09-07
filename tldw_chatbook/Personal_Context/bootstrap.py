"""Construct the process-local Personal Context service once at composition."""

from __future__ import annotations

import os
import sqlite3

from .key_protector import ProfileKeyProtector, ProfileLockedError
from .paths import get_personal_context_db_path
from .repository import (
    PersonalContextRepository,
    ProfileIntegrityError,
    RepositorySchemaError,
    profile_presence_hint,
)
from .service import PersonalContextService


def _profile_presence_hint(db_path: str | os.PathLike[str]) -> bool:
    """Inspect only unencrypted profile metadata without creating a database."""

    return profile_presence_hint(db_path)


def bootstrap_personal_context_service(
    *,
    db_path: str | os.PathLike[str] | None = None,
    key_protector: ProfileKeyProtector | None = None,
    recovery_integrity_key: bytes | None = None,
    expected_recovery_profile_id: str | None = None,
) -> PersonalContextService:
    """Return an available service or a locked fail-closed facade."""

    destination = db_path or get_personal_context_db_path()
    try:
        repository = PersonalContextRepository(
            destination,
            key_protector=key_protector,
            recovery_integrity_key=recovery_integrity_key,
            expected_recovery_profile_id=expected_recovery_profile_id,
        )
    except ProfileLockedError as exc:
        return PersonalContextService.locked(
            getattr(exc, "reason_code", "profile_locked"),
            profile_present=_profile_presence_hint(destination),
        )
    except RepositorySchemaError:
        return PersonalContextService.locked(
            "repository_schema_invalid",
            profile_present=_profile_presence_hint(destination),
        )
    except ProfileIntegrityError:
        return PersonalContextService.locked(
            "profile_integrity_invalid",
            profile_present=_profile_presence_hint(destination),
        )
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return PersonalContextService.locked(
            "repository_unavailable",
            profile_present=_profile_presence_hint(destination),
        )
    return PersonalContextService(repository)
