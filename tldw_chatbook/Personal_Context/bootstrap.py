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

    from .native_compatibility import (
        ProfileCompatibilityError,
        require_native_consumer,
        validate_native_consumers,
    )

    destination = db_path or get_personal_context_db_path()
    try:
        validate_native_consumers()
        require_native_consumer("bootstrap.bootstrap_personal_context_service")
        repository = PersonalContextRepository(
            destination,
            key_protector=key_protector,
            recovery_integrity_key=recovery_integrity_key,
            expected_recovery_profile_id=expected_recovery_profile_id,
        )
        if not repository.is_destroyed():
            compatibility = repository.read_compatibility()
            if compatibility is not None and compatibility.state != "legacy_v1":
                raise ProfileCompatibilityError()
    except ProfileCompatibilityError:
        return PersonalContextService.locked(
            ProfileCompatibilityError.reason_code,
            profile_present=_profile_presence_hint(destination),
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
