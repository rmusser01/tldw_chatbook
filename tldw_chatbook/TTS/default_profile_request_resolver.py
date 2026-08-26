"""Fail-closed app-wide default-voice-profile resolution for Console speech.

This is the sibling `character_request_resolver.py` names in its own module
docstring: it resolves the single **stored** id at `[app_tts]
default_profile_id` (Task 2's persisted setting) into an exact,
synthesizable `CharacterTTSRequestResolution`, or a bounded, UI-safe
`CharacterTTSResolutionError` naming the *default voice* specifically --
never the "assigned"/"character" wording those existing codes use, since
that copy would misdescribe a failure that has nothing to do with any
character.

A sibling module, not a widened `CharacterTTSRequestResolver.resolve()`,
because the two resolutions differ in every input that matters: the
identity being resolved (a stored profile UUID here, a `CharacterRef`
there), the service capability used (`get_profile` here,
`get_assigned_profile` there), and the "nothing configured" case (simply
never called here -- the caller only reaches this module once it already
knows no character voice applies AND a default profile id is configured).
Cramming both into one class would have made `CharacterTTSRequestResolver`
carry app-default failures under a name that says "character" -- exactly
what this module exists to avoid. The bounded-code *mechanism* is still
fully reused, not reinvented: this module raises the same
`CharacterTTSResolutionError` type, using two more members
(`"default_profile_missing"`, `"default_profile_store_unavailable"`) of the
SAME code table `character_request_resolver.py` already owns, so the
existing refuse+override CONTROL FLOW (`tts_events.py`'s catch site,
`app.py::_offer_tts_global_override`'s prompt-then-post-decision shape)
needed no *structural* change. Its confirmation-dialog **copy** did need a
change -- `CharacterTTSResolutionError.domain` (also computed from this
same code table) is threaded through `_issue_global_override` /
`_PendingGlobalOverride` so the dialog names the actual domain that
failed, matching the toast (`TTSCompleteEvent.error`) it already got right.

**Failure is honest, by name**, following `Subscriptions/briefing_voices.
py::resolve_roster_voices` -- this module's own precedent for "one stored
profile UUID, one `profile_service.get_profile` call, no partial
resolution": a profile id that no longer resolves (deleted) is
`"default_profile_missing"`; anything else the store itself cannot
guarantee (unbound service, repository failure, malformed stored id) is
`"default_profile_store_unavailable"` for anything short of a confirmed
"gone", and `"default_profile_missing"` for a stored value that is not
even a well-formed UUID -- Task 2's loader deliberately still loads a
malformed value as a defined **dangling state** rather than discarding it,
and this is where that dangling state is finally interpreted: as
unusable, exactly like a deleted profile, never silently dropped into the
global voice.
"""

from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import UUID

from loguru import logger

from tldw_chatbook.TTS.adapter_types import TTSRequest
from tldw_chatbook.TTS.character_request_resolver import (
    CharacterTTSRequestResolution,
    CharacterTTSResolutionCode,
    CharacterTTSResolutionError,
)
from tldw_chatbook.TTS.profile_errors import (
    ProfileRepositoryError,
    ProfileServiceError,
    ProfileValidationError,
)
from tldw_chatbook.TTS.profile_service import LoadedTTSProfile
from tldw_chatbook.TTS.profile_reference_types import TTSCloneReference


class _DefaultProfileService(Protocol):
    async def get_profile(self, profile_id: UUID) -> LoadedTTSProfile: ...

    async def get_reference(
        self,
        profile_id: UUID,
        *,
        expected_revision: int,
        expected_generation: int,
    ) -> TTSCloneReference: ...


def _log_default_profile_resolution_failure(
    code: CharacterTTSResolutionCode,
    error: Exception,
) -> None:
    detail_code = (
        error.code
        if isinstance(
            error,
            (ProfileRepositoryError, ProfileServiceError, ProfileValidationError),
        )
        else "not_available"
    )
    logger.warning(
        "Default-profile TTS request resolution failed "
        "(operation=default_profile_tts_resolution, outcome_code={}, "
        "exception_category={}, detail_code={})",
        code,
        type(error).__name__,
        detail_code,
    )


async def resolve_default_profile(
    *,
    text: str,
    default_profile_id: str,
    profile_service: _DefaultProfileService | None,
) -> CharacterTTSRequestResolution:
    """Resolve the app-wide default voice profile into one exact request.

    Called only once a caller has already established that no
    higher-priority voice (a per-character assignment) applies to this
    message and that a non-blank `default_profile_id` is configured --
    this function never decides whether the default profile SHOULD be
    consulted, only whether the configured one CAN be used.

    Args:
        text: Trusted, already-validated message text to synthesize.
        default_profile_id: The raw, non-blank stored value of
            `[app_tts] default_profile_id` -- may be a well-formed UUID
            string or Task 2's defined malformed dangling state; both are
            resolved (or refused) here, never upstream.
        profile_service: The app's TTS profile service -- duck-typed to
            one async `get_profile(profile_id: UUID) -> LoadedTTSProfile`
            method (`TTSProfileService.get_profile`) -- or `None` if no
            profile store is bound to this app instance.

    Returns:
        One immutable `source="default_profile"` resolution carrying the
        exact provider/model/voice request to synthesize.

    Raises:
        CharacterTTSResolutionError: With code `"default_profile_missing"`
            when the stored id is not a well-formed UUID or no longer
            resolves to a stored profile, or
            `"default_profile_store_unavailable"` when the profile store
            is unbound or its read fails for any other reason. Both codes
            allow the global-voice override.
        asyncio.CancelledError: If the profile-store read is cancelled.
    """

    if profile_service is None:
        raise CharacterTTSResolutionError("default_profile_store_unavailable")

    try:
        profile_uuid = UUID(default_profile_id)
    except (ValueError, AttributeError, TypeError):
        raise CharacterTTSResolutionError("default_profile_missing") from None

    load_failure: CharacterTTSResolutionCode | None = None
    try:
        loaded = await profile_service.get_profile(profile_uuid)
    except asyncio.CancelledError:
        raise
    except ProfileRepositoryError as error:
        code: CharacterTTSResolutionCode = (
            "default_profile_missing"
            if error.code == "missing"
            else "default_profile_store_unavailable"
        )
        _log_default_profile_resolution_failure(code, error)
        load_failure = code
    except (ProfileServiceError, ProfileValidationError) as error:
        _log_default_profile_resolution_failure(
            "default_profile_store_unavailable", error
        )
        load_failure = "default_profile_store_unavailable"
    except Exception as error:
        _log_default_profile_resolution_failure(
            "default_profile_store_unavailable", error
        )
        load_failure = "default_profile_store_unavailable"
    if load_failure is not None:
        raise CharacterTTSResolutionError(load_failure) from None

    try:
        if type(loaded) is not LoadedTTSProfile:
            raise TypeError
        profile = loaded.profile
        reference: TTSCloneReference | None = None
        if profile.reference is not None:
            reference_failure: CharacterTTSResolutionCode | None = None
            try:
                reference = await profile_service.get_reference(
                    profile.profile_id,
                    expected_revision=profile.revision,
                    expected_generation=loaded.repository_generation,
                )
            except asyncio.CancelledError:
                raise
            except (
                ProfileRepositoryError,
                ProfileServiceError,
                ProfileValidationError,
            ) as error:
                _log_default_profile_resolution_failure(
                    "default_profile_store_unavailable",
                    error,
                )
                reference_failure = "default_profile_store_unavailable"
            except Exception as error:
                _log_default_profile_resolution_failure(
                    "default_profile_store_unavailable",
                    error,
                )
                reference_failure = "default_profile_store_unavailable"
            if reference_failure is not None:
                raise CharacterTTSResolutionError(reference_failure) from None
            if type(reference) is not TTSCloneReference:
                raise ValueError
            assert reference is not None
            if reference.summary != profile.reference:
                raise ValueError
        request = TTSRequest(
            provider_id=profile.provider_id,
            model_id=profile.model_id,
            text=text,
            voice=profile.voice_id,
            response_format=profile.response_format,
            speed=profile.speed,
            options=profile.options,
        )
        return CharacterTTSRequestResolution(
            source="default_profile",
            request=request,
            repository_generation=loaded.repository_generation,
            profile_id=profile.profile_id,
            profile_revision=profile.revision,
            reference=reference,
        )
    except CharacterTTSResolutionError:
        raise
    except Exception as error:
        _log_default_profile_resolution_failure(
            "default_profile_store_unavailable", error
        )
        raise CharacterTTSResolutionError("default_profile_store_unavailable") from None


__all__ = ["resolve_default_profile"]
