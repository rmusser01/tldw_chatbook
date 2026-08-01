"""Resolve a briefing script's roster to concrete synthesizable voices.

A cast script (`briefing_cast.generate_script`) stores a **roster
snapshot** on its `briefing_scripts` row -- one dict per speaker, carrying
(among other fields) `voice_profile_id`, a stringified TTS profile UUID
recorded by the preset/script editor but otherwise inert until now (spec
#2 phase 2b). This module is what gives that field meaning: turning a
speaker's frozen snapshot into a concrete, synthesizable `VoiceSelection`
that Task 5 uses to build one synthesis request per turn, and Task 6
stores (via `dump_voice_snapshot`) on the `briefing_audio` row.

**The identity landscape has two different keys.** A roster entry's
`voice_profile_id` is a **stringified TTS profile UUID** (the preset
modal's voice picker builds its options as
`(profile.display_name, str(profile.profile_id))` --
`UI/Screens/watchlists_collections_screen.py:3766` -- and writes that
string back onto the roster). A roster entry's `character_card_id` is an
entirely different key: a `ChaChaNotes` row id (an `int`), not a
`CharacterRef` (`TTS/profile_types.py:382`, a three-part authority-scoped
identity used by the *character-assignment* feature). Character-bound
voice assignment is OUT OF SCOPE here -- a speaker with a card but no
`voice_profile_id` fails the same "no profile assigned" way as any other
speaker with none. This module never reads `character_card_id` at all.
`resolve_roster_voices` converts the roster's `str` id to a `UUID`
**exactly once**, at the service boundary (immediately before the one
`profile_service.get_profile` call); a malformed string is a resolution
failure naming the speaker, never a crash.

**`response_format` is always forced to `"wav"`**, regardless of what the
resolved profile itself says (a decision locked before this task, not
relitigated here): `audio_cpp` accepts nothing else, pydub decodes wav
without codec surprises, and Task 3's stitcher (`TTS/audio_stitch.py`) is
WAV-only. This is the constraint the whole audio pipeline rests on.

**Failure is honest, by name** (spec's "Error handling ethos", the same
rule `briefing_cast` follows): every `VoiceResolutionError` names the
speaker; a `voice_profile_id` that no longer resolves (a deleted profile)
names **both** the speaker and the id, so a user can tell which profile
vanished; an unbound voice service says plainly that no voice service is
available. Task 6 surfaces these messages directly to users.

Testing: the only faked seam is the profile service (one async
`get_profile(profile_id: UUID) -> LoadedTTSProfile` method) -- mirroring
`briefing_cast`'s own rule of faking exactly one collaborator.

Nothing here logs a voice id, a speaker name, or turn text at any level --
only exception types, for the same reason `briefing_cast` avoids it: this
app's log sink runs with `diagnose=True`, which dumps a failing frame's
locals, and those locals would hold exactly the content this module must
not leak.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

#: `VoiceSelection.response_format` is always forced to this value. See the
#: module docstring: this is a locked decision, not a per-profile choice.
_FORCED_RESPONSE_FORMAT = "wav"

#: The provider id identifying the app's own exact/native synthesis path
#: (`TTSService.synthesize_exact`). Every other provider id is a legacy
#: provider, synthesized through the raw streaming path (Task 5).
_EXACT_PROVIDER_ID = "audio_cpp"


class VoiceResolutionError(RuntimeError):
    """Raised when a roster speaker cannot be resolved to a concrete voice.

    Every raise site names the specific speaker -- and, for a stored
    profile id that no longer resolves, the id too -- so a caller (Task 6)
    can surface exactly which speaker or profile is at fault rather than a
    generic "voice resolution failed".
    """


@dataclass(frozen=True)
class VoiceSelection:
    """One roster speaker's resolved, synthesizable voice.

    Everything Task 5 needs to build one synthesis request for one turn by
    this speaker. A plain, already-resolved value -- once built, it never
    reaches back into the profile store.

    Attributes:
        speaker: The roster entry's speaker name (the same name
            `briefing_cast.parse_script_turns` uses on each turn).
        provider_id: The resolved profile's TTS provider id.
        model_id: The resolved profile's model id.
        voice_id: The resolved profile's voice id, if any.
        response_format: Always `"wav"` -- see the module docstring.
        speed: The resolved profile's speed.
        options: The resolved profile's provider-specific options.
        profile_id: The resolved profile's id, as a string (the same
            stringified form the roster itself stores), or `None` if this
            selection was not built from a stored profile.
        profile_revision: The resolved profile's revision at resolution
            time -- snapshotted so a later profile edit cannot silently
            reinterpret an already-synthesized artifact.
    """

    speaker: str
    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, Any]
    profile_id: str | None
    profile_revision: int | None

    def is_exact_provider(self) -> bool:
        """Return whether this selection synthesizes via the exact path.

        Returns:
            `True` when `provider_id` is `"audio_cpp"`, the app's own
            native provider, routed through `TTSService.synthesize_exact`.
            `False` for every legacy provider, which Task 5 routes through
            the raw streaming path instead.
        """

        return self.provider_id == _EXACT_PROVIDER_ID


def _json_safe(value: Any) -> Any:
    """Recursively convert a profile's frozen options into plain JSON types.

    `TTSGenerationProfile.options` freezes nested dicts/lists into
    `Mapping`/`tuple` (`TTS/profile_types.py`'s `FrozenJsonOptions`), which
    `json.dumps` cannot serialize directly (a `Mapping` that is not itself
    a `dict`, and a `tuple`, are both rejected). This mirrors that module's
    own private `_json_ready` helper (not imported -- it is private to its
    own module).
    """

    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


async def resolve_roster_voices(
    roster_snapshot: Sequence[Mapping[str, Any]],
    *,
    profile_service: Any | None,
) -> list[VoiceSelection]:
    """Resolve every roster speaker's stored `voice_profile_id` to a voice.

    Maps speakers in roster order and raises on the first unresolvable one
    -- a script with one bad voice assignment fails cleanly rather than
    partially resolving.

    Args:
        roster_snapshot: A cast script's stored roster snapshot -- each
            entry at least `{"name": str, "voice_profile_id": str | None}`,
            in roster order (e.g. `briefing_cast.load_roster`'s decoded
            `roster_snapshot_json`).
        profile_service: The app's TTS profile service -- duck-typed to
            one async `get_profile(profile_id: UUID) -> LoadedTTSProfile`
            method (`TTSProfileService.get_profile`) -- or `None` if no
            voice service is bound to this app instance.

    Returns:
        One `VoiceSelection` per roster entry, in roster order.

    Raises:
        VoiceResolutionError: Naming the speaker, for a speaker with no
            `voice_profile_id`, a malformed id, an unbound voice service,
            or any other resolution failure; naming both the speaker and
            the id specifically when a `voice_profile_id` no longer
            resolves to a stored profile (e.g. it was deleted).
    """

    selections: list[VoiceSelection] = []
    for entry in roster_snapshot:
        name = str(entry.get("name") or "").strip()
        raw_profile_id = entry.get("voice_profile_id")

        if not raw_profile_id:
            raise VoiceResolutionError(f"speaker {name!r} has no voice profile assigned")
        if profile_service is None:
            raise VoiceResolutionError(
                f"speaker {name!r} has a voice profile assigned, but no voice "
                "service is available to resolve it"
            )

        try:
            profile_uuid = UUID(str(raw_profile_id))
        except (ValueError, AttributeError, TypeError):
            raise VoiceResolutionError(
                f"speaker {name!r} has a malformed voice profile id "
                f"{raw_profile_id!r}"
            ) from None

        try:
            loaded = await profile_service.get_profile(profile_uuid)
        except ProfileRepositoryError as exc:
            if exc.code == "missing":
                raise VoiceResolutionError(
                    f"speaker {name!r} references voice profile "
                    f"{raw_profile_id!r}, which no longer exists"
                ) from None
            raise VoiceResolutionError(
                f"speaker {name!r}'s voice profile could not be resolved: "
                f"{type(exc).__name__}"
            ) from None
        except Exception as exc:  # noqa: BLE001 - every resolution failure names the speaker
            raise VoiceResolutionError(
                f"speaker {name!r}'s voice profile could not be resolved: "
                f"{type(exc).__name__}"
            ) from None

        profile = loaded.profile
        selections.append(
            VoiceSelection(
                speaker=name,
                provider_id=profile.provider_id,
                model_id=profile.model_id,
                voice_id=profile.voice_id,
                response_format=_FORCED_RESPONSE_FORMAT,
                speed=profile.speed,
                options=profile.options,
                profile_id=str(profile.profile_id),
                profile_revision=profile.revision,
            )
        )
    return selections


def dump_voice_snapshot(selections: Sequence[VoiceSelection]) -> str:
    """Canonical JSON encoding of resolved voice selections, for storage.

    Mirrors `briefing_cast.dump_roster`'s determinism contract
    (`sort_keys=True`): two calls over equal data produce byte-identical
    text. Task 6 stores this string once, on a `briefing_audio` row.

    Args:
        selections: `resolve_roster_voices`'s output, in roster order.

    Returns:
        Deterministic JSON text: an array with one object per selection,
        including `profile_revision` -- the snapshot's whole point being
        that a profile edited after synthesis must not silently
        reinterpret an already-produced artifact.
    """

    return json.dumps(
        [
            {
                "speaker": selection.speaker,
                "provider_id": selection.provider_id,
                "model_id": selection.model_id,
                "voice_id": selection.voice_id,
                "response_format": selection.response_format,
                "speed": selection.speed,
                "options": _json_safe(selection.options),
                "profile_id": selection.profile_id,
                "profile_revision": selection.profile_revision,
            }
            for selection in selections
        ],
        sort_keys=True,
    )
