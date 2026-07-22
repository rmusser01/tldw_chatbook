"""Roleplay P3b Task 5: net-new `personality_traits` on the persona profile DTO.

The persona service is DTO-driven (`create_persona_profile` validates
`PersonaProfileCreate` -> `model_dump` -> JSON store; `_persona_profile_view`
is a passthrough `dict(record)`), so adding `personality_traits` to the DTO is
sufficient for it to persist and reload - no service-method change needed.
Mirrors `test_local_character_persona_service_persists_persona_profile_crud`
in `test_local_character_persona_service.py` for the ctor/store round-trip
pattern (db=None: persona-profile CRUD never touches `self.db`).
"""

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.tldw_api.character_persona_schemas import (
    PersonaProfileCreate,
    PersonaProfileUpdate,
)


def test_personality_traits_persists_and_reloads(tmp_path):
    store_path = tmp_path / "personas.json"
    svc = LocalCharacterPersonaService(None, persona_store_path=store_path)

    created = svc.create_persona_profile(
        PersonaProfileCreate(
            id="guide", name="Guide", personality_traits="brave, kind"
        )
    )
    assert created["personality_traits"] == "brave, kind"

    # Reload from a FRESH service instance backed by the same JSON store -
    # proves the field round-trips through persistence, not just in-memory.
    svc2 = LocalCharacterPersonaService(None, persona_store_path=store_path)
    got = svc2.get_persona_profile("guide")
    assert got["personality_traits"] == "brave, kind"


def test_personality_traits_defaults_to_empty_string():
    svc = LocalCharacterPersonaService(None)
    created = svc.create_persona_profile(PersonaProfileCreate(name="Blank"))
    assert created["personality_traits"] == ""


def test_personality_traits_updates_and_reloads(tmp_path):
    store_path = tmp_path / "personas.json"
    svc = LocalCharacterPersonaService(None, persona_store_path=store_path)
    created = svc.create_persona_profile(
        PersonaProfileCreate(id="guide", name="Guide", personality_traits="calm")
    )

    updated = svc.update_persona_profile(
        "guide",
        PersonaProfileUpdate(personality_traits="fierce, loyal"),
        expected_version=created["version"],
    )
    assert updated["personality_traits"] == "fierce, loyal"

    svc2 = LocalCharacterPersonaService(None, persona_store_path=store_path)
    got = svc2.get_persona_profile("guide")
    assert got["personality_traits"] == "fierce, loyal"


def test_personality_traits_update_to_empty_string_clears_and_reloads(tmp_path):
    """An explicit ``personality_traits=""`` update must persist the empty
    string, not silently preserve the old value.

    ``update_persona_profile`` builds its patch via
    ``PersonaProfileUpdate.model_dump(exclude_none=True)`` - which excludes
    only ``None`` fields, not falsy-but-set ones like ``""`` - so a real
    clear-to-empty edit round-trips correctly. This pins that behavior.
    """
    store_path = tmp_path / "personas.json"
    svc = LocalCharacterPersonaService(None, persona_store_path=store_path)
    created = svc.create_persona_profile(
        PersonaProfileCreate(id="guide", name="Guide", personality_traits="x")
    )
    assert created["personality_traits"] == "x"

    updated = svc.update_persona_profile(
        "guide",
        PersonaProfileUpdate(personality_traits=""),
        expected_version=created["version"],
    )
    assert updated["personality_traits"] == ""

    # Reload from a FRESH service instance backed by the same JSON store -
    # proves the cleared value genuinely persisted, not just that the
    # in-memory record returned by update_persona_profile looks right.
    svc2 = LocalCharacterPersonaService(None, persona_store_path=store_path)
    got = svc2.get_persona_profile("guide")
    assert got["personality_traits"] == ""
