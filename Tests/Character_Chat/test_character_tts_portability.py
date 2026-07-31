"""Character-card integration tests for sanitized TTS profile attachments."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from loguru import logger as loguru_logger
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import tldw_chatbook.Character_Chat.Character_Chat_Lib as character_lib
import tldw_chatbook.UI.CCP_Modules.ccp_character_handler as character_handler
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.TTS.profile_portability import (
    CHARACTER_CARD_TTS_EXTENSION_KEY,
    PortableTTSProfile,
)
from tldw_chatbook.TTS.profile_types import TTSProfileDraft


@pytest.fixture
def db_instance(tmp_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB(tmp_path / "characters.sqlite", "portability-test")
    yield db
    db.close_connection()


def _card(name: str = "Portable Character", *, attachment: object) -> dict[str, Any]:
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": name,
            "description": "A local character",
            "extensions": {
                "unrelated/example": {"kept": True},
                CHARACTER_CARD_TTS_EXTENSION_KEY: attachment,
            },
        },
    }


def _valid_attachment() -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_id": "00000000-0000-4000-8000-000000000000",
        "name": "Character voice",
        "provider_id": "audio_cpp",
        "model_id": "supertonic-3",
        "voice_id": "M1",
        "response_format": "wav",
        "speed": 1.0,
        "options": {},
    }


def _portable_profile() -> PortableTTSProfile:
    attachment = _valid_attachment()
    return PortableTTSProfile(
        profile_id=UUID(cast(str, attachment["profile_id"])),
        draft=TTSProfileDraft(
            display_name=cast(str, attachment["name"]),
            provider_id=cast(str, attachment["provider_id"]),
            model_id=cast(str, attachment["model_id"]),
            voice_id=cast(str, attachment["voice_id"]),
            response_format=cast(str, attachment["response_format"]),
            speed=cast(float, attachment["speed"]),
            options={},
        ),
    )


def _png_card_bytes(card: dict[str, Any], *, encoded: str | None = None) -> bytes:
    """Return a minimal PNG with one Tavern-compatible card text chunk."""

    metadata = PngInfo()
    card_text = json.dumps(card) if encoded is None else encoded
    if encoded is None:
        card_text = base64.b64encode(card_text.encode("utf-8")).decode("ascii")
    metadata.add_text("chara", card_text)
    output = io.BytesIO()
    Image.new("RGB", (1, 1), color="gray").save(
        output,
        format="PNG",
        pnginfo=metadata,
    )
    return output.getvalue()


def _add_character(
    db: CharactersRAGDB,
    *,
    name: str,
    extensions: object,
) -> int:
    character_id = db.add_character_card(
        {
            "name": name,
            "description": "Export portability test",
            "extensions": extensions,
        }
    )
    assert type(character_id) is int
    return character_id


def _detailed_import() -> Callable[..., object]:
    importer = getattr(
        character_lib,
        "import_and_save_character_from_file_with_outcome",
        None,
    )
    assert callable(importer), "the structured character import API is missing"
    return cast(Callable[..., object], importer)


def test_valid_attachment_is_returned_but_stripped_before_character_persistence(
    db_instance: CharactersRAGDB,
) -> None:
    outcome = _detailed_import()(db_instance, json.dumps(_card(attachment=_valid_attachment())).encode())

    assert outcome is not None
    assert outcome.created is True
    assert outcome.warning_code is None
    assert str(outcome.portable_profile.profile_id) == _valid_attachment()["profile_id"]
    stored = db_instance.get_character_card_by_id(outcome.character_id)
    assert stored["extensions"] == {"unrelated/example": {"kept": True}}


def test_duplicate_name_returns_reused_outcome_without_mutating_existing_card(
    db_instance: CharactersRAGDB,
) -> None:
    first = _detailed_import()(
        db_instance,
        json.dumps(_card(attachment=_valid_attachment())).encode(),
    )
    duplicate_card = _card(attachment=_valid_attachment())
    duplicate_card["data"]["description"] = "must not replace the existing row"

    second = _detailed_import()(db_instance, json.dumps(duplicate_card).encode())

    assert first is not None and second is not None
    assert first.created is True
    assert second.created is False
    assert second.character_id == first.character_id
    stored = db_instance.get_character_card_by_id(first.character_id)
    assert stored["description"] == "A local character"


@pytest.mark.parametrize(
    ("attachment", "warning_code"),
    [
        ({**_valid_attachment(), "schema_version": 99}, "unsupported_version"),
        ({**_valid_attachment(), "provider_id": "future_tts"}, "unsupported_provider"),
        ({**_valid_attachment(), "unexpected": True}, "invalid_attachment"),
    ],
)
def test_skipped_or_invalid_attachment_does_not_block_or_persist_character(
    db_instance: CharactersRAGDB,
    attachment: object,
    warning_code: str,
) -> None:
    outcome = _detailed_import()(
        db_instance,
        json.dumps(_card(name=f"Character {warning_code}", attachment=attachment)).encode(),
    )

    assert outcome is not None
    assert outcome.created is True
    assert outcome.portable_profile is None
    assert outcome.warning_code == warning_code
    stored = db_instance.get_character_card_by_id(outcome.character_id)
    assert CHARACTER_CARD_TTS_EXTENSION_KEY not in stored["extensions"]
    assert stored["extensions"]["unrelated/example"] == {"kept": True}


def test_legacy_import_wrapper_still_returns_only_the_character_id(
    db_instance: CharactersRAGDB,
) -> None:
    imported_id = character_lib.import_and_save_character_from_file(
        db_instance,
        json.dumps(_card(attachment=_valid_attachment())).encode(),
    )

    assert type(imported_id) is int


def test_import_logging_does_not_disclose_the_source_filesystem_path(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    private_dir = tmp_path / "private-provider-origin-and-credential"
    private_dir.mkdir()
    secret_suffix = "credential-origin-secret"
    card_path = private_dir / f"untrusted-card.{secret_suffix}"
    card_path.write_text("{}", encoding="utf-8")
    messages: list[str] = []
    sink = loguru_logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
    )
    try:
        outcome = _detailed_import()(db_instance, str(card_path))
    finally:
        loguru_logger.remove(sink)

    assert outcome is None
    assert str(card_path) not in "".join(messages)
    assert secret_suffix not in "".join(messages)


def test_personas_local_wrapper_exposes_the_structured_import_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()
    fake_db = object()
    monkeypatch.setattr(character_handler, "_default_character_db", lambda: fake_db)
    monkeypatch.setattr(
        character_lib,
        "import_and_save_character_from_file_with_outcome",
        lambda db, path: expected if (db, path) == (fake_db, "/chosen/card.json") else None,
    )
    importer = getattr(character_handler, "import_character_card_with_outcome", None)

    assert callable(importer), "the Personas structured import wrapper is missing"
    assert importer("/chosen/card.json") is expected


def test_read_only_attachment_inspection_returns_valid_profile_without_a_database() -> (
    None
):
    inspector = getattr(
        character_lib,
        "inspect_character_card_tts_attachment",
        None,
    )
    assert callable(inspector), "the read-only attachment inspection API is missing"

    result = inspector(json.dumps(_card(attachment=_valid_attachment())).encode())

    assert result.portable_profile == _portable_profile()
    assert result.warning_code is None


def test_string_attachment_inspection_rejects_path_validation_violation(
    tmp_path: Path,
) -> None:
    unsafe_path = tmp_path / "private;card.json"
    unsafe_path.write_text(
        json.dumps(_card(attachment=_valid_attachment())),
        encoding="utf-8",
    )

    result = character_lib.inspect_character_card_tts_attachment(str(unsafe_path))

    assert result is None


def test_read_only_attachment_inspection_reports_absence_and_bounded_warning() -> None:
    inspector = getattr(character_lib, "inspect_character_card_tts_attachment", None)
    assert callable(inspector), "the read-only attachment inspection API is missing"
    without_attachment = _card(attachment=_valid_attachment())
    without_attachment["data"]["extensions"].pop(
        CHARACTER_CARD_TTS_EXTENSION_KEY
    )

    absent = inspector(json.dumps(without_attachment).encode())
    unsupported = inspector(
        json.dumps(
            _card(
                attachment={**_valid_attachment(), "schema_version": 99},
            )
        ).encode()
    )

    assert absent.portable_profile is None
    assert absent.warning_code is None
    assert unsupported.portable_profile is None
    assert unsupported.warning_code == "unsupported_version"


def test_read_only_attachment_inspection_never_logs_character_text() -> None:
    secret = "private roleplay message and credential https://user:key@example.test"
    payload = {
        "name": secret,
        "description": secret,
        "extensions": {},
    }
    messages: list[str] = []
    sink = loguru_logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        result = character_lib.inspect_character_card_tts_attachment(
            json.dumps(payload).encode()
        )
    finally:
        loguru_logger.remove(sink)

    assert result.portable_profile is None
    assert secret not in "".join(messages)


def test_attachment_inspection_never_logs_spec_or_lorebook_values() -> None:
    secret = "credential https://user:key@private-origin.invalid/message-text"
    payload = _card(attachment=_valid_attachment())
    payload["spec"] = secret
    payload["data"]["character_book"] = {
        "name": secret,
        "entries": [secret],
    }
    messages: list[str] = []
    sink = loguru_logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        result = character_lib.inspect_character_card_tts_attachment(
            json.dumps(payload).encode()
        )
    finally:
        loguru_logger.remove(sink)

    assert result.portable_profile == _portable_profile()
    assert secret not in "".join(messages)


def test_attachment_inspection_hides_lorebook_conversion_exception_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "credential-private-origin-message-text"
    payload = _card(attachment=_valid_attachment())
    payload["data"]["character_book"] = {"entries": []}
    monkeypatch.setattr(
        character_lib,
        "character_book_to_world_book_block",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    messages: list[str] = []
    sink = loguru_logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        result = character_lib.inspect_character_card_tts_attachment(
            json.dumps(payload).encode()
        )
    finally:
        loguru_logger.remove(sink)

    assert result.portable_profile == _portable_profile()
    assert secret not in "".join(messages)


def test_default_json_export_is_tts_free_and_does_not_mutate_extensions(
    db_instance: CharactersRAGDB,
) -> None:
    extensions = {"unrelated/example": {"kept": True}}
    character_id = _add_character(
        db_instance,
        name="Default export",
        extensions=extensions,
    )

    exported = character_lib.export_character_card_to_json(
        db_instance,
        character_id,
        include_image=False,
    )

    assert exported is not None
    payload = json.loads(exported)
    assert payload["data"]["extensions"] == extensions
    assert CHARACTER_CARD_TTS_EXTENSION_KEY not in payload["data"]["extensions"]
    stored = db_instance.get_character_card_by_id(character_id)
    assert stored["extensions"] == extensions


def test_opt_in_json_export_adds_sanitized_attachment_to_transient_copy(
    db_instance: CharactersRAGDB,
) -> None:
    extensions = {"unrelated/example": {"kept": True}}
    character_id = _add_character(
        db_instance,
        name="Opt-in export",
        extensions=extensions,
    )

    exported = character_lib.export_character_card_to_json(
        db_instance,
        character_id,
        include_image=False,
        portable_tts_profile=_portable_profile(),
    )

    assert exported is not None
    payload = json.loads(exported)
    assert payload["data"]["extensions"]["unrelated/example"] == {"kept": True}
    assert (
        payload["data"]["extensions"][CHARACTER_CARD_TTS_EXTENSION_KEY]
        == _valid_attachment()
    )
    stored = db_instance.get_character_card_by_id(character_id)
    assert stored["extensions"] == extensions


@pytest.mark.parametrize(
    "extensions",
    [
        "malformed",
        {CHARACTER_CARD_TTS_EXTENSION_KEY: {"do_not": "overwrite"}},
    ],
)
def test_opt_in_json_export_fails_closed_for_invalid_reserved_namespace(
    db_instance: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    extensions: object,
) -> None:
    monkeypatch.setattr(
        db_instance,
        "get_character_card_by_id",
        lambda _character_id: {
            "name": "Invalid extension",
            "extensions": extensions,
        },
    )

    exported = character_lib.export_character_card_to_json(
        db_instance,
        1,
        include_image=False,
        portable_tts_profile=_portable_profile(),
    )

    assert exported is None


def test_opt_in_png_export_embeds_the_same_sanitized_attachment(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    character_id = _add_character(
        db_instance,
        name="PNG export",
        extensions={"unrelated/example": {"kept": True}},
    )
    target = tmp_path / "portable-card.png"

    exported = character_lib.export_character_card_to_png(
        db_instance,
        character_id,
        str(target),
        base_directory=str(tmp_path),
        portable_tts_profile=_portable_profile(),
    )

    assert exported is True
    embedded = character_lib.extract_json_from_image_file(target.read_bytes())
    assert embedded is not None
    payload = json.loads(embedded)
    assert (
        payload["data"]["extensions"][CHARACTER_CARD_TTS_EXTENSION_KEY]
        == _valid_attachment()
    )


def test_png_export_path_failure_never_logs_sensitive_destination(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    secret = "credential-private-origin-message-text"
    hidden_parent = tmp_path / f".{secret}"
    hidden_parent.mkdir()
    target = hidden_parent / "portable-card.png"
    character_id = _add_character(
        db_instance,
        name="Private PNG destination",
        extensions={},
    )
    messages: list[str] = []
    sink = loguru_logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        exported = character_lib.export_character_card_to_png(
            db_instance,
            character_id,
            str(target),
            base_directory=str(hidden_parent),
            portable_tts_profile=_portable_profile(),
        )
    finally:
        loguru_logger.remove(sink)

    assert exported is False
    assert not target.exists()
    assert secret not in "".join(messages)


def test_default_png_export_is_tts_free_and_preserves_unrelated_extensions(
    db_instance: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    extensions = {"unrelated/example": {"kept": True}}
    character_id = _add_character(
        db_instance,
        name="Default PNG export",
        extensions=extensions,
    )
    target = tmp_path / "ordinary-card.png"

    exported = character_lib.export_character_card_to_png(
        db_instance,
        character_id,
        str(target),
        base_directory=str(tmp_path),
    )

    assert exported is True
    embedded = character_lib.extract_json_from_image_file(target.read_bytes())
    assert embedded is not None
    payload = json.loads(embedded)
    assert payload["data"]["extensions"] == extensions
    assert CHARACTER_CARD_TTS_EXTENSION_KEY not in payload["data"]["extensions"]
    assert db_instance.get_character_card_by_id(character_id)["extensions"] == extensions


def test_valid_png_attachment_is_returned_and_stripped_before_persistence(
    db_instance: CharactersRAGDB,
) -> None:
    outcome = _detailed_import()(
        db_instance,
        _png_card_bytes(_card(name="Portable PNG", attachment=_valid_attachment())),
    )

    assert outcome is not None
    assert outcome.created is True
    assert outcome.portable_profile == _portable_profile()
    assert outcome.warning_code is None
    stored = db_instance.get_character_card_by_id(outcome.character_id)
    assert stored["extensions"] == {"unrelated/example": {"kept": True}}


def test_hostile_png_attachment_is_skipped_and_never_persisted(
    db_instance: CharactersRAGDB,
) -> None:
    hostile_attachment = {**_valid_attachment(), "unexpected": "private value"}

    outcome = _detailed_import()(
        db_instance,
        _png_card_bytes(
            _card(name="Hostile portable PNG", attachment=hostile_attachment)
        ),
    )

    assert outcome is not None
    assert outcome.created is True
    assert outcome.portable_profile is None
    assert outcome.warning_code == "invalid_attachment"
    stored = db_instance.get_character_card_by_id(outcome.character_id)
    assert CHARACTER_CARD_TTS_EXTENSION_KEY not in stored["extensions"]
    assert stored["extensions"]["unrelated/example"] == {"kept": True}


def test_malformed_png_metadata_never_logs_untrusted_content() -> None:
    secret = "credential-private-origin-message-text"
    messages: list[str] = []
    sink = loguru_logger.add(lambda message: messages.append(str(message)), level="DEBUG")
    try:
        result = character_lib.inspect_character_card_tts_attachment(
            _png_card_bytes({}, encoded=f"!!!!{secret}!!!!")
        )
    finally:
        loguru_logger.remove(sink)

    assert result is None
    assert secret not in "".join(messages)
