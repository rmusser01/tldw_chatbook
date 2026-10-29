"""One file of every format today's four Roleplay import pickers accept (B2b).

Frame slice B2b (spec 1.4.6) puts the four pickers behind one Import picker;
these fixtures exercise it. Not a test module (no ``test_`` prefix), so pytest
never collects it. ``write_import_fixtures`` writes real, importable files:

* character cards: V1, V2 and V3 JSON, a PNG with a ``chara`` tEXt chunk (the
  live harness's ``_png_card_bytes`` shape), a WebP with the card in EXIF
  UserComment, Markdown with YAML frontmatter and with a json block, and a
  frontmatter card in a ``.yaml`` file (dev's card picker read any non-image
  file as text under All Files);
* a Buddy pack (``.tldw-persona-vpack``, built like
  ``Tests/Persona_Visual``'s importer tests) and an Actor Pack (the committed
  ``minimal-character.tldw-actor-pack`` golden);
* lore books: tldw's export, the plain ``entries`` list form and SillyTavern's
  World Info object form;
* chat dictionaries: tldw's export (``dictionary_id`` + ``data``), the
  ``data``-wrapped import form, the flat form and Markdown.

``EXPECTED_KINDS`` maps each fixture name to the kind it imports as.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ACTOR_PACK_GOLDEN = (
    REPO / "Tests/Actor_Packs/fixtures/export-golden/minimal-character.tldw-actor-pack"
)

_V2_DATA = {
    "name": "Wren Halloway",
    "description": "A cartographer of places that move.",
    "personality": "Curious, restless.",
    "scenario": "",
    "first_mes": "Hold the map still -- it's trying to fold itself again.",
    "mes_example": "",
    "creator_notes": "B2b import fixture.",
    "system_prompt": "",
    "post_history_instructions": "",
    "alternate_greetings": [],
    "tags": ["imported"],
    "creator": "b2b",
    "character_version": "1.0",
    "extensions": {},
}

#: Fixture file name -> the kind it imports as.
EXPECTED_KINDS: dict[str, str] = {
    "card_v1.json": "character",
    "card_v2.json": "character",
    "card_v3.json": "character",
    "card.png": "character",
    "card.webp": "character",
    "card_frontmatter.md": "character",
    "card_json_block.markdown": "character",
    "card_frontmatter.yaml": "character",
    "buddy.tldw-persona-vpack": "buddy",
    "actor.tldw-actor-pack": "actor_pack",
    "lore_tldw_export.json": "lore",
    "lore_entries_list.json": "lore",
    "lore_sillytavern.json": "lore",
    "dictionary_export.json": "dictionary",
    "dictionary_data.json": "dictionary",
    "dictionary_flat.json": "dictionary",
    "dictionary.md": "dictionary",
}


def card_v2(name: str = "Wren Halloway") -> dict:
    """A CCv2 card named ``name``."""
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {**_V2_DATA, "name": name},
    }


def png_card_bytes(card: dict | None = None) -> bytes:
    """A PNG character card: the card JSON, base64, in a ``chara`` tEXt chunk."""
    from PIL import Image, PngImagePlugin

    meta = PngImagePlugin.PngInfo()
    payload = json.dumps(card or card_v2()).encode()
    meta.add_text("chara", base64.b64encode(payload).decode())
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (70, 110, 160)).save(buffer, "PNG", pnginfo=meta)
    return buffer.getvalue()


def webp_card_bytes(card: dict | None = None) -> bytes:
    """A WebP character card: the card JSON, base64, in EXIF UserComment."""
    from PIL import Image

    exif = Image.Exif()
    exif[37510] = base64.b64encode(json.dumps(card or card_v2()).encode()).decode()
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (70, 110, 160)).save(buffer, "WEBP", exif=exif)
    return buffer.getvalue()


def _texts() -> dict[str, str]:
    v3_card = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        # Only the name, the description and notes: nothing but the spec
        # says "card" (CCv3 makes most fields optional).
        "data": {
            "name": "Ilse Varn",
            "description": "A lighthouse keeper.",
            "creator_notes": "V3",
        },
    }
    v1_card = {
        "name": "Corwin Ash",
        "description": "A knight who left the order.",
        "personality": "Gruff.",
        "first_mes": "Seat's taken.",
        "mes_example": "",
    }
    lore_export = {
        "name": "Kepler Atlas",
        "description": "Stars.",
        "scan_depth": 3,
        "token_budget": 500,
        "recursive_scanning": False,
        "entries": [{"keys": ["kepler"], "content": "A star map.", "enabled": True}],
    }
    lore_list = {
        "name": "Harbour Lore",
        "entries": [{"keys": ["harbour"], "content": "Fog at dawn."}],
    }
    lore_sillytavern = {
        "entries": {
            "0": {
                "uid": 0,
                "key": ["tide"],
                "keysecondary": [],
                "comment": "Tides",
                "content": "Twice a day.",
            }
        }
    }
    dictionary_entries = [
        {"key": "BP", "content": "blood pressure", "probability": 100},
        {"key": "HR", "content": "heart rate", "probability": 100},
    ]
    dictionary_export = {
        "dictionary_id": 7,
        "data": {
            "name": "Medical Abbrev",
            "description": "",
            "entries": dictionary_entries,
            "strategy": "sorted_evenly",
            "max_tokens": 1000,
            "enabled": True,
        },
        "source": "local",
    }
    dictionary_data = {
        "data": {"name": "Ward Shorthand", "entries": dictionary_entries}
    }
    dictionary_flat = {
        "name": "Ship Talk",
        "entries": [{"pattern": "aye", "replacement": "yes"}],
        "strategy": "sorted_evenly",
        "max_tokens": 500,
    }
    frontmatter_card = (
        "---\nname: Mara Quill\ndescription: A scribe.\n---\nShe writes.\n"
    )
    yaml_card = "---\nname: Tam Reyes\ndescription: A pilot.\n---\n"
    json_block_card = (
        "# Card\n\n```json\n" + json.dumps(card_v2("Odo Fenn")) + "\n```\n"
    )
    dictionary_md = "BP: blood pressure\n---@@@---\nHR: heart rate\n"
    return {
        "card_v1.json": json.dumps(v1_card),
        "card_v2.json": json.dumps(card_v2()),
        "card_v3.json": json.dumps(v3_card),
        "card_frontmatter.md": frontmatter_card,
        "card_json_block.markdown": json_block_card,
        "card_frontmatter.yaml": yaml_card,
        "lore_tldw_export.json": json.dumps(lore_export),
        "lore_entries_list.json": json.dumps(lore_list),
        "lore_sillytavern.json": json.dumps(lore_sillytavern),
        "dictionary_export.json": json.dumps(dictionary_export),
        "dictionary_data.json": json.dumps(dictionary_data),
        "dictionary_flat.json": json.dumps(dictionary_flat),
        "dictionary.md": dictionary_md,
    }


def write_import_fixtures(directory: Path) -> dict[str, Path]:
    """Write one file per name in ``EXPECTED_KINDS`` into ``directory``.

    Returns:
        Fixture name -> its path.
    """
    from Tests.Persona_Visual.test_persona_visual_importer import _write_archive

    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, text in _texts().items():
        paths[name] = directory / name
        paths[name].write_text(text, encoding="utf-8")
    paths["card.png"] = directory / "card.png"
    paths["card.png"].write_bytes(png_card_bytes())
    paths["card.webp"] = directory / "card.webp"
    paths["card.webp"].write_bytes(webp_card_bytes())
    paths["buddy.tldw-persona-vpack"] = _write_archive(
        directory / "buddy.tldw-persona-vpack"
    )
    paths["actor.tldw-actor-pack"] = directory / "actor.tldw-actor-pack"
    shutil.copyfile(ACTOR_PACK_GOLDEN, paths["actor.tldw-actor-pack"])
    assert set(paths) == set(EXPECTED_KINDS)
    return paths


def write_sparse(path: Path, size: int) -> Path:
    """A file of ``size`` bytes that takes no disk (a hole after one byte)."""
    with open(path, "wb") as handle:
        handle.write(b"{")
        handle.truncate(size)
    return path


def write_nested_json(path: Path, nbytes: int, *, opener: str = "[") -> Path:
    """``nbytes`` of one JSON opener repeated: nesting far past any parser's limit.

    Args:
        path: Where to write.
        nbytes: About how many bytes to write.
        opener: ``"["`` or ``'{"a":'``.
    """
    path.write_bytes(opener.encode() * max(1, nbytes // len(opener)))
    return path


__all__ = [
    "ACTOR_PACK_GOLDEN",
    "EXPECTED_KINDS",
    "card_v2",
    "png_card_bytes",
    "webp_card_bytes",
    "write_import_fixtures",
    "write_nested_json",
    "write_sparse",
]
