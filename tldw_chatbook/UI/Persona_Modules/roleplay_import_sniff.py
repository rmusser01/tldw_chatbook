"""Roleplay's bounded import sniff: which kind a picked file is (frame slice B2b).

Spec 1.4.6. The one Import picker accepts every format today's four pickers
accept, and routes each file to its own kind's importer. The type comes from
the extension plus a bounded, advisory look at the file's first bytes:

* Actor packs (``.tldw-actor-pack``) and Buddy packs (``.tldw-persona-vpack``)
  route by extension alone, before any size cap or read: they are archives
  with their own limits (``Actor_Packs/importer.py``, ``Persona_Visual``),
  legitimately far larger than 10 MB, and the sniff never opens an archive.
* Everything else is checked with ``validate_path_simple``, refused above the
  caller's size cap by ``stat`` (before any read), then read once, at most
  ``SNIFF_MAX_BYTES``.
* PNG and WebP are decided by signature (a character card image); no pixel is
  decoded and no chunk is parsed.
* JSON is parsed partially (``pydantic_core.from_json(..., allow_partial=True)``)
  for its top-level keys and its ``data`` keys. A card ``spec`` wins; a chat
  dictionary export (``dictionary_id`` with ``data``) is told apart from the
  generic V2 ``data`` shape before that shape is read as a card.
* Markdown gets linear checks only (never the card parser, which is
  quadratic on adversarial input). A Markdown file whose text starts with
  ``{`` is read as JSON first, so a card saved as ``.md`` keeps its
  ``spec``; dev's card importer reads such a file's raw JSON. Only a
  ``---@@@---`` line decides a chat dictionary: ``key: value`` lines alone
  (a README has them too) ask, with Chat dictionary preselected (owner
  decision D8).
* Any other extension (``.yaml``, ``.txt``, none) is a character card when
  its text is YAML frontmatter or holds a json block (fenced with or without
  ``json``: the card importer's own pattern takes both), the same linear
  checks: dev's card picker read every non-image file as text under All
  Files, and no other kind's importer reads such a file.

Verdicts: ``decided`` (one kind), ``ask`` (a valid file that fits more than one
kind, or whose deciding keys lie past the bytes read: the screen asks "Import
as:"), ``not_roleplay`` (undecodable, too deeply nested, a signature that does
not match, or no Roleplay shape at all) and ``too_large``. Inconclusive means
``ask``, never ``not_roleplay``: a strict sniff would refuse files dev's
per-kind pickers accept.

The screen imports this module lazily, from its import worker, so the
Roleplay route pre-imports nothing new (spec 5.10).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic_core import from_json

from tldw_chatbook.Utils.path_validation import validate_path_simple

ImportKind = Literal["character", "lore", "dictionary", "actor_pack", "buddy"]
Verdict = Literal["decided", "ask", "not_roleplay", "too_large"]

#: The most the sniff ever reads of one file.
SNIFF_MAX_BYTES = 65_536

#: Archive formats, routed by extension before any cap or read.
PACK_SUFFIXES: dict[str, ImportKind] = {
    ".tldw-actor-pack": "actor_pack",
    ".tldw-persona-vpack": "buddy",
}
JSON_SUFFIXES = frozenset({".json"})
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
IMAGE_SUFFIXES = frozenset({".png", ".webp"})
#: The kinds a JSON or Markdown file can be ("Import as:"), in that order.
TEXT_KINDS: tuple[ImportKind, ...] = ("character", "lore", "dictionary")

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_UTF8_BOM = b"\xef\xbb\xbf"
_CARD_SPECS = frozenset({"chara_card_v2", "chara_card_v3"})
#: Root keys only character cards use (V1, and the formats
#: ``character_card_formats.py`` detects: Agnai, CharacterAI, KoboldAI,
#: TextGen, Chub).
_CARD_KEYS = frozenset(
    {
        "first_mes",
        "mes_example",
        "personality",
        "scenario",
        "alternate_greetings",
        "char_name",
        "char_persona",
        "char_greeting",
        "greeting",
        "persona",
        "participant__name",
        "external_id",
        "world_scenario",
    }
)
#: Root keys of tldw's lore export.
_LORE_KEYS = frozenset({"scan_depth", "token_budget", "recursive_scanning"})
#: Root keys of a flat chat dictionary.
_DICTIONARY_KEYS = frozenset({"strategy", "max_tokens", "default_token_budget"})
_DICTIONARY_SEPARATOR = "---@@@---"
#: The opening of a json block, fenced ```json or bare ``` (the card
#: importer's pattern, ```(?:json)?\s*(\{.*?\})\s*```, takes both). Linear:
#: only the opening is matched, never the block.
_JSON_FENCE = re.compile(r"```(?:json)?\s*\{", re.IGNORECASE)


@dataclass(frozen=True)
class SniffResult:
    """What the sniff concluded about one file.

    Attributes:
        verdict: ``decided``, ``ask``, ``not_roleplay`` or ``too_large``.
        kinds: The one kind when decided; the candidates when asking (in
            ``TEXT_KINDS`` order); empty otherwise.
        reason: Why, in a few plain words (for logs and tests; the screen
            writes its own toast copy).
        preselect: When asking, the kind "Import as:" opens on; None opens
            on the first row the screen enables.
    """

    verdict: Verdict
    kinds: tuple[ImportKind, ...]
    reason: str
    preselect: ImportKind | None = None


def _decided(kind: ImportKind, reason: str) -> SniffResult:
    return SniffResult("decided", (kind,), reason)


def _not_roleplay(reason: str) -> SniffResult:
    return SniffResult("not_roleplay", (), reason)


def _kinds_result(found: set[ImportKind], reason: str) -> SniffResult:
    """One kind decides; several (or none: inconclusive) ask."""
    if len(found) == 1:
        return _decided(next(iter(found)), reason)
    kinds = tuple(kind for kind in TEXT_KINDS if kind in found) or TEXT_KINDS
    return SniffResult("ask", kinds, reason)


def _complete_utf8(prefix: bytes) -> bytes:
    """Drop a multi-byte character the read cut in half at the prefix's end."""
    for cut in range(4):
        head = prefix[: len(prefix) - cut]
        try:
            head.decode("utf-8")
        except UnicodeDecodeError:
            continue
        return head
    return prefix


def _is_dict_list(value: object) -> bool:
    return isinstance(value, list) and any(isinstance(item, Mapping) for item in value)


def _entry_kinds(entries: object) -> set[ImportKind]:
    """Which kinds an ``entries`` value's items look like.

    Lore entries carry ``keys`` (a list) or, in SillyTavern's object form,
    ``key`` (a list) and ``uid``; chat dictionary entries carry ``pattern``
    and ``replacement``, or ``key`` as one string with ``content``.
    """
    if isinstance(entries, Mapping):
        items = list(entries.values())
    elif isinstance(entries, list):
        items = entries
    else:
        return set()
    found: set[ImportKind] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if isinstance(item.get("keys"), list) or isinstance(item.get("key"), list):
            found.add("lore")
        elif "uid" in item:
            found.add("lore")
        if "pattern" in item or "replacement" in item:
            found.add("dictionary")
        elif isinstance(item.get("key"), str) and "content" in item:
            found.add("dictionary")
    return found


def _object_kinds(obj: Mapping[str, object]) -> set[ImportKind]:
    """The kinds one JSON object's own keys point at (no ``spec`` rule here)."""
    found: set[ImportKind] = set()
    keys = set(obj)
    if keys & _CARD_KEYS:
        found.add("character")
    if keys & _LORE_KEYS:
        found.add("lore")
    if keys & _DICTIONARY_KEYS:
        found.add("dictionary")
    found |= _entry_kinds(obj.get("entries"))
    return found


def _classify_json(text: bytes) -> SniffResult:
    try:
        parsed = from_json(text, allow_partial=True)
    except (ValueError, RecursionError) as exc:
        # Invalid UTF-8, invalid JSON and pydantic_core's own recursion limit
        # (deep nesting raises ValueError) land here, in microseconds.
        return _not_roleplay(f"not readable JSON ({type(exc).__name__})")
    if not isinstance(parsed, Mapping):
        return _not_roleplay("JSON that is not an object")
    spec = parsed.get("spec")
    if isinstance(spec, str) and spec in _CARD_SPECS:
        return _decided("character", f"card spec {spec}")
    data = parsed.get("data")
    if "dictionary_id" in parsed and isinstance(data, Mapping):
        return _decided("dictionary", "chat dictionary export")
    found = _object_kinds(parsed)
    if isinstance(data, Mapping):
        # The V2 shape and the dictionary import shape both wrap a ``data``
        # object: read its keys like the root's.
        found |= _object_kinds(data)
    return _kinds_result(found, "JSON keys")


def _text_card_markers(body: str, lines: list[str]) -> bool:
    """YAML frontmatter (the first non-blank line is ``---``) or a json block,
    fenced ```json or bare ```: the two text forms the card importer reads
    besides raw JSON."""
    first = next((line.strip() for line in lines if line.strip()), "")
    return first == "---" or _JSON_FENCE.search(body) is not None


def _classify_markdown(text: bytes) -> SniffResult:
    try:
        body = text.decode("utf-8")
    except UnicodeDecodeError:
        return _not_roleplay("Markdown that is not UTF-8 text")
    lines = body.splitlines()
    card = _text_card_markers(body, lines)
    dictionary = any(line.strip() == _DICTIONARY_SEPARATOR for line in lines)
    if card and dictionary:
        return SniffResult("ask", ("character", "dictionary"), "card and dictionary")
    if card:
        return _decided("character", "Markdown card (frontmatter or a json block)")
    if dictionary:
        return _decided("dictionary", "Markdown rules (a ---@@@--- line)")
    if any(":" in line and line.split(":", 1)[1].strip() for line in lines):
        # A README or notes file has ``key: value`` lines too, so they only
        # suggest a chat dictionary (owner decision D8, 2026-10-29). A card's
        # json block may still lie past the bytes read.
        return SniffResult(
            "ask",
            ("character", "dictionary"),
            "Markdown key: value lines",
            preselect="dictionary",
        )
    return _not_roleplay("Markdown with no card and no rules")


def _classify_card_text(text: bytes) -> SniffResult:
    """Another extension (``.yaml``, ``.txt``, none) that is not JSON: a card
    when it carries frontmatter or a json block, else no Roleplay format."""
    try:
        body = text.decode("utf-8")
    except UnicodeDecodeError:
        return _not_roleplay("no Roleplay format")
    if _text_card_markers(body, body.splitlines()):
        return _decided("character", "text card (frontmatter or a json block)")
    return _not_roleplay("no Roleplay format")


def classify(suffix: str, prefix: bytes) -> SniffResult:
    """Decide a file's kind from its extension and its first bytes (pure).

    Args:
        suffix: The file's extension, any case (``".json"``, ``""``).
        prefix: At most ``SNIFF_MAX_BYTES`` from the start of the file.

    Returns:
        The verdict. Never raises on any input.
    """
    suffix = suffix.lower()
    pack = PACK_SUFFIXES.get(suffix)
    if pack is not None:
        return _decided(pack, "pack extension")
    if prefix.startswith(_PNG_SIGNATURE) or (
        prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP"
    ):
        return _decided("character", "card image signature")
    if suffix in IMAGE_SUFFIXES:
        return _not_roleplay("an image extension without an image signature")
    text = _complete_utf8(prefix.removeprefix(_UTF8_BOM))
    if suffix in MARKDOWN_SUFFIXES:
        if text.lstrip()[:1] == b"{":
            # A JSON card saved as Markdown: dev's card importer reads its raw
            # JSON, so its ``spec`` must still win (spec 1.4.6). Text that is
            # not readable JSON gets the Markdown checks.
            as_json = _classify_json(text)
            if as_json.verdict != "not_roleplay":
                return as_json
        return _classify_markdown(text)
    if suffix in JSON_SUFFIXES:
        return _classify_json(text)
    if text.lstrip()[:1] in (b"{", b"["):
        # Any other extension that starts like JSON is sniffed as JSON; text
        # that is not readable JSON (``[Character card]`` over a json block)
        # still gets the card-text checks, as dev's card picker reads it.
        as_json = _classify_json(text)
        if as_json.verdict != "not_roleplay":
            return as_json
    return _classify_card_text(text)


def sniff_import_file(path: str | os.PathLike[str], *, max_bytes: int) -> SniffResult:
    """Sniff one picked file, reading at most ``SNIFF_MAX_BYTES`` of it.

    Packs route by extension first and are never validated, measured or read
    here. Everything else is refused above ``max_bytes`` by ``stat``, before
    any read.

    Args:
        path: The picked file.
        max_bytes: The size cap for cards, lore books and chat dictionaries.

    Returns:
        The verdict.

    Raises:
        ValueError: The path fails ``validate_path_simple``.
        OSError: The file cannot be stat'ed or read.
    """
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    pack = PACK_SUFFIXES.get(suffix)
    if pack is not None:
        return _decided(pack, "pack extension")
    checked = validate_path_simple(path, require_exists=True)
    size = checked.stat().st_size
    if size > max_bytes:
        return SniffResult("too_large", (), f"{size} bytes, over {max_bytes}")
    with open(checked, "rb") as handle:
        prefix = handle.read(SNIFF_MAX_BYTES)
    return classify(suffix, prefix)


__all__ = [
    "IMAGE_SUFFIXES",
    "JSON_SUFFIXES",
    "MARKDOWN_SUFFIXES",
    "PACK_SUFFIXES",
    "SNIFF_MAX_BYTES",
    "TEXT_KINDS",
    "ImportKind",
    "SniffResult",
    "Verdict",
    "classify",
    "sniff_import_file",
]
