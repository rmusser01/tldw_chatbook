"""Roleplay frame B2b: the bounded import sniff (spec 1.4.6; AC#4, AC#6).

``UI/Persona_Modules/roleplay_import_sniff.py`` decides which kind a picked
file is before the one Import picker routes it. Pure and file-level contracts
only: nothing here mounts ``PersonasScreen`` (UI Fast Lane). The routing
through the picker is ``test_roleplay_import_app.py``'s.
"""

from __future__ import annotations

import builtins
import json
import pathlib
import time
import zipfile

import pytest

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
import tldw_chatbook.UI.Persona_Modules.roleplay_import_sniff as sniff
from Tests.UI.roleplay_import_fixtures import (
    EXPECTED_KINDS,
    card_v2,
    write_import_fixtures,
    write_nested_json,
    write_sparse,
)
from tldw_chatbook.Character_Chat import Character_Chat_Lib

CAP = 10 * 1024 * 1024


@pytest.fixture
def fixtures(tmp_path):
    return write_import_fixtures(tmp_path / "fixtures")


@pytest.mark.parametrize("name", sorted(EXPECTED_KINDS))
def test_every_format_of_the_four_pickers_sniffs_to_its_kind(fixtures, name):
    result = sniff.sniff_import_file(fixtures[name], max_bytes=CAP)
    assert (result.verdict, result.kinds) == ("decided", (EXPECTED_KINDS[name],))


def test_cross_kind_shapes_dev_accepts_silently_are_told_apart(fixtures):
    """On dev each per-kind importer accepts the other kinds' JSON: SillyTavern
    lore imports as a character "Book", a dictionary as a character "Dict", a
    CCv2 card as a lore book with no entries, and the dictionary export
    matches the generic V2 ``data`` check (Survey 2 §0 item 6, measured). B2b
    routes each to its own kind; the per-kind importers stay as they are."""
    cases = {
        # SillyTavern World Info carries a name, like a card would.
        "lore": {
            **json.loads(fixtures["lore_sillytavern.json"].read_text()),
            "name": "Book",
        },
        "dictionary": {
            "name": "Dict",
            "entries": [{"key": "a", "content": "b"}],
            "max_tokens": 10,
        },
        "character": card_v2(),
    }
    for kind, payload in cases.items():
        result = sniff.classify(".json", json.dumps(payload).encode())
        assert (result.verdict, result.kinds) == ("decided", (kind,)), kind
    export = fixtures["dictionary_export.json"].read_bytes()
    assert sniff.classify(".json", export).kinds == ("dictionary",)


def test_an_11_mb_file_is_refused_before_any_read(tmp_path, monkeypatch):
    path = write_sparse(tmp_path / "huge.json", 11 * 1024 * 1024)
    opened: list[object] = []

    def spy_open(*args, **kwargs):
        opened.append(args[0] if args else kwargs)
        return builtins.open(*args, **kwargs)

    monkeypatch.setattr(sniff, "open", spy_open, raising=False)
    monkeypatch.setattr(
        pathlib.Path, "read_bytes", lambda self: opened.append(self) or b""
    )
    monkeypatch.setattr(
        pathlib.Path, "open", lambda self, *a, **k: opened.append(self) or None
    )
    result = sniff.sniff_import_file(path, max_bytes=CAP)
    assert result.verdict == "too_large"
    assert opened == []


@pytest.mark.parametrize("suffix", [".tldw-actor-pack", ".tldw-persona-vpack"])
def test_packs_route_by_extension_and_skip_the_cap(tmp_path, suffix):
    """Actor and Buddy packs legitimately pass 10 MB (archive limits are the
    importers' own); the sniff never measures or opens them (OQ3 iii)."""
    path = write_sparse(tmp_path / f"big{suffix}", 11 * 1024 * 1024)
    result = sniff.sniff_import_file(path, max_bytes=CAP)
    assert result.verdict == "decided"
    assert result.kinds == (sniff.PACK_SUFFIXES[suffix],)


@pytest.mark.parametrize("opener", ["[", '{"a":'])
def test_a_deeply_nested_9_mb_json_is_not_a_roleplay_file_and_raises_nothing(
    tmp_path, opener
):
    path = write_nested_json(tmp_path / "nested.json", 9 * 1024 * 1024, opener=opener)
    started = time.monotonic()
    result = sniff.sniff_import_file(path, max_bytes=CAP)
    assert result.verdict == "not_roleplay"
    assert result.kinds == ()
    # Evidence, not a threshold: a bounded read never needs a full parse.
    assert time.monotonic() - started < 5


def test_the_sniff_reads_at_most_64_kib(tmp_path, monkeypatch):
    card = card_v2()
    card["data"]["description"] = "x" * (5 * 1024 * 1024)
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    reads: list[int] = []
    real_open = builtins.open

    class Counting:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()

        def read(self, size=-1):
            data = self._handle.read(size)
            reads.append(len(data))
            return data

    monkeypatch.setattr(
        sniff, "open", lambda *a, **k: Counting(real_open(*a, **k)), raising=False
    )
    result = sniff.sniff_import_file(path, max_bytes=CAP)
    assert (result.verdict, result.kinds) == ("decided", ("character",))
    assert sum(reads) <= sniff.SNIFF_MAX_BYTES
    assert reads


def test_the_sniffer_never_opens_an_archive_or_decodes_pixels(
    fixtures, tmp_path, monkeypatch
):
    from PIL import Image

    calls: list[tuple] = []

    def refuse(*args, **kwargs):
        # Recorded as well as raised: a caller that swallows the error still fails.
        calls.append(args)
        raise AssertionError("the sniff opened an archive or decoded an image")

    zip_as_json = tmp_path / "archive.json"
    zip_as_json.write_bytes(fixtures["actor.tldw-actor-pack"].read_bytes())
    monkeypatch.setattr(zipfile, "ZipFile", refuse)
    monkeypatch.setattr(Image, "open", refuse)
    monkeypatch.setattr(Image.Image, "load", refuse)
    for name, path in fixtures.items():
        assert sniff.sniff_import_file(path, max_bytes=CAP).kinds == (
            EXPECTED_KINDS[name],
        ), name
    assert sniff.sniff_import_file(zip_as_json, max_bytes=CAP).verdict == "not_roleplay"
    assert calls == []


@pytest.mark.parametrize(
    ("suffix", "prefix"),
    [
        (".json", b"\xff\xfe\x00{"),
        (".json", b"[" * 70_000),
        (".json", b'{"a":' * 20_000),
        (".json", b'{"name": "x", "description": '),
        (".json", b"[1, 2, 3]"),
        (".md", b"\xff\xfe\xfd"),
        (".png", b"GIF89a"),
        (".webp", b"RIFF\x00\x00\x00\x00WAVE"),
        ("", b"plain words"),
        (".json", b""),
    ],
    # Short ids: a 70 KB prefix as a node id defeats grep-based replay tools.
    ids=[
        "json-not-utf8",
        "json-nested-lists",
        "json-nested-objects",
        "json-truncated-object",
        "json-list",
        "md-not-utf8",
        "png-gif-bytes",
        "webp-wave-bytes",
        "no-suffix-prose",
        "json-empty",
    ],
)
def test_undecodable_or_recursive_prefixes_mean_unknown_never_an_exception(
    suffix, prefix
):
    result = sniff.classify(suffix, prefix)
    assert result.verdict in ("not_roleplay", "ask")
    if result.verdict == "ask":
        # A truncated object is still a JSON object: ask, never refuse.
        assert result.kinds == sniff.TEXT_KINDS
    else:
        assert result.kinds == ()


@pytest.mark.parametrize("kind", ["character", "dictionary"])
def test_a_multibyte_character_cut_by_the_read_does_not_refuse_the_file(tmp_path, kind):
    """The 64 KiB read can end inside a UTF-8 character: that half character
    is dropped, never read as an undecodable file (a JSON card or a Markdown
    dictionary with non-ASCII text past 64 KiB). The dictionary's
    ``---@@@---`` line comes first, inside the read: only that line decides a
    chat dictionary (owner decision D8)."""
    for pad in ("", "x"):
        if kind == "character":
            card = card_v2(f"Æsir{pad}")
            card["data"]["description"] = "é" * 70_000
            name, raw = "card.json", json.dumps(card, ensure_ascii=False).encode()
        else:
            text = f"BP: blood pressure\n---@@@---\nHR{pad}: " + "é" * 70_000 + "\n"
            name, raw = "rules.md", text.encode()
        try:
            raw[: sniff.SNIFF_MAX_BYTES].decode("utf-8")
        except UnicodeDecodeError:
            break  # the read ends inside a 2-byte "é"
    else:
        pytest.fail("no padding puts the read's end inside a character")
    path = tmp_path / name
    path.write_bytes(raw)
    assert sniff.sniff_import_file(path, max_bytes=CAP).kinds == (kind,)


def test_keys_past_the_prefix_ask_rather_than_refuse(tmp_path):
    """A V1 card whose leading ``avatar`` holds 100 KB of base64: its
    discriminating keys lie past the 64 KiB read. Dev's card picker imports
    it, so refusing it would be a regression new with B2b."""
    card = {
        "avatar": "QUFB" * 25_000,
        "name": "Late Keys",
        "description": "d",
        "personality": "p",
        "first_mes": "hi",
        "mes_example": "",
    }
    path = tmp_path / "late.json"
    path.write_text(json.dumps(card), encoding="utf-8")
    result = sniff.sniff_import_file(path, max_bytes=CAP)
    assert (result.verdict, result.kinds) == ("ask", sniff.TEXT_KINDS)


def test_markdown_that_fits_two_kinds_asks():
    text = b"---\nname: Both\n---\nBP: blood pressure\n---@@@---\nHR: heart rate\n"
    result = sniff.classify(".md", text)
    assert (result.verdict, result.kinds) == ("ask", ("character", "dictionary"))
    assert sniff.classify(".md", b"# notes\n\njust prose\n").verdict == "not_roleplay"


def test_markdown_key_value_lines_alone_ask_with_the_dictionary_preselected():
    """Owner decision D8 (2026-10-29): only a ``---@@@---`` line decides a chat
    dictionary. A README or notes file has ``key: value`` lines too ("Note:
    ..."), so such a file asks "Import as:", opening on Chat dictionary: a
    real dictionary without a separator costs one Enter. The plan's first
    sniff imported it as a chat dictionary without asking, which would have
    been new with B2b for the Characters, Personas and Lore pickers; on dev
    the Dictionaries picker imports it without asking, and B2b's one picker
    asks first."""
    readme = b"# Field kit\n\nNote: pack light.\nAuthor: Tam Reyes\n"
    result = sniff.classify(".md", readme)
    assert (result.verdict, result.kinds) == ("ask", ("character", "dictionary"))
    assert result.preselect == "dictionary"
    rules = sniff.classify(".md", b"BP: blood pressure\n---@@@---\nHR: heart rate\n")
    assert (rules.verdict, rules.kinds) == ("decided", ("dictionary",))
    assert rules.preselect is None


@pytest.mark.parametrize(
    "suffix", [".yaml", ".yml", ".txt", ""], ids=["yaml", "yml", "txt", "no-suffix"]
)
def test_a_text_card_in_another_extension_still_reads_as_a_card(suffix):
    """Dev's card picker read any non-image file as text under All Files, so
    YAML frontmatter or a json block (fenced ```json or bare ```) in a
    ``.yaml`` or ``.txt`` file imported as a card; refusing it would be a
    regression new with B2b. Plain text there is still not a Roleplay file
    (no other kind's importer reads it)."""
    frontmatter = b"---\nname: Tam Reyes\ndescription: A pilot.\n---\n"
    json_block = b'# Card\n\n```json\n{"name": "Tam Reyes"}\n```\n'
    bare_block = b'# Card\n\n```\n{"name": "Tam Reyes"}\n```\n'
    for text in (frontmatter, json_block, bare_block):
        result = sniff.classify(suffix, text)
        assert (result.verdict, result.kinds) == ("decided", ("character",)), text
    assert sniff.classify(suffix, b"key: value\nmore words\n").verdict == (
        "not_roleplay"
    )


@pytest.mark.parametrize("suffix", [".md", ".markdown"], ids=["md", "markdown"])
def test_a_json_card_saved_as_markdown_reads_as_a_card(suffix):
    """Dev's card importer reads a Markdown file's raw JSON, or a json block
    fenced with or without ``json``, so a CCv2 card saved as ``.md`` imported
    as a character on dev. Its ``spec`` must win here too (spec 1.4.6): read
    as Markdown, the card's ``"key": "value"`` lines would make it a likely
    chat dictionary, which would be new with B2b. Text that only looks like
    JSON still gets the Markdown checks."""
    card = json.dumps(card_v2(), indent=2).encode()
    bare_block = b'Notes: a pilot.\n\n```\n{"name": "Tam Reyes"}\n```\n'
    for text in (card, b"\n" + card, bare_block):
        result = sniff.classify(suffix, text)
        assert (result.verdict, result.kinds) == ("decided", ("character",)), text[:24]
    rules = sniff.classify(suffix, b"{not json}\nBP: blood pressure\n---@@@---\n")
    assert (rules.verdict, rules.kinds) == ("decided", ("dictionary",))


def test_the_sniff_never_runs_the_markdown_card_parser(tmp_path, monkeypatch):
    """The card parser's json-block regex is quadratic (0.39 s at 25 KB, 5.7 s
    at 100 KB on dev, measured): it exists on dev and B2b does not fix it,
    so the sniff must never call it."""

    calls: list[tuple] = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the sniff ran the Markdown card parser")

    monkeypatch.setattr(
        Character_Chat_Lib, "load_character_card_from_string_content", refuse
    )
    path = tmp_path / "adversarial.md"
    path.write_text("```json\n{" + " " * 100_000, encoding="utf-8")
    started = time.monotonic()
    result = sniff.sniff_import_file(path, max_bytes=CAP)
    assert (result.verdict, result.kinds) == ("decided", ("character",))
    assert calls == []
    # Evidence, not a threshold (the spy above is the proof): a linear check
    # of 64 KiB never takes seconds, even under xdist and load.
    assert time.monotonic() - started < 5
