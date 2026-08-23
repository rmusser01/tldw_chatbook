"""The student story, end to end (chunking-agent-tools spec §7.6, upgraded
by student-workflow spec §7.6 to close the write loop).

The program's motivating payoff: a student ingests a book and wants
per-chapter notes. The ``library_*`` chunk tools deliver the read from the
item's OWN stored chunks — structure map -> chapter node -> chunk address
span -> unit fetches -> note text — with no window-walking and no
re-chunking behind anyone's back; ``library_save_note`` then lands the
note (provenance-headered, folder-grouped), a re-run leg proves
search-first + update-not-duplicate, and a flashcard leg saves the Q/A
markdown convention (student-workflow spec §5/§6).

The ingestion is REAL end to end at the chunking seam (the established
convention from ``Tests/Local_Ingestion/test_book_ingestion_chunking.py``
and ``test_ingest_template_resolution.py``):
``parse_local_file_for_ingest`` -> ``persist_parsed_media`` ->
``add_media_with_keywords(chunks=...)`` -> ``UnvectorizedMediaChunks`` rows.
Only the optional-dependency EXTRACTION seam is stubbed (ebooklib is absent
in this venv, so ``read_epub_filtered`` returns a chaptered fixture); the
chunker, the spans, the stamps, and the DB writes all run for real.

Two behaviors worth knowing before reading the assertions:

* **The mutation pin** patches ``improved_chunking_process`` in BOTH
  ``RAG_Search.chunking_service`` and ``Library.library_rechunk_service``
  (the latter binds the name via ``from ... import`` at module scope, so
  patching the source module alone would miss it) — the story's fetches
  must be served by stored rows only.
* **The chapter-marker bleed.** The vendored engine's ``ebook_chapters``
  strategy ends each chapter at the NEXT marker's start, and the marker
  regex matches the title text (after the ``#``), so every chapter chunk's
  text carries the next heading's ``#`` as its last character. A node span
  begins AT the ``#``, so the previous chapter's unit overlaps the node by
  exactly that one character and joins the node's ``chunk_span``. This is
  pinned vendored behavior (engine-parity sub-project), not a tool bug:
  the span honestly reports every overlapping unit, and the chapter's own
  unit is still exactly recoverable (the note derivation below).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_tool_contract import (
    ERROR_FEATURE_UNAVAILABLE,
    make_public_id,
)
from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService
from tldw_chatbook.Library.local_media_chunk_tool_service import (
    LocalMediaChunkToolService,
)
from tldw_chatbook.Local_Ingestion import Book_Ingestion_Lib
from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    parse_local_file_for_ingest,
    persist_parsed_media,
)
from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType

# ---------------------------------------------------------------------------
# Fixture books: chapters with recognizable headings
# ---------------------------------------------------------------------------

#: The book the student ingested: nine short chapters, each three
#: paragraphs, markdown headings the navigation tree recognizes.
STUDENT_READER = "\n\n".join(
    f"# Chapter {i}\n\n" + "\n\n".join(
        f"Paragraph {j} of chapter {i} carries sentences worth noting for the exam."
        for j in range(3)
    )
    for i in range(1, 10)
)

#: A second, differently-worded book for the degradation leg. Different
#: body text on purpose: ``add_media_with_keywords`` dedups on content
#: hash, so an identically-worded "chunking off" ingest would silently
#: UPDATE the first item instead of creating the unchunked one.
UNREAD_READER = "\n\n".join(
    f"# Chapter {i}\n\n" + "\n\n".join(
        f"Paragraph {j} of chapter {i} holds lines the student has not read yet."
        for j in range(3)
    )
    for i in range(1, 4)
)

#: The ebook family's own chapter method (the Import panel's default
#: choice for e-books), with an explicit zero overlap.
CHAPTER_CHUNK_OPTIONS = {"method": "ebook_chapters", "max_size": 1500, "overlap": 0}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def media_db(tmp_path: Path) -> MediaDatabase:
    return MediaDatabase(tmp_path / "media.db", client_id="student-story")


@pytest.fixture(autouse=True)
def _deterministic_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reads the developer machine's config."""
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda section, key=None, default=None: (
            default if default is not None else None
        ),
    )


_STUBBED_BOOKS: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _stub_epub_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ONLY the ebooklib-backed extraction seams (house convention).

    The chunking seam below them stays fully real. Each stubbed book is
    registered by file name, so one fixture serves both books.
    """
    monkeypatch.setattr(
        Book_Ingestion_Lib, "EBOOK_PROCESSING_AVAILABLE", True
    )

    def _read_epub_filtered(path: str) -> tuple[str, Any]:
        name = Path(path).name
        try:
            return _STUBBED_BOOKS[name], SimpleNamespace(metadata={})
        except KeyError as exc:  # pragma: no cover - fixture wiring only
            raise ValueError(f"no stubbed extraction for {name}") from exc

    monkeypatch.setattr(Book_Ingestion_Lib, "read_epub_filtered", _read_epub_filtered)
    monkeypatch.setattr(
        Book_Ingestion_Lib,
        "extract_epub_metadata_from_epub_obj",
        lambda ebook_obj: ("The Student Reader", "A. Author"),
    )


def _ingest_ebook(
    db: MediaDatabase, tmp_path: Path, name: str, book: str, chunk_options: dict | None
) -> tuple[int, str]:
    """Ingest one fixture ebook for real: parse -> persist -> stored rows.

    ``chunk_options=None`` is the Library queue's "Chunk content OFF"
    state (task-3301) and stores no chunk rows.
    """
    _STUBBED_BOOKS[name] = book
    source = tmp_path / name
    source.write_bytes(b"PK\x03\x04 fake epub bytes; extraction is stubbed only")
    payload = parse_local_file_for_ingest(
        str(source), {"chunk_options": chunk_options, "perform_analysis": False}
    )
    media_id, media_uuid, _ = persist_parsed_media(
        payload, db, overwrite_existing=True, generate_embeddings=False
    )
    assert media_id is not None and media_uuid is not None
    return int(media_id), str(media_uuid)


def _service(db: MediaDatabase) -> LocalMediaChunkToolService:
    return LocalMediaChunkToolService(
        db, LocalMediaReadingService(db), template_interop=None
    )


def _agent_surface(
    db: MediaDatabase, tmp_path: Path
) -> tuple[LocalLibraryToolService, NotesScopeService]:
    """One tool surface for the WHOLE story: the shared dispatcher the
    Console provider and local MCP both call, with the media chunk tools
    and the real note row/folder seams wired (student-workflow Task 1's
    ``library_save_note`` landing).

    Rows go through a real ``NotesInteropService`` over a real ChaChaNotes
    DB; folders through a real ``NotesScopeService`` — the same construction
    Task 1's real-DB save test pins. The scope handle comes back too, so the
    story can assert what the notes screen would list.
    """
    notes_db = CharactersRAGDB(tmp_path / "notes.db", "student-story-notes")
    notes = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="student-story",
        global_db_to_use=notes_db,
    )
    scope = NotesScopeService(
        local_notes_service=notes,
        server_service=None,
        folder_repository=LocalNoteFolderRepository(notes_db),
    )
    service = LocalLibraryToolService(
        media_chunk_service=_service(db),
        notes_service=notes,
        notes_scope_service=scope,
        notes_user_id="student",
    )
    return service, scope


def _public(media_uuid: str) -> str:
    return make_public_id("media", media_uuid)


def _pin_no_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reuse-stored-chunks pin: any chunking call fails the test.

    BOTH bindings are patched (Task 3's review minor):
    ``library_rechunk_service`` binds ``improved_chunking_process`` into
    its own module via ``from ... import``, so patching the source module
    alone would not catch a call routed through the re-chunk machinery.
    """
    import tldw_chatbook.Library.library_rechunk_service as rechunk_service
    from tldw_chatbook.RAG_Search import chunking_service

    def _fail(*args: Any, **kwargs: Any):  # pragma: no cover - pin only
        raise AssertionError("chunking ran during a stored-chunk story fetch")

    monkeypatch.setattr(chunking_service, "improved_chunking_process", _fail)
    monkeypatch.setattr(rechunk_service, "improved_chunking_process", _fail)


def _note_text(text: str) -> str:
    """Normalize text for note comparison.

    Whitespace collapses, and markdown heading markers (``#``) drop from
    BOTH sides: the navigation heading line and the engine's chapter
    chunks render the same title with and without its ``#``/``##`` marker
    (the chunk strategy matches the title text), and — the vendored
    engine's chapter-marker quirk — each chapter chunk's text ends with
    the NEXT heading's stray ``#``. Markers are formatting, not content;
    the note comparison is over the chapter's words.
    """
    return " ".join(text.replace("#", " ").split())


# ---------------------------------------------------------------------------
# The story (spec §7.6): notes for Chapter 7, from stored chunks only
# ---------------------------------------------------------------------------


def test_student_story_chapter_notes_from_stored_chunks_only(
    media_db: MediaDatabase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    media_id, media_uuid = _ingest_ebook(
        media_db, tmp_path, "student-reader.epub", STUDENT_READER, CHAPTER_CHUNK_OPTIONS
    )
    content = str(media_db.get_media_by_id(media_id)["content"])
    # One surface for the whole story — the shared dispatcher a Console
    # agent actually calls; the chunk reads below route through it to the
    # same chunk service, unchanged.
    service, scope = _agent_surface(media_db, tmp_path)
    public = _public(media_uuid)

    # Ground truth first: the stored rows are exact, engine-stamped slices
    # of the ingested source (one chapter unit per chapter).
    rows = media_db.get_connection().execute(
        "SELECT chunk_index, chunk_text, start_char, end_char, "
        "chunk_engine_version FROM UnvectorizedMediaChunks "
        "WHERE media_id = ? AND deleted = 0 ORDER BY chunk_index",
        (media_id,),
    ).fetchall()
    assert len(rows) == 9
    for row in rows:
        assert content[row["start_char"] : row["end_char"]] == row["chunk_text"]
        assert row["chunk_engine_version"]  # engine-stamped, not legacy

    # 1. "Where are the chapters?" — the structure map, no window-walking.
    structure = service.invoke("library_get_media_structure", {"id": public})
    assert "error" not in structure
    assert [node["title"] for node in structure["nodes"]] == [
        f"Chapter {i}" for i in range(1, 10)
    ]
    assert structure["chunk_summary"]["available"] is True
    assert structure["chunk_summary"]["chunk_count"] == 9

    # 2. Find Chapter 7's node and its chunk address span.
    node = next(n for n in structure["nodes"] if n["title"] == "Chapter 7")
    span = node.get("chunk_span")
    assert isinstance(span, list) and len(span) == 2

    # 3. Fetch every unit the span addresses — from stored rows only.
    _pin_no_chunking(monkeypatch)
    fetched: dict[int, dict] = {}
    for index in range(span[0], span[1] + 1):
        payload = service.invoke(
            "library_get_media_chunk", {"id": public, "chunk_index": index}
        )
        assert "error" not in payload, payload
        chunk = payload["chunk"]
        assert chunk["chunk_index"] == index
        # Verbatim stored row: the text IS the source slice at its span.
        assert content[chunk["start_char"] : chunk["end_char"]] == chunk["text"]
        assert payload["revision"] == structure["revision"]
        fetched[index] = chunk

    # The span's last unit is Chapter 7's own; the member before it is
    # Chapter 6's unit, pulled in by the vendored engine's one-character
    # chapter-marker overrun (see the module docstring) — the span reports
    # every overlapping unit and nothing else.
    assert _note_text(fetched[span[1]]["text"]).startswith("Chapter 7")
    if span[0] != span[1]:
        assert _note_text(fetched[span[0]]["text"]).startswith("Chapter 6")

    # 4. The payoff: the note derived from the fetched chunks is exactly
    #    the source chapter text. The student's agent picks the chapter's
    #    unit the way a reader would — the one whose text opens with the
    #    chapter's recognizable heading.
    owning = next(
        chunk
        for chunk in fetched.values()
        if chunk["text"].strip().startswith("Chapter 7")
    )
    chapter_source = content[node["span"][0] : node["span"][1]]
    assert _note_text(owning["text"]) == _note_text(chapter_source)
    # Not a coincidence of normalization: every sentence of the chapter is
    # present, word for word.
    for paragraph_index in range(3):
        sentence = (
            f"Paragraph {paragraph_index} of chapter 7 carries sentences"
            " worth noting for the exam."
        )
        assert _note_text(sentence) in _note_text(owning["text"])

    # 5. The write loop closes (student-workflow spec §7.6): the agent
    #    lands the Chapter-7 note WITH the provenance header — source,
    #    revision (the media revision from the structure payload), chapter,
    #    chunks — grouped in one per-book folder.
    note_title = "The Student Reader — Chapter 7 notes"
    provenance_header = (
        f"source: {public}\n"
        f"revision: {structure['revision']}\n"
        f"chapter: {node['title']}\n"
        f"chunks: {span[0]}-{span[1]}"
    )
    note_content = (
        f"{provenance_header}\n\n"
        "Key points:\n"
        "- Three paragraphs, each carrying sentences worth noting for the"
        " exam.\n"
        f"- Verbatim source: {_note_text(owning['text'])}\n"
    )
    saved = service.invoke(
        "library_save_note",
        {"title": note_title, "content": note_content, "folder": "Student Reader"},
    )
    assert "error" not in saved, saved
    assert saved["created"] is True
    assert saved["version"] == 1
    assert saved["item"]["title"] == note_title
    assert saved["item"]["folder"] == "Student Reader"

    # 6. The re-read: library_get_note serves the note back whole, the
    #    payload's revision is the note's own version (what an update
    #    needs), and the header round-trips verbatim — media revision
    #    included, so staleness stays detectable.
    reread = service.invoke("library_get_note", {"id": saved["item"]["id"]})
    assert "error" not in reread, reread
    assert reread["content"]["revision"] == str(saved["version"])
    assert reread["content"]["has_more"] is False
    assert reread["content"]["text"].startswith(provenance_header)

    # 7. The re-run leg: a later session rediscovers the note by title —
    #    SEARCH-based, because the list tool has no folder filter and its
    #    payloads carry no folder info — and UPDATES it in place instead of
    #    minting a duplicate (notes have no unique title; the convention is
    #    the agent's explicit choice).
    found = service.invoke("library_search_notes", {"query": note_title})
    assert "error" not in found, found
    assert found["total"] == 1
    assert found["items"][0]["id"] == saved["item"]["id"]
    assert found["items"][0]["title"] == note_title
    assert "title" in found["items"][0]["matched_fields"]

    rerun_content = note_content + "\nRe-run addition: reviewed again.\n"
    updated = service.invoke(
        "library_save_note",
        {
            "title": note_title,
            "content": rerun_content,
            "folder": "Student Reader",
            "note_id": found["items"][0]["id"],
            "expected_version": int(reread["content"]["revision"]),
        },
    )
    assert "error" not in updated, updated
    assert updated["created"] is False
    assert updated["version"] == 2

    # Still exactly ONE note — the re-run updated, never duplicated.
    still_one = service.invoke("library_search_notes", {"query": note_title})
    assert still_one["total"] == 1
    updated_read = service.invoke("library_get_note", {"id": updated["item"]["id"]})
    assert updated_read["content"]["revision"] == "2"
    assert updated_read["content"]["text"].endswith("reviewed again.\n")

    # 8. The flashcard convention (student-workflow spec §6): flashcards
    #    are Q/A markdown inside notes — visible the moment they land.
    flashcard_title = "The Student Reader — Chapter 7 flashcards"
    flashcard_content = (
        f"{provenance_header}\n\n"
        "Q: How many paragraphs does Chapter 7 carry?\n"
        "A: Three, each worth noting for the exam.\n"
        "Q: What does every paragraph of Chapter 7 carry?\n"
        "A: Sentences worth noting for the exam.\n"
    )
    flashcards = service.invoke(
        "library_save_note",
        {
            "title": flashcard_title,
            "content": flashcard_content,
            "folder": "Student Reader",
        },
    )
    assert "error" not in flashcards, flashcards
    assert flashcards["created"] is True
    flashcard_read = service.invoke(
        "library_get_note", {"id": flashcards["item"]["id"]}
    )
    assert "error" not in flashcard_read, flashcard_read
    assert flashcard_read["content"]["text"].startswith(provenance_header)
    assert flashcard_read["content"]["text"].count("Q:") == 2
    assert flashcard_read["content"]["text"].count("A:") == 2

    # Both saves converged on ONE per-book folder — what the notes screen
    # lists, because the folder seam is pinned to the notes UI's scope.
    children = asyncio.run(
        scope.list_note_folder_children(
            scope=ScopeType.LOCAL_NOTE,
            parent_id=None,
            limit=50,
            offset=0,
            user_id="student",
        )
    )
    assert [folder.name for folder in children.folders] == ["Student Reader"]


# ---------------------------------------------------------------------------
# The degradation leg (spec §8.13): chunking off at ingest, then repaired
# ---------------------------------------------------------------------------


def test_degradation_unchunked_item_hints_rechunk_then_fetches_work(
    media_db: MediaDatabase, tmp_path: Path
):
    media_id, media_uuid = _ingest_ebook(
        media_db, tmp_path, "unread-reader.epub", UNREAD_READER, None
    )
    service = _service(media_db)
    public = _public(media_uuid)

    # Ingest with chunking OFF stored no chunk rows.
    stored = media_db.get_connection().execute(
        "SELECT COUNT(*) AS n FROM UnvectorizedMediaChunks "
        "WHERE media_id = ? AND deleted = 0",
        (media_id,),
    ).fetchone()["n"]
    assert stored == 0

    # The structure tool keeps the story alive: heading tree, available
    # false, and the re-chunk hint note.
    structure = service.invoke("library_get_media_structure", {"id": public})
    assert "error" not in structure
    assert [node["title"] for node in structure["nodes"]] == [
        "Chapter 1",
        "Chapter 2",
        "Chapter 3",
    ]
    assert all("chunk_span" not in node for node in structure["nodes"])
    assert structure["chunk_summary"]["available"] is False
    assert any(
        "library_rechunk_media" in note for note in structure["notes"]
    ), structure["notes"]

    # A unit fetch is the named degradation error naming the way out.
    refused = service.invoke(
        "library_get_media_chunk", {"id": public, "chunk_index": 0}
    )
    assert refused["error"]["code"] == ERROR_FEATURE_UNAVAILABLE
    assert "library_rechunk_media" in refused["error"]["message"]

    # The agent opts into the write: one re-chunk with a flat spec.
    rechunked = service.invoke(
        "library_rechunk_media", {"id": public, "spec": CHAPTER_CHUNK_OPTIONS}
    )
    assert "error" not in rechunked, rechunked
    assert rechunked["status"] == "rechunked"
    assert rechunked["chunk_summary"]["chunk_count"] == 3
    assert rechunked["chunk_summary"]["spans_present"] is True
    # Default-off reindex posture: the disclosure note, never a reindexed
    # key the run did not earn.
    assert "reindexed" not in rechunked
    assert any("reindex" in note for note in rechunked["notes"])

    # The fetch path works now, and the structure map says so.
    structure_after = service.invoke(
        "library_get_media_structure", {"id": public}
    )
    assert structure_after["chunk_summary"]["available"] is True
    fetched = service.invoke(
        "library_get_media_chunk", {"id": public, "chunk_index": 0}
    )
    assert "error" not in fetched
    assert fetched["chunk"]["text"].strip().startswith("Chapter 1")
