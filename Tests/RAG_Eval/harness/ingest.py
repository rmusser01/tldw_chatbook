# Tests/RAG_Eval/harness/ingest.py
"""Corpus -> real source DBs -> isolated, indexed RAG service.

`build_eval_runtime` is the harness's one impure seam. It stands up a
throwaway but *genuine* installation of the app's retrieval stack:

* the fixture corpus written through the **real writer APIs**
  (`MediaDatabase.add_media_with_keywords`, `NotesInteropService.add_note`,
  `CharactersRAGDB.add_conversation`/`add_message`,
  `PromptsDatabase.add_prompt`) into scratch SQLite files, so FTS triggers,
  id assignment and row shapes are the production ones and not a fixture's
  guess at them;
* those rows read back through the **same readers production reads them
  with**, turned into documents by the **real document builders**
  (`media_document`/`note_document`/`conversation_document`, reached via
  their `*_index_entry` wrappers) and indexed through the real batch
  indexing helper;
* an app-shaped object wired to those DBs plus the harness's own service,
  so retrieval can be measured through `LibraryLocalRagSearchService` — the
  production seam — rather than by calling the engine directly.

Nothing here is shared with the running app: the vector store persists under
the caller's `tmp_path`, the keyword leg is pointed at the scratch media,
ChaChaNotes and Prompts DBs (`config.search.media_db_path` -- P0's validated
injection point -- plus `config.search.chachanotes_db_path`, TASK-3996's, and
`config.search.prompts_db_path`, TASK-15020/B2's), and
`get_shared_rag_service()` is never called. A harness that measured the
process-wide singleton would measure whatever an earlier test left in it.

**Which indexing seam.** `index_entries` (this module's narrow helper), not
`RAGService.index_document`. Both avoid `IngestionIndexer`'s daemon-thread
queue, but `index_entries` is *exactly* what both production indexing routes
call once they have entries in hand — the ingest-time worker and
`backfill_semantic_index` differ from each other only in how they *produce*
entries, and converge here. It carries the batch embedding path
(`index_batch_optimized`), the stale-chunk delete, and the search-cache
invalidation that `index_document` does not; measuring retrieval over an
index built by a path production never uses would measure the wrong thing.
`backfill_semantic_index` itself was the other candidate and was rejected
for one reason: it consults `semantic_indexing_available()`, a *user config*
kill switch, and returns `status="unavailable"` when it is off — a
measurement harness must not be silently disarmed by a TOML setting.

**One event loop.** The runtime owns a loop and exposes `run()`. Indexing
and every later search must share it: `RAGService`'s cache and pools hold
`asyncio` primitives that bind to the first loop that touches them, so
indexing under `asyncio.run(...)` and then searching under a second
`asyncio.run(...)` raises "attached to a different loop" — an error that
looks like a harness bug but is really a loop-ownership bug.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Iterator, Sequence, TypeVar

from Tests.RAG_Eval.harness.environment import PROFILE_EMBEDDING_MODEL, PROFILE_NAME
from Tests.RAG_Eval.harness.goldenset import CorpusDoc

__all__ = [
    "CLIENT_ID",
    "EvalRuntime",
    "EvalRuntimeError",
    "NOTES_USER_ID",
    "UNINDEXED_SOURCE_TYPES",
    "build_eval_runtime",
]

T = TypeVar("T")

#: Client id stamped on every scratch row. Constant so a failure inspecting
#: the scratch DBs can tell harness rows from anything else.
CLIENT_ID = "rag-eval-harness"

#: The notes seam is per-user; `app.notes_user_id` must name the same user
#: the corpus was written as or keyword search over notes finds nothing.
NOTES_USER_ID = "rag-eval-user"

#: Media rows need a type; every fixture doc is prose.
MEDIA_TYPE = "document"

#: Matches `backfill_semantic_index`'s default, so batching behaviour (and
#: therefore embedding batching) is the production one.
INDEX_BATCH_SIZE = 16

#: Corpus source types that are written to a real database but never reach
#: the VECTOR index — retrievable through the keyword leg alone.
#:
#: This constant used to be `UNWRITABLE_SOURCE_TYPES`, and `prompt` was in
#: it for a stronger reason: before TASK-15020/B2 there was no prompts
#: writer here, no prompts sub-leg in the engine and no vector index either,
#: so a prompt fixture was invisible to every mode and the harness recorded
#: it as absent rather than pretending to have ingested it. B2 shipped the
#: writer and the sub-leg; what it deliberately did NOT ship is semantic
#: indexing of prompts (spec: out of scope, a P4-adjacent ingestion
#: question). So prompts are now written like everything else, live in
#: `slug_to_source`, and are reachable through hybrid's FTS leg — while
#: staying structurally invisible to the semantic leg.
#:
#: That surviving asymmetry is worth recording rather than inferring, which
#: is what `EvalRuntime.unindexed` is for: "in the corpus but not in the
#: vector store" and "in the vector store and not retrieved" are different
#: findings that can produce the same 0.000.
#:
#: An UNDECLARED source type still raises: a typo'd `source_type` must never
#: reach a quiet path (`goldenset.validate` rejects it first, and the raise
#: in `build_eval_runtime` is the second line of that defence).
UNINDEXED_SOURCE_TYPES: tuple[str, ...] = ("prompt",)


class EvalRuntimeError(RuntimeError):
    """Raised when the harness could not be stood up completely.

    Always raised rather than returning a partially indexed runtime: a
    harness that quietly indexes 40 of 48 documents does not error, it
    reports *plausible numbers that mean something else*.
    """


@dataclass
class EvalRuntime:
    """A live, isolated retrieval installation for one eval run.

    Attributes:
        app: App-shaped object accepted by `LibraryLocalRagSearchService`,
            wired to this runtime's scratch DBs and service.
        service: This runtime's own `EnhancedRAGServiceV2`. Task 6 flips
            `service.config.search.default_search_mode` between passes.
        slug_to_source: Fixture slug -> (source_type, source_id). Source ids
            are assigned by the real writers at write time, so the golden
            set's slugs are the only stable handle; this is the map that
            resolves them. Every corpus document is in here, prompts
            included (TASK-15020/B2) — being present here is what makes a
            document *scoreable*, not what makes it semantically indexed.
        index_summary: The accumulated `index_entries` summary
            ({'indexed', 'skipped', 'failed', 'errors'}).
        unindexed: Fixture slug -> source_type, for every corpus document
            that was written to its real database but never entered the
            VECTOR index (`UNINDEXED_SOURCE_TYPES`). Kept so a zero score
            can be attributed: "the semantic leg cannot see this type at
            all" and "the semantic leg saw it and ranked it badly" are
            different findings with the same number.
    """

    app: SimpleNamespace
    service: Any
    slug_to_source: dict[str, tuple[str, str]]
    index_summary: dict[str, Any]
    _loop: asyncio.AbstractEventLoop
    _closers: list[Callable[[], None]] = field(default_factory=list)
    _closed: bool = False
    unindexed: dict[str, str] = field(default_factory=dict)

    def run(self, awaitable: Awaitable[T]) -> T:
        """Drive an awaitable on this runtime's loop.

        Every async call against this runtime — indexing, and every search
        a caller makes afterwards — must go through here. See the module
        docstring on loop ownership.
        """
        if self._closed:
            raise EvalRuntimeError("EvalRuntime.run() called after close()")
        return self._loop.run_until_complete(awaitable)

    def close(self) -> None:
        """Release the service, the scratch DB handles, and the loop.

        Idempotent, and every closer runs even if an earlier one raises —
        a leaked SQLite handle in a `finally:` block would otherwise be
        masked by whatever failed first.
        """
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        # LIFO: the service (and its pooled SQLite handles onto the scratch
        # media DB) must go before the DB objects it was pointed at.
        for closer in reversed(self._closers):
            try:
                closer()
            except Exception as exc:  # keep closing the rest
                errors.append(f"{closer!r}: {exc}")
        try:
            self._loop.close()
        except Exception as exc:
            errors.append(f"event loop: {exc}")
        if errors:
            raise EvalRuntimeError(
                "EvalRuntime.close() had failures:\n"
                + "\n".join(f"  - {error}" for error in errors)
            )


def _batched(items: Sequence[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _write_media(media_db: Any, doc: CorpusDoc) -> tuple[str, dict[str, Any]]:
    """Write one media fixture and read its row back the way production does."""
    media_id, _uuid, message = media_db.add_media_with_keywords(
        title=doc.title,
        media_type=MEDIA_TYPE,
        content=doc.content,
    )
    if media_id is None:
        raise EvalRuntimeError(f"media write failed for {doc.slug!r}: {message}")
    row = media_db.get_media_by_id(media_id)
    if not row:
        raise EvalRuntimeError(f"media row for {doc.slug!r} vanished after write")
    return str(media_id), dict(row)


def _write_note(
    notes_service: Any, chachanotes_db: Any, doc: CorpusDoc
) -> tuple[str, dict[str, Any]]:
    """Write one note fixture through the notes seam's own writer.

    Deliberately `NotesInteropService.add_note` rather than
    `CharactersRAGDB.add_note`: the interop service is what the app writes
    notes with, and it is the same object the runtime's
    `notes_scope_service` reads them back through, so a mismatch between
    "written" and "searchable" is impossible by construction.
    """
    note_id = notes_service.add_note(NOTES_USER_ID, doc.title, doc.content)
    if not note_id:
        raise EvalRuntimeError(f"note write failed for {doc.slug!r}")
    row = chachanotes_db.get_note_by_id(note_id)
    if not row:
        raise EvalRuntimeError(f"note row for {doc.slug!r} vanished after write")
    return str(note_id), dict(row)


def _write_conversation(
    chachanotes_db: Any, doc: CorpusDoc
) -> tuple[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Write one conversation fixture as a single-turn transcript."""
    conversation_id = chachanotes_db.add_conversation({"title": doc.title})
    if not conversation_id:
        raise EvalRuntimeError(f"conversation write failed for {doc.slug!r}")
    chachanotes_db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": doc.content,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    )
    conversation = chachanotes_db.get_conversation_by_id(conversation_id)
    if not conversation:
        raise EvalRuntimeError(
            f"conversation row for {doc.slug!r} vanished after write"
        )
    messages = chachanotes_db.get_messages_for_conversation(conversation_id, limit=500)
    return str(conversation_id), (dict(conversation), [dict(m) for m in messages])


def _write_prompt(prompts_db: Any, doc: CorpusDoc) -> str:
    """Write one prompt fixture through `PromptsDatabase`'s own writer.

    `add_prompt` is what every prompt-creating surface in the app calls, and
    it is what maintains `prompts_fts` (the index the engine's prompts
    sub-leg reads). Writing the row with raw SQL would leave the FTS index
    empty and measure nothing.

    The fixture body goes into `system_prompt`: these fixtures are saved
    instructions to a model, which is what that column holds, and it is one
    of the three body columns the sub-leg renders as the row's document
    (`rag_service.PROMPT_DOCUMENT_COLUMNS`). `details` and `user_prompt` are
    left unset, exactly as a real one-field saved prompt would be.

    Unlike the other three writers there is no read-back: prompts are not
    semantically indexed (`UNINDEXED_SOURCE_TYPES`), so there is no index
    entry to build from the row, and the id `add_prompt` returns is the FTS
    rowid the sub-leg will emit as `source_id`.
    """
    prompt_id, _uuid, message = prompts_db.add_prompt(
        name=doc.title,
        author=None,
        details=None,
        system_prompt=doc.content,
    )
    if prompt_id is None:
        raise EvalRuntimeError(f"prompt write failed for {doc.slug!r}: {message}")
    return str(prompt_id)


def _build_config(
    profile_name: str,
    persist_directory: Path,
    media_db_path: Path,
    chachanotes_db_path: Path,
    prompts_db_path: Path,
):
    """Clone the profile's config and repoint it at this run's scratch state.

    A deep copy, not the profile's own object: `get_profile_manager()` hands
    out a process-wide singleton's config, and mutating it in place would
    leak this run's temp paths into every later profile consumer in the
    process.
    """
    from tldw_chatbook.RAG_Search.config_profiles import get_profile_manager
    from tldw_chatbook.RAG_Search.simplified.config import VECTOR_STORE_TYPE_CHROMA

    profile = get_profile_manager().get_profile(profile_name)
    if profile is None:
        raise EvalRuntimeError(f"RAG profile {profile_name!r} is not available")
    config = copy.deepcopy(profile.rag_config)
    # The env gate checked ONE model id was cached; loading a different one
    # would either download (blocked — the conftest forces HF offline) or,
    # worse on some future config, quietly succeed against a model nobody
    # gated on. Fail here instead, naming both sides.
    if config.embedding.model != PROFILE_EMBEDDING_MODEL:
        raise EvalRuntimeError(
            f"profile {profile_name!r} now embeds with "
            f"{config.embedding.model!r}, but the harness env gate checks "
            f"{PROFILE_EMBEDDING_MODEL!r} is cached; update "
            "harness/environment.py before changing the profile"
        )
    # Persistent Chroma, not the in-memory store: production retrieval runs
    # against Chroma, and its ANN behaviour is part of what is measured.
    config.vector_store.type = VECTOR_STORE_TYPE_CHROMA
    config.vector_store.persist_directory = persist_directory
    # The keyword/hybrid FTS leg's explicit media DB override (P0). Without
    # it the leg resolves the *real* user media DB and the harness would
    # measure retrieval over the developer's own library.
    config.search.media_db_path = media_db_path
    # Same reasoning for the notes/conversation sub-legs of that leg
    # (TASK-3996). Without this override they resolve the *real* user
    # ChaChaNotes DB: under pytest's env isolation that path does not exist,
    # so the FTS leg would silently measure media only (every note and
    # conversation unreachable -- 107 of the 172 fixture docs as the corpus
    # stands 2026-08-13, exactly the defect being fixed); outside it, the
    # harness would read the developer's own notes and conversations.
    config.search.chachanotes_db_path = chachanotes_db_path
    # And the prompts sub-leg (TASK-15020/B2). Same hazard, one step worse:
    # prompts have no vector index, so an unset override does not merely bias
    # the keyword leg — it makes the prompt category unmeasurable in every
    # mode while the run still reports numbers.
    config.search.prompts_db_path = prompts_db_path
    return config


def build_eval_runtime(
    corpus: Sequence[CorpusDoc],
    tmp_path: Path | str,
    *,
    profile_name: str = PROFILE_NAME,
) -> EvalRuntime:
    """Ingest `corpus` into a fresh, isolated, indexed retrieval installation.

    Args:
        corpus: Fixture documents, as loaded by `goldenset.load_corpus`.
        tmp_path: A per-run scratch directory (pytest's `tmp_path`). The
            scratch DBs and the vector store live under it.
        profile_name: RAG profile to build from. The default's
            `default_search_mode` is "hybrid"; callers wanting a different
            retrieval mode flip `runtime.service.config.search.
            default_search_mode` rather than rebuilding.

    Returns:
        A live `EvalRuntime`. The caller **must** `close()` it.

    Raises:
        EvalRuntimeError: A write, a read-back, or any part of indexing
            failed, or a document produced no indexable content. Never
            returns a partially built runtime.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.Prompt_Management.prompt_scope_service import (
        ServerPromptService,
        build_prompt_scope_service,
    )
    from tldw_chatbook.Media.local_media_reading_service import LocalMediaReadingService
    from tldw_chatbook.Media.media_reading_scope_service import MediaReadingScopeService
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
    from tldw_chatbook.RAG_Search.ingestion_indexing import (
        IndexEntry,
        conversation_index_entry,
        index_entries,
        media_index_entry,
        note_index_entry,
    )
    from tldw_chatbook.RAG_Search.simplified.rag_factory import create_rag_service

    if not corpus:
        raise EvalRuntimeError("refusing to build an eval runtime over an empty corpus")
    if all(doc.source_type in UNINDEXED_SOURCE_TYPES for doc in corpus):
        # An empty VECTOR index is not a runtime: semantic would score 0.000
        # on every query and read as total retrieval failure rather than as
        # "this corpus has nothing the semantic leg can hold". Since
        # TASK-15020/B2 such a corpus is no longer entirely unretrievable
        # (its prompts are reachable through the keyword leg), which makes
        # the confusion MORE likely, not less — hybrid would report real
        # numbers next to a semantic column of zeros.
        raise EvalRuntimeError(
            "refusing to build an eval runtime with no semantically indexable "
            "document: every corpus document has an unindexed source type "
            f"({', '.join(UNINDEXED_SOURCE_TYPES)})"
        )

    tmp_path = Path(tmp_path)
    persist_directory = tmp_path / "chroma"
    media_db_path = tmp_path / "eval_media.db"
    chachanotes_db_path = tmp_path / "eval_chachanotes.db"
    prompts_db_path = tmp_path / "eval_prompts.db"
    notes_user_dbs = tmp_path / "notes_user_dbs"
    notes_user_dbs.mkdir(mode=0o700, parents=True, exist_ok=True)

    closers: list[Callable[[], None]] = []
    loop = asyncio.new_event_loop()

    try:
        media_db = MediaDatabase(media_db_path, client_id=CLIENT_ID)
        closers.append(media_db.close_connection)
        chachanotes_db = CharactersRAGDB(chachanotes_db_path, client_id=CLIENT_ID)
        closers.append(chachanotes_db.close_connection)
        prompts_db = PromptsDatabase(prompts_db_path, client_id=CLIENT_ID)
        closers.append(prompts_db.close_connection)
        notes_service = NotesInteropService(
            base_db_directory=notes_user_dbs,
            api_client_id=CLIENT_ID,
            global_db_to_use=chachanotes_db,
        )
        # NotesInteropService opens its own per-user CharactersRAGDB onto the
        # same file (it does NOT reuse the template object), so those handles
        # are ours to close too — but never the template itself twice.
        closers.append(
            lambda: [
                db.close_connection()
                for db in notes_service._db_instances.values()
                if db is not chachanotes_db
            ]
        )

        slug_to_source: dict[str, tuple[str, str]] = {}
        unindexed: dict[str, str] = {}
        entries: list[IndexEntry] = []
        for doc in corpus:
            entry: IndexEntry | None
            if doc.source_type == "media":
                source_id, row = _write_media(media_db, doc)
                entry = media_index_entry(row)
            elif doc.source_type == "note":
                source_id, row = _write_note(notes_service, chachanotes_db, doc)
                entry = note_index_entry(row)
            elif doc.source_type == "conversation":
                source_id, (conversation, messages) = _write_conversation(
                    chachanotes_db, doc
                )
                entry = conversation_index_entry(conversation, messages)
            elif doc.source_type == "prompt":
                # Written for real (TASK-15020/B2) and then deliberately NOT
                # indexed: there is no `prompt_index_entry`, because nothing
                # in the app indexes prompts semantically. Recording the slug
                # keeps that a stated property of the run rather than
                # something a reader has to infer from a column of zeros.
                source_id = _write_prompt(prompts_db, doc)
                unindexed[doc.slug] = doc.source_type
                slug_to_source[doc.slug] = (doc.source_type, source_id)
                continue
            else:
                raise EvalRuntimeError(
                    f"corpus doc {doc.slug!r} has unsupported source_type "
                    f"{doc.source_type!r}"
                )
            if entry is None:
                # The builders return None for "not indexable" (no content,
                # no messages). Silently dropping such a doc would leave a
                # golden query permanently unanswerable for a reason no
                # report would show.
                raise EvalRuntimeError(
                    f"corpus doc {doc.slug!r} ({doc.source_type} {source_id}) "
                    "produced no indexable document"
                )
            slug_to_source[doc.slug] = (doc.source_type, source_id)
            entries.append(entry)

        config = _build_config(
            profile_name,
            persist_directory,
            media_db_path,
            chachanotes_db_path,
            prompts_db_path,
        )
        service = create_rag_service(profile_name, config=config)
        closers.append(service.close)

        summary: dict[str, Any] = {
            "indexed": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }
        for batch in _batched(entries, INDEX_BATCH_SIZE):
            # indexing_db=None on purpose: the incremental-skip state DB
            # lives under the user data dir and would make a rerun's
            # "indexed" count depend on a previous run's leftovers.
            batch_summary = loop.run_until_complete(
                index_entries(service, None, batch)
            )
            for key in ("indexed", "skipped", "failed"):
                summary[key] += batch_summary[key]
            summary["errors"].extend(batch_summary["errors"])

        if summary["indexed"] != len(entries) or summary["failed"]:
            raise EvalRuntimeError(
                f"indexed {summary['indexed']}/{len(entries)} documents "
                f"({summary['failed']} failed): "
                + ("; ".join(summary["errors"]) or "no error detail reported")
            )

        app = SimpleNamespace(
            media_reading_scope_service=MediaReadingScopeService(
                LocalMediaReadingService(media_db),
                None,
            ),
            chachanotes_db=chachanotes_db,
            notes_scope_service=NotesScopeService(
                local_notes_service=notes_service,
                server_service=None,
            ),
            notes_user_id=NOTES_USER_ID,
            # The prompts seam, wired (TASK-18255). It used to be `None`
            # "deliberately", and that omission cost a wrong finding: the
            # plain `category.prompt.*` cells read 0.000 because
            # `_search_prompts` returns `(False, [])` — the seam reporting
            # itself UNAVAILABLE — and TASK-17855 read that as a production
            # retrieval defect. The metrics table renders "not measured" and
            # "measured, found nothing" identically, so an unwired seam is
            # indistinguishable from a broken one.
            #
            # `server_service` is passed EXPLICITLY as a client-less service
            # rather than left to `app_config`: the default path runs
            # `derive_configured_server_binding`, and a harness that consults
            # ambient config could bind to the developer's own server — the
            # same hazard the `*_db_path` overrides above exist to prevent.
            # Local-only, no network, by construction.
            prompt_scope_service=build_prompt_scope_service(
                prompt_db=prompts_db,
                server_service=ServerPromptService(client=None),
            ),
            # An UNSTAMPED `_rag_service` wins outright in the seam's
            # resolver (`semantic_availability.current_app_rag_service`'s
            # direct-injection carve-out), so the seam retrieves through
            # THIS service and never builds the process-wide one.
            _rag_service=service,
        )
    except BaseException:
        # Partial build: release whatever was already opened (LIFO) and let
        # the original failure propagate. Suppressing a closer's own error
        # here is deliberate — it must never mask the real cause.
        for closer in reversed(closers):
            try:
                closer()
            except Exception:
                pass
        loop.close()
        raise

    return EvalRuntime(
        app=app,
        service=service,
        slug_to_source=slug_to_source,
        index_summary=summary,
        _loop=loop,
        _closers=closers,
        unindexed=unindexed,
    )
