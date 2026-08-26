"""TASK-16588 Task 2 -- the dual-index semantic/hybrid route probe.

Measures `expand_document` on the routes TASK-16174's oracle run structurally
could not reach (`semantic` and `hybrid`), against BOTH index kinds the spec
pre-registered:

* **canonical** -- notes/media/conversations indexed through the app's own
  `note_document` / `media_document` / `conversation_document` builders +
  `index_entries`, so every chunk's metadata carries `source_id`/`source_type`
  and `_semantic_row` resolves a real database id.
* **non-canonical** -- the same notes indexed through a HAND-BUILT `IndexEntry`
  whose metadata is `{"type", "note_id", "title"}` (the shape TASK-15810's
  committed QA seeder actually writes,
  `Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/seed_profile.py:64-72`).
  With no `source_id`/`document_id` in the metadata, `_semantic_row` falls
  through to the vector store's POINT id -- the case AC#2 exists for.

For every route the probe drives the PRODUCTION surface
(`LibraryRagToolProvider.invoke`, `mode="rag"`, the sealed payload) and, as a
control that shows what metadata actually arrived, the direct engine
(`rag_service.search(search_type=...)`). Every row the payload declares
expandable is then expanded by a DIRECT `ExpandDocumentTool().execute(...)`
call in three arms over the SAME row:

* `pre`  -- the payload as it was BEFORE TASK-16588 (`note_id`/`doc_id`
            stripped by this probe, no checkout dance, no re-commit),
* `post` -- the payload as shipped at this HEAD,
* `head` -- `post` minus `chunk_start`, the control that shows a document-head
            window CANNOT contain a marker planted past the 8000-char budget.

Isolation follows the live-run rules (`backlog/docs/lessons-live-verification.md`):
a scratch HOME/XDG/`TLDW_CONFIG_PATH` per index kind, set BEFORE any
`tldw_chatbook` import (importing `config.py` resolves the real profile's
paths at import time), the embedding model copied into the scratch
profile-local cache with HF downloads made impossible, and the real config's
sha256 recorded before and after. The probe never boots the TUI.

Usage
-----
    route_probe.py all <scratch-root> <out-dir>     # driver: both kinds + merge
    route_probe.py one <scratch-root> <kind> <out>  # one isolated worker

`all` re-execs THIS file once per index kind so each kind gets its own
process, its own profile and its own chroma collection -- the shared RAG
service is a process-wide singleton, so two indexes cannot honestly coexist
in one interpreter.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else ""

# ---------------------------------------------------------------------------
# Scratch environment. This block MUST run before any tldw_chatbook import --
# `tldw_chatbook.config` resolves the data/config directories at import time,
# so hoisting the project imports above it would bind the REAL profile and
# silently invalidate every isolation claim in the report (the `data_dir`
# assert in `_assert_isolated` is what catches that mistake).
# ---------------------------------------------------------------------------
def _validated_scratch_path(raw: str) -> pathlib.Path:
    """Resolve an operator-supplied path and refuse unsafe targets.

    Qodo PR-1729 finding 1: CLI paths flow into ``mkdir``/``write_text``.
    This probe may only ever write OUTSIDE the repository tree (a scratch
    dir or an artifact path the operator owns), so the one containment
    check that matters here is "never inside the repo checkout" -- a
    traversal that lands in the tree could clobber tracked files.

    Args:
        raw: The path string exactly as passed on the command line.

    Returns:
        The fully resolved path.

    Raises:
        SystemExit: If the resolved path falls inside this repository.
    """
    resolved = pathlib.Path(raw).resolve()
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if resolved == repo_root or repo_root in resolved.parents:
        raise SystemExit(
            f"refusing repo-tree target {resolved} (repo: {repo_root})"
        )
    return resolved


if MODE == "one":
    SCRATCH = _validated_scratch_path(sys.argv[2])
    INDEX_KIND = sys.argv[3]
    if INDEX_KIND not in ("canonical", "noncanonical"):
        raise SystemExit(f"unknown index kind {INDEX_KIND!r}")
    OUT_JSON = _validated_scratch_path(sys.argv[4])
    PROFILE_ROOT = SCRATCH / INDEX_KIND
    USER_NAME = f"probe16588_{INDEX_KIND}"
    # Captured BEFORE HOME is overwritten: `expanduser("~")` reads
    # os.environ["HOME"], so computing it afterwards would point at the
    # scratch home and change which model cache is reachable.
    _REAL_HOME = pathlib.Path(os.path.expanduser("~"))
    _MODEL_CACHE_SRC = pathlib.Path(
        os.environ.get(
            "MODEL_CACHE",
            str(_REAL_HOME / ".local/share/tldw_cli/default_user/models/embeddings"),
        )
    )

    _config_dir = PROFILE_ROOT / "home/.config/tldw_cli"
    _data_dir = PROFILE_ROOT / "data"
    _config_dir.mkdir(parents=True, exist_ok=True)
    (_data_dir / USER_NAME / "models").mkdir(parents=True, exist_ok=True)
    _config_path = _config_dir / "config.toml"
    _config_path.write_text(
        "\n".join(
            [
                "[general]",
                f'users_name = "{USER_NAME}"',
                'default_tab = "chat"',
                "",
                "[paths]",
                f'data_dir = "{_data_dir}"',
                "",
                "[first_run]",
                "setup_started = true",
                "setup_completed = true",
                "",
                "[splash_screen]",
                "enabled = false",
                "",
                "[embeddings]",
                'default_model_id = "all-MiniLM-L6-v2"',
                "",
                "[rag]",
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _model_cache_dst = _data_dir / USER_NAME / "models" / "embeddings"
    if not _model_cache_dst.exists():
        if not _MODEL_CACHE_SRC.exists():
            raise SystemExit(
                f"FATAL: embedding model cache not found at {_MODEL_CACHE_SRC}; "
                "the app's cache is PROFILE-LOCAL, so a scratch profile starts "
                "empty and cannot load the model under HF_HUB_OFFLINE=1."
            )
        shutil.copytree(_MODEL_CACHE_SRC, _model_cache_dst)

    os.environ["HOME"] = str(PROFILE_ROOT / "home")
    os.environ["XDG_CONFIG_HOME"] = str(PROFILE_ROOT / "home/.config")
    os.environ["XDG_DATA_HOME"] = str(PROFILE_ROOT / "home/.local/share")
    os.environ["XDG_CACHE_HOME"] = str(PROFILE_ROOT / "home/.cache")
    os.environ["TLDW_CONFIG_PATH"] = str(_config_path)
    # The model must already be on disk and a download must be impossible.
    os.environ.setdefault("HF_HOME", str(_REAL_HOME / ".cache/huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

#: Generic maintenance vocabulary used for FILLER only. Deliberately shares no
#: token with any marker phrase, so a marker-targeted query cannot be answered
#: by filler and the matched chunk is the marker's own.
_FILLER_WORDS = (
    "inspection interval routine schedule technician logged replacement gasket "
    "fastener torque wrench grease fitting bearing housing pump seal valve "
    "flange bolt washer shim plate bracket panel cover guard rail ladder "
    "platform walkway handrail signage label tag record sheet binder cabinet "
    "storeroom spare consumable filter element cartridge hose clamp fitting "
    "coupling adapter reducer elbow tee union nipple bushing sleeve gland "
    "packing lantern ring stuffing box shaft sleeve wear plate liner"
).split()

#: Marker sentences are planted PAST this offset so a document-head window
#: (`expand_document`'s `DEFAULT_MAX_CHARS`, 8000) structurally cannot contain
#: one. The corpus builder asserts the achieved offset rather than assuming it.
MIN_MARKER_OFFSET = 8000

LONG_DOCS = (
    {
        "slug": "marrowvane-coupler",
        "seam": "note",
        "title": "Marrowvane coupler service log",
        "marker": (
            "The marrowvane coupler seized after eleven hundred cycles of "
            "reversed torque."
        ),
        "context": (
            "Marrowvane coupler teardown notes. The marrowvane splines were "
            "galled across their full engagement length and the reversed "
            "torque counter had rolled past its reset. Reversed torque cycling "
            "is the only duty this coupler sees on the marrowvane line, so the "
            "cycle count is the whole diagnosis. Eleven hundred cycles is well "
            "short of the marrowvane coupler's rated endurance, and the "
            "supplier was asked to review the reversed torque derating curve "
            "before the next marrowvane coupler is fitted."
        ),
        "query": "marrowvane coupler seized reversed torque cycles",
    },
    {
        "slug": "sable-flume-weir",
        "seam": "note",
        "title": "Sable flume weir freshet report",
        "marker": (
            "The sable flume weir overtopped at nine hundred litres per minute "
            "during the freshet."
        ),
        "context": (
            "Sable flume weir hydraulics. The freshet arrived early and the "
            "sable flume crest was submerged before the overtopping alarm "
            "latched. Nine hundred litres per minute is the highest flow the "
            "sable flume weir has passed since it was rebuilt, and the freshet "
            "gauge board was read by two observers to confirm it. Overtopping "
            "the sable flume weir floods the lower freshet channel, so the "
            "crest height is being reconsidered."
        ),
        "query": "sable flume weir overtopped freshet litres per minute",
    },
    {
        "slug": "tindalos-encoder",
        "seam": "note",
        "title": "Tindalos encoder fault history",
        "marker": (
            "The tindalos encoder logged a quadrature fault every fourth index "
            "pulse."
        ),
        "context": (
            "Tindalos encoder diagnostics. Quadrature faults on the tindalos "
            "encoder were captured against the index pulse train, and the "
            "pattern repeated on every fourth index pulse without exception. "
            "A quadrature fault that tracks the index pulse points at the "
            "tindalos encoder disc rather than its cabling, and swapping the "
            "cable did not move the quadrature fault at all."
        ),
        "query": "tindalos encoder quadrature fault index pulse",
    },
    {
        "slug": "hollowmere-ballast",
        "seam": "note",
        "title": "Hollowmere ballast tank incident",
        "marker": (
            "The hollowmere ballast tank vented brine through a cracked "
            "standpipe collar."
        ),
        "context": (
            "Hollowmere ballast tank incident record. Brine was found on the "
            "hollowmere deck plates and traced to the standpipe collar, which "
            "had cracked circumferentially. The hollowmere ballast tank was "
            "isolated and the brine line blanked before the standpipe collar "
            "was cut out. A cracked standpipe collar on a ballast tank vents "
            "brine under head, so the hollowmere isolation sequence was "
            "rewritten."
        ),
        "query": "hollowmere ballast tank vented brine standpipe collar",
    },
    {
        "slug": "catafract-lathe",
        "seam": "media",
        "title": "Catafract lathe commissioning transcript",
        "marker": (
            "The catafract lathe headstock showed spiral scoring after the dry "
            "run."
        ),
        "context": (
            "Catafract lathe commissioning. The dry run was cut short when the "
            "catafract headstock bore was found spiral scored along its whole "
            "length. Spiral scoring after a dry run means the catafract lathe "
            "headstock lost its oil film, and the dry run procedure was "
            "amended to prime the catafract headstock first. No spiral scoring "
            "was present before the dry run."
        ),
        "query": "catafract lathe headstock spiral scoring dry run",
    },
    {
        "slug": "vermillion-kiln",
        "seam": "media",
        "title": "Vermillion kiln refractory survey",
        "marker": (
            "The vermillion kiln refractory spalled along the seventh course "
            "of firebrick."
        ),
        "context": (
            "Vermillion kiln refractory survey. Spalling was mapped course by "
            "course and the seventh course of firebrick had lost face on both "
            "sides of the vermillion arch. Refractory spalling concentrated in "
            "one course of firebrick points at a thermal gradient, so the "
            "vermillion kiln firing curve was pulled for review. The seventh "
            "course was the only course of firebrick that had spalled."
        ),
        "query": "vermillion kiln refractory spalled firebrick course",
    },
    {
        "slug": "starling-gantry",
        "seam": "conversation",
        "title": "Starling gantry descent test debrief",
        "marker": (
            "The starling gantry brake pads glazed over after the descent "
            "test."
        ),
        "context": (
            "Starling gantry descent test debrief. The brake pads came off the "
            "starling gantry with a mirror glaze across their whole friction "
            "face. Glazed brake pads after a descent test mean the starling "
            "gantry brakes were dragged rather than cycled, and the descent "
            "test profile was rewritten to cycle them. New brake pads were "
            "fitted to the starling gantry and bedded in before the next "
            "descent test."
        ),
        "query": "starling gantry brake pads glazed descent test",
    },
)

#: Queries that carry no marker: they exist so the probe's row population is
#: not made entirely of marker hits, and so label-only keyword rows (which only
#: the hybrid route's FTS leg can produce) get a chance to appear.
GENERAL_QUERIES = (
    {
        "slug": "general-lubrication",
        "query": "routine preventive maintenance scheduling and lubrication intervals",
    },
    {
        "slug": "general-interlock",
        "query": "safety interlock inspection checklist",
    },
    {
        "slug": "general-handover",
        "query": "shift handover notes and outstanding defects",
    },
)

SHORT_NOTE_TITLES = (
    "Weekly lubrication round",
    "Gasket stock reconciliation",
    "Guard rail inspection sheet",
    "Filter cartridge changeout",
    "Storeroom spares audit",
    "Walkway grating repair",
    "Hose clamp survey",
    "Shift handover summary",
)
SHORT_MEDIA_TITLES = (
    "Safety interlock inspection checklist",
    "Bearing housing rebuild procedure",
)
SHORT_CONVERSATION_TITLES = (
    "Outstanding defects review",
    "Spare parts expediting",
    "Night shift handover",
)


def _filler(seed: int, target_chars: int) -> str:
    """Deterministic topical filler of at least ``target_chars`` characters."""
    rng = random.Random(seed)
    sentences: list[str] = []
    total = 0
    while total < target_chars:
        words = [rng.choice(_FILLER_WORDS) for _ in range(rng.randint(12, 20))]
        sentence = " ".join(words).capitalize() + "."
        sentences.append(sentence)
        total += len(sentence) + 1
    return " ".join(sentences)


def build_long_document(doc: dict) -> str:
    """Prefix filler, then the marker paragraph, then suffix filler.

    Asserts the marker's achieved character offset is past
    ``MIN_MARKER_OFFSET`` -- if the corpus ever stops satisfying its own
    design, the window-contains-marker check silently becomes unfailable, so
    this is a hard assert rather than a comment.

    Args:
        doc: A long-doc spec mapping with ``slug``, ``title`` and
            ``marker`` keys (see ``LONG_DOCS``).

    Returns:
        The full document text with the marker paragraph planted past the
        minimum offset.
    """
    # A STABLE digest, never `hash()` -- Python randomizes string hashing per
    # process, so the two workers (one per index kind) would otherwise build
    # different bodies for the same slug and the two arms would not be
    # comparable.
    seed = int(hashlib.sha256(doc["slug"].encode("utf-8")).hexdigest()[:8], 16)
    prefix = _filler(seed, MIN_MARKER_OFFSET + 1200)
    suffix = _filler(seed + 1, 2500)
    body = f"{doc['title']}\n\n{prefix}\n\n{doc['context']} {doc['marker']}\n\n{suffix}"
    offset = body.find(doc["marker"])
    if offset <= MIN_MARKER_OFFSET:
        raise AssertionError(
            f"corpus design failure: marker for {doc['slug']} at offset "
            f"{offset}, needs > {MIN_MARKER_OFFSET}"
        )
    if "[" in body or "]" in body:
        raise AssertionError(
            f"corpus design failure: {doc['slug']} contains a bracket; the "
            "row snippet is escape_markup-escaped and the substring checks "
            "would compare escaped text against raw document text"
        )
    return body


def build_short_document(title: str, seed: int) -> str:
    """Build a short (~1.5k-char) corpus document with no planted marker.

    Args:
        title: The document's first line, used as its title.
        seed: Deterministic filler seed, so re-runs produce identical text.

    Returns:
        The document text: title, blank line, then filler.
    """
    return f"{title}\n\n{_filler(seed, 1500)}"


# ---------------------------------------------------------------------------
# Analysis helpers (pure)
# ---------------------------------------------------------------------------

#: Raw provenance `source_type` spellings `_SEMANTIC_SOURCE_TYPE_MAP` treats as
#: live but `library_expand_policy.EXPANDABLE_SOURCE_TYPES` (singular only)
#: does not -- the canonicalization VARIANTS TASK-16174's final review
#: finding 6 asked this probe to COUNT (the fix lives in TASK-16688).
VARIANT_SOURCE_TYPES = ("notes", "media_chunk", "conversations", "chat", "prompts")

_IDENTITY_KEYS = ("source_id", "chunk_id", "chunk_start", "note_id", "doc_id")


def _norm(text: str) -> str:
    """Whitespace-normalized text, so a collapsed snippet and the raw document
    it came from compare on their words rather than on their line breaks."""
    return " ".join(str(text or "").split())


def _expand_kwargs(row: dict, *, strip_fallbacks: bool, strip_anchor: bool) -> dict:
    kwargs: dict = {
        "source_type": row.get("source_type", ""),
        "source_id": row.get("source_id", ""),
    }
    if not strip_anchor and "chunk_start" in row:
        kwargs["chunk_start"] = row["chunk_start"]
    if not strip_fallbacks:
        for key in ("note_id", "doc_id"):
            if key in row:
                kwargs[key] = row[key]
    return kwargs


# ---------------------------------------------------------------------------
# The worker (`one` mode)
# ---------------------------------------------------------------------------


def _assert_isolated(scratch: pathlib.Path) -> pathlib.Path:
    from tldw_chatbook.config import get_user_data_dir

    data_dir = get_user_data_dir()
    if not str(data_dir).startswith(str(scratch)):
        raise SystemExit(f"NOT ISOLATED: data_dir={data_dir} scratch={scratch}")
    return data_dir


def _seed_canonical(data_dir: pathlib.Path) -> tuple[list, dict]:
    """Seed notes/media/conversations and build entries through the APP's own
    canonical document builders (`source_id`/`source_type` in every metadata)."""
    from datetime import datetime, timezone

    from tldw_chatbook.config import get_chachanotes_db_lazy, get_media_db_lazy
    from tldw_chatbook.RAG_Search.ingestion_indexing import (
        conversation_index_entry,
        media_index_entry,
        note_index_entry,
    )

    # The SAME handles `expand_document` resolves (`get_*_db_lazy`), so the
    # probe cannot seed one database and expand another.
    chacha = get_chachanotes_db_lazy()
    media_db = get_media_db_lazy()
    if chacha is None or media_db is None:
        raise SystemExit("FATAL: could not open the scratch profile databases")
    entries = []
    planted: dict[str, dict] = {}

    for doc in LONG_DOCS:
        body = build_long_document(doc)
        if doc["seam"] == "note":
            note_id = chacha.add_note(title=doc["title"], content=body)
            row = chacha.get_note_by_id(note_id)
            entries.append(note_index_entry(row))
            planted[doc["slug"]] = {
                "seam": "note",
                "db_id": str(note_id),
                "marker": doc["marker"],
                "marker_offset": body.find(doc["marker"]),
                "length": len(body),
            }
        elif doc["seam"] == "media":
            media_id, _uuid, _msg = media_db.add_media_with_keywords(
                url=f"probe16588://{doc['slug']}",
                title=doc["title"],
                media_type="document",
                content=body,
                keywords=["probe16588"],
                ingestion_date=datetime.now(timezone.utc).isoformat(),
            )
            row = media_db.get_media_by_id(media_id)
            entries.append(media_index_entry(row))
            planted[doc["slug"]] = {
                "seam": "media",
                "db_id": str(media_id),
                "marker": doc["marker"],
                "marker_offset": body.find(doc["marker"]),
                "length": len(body),
            }
        else:
            conv_id = chacha.add_conversation({"title": doc["title"]})
            # One message per paragraph, so the rendered transcript
            # (`sender: content`) stays the text `conversation_document`
            # chunks AND the text `expand_document` renders back.
            for index, para in enumerate(body.split("\n\n")):
                if not para.strip():
                    continue
                chacha.add_message(
                    {
                        "conversation_id": conv_id,
                        "sender": "user" if index % 2 == 0 else "assistant",
                        "content": para,
                    }
                )
            conversation = chacha.get_conversation_by_id(conv_id)
            messages = chacha.get_messages_for_conversation(conv_id, limit=500)
            entry = conversation_index_entry(conversation, messages)
            entries.append(entry)
            rendered = entry.document["content"]
            planted[doc["slug"]] = {
                "seam": "conversation",
                "db_id": str(conv_id),
                "marker": doc["marker"],
                "marker_offset": rendered.find(doc["marker"]),
                "length": len(rendered),
            }

    for index, title in enumerate(SHORT_NOTE_TITLES):
        note_id = chacha.add_note(
            title=title, content=build_short_document(title, 9000 + index)
        )
        entries.append(note_index_entry(chacha.get_note_by_id(note_id)))
    for index, title in enumerate(SHORT_MEDIA_TITLES):
        media_id, _uuid, _msg = media_db.add_media_with_keywords(
            url=f"probe16588://short-media-{index}",
            title=title,
            media_type="document",
            content=build_short_document(title, 9100 + index),
            keywords=["probe16588"],
            ingestion_date=datetime.now(timezone.utc).isoformat(),
        )
        entries.append(media_index_entry(media_db.get_media_by_id(media_id)))
    for index, title in enumerate(SHORT_CONVERSATION_TITLES):
        conv_id = chacha.add_conversation({"title": title})
        for line_index, para in enumerate(
            build_short_document(title, 9200 + index).split(". ")[:8]
        ):
            chacha.add_message(
                {
                    "conversation_id": conv_id,
                    "sender": "user" if line_index % 2 == 0 else "assistant",
                    "content": para + ".",
                }
            )
        entries.append(
            conversation_index_entry(
                chacha.get_conversation_by_id(conv_id),
                chacha.get_messages_for_conversation(conv_id, limit=500),
            )
        )
    return [entry for entry in entries if entry is not None], planted


def _seed_non_canonical(data_dir: pathlib.Path) -> tuple[list, dict]:
    """Seed the same NOTES, then index them the way TASK-15810's committed QA
    seeder does: a hand-built `IndexEntry` whose metadata is
    `{"type", "note_id", "title"}` -- no `source_id`, no `document_id`, so
    `_semantic_row` falls through to the vector store's point id."""
    from datetime import datetime, timezone

    from tldw_chatbook.config import get_chachanotes_db_lazy
    from tldw_chatbook.RAG_Search.ingestion_indexing import IndexEntry

    chacha = get_chachanotes_db_lazy()
    if chacha is None:
        raise SystemExit("FATAL: could not open the scratch profile database")
    entries = []
    planted: dict[str, dict] = {}

    def _add(title: str, body: str) -> str:
        note_id = chacha.add_note(title=title, content=body)
        entries.append(
            IndexEntry(
                item_id=str(note_id),
                item_type="note",
                last_modified=datetime.now(timezone.utc),
                document={
                    "id": f"note_{note_id}",
                    "content": body,
                    "title": title,
                    "metadata": {
                        "type": "note",
                        "note_id": str(note_id),
                        "title": title,
                    },
                },
            )
        )
        return str(note_id)

    for doc in LONG_DOCS:
        if doc["seam"] != "note":
            continue
        body = build_long_document(doc)
        note_id = _add(doc["title"], body)
        planted[doc["slug"]] = {
            "seam": "note",
            "db_id": note_id,
            "marker": doc["marker"],
            "marker_offset": body.find(doc["marker"]),
            "length": len(body),
        }
    for index, title in enumerate(SHORT_NOTE_TITLES):
        _add(title, build_short_document(title, 9000 + index))
    return entries, planted


class _ProbeApp:
    """Minimal stand-in for the Textual app: only what the Library service reads."""

    def __init__(self) -> None:
        from tldw_chatbook.config import get_chachanotes_db_lazy
        from tldw_chatbook.Library.library_local_rag_search_service import (
            LibraryLocalRagSearchService,
        )

        self.chachanotes_db = get_chachanotes_db_lazy()
        self.notes_user_id = "probe16588"
        self.library_rag_search_service = LibraryLocalRagSearchService(self)

    def notify(self, *args, **kwargs) -> None:  # pragma: no cover - shim
        pass


def _run_one(scratch: pathlib.Path, index_kind: str, out_json: pathlib.Path) -> None:
    import asyncio

    data_dir = _assert_isolated(scratch)
    print(f"RESOLVED data_dir: {data_dir}", flush=True)

    from tldw_chatbook.Agents.library_rag_tool_provider import (
        RAG_TOOL_NAME,
        LibraryRagToolProvider,
    )
    from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
    from tldw_chatbook.Library.library_tool_contract import (
        MAX_RESULT_BYTES,
        serialized_size,
    )
    from tldw_chatbook.Library.library_rag_service import (
        LibraryRagSearchRequest,
        run_library_rag_search,
    )
    from tldw_chatbook.RAG_Search.ingestion_indexing import (
        get_shared_rag_service,
        index_entries,
    )
    from tldw_chatbook.Tools.document_expansion_tool import ExpandDocumentTool

    seeder = _seed_canonical if index_kind == "canonical" else _seed_non_canonical
    entries, planted = seeder(data_dir)
    print(f"SEEDED entries: {len(entries)} (kind={index_kind})", flush=True)

    service = get_shared_rag_service()
    if service is None:
        raise SystemExit("FATAL: get_shared_rag_service() returned None")
    indexing_db = RAGIndexingDB(data_dir / "tldw_chatbook_rag_indexing.db")
    summary = asyncio.run(index_entries(service, indexing_db, entries))
    print(f"index_entries summary: {summary}", flush=True)
    stats = service.vector_store.get_collection_stats()
    print(f"vector store stats: {stats}", flush=True)

    app = _ProbeApp()
    provider = LibraryRagToolProvider(app.library_rag_search_service)
    tool = ExpandDocumentTool()

    queries = [
        {"slug": doc["slug"], "query": doc["query"], "marker": doc["marker"]}
        for doc in LONG_DOCS
        if index_kind == "canonical" or doc["seam"] == "note"
    ] + [dict(entry, marker=None) for entry in GENERAL_QUERIES]

    artifact: dict = {
        "index_kind": index_kind,
        "data_dir": str(data_dir),
        "entries_indexed": len(entries),
        "index_summary": summary,
        "vector_store_stats": stats,
        "planted": planted,
        "routes": {},
    }

    for route in ("semantic", "hybrid"):
        # The route the Library's `rag` mode takes is read fresh from the
        # ACTIVE profile on every call (`_resolve_profile_search_mode` reads
        # `rag_service.config.search.default_search_mode`), so setting it here
        # is exactly what selecting a semantic or hybrid profile in Settings
        # does -- no monkeypatching of the search path itself.
        service.config.search.default_search_mode = route
        route_record: dict = {"queries": [], "engine_control": []}
        for spec in queries:
            request = LibraryRagSearchRequest(
                query=spec["query"],
                source_types=("notes", "media", "conversations"),
                mode="rag",
                top_k=10,
                include_citations=True,
            )
            started = time.perf_counter()
            outcome = asyncio.run(run_library_rag_search(app, request))
            elapsed = time.perf_counter() - started

            # The PRODUCTION surface: the sealed payload an agent actually
            # receives. `library_rows` is the same call one layer down, kept
            # only so the probe can read the provenance the payload (by
            # design) never carries.
            tool_result = provider.invoke(RAG_TOOL_NAME, {"query": spec["query"], "top_k": 10})
            payload = json.loads(tool_result.content) if tool_result.ok else {}
            projected_rows = payload.get("results", [])
            library_rows = list(outcome.results or ())

            # Direct engine control: what metadata actually arrived, before
            # any Library normalization.
            raw = asyncio.run(
                service.search(
                    query=spec["query"],
                    top_k=10,
                    search_type=route,
                    include_citations=True,
                )
            )
            route_record["engine_control"].append(
                {
                    "slug": spec["slug"],
                    "rows": [
                        {
                            "point_id": getattr(item, "id", None),
                            "metadata_keys": sorted(
                                (getattr(item, "metadata", None) or {}).keys()
                            ),
                            "source_id": (getattr(item, "metadata", None) or {}).get(
                                "source_id"
                            ),
                            "doc_id": (getattr(item, "metadata", None) or {}).get(
                                "doc_id"
                            ),
                            "note_id": (getattr(item, "metadata", None) or {}).get(
                                "note_id"
                            ),
                            "chunk_start": (getattr(item, "metadata", None) or {}).get(
                                "chunk_start"
                            ),
                        }
                        for item in (raw or ())[:10]
                    ],
                }
            )

            # AC#4 corroborated on the REAL route payload, by the same
            # strip-and-reserialize method Task 1 used on a synthetic ten-row
            # payload: serialize as shipped, then again with the two fallback
            # keys removed from every row.
            stripped_payload = dict(payload)
            stripped_payload["results"] = [
                {key: value for key, value in row.items()
                 if key not in ("note_id", "doc_id")}
                for row in projected_rows
            ]
            query_record: dict = {
                "slug": spec["slug"],
                "query": spec["query"],
                "marker": spec["marker"],
                "status": outcome.status,
                "runtime_backend": outcome.runtime_backend,
                "seconds": round(elapsed, 3),
                "payload_status": payload.get("status"),
                "returned": payload.get("returned"),
                "payload_bytes": serialized_size(payload) if payload else 0,
                "payload_bytes_without_fallbacks": (
                    serialized_size(stripped_payload) if payload else 0
                ),
                "payload_ceiling_bytes": MAX_RESULT_BYTES,
                "rows_carrying_fallbacks": sum(
                    1 for row in projected_rows
                    if "note_id" in row or "doc_id" in row
                ),
                "rows": [],
            }

            for index, projected in enumerate(projected_rows):
                library_row = library_rows[index] if index < len(library_rows) else None
                provenance = dict(getattr(library_row, "provenance", {}) or {})
                raw_source_type = str(provenance.get("source_type") or "")
                snippet = str(projected.get("snippet") or "")
                hint = projected.get("expand_hint")
                row_record: dict = {
                    "rank": index + 1,
                    "result_id": projected.get("result_id"),
                    "title": projected.get("title"),
                    "raw_source_type": raw_source_type,
                    "hint": hint,
                    "identity": {
                        key: projected[key]
                        for key in _IDENTITY_KEYS
                        if key in projected
                    },
                    "snippet_has_marker": bool(
                        spec["marker"] and _norm(spec["marker"]) in _norm(snippet)
                    ),
                    "is_variant_without_hint": bool(
                        hint is None and raw_source_type in VARIANT_SOURCE_TYPES
                    ),
                    "expansions": {},
                }
                if hint is not None:
                    for arm, strip_fallbacks, strip_anchor in (
                        ("pre", True, False),
                        ("post", False, False),
                        ("head", False, True),
                    ):
                        kwargs = _expand_kwargs(
                            projected,
                            strip_fallbacks=strip_fallbacks,
                            strip_anchor=strip_anchor,
                        )
                        result = asyncio.run(tool.execute(**kwargs))
                        window_text = str(result.get("text") or "")
                        row_record["expansions"][arm] = {
                            "kwargs_keys": sorted(kwargs),
                            "chunk_start_passed": kwargs.get("chunk_start"),
                            "status": result.get("status"),
                            "resolved_source_id": result.get("source_id"),
                            "total_size": result.get("total_size"),
                            "window": result.get("window"),
                            "truncated": result.get("truncated"),
                            "window_has_marker": bool(
                                spec["marker"]
                                and _norm(spec["marker"]) in _norm(window_text)
                            ),
                            "window_has_snippet_head": bool(
                                snippet
                                and _norm(snippet)[:160] in _norm(window_text)
                            ),
                        }
                query_record["rows"].append(row_record)
            route_record["queries"].append(query_record)
        artifact["routes"][route] = route_record

    artifact["variant_detector_control"] = _variant_detector_control()
    artifact["counts"] = {
        route: _count_route(record) for route, record in artifact["routes"].items()
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(artifact, indent=1, default=str), encoding="utf-8")
    print(f"WROTE {out_json}", flush=True)
    for route, counts in artifact["counts"].items():
        print(f"COUNTS[{index_kind}/{route}] {json.dumps(counts)}", flush=True)


def _count_route(record: dict) -> dict:
    """Every count the plan pre-registered, per route.

    ``hinted`` is every row that carries identity at all (the hint's own
    precondition, ``expandable`` either way); ``expandable_true`` is the
    stricter subset the hint actually recommends following. Both are counted
    because ``not_found`` on either is a row whose identity the payload
    declared and the tool could not use.
    """
    counts = {
        "rows": 0,
        "hinted": 0,
        "expandable_true": 0,
        "not_found_pre_hinted": 0,
        "not_found_pre_expandable": 0,
        "not_found_post_hinted": 0,
        "not_found_post_expandable": 0,
        "ok_pre_hinted": 0,
        "ok_post_hinted": 0,
        "chunk_start_carried": 0,
        "fallbacks_carried": 0,
        "marker_rows": 0,
        "marker_window_post": 0,
        "marker_window_head": 0,
        "long_doc_windows_missing_marker": 0,
        "variant_rows_without_hint": 0,
        "payload_bytes_total": 0,
        "payload_bytes_without_fallbacks_total": 0,
        "payload_bytes_max": 0,
        "payload_ceiling_bytes": 0,
        "hint_reasons": {},
    }
    for query in record["queries"]:
        counts["payload_bytes_total"] += query.get("payload_bytes", 0)
        counts["payload_bytes_without_fallbacks_total"] += query.get(
            "payload_bytes_without_fallbacks", 0
        )
        counts["payload_bytes_max"] = max(
            counts["payload_bytes_max"], query.get("payload_bytes", 0)
        )
        counts["payload_ceiling_bytes"] = query.get("payload_ceiling_bytes", 0)
        for row in query["rows"]:
            counts["rows"] += 1
            if row["is_variant_without_hint"]:
                counts["variant_rows_without_hint"] += 1
            hint = row["hint"]
            if hint is None:
                continue
            counts["hinted"] += 1
            reason_key = f"{bool(hint.get('expandable'))}/{hint.get('reason')}"
            counts["hint_reasons"][reason_key] = (
                counts["hint_reasons"].get(reason_key, 0) + 1
            )
            expandable = bool(hint.get("expandable"))
            counts["expandable_true"] += expandable
            identity = row["identity"]
            if "chunk_start" in identity:
                counts["chunk_start_carried"] += 1
            if "note_id" in identity or "doc_id" in identity:
                counts["fallbacks_carried"] += 1
            pre = row["expansions"].get("pre", {})
            post = row["expansions"].get("post", {})
            head = row["expansions"].get("head", {})
            pre_missing = pre.get("status") == "not_found"
            post_missing = post.get("status") == "not_found"
            counts["not_found_pre_hinted"] += pre_missing
            counts["not_found_post_hinted"] += post_missing
            counts["not_found_pre_expandable"] += pre_missing and expandable
            counts["not_found_post_expandable"] += post_missing and expandable
            counts["ok_pre_hinted"] += pre.get("status") == "ok"
            counts["ok_post_hinted"] += post.get("status") == "ok"
            if row["snippet_has_marker"]:
                counts["marker_rows"] += 1
                hit = bool(post.get("window_has_marker"))
                counts["marker_window_post"] += hit
                counts["long_doc_windows_missing_marker"] += not hit
                counts["marker_window_head"] += bool(head.get("window_has_marker"))
    counts["fallback_bytes_total"] = (
        counts["payload_bytes_total"] - counts["payload_bytes_without_fallbacks_total"]
    )
    counts["fallback_bytes_per_carrying_row"] = round(
        counts["fallback_bytes_total"] / counts["fallbacks_carried"], 2
    ) if counts["fallbacks_carried"] else 0.0
    return counts


def _variant_detector_control() -> dict:
    """Prove the variant counter is not vacuous.

    A zero ``variant_rows_without_hint`` reading is only evidence if a variant
    row WOULD have been counted. This runs the same policy helper the adapter
    runs, over synthetic rows, and records that every VARIANT spelling gets no
    hint (so it would be counted) while every SINGULAR spelling does (so the
    detector is not simply always-true).
    """
    from tldw_chatbook.Library.library_expand_policy import (
        EXPANDABLE_SOURCE_TYPES,
        expand_hint,
    )

    def _probe(source_type: str) -> bool:
        row = {
            "source_id": "1",
            "chunk_id": "",
            "snippet": "text",
            "provenance": {"source_type": source_type},
        }
        return expand_hint(row) is None

    return {
        "variant_spellings_get_no_hint": {
            spelling: _probe(spelling) for spelling in VARIANT_SOURCE_TYPES
        },
        "singular_spellings_get_a_hint": {
            spelling: not _probe(spelling) for spelling in EXPANDABLE_SOURCE_TYPES
        },
    }


# ---------------------------------------------------------------------------
# The driver (`all` mode)
# ---------------------------------------------------------------------------


def _sha256(path: pathlib.Path) -> str:
    if not path.exists():
        return "<absent>"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_all(scratch: pathlib.Path, out_dir: pathlib.Path) -> None:
    real_config = pathlib.Path(os.path.expanduser("~/.config/tldw_cli/config.toml"))
    before = _sha256(real_config)
    print(f"REAL CONFIG BEFORE: {before}  {real_config}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    child_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "TERM": os.environ.get("TERM", "dumb"),
        "HOME": os.path.expanduser("~"),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "PYTHONUNBUFFERED": "1",
    }
    if "MODEL_CACHE" in os.environ:
        child_env["MODEL_CACHE"] = os.environ["MODEL_CACHE"]

    merged: dict = {}
    for kind in ("canonical", "noncanonical"):
        target = out_dir / f"probe-{kind}.json"
        print(f"== running {kind} ==", flush=True)
        completed = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "one",
             str(scratch), kind, str(target)],
            env=child_env,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(f"FATAL: {kind} worker exited {completed.returncode}")
        merged[kind] = json.loads(target.read_text(encoding="utf-8"))

    after = _sha256(real_config)
    merged["isolation"] = {
        "real_config_path": str(real_config),
        "real_config_sha256_before": before,
        "real_config_sha256_after": after,
        "unchanged": before == after,
    }
    (out_dir / "probe-artifacts.json").write_text(
        json.dumps(merged, indent=1, default=str), encoding="utf-8"
    )
    print(f"REAL CONFIG AFTER : {after}", flush=True)
    print(f"REAL CONFIG UNCHANGED: {before == after}", flush=True)
    print(f"WROTE {out_dir / 'probe-artifacts.json'}", flush=True)


def main() -> None:
    """Run one probe arm (seed, drive routes, expand, dump artifacts)."""
    if MODE == "one":
        _run_one(SCRATCH, INDEX_KIND, OUT_JSON)
    elif MODE == "all":
        _run_all(pathlib.Path(sys.argv[2]).resolve(), pathlib.Path(sys.argv[3]).resolve())
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
