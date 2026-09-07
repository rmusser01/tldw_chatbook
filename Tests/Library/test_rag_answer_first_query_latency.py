"""TASK-15810: what the Library RAG Answer first query costs HEADLESSLY.

This file is the arc's **control**, not its reproduction. The defect --
RAG Answer's first query on a fresh profile never returns -- reproduces
under the real TUI (11+ minutes on `searching · Notes…` at ~98% CPU,
transcript in
`.superpowers/sdd/2026-08-14-rag-answer-first-query-hang/task-1-report.md`)
and does **not** reproduce here: driving the same coroutine
(`run_library_rag_search`, `mode="rag"`) against the same kind of profile,
in a process where the embedding factory has never initialized, the first
query returned in ~5.4s and the second in ~0.00s.

That asymmetry is evidence about the mechanism, so this test's job is to
keep the headless leg honest: if the engine path itself ever regresses into
minutes, the TUI-only attribution stops being true and this test says so.

The precondition the measurement needs is process-level ("the embedding
factory has NEVER initialized"), which indexing would destroy -- indexing
loads the model. Both legs therefore run in their own subprocess: one seeds,
one measures.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.slow]

# Generous by design. The observed headless first query is ~5.4s (model load
# dominated); the TUI stall it is the control for runs 11+ minutes. Anything
# in between is a real regression worth failing on, and this bound will not
# flake on a cold or loaded machine.
FIRST_QUERY_BOUND_S = 120.0

_SEED = textwrap.dedent(
    """
    import os, pathlib, sys
    SCRATCH = pathlib.Path(sys.argv[1])
    REPO = pathlib.Path(sys.argv[2])
    _REAL_HOME = os.path.expanduser("~")
    os.environ["HOME"] = str(SCRATCH / "home")
    os.environ["XDG_CONFIG_HOME"] = str(SCRATCH / "home/.config")
    os.environ["XDG_DATA_HOME"] = str(SCRATCH / "home/.local/share")
    os.environ["XDG_CACHE_HOME"] = str(SCRATCH / "home/.cache")
    os.environ["TLDW_CONFIG_PATH"] = str(SCRATCH / "home/.config/tldw_cli/config.toml")
    os.environ.setdefault("HF_HOME", os.path.join(_REAL_HOME, ".cache/huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import asyncio, json
    from datetime import datetime, timezone
    from tldw_chatbook.config import get_user_data_dir
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
    from tldw_chatbook.RAG_Search.ingestion_indexing import (
        IndexEntry, get_shared_rag_service, index_entries,
    )

    data_dir = get_user_data_dir()
    assert str(data_dir).startswith(str(SCRATCH)), data_dir
    db = CharactersRAGDB(data_dir / "tldw_chatbook_ChaChaNotes.db", client_id="seed15810")

    pages = sorted((REPO / "Docs" / "User_Guide").rglob("*.md"))[:8]
    entries = []
    for page in pages:
        text = page.read_text(encoding="utf-8", errors="replace")
        title = page.name
        note_id = db.add_note(title=title, content=text)
        entries.append(IndexEntry(
            item_id=str(note_id), item_type="note",
            last_modified=datetime.now(timezone.utc),
            document={"id": f"note_{note_id}", "content": text, "title": title,
                      "metadata": {"type": "note", "note_id": str(note_id), "title": title}},
        ))

    service = get_shared_rag_service()
    if service is None:
        print("RESULT " + json.dumps({"ok": False, "why": "no rag service"}))
        raise SystemExit(0)
    summary = asyncio.run(index_entries(
        service, RAGIndexingDB(data_dir / "tldw_chatbook_rag_indexing.db"), entries))
    stats = service.vector_store.get_collection_stats()
    print("RESULT " + json.dumps(
        {"ok": summary["indexed"] > 0, "indexed": summary["indexed"],
         "chunks": stats.get("count"), "errors": summary["errors"][:2]}))
    """
)

_MEASURE = textwrap.dedent(
    """
    import os, pathlib, sys
    SCRATCH = pathlib.Path(sys.argv[1])
    _REAL_HOME = os.path.expanduser("~")
    os.environ["HOME"] = str(SCRATCH / "home")
    os.environ["XDG_CONFIG_HOME"] = str(SCRATCH / "home/.config")
    os.environ["XDG_DATA_HOME"] = str(SCRATCH / "home/.local/share")
    os.environ["XDG_CACHE_HOME"] = str(SCRATCH / "home/.cache")
    os.environ["TLDW_CONFIG_PATH"] = str(SCRATCH / "home/.config/tldw_cli/config.toml")
    os.environ.setdefault("HF_HOME", os.path.join(_REAL_HOME, ".cache/huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import asyncio, json, time
    from tldw_chatbook.config import get_user_data_dir
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )
    from tldw_chatbook.Library.library_rag_service import (
        LibraryRagSearchRequest, run_library_rag_search,
    )

    # Nothing above has touched the embeddings factory: this process is in the
    # exact "never initialized" state the first query on a fresh profile is in.
    class FakeApp:
        def __init__(self, data_dir):
            self.chachanotes_db = CharactersRAGDB(
                data_dir / "tldw_chatbook_ChaChaNotes.db", client_id="probe15810")
            self.library_rag_search_service = LibraryLocalRagSearchService(self)
        def notify(self, *a, **k):
            pass

    async def main():
        data_dir = get_user_data_dir()
        assert str(data_dir).startswith(str(SCRATCH)), data_dir
        app = FakeApp(data_dir)
        request = LibraryRagSearchRequest(
            query="how do I use the command palette",
            source_types=("notes",), mode="rag", top_k=15, include_citations=True)
        t0 = time.perf_counter()
        first = await run_library_rag_search(app, request)
        first_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        second = await run_library_rag_search(app, request)
        second_s = time.perf_counter() - t1
        print("RESULT " + json.dumps({
            "first_seconds": first_s, "second_seconds": second_s,
            "first_status": first.status, "first_rows": len(first.results),
            "second_status": second.status}))

    asyncio.run(main())
    """
)


def _run(script: str, *args: str, timeout: float) -> dict:
    """Run `script` in a fresh interpreter and return its RESULT payload."""
    proc = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            k: v
            for k, v in os.environ.items()
            # Let the child resolve its own profile from the scratch config the
            # script sets; the suite's isolation vars would otherwise win.
            if k not in {"TLDW_CONFIG_PATH", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "HOME"}
        },
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError(
        f"subprocess produced no RESULT line (rc={proc.returncode})\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
    )


def first_query_seconds(scratch: Path, repo: Path) -> dict:
    """Seed a scratch profile, then time the FIRST RAG-mode query in a fresh process.

    This is the number Task 3 compares before/after on the headless leg. The
    TUI leg's number comes from `repro/reproduce.sh`.
    """
    (scratch / "home/.config/tldw_cli").mkdir(parents=True, exist_ok=True)
    (scratch / "data/verify15810/models").mkdir(parents=True, exist_ok=True)
    (scratch / "home/.config/tldw_cli/config.toml").write_text(
        "[general]\n"
        'users_name = "verify15810"\n\n'
        "[paths]\n"
        f'data_dir = "{scratch / "data"}"\n\n'
        "[first_run]\nsetup_started = true\nsetup_completed = true\n\n"
        "[embeddings]\n"
        'default_model_id = "all-MiniLM-L6-v2"\n\n'
        "[rag]\nenabled = true\n"
    )
    # The app's HuggingFace cache is PROFILE-LOCAL (<data_dir>/<user>/models/
    # embeddings), not $HF_HOME. A scratch profile therefore starts with an
    # empty model cache and, offline, cannot load the model at all -- so the
    # real profile's cache is copied in before first use, which is also what
    # "the embedding model was already on disk" meant in the 15700 check.
    # `Path.home()` reads $HOME, which Tests/conftest.py has already redirected
    # to a temp directory -- it would look for the model inside the very scratch
    # tree this function is creating and skip every time. The passwd database is
    # the only reading of "the real user's home" the fixture cannot move.
    # POSIX-only: imported here, not at module scope, so collection cannot
    # fail on a platform without `pwd` (markers do not prevent import-time
    # errors).
    import pwd

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    model_cache = real_home / ".local/share/tldw_cli/default_user/models/embeddings"
    if not model_cache.is_dir():
        pytest.skip(f"embedding model not on disk at {model_cache}")
    shutil.copytree(model_cache, scratch / "data/verify15810/models/embeddings")

    seeded = _run(_SEED, str(scratch), str(repo), timeout=600)
    if not seeded.get("ok"):
        pytest.skip(f"could not seed a real index: {seeded}")
    measured = _run(_MEASURE, str(scratch), timeout=600)
    measured["seed"] = seeded
    return measured


def test_headless_first_rag_answer_query_is_not_the_tui_stall(tmp_path):
    """The engine path completes headlessly -- the stall is TUI-side.

    Fails if the engine leg itself starts taking minutes, which would move
    the attribution off "TUI-only" and invalidate this arc's control.
    """
    repo = Path(__file__).resolve().parents[2]
    result = first_query_seconds(tmp_path / "profile", repo)
    # Visible under `pytest -s`: the arc compares these numbers across tasks.
    print(
        f"\nTASK-15810 headless: first={result['first_seconds']:.2f}s "
        f"second={result['second_seconds']:.2f}s "
        f"rows={result['first_rows']} seed={result['seed']}"
    )

    assert result["first_status"] == "ready", result
    assert result["first_rows"] > 0, result
    assert result["first_seconds"] < FIRST_QUERY_BOUND_S, (
        f"headless first RAG query took {result['first_seconds']:.1f}s "
        f"(bound {FIRST_QUERY_BOUND_S}s). The engine leg has regressed, or the "
        f"TASK-15810 stall is no longer TUI-only: {result}"
    )
    # The 'first, fresh' qualifier is the record's strongest clue: whatever the
    # first query pays for, the second must not pay again.
    assert result["second_seconds"] < result["first_seconds"], result
