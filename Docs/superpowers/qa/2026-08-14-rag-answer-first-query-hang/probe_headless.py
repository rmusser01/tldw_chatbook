"""Headless probe: drive `run_library_rag_search` (RAG mode) against the SAME
seeded scratch profile the TUI reproduction uses, in a FRESH process where the
embedding factory has never initialized. Times the first query.
"""

import os
import asyncio
import time
import pathlib
import sys

SCRATCH = pathlib.Path(sys.argv[1]).resolve()
# Capture the REAL home BEFORE overwriting HOME -- expanduser("~") reads
# os.environ["HOME"], so computing it afterwards would point at the scratch
# home and silently change which model cache is reachable.
_REAL_HOME = os.path.expanduser("~")
os.environ["HOME"] = str(SCRATCH / "home")
os.environ["XDG_CONFIG_HOME"] = str(SCRATCH / "home/.config")
os.environ["XDG_DATA_HOME"] = str(SCRATCH / "home/.local/share")
os.environ["XDG_CACHE_HOME"] = str(SCRATCH / "home/.cache")
os.environ["TLDW_CONFIG_PATH"] = str(SCRATCH / "home/.config/tldw_cli/config.toml")
os.environ.setdefault("HF_HOME", os.path.join(_REAL_HOME, ".cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _load_app_modules():
    """Import the project's modules AFTER the scratch environment is set.

    Deliberate, and the reason this file has no module-scope project import:
    `tldw_chatbook.config` resolves data/config paths at import time, so
    importing it before the `os.environ` block above would bind the REAL
    profile and silently invalidate the whole probe.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )
    from tldw_chatbook.Library.library_rag_service import (
        LibraryRagSearchRequest,
        run_library_rag_search,
    )

    return (
        CharactersRAGDB,
        LibraryLocalRagSearchService,
        LibraryRagSearchRequest,
        run_library_rag_search,
    )


class FakeApp:
    """Minimal stand-in for the Textual app: only what the service reads."""

    def __init__(self, data_dir):
        self.chachanotes_db = CharactersRAGDB(
            data_dir / "tldw_chatbook_ChaChaNotes.db", client_id="probe15810"
        )
        self.library_rag_search_service = LibraryLocalRagSearchService(self)

    def notify(self, *a, **k):
        pass


async def main():
    (
        CharactersRAGDB,
        LibraryLocalRagSearchService,
        LibraryRagSearchRequest,
        run_library_rag_search,
    ) = _load_app_modules()
    # Project-local imports live HERE, not at module scope: the scratch
    # environment above must be set BEFORE tldw_chatbook is first
    # imported, or config/data paths resolve to the real profile.
    from tldw_chatbook.config import get_user_data_dir

    data_dir = get_user_data_dir()
    assert str(data_dir).startswith(str(SCRATCH)), f"NOT ISOLATED: {data_dir}"
    app = FakeApp(data_dir)
    request = LibraryRagSearchRequest(
        query="how do I use the command palette",
        source_types=("notes",),
        mode="rag",
        top_k=15,
        include_citations=True,
    )
    print("PROBE: first run_library_rag_search (rag mode), fresh process", flush=True)
    t0 = time.perf_counter()
    outcome = await run_library_rag_search(app, request)
    first = time.perf_counter() - t0
    print(f"FIRST QUERY: {first:.2f}s status={outcome.status} rows={len(outcome.results)}", flush=True)

    t1 = time.perf_counter()
    outcome2 = await run_library_rag_search(app, request)
    second = time.perf_counter() - t1
    print(f"SECOND QUERY: {second:.2f}s status={outcome2.status} rows={len(outcome2.results)}", flush=True)


asyncio.run(main())
