"""Seed the TASK-15810 scratch profile the way the 15700 live check did.

36 real Docs/User_Guide pages written through the app's own `add_note`, then
indexed through the app's own `index_entries` against the shared RAG service.

Environment isolation is set BEFORE any tldw_chatbook import (lessons-live-
verification: "A bare interpreter call is not an isolated test").
"""

import asyncio
import os
import pathlib
import sys
from datetime import datetime, timezone

SCRATCH = pathlib.Path(sys.argv[1]).resolve()
REPO = pathlib.Path(sys.argv[2]).resolve()

# Capture the REAL home BEFORE overwriting HOME -- expanduser("~") reads
# os.environ["HOME"], so computing it afterwards would point at the scratch
# home and silently change which model cache is reachable.
_REAL_HOME = os.path.expanduser("~")
os.environ["HOME"] = str(SCRATCH / "home")
os.environ["XDG_CONFIG_HOME"] = str(SCRATCH / "home" / ".config")
os.environ["XDG_DATA_HOME"] = str(SCRATCH / "home" / ".local" / "share")
os.environ["XDG_CACHE_HOME"] = str(SCRATCH / "home" / ".cache")
os.environ["TLDW_CONFIG_PATH"] = str(SCRATCH / "home/.config/tldw_cli/config.toml")
# The embedding model must ALREADY be on disk and a download must be impossible.
os.environ.setdefault("HF_HOME", os.path.join(_REAL_HOME, ".cache/huggingface"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# DELIBERATE E402: every import below must happen AFTER the os.environ block
# above. `tldw_chatbook.config` resolves the data/config directories at
# import time, so hoisting these would bind the REAL profile and silently
# invalidate the isolation this script exists to provide (the assert on
# `data_dir` a few lines down is what catches that mistake). This is a flat
# script with module-level state; wrapping it in a function to satisfy the
# import-order rule would restructure a working reproduction for style.
from tldw_chatbook.config import get_user_data_dir  # noqa: E402
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB  # noqa: E402
from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB  # noqa: E402
from tldw_chatbook.RAG_Search.ingestion_indexing import (  # noqa: E402
    IndexEntry,
    get_shared_rag_service,
    index_entries,
)

data_dir = get_user_data_dir()
print("RESOLVED data_dir:", data_dir, flush=True)
assert str(data_dir).startswith(str(SCRATCH)), f"NOT ISOLATED: {data_dir}"

db = CharactersRAGDB(data_dir / "tldw_chatbook_ChaChaNotes.db", client_id="seed15810")

pages = sorted((REPO / "Docs" / "User_Guide").rglob("*.md"))
print(f"User Guide pages found: {len(pages)}", flush=True)

entries = []
for page in pages:
    text = page.read_text(encoding="utf-8", errors="replace")
    title = page.relative_to(REPO / "Docs" / "User_Guide").as_posix()
    note_id = db.add_note(title=title, content=text)
    entries.append(
        IndexEntry(
            item_id=str(note_id),
            item_type="note",
            last_modified=datetime.now(timezone.utc),
            document={
                "id": f"note_{note_id}",
                "content": text,
                "title": title,
                "metadata": {"type": "note", "note_id": str(note_id), "title": title},
            },
        )
    )

print(f"Notes written via add_note: {len(entries)}", flush=True)

service = get_shared_rag_service()
if service is None:
    raise SystemExit("FATAL: get_shared_rag_service() returned None")
print("RAG service:", type(service).__name__, flush=True)

indexing_db = RAGIndexingDB(data_dir / "tldw_chatbook_rag_indexing.db")
summary = asyncio.run(index_entries(service, indexing_db, entries))
print("index_entries summary:", summary, flush=True)

stats = service.vector_store.get_collection_stats()
print("vector store stats:", stats, flush=True)
