"""Single source of truth for the chunking-engine version pin (task-21102).

``ENGINE_VERSION`` is the identity stamped into every chunk (spec §8,
task-12): ``Chunk_Lib`` writes it into ``metadata["chunk_engine_version"]``
for in-memory consumers, and the ingestion persist seam
(``Local_Ingestion.local_file_ingestion.persist_parsed_media``) stamps the
same value as the top-level ``chunk_engine_version`` key the DB writer
persists to ``UnvectorizedMediaChunks``.

It lives OUTSIDE the ``Chunking`` package on purpose: importing any
``tldw_chatbook.Chunking`` submodule executes ``Chunking/__init__.py`` and
with it the full shim + vendored engine (~15k LOC, 28/38 engine modules, an
``import langdetect`` attempt, an nltk ``find_spec`` path scan). The persist
seam is on the app's boot-import path and needs only this string, so the
string must be importable for the cost of a module this size.
``Chunking/Chunk_Lib.py`` re-imports and re-exports this name, keeping the
package surface (``tldw_chatbook.Chunking.ENGINE_VERSION``) the same object.
Guarded by ``Tests/Packaging/test_chunking_import_closure.py``.

The value tracks the vendored engine pin in
``Chunking/engine/VENDOR_MANIFEST.toml`` (``parity-1@<short upstream sha>``);
update it only when the vendored engine is re-synced.

This module must stay stdlib-only (currently: no imports at all).
"""

ENGINE_VERSION = "parity-1@385afa95"
