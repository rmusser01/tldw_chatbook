#!/usr/bin/env python3
"""Generate golden parity fixtures for the vendored Chunking engine (Task 4).

Spec §10.2 requires golden fixtures generated with test mode explicitly
disabled, because the engine's input sanitization relaxes itself under
PYTEST_CURRENT_TEST / is_test_mode() and would otherwise exercise a different
code path than production.

Why this script runs the VENDORED chatbook engine rather than the server
engine: the vendored tree at tldw_chatbook/Chunking/engine/ is byte-for-byte
the server engine (pinned dev @ 385afa95) modulo the mechanical import
rewrite applied by Helper_Scripts/sync_chunking_engine.py (dotted Chunking →
engine, dotted other app.core → _shims, slashed path → engine path) — none of
which touches behavior. Generating from the vendored engine with test mode
off is therefore equivalent to generating from the server engine with test
mode off, and it keeps generation reproducible from this repo alone (no
tldw_server checkout required at fixture-regeneration time). Verified while
porting: both engines raise the same InvalidInputError for json/xml methods
on non-json/non-xml text, so even the error outcomes match.

Where a (corpus, method) combination legitimately raises (e.g. the `json`
method on prose — the engine requires parseable JSON), the fixture records
the deterministic error outcome instead of the chunk list, and the parity
test asserts the same error type is raised. A "skips" outcome is not parity:
same input must produce the same result or the same error on both sides.

Re-run this script at every sync_chunking_engine.py execution; the corpus and
options are frozen. A diff in any golden file means the engine's chunking
behavior moved and the parity claim must be re-examined.

Usage:
    .venv/bin/python Tests/Chunking/golden/generate_golden.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- Freeze test mode OFF before importing the engine (spec §10.2) ----------
# The engine's detection (engine/chunker.py:1373) is
#   os.getenv("PYTEST_CURRENT_TEST") != "" or is_test_mode()
# where is_test_mode is BOUND into the engine module at import time. Run
# outside pytest PYTEST_CURRENT_TEST is unset, and patching the engine-bound
# name (not just the shim's) guarantees the production branch regardless of
# ambient TLDW_TEST_MODE.
os.environ.pop("PYTEST_CURRENT_TEST", None)
os.environ.pop("TLDW_TEST_MODE", None)

REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

import tldw_chatbook.Chunking.engine.chunker as _chunker_module  # noqa: E402

_chunker_module.is_test_mode = lambda: False

from tldw_chatbook.Chunking.engine import Chunker, ChunkerConfig  # noqa: E402

# Must match Tests/Chunking/test_golden_parity.py (spec §10.2 corpus).
CORPUS = {
    "prose": "The quick brown fox jumps over the lazy dog. " * 20,
    "markdown_atx": "# Title\n\n## Section A\n\nPara one.\n\n## Section B\n\nPara two.\n",
    "ebook": "# Chapter 1\n\nFirst chapter text.\n\n# Chapter 2\n\nSecond chapter text.\n",
    "json": '{"data": [' + ", ".join(f'{{"item": {i}, "text": "value {i}"}}' for i in range(20)) + ']}',
    "xml": "<root>" + "".join(f"<item id='{i}'>text {i}</item>" for i in range(20)) + "</root>",
    "code": "def f%d():\n    return %d\n" * 10,
    "cjk": "这是一段中文文本。" * 10,
}
METHODS = [
    "words", "sentences", "paragraphs", "tokens", "json", "xml",
    "ebook_chapters", "structure_aware", "code", "fixed_size",
    "propositions",
]


def main() -> int:
    chunker = Chunker(ChunkerConfig())
    written = 0
    for corpus_key, text in CORPUS.items():
        for method in METHODS:
            try:
                result = chunker.process_text(
                    text, {"method": method, "max_size": 50, "overlap": 10}
                )
                payload = {"outcome": "ok", "chunks": result}
            except Exception as exc:  # deterministic error outcome (see docstring)
                payload = {"outcome": "error", "error_type": type(exc).__name__}
            path = HERE / f"{corpus_key}_{method}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written += 1
    print(f"wrote {written} golden fixtures to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
