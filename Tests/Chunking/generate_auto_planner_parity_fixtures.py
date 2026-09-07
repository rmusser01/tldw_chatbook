#!/usr/bin/env python3
"""Generate the auto-planner parity fixtures (chunking auto-selection Task 3,
AC 6 — the #2-goldens pattern of Tests/Chunking/golden/generate_golden.py).

Pins the vendored planner's outputs for the FIXED input family the
chatbook call site produces: every ``MEDIA_TYPE_MAP`` key (mapped through
the same expression ``auto_selection._plan_or_plain`` uses) plus the
no-metadata (``None``) and unmapped-passthrough cases, at the capability
flags chatbook passes (``perform_chunking=True``, ``chunking_mode="auto"``,
``goal="balanced"``, ``requested_llm=False``, ``llm_available=False``,
``semantic_available=True``). Both decision payloads are recorded as
JSON-safe dicts — ``chunk_options`` and the ``to_metadata()`` plan (which
carries the derived_views list, the rationale string, and the asdict-shaped
profile) — so a byte-level diff of the fixtures file is the parity evidence.

Why generation runs the VENDORED planner with test mode explicitly off:
spec §6.1/AC 6 requires fixtures "generated from the vendored module with
test mode off". The planner itself is stdlib-only and has no test-mode
detection, so the production-path discipline (pop PYTEST_CURRENT_TEST /
TLDW_TEST_MODE, force the engine-bound is_test_mode False) is belt-and-braces —
it matters for anything the planner's imports transitively touch and keeps
this generator honest against the same trap generate_golden.py documents.

Re-run this script at every sync_chunking_engine.py execution (the input
family is frozen; only a planner behavior change can move the bytes). A
diff means the planner's planning behavior moved and the parity claim must
be re-examined.

Usage:
    .venv/bin/python Tests/Chunking/generate_auto_planner_parity_fixtures.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- Freeze test mode OFF before importing anything engine-side (§10.2) ----
os.environ.pop("PYTEST_CURRENT_TEST", None)
os.environ.pop("TLDW_TEST_MODE", None)

REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import tldw_chatbook.Chunking.engine.chunker as _chunker_module  # noqa: E402

_chunker_module.is_test_mode = lambda: False

from tldw_chatbook.Chunking.auto_selection import (  # noqa: E402
    MEDIA_TYPE_MAP,
    SEMANTIC_AVAILABLE_DEFAULT,
)
from tldw_chatbook.Chunking.engine.auto_planner import plan_auto_chunking  # noqa: E402

#: The capability flags chatbook's tier-2 call site passes (spec §4.2/AC 5).
FLAGS = {
    "perform_chunking": True,
    "chunking_mode": "auto",
    "goal": "balanced",
    "requested_llm": False,
    "llm_available": False,
    "semantic_available": SEMANTIC_AVAILABLE_DEFAULT,
}

#: Cases beyond the map keys: no metadata at all, and a value the frozen
#: table does not name (the passthrough surfaces — add_media(media_type=…),
#: reading-import origin_type rows, chatbook-export imports — can carry
#: arbitrary caller strings; the identity fallback is part of the pinned
#: contract).
EXTRA_CASES = [None, "unmapped_legacy_passthrough"]


def main() -> int:
    cases = []
    for media_type in [*MEDIA_TYPE_MAP.keys(), *EXTRA_CASES]:
        planner_media_type = MEDIA_TYPE_MAP.get(media_type, media_type)
        decision = plan_auto_chunking(
            media_type=planner_media_type,
            **FLAGS,
        )
        cases.append(
            {
                "media_type": media_type,
                "planner_media_type": planner_media_type,
                "chunk_options": decision.chunk_options,
                "chunking_plan": decision.chunking_plan,
            }
        )
    payload = {
        "description": (
            "Vendored auto_planner parity fixtures: plan_auto_chunking outputs "
            "for every auto_selection.MEDIA_TYPE_MAP key (mapped exactly as "
            "the tier-2 call site maps them) plus the None and "
            "unmapped-passthrough cases, at the chatbook capability flags. "
            "Generated with test mode explicitly off; re-run at every "
            "sync_chunking_engine.py execution."
        ),
        "flags": FLAGS,
        "cases": cases,
    }
    out = HERE / "auto_planner_parity_fixtures.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
