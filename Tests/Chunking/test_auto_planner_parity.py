"""Auto-planner parity fixtures (spec §6.1, AC 6 — the #2-goldens pattern).

``Tests/Chunking/auto_planner_parity_fixtures.json`` pins the vendored
planner's outputs for the fixed input family chatbook produces: every
``MEDIA_TYPE_MAP`` key (mapped exactly as the tier-2 call site maps them)
plus the no-metadata and unmapped-passthrough cases, at the capability
flags ``auto_selection._plan_or_plain`` passes. The fixtures were generated
by ``Tests/Chunking/generate_auto_planner_parity_fixtures.py`` with test
mode explicitly off; re-run the generator at every
``sync_chunking_engine.py`` execution. This test runs under the
``production_path`` marker so the asserted path is the same one the
fixtures were generated on; a mismatch means the vendored planner's
planning behavior drifted and the parity claim must be re-examined.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.Chunking.auto_selection import (
    MEDIA_TYPE_MAP,
    SEMANTIC_AVAILABLE_DEFAULT,
)
from tldw_chatbook.Chunking.engine.auto_planner import plan_auto_chunking

pytestmark = [pytest.mark.unit, pytest.mark.production_path]

FIXTURES = Path(__file__).parent / "auto_planner_parity_fixtures.json"

#: The capability flags the tier-2 call site passes (spec §4.2/AC 5). The
#: fixtures' own "flags" block must equal this, so a change to the call
#: contract fails here rather than generating silent fixture rot.
CALL_SITE_FLAGS = {
    "perform_chunking": True,
    "chunking_mode": "auto",
    "goal": "balanced",
    "requested_llm": False,
    "llm_available": False,
    "semantic_available": SEMANTIC_AVAILABLE_DEFAULT,
}


def _load():
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return payload


def test_fixture_flags_match_the_call_site_contract():
    assert _load()["flags"] == CALL_SITE_FLAGS


def test_fixtures_cover_every_map_entry_plus_passthrough_cases():
    cases = _load()["cases"]
    covered = {case["media_type"] for case in cases}
    # Every frozen map entry has a pinned output...
    assert set(MEDIA_TYPE_MAP) <= covered
    # ...plus the no-metadata and identity-fallback passthrough cases.
    assert None in covered
    assert any(
        case["media_type"] not in MEDIA_TYPE_MAP
        and case["planner_media_type"] == case["media_type"]
        for case in cases
    )


@pytest.mark.parametrize(
    "case", _load()["cases"], ids=lambda case: str(case["media_type"])
)
def test_planner_output_byte_matches_fixture(case):
    # The mapping outcome is pinned too: the table must still produce the
    # same planner input it produced at generation time.
    assert MEDIA_TYPE_MAP.get(case["media_type"], case["media_type"]) == case[
        "planner_media_type"
    ]
    decision = plan_auto_chunking(
        media_type=case["planner_media_type"],
        **CALL_SITE_FLAGS,
    )
    assert decision.chunk_options == case["chunk_options"]
    assert decision.chunking_plan == case["chunking_plan"]
