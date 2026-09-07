"""Task 2 tests: ``Chunking/auto_selection.py`` — the three-tier decision
engine (spec ``2026-08-22-chunking-auto-selection-design.md`` §4.2; ACs 2-5,
14).

Fixture pattern: the in-memory v7 ``ChunkingTemplates`` store from
``Tests/Chunking/test_template_runtime.py`` (``_ConnDb`` over a raw
sqlite3 connection), seeded with flat bodies. Tier 1 is exercised through
#2's listing surface (``LocalRAGAdminService.list_templates`` — the
deleted-filtered interop listing decorated with the AC-24a validity flag),
so the seeds must be validator-valid unless a test explicitly wants a
stored-invalid row.
"""

from __future__ import annotations

import json
import sqlite3
import uuid as uuid_module

from tldw_chatbook.Chunking import auto_selection as aus
from tldw_chatbook.Chunking.auto_selection import AUTO_SENTINEL, resolve_auto

# ---------------------------------------------------------------------------
# Store fixtures (test_template_runtime.py's in-memory v7 pattern)
# ---------------------------------------------------------------------------


class _ConnDb:
    """Minimal DB handle shaped like the Media DB wrappers."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def get_connection(self) -> sqlite3.Connection:
        return self._conn


def _v7_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE ChunkingTemplates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT NOT NULL UNIQUE,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            template_json TEXT NOT NULL,
            tags TEXT,
            is_builtin BOOLEAN DEFAULT 0,
            version INTEGER DEFAULT 1,
            deleted BOOLEAN DEFAULT 0,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


def _store_with(rows) -> _ConnDb:
    """Seed an in-memory v7 store.

    ``rows`` is an iterable of ``(name, body)`` (dict body, stored as JSON)
    or ``(name, body, overrides)`` with any of ``is_builtin`` / ``deleted``.
    """
    conn = _v7_conn()
    for row in rows:
        name, body = row[0], row[1]
        overrides = row[2] if len(row) > 2 else {}
        conn.execute(
            "INSERT INTO ChunkingTemplates "
            "(uuid, name, description, template_json, is_builtin, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(uuid_module.uuid4()),
                name,
                "fixture",
                body if isinstance(body, str) else json.dumps(body),
                int(overrides.get("is_builtin", 0)),
                int(overrides.get("deleted", 0)),
            ),
        )
    return _ConnDb(conn)


def _classifier_body(
    *,
    media_types=("document",),
    min_score=0.4,
    priority=None,
    method="words",
    extra=None,
) -> dict:
    """A validator-valid flat body whose classifier block opts into auto."""
    classifier: dict = {"media_types": list(media_types)}
    if min_score is not None:
        classifier["min_score"] = min_score
    if priority is not None:
        classifier["priority"] = priority
    if extra:
        classifier.update(extra)
    return {
        "chunking": {"method": method, "config": {"max_size": 100, "overlap": 10}},
        "classifier": classifier,
    }


def _plain_body(method="words") -> dict:
    """A validator-valid flat body with NO classifier block (never opts in)."""
    return {
        "chunking": {"method": method, "config": {"max_size": 100, "overlap": 10}},
    }


# ---------------------------------------------------------------------------
# AC 14 surface — the sentinel
# ---------------------------------------------------------------------------


def test_auto_sentinel_is_the_reserved_name():
    assert AUTO_SENTINEL == "auto"


def test_reserved_legacy_row_never_candidate_never_shadowed():
    # A legacy row named "auto" (created before the reservation) carries a
    # would-win classifier block: tier 1 must skip it by name (never
    # selected), and the decision must fall through to the plan tier
    # (never shadowed — the sentinel does not resolve to this row).
    db = _store_with([("auto", _classifier_body(media_types=["document"]))])
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier == "plan"
    assert d.template is None
    assert any("auto" in r for r in d.fallback_reasons)


def test_cased_reserved_legacy_row_never_candidate():
    # (Qodo #4) The cased whole-word variants ("Auto", "AUTO") of a legacy
    # reserved row are flagged by the widened listing decoration and
    # skipped by tier 1 exactly like the exact-sentinel row — never
    # selected, never shadowed.
    db = _store_with([("Auto", _classifier_body(media_types=["document"]))])
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier == "plan"
    assert d.template is None
    assert any("Auto" in r for r in d.fallback_reasons)


# ---------------------------------------------------------------------------
# Tier 1 — the template tier
# ---------------------------------------------------------------------------


def test_template_tier_selects_positive_score_full_block():
    db = _store_with([("winner", _classifier_body(media_types=["document"], min_score=0.4))])
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier == "template"
    assert d.template is not None
    assert d.template["name"] == "winner"
    assert d.template["chunking"]["method"] == "words"
    assert d.chunk_options is None
    assert d.rationale and any("winner" in r for r in d.rationale)


def test_no_block_never_selected_absent_min_score_selects():
    # Spec §0.1's two pins, in one store each:
    # pin 1 — no classifier block at all → score 0.0 → NEVER selected
    #         (the six #2 built-ins included; the caller-side score > 0
    #         guard is the opt-in).
    db = _store_with([("no_block", _plain_body())])
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier != "template"
    # pin 2 — a present block with ABSENT min_score selects at any positive
    #         score (the vendored clamp defaults min_score to 0.0 — the
    #         brainstorm's absent-min_score veto was corrected in §0.1).
    db = _store_with([("no_min_score", _classifier_body(min_score=None))])
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier == "template"
    assert d.template["name"] == "no_min_score"


def test_stored_invalid_excluded_from_candidacy():
    # Ruling 8.8: a stored-invalid template (decoration's
    # template_valid is False) is not a candidate even with a would-win
    # classifier block — auto must never pick a body the apply path would
    # then refuse. Losing to nothing → the plan tier.
    db = _store_with(
        [
            (
                "invalid",
                {
                    "chunking": {"method": "no_such_method", "config": {}},
                    "classifier": {"media_types": ["document"]},
                },
            )
        ]
    )
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier == "plan"
    assert d.template is None
    assert any("invalid" in r for r in d.fallback_reasons)


def test_soft_deleted_rows_are_not_listed_candidates():
    # The interop listing is deleted-filtered; a deleted row with a
    # would-win block must not surface through auto either.
    db = _store_with(
        [
            ("gone", _classifier_body(), {"deleted": 1}),
        ]
    )
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier == "plan"


def test_tiebreak_priority_then_listing_order():
    # Higher classifier priority wins at equal score...
    db = _store_with(
        [
            ("aaa", _classifier_body(priority=5)),
            ("zzz", _classifier_body(priority=10)),
        ]
    )
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.template["name"] == "zzz"
    # ...and at equal priority the first-listed row wins (#2's listing is
    # name-ordered, so ties keep the alphabetically-first name).
    db = _store_with(
        [
            ("zzz", _classifier_body(priority=3)),
            ("aaa", _classifier_body(priority=3)),
        ]
    )
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.template["name"] == "aaa"


def test_highest_score_beats_priority():
    # The key is (score, priority) strictly-greater with score FIRST: a
    # media+regex match (0.5 + 1/6) beats a media-only match (0.5) whatever
    # the priorities say.
    low = _classifier_body(priority=99)
    high = _classifier_body(priority=0, extra={"title_regex": "^Doc"})
    db = _store_with([("low", low), ("high", high)])
    d = resolve_auto(db, media_type="document", title="Doc Title", filename=None, url=None)
    assert d.template["name"] == "high"


def test_min_score_clamp_can_veto_a_match():
    # A block whose min_score exceeds its own score clamps to 0.0 upstream
    # (score <= 0 → continue): media-match 0.5 under min_score 0.6 → never.
    db = _store_with([("clamped", _classifier_body(min_score=0.6))])
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier != "template"


def test_regex_only_match_selects():
    # No media_types in the block: a filename_regex hit alone (0.5/3) is a
    # positive score and selects.
    db = _store_with(
        [
            (
                "by_regex",
                _classifier_body(media_types=[], min_score=None, extra={"filename_regex": r"\.pdf$"}),
            )
        ]
    )
    d = resolve_auto(db, media_type="pdf", title="t", filename="report.pdf", url=None)
    assert d.tier == "template"
    assert d.template["name"] == "by_regex"


def test_malformed_block_skipped_with_reason(monkeypatch):
    # One poisoned evaluation + one healthy: the healthy template wins, and
    # the skip is explained — never fatal (spec §4.2 per-candidate guard).
    real = aus.TemplateClassifier

    class ExplodingClassifier:
        @staticmethod
        def score(template_cfg, *, media_type, title, url, filename):
            marker = (template_cfg.get("classifier") or {}).get("marker")
            if marker == "poison":
                raise ValueError("poisoned classifier block")
            return real.score(
                template_cfg,
                media_type=media_type,
                title=title,
                url=url,
                filename=filename,
            )

    monkeypatch.setattr(aus, "TemplateClassifier", ExplodingClassifier)
    db = _store_with(
        [
            ("poisoned", _classifier_body(extra={"marker": "poison"})),
            ("healthy", _classifier_body()),
        ]
    )
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier == "template"
    assert d.template["name"] == "healthy"
    assert any("poisoned" in r for r in d.fallback_reasons)


def test_six_builtins_never_auto_selected():
    # The opt-in proof as a standing test (AC 3): all six #2 built-ins ship
    # without classifier blocks, so auto over every media type lands on a
    # non-template tier.
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    rows = [
        (seed["name"], seed["template"], {"is_builtin": 1})
        for seed in MediaDatabase._SERVER_BUILTIN_CHUNKING_TEMPLATES
    ]
    assert len(rows) == 6
    db = _store_with(rows)
    for media_type in (
        "document",
        "pdf",
        "ebook",
        "audio",
        "video",
        "email",
        "web_document",
        "message",
    ):
        d = resolve_auto(db, media_type=media_type, title="t", filename=None, url=None)
        assert d.tier != "template", media_type


# ---------------------------------------------------------------------------
# Tier 1 → tier 2 suppression (ruling 8.2 — the never-runs pin)
# ---------------------------------------------------------------------------


def test_planner_never_runs_when_template_won(monkeypatch):
    called = []
    monkeypatch.setattr(aus, "plan_auto_chunking", lambda **kw: called.append(kw))
    db = _store_with([("winner", _classifier_body())])
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier == "template"
    assert not called  # mutation target for the reverse pin


def test_planner_runs_when_no_template_won(monkeypatch):
    called = []
    monkeypatch.setattr(
        aus,
        "plan_auto_chunking",
        lambda **kw: called.append(kw)
        or aus.AutoChunkingDecision(chunk_options={"method": "sentences"}, chunking_plan={}),
    )
    db = _store_with([])
    d = resolve_auto(db, media_type="document", title="t", filename=None, url=None)
    assert d.tier == "plan"
    assert len(called) == 1


# ---------------------------------------------------------------------------
# Tier 2 — the plan tier
# ---------------------------------------------------------------------------


def test_plan_tier_when_no_candidate():
    db = _store_with([])
    d = resolve_auto(db, media_type="pdf", title="t", filename=None, url=None)
    assert d.tier == "plan"
    assert d.template is None
    assert d.chunk_options is not None
    assert d.chunk_options["method"]
    assert "max_size" in d.chunk_options
    # Spec §5: an empty store is one fallback reason, then tier 2.
    assert any("empty" in r for r in d.fallback_reasons)


def test_plan_tier_pins_planner_call_contract(monkeypatch):
    # AC 5: llm_available=False and requested_llm=False (#6's boundary
    # assistant owns availability later), perform_chunking=True +
    # chunking_mode="auto", goal rides through, semantic_available=True
    # (the engine's semantic strategy is registered unconditionally — no
    # embeddings-config reader gates it at this layer), media_type rides
    # MEDIA_TYPE_MAP (web normalization seed).
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return aus.AutoChunkingDecision(
            chunk_options={"method": "sentences", "max_size": 1, "overlap": 0},
            chunking_plan={},
        )

    monkeypatch.setattr(aus, "plan_auto_chunking", fake)
    db = _store_with([])
    resolve_auto(db, media_type="web_document", title="t", filename=None, url=None)
    assert seen["perform_chunking"] is True
    assert seen["chunking_mode"] == "auto"
    assert seen["goal"] == "balanced"
    assert seen["requested_llm"] is False
    assert seen["llm_available"] is False
    assert seen["semantic_available"] is True
    assert seen["media_type"] == "web"  # MEDIA_TYPE_MAP web normalization seed


def test_goal_rides_through_to_planner(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return aus.AutoChunkingDecision(
            chunk_options={"method": "sentences", "max_size": 1, "overlap": 0},
            chunking_plan={},
        )

    monkeypatch.setattr(aus, "plan_auto_chunking", fake)
    db = _store_with([])
    resolve_auto(db, media_type="pdf", title="t", filename=None, url=None, goal="qa_search")
    assert seen["goal"] == "qa_search"


# ---------------------------------------------------------------------------
# Tier 3 — the plain tier
# ---------------------------------------------------------------------------


def test_plain_tier_when_perform_chunking_context_declines(monkeypatch):
    # The planner declining to produce options (chunk_options None) is the
    # plain tier: the caller keeps today's defaults. Auto never raises for
    # a selection outcome.
    monkeypatch.setattr(
        aus,
        "plan_auto_chunking",
        lambda **kw: aus.AutoChunkingDecision(chunk_options=None, chunking_plan=None),
    )
    db = _store_with([])
    d = resolve_auto(db, media_type="pdf", title="t", filename=None, url=None)
    assert d.tier == "plain"
    assert d.template is None
    assert d.chunk_options is None
    assert d.rationale


def test_planner_exception_falls_to_plain(monkeypatch):
    # "Auto cannot fail; it can only explain why it declined" (§4.2): an
    # exploding planner is a plain-tier decision with a reason, never an
    # exception out of resolve_auto.
    def boom(**kw):
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(aus, "plan_auto_chunking", boom)
    db = _store_with([])
    d = resolve_auto(db, media_type="pdf", title="t", filename=None, url=None)
    assert d.tier == "plain"
    assert d.chunk_options is None
    assert any("planner" in r for r in d.fallback_reasons)
