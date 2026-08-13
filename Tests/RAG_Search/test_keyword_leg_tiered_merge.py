"""The keyword leg's FORM-TIERED sub-leg merge (TASK-15700, Part A).

`RAGService._keyword_search` runs four source sub-legs (media, notes,
conversations, prompts) and merges them into ONE leg that hybrid fusion
then consumes by LEG RANK. Until this arc that merge was a plain
round-robin over sub-leg POSITION in a fixed source order (media first),
so *how many rows one sub-leg returned changed every other sub-leg's leg
rank* -- a property with no relation to the query.

**The incident (TASK-15400 Task 3, measured over the 172-doc golden
corpus).** Under the `and_then_or` construction, `kw-plant-maintenance-
record` -> `note-saltmarsh-hide` had the notes sub-leg's UNTOUCHED rank-1
AND row. The media and conversations sub-legs found zero AND rows, fell
back to the OR form, and injected 10 rows each; the round-robin put media
FIRST, demoting the untouched notes row from leg rank 1 to leg rank 2.
Fusion (alpha 0.7, rrf_k 5) then scored the vector rank-9 row above it by
6.94e-18 and the fixture lost its hybrid rescue. The same displacement
decomposes the scoped category exactly: the four NOTE-targeted scoped
queries each dropped behind a media fallback row while the three
MEDIA-targeted ones kept leg rank 1 -- 3 of 7 = 0.429, the measured cell
to the digit.

**The fix, pinned here.** The sub-leg rankings are partitioned by the FORM
their rows carry: tier 1 = sub-legs whose rows came from the
construction's PRIMARY form (`_fts5_primary_form()`), tier 2 = sub-legs
that fell back. Round-robin runs WITHIN each tier exactly as before, and
tier 1 wholly precedes tier 2. Because the merge truncates to `top_k`
AFTER concatenation, tier 2 only ever FILLS slots tier 1 left empty.

Five properties, each with the mutation that reds it:

* **(a) The AC#2 displacement pin.** One sub-leg's primary rank-1 row must
  lead a fallback sub-leg's many rows. RED on the pre-arc round-robin
  (media interleaves first); reds again if the tiering is removed. Pinned
  twice, once per fallback form the engine can run: the incident's own OR
  fallback, and (TASK-15700 Part B) the PREFIX fallback of
  `and_then_prefix` -- the composition whose whole case rests on this
  tiering holding for a form the incident never involved.
* **(b) All-primary byte-identity.** When no sub-leg fell back -- which is
  every sub-leg under a construction that HAS no fallback form: the legacy
  `and`, the former default `and_stopword_trim`, `or`, and TASK-15700's
  `prefix` -- the merged list is the SAME OBJECTS IN THE SAME ORDER a
  single unpartitioned `interleave_rankings` produces. Compared by object
  identity against the real gathered rankings, so this is byte identity and
  not a re-derivation.

  **Note what this case is NOT, since 2026-08-13:** it is no longer the
  default path. The shipped `and_then_prefix` defines a fallback, so a
  default search can and does populate tier 2, and the partition is live
  rather than a latent no-op. This property now covers the fallback-less
  constructions specifically; property (a) covers the shipped one.
  Reds if the tier order is inverted -- for a construction that HAS a
  fallback; under an all-primary one tier 2 is empty and the inversion is a
  provable no-op (Task 1's mutation note, and the reason the prefix
  parametrization here is coverage rather than a second inversion pin).
* **(c) Rank-fairness between primaries is KEPT.** Two primary sub-legs,
  many rows vs one: the round-robin order is unchanged. This is the pin
  that stops the fix overreaching -- rank-fairness among primaries is
  correct behaviour (raw FTS5 scores are not comparable across sources)
  and only the fallback case was ever the defect.
* **(d) Tier 2 fills, never displaces.** Tier 1 holding >= `top_k` rows
  leaves ZERO tier-2 rows in the output. Reds if the tier order is
  inverted.
* **(e) No tier 2 without a fallback.** Structural: a construction with no
  fallback concept cannot produce a tier-2 entry, because
  `_fts5_match_expressions` returns no fallback expression at all and every
  row is stamped with the primary form.

Real databases throughout (a real `MediaDatabase` / `CharactersRAGDB` /
`PromptsDatabase`, their own writers maintaining the FTS indexes), per the
`test_fts5_match_construction` / `test_keyword_leg_chacha` template: the
thing under test is which rows FTS5 actually returns for which form, and a
mock would pin nothing.
"""
import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.RAG_Search.fusion import interleave_rankings
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import (
    FTS_MATCH_AND,
    FTS_MATCH_CONSTRUCTION_AND,
    FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM,
    FTS_MATCH_CONSTRUCTION_AND_THEN_OR,
    FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX,
    FTS_MATCH_CONSTRUCTION_OR,
    FTS_MATCH_CONSTRUCTION_PREFIX,
    FTS_MATCH_CONSTRUCTIONS,
    FTS_MATCH_FORMS_BY_CONSTRUCTION,
    FTS_MATCH_OR,
    FTS_MATCH_PREFIX,
    RAGService,
    _fusion_doc_key,
)

CLIENT_ID = "test_keyword_leg_tiered_merge"

# The displacement fixture, at unit scale. Every content token of the
# query is present in the NOTE, so the note sub-leg matches on the full
# implicit AND (the `and_then_or` PRIMARY) and is never widened. No media
# row contains all of them, so the media sub-leg finds zero AND rows,
# falls back to the OR form, and returns SEVERAL rows -- the exact shape
# that demoted `note-saltmarsh-hide` to leg rank 2 in the gated run.
DISPLACEMENT_QUERY = "how does the wombat template work"

NOTE_AND_HIT = (
    "Saltmarsh hide",
    "How does the wombat template work when the hide is unstaffed?",
)
MEDIA_OR_ONLY_ROWS = [
    ("Wombat burrow survey", "Notes on the wombat burrow entrance survey."),
    ("Template library", "The template library indexes every studio template."),
    ("Work rota", "The work rota covers the dusk and dawn shifts."),
    ("Wombat rescue log", "A wombat rescue log entry from the salt flats."),
]

# The same displacement shape, driven by the PREFIX form (TASK-15700 Part B,
# the `and_then_prefix` row). The note carries every query token verbatim, so
# the notes sub-leg matches the full implicit AND and is never widened. No
# media row contains "template" or "work" as whole words -- only longer words
# they are prefixes OF -- so the media sub-leg finds zero AND rows, falls back
# to `"wombat"* "template"* "work"*` and returns three.
PREFIX_DISPLACEMENT_QUERY = "wombat template work"
PREFIX_NOTE_AND_HIT = (
    "Saltmarsh hide",
    "The wombat template work is logged at the hide each dusk.",
)
MEDIA_PREFIX_ONLY_ROWS = [
    ("Wombat templates rota", "The wombat templates working rota for the burrow."),
    ("Wombat templating notes", "Wombat templating workshop notes from the studio."),
    ("Wombat templated log", "A wombat templated workbook entry from the flats."),
]


def _make_service(
    construction=None,
    media_db_path=None,
    chachanotes_db_path=None,
    prompts_db_path=None,
):
    """A RAGService with the in-memory vector store and mock embeddings.

    Args:
        construction: `fts_match_construction` value, or None for the
            shipped default.
        media_db_path: Seeded media DB path, if the media sub-leg runs.
        chachanotes_db_path: Seeded ChaChaNotes DB path, if the notes or
            conversations sub-legs run.
        prompts_db_path: Seeded prompts DB path, if the prompts sub-leg runs.

    Returns:
        The configured `RAGService`.
    """
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = False
    if construction is not None:
        cfg.search.fts_match_construction = construction
    if media_db_path is not None:
        cfg.search.media_db_path = Path(media_db_path)
    if chachanotes_db_path is not None:
        cfg.search.chachanotes_db_path = Path(chachanotes_db_path)
    if prompts_db_path is not None:
        cfg.search.prompts_db_path = Path(prompts_db_path)
    return RAGService(cfg)


def _seed_media(tmp_path, rows, name="tldw_cli_media_v2.db"):
    """A real MediaDatabase (its writer maintains `media_fts`).

    Args:
        tmp_path: Directory to hold the database file.
        rows: `(title, content)` pairs, seeded in order.
        name: Database file name.

    Returns:
        The database path.
    """
    db_path = tmp_path / name
    db = MediaDatabase(db_path=str(db_path), client_id=CLIENT_ID)
    try:
        for title, content in rows:
            media_id, _, message = db.add_media_with_keywords(
                title=title,
                content=content,
                media_type="article",
                author="Tester",
                url=f"https://example.com/{title.replace(' ', '-').lower()}",
            )
            assert media_id is not None, f"media seed failed: {message}"
    finally:
        db.close_connection()
    return db_path


def _seed_notes(tmp_path, rows, name="chachanotes.db"):
    """A real CharactersRAGDB with notes (its writer maintains `notes_fts`).

    Args:
        tmp_path: Directory to hold the database file.
        rows: `(title, content)` pairs, seeded in order.
        name: Database file name.

    Returns:
        The database path.
    """
    db_path = tmp_path / name
    db = CharactersRAGDB(db_path, client_id=CLIENT_ID)
    try:
        for title, content in rows:
            db.add_note(title=title, content=content)
    finally:
        db.close_connection()
    return db_path


def _seed_prompts(tmp_path, rows, name="prompts.db"):
    """A real PromptsDatabase (its writer maintains `prompts_fts`).

    Args:
        tmp_path: Directory to hold the database file.
        rows: `(name, system_prompt)` pairs, seeded in order.
        name: Database file name.

    Returns:
        The database path.
    """
    db_path = tmp_path / name
    db = PromptsDatabase(db_path, client_id=CLIENT_ID)
    try:
        for prompt_name, system_prompt in rows:
            prompt_id, _uuid, message = db.add_prompt(
                name=prompt_name,
                author=None,
                details=None,
                system_prompt=system_prompt,
            )
            assert prompt_id is not None, f"prompt seed failed: {message}"
    finally:
        db.close_connection()
    return db_path


def _capture_sublegs(monkeypatch):
    """Record every sub-leg's returned ranking, keyed by method name.

    The captured lists hold the SAME `SearchResult` objects the gather site
    merges, so a test can rebuild the gathered `rankings` argument exactly
    and compare the merge against a direct `interleave_rankings` call by
    object identity.

    Args:
        monkeypatch: pytest's monkeypatch fixture.

    Returns:
        A dict populated during the call: `_media_keyword_subleg` and
        `_prompts_keyword_subleg` map to one ranking each,
        `_chacha_keyword_sublegs` to a list of rankings. A sub-leg that was
        not selected never runs and is absent.
    """
    captured: dict = {}

    def install(name):
        original = getattr(RAGService, name)

        async def wrapper(self, *args, **kwargs):
            rows = await original(self, *args, **kwargs)
            captured[name] = rows
            return rows

        monkeypatch.setattr(RAGService, name, wrapper)

    for method in (
        "_media_keyword_subleg",
        "_chacha_keyword_sublegs",
        "_prompts_keyword_subleg",
    ):
        install(method)
    return captured


def _gathered_rankings(captured):
    """Rebuild the gather site's `rankings` list from captured sub-legs.

    Mirrors `_keyword_search`'s own construction order verbatim: media,
    then the chacha sub-legs (notes, conversations), then prompts, with
    empty rankings dropped.

    Args:
        captured: `_capture_sublegs`' dict.

    Returns:
        The list of non-empty per-sub-leg rankings, in gather order.
    """
    media = captured.get("_media_keyword_subleg") or []
    chacha = captured.get("_chacha_keyword_sublegs") or []
    prompts = captured.get("_prompts_keyword_subleg") or []
    return [ranking for ranking in (media, *chacha, prompts) if ranking]


# --- (a) THE AC#2 PIN: a fallback sub-leg may not displace a primary row ---


def test_a_primary_rank_1_row_leads_a_fallback_sub_legs_many_rows(
    tmp_path: Path,
) -> None:
    """THE AC#2 PIN -- RED on the pre-TASK-15700 round-robin.

    The notes sub-leg matches the full implicit AND (`and_then_or`'s
    primary) and is never widened; the media sub-leg finds zero AND rows,
    falls back to the OR form and returns four. Before the tiering, media
    interleaved FIRST, so a media FALLBACK row took leg rank 1 and the
    untouched notes row was demoted to leg rank 2 -- the displacement that
    cost `kw-plant-maintenance-record` its hybrid rescue and dropped scoped
    recall to 3/7.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            media and notes databases.
    """
    media_path = _seed_media(tmp_path, MEDIA_OR_ONLY_ROWS)
    notes_path = _seed_notes(tmp_path, [NOTE_AND_HIT])
    service = _make_service(
        construction=FTS_MATCH_CONSTRUCTION_AND_THEN_OR,
        media_db_path=media_path,
        chachanotes_db_path=notes_path,
    )

    results = asyncio.run(
        service._keyword_search(
            DISPLACEMENT_QUERY, top_k=10, keyword_source_types={"media", "note"}
        )
    )

    # The fixture must actually reproduce the shape: one untouched primary
    # row, several fallback rows. Otherwise the pin passes vacuously.
    stamps = [r.metadata["fts_match"] for r in results]
    assert stamps.count(FTS_MATCH_AND) == 1, stamps
    assert stamps.count(FTS_MATCH_OR) >= 2, stamps

    assert results[0].metadata["source_type"] == "note", [
        (r.metadata["source_type"], r.metadata["fts_match"]) for r in results
    ]
    assert results[0].metadata["doc_title"] == "Saltmarsh hide"
    assert results[0].metadata["fts_match"] == FTS_MATCH_AND
    # ...and every fallback row is behind it, not merely one of them.
    assert stamps[0] == FTS_MATCH_AND
    assert set(stamps[1:]) == {FTS_MATCH_OR}, stamps


def test_a_prefix_fallback_sub_leg_tiers_behind_a_primary_sub_leg(
    tmp_path: Path,
) -> None:
    """THE AC#2 PIN's shape under TASK-15700's own new construction.

    `and_then_prefix` is the composition the arc pre-registered on the
    strength of BOTH protections: the AND primary preserves every current
    hit inside a sub-leg, and Part A's tiering keeps the widened rows from
    re-ranking the untouched ones across sub-legs. The second half is only
    true if the tier partition reads the FORM the fallback actually ran --
    it is asserted here against real prefix rows rather than against the
    monkeypatched stand-in the stamp pin uses.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            media and notes databases.
    """
    media_path = _seed_media(tmp_path, MEDIA_PREFIX_ONLY_ROWS)
    notes_path = _seed_notes(tmp_path, [PREFIX_NOTE_AND_HIT])
    service = _make_service(
        construction=FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX,
        media_db_path=media_path,
        chachanotes_db_path=notes_path,
    )

    results = asyncio.run(
        service._keyword_search(
            PREFIX_DISPLACEMENT_QUERY, top_k=10, keyword_source_types={"media", "note"}
        )
    )

    # The fixture must reproduce the shape or the pin passes vacuously: one
    # untouched primary row, several prefix-fallback rows.
    stamps = [r.metadata["fts_match"] for r in results]
    assert stamps.count(FTS_MATCH_AND) == 1, stamps
    assert stamps.count(FTS_MATCH_PREFIX) >= 2, stamps
    assert FTS_MATCH_OR not in stamps, stamps

    assert results[0].metadata["source_type"] == "note", [
        (r.metadata["source_type"], r.metadata["fts_match"]) for r in results
    ]
    assert results[0].metadata["doc_title"] == "Saltmarsh hide"
    assert stamps[0] == FTS_MATCH_AND
    assert set(stamps[1:]) == {FTS_MATCH_PREFIX}, stamps


# --- (b) all-primary byte-identity: the shipped default cannot move ---


@pytest.mark.parametrize(
    "construction",
    [
        FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM,
        FTS_MATCH_CONSTRUCTION_OR,
        FTS_MATCH_CONSTRUCTION_AND,
        FTS_MATCH_CONSTRUCTION_PREFIX,
    ],
)
def test_all_primary_constructions_merge_byte_identically_to_a_plain_interleave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, construction: str
) -> None:
    """No fallback exists, so every sub-leg is tier 1 and NOTHING moves.

    Compared against a direct, unpartitioned `interleave_rankings` call over
    the real gathered rankings, by OBJECT IDENTITY -- so this pins the
    shipped default's leg order byte for byte, not a re-derivation of it.
    Inverting the tier order reds this.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            media, notes and prompts databases.
        monkeypatch: pytest's monkeypatch fixture; installs the sub-leg
            capture.
        construction: The all-primary construction under test.
    """
    media_path = _seed_media(
        tmp_path,
        [
            ("Wombat burrow survey", "Notes on the wombat burrow entrance survey."),
            ("Wombat rescue log", "A wombat rescue log entry from the salt flats."),
            ("Wombat census", "The annual wombat census for the reserve."),
        ],
    )
    notes_path = _seed_notes(
        tmp_path, [("Saltmarsh hide", "The hide overlooks the wombat burrow.")]
    )
    prompts_path = _seed_prompts(
        tmp_path,
        [("Wombat shift handover", "Summarise the wombat burrow inspection log.")],
    )
    service = _make_service(
        construction=construction,
        media_db_path=media_path,
        chachanotes_db_path=notes_path,
        prompts_db_path=prompts_path,
    )
    captured = _capture_sublegs(monkeypatch)

    top_k = 10
    results = asyncio.run(service._keyword_search("wombat burrow", top_k=top_k))

    rankings = _gathered_rankings(captured)
    assert len(rankings) >= 3, "the fixture must exercise several sub-legs"
    expected = interleave_rankings(rankings, key=_fusion_doc_key)[:top_k]

    assert len(results) == len(expected)
    for position, (got, want) in enumerate(zip(results, expected)):
        assert got is want, (
            f"position {position} moved under {construction!r}: "
            f"{_fusion_doc_key(got)} != {_fusion_doc_key(want)}"
        )
    # The premise of the identity: nothing fell back.
    assert {r.metadata["fts_match"] for r in results} == {
        service._fts5_primary_form()
    }


# --- (c) rank-fairness BETWEEN primaries is kept (no overreach) -------------


def test_rank_fairness_between_two_primary_sub_legs_is_unchanged(
    tmp_path: Path,
) -> None:
    """Many-vs-one between two PRIMARY sub-legs still round-robins.

    Rank-fairness among primaries is correct behaviour -- raw FTS5 scores
    are not comparable across sources, so rank position is the only
    cross-source signal there is. Only the FALLBACK case was ever the
    defect, and this pin is what stops the fix from spreading into the
    case it was not measured against.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            media and notes databases.
    """
    media_path = _seed_media(
        tmp_path,
        [
            ("Wombat burrow survey", "Notes on the wombat burrow entrance survey."),
            ("Wombat burrow census", "The annual wombat burrow census."),
            ("Wombat burrow rescue", "A wombat burrow rescue log entry."),
        ],
    )
    notes_path = _seed_notes(
        tmp_path, [("Saltmarsh hide", "The hide overlooks the wombat burrow.")]
    )
    service = _make_service(
        construction=FTS_MATCH_CONSTRUCTION_AND_THEN_OR,
        media_db_path=media_path,
        chachanotes_db_path=notes_path,
    )

    results = asyncio.run(
        service._keyword_search(
            "wombat burrow", top_k=10, keyword_source_types={"media", "note"}
        )
    )

    # Both sub-legs matched the AND primary: one tier, round-robin intact.
    assert {r.metadata["fts_match"] for r in results} == {FTS_MATCH_AND}
    assert [r.metadata["source_type"] for r in results] == [
        "media",
        "note",
        "media",
        "media",
    ], [(r.metadata["source_type"], r.metadata["doc_title"]) for r in results]


# --- (d) tier 2 FILLS, never displaces -------------------------------------


def test_tier_two_rows_never_take_a_slot_tier_one_wanted(tmp_path: Path) -> None:
    """Tier 1 holding >= top_k rows leaves ZERO fallback rows in the output.

    The merge truncates AFTER concatenating the two tiers, so a fallback row
    can only ever appear in a slot tier 1 left empty. Inverting the tier
    order reds this.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            media and notes databases.
    """
    media_path = _seed_media(tmp_path, MEDIA_OR_ONLY_ROWS)
    notes_path = _seed_notes(
        tmp_path,
        [
            NOTE_AND_HIT,
            (
                "Saltmarsh rota",
                "How does the wombat template work for the dawn rota?",
            ),
            (
                "Saltmarsh handover",
                "How does the wombat template work at handover time?",
            ),
        ],
    )
    service = _make_service(
        construction=FTS_MATCH_CONSTRUCTION_AND_THEN_OR,
        media_db_path=media_path,
        chachanotes_db_path=notes_path,
    )

    results = asyncio.run(
        service._keyword_search(
            DISPLACEMENT_QUERY, top_k=2, keyword_source_types={"media", "note"}
        )
    )

    assert len(results) == 2
    assert [r.metadata["source_type"] for r in results] == ["note", "note"]
    assert {r.metadata["fts_match"] for r in results} == {FTS_MATCH_AND}


# --- (e) no tier 2 without a fallback (structural) --------------------------


@pytest.mark.parametrize(
    "construction, expected_form",
    [
        (FTS_MATCH_CONSTRUCTION_AND, FTS_MATCH_AND),
        (FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM, FTS_MATCH_AND),
        (FTS_MATCH_CONSTRUCTION_OR, FTS_MATCH_OR),
        (FTS_MATCH_CONSTRUCTION_AND_THEN_OR, FTS_MATCH_AND),
        (FTS_MATCH_CONSTRUCTION_PREFIX, FTS_MATCH_PREFIX),
        (FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX, FTS_MATCH_AND),
    ],
)
def test_primary_form_is_one_definition_per_construction(
    construction: str, expected_form: str
) -> None:
    """`_fts5_primary_form` names the form a construction's PRIMARY runs.

    One definition, because the tier partition and (from Task 2) the
    sweep's negative-composition counter must not disagree about which form
    is "the primary" for a construction. `or` is the one construction whose
    primary IS the OR form -- reading fallback-ness off the stamp alone
    would put every `or` row in tier 2.

    Args:
        construction: The construction under test.
        expected_form: The `FTS_MATCH_*` form its primary expression runs.
    """
    service = _make_service(construction=construction)
    assert service._fts5_primary_form() == expected_form


def test_an_unknown_construction_takes_the_conservative_primary_form() -> None:
    """An unrecognized construction degrades to `and` -- form included.

    `_resolved_fts_match_construction` fail-safes to `"and"`, so the primary
    form must follow it rather than being computed off the raw config value;
    otherwise the partition would tier rows against a construction the leg
    never ran.
    """
    service = _make_service(construction="not-a-construction")
    assert service._fts5_primary_form() == FTS_MATCH_AND


def test_every_construction_names_the_forms_it_can_run() -> None:
    """`FTS_MATCH_FORMS_BY_CONSTRUCTION` is total, and honest both ways.

    Every construction has an entry, a construction that can produce a
    fallback EXPRESSION names a fallback FORM, and one that cannot names
    ``None``. This is the pin a new construction trips if it adds an
    expression without adding its form -- the omission the hardcoded
    fallback stamp used to hide (`_fts_rows_with_fallback` stamped
    ``FTS_MATCH_OR`` whatever form actually ran).
    """
    assert set(FTS_MATCH_FORMS_BY_CONSTRUCTION) == set(FTS_MATCH_CONSTRUCTIONS)

    probes = (
        "wombat",
        "wombat burrow",
        "how does the wombat template work",
        "notes about the vendor",
        "what about the",
    )
    for construction in FTS_MATCH_CONSTRUCTIONS:
        service = _make_service(construction=construction)
        _primary_form, fallback_form = service._fts5_match_forms()
        can_fall_back = any(
            service._fts5_match_expressions(query)[1] is not None
            for query in probes
        )
        assert can_fall_back == (fallback_form is not None), (
            f"{construction!r}: produces a fallback expression="
            f"{can_fall_back}, names a fallback form={fallback_form!r}"
        )


def test_fallback_rows_are_stamped_with_the_fallbacks_own_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback stamp is DERIVED, not hardcoded to the OR form.

    Today every fallback the engine ships happens to BE the OR form, so no
    fixture can tell a derived stamp from a hardcoded one by its value
    alone. This drives the seam instead: the construction is made to declare
    a fallback form no construction currently uses (standing in for the
    pre-registered `and_then_prefix` row), and the rows the fallback returns
    must carry THAT form. A hardcoded `FTS_MATCH_OR` in the fallback branch
    stamps `"or"` here and reds -- which is exactly the defect that would
    have made the sweep's negative-composition table attribute prefix rows
    to the or-form.

    The tiering is asserted alongside it: a novel fallback form is still
    not the primary, so its rows still land in tier 2.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            media and notes databases.
        monkeypatch: pytest's monkeypatch fixture; declares the novel
            fallback form.
    """
    novel_form = "prefix"
    monkeypatch.setattr(
        RAGService,
        "_fts5_match_forms",
        lambda self: (FTS_MATCH_AND, novel_form),
    )

    media_path = _seed_media(tmp_path, MEDIA_OR_ONLY_ROWS)
    notes_path = _seed_notes(tmp_path, [NOTE_AND_HIT])
    service = _make_service(
        construction=FTS_MATCH_CONSTRUCTION_AND_THEN_OR,
        media_db_path=media_path,
        chachanotes_db_path=notes_path,
    )

    results = asyncio.run(
        service._keyword_search(
            DISPLACEMENT_QUERY, top_k=10, keyword_source_types={"media", "note"}
        )
    )

    stamps = [
        (r.metadata["source_type"], r.metadata["fts_match"]) for r in results
    ]
    assert (
        "note",
        FTS_MATCH_AND,
    ) in stamps, stamps
    media_stamps = {form for source, form in stamps if source == "media"}
    assert media_stamps == {novel_form}, stamps
    assert FTS_MATCH_OR not in {form for _source, form in stamps}, stamps

    # ...and a novel fallback form is still not the primary, so tier 2.
    assert stamps[0] == ("note", FTS_MATCH_AND), stamps
    assert {form for _source, form in stamps[1:]} == {novel_form}, stamps


def test_an_unstamped_row_lands_in_tier_one_under_every_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing stamp must never DEMOTE a row -- the fail-safe's direction.

    `_keyword_row_metadata` substitutes a default before `_keyword_search`'s
    partition can see the absence, so that default is the whole fail-safe.
    It used to be a hardcoded `FTS_MATCH_AND`: probed under the `or`
    construction, where the primary form is `or`, an unstamped row therefore
    compared UNEQUAL to the primary and was demoted to tier 2 -- the exact
    opposite of the stated fail-safe, and live the moment a construction
    ships whose primary is not the AND form.

    Driven through a sub-leg helper that returns rows without the raw stamp
    (the shape a new sub-leg author actually produces), not by editing
    metadata after the fact.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            media and notes databases.
        monkeypatch: pytest's monkeypatch fixture; strips the media
            sub-leg's raw stamp.
    """
    original = RAGService._perform_fts5_search

    def unstamped(self, pool, query, limit, allowed_ids=None):
        rows = original(self, pool, query, limit, allowed_ids)
        for row in rows:
            row.pop("fts_match", None)
        return rows

    monkeypatch.setattr(RAGService, "_perform_fts5_search", unstamped)

    media_path = _seed_media(
        tmp_path,
        [
            ("Wombat burrow survey", "Notes on the wombat burrow entrance survey."),
            ("Wombat burrow census", "The annual wombat burrow census."),
        ],
    )
    notes_path = _seed_notes(
        tmp_path, [("Saltmarsh hide", "The hide overlooks the wombat burrow.")]
    )
    service = _make_service(
        construction=FTS_MATCH_CONSTRUCTION_OR,
        media_db_path=media_path,
        chachanotes_db_path=notes_path,
    )

    results = asyncio.run(
        service._keyword_search(
            "wombat burrow", top_k=10, keyword_source_types={"media", "note"}
        )
    )

    # The media sub-leg arrived unstamped; the notes sub-leg carries the
    # real `or` primary stamp. One tier, so the round robin is intact and
    # media (first in source order) still leads. A hardcoded `and` default
    # tiers the unstamped media rows BEHIND the notes row and reds this.
    assert [r.metadata["source_type"] for r in results] == [
        "media",
        "note",
        "media",
    ], [(r.metadata["source_type"], r.metadata["fts_match"]) for r in results]
    assert {r.metadata["fts_match"] for r in results} == {
        service._fts5_primary_form()
    }


@pytest.mark.parametrize(
    "construction",
    [
        FTS_MATCH_CONSTRUCTION_AND,
        FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM,
        FTS_MATCH_CONSTRUCTION_OR,
        FTS_MATCH_CONSTRUCTION_PREFIX,
    ],
)
def test_a_construction_without_a_fallback_can_never_produce_a_tier_two_row(
    tmp_path: Path, construction: str
) -> None:
    """Structural: no fallback expression exists, so no second form can be.

    Asserted at BOTH ends -- `_fts5_match_expressions` returns no fallback
    for any query shape (so no sub-leg can run a second form), and a live
    multi-sub-leg query's rows all carry the primary form. Together these
    are why the shipped default's merge is a single tier by construction
    rather than by luck.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            media and notes databases.
        construction: The fallback-free construction under test.
    """
    service = _make_service(construction=construction)
    for query in (
        "wombat",
        "wombat burrow",
        "how does the wombat template work",
        "notes about the vendor",
        "what about the",
    ):
        assert service._fts5_match_expressions(query)[1] is None, query

    media_path = _seed_media(tmp_path, MEDIA_OR_ONLY_ROWS)
    notes_path = _seed_notes(tmp_path, [NOTE_AND_HIT])
    live = _make_service(
        construction=construction,
        media_db_path=media_path,
        chachanotes_db_path=notes_path,
    )
    results = asyncio.run(
        live._keyword_search(
            DISPLACEMENT_QUERY, top_k=10, keyword_source_types={"media", "note"}
        )
    )
    assert results, "the fixture must return rows for this pin to mean anything"
    assert {r.metadata["fts_match"] for r in results} == {
        live._fts5_primary_form()
    }
