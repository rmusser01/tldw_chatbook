"""The engine keyword leg's MATCH construction seam (TASK-15400, Tasks 1+4).

`_escape_fts5_query` builds an implicit AND over every query token, which
is why the leg returned zero rows for 40 of the 60 golden queries (the
census in TASK-15400's description). This file pins the SEAM that let the
arc's sweep measure four candidate constructions against each other.

**The default flipped in Task 4 (2026-08-11), and these pins moved with
it.** `SearchConfig.fts_match_construction` was `"and"` (the pre-arc
implicit AND over every token) and became `"and_stopword_trim"`, the
construction the sweep's matrix chose under the arc's pre-registered rule
(row `and_trim`: leg census 20 → 21 of 53, hybrid prompt/recall
0.000 → 0.200, no gated cell down in any mode, zero extra FTS queries; the
two OR-bearing rows scored 28/29 and were disqualified for losing the
vector-blind fixture's hybrid rescue — see `_escape_fts5_query`'s docstring
for the interleave-displacement mechanism).

**TASK-15700 Part B (2026-08-13) added two more constructions**, the rows
its re-run of the sweep pre-registers: `prefix` (the content tokens as
PREFIX terms, promoting the 15400 sweep's report-only probe) and
`and_then_prefix` (the AND primary with a prefix fallback). Their pins live
in their own section below; the properties they extend are the same four.

**AND TASK-15700 Task 4 (2026-08-13) FLIPPED THE DEFAULT AGAIN**, to
`"and_then_prefix"` — **by OWNER RULING, not by the pre-registered rule's
output.** The rule was applied verbatim: it disqualified the census-maximal
`and_then_or` on constraint (b), tied `prefix` and `and_then_prefix` at
census 23 (measurement-identical on every captured axis), and its tie-break
— fewest extra FTS statements, 240 vs 460 — selected `prefix`. The owner
overrode that tie-break, applying the standing stability-over-quick-wins
ruling to a dimension it predates: `prefix` widens as the PRIMARY form and
can self-displace inside one bm25-limited sub-leg, where the tiered merge
protects nothing. Never describe the shipped value as the sweep's winner.

Three pins here name both defaults explicitly:
`test_the_shipped_default_is_the_owner_ruled_construction` (the flip and
the ruling), `test_the_shipped_construction_runs_one_fallback_per_zero_row_
sub_leg` (the flip's accepted PRICE — this pin previously asserted the
exact opposite) and
`test_and_construction_is_byte_identical_to_the_shipped_escaper` (same
property, asked for explicitly). Every OTHER pin here sets its construction
by hand and is unaffected by which one ships.

**One fact worth carrying, because it is the easiest thing to get wrong
about the current default:** `and_then_prefix`'s PRIMARY is the FULL AND
(every token, function words included) — NOT the trimmed AND. The trim only
appears in its per-sub-leg fallback. So the shipped default is not a
superset of the previous one by construction; that it lost nothing is a
MEASURED result on the golden corpus, not a structural guarantee.

Four properties, each with a mutation that reds it:

* **Expression shape per construction.** `_fts5_match_expressions` returns
  `(primary, fallback | None)`. `and` is byte-identical to
  `_escape_fts5_query`; `and_stopword_trim` ANDs the content tokens (and
  falls back to the FULL AND when trimming empties the query, never to an
  empty MATCH expression -- an FTS5 syntax error); `or` ORs the content
  tokens and returns `""` (= "no rows", the existing skip contract) when
  trimming empties them; `and_then_or` returns both forms; `prefix` stars
  the content tokens (`""` on an empty trim, as `or`) and `and_then_prefix`
  returns the full AND with that prefix form as its fallback.
* **The fallback fires ONLY on zero primary rows.** Counted with a spy on
  the SQL-executing helper, per sub-leg: one AND row means exactly one
  query. Dropping the zero-row guard (running the fallback unconditionally)
  reds the count assertions; dropping the fallback loop entirely reds the
  rescue assertions.
* **Provenance.** Every keyword row carries `metadata["fts_match"]`, naming
  the FORM that matched it and nothing else: `"and"` for an implicit-AND
  expression (full or stopword-trimmed), `"or"` for the content-token OR
  form -- whether that form was reached as `and_then_or`'s fallback or run
  as the `or` construction's primary -- and `"prefix"` likewise for the two
  prefix-bearing constructions. Task 2's negative-composition counter
  and Task 5's mechanism prose both read this key, so it must name the form,
  not the position: under the `or` construction NO row is a fallback row,
  and fallback-ness is derivable from (construction, form) whenever it is
  wanted.
* **The construction is in the cache key.** A per-service cache plus a
  runtime-mutable construction is exactly the shape that made TASK-4110's
  fusion sweep report "k doesn't matter". `"and"` renders the key
  byte-identically to the pre-arc rendering (reproduced literally below,
  the way `test_no_allowlist_keeps_the_legacy_key_byte_identical` does it);
  every other construction renders a different key.

Real databases throughout for the behavioural half (a real
`PromptsDatabase`/`CharactersRAGDB`/`MediaDatabase`, their own writers
maintaining the FTS indexes), because the thing under test is FTS5's own
parse of the expression -- a mock would pin nothing.
"""
import asyncio
import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.rag_service import (
    FTS_MATCH_AND,
    FTS_MATCH_CONSTRUCTION_AND,
    FTS_MATCH_CONSTRUCTION_AND_STOPWORD_TRIM,
    FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX,
    FTS_MATCH_CONSTRUCTION_PREFIX,
    FTS_MATCH_CONSTRUCTIONS,
    FTS_MATCH_OR,
    FTS_MATCH_PREFIX,
    _FTS5_STOPWORDS,
    RAGService,
)
from tldw_chatbook.RAG_Search.simplified.simple_cache import SimpleRAGCache

CLIENT_ID = "test_fts_match_construction"


# --- fixtures ---------------------------------------------------------------


def _make_service(
    construction=None,
    media_db_path=None,
    chachanotes_db_path=None,
    prompts_db_path=None,
):
    """A RAGService with the in-memory vector store and mock embeddings."""
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


def _seed_prompts(tmp_path, rows, name="prompts.db"):
    """A real PromptsDatabase (its writer maintains `prompts_fts`)."""
    db_path = tmp_path / name
    db = PromptsDatabase(db_path, client_id=CLIENT_ID)
    ids = {}
    try:
        for prompt_name, system_prompt in rows:
            prompt_id, _uuid, message = db.add_prompt(
                name=prompt_name,
                author=None,
                details=None,
                system_prompt=system_prompt,
            )
            assert prompt_id is not None, f"prompt seed failed: {message}"
            ids[prompt_name] = prompt_id
    finally:
        db.close_connection()
    return db_path, ids


def _seed_media(tmp_path, rows, name="tldw_cli_media_v2.db"):
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
    db_path = tmp_path / name
    db = CharactersRAGDB(db_path, client_id=CLIENT_ID)
    try:
        for title, content in rows:
            db.add_note(title=title, content=content)
    finally:
        db.close_connection()
    return db_path


def _seed_conversation(tmp_path, title, messages, name="chachanotes.db"):
    """A real conversation with real messages (`messages_fts` is the index).

    Same shape as `test_keyword_leg_chacha._add_conversation`.
    """
    db_path = tmp_path / name
    db = CharactersRAGDB(db_path, client_id=CLIENT_ID)
    try:
        conv_id = db.add_conversation({"title": title})
        assert conv_id, "conversation seed failed"
        for sender, content in messages:
            assert db.add_message(
                {"conversation_id": conv_id, "sender": sender, "content": content}
            ), "message seed failed"
    finally:
        db.close_connection()
    return db_path, conv_id


@contextmanager
def _captured_warnings():
    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


# The seed prompt matches "wombat"/"burrow" but carries neither "template"
# nor "checklist" -- so an AND over the natural-language query is empty
# while the content-token OR still finds it. That gap IS the defect
# TASK-15400 measured, reproduced at fixture scale.
PROMPT_ROWS = [
    (
        "Wombat shift handover",
        "Summarise the wombat burrow inspection log for the incoming supervisor.",
    ),
    (
        "Kiln firing schedule",
        "Draft the ceramic kiln firing schedule for the studio.",
    ),
]
AND_HIT_QUERY = "wombat burrow"                      # every token present
OR_ONLY_QUERY = "how does the wombat template work"  # "template"/"work" absent
# The PREFIX mechanism at fixture scale (TASK-15700): the prompt says
# "inspection", the query says "inspect". Both tokens are content words, so
# no stopword trim reaches it and the OR form would only find it by dropping
# the second token entirely -- while `"wombat"* "inspect"*` matches it as the
# implicit AND it is. This is the shape behind the 15400 probe's 3 rescues.
PREFIX_ONLY_QUERY = "wombat inspect"


# --- expression shape -------------------------------------------------------


def test_the_shipped_default_is_the_owner_ruled_construction(
    tmp_path: Path,
) -> None:
    """DISCLOSED ORACLE FLIP (2026-08-13, TASK-15700 Task 4): the default was
    `"and_stopword_trim"` and is now `"and_then_prefix"`.

    Both states named on purpose, and so is the REASON, because this flip is
    the one place in the suite where the shipped value is NOT the
    pre-registered rule's own output:

    * `"and_stopword_trim"` was TASK-15400's measured winner (census 20 → 21
      of 53) and shipped 2026-08-11 → 2026-08-13.
    * TASK-15700 re-ran the sweep as SIX rows under the form-tiered merge.
      The rule, applied verbatim, disqualified the census-maximal
      `and_then_or` (29) on constraint (b), then TIED `prefix` and
      `and_then_prefix` at census 23 — measurement-identical on every
      captured axis (all 105 gated cells unmoved, all 60 hybrid top-10s and
      all 60 keyword-leg top-10s identical, same rescues, `lost` 0) — and
      its tie-break (fewest extra FTS statements, 240 vs 460) selected
      **`prefix`**.
    * **The OWNER RULED `and_then_prefix` ships instead**, applying the
      standing stability-over-quick-wins ruling to a dimension the tie-break
      predates: `prefix` widens as the PRIMARY form and self-displaces
      inside a single bm25-limited sub-leg (where tiering can protect
      nothing), while `and_then_prefix` never widens a NON-EMPTY AND primary
      and confines widening rows to tier 2. Price: 220 extra SQLite
      statements on this corpus, zero measured retrieval difference.

    So this pin is deliberately NOT named "the sweep's winner" — it was, up
    to 2026-08-13, and renaming it is part of the disclosure. A default
    reverted to `"and_stopword_trim"` reds this assertion and the census pin
    in `Tests/RAG_Eval/test_fusion_decision_rule.py`.

    Args:
        tmp_path: pytest's per-test temporary directory; unused by this
            pin (the service under test uses the in-memory vector store
            and no on-disk databases), kept for parity with the sibling
            construction pins in this file.
    """
    service = _make_service()
    assert (
        service.config.search.fts_match_construction
        == FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX
    )

    # ...and the SHAPE at the DEFAULT, not merely the name. Read this pair
    # carefully, because it is the flip's most easily mis-stated fact: the
    # PRIMARY is the FULL AND -- every token, function words INCLUDED -- and
    # NOT the trimmed AND the outgoing default ran. The trim now lives only
    # in the FALLBACK, which fires per sub-leg on a zero-row primary.
    # Consequence, recorded rather than left to be rediscovered: the new
    # default is NOT a superset of the old one by construction (a sub-leg
    # whose full AND returns rows never seeks the trim-only hits). That it
    # loses nothing is a MEASURED fact on the golden corpus (TASK-15700
    # Task 3: `lost` 0 against both the control and the shipped row), never
    # a structural guarantee.
    assert service._fts5_match_expressions("notes about the vendor") == (
        '"notes" "about" "the" "vendor"',
        '"notes"* "vendor"*',
    )


def test_and_construction_is_byte_identical_to_the_shipped_escaper(
    tmp_path: Path,
) -> None:
    """`and` still produces exactly the pre-arc MATCH expression.

    DISCLOSED (2026-08-11): this used to be measured at the DEFAULT
    construction, which was `"and"`; it now asks for `"and"` explicitly. The
    property is unchanged and still load-bearing — `"and"` is what
    `_resolved_fts_match_construction` degrades an unknown value to, and
    what `and_stopword_trim` itself falls back to when trimming empties the
    query, so the byte-identity is a live path, not a historical one.

    Args:
        tmp_path: pytest's per-test temporary directory; unused by this
            pin (the service under test uses the in-memory vector store
            and no on-disk databases), kept for parity with the sibling
            construction pins in this file.
    """
    service = _make_service(construction=FTS_MATCH_CONSTRUCTION_AND)

    for query in ("wombat", "Obsidian-3 spindle runout", "notes about the vendor"):
        primary, fallback = service._fts5_match_expressions(query)
        assert primary == service._escape_fts5_query(query), query
        assert fallback is None, query


def test_and_stopword_trim_ands_only_the_content_tokens():
    """`pm-vendor-chaser`'s shape: one function word blocks the whole AND."""
    service = _make_service(construction="and_stopword_trim")

    primary, fallback = service._fts5_match_expressions("notes about the vendor")
    assert primary == '"notes" "vendor"'
    assert fallback is None


def test_and_stopword_trim_falls_back_to_the_full_and_when_trimming_empties():
    """An all-stopword query must never produce an empty MATCH expression."""
    service = _make_service(construction="and_stopword_trim")

    primary, fallback = service._fts5_match_expressions("what about the")
    assert primary == '"what" "about" "the"'
    assert fallback is None


def test_or_construction_ors_the_content_tokens():
    """Quoted tokens joined by ` OR ` -- FTS5's own disjunction syntax."""
    service = _make_service(construction="or")

    primary, fallback = service._fts5_match_expressions("notes about the vendor")
    assert primary == '"notes" OR "vendor"'
    assert fallback is None


def test_or_construction_returns_no_rows_when_trimming_empties_the_query():
    """Honest emptiness, never a syntax error: `""` is the existing "skip".

    A raw OR over every token would match every document containing "the";
    trimming is what keeps the OR form from flooding fusion with junk. When
    trimming leaves nothing, the answer is no rows.
    """
    service = _make_service(construction="or")

    assert service._fts5_match_expressions("what about the") == ("", None)


def test_and_then_or_returns_the_and_primary_and_the_or_fallback():
    """The data's own suggestion: widen only where AND returns nothing."""
    service = _make_service(construction="and_then_or")

    primary, fallback = service._fts5_match_expressions("notes about the vendor")
    assert primary == '"notes" "about" "the" "vendor"'
    assert fallback == '"notes" OR "vendor"'


def test_and_then_or_has_no_fallback_when_every_token_is_a_stopword():
    """Nothing to widen to -- and an empty MATCH expression is a syntax error."""
    service = _make_service(construction="and_then_or")

    assert service._fts5_match_expressions("what about the") == (
        '"what" "about" "the"',
        None,
    )


def test_and_then_or_has_no_redundant_fallback_when_the_forms_are_identical():
    """A one-token OR IS the one-token AND: re-running it costs a query and
    can only return the same zero rows."""
    service = _make_service(construction="and_then_or")

    assert service._fts5_match_expressions("wombat") == ('"wombat"', None)

    # Stopwords AROUND a single content token are a different case: the
    # trimmed OR is strictly weaker than the AND (it drops the function
    # words the AND still demands), so the fallback is real work --
    # `pm-vendor-chaser`'s exact shape, the one query the census measured as
    # blocked solely by a function word.
    assert service._fts5_match_expressions("what about the wombat") == (
        '"what" "about" "the" "wombat"',
        '"wombat"',
    )


def test_a_repeated_token_is_still_one_term_and_suppresses_the_fallback():
    """Suppression is decided on FTS5 TERMS, not on expression strings.

    `"wombat" "wombat"` and `"wombat" OR "wombat"` are the same single-term
    query to FTS5, so widening is impossible -- but the two expression
    STRINGS differ, and a string comparison would spend one extra FTS query
    per zero-row sub-leg on it (and inflate the sweep's extra-query
    tie-break). Case and trailing punctuation fold the same way.
    """
    service = _make_service(construction="and_then_or")

    assert service._fts5_match_expressions("wombat wombat") == (
        '"wombat" "wombat"',
        None,
    )
    assert service._fts5_match_expressions("Wombat wombat,") == (
        '"Wombat" "wombat,"',
        None,
    )
    # Two DISTINCT content terms still widen -- suppression is not a
    # blanket "single OR term" rule.
    assert service._fts5_match_expressions("wombat burrow")[1] == (
        '"wombat" OR "burrow"'
    )


# --- the prefix constructions (TASK-15700 Part B) --------------------------
#
# Two pre-registered rows, added for the 15700 re-run. `prefix` is the
# construction the 15400 sweep's report-only probe measured a 3-rescue lead
# on; `and_then_prefix` is the composition (AND primary, prefix fallback)
# that gets BOTH protections -- every current AND hit preserved by
# construction, and its widening rows confined to tier 2 by Part A's merge.
#
# THE PROVENANCE RULE: `prefix` must build the expression the probe built,
# byte for byte, or the lead it is being promoted on was measured on a
# different query. The probe (`Tests/RAG_Eval/harness/fusion_sweep.py`,
# `prefix_probe_expression`) SPACE-joins one `"tok"*` per CONTENT token --
# an implicit AND over prefix terms, not a disjunction -- and returns ""
# when trimming empties the token list. The byte-identity is pinned at the
# probe's own site (`test_fusion_decision_rule.py`); the shape is pinned
# here on the probe's own example query.


def test_prefix_construction_stars_each_content_token_outside_the_quotes():
    """The probe's form: implicit AND over prefix terms, star OUTSIDE.

    `"tok"*` is "a phrase whose last token is a prefix"; the star INSIDE the
    quotes is part of the literal string and matches nothing (FTS5's own
    tokenizer drops it), which would silently reduce this construction to
    the trimmed AND. The expression below is character-for-character the one
    `prefix_probe_expression` pins for the same query.
    """
    service = _make_service(construction=FTS_MATCH_CONSTRUCTION_PREFIX)

    primary, fallback = service._fts5_match_expressions("the shift log")
    assert primary == '"shift"* "log"*'
    assert fallback is None


def test_prefix_construction_returns_no_rows_when_trimming_empties_the_query():
    """A stopword prefix (`"the"*`) is junk that matches most of the corpus.

    So the content-token trim is not optional here, and when it leaves
    nothing the answer is honestly no rows -- `""`, the existing skip
    contract -- exactly as under `or`. The probe returns `""` for this query
    too.
    """
    service = _make_service(construction=FTS_MATCH_CONSTRUCTION_PREFIX)

    assert service._fts5_match_expressions("what about the") == ("", None)


def test_and_then_prefix_returns_the_and_primary_and_the_prefix_fallback():
    """The composition: no AND hit is widened, and zero-row sub-legs widen."""
    service = _make_service(construction=FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX)

    primary, fallback = service._fts5_match_expressions("notes about the vendor")
    assert primary == '"notes" "about" "the" "vendor"'
    assert fallback == '"notes"* "vendor"*'


def test_and_then_prefix_has_no_fallback_when_every_token_is_a_stopword():
    """Nothing to widen TO: a stopword-only prefix form is junk, not a query."""
    service = _make_service(construction=FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX)

    assert service._fts5_match_expressions("what about the") == (
        '"what" "about" "the"',
        None,
    )


def test_and_then_prefix_never_suppresses_a_fallback_on_an_identical_term_set():
    """THE SUPPRESSION PIN -- a prefix fallback is ALWAYS wider.

    `and_then_or`'s suppression asks "do both forms reduce to the same
    single FTS5 term?", because an OR over one term IS that term and
    re-running it can only return the same zero rows. That reasoning does
    NOT transfer: `"wombat"*` and `"wombat"` are the same TERM SET and
    different QUERIES -- the prefix form matches every word starting with
    it. Copying the suppression across would silence the fallback on exactly
    the single-content-token queries the probe's rescues came from.

    Suppression fires here only when the prefix expression is EMPTY (the
    all-stopword case above), which is the "nothing to widen to" case rather
    than a term-set comparison.
    """
    service = _make_service(construction=FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX)

    assert service._fts5_match_expressions("wombat") == ('"wombat"', '"wombat"*')
    # ...including the repeated-token shape `and_then_or` folds to one term.
    assert service._fts5_match_expressions("wombat wombat") == (
        '"wombat" "wombat"',
        '"wombat"* "wombat"*',
    )
    assert service._fts5_match_expressions("Wombat wombat,") == (
        '"Wombat" "wombat,"',
        '"Wombat"* "wombat,"*',
    )

    # The contrast that makes the pin discriminating: the OR composition
    # suppresses every one of those, because for IT the two forms really are
    # the same query.
    or_service = _make_service(construction="and_then_or")
    assert or_service._fts5_match_expressions("wombat") == ('"wombat"', None)
    assert or_service._fts5_match_expressions("wombat wombat")[1] is None


def test_the_prefix_form_really_is_wider_than_the_and_over_the_same_terms():
    """The suppression pin's premise, against real FTS5.

    "Semantically wider" is a measurable claim, not a slogan: the same
    single term matches strictly more documents with the star than without.
    If this ever stopped being true, suppressing the fallback would be
    correct and the pin above would be the wrong rule.
    """
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE VIRTUAL TABLE docs USING fts5(title, content)")
        conn.execute(
            "INSERT INTO docs(title, content) VALUES (?, ?)",
            ("Burrow survey", "The wombats were counted at dusk."),
        )
        conn.commit()

        def match(expression):
            return conn.execute(
                "SELECT rowid FROM docs WHERE docs MATCH ?", (expression,)
            ).fetchall()

        service = _make_service(
            construction=FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX
        )
        primary, fallback = service._fts5_match_expressions("wombat")

        assert match(primary) == [], "the AND form must NOT reach 'wombats'"
        assert match(fallback) == [(1,)], "the prefix form must"


def test_a_zero_row_and_falls_back_to_the_prefix_form_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prefix mechanism end to end: one extra query, one rescue.

    The seeded prompt says "inspection"; the query says "inspect". No
    stopword list and no disjunction reaches that -- the prefix form does,
    which is the mechanism behind the probe's 3 rescues.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            prompts database.
        monkeypatch: pytest's monkeypatch fixture; installs the
            `_prompts_fts` call-counting spy.
    """
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(
        construction=FTS_MATCH_CONSTRUCTION_AND_THEN_PREFIX,
        prompts_db_path=db_path,
    )
    calls = _prompts_fts_spy(monkeypatch)

    results = asyncio.run(
        service._keyword_search(
            PREFIX_ONLY_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Wombat shift handover"]
    assert len(calls) == 2, f"expected primary + ONE fallback, got {calls}"
    assert calls[0] == '"wombat" "inspect"'
    assert calls[1] == '"wombat"* "inspect"*'
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_PREFIX]


def test_the_prefix_construction_stamps_its_rows_as_the_prefix_form(
    tmp_path: Path,
) -> None:
    """Under `prefix` the prefix expression IS the primary -- and the stamp
    still names the FORM, so the sweep's negative-composition counter reads
    the same value under both prefix-bearing constructions and lets the
    construction column say which of them was a fallback.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            prompts database.
    """
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(
        construction=FTS_MATCH_CONSTRUCTION_PREFIX, prompts_db_path=db_path
    )

    results = asyncio.run(
        service._keyword_search(
            PREFIX_ONLY_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Wombat shift handover"]
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_PREFIX]


def test_stopword_trimming_is_case_and_punctuation_insensitive():
    """FTS5 reads `About,` as the term `about`; so must the trimmer."""
    service = _make_service(construction="or")

    primary, _ = service._fts5_match_expressions("About, THE vendor's chaser")
    assert primary == '"vendor\'s" OR "chaser"'


def test_stopword_list_is_lowercase_and_covers_the_measured_blocker():
    """`pm-vendor-chaser` is the one golden query blocked solely by "about"."""
    assert "about" in _FTS5_STOPWORDS
    assert all(word == word.lower() for word in _FTS5_STOPWORDS)
    # A small fixed list of function words -- not a content-word vocabulary.
    # The EXACT size is pinned, not a range: Task 5's prose quotes this
    # number, and a range let an off-by-one claim ("66") survive a review.
    assert len(_FTS5_STOPWORDS) == 67


def test_every_token_stays_individually_quoted_in_every_construction():
    """The load-bearing injection property (TASK-3995), across the seam."""
    query = 'Obsidian-3 content:foo about the "quoted'
    for construction in FTS_MATCH_CONSTRUCTIONS:
        service = _make_service(construction=construction)
        primary, fallback = service._fts5_match_expressions(query)
        for expression in (primary, fallback):
            if not expression:
                continue
            # Tokens never contain whitespace, so stripping the OR
            # operators and splitting on spaces yields exactly the terms --
            # every one of which must be a quoted literal. A prefix term is
            # that same quoted literal with FTS5's star appended OUTSIDE the
            # closing quote (TASK-15700); stripping one trailing star is
            # therefore part of reading the term, and a star moved INSIDE
            # the quotes leaves the term looking quoted here but reds the
            # expression-shape and behavioural pins above.
            terms = expression.replace(" OR ", " ").split(" ")
            bare = [
                term
                for term in terms
                if not (
                    term.removesuffix("*").startswith('"')
                    and term.removesuffix("*").endswith('"')
                )
            ]
            assert bare == [], (
                f"{construction}: unquoted term(s) {bare} in {expression!r}"
            )


def test_an_invalid_construction_warns_once_and_behaves_as_and():
    """Fail-safe to the FULL AND -- never a crash, never silence.

    DISCLOSED (2026-08-11, TASK-15400 Task 4): this used to read "fail-safe
    to the shipped behaviour", which was the same thing when `and` shipped.
    It is not any more — the fail-safe deliberately stays on the pre-arc
    full AND (the most conservative construction, the one that never widens
    a query) while the default moves. The assertion below is unchanged and
    is what pins that: an invalid value emits
    `'"notes" "about" "the" "vendor"'`, and never a widened form.

    RE-DISCLOSED (2026-08-13, TASK-15700 Task 4): the default moved a second
    time, to `and_then_prefix`. Note the assertion is now byte-identical to
    the shipped default's PRIMARY — the fail-safe and the default's primary
    are the same full AND — so what this pin still discriminates is the
    FALLBACK: an unrecognized value must emit no second expression at all,
    where the shipped default emits `'"notes"* "vendor"*'`. That is asserted
    explicitly below rather than left implied by the primary alone.
    """
    service = _make_service(construction="or_of_ands_probably")

    with _captured_warnings() as warnings:
        first = service._fts5_match_expressions("notes about the vendor")
        second = service._fts5_match_expressions("a different query entirely")

    assert first == ('"notes" "about" "the" "vendor"', None)
    assert second[1] is None
    # EVENT-ONLY contract (dev TASK-15103 Batch C, rebased in 2026-08-13):
    # resolver diagnostics are redacted — the warning is a fixed event and
    # never echoes the user-controlled value. Assert the event + the count,
    # not the value.
    matching = [m for m in warnings if "Unknown fts_match_construction" in m]
    assert len(matching) == 1, (
        f"an invalid construction must warn exactly once per service: {matching}"
    )


def test_a_non_str_construction_degrades_instead_of_crashing():
    """Qodo PR-1574: `SearchConfig` is built from an untyped dict
    (`SearchConfig(**search_data)` in config.py, reachable from
    user-editable profile JSON), so `fts_match_construction` can arrive as
    a list or dict rather than a string. Both are unhashable, and the
    warn-once dedup set's membership check (`construction not in
    self._warned_fts_constructions`) and `.add()` raised TypeError on
    them -- crashing hybrid AND keyword search instead of degrading to
    the conservative full AND. Neither shape should raise.
    """
    for bad_value in ([], {}):
        service = _make_service(construction=bad_value)
        assert (
            service._resolved_fts_match_construction() == FTS_MATCH_CONSTRUCTION_AND
        ), bad_value


def test_a_non_str_construction_warns_once_per_distinct_bad_value():
    """The dedup set now keys non-str values on a hashable surrogate
    (`f"{type(construction).__name__}:{construction!r}"`), so repeating the
    SAME bad shape stays silent on the second call, and a DIFFERENT bad
    shape -- a distinct surrogate key -- warns again."""
    service = _make_service(construction=[])

    with _captured_warnings() as warnings:
        first = service._resolved_fts_match_construction()
        second = service._resolved_fts_match_construction()

    assert first == FTS_MATCH_CONSTRUCTION_AND
    assert second == FTS_MATCH_CONSTRUCTION_AND
    # EVENT-ONLY contract (dev TASK-15103 Batch C): the fixed event, no
    # value echo — dedup is still per distinct surrogate key, so the count
    # is what discriminates.
    matching = [m for m in warnings if "Unknown fts_match_construction" in m]
    assert len(matching) == 1, (
        f"a repeated non-str bad value must warn exactly once: {matching}"
    )

    # A DIFFERENT bad shape -- {} instead of [] -- is a distinct surrogate
    # key, so it gets its own one-shot warning rather than staying silent.
    service.config.search.fts_match_construction = {}
    with _captured_warnings() as more_warnings:
        service._resolved_fts_match_construction()
    assert len(more_warnings) == 1, more_warnings


# --- the fallback loop: zero rows only, once, per sub-leg -------------------


def _prompts_fts_spy(monkeypatch):
    """Count the prompts sub-leg's SQL executions, per MATCH expression."""
    calls = []
    original = RAGService._prompts_fts

    def spy(conn, escaped_query, limit, allowed_ids=None):
        calls.append(escaped_query)
        return original(conn, escaped_query, limit, allowed_ids)

    monkeypatch.setattr(RAGService, "_prompts_fts", staticmethod(spy))
    return calls


def test_a_matching_and_never_runs_the_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`kw-plant-maintenance-record`'s protection, mechanically: a non-empty
    AND result is returned as-is and the OR form is never executed.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            prompts database.
        monkeypatch: pytest's monkeypatch fixture; installs the
            `_prompts_fts` call-counting spy.
    """
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(construction="and_then_or", prompts_db_path=db_path)
    calls = _prompts_fts_spy(monkeypatch)

    results = asyncio.run(
        service._keyword_search(
            AND_HIT_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Wombat shift handover"]
    assert len(calls) == 1, f"the fallback ran on a non-empty AND: {calls}"
    assert " OR " not in calls[0]
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_AND]


def test_a_zero_row_and_falls_back_to_the_or_form_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rescue: a natural-language query that finds nothing under AND
    reaches the prompt through the content-token OR -- one extra query.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            prompts database.
        monkeypatch: pytest's monkeypatch fixture; installs the
            `_prompts_fts` call-counting spy.
    """
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(construction="and_then_or", prompts_db_path=db_path)
    calls = _prompts_fts_spy(monkeypatch)

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Wombat shift handover"]
    assert len(calls) == 2, f"expected primary + ONE fallback, got {calls}"
    assert " OR " not in calls[0]
    assert calls[1] == '"wombat" OR "template" OR "work"'
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_OR]


def test_the_shipped_construction_runs_one_fallback_per_zero_row_sub_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DISCLOSED ORACLE FLIP (2026-08-13, TASK-15700 Task 4): this pin
    asserted the OPPOSITE, and the reversal is the cost the owner accepted.

    Both states, explicitly:

    * BEFORE — under `and_stopword_trim` this test was
      `test_the_shipped_and_construction_never_runs_a_second_query` and
      asserted `len(calls) == 1`: the shipped construction had no fallback
      form at all, so a zero-row sub-leg simply returned nothing.
    * AFTER — the shipped `and_then_prefix` defines a fallback, so a sub-leg
      whose primary returns zero rows runs a SECOND statement. Measured over
      the 60 golden queries: 460 statements against the old default's 240,
      i.e. **220 extra, 92% of sub-legs falling back** on the 172-document
      eval corpus (an upper bound — the fallback fires only where the
      AND primary found nothing, so a denser corpus hits it less).

    That extra work is precisely what the pre-registered tie-break weighed,
    and precisely what the OWNER RULING overrode in favour of structural
    immunity to intra-sub-leg self-displacement (see
    `test_the_shipped_default_is_the_owner_ruled_construction`). Pinning the
    count here keeps the price VISIBLE rather than letting it drift: if this
    ever reads 1, the default silently lost its fallback; if it reads more
    than 2 for one sub-leg, the zero-row guard broke and the fallback is
    running unconditionally.

    The query also shows why the prefix fallback is not a widening free
    lunch: it ANDs its prefix terms, so unlike `and_then_or`'s OR fallback
    (which answers this same query) it still returns nothing here.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            prompts database.
        monkeypatch: pytest's monkeypatch fixture; installs the
            `_prompts_fts` call-counting spy.
    """
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(prompts_db_path=db_path)
    calls = _prompts_fts_spy(monkeypatch)

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert results == []
    assert len(calls) == 2, (
        f"expected the shipped construction's primary + ONE fallback: {calls}"
    )
    # The primary is the FULL AND (function words included) -- NOT the
    # trimmed AND the previous default ran. The trim lives in the fallback.
    assert calls[0] == '"how" "does" "the" "wombat" "template" "work"'
    assert calls[1] == '"wombat"* "template"* "work"*'


def test_the_or_construction_stamps_its_rows_as_the_or_form(tmp_path: Path) -> None:
    """The stamp names the FORM, not the position: under `or` the OR
    expression IS the primary, and calling those rows `and` would make Task
    2's negative-composition counter read zero for the widest candidate.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            prompts database.
    """
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(construction="or", prompts_db_path=db_path)

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Wombat shift handover"]
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_OR]


def test_notes_sub_leg_falls_back_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop wraps every sub-leg's SQL helper, not just the prompts one.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            notes database.
        monkeypatch: pytest's monkeypatch fixture; installs the
            `_chacha_notes_fts` call-counting spy.
    """
    db_path = _seed_notes(
        tmp_path,
        [("Saltmarsh hide", "The hide overlooks the wombat burrow at dusk.")],
    )
    service = _make_service(construction="and_then_or", chachanotes_db_path=db_path)

    calls = []
    original = RAGService._chacha_notes_fts

    def spy(conn, escaped_query, limit, allowed_ids=None):
        calls.append(escaped_query)
        return original(conn, escaped_query, limit, allowed_ids)

    monkeypatch.setattr(RAGService, "_chacha_notes_fts", staticmethod(spy))

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"note"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Saltmarsh hide"]
    assert len(calls) == 2, calls
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_OR]


def test_conversations_sub_leg_falls_back_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conversations helper is the one that issues TWO MATCH statements
    off a single expression (the conversation ranking, then the matched
    message lines). Both must run on whichever expression actually matched,
    or the row comes back with an empty document.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            conversation database.
        monkeypatch: pytest's monkeypatch fixture; installs the
            `_chacha_conversations_fts` call-counting spy.
    """
    db_path, conv_id = _seed_conversation(
        tmp_path,
        "Burrow handover",
        [
            ("user", "Where did we leave the wombat burrow inspection?"),
            ("assistant", "The dusk pass is still outstanding."),
        ],
    )
    service = _make_service(construction="and_then_or", chachanotes_db_path=db_path)

    calls = []
    original = RAGService._chacha_conversations_fts

    def spy(conn, escaped_query, limit, allowed_ids=None):
        calls.append(escaped_query)
        return original(conn, escaped_query, limit, allowed_ids)

    monkeypatch.setattr(
        RAGService, "_chacha_conversations_fts", staticmethod(spy)
    )

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"conversation"}
        )
    )

    assert [r.metadata["source_id"] for r in results] == [str(conv_id)]
    assert len(calls) == 2, calls
    assert " OR " not in calls[0]
    assert calls[1] == '"wombat" OR "template" OR "work"'
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_OR]
    # The second statement ran on the FALLBACK expression too: without that,
    # the conversation ranks but carries no message text at all.
    assert "wombat burrow" in results[0].document.lower(), results[0].document


def test_media_sub_leg_falls_back_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same loop over the media sub-leg's pooled FTS5 execution.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            media database.
        monkeypatch: pytest's monkeypatch fixture; installs the
            `_perform_fts5_search` call-counting spy.
    """
    db_path = _seed_media(
        tmp_path,
        [("Burrow survey", "Notes on the wombat burrow entrance survey.")],
    )
    service = _make_service(construction="and_then_or", media_db_path=db_path)

    calls = []
    original = RAGService._perform_fts5_search

    def spy(self, pool, query, limit, allowed_ids=None):
        calls.append(query)
        return original(self, pool, query, limit, allowed_ids)

    monkeypatch.setattr(RAGService, "_perform_fts5_search", spy)

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"media"}
        )
    )

    assert [r.metadata["doc_title"] for r in results] == ["Burrow survey"]
    # ONE call into the pooled helper; the fallback happens inside it, so
    # the retry wrapper cannot multiply it.
    assert len(calls) == 1, calls
    assert [r.metadata["fts_match"] for r in results] == [FTS_MATCH_OR]


def test_sub_legs_carry_and_and_fallback_rows_in_one_query_primary_first(
    tmp_path: Path,
) -> None:
    """One query, two provenances -- and the primary one leads.

    Media matches every token (AND); the prompt only matches through the OR
    fallback. Both rows come back, each stamped with the form that found it
    -- which is what keeps Task 5's mechanism prose table-derived.

    RENAMED (TASK-15700, was
    `test_sub_legs_interleave_and_and_fallback_rows_in_one_query`): the two
    provenances are no longer INTERLEAVED. `_keyword_search` tiers them, so
    the ORDER is now load-bearing and this test asserts it instead of
    collapsing the results into an order-blind dict. Media happens to be
    first in the source order here, so the ordering assertion would pass
    under the old round-robin too -- it is a second witness for the tier
    order, not the primary one (`Tests/RAG_Search/test_keyword_leg_tiered_merge.py`
    owns that, with the sources in the order that discriminates).

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            media and prompts databases.
    """
    media_path = _seed_media(
        tmp_path,
        [("Template work log", "How does the wombat template work in practice?")],
    )
    prompts_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(
        construction="and_then_or",
        media_db_path=media_path,
        prompts_db_path=prompts_path,
    )

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=10, keyword_source_types={"media", "prompt"}
        )
    )

    stamps = {
        r.metadata["source_type"]: r.metadata["fts_match"] for r in results
    }
    assert stamps == {"media": FTS_MATCH_AND, "prompt": FTS_MATCH_OR}

    # The tier order, in sequence: every primary-form row precedes every
    # fallback row.
    forms = [r.metadata["fts_match"] for r in results]
    assert forms == [FTS_MATCH_AND, FTS_MATCH_OR], forms


def test_a_query_with_no_searchable_tokens_still_touches_no_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The early exit reads the CONSTRUCTION's primary expression, so the
    `or` construction's all-stopword emptiness short-circuits too.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            prompts database.
        monkeypatch: pytest's monkeypatch fixture; installs the
            `_prompts_fts` call-counting spy.
    """
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(construction="or", prompts_db_path=db_path)
    calls = _prompts_fts_spy(monkeypatch)

    results = asyncio.run(
        service._keyword_search(
            "what about the", top_k=5, keyword_source_types={"prompt"}
        )
    )

    assert results == []
    assert calls == [], f"an empty MATCH expression reached FTS5: {calls}"


def test_a_failing_fallback_degrades_the_sub_leg_like_the_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No new failure modes: the fallback inherits the degrade path.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            prompts database.
        monkeypatch: pytest's monkeypatch fixture; installs a `_prompts_fts`
            spy that raises on the OR-fallback expression.
    """
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    service = _make_service(construction="and_then_or", prompts_db_path=db_path)

    original = RAGService._prompts_fts

    def spy(conn, escaped_query, limit, allowed_ids=None):
        if " OR " in escaped_query:
            raise sqlite3.OperationalError("fallback exploded")
        return original(conn, escaped_query, limit, allowed_ids)

    monkeypatch.setattr(RAGService, "_prompts_fts", staticmethod(spy))

    results = asyncio.run(
        service._keyword_search(
            OR_ONLY_QUERY, top_k=5, keyword_source_types={"prompt"}
        )
    )
    assert results == []


# --- the cache key ----------------------------------------------------------


def _legacy_hash(key_str):
    """`_make_key`'s hashing tail, reproduced so "byte-identical" is checkable."""
    try:
        import xxhash

        return xxhash.xxh64(key_str.encode()).hexdigest()
    except ImportError:
        import hashlib

        return hashlib.md5(key_str.encode()).hexdigest()


def test_the_and_construction_keeps_the_hybrid_key_byte_identical():
    """The pre-arc rendering, reproduced literally (B1a's discipline)."""
    cache = SimpleRAGCache(enabled=True)

    legacy_parts = [
        "quokka",
        "hybrid",
        "10",
        json.dumps({}, sort_keys=True),
        "fusion:"
        + json.dumps(
            {"alpha": 0.7, "rrf_k": 5, "pool_multiplier": 2}, sort_keys=True
        ),
    ]
    assert cache._make_key(
        "quokka",
        "hybrid",
        10,
        None,
        None,
        None,
        (0.7, 5, 2),
        FTS_MATCH_AND,
    ) == _legacy_hash("|".join(legacy_parts))

    # ... and omitting the argument entirely is the same key again.
    assert cache._make_key(
        "quokka", "hybrid", 10, None, None, None, (0.7, 5, 2)
    ) == _legacy_hash("|".join(legacy_parts))


def test_each_construction_gets_its_own_hybrid_cache_key():
    """A per-service cache plus a sweep that mutates the construction is
    exactly the shape that reported "k doesn't matter" in TASK-4110."""
    cache = SimpleRAGCache(enabled=True)

    keys = {
        construction: cache._make_key(
            "quokka", "hybrid", 10, None, None, None, (0.7, 5, 2), construction
        )
        for construction in FTS_MATCH_CONSTRUCTIONS
    }
    assert len(set(keys.values())) == len(FTS_MATCH_CONSTRUCTIONS), keys


def test_the_two_prefix_constructions_key_distinctly_from_every_other_row():
    """TASK-15700's two new values, named rather than counted.

    The test above is a set-size check over `FTS_MATCH_CONSTRUCTIONS`, so it
    would still pass if a new construction were simply never added to that
    tuple. These two are asserted by name: a per-service cache plus a sweep
    that mutates the construction between rows is exactly the shape that
    reported "k doesn't matter" in TASK-4110, and the two new rows enter that
    sweep.
    """
    cache = SimpleRAGCache(enabled=True)

    def key(construction):
        return cache._make_key(
            "quokka", "hybrid", 10, None, None, None, (0.7, 5, 2), construction
        )

    assert key("prefix") != key(FTS_MATCH_AND)
    assert key("and_then_prefix") != key(FTS_MATCH_AND)
    assert key("prefix") != key("and_then_prefix")
    assert key("prefix") != key("and_then_or")
    assert key("and_then_prefix") != key("and_then_or")


def test_the_keyword_search_type_keys_the_construction_too():
    """A keyword-only search reads the same leg, so it needs the same key
    part -- and `"and"` still renders the pre-arc bytes."""
    cache = SimpleRAGCache(enabled=True)

    legacy = cache._make_key("quokka", "keyword", 10)
    assert cache._make_key(
        "quokka", "keyword", 10, None, None, None, None, FTS_MATCH_AND
    ) == legacy
    assert cache._make_key(
        "quokka", "keyword", 10, None, None, None, None, "and_then_or"
    ) != legacy


def test_the_search_path_passes_the_construction_into_the_cache_key(
    tmp_path: Path,
) -> None:
    """End to end: two searches identical except for the construction must
    not share a cached entry.

    Args:
        tmp_path: pytest's per-test temporary directory; holds the seeded
            prompts database.
    """
    db_path, _ids = _seed_prompts(tmp_path, PROMPT_ROWS)
    cfg = RAGConfig()
    cfg.embedding.model = "mock"
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None
    cfg.search.enable_cache = True
    cfg.search.prompts_db_path = Path(db_path)
    service = RAGService(cfg)

    first = asyncio.run(
        service.search(
            OR_ONLY_QUERY,
            top_k=5,
            search_type="keyword",
            keyword_source_types={"prompt"},
        )
    )
    assert first == []

    service.config.search.fts_match_construction = "and_then_or"
    second = asyncio.run(
        service.search(
            OR_ONLY_QUERY,
            top_k=5,
            search_type="keyword",
            keyword_source_types={"prompt"},
        )
    )
    assert [r.metadata["doc_title"] for r in second] == ["Wombat shift handover"], (
        "the second search was served the first construction's cached rows"
    )
