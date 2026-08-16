"""Per-row expansion policy for Library retrieval rows (TASK-16174, Phase P).

The arc's premise is that a policy nothing consumes is inert surface. This
module's helper is the consumed half: it labels each retrieval row with
whether following it into its document (`expand_document`, Phase T) would
add anything, and why.

Label detection is deliberately a TRIPLE -- `provenance.source_type` +
empty `chunk_id` + the label snippet prefix -- never the label text alone:
a real media chunk whose text happens to open with "Matched media" is a
text-bearing row, and a user's note that quotes the label is not a media
row at all. All three must agree before a row is called label-only.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_expand_policy import (
    DEFAULT_SNIPPET_CAP,
    REASON_LABEL_ONLY,
    REASON_TEXT_BEARING,
    REASON_TRUNCATED_SNIPPET,
    expand_hint,
)


def _row(**overrides) -> dict:
    """A retrieval row shaped like `LibraryRagResultRow`'s mapping form."""
    row = {
        "source_id": "42",
        "chunk_id": "",
        "title": "Some source",
        "snippet": "",
        "provenance": {"source_type": "note"},
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------
# The plan's four policy branches
# --------------------------------------------------------------------------


def test_media_label_row_hint_is_label_only():
    """`library_local_rag_search_service._media_row`'s exact shape."""
    hint = expand_hint(
        _row(
            source_id="12",
            chunk_id="",
            snippet="Matched media · pdf",
            provenance={"source_type": "media"},
        )
    )

    assert hint == {"expandable": True, "reason": REASON_LABEL_ONLY}


def test_conversation_label_row_hint_is_label_only():
    """`_conversation_row`'s exact shape (count_noun-rendered message count)."""
    hint = expand_hint(
        _row(
            source_id="conv-uuid",
            chunk_id="",
            snippet="Matched conversation · 7 messages",
            provenance={"source_type": "conversation"},
        )
    )

    assert hint == {"expandable": True, "reason": REASON_LABEL_ONLY}


def test_text_bearing_note_row_hint_is_text_bearing():
    hint = expand_hint(
        _row(
            source_id="note-uuid",
            chunk_id="",
            snippet="The quarterly plan sets a 12% target for the EU region.",
            provenance={"source_type": "note"},
        )
    )

    assert hint == {"expandable": False, "reason": REASON_TEXT_BEARING}


def test_truncated_semantic_snippet_hint():
    """A snippet longer than the projection cap is cut mid-sentence."""
    long_snippet = ("The migration plan continues for a long while. " * 60)[
        : DEFAULT_SNIPPET_CAP + 120
    ]

    hint = expand_hint(
        _row(
            source_id="media_12",
            chunk_id="media_12_chunk_3",
            snippet=long_snippet,
            provenance={"source_type": "media", "chunk_start": 900},
        )
    )

    assert hint == {"expandable": True, "reason": REASON_TRUNCATED_SNIPPET}


# --------------------------------------------------------------------------
# Edge behaviour the branches above depend on
# --------------------------------------------------------------------------


def test_label_text_alone_does_not_make_a_note_row_label_only():
    """The triple, not the text: a note quoting the label stays text-bearing."""
    hint = expand_hint(
        _row(
            snippet="Matched media · pdf was the phrase in my inbox today.",
            provenance={"source_type": "note"},
        )
    )

    assert hint == {"expandable": False, "reason": REASON_TEXT_BEARING}


def test_media_chunk_row_opening_with_the_label_text_is_not_label_only():
    """A real media chunk carries a chunk_id, so the triple fails."""
    hint = expand_hint(
        _row(
            source_id="12",
            chunk_id="media_12_chunk_0",
            snippet="Matched media · pdf, the report's own opening line.",
            provenance={"source_type": "media"},
        )
    )

    assert hint == {"expandable": False, "reason": REASON_TEXT_BEARING}


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_row(provenance={}), id="no-source-type"),
        pytest.param(_row(provenance={"source_type": "widget"}), id="unknown-type"),
        pytest.param(_row(provenance=None), id="no-provenance"),
        pytest.param(_row(source_id=""), id="no-source-id"),
        pytest.param("not a row", id="not-a-mapping"),
    ],
)
def test_unexpandable_rows_get_no_hint(row):
    """No hint at all beats a hint the agent cannot act on."""
    assert expand_hint(row) is None


def test_snippet_cap_is_caller_supplied_so_it_cannot_drift():
    snippet = "x" * 300

    assert expand_hint(_row(snippet=snippet))["reason"] == REASON_TEXT_BEARING
    assert (
        expand_hint(_row(snippet=snippet), snippet_cap=200)["reason"]
        == REASON_TRUNCATED_SNIPPET
    )


def test_ellipsis_terminated_snippet_is_truncated():
    hint = expand_hint(_row(snippet="The report concludes that the region…"))

    assert hint == {"expandable": True, "reason": REASON_TRUNCATED_SNIPPET}


def test_hint_carries_no_identity_or_provenance():
    hint = expand_hint(
        _row(
            source_id="secret-id",
            snippet="Matched media · pdf",
            chunk_id="",
            provenance={"source_type": "media", "note_id": "secret-note"},
        )
    )

    assert set(hint) == {"expandable", "reason"}


def test_helper_does_not_mutate_the_row():
    row = _row(snippet="Matched media · pdf", provenance={"source_type": "media"})
    before = {"snippet": row["snippet"], "provenance": dict(row["provenance"])}

    expand_hint(row)

    assert row["snippet"] == before["snippet"]
    assert dict(row["provenance"]) == before["provenance"]


# --------------------------------------------------------------------------
# The other half of the wired policy: the tool's own description (Phase T)
# --------------------------------------------------------------------------


def test_tool_description_states_the_policy_verbatim():
    """The hint says WHICH row; the description says WHAT TO DO about it."""
    from tldw_chatbook.Tools.document_expansion_tool import ExpandDocumentTool

    assert (
        "Expand a retrieval hit into its document. Use when a high-ranked "
        "hit is label-only (media/conversation rows) or its snippet is "
        "truncated and the answer needs the content. Re-query instead if "
        "the hit itself looks irrelevant. Never expand the same source "
        "twice — reuse the earlier result."
    ) in " ".join(ExpandDocumentTool().description.split())


def test_description_carries_the_two_branches_no_code_enforces():
    """The spec named FOUR policy branches; the helper implements two.

    "budget exhausted -> no" and "repeat expansion of the same source -> no"
    need per-conversation agent-loop state (what has already been expanded,
    what context budget is left) that a stateless per-call tool does not
    have. They therefore ship as INSTRUCTION in the description and nothing
    enforces or measures them -- disclosed in TASK-16174's AC#3 rather than
    silently narrowed. This test is the only thing keeping them from
    evaporating entirely, so it pins both sentences verbatim.
    """
    from tldw_chatbook.Tools.document_expansion_tool import ExpandDocumentTool

    description = " ".join(ExpandDocumentTool().description.split())

    assert (
        "Never expand the same source twice — reuse the earlier result."
    ) in description
    assert (
        "Stop expanding once your remaining context budget is short — a "
        "window you cannot afford to read is spent for nothing."
    ) in description


# --------------------------------------------------------------------------
# TASK-16688 AC#2: the canonicalization-variant EXCLUSION is deliberate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    ["notes", "media_chunk", "conversations", "chat", "prompts"],
)
def test_canonicalization_variant_rows_get_no_hint(variant):
    """A VARIANT spelling gets no hint, and that is the recorded decision.

    `library_local_rag_search_service._SEMANTIC_SOURCE_TYPE_MAP` treats
    these five spellings as live raw provenance values, so a row could in
    principle carry one; `EXPANDABLE_SOURCE_TYPES` accepts only the four
    singulars. TASK-16174's final review (finding 6) asked whether to
    broaden the allowlist. TASK-16588's route probe then MEASURED the case
    -- 0 variant rows across all 340 rows on four (index x route) arms,
    with a committed positive control showing its detector fires on every
    variant and on no singular -- and today's indexer
    (`RAG_Search/ingestion_indexing.py`) stamps only singulars.

    Broadening on a measured zero would ship speculative surface, so the
    exclusion stands and this test is what makes it deliberate rather than
    accidental: it reds if someone widens the allowlist without revisiting
    the reasoning in `library_expand_policy`'s module docstring.
    """
    assert (
        expand_hint(
            _row(
                source_id="42",
                chunk_id="",
                snippet="Matched media · pdf",
                provenance={"source_type": variant},
            )
        )
        is None
    )


@pytest.mark.parametrize(
    "singular", ["note", "media", "conversation", "prompt"]
)
def test_singular_twin_of_each_variant_still_gets_a_hint(singular):
    """The control that makes the exclusion above a reading, not a tautology.

    Mirrors the probe's own positive control: the same helper, the same
    row, only the spelling differs -- so "no hint" for a variant is a
    statement about the allowlist and not about a malformed row.
    """
    assert (
        expand_hint(
            _row(
                source_id="42",
                chunk_id="",
                snippet="Matched media · pdf",
                provenance={"source_type": singular},
            )
        )
        is not None
    )
