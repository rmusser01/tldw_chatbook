"""Deep-search citation verification (task-16331). Pins the matching ladder
ported from tldw_server's Claims_Extraction/guardrails verbatim-first design:
exact substring, then casefold/whitespace-normalized containment, then a
bounded token-level fuzzy fallback. Fakes live only at chat_api_call / the
tool's phase boundaries; the verification code itself runs real."""

from tldw_chatbook.Tools import web_tool_impls
from tldw_chatbook.Tools.web_tool_impls import web_deep_search
from tldw_chatbook.Web_Scraping import WebSearch_APIs
from tldw_chatbook.Web_Scraping.deep_search_citations import (
    match_quote_in_sources,
    summarize_for_footer,
    verify_citations,
)


# --- matching ladder -----------------------------------------------------------


def test_quote_match_exact_substring():
    out = match_quote_in_sources(
        "the ocean is blue",
        ["Sky looks grey.", "Studies show the ocean is blue year-round."],
    )
    assert (
        out["matched"] is True and out["level"] == "exact" and out["source_index"] == 1
    )


def test_quote_match_normalized_survives_case_and_whitespace():
    out = match_quote_in_sources(
        "The   OCEAN is\tblue", ["studies show the ocean is  blue year-round"]
    )
    assert out["matched"] is True and out["level"] == "normalized"


def test_quote_match_fuzzy_covers_minor_token_drift():
    out = match_quote_in_sources(
        "the results were highly significant across cohorts",
        ["In 2023 the results were significant across all tested cohorts."],
    )
    assert out["matched"] is True and out["level"] == "fuzzy"


def test_quote_match_no_match_for_unrelated_text():
    out = match_quote_in_sources(
        "quantum tunneling at absolute zero", ["The chef folded eggs into the batter."]
    )
    assert (
        out["matched"] is False and out["level"] is None and out["source_index"] is None
    )


def test_quote_match_empty_or_short_inputs_do_not_match():
    assert match_quote_in_sources("", ["text"])["matched"] is False
    assert match_quote_in_sources("quote", [])["matched"] is False


# --- verify_citations ----------------------------------------------------------

_EVIDENCE = [
    {
        "id": 1,
        "content": "sum one",
        "original_content": "The ocean is blue and vast.",
        "url": "https://1.example/",
    },
    {
        "id": 2,
        "content": "sum two",
        "original_content": "Fish live in water.",
        "url": "https://2.example/",
    },
]


def test_verify_resolves_markers_and_flags_unknown_ids_inline():
    out = verify_citations(
        "Fact one[1] and fact two[2] but also[99] something.", _EVIDENCE
    )
    assert out["markers_total"] == 3
    assert out["markers_resolved"] == 2
    assert out["unknown_marker_ids"] == [99]
    # flagged with an inline marker, never deleted
    assert "[99?]" in out["annotated_text"]
    assert "something." in out["annotated_text"]


def test_verify_checks_quoted_spans_against_original_content():
    answer = 'As one source puts it, "the ocean is blue and vast"[1].'
    out = verify_citations(answer, _EVIDENCE)
    assert out["quotes_checked"] == 1 and out["quotes_verified"] == 1
    assert out["quotes_misquoted"] == 0


def test_verify_flags_misquoted_spans_without_removing_them():
    answer = 'One paper claims "mars hosts a breathable atmosphere"[1].'
    out = verify_citations(answer, _EVIDENCE)
    assert out["quotes_checked"] == 1 and out["quotes_verified"] == 0
    assert out["quotes_misquoted"] == 1
    assert '"mars hosts a breathable atmosphere"' in out["annotated_text"]


def test_verify_ignores_short_quotes():
    out = verify_citations('He said "ok"[1].', _EVIDENCE)
    assert out["quotes_checked"] == 0


def test_verify_counts_uncited_sentences():
    out = verify_citations(
        "Cited sentence[1]. Uncited sentence. Also cited[2].", _EVIDENCE
    )
    assert out["uncited_sentences"] == 1


def test_verify_all_resolved_leaves_text_untouched():
    answer = "Fact one[1]. Fact two[2]."
    out = verify_citations(answer, _EVIDENCE)
    assert out["annotated_text"] == answer
    assert out["unknown_marker_ids"] == []
    assert out["markers_total"] == 2 and out["markers_resolved"] == 2


# --- footer summary ------------------------------------------------------------


def test_footer_summary_omits_zero_sections():
    note = summarize_for_footer(
        {
            "markers_total": 2,
            "markers_resolved": 2,
            "unknown_marker_ids": [],
            "quotes_checked": 0,
            "quotes_verified": 0,
            "quotes_misquoted": 0,
            "uncited_sentences": 0,
        }
    )
    assert note == "Citations: 2/2 resolved"


def test_footer_summary_names_unknown_and_misquotes_and_uncited():
    note = summarize_for_footer(
        {
            "markers_total": 3,
            "markers_resolved": 2,
            "unknown_marker_ids": [99],
            "quotes_checked": 2,
            "quotes_verified": 1,
            "quotes_misquoted": 1,
            "uncited_sentences": 4,
        }
    )
    assert "Citations: 2/3 resolved (1 unknown)" in note
    assert "quotes 1/2 verified (1 misquoted)" in note
    assert "4 uncited sentence(s)" in note


# --- pipeline integration ------------------------------------------------------


def test_aggregate_success_verifies_and_annotates_citations(monkeypatch):
    monkeypatch.setattr(
        WebSearch_APIs, "chat_api_call", lambda **kwargs: "Answer citing [1] and [7]."
    )
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib

    monkeypatch.setattr(
        Summarization_General_Lib, "analyze", lambda *a, **k: "chunk summary"
    )

    out = WebSearch_APIs.aggregate_results(
        {
            "1": {
                "content": "sum one",
                "original_content": "orig",
                "reasoning": "r1",
                "url": "https://one.example/",
                "title": "One",
            }
        },
        "q",
        [],
        "openai",
    )
    cv = out["citation_verification"]
    assert cv["markers_total"] == 2 and cv["markers_resolved"] == 1
    assert cv["unknown_marker_ids"] == [7]
    assert out["text"] == "Answer citing [1] and [7?]."


def test_aggregate_failure_branches_carry_no_verification(monkeypatch):
    monkeypatch.setattr(WebSearch_APIs, "chat_api_call", lambda **kwargs: None)
    failure = WebSearch_APIs.aggregate_results(
        {
            "1": {
                "content": "c",
                "original_content": "o",
                "reasoning": "r",
                "url": "https://one.example/",
                "title": "One",
            }
        },
        "q",
        [],
        "openai",
    )
    assert "citation_verification" not in failure
    empty = WebSearch_APIs.aggregate_results({}, "q", [], "openai")
    assert "citation_verification" not in empty


# --- tool footer integration ---------------------------------------------------


def _tool_env(monkeypatch, final_answer):
    monkeypatch.setattr(
        WebSearch_APIs,
        "generate_and_search",
        lambda q, p: {
            "web_search_results_dict": {
                "results": [{"title": "T", "url": "https://e.com/"}],
                "warnings": [],
            },
            "sub_query_dict": {"sub_questions": [], "main_goal": "q"},
        },
    )

    async def fake_aa(wsr, sqd, params, cancel_event=None):
        return {
            "final_answer": final_answer,
            "relevant_results": {"1": {}},
            "web_search_results_dict": wsr,
        }

    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", fake_aa)
    monkeypatch.setattr(
        web_tool_impls,
        "_deep_search_settings",
        lambda: dict(
            {
                "search_provider_default": "google",
                "relevance_analysis_llm": "openai",
                "final_answer_llm": "openai",
                "search_enable_subquery": False,
                "search_default_max_queries": 5,
                "search_result_max": 10,
                "relevance_llm_timeout_s": 30,
                "relevance_scrape_timeout_s": 30,
                "deep_search_timeout_s": 240,
            }
        ),
    )


def test_tool_footer_surfaces_citation_verification(monkeypatch):
    _tool_env(
        monkeypatch,
        {
            "text": "Deep answer [1] and [9?].",
            "evidence": [
                {
                    "id": 1,
                    "url": "https://e.com/",
                    "title": "T",
                    "content": "c",
                    "original_content": "o",
                    "reasoning": "r",
                    "chunk_index": 0,
                }
            ],
            "confidence": 0.78,
            "chunks": [{}],
            "citation_verification": {
                "markers_total": 2,
                "markers_resolved": 1,
                "unknown_marker_ids": [9],
                "quotes_checked": 1,
                "quotes_verified": 1,
                "quotes_misquoted": 0,
                "uncited_sentences": 2,
            },
        },
    )
    out = web_deep_search("what is love")
    assert "Citations: 1/2 resolved (1 unknown)" in out
    assert "quotes 1/1 verified" in out
    assert "2 uncited sentence(s)" in out


def test_tool_footer_silent_without_verification(monkeypatch):
    _tool_env(
        monkeypatch,
        {
            "text": "Deep answer [1].",
            "evidence": [
                {
                    "id": 1,
                    "url": "https://e.com/",
                    "title": "T",
                    "content": "c",
                    "original_content": "o",
                    "reasoning": "r",
                    "chunk_index": 0,
                }
            ],
            "confidence": 0.78,
            "chunks": [{}],
        },
    )
    out = web_deep_search("what is love")
    assert "Citations:" not in out


# --- per-claim detail (task-16325) ----------------------------------------------

def test_verify_extracts_sentence_level_claims_with_source_ids():
    answer = "Ice is less dense than water[1]. Paris is the capital of France[1][2]. Mars hosts cities[99]."
    out = verify_citations(answer, _EVIDENCE)
    claims = out["claims"]
    assert [c["claim_id"] for c in claims] == ["claim-1", "claim-2", "claim-3"]
    assert claims[0]["text"] == "Ice is less dense than water[1]."
    assert claims[0]["source_ids"] == [1]
    assert claims[0]["status"] == "supported"
    assert claims[1]["source_ids"] == [1, 2]
    assert claims[2]["source_ids"] == []
    assert claims[2]["unknown_marker_ids"] == [99]
    assert claims[2]["status"] == "unverified"


def test_verify_claims_carry_per_sentence_quote_verdicts():
    answer = 'One source says "the ocean is blue and vast"[1]. Another claims "mars hosts a breathable atmosphere"[1].'
    out = verify_citations(answer, _EVIDENCE)
    claims = {c["claim_id"]: c for c in out["claims"]}
    assert claims["claim-1"]["quotes_checked"] == 1
    assert claims["claim-1"]["quotes_verified"] == 1
    assert claims["claim-1"]["status"] == "supported"
    assert claims["claim-2"]["quotes_verified"] == 0
    assert claims["claim-2"]["status"] == "unverified"


def test_verify_claims_skip_uncited_sentences():
    out = verify_citations("Cited[1]. Uncited sentence.", _EVIDENCE)
    assert [c["text"] for c in out["claims"]] == ["Cited[1]."]


# --- Qodo remediation (task-16814) ------------------------------------------------

def test_unknown_marker_sentence_is_not_counted_uncited():
    out = verify_citations("Cited sentence[1]. Unknown citation attempt[99].", _EVIDENCE)
    assert out["uncited_sentences"] == 0  # both sentences ATTEMPTED citations
