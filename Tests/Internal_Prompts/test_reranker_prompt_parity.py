# Tests/Internal_Prompts/test_reranker_prompt_parity.py
"""Old .format() rendering vs new safe_substitute rendering must be
byte-identical. The double-braced templates below are verbatim copies of the
pre-migration reranker.py literals."""

from tldw_chatbook.Internal_Prompts import CATALOG, safe_substitute

ORIGINAL_POINTWISE_TEMPLATE = """Query: {query}

Search Result:
Title: {title}
Content: {content}

How relevant is this search result to the query? Consider:
1. Direct answer to the query
2. Topical relevance
3. Information quality
4. Completeness

Return JSON: {{"score": 0.0-1.0{reasoning}}}"""

ORIGINAL_PAIRWISE_TEMPLATE = """Query: {query}

Result 1:
Title: {title1}
Content: {content1}

Result 2:
Title: {title2}
Content: {content2}

Which result better answers the query? Consider relevance, accuracy, and completeness.

Return JSON: {{"choice": 1 or 2{reasoning}}}"""

ORIGINAL_LISTWISE_TEMPLATE = """Query: {query}

Search Results:
{results_list}

Reorder these results by relevance to the query (most relevant first).
Return the indices in order.

Return JSON: {{"ranking": [indices in order]{reasoning}}}"""


def test_pointwise_template_parity():
    # content deliberately contains a brace sequence ("{with braces}"). Both
    # old .format() and new safe_substitute() are single-pass over the
    # template string only — neither re-parses braces that arrive inside a
    # substituted value — so this does not raise and both must render it
    # through unchanged and identically.
    values = dict(
        query="what is rust",
        title="Rust book",
        content="Rust is a systems language {with braces}",
        reasoning=', "reasoning": "explanation"',
    )
    assert safe_substitute(
        CATALOG["rag_reranker.pointwise_template"].default, **values
    ) == ORIGINAL_POINTWISE_TEMPLATE.format(**values)


def test_pointwise_template_parity_without_reasoning():
    values = dict(query="q", title="t", content="c", reasoning="")
    assert safe_substitute(
        CATALOG["rag_reranker.pointwise_template"].default, **values
    ) == ORIGINAL_POINTWISE_TEMPLATE.format(**values)


def test_pairwise_template_parity():
    values = dict(
        query="best db",
        title1="Postgres",
        content1="c1",
        title2="SQLite",
        content2="c2",
        reasoning="",
    )
    assert safe_substitute(
        CATALOG["rag_reranker.pairwise_template"].default, **values
    ) == ORIGINAL_PAIRWISE_TEMPLATE.format(**values)


def test_listwise_template_parity():
    values = dict(
        query="q",
        results_list="0. Title: A\n   Content: a...",
        reasoning=', "reasoning": "explanation"',
    )
    assert safe_substitute(
        CATALOG["rag_reranker.listwise_template"].default, **values
    ) == ORIGINAL_LISTWISE_TEMPLATE.format(**values)


def test_substitute_tolerates_braces_in_values():
    # safe_substitute is single-pass over the declared {name} tokens only
    # (see test_pointwise_template_parity), so arbitrary braces inside a
    # substituted value pass through untouched — this is the whole point of
    # the new resolver, and worth pinning down directly rather than only via
    # the parity comparison above.
    out = safe_substitute(
        CATALOG["rag_reranker.pointwise_template"].default,
        query="q",
        title="t",
        content="has {curly} text",
        reasoning="",
    )
    assert "has {curly} text" in out


def test_system_prompts_match_source():
    # System prompts are verbatim moves; spot-check load-bearing lines.
    assert CATALOG["rag_reranker.pointwise_system"].default.startswith(
        "You are a search result relevance evaluator."
    )
    assert "'choice' (1 or 2)" in CATALOG["rag_reranker.pairwise_system"].default
    assert "'ranking' as an array" in CATALOG["rag_reranker.listwise_system"].default
