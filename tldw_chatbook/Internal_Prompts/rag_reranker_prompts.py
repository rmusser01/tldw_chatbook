# tldw_chatbook/Internal_Prompts/rag_reranker_prompts.py
"""LLM-reranker prompt specs. Defaults moved from RAG_Search/reranker.py
__init__ fallbacks; templates converted from .format ({{) to final
single-brace form. Parity tests compare rendered output against the
original .format results. Rerankers snapshot config at construction, so
overrides apply on the next search (applies="next search")."""

from .catalog import PromptSpec, register

_APPLIES = "next search"

register(
    PromptSpec(
        id="rag_reranker.pointwise_system",
        subsystem="rag_reranker",
        title="Pointwise reranker — system",
        description="System prompt for scoring one search result at a time.",
        used_in="RAG search LLM reranking (PointwiseReranker)",
        default="""You are a search result relevance evaluator. 
Your task is to score how relevant a search result is to a given query.
Return only a JSON object with a 'score' field (0.0 to 1.0) and optionally a 'reasoning' field.
Higher scores indicate better relevance.""",
        contract_note="Model must return JSON with a numeric 'score' field.",
        applies=_APPLIES,
    )
)

register(
    PromptSpec(
        id="rag_reranker.pointwise_template",
        subsystem="rag_reranker",
        title="Pointwise reranker — scoring template",
        description="Per-result scoring prompt.",
        used_in="RAG search LLM reranking (PointwiseReranker)",
        default="""Query: {query}

Search Result:
Title: {title}
Content: {content}

How relevant is this search result to the query? Consider:
1. Direct answer to the query
2. Topical relevance
3. Information quality
4. Completeness

Return JSON: {"score": 0.0-1.0{reasoning}}""",
        required_placeholders=("query", "title", "content", "reasoning"),
        contract_note=(
            "Output is parsed as JSON with a numeric 'score'; {reasoning} "
            "expands to the optional reasoning JSON fragment."
        ),
        applies=_APPLIES,
    )
)

register(
    PromptSpec(
        id="rag_reranker.pairwise_system",
        subsystem="rag_reranker",
        title="Pairwise reranker — system",
        description="System prompt for choosing the better of two results.",
        used_in="RAG search LLM reranking (PairwiseReranker)",
        default="""You are a search result comparator.
Given a query and two search results, determine which one is more relevant.
Return only a JSON object with 'choice' (1 or 2) and optionally 'reasoning'.""",
        contract_note="Model must return JSON with 'choice' of 1 or 2.",
        applies=_APPLIES,
    )
)

register(
    PromptSpec(
        id="rag_reranker.pairwise_template",
        subsystem="rag_reranker",
        title="Pairwise reranker — comparison template",
        description="Two-result comparison prompt.",
        used_in="RAG search LLM reranking (PairwiseReranker)",
        default="""Query: {query}

Result 1:
Title: {title1}
Content: {content1}

Result 2:
Title: {title2}
Content: {content2}

Which result better answers the query? Consider relevance, accuracy, and completeness.

Return JSON: {"choice": 1 or 2{reasoning}}""",
        required_placeholders=(
            "query",
            "title1",
            "content1",
            "title2",
            "content2",
            "reasoning",
        ),
        contract_note="Output is parsed as JSON with 'choice' of 1 or 2.",
        applies=_APPLIES,
    )
)

register(
    PromptSpec(
        id="rag_reranker.listwise_system",
        subsystem="rag_reranker",
        title="Listwise reranker — system",
        description="System prompt for reordering a result list.",
        used_in="RAG search LLM reranking (ListwiseReranker)",
        default="""You are a search result ranker.
Given a query and a list of search results, reorder them by relevance.
Return a JSON object with 'ranking' as an array of result indices in order of relevance.""",
        contract_note="Model must return JSON with a 'ranking' index array.",
        applies=_APPLIES,
    )
)

register(
    PromptSpec(
        id="rag_reranker.listwise_template",
        subsystem="rag_reranker",
        title="Listwise reranker — ranking template",
        description="Whole-list reordering prompt.",
        used_in="RAG search LLM reranking (ListwiseReranker)",
        default="""Query: {query}

Search Results:
{results_list}

Reorder these results by relevance to the query (most relevant first).
Return the indices in order.

Return JSON: {"ranking": [indices in order]{reasoning}}""",
        required_placeholders=("query", "results_list", "reasoning"),
        contract_note="Output is parsed as JSON with a 'ranking' index array.",
        applies=_APPLIES,
    )
)
