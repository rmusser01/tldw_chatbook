# library_search_engine.py (TASK-34904)
#########################################
# One engine for the search_library_*_page family.
#
# Media, Prompts, and Conversations carried three hand-copies of the same
# match-branch machinery: an exact-title branch, escaped-LIKE substring
# branches, optional tokenized-quoted FTS branches, a keyword-substring
# branch, per-row hit_N projection so matched_fields is evidence-based,
# COUNT + LIMIT/OFFSET pagination in one transaction, and substring
# matched_keywords attribution. Per ADR-177 this is a HELPER parameterized
# by a ``LibrarySearchSpec`` -- not a base class.
#
# Notes (``search_library_notes_page``) is a multi-selector AND-shape
# (lexical group ANDed with exact-keyword and folder selectors), so only
# its LEXICAL GROUP and hit attribution route through this module via
# ``build_lexical_group`` / ``matched_fields_from_hits``; the selector
# validation, AND assembly, and organization projection stay in the host.
#
# Branch order is fixed: exact, substrings, FTS (conditional), keyword
# last. Each branch contributes exactly one bind parameter. FTS input is
# tokenized (``\\w+``) and per-token quoted via
# ``Utils.fts5_match_forms.quote_fts5_token`` so wildcards and FTS
# operators in the user query match literally; LIKE input is escaped so
# ``%``/``_`` match literally.
####

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from tldw_chatbook.Utils.fts5_match_forms import quote_fts5_token

_FTS_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_FTS_TOKEN_LIMIT = 20
_KEYWORD_CAP = 20
_PREVIEW_CHARS = 241


def escape_library_like(value: str) -> str:
    """Escape LIKE wildcards so ``%``/``_`` in the query match literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def library_fts_query(raw_query: str) -> Optional[str]:
    """Tokenize + per-token quote a query into a safe FTS5 MATCH expression.

    Returns None when the query has no ``\\w+`` tokens (FTS branch skips).
    """
    tokens = _FTS_TOKEN_PATTERN.findall(raw_query)
    if not tokens:
        return None
    tokens = tokens[:_FTS_TOKEN_LIMIT]
    return " ".join(quote_fts5_token(token) for token in tokens)


@dataclass(frozen=True)
class LibrarySearchSpec:
    """Per-surface parameterization of the shared library page search.

    Attributes:
        table: The searched table (``Media`` / ``Prompts`` / ``conversations``).
        base_filter: Constant row-visibility predicate (``deleted = 0``).
        extra_where: Additional parameterless predicate ANDed after
            ``base_filter`` (e.g. conversations' validated archive clause).
        exact_column: Column for the exact-match branch and ranking probe.
        substring_branches: ``(branch_sql, field_name)`` pairs; each SQL
            fragment carries exactly one ``?`` bound to the escaped LIKE
            pattern. ``field_name`` is the matched_fields vocabulary entry.
        fts_branches: ``(branch_sql, field_name)`` pairs appended (in
            order) only when the query yields FTS tokens; one ``?`` each
            bound to the shared FTS match expression.
        keyword_branch: SQL fragment with one ``?`` bound to the LIKE
            pattern (keyword-substring selector over the link relation).
        page_columns: SQL column list for the page query. When
            ``preview_first`` is true the engine binds ``_PREVIEW_CHARS``
            ahead of the hit parameters (the caller's column list contains
            its own ``substr(..., 1, ?)`` preview expression).
        tiebreaker: Final ORDER BY key (``id DESC`` / ``rowid DESC``).
        fetch_keywords: ``(conn, ids) -> Dict[id, List[str]]`` -- the
            surface's existing keyword-fetch helper.
        make_item: ``(row, keywords_by_id) -> Dict`` -- the surface's
            existing item projector.
        fts_attribution: Optional ``(conn, row_id, fts_query) -> List[str]``
            invoked for rows whose ONLY match is FTS, to attribute the
            combined-FTS hit to concrete columns (Prompts' per-column
            probes).
        make_database_error: ``message -> Exception`` constructing the
            host's database error.
    """

    table: str
    base_filter: str
    exact_column: str
    substring_branches: Tuple[Tuple[str, str], ...]
    fts_branches: Tuple[Tuple[str, str], ...]
    keyword_branch: str
    page_columns: str
    tiebreaker: str
    fetch_keywords: Callable[[sqlite3.Connection, List[Any]], Dict[Any, List[str]]]
    make_item: Callable[[Any, Dict[Any, List[str]]], Dict[str, Any]]
    make_database_error: Callable[[str], Exception]
    error_prefix: str  # e.g. "Failed to search library media: "
    extra_where: str = ""
    preview_first: bool = False
    fts_attribution: Optional[Callable[[sqlite3.Connection, Any, str], List[str]]] = (
        None
    )
    log_label: str = "library search"


def build_lexical_group(
    exact_column: str,
    substring_branches: Sequence[Tuple[str, str]],
    fts_branches: Sequence[Tuple[str, str]],
    keyword_branch: str,
    query: str,
) -> Tuple[List[str], List[Any], Dict[int, List[str]], Optional[str]]:
    """Build the OR'd lexical branch group shared by every surface.

    Returns ``(branches, params, hit_index_to_fields, fts_query)`` where
    ``hit_index_to_fields`` maps each branch index to its matched_fields
    vocabulary entries (the exact branch and the title substring share the
    same entry, mirroring the originals' ``hit_0 or hit_1`` logic).
    """
    like_pattern = f"%{escape_library_like(query)}%"
    fts_query = library_fts_query(query)

    branches: List[str] = [f"LOWER({exact_column}) = LOWER(?)"]
    params: List[Any] = [query]
    hit_fields: Dict[int, List[str]] = {}
    exact_field = substring_branches[0][1] if substring_branches else exact_column
    hit_fields[0] = [exact_field]

    for index, (branch_sql, field_name) in enumerate(substring_branches, start=1):
        branches.append(branch_sql)
        params.append(like_pattern)
        hit_fields[index] = [field_name]

    if fts_query is not None:
        for branch_sql, field_name in fts_branches:
            branches.append(branch_sql)
            params.append(fts_query)
            if field_name is not None:
                # None means the FTS hit attributes nothing by itself;
                # the surface's fts_attribution probes name the columns.
                hit_fields[len(branches) - 1] = [field_name]

    branches.append(keyword_branch)
    params.append(like_pattern)
    hit_fields[len(branches) - 1] = ["keywords"]
    return branches, params, hit_fields, fts_query


def matched_fields_from_hits(row: Any, hit_fields: Dict[int, List[str]]) -> List[str]:
    """Project a page row's hit_N columns into sorted matched_fields."""
    fields = {
        name
        for index, names in hit_fields.items()
        if row[f"hit_{index}"]
        for name in names
    }
    return sorted(fields)


def library_search_page(
    db: Any,
    spec: LibrarySearchSpec,
    *,
    query: str,
    limit: int,
    offset: int,
) -> Dict[str, Any]:
    """Run one library page search (COUNT + page + attribution, one tx).

    Args:
        db: Host database instance; duck-types ``transaction()``.
        spec: Per-surface parameterization.
        query: The user query (escaped for LIKE, tokenized for FTS).
        limit: Page size.
        offset: Page offset.

    Returns:
        Dict with ``items`` (spec-projected rows plus ``matched_fields``
        and ``matched_keywords``) and ``total``.

    Raises:
        Exception: The host's database error on sqlite failures.
    """
    branches, params, hit_fields, fts_query = build_lexical_group(
        spec.exact_column,
        spec.substring_branches,
        spec.fts_branches,
        spec.keyword_branch,
        query,
    )
    where_clause = " OR ".join(f"({branch})" for branch in branches)
    hit_selects = ", ".join(
        f"({branch}) AS hit_{index}" for index, branch in enumerate(branches)
    )
    # The OR group must be parenthesized: without it AND binds only the
    # first branch and later branches would escape the row-visibility
    # filters (caught by the conversations soft-delete equivalence class).
    row_filter = " AND ".join(
        part
        for part in (spec.base_filter, spec.extra_where, f"({where_clause})")
        if part
    )

    fts_indexes = _fts_indexes(spec, library_fts_query(query))
    try:
        with db.transaction() as conn:
            count_cursor = conn.execute(
                f"SELECT COUNT(*) AS count FROM {spec.table} WHERE {row_filter}",
                params,
            )
            total = int(count_cursor.fetchone()["count"])

            page_sql = (
                f"SELECT {spec.page_columns}, {hit_selects} "
                f"FROM {spec.table} WHERE {row_filter} "
                f"ORDER BY (LOWER({spec.exact_column}) = LOWER(?)) DESC, "
                f"last_modified DESC, {spec.tiebreaker} "
                f"LIMIT ? OFFSET ?"
            )
            # Branch params bind twice: once for the hit_N projection,
            # once for the WHERE clause (plus the ORDER-BY exact probe).
            page_params: List[Any] = list(params) * 2 + [query, limit, offset]
            if spec.preview_first:
                page_params = [_PREVIEW_CHARS] + page_params
            page_cursor = conn.execute(page_sql, page_params)
            rows = page_cursor.fetchall()

            row_ids = [row["id"] for row in rows]
            keywords_by_id = spec.fetch_keywords(conn, row_ids) if row_ids else {}

            query_lower = query.lower()
            items: List[Dict[str, Any]] = []
            for row in rows:
                item = spec.make_item(row, keywords_by_id)
                item["matched_fields"] = matched_fields_from_hits(row, hit_fields)
                if spec.fts_attribution is not None and fts_query is not None:
                    fts_only = any(
                        row[f"hit_{index}"] for index in fts_indexes
                    ) and not any(
                        row[f"hit_{index}"]
                        for index in range(len(branches))
                        if index not in fts_indexes
                    )
                    if fts_only:
                        item["matched_fields"] = sorted(
                            set(item["matched_fields"])
                            | set(spec.fts_attribution(conn, row["id"], fts_query))
                        )
                item["matched_keywords"] = [
                    keyword
                    for keyword in keywords_by_id.get(row["id"], [])
                    if query_lower in keyword.lower()
                ][:_KEYWORD_CAP]
                items.append(item)

            return {"items": items, "total": total}
    except sqlite3.Error as e:
        logger.opt(exception=True).error(
            f"{spec.log_label} failed; exception_type={type(e).__name__}"
        )
        logging.debug(f"{spec.log_label} sqlite failure details suppressed")
        raise spec.make_database_error(f"{spec.error_prefix}{e}") from e


def _fts_indexes(spec: LibrarySearchSpec, fts_query: Optional[str]) -> List[int]:
    """Branch indexes of the FTS branches (given they were appended)."""
    if fts_query is None:
        return []
    start = 1 + len(spec.substring_branches)
    return list(range(start, start + len(spec.fts_branches)))
