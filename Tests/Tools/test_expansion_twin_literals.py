"""TASK-16688 AC#1: the expansion arc's two TWIN LITERALS, pinned.

TASK-16174's final review (finding 5) found two pairs of independent
literals with identical content and nothing keeping them that way:

* ``document_expansion_tool.SUPPORTED_SOURCE_TYPES`` (the seams the tool
  can fetch) and ``library_expand_policy.EXPANDABLE_SOURCE_TYPES`` (the
  seams the policy hints). Drift in either direction reproduces exactly
  the failure the policy exists to prevent -- the policy hinting a row the
  tool answers ``unsupported``, or withholding a hint for a seam the tool
  could have opened.
* ``document_expansion_tool.PROMPT_BODY_COLUMNS`` and
  ``rag_service.PROMPT_DOCUMENT_COLUMNS`` -- the three prompt BODY columns
  rendered as a prompt's document. Lower stakes, same shape: the agent
  would read back a different rendering than the one that was indexed.

**Equality pins, deliberately not an import.** Each side has a recorded
reason to stay decoupled: the tool (``Tools/``) is constructed by the
Settings-side gate enumerator purely to read its description, so importing
``rag_service`` would drag the embeddings/vector stack into that path, and
the policy (``Library/``) is a pure, dependency-free helper. A test that
reds on drift is the whole requirement; the in-file reasons for not
importing stay as they are.
"""

from __future__ import annotations

from tldw_chatbook.Library.library_expand_policy import EXPANDABLE_SOURCE_TYPES
from tldw_chatbook.RAG_Search.simplified.rag_service import PROMPT_DOCUMENT_COLUMNS
from tldw_chatbook.Tools.document_expansion_tool import (
    PROMPT_BODY_COLUMNS,
    SUPPORTED_SOURCE_TYPES,
)


def test_source_type_allowlists_cannot_drift():
    """The seams the tool fetches and the seams the policy hints are one set.

    Set equality, not tuple equality: order carries no meaning on either
    side (both are membership tests), so pinning it would fail a harmless
    reordering and teach the next reader to edit the test rather than
    think about the drift.
    """
    assert set(SUPPORTED_SOURCE_TYPES) == set(EXPANDABLE_SOURCE_TYPES), (
        "document_expansion_tool.SUPPORTED_SOURCE_TYPES and "
        "library_expand_policy.EXPANDABLE_SOURCE_TYPES have drifted: the "
        "policy would hint a row the tool cannot fetch, or withhold a hint "
        "for a seam the tool can open (TASK-16174 finding 5)."
    )


def test_prompt_body_columns_match_rag_service():
    """ORDER matters here: the rendering is these columns joined in order.

    ``rag_service._prompt_document_text`` joins the non-empty columns with
    ``"\\n\\n"`` in ``PROMPT_DOCUMENT_COLUMNS`` order, and the tool renders
    a prompt the same way. A reordering alone would hand the agent a
    document that is not the one that was chunked and indexed.
    """
    assert tuple(PROMPT_BODY_COLUMNS) == tuple(PROMPT_DOCUMENT_COLUMNS), (
        "document_expansion_tool.PROMPT_BODY_COLUMNS mirrors "
        "rag_service.PROMPT_DOCUMENT_COLUMNS including ORDER (the join rule "
        "depends on it); they have drifted (TASK-16174 finding 5)."
    )
