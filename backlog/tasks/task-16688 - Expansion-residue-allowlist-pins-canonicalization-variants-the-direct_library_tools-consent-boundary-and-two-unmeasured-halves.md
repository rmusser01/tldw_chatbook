---
id: TASK-16688
title: >-
  Expansion residue: allowlist pins, canonicalization variants, the
  direct_library_tools consent boundary, and two unmeasured halves
status: To Do
assignee: []
created_date: '2026-08-15 20:20'
labels:
  - rag
  - agents
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16174's final whole-branch review returned SHIP-WITH-FIXES. Its three
blockers (a dead `chunk_id` parameter, the AC#3 four-branch narrowing, and a
>500-message conversation reported as a complete read) plus the pin and docs
findings were fixed in a bounded fix wave on
`feat/rag-16174-agentic-expansion`. Five findings were deliberately NOT
fixed there — each is either a correctness gap on a route that arc could not
measure, or a recorded-decision gap — and this task carries them so they do
not evaporate with the branch.

**(1) Finding 5 — two source-type allowlists with nothing pinning them
together.** `document_expansion_tool.SUPPORTED_SOURCE_TYPES` and
`library_expand_policy.EXPANDABLE_SOURCE_TYPES` are independent literals
with identical content today. Drift in either direction reproduces exactly
the failure Task 3b existed to prevent: the policy hinting a row the tool
answers `unsupported`, or withholding a hint for a seam the tool can open.
Same shape, lower stakes: `document_expansion_tool.PROMPT_BODY_COLUMNS`
mirrors `rag_service.PROMPT_DOCUMENT_COLUMNS` (identical today, including
the `"\n\n"` join and the strip/skip-empty rule) with a documented reason
for not importing it, and nothing pins that either.

**(2) Finding 6 — the policy allowlist is narrower than the codebase's own
canonicalization surface.** `_SEMANTIC_SOURCE_TYPE_MAP`
(`Library/library_local_rag_search_service.py`, mirrored in
`library_rag_state.py`) treats `media_chunk`, `chat` and the plural
spellings as live raw provenance values. `EXPANDABLE_SOURCE_TYPES` accepts
only the four singulars, so such a row gets **no hint and — by Task 3b's
shared precondition — no identity**, silently, while the UI's own "open this
row" path canonicalizes it fine and the tool would fetch it if handed
`source_type="media"`. Today's local indexer stamps only singulars
(`RAG_Search/ingestion_indexing.py`), which is why this is not urgent; it is
a semantic/hybrid-route gap of exactly TASK-16588's family, and those rows
are in-scope evidence for 16588's route measurement.

**(3) Finding 13 — the consent-boundary relationship with `[console]
direct_library_tools` is nowhere recorded.** The Library agent-tools spec
makes that toggle the Console consent boundary: OFF, the agent gets bounded
`search_library_rag` excerpts and no direct Library reads. `expand_document`
returns whole documents by **raw** backing id regardless of that setting,
under its own `[tools]` gate plus the per-call ask floor; ON, it duplicates
the 18 direct get-tools while bypassing their opaque public-ID codec. Two
mitigations are real and pinned (OFF by default; `risk_tags=("reads",)` →
`ask`, 7 approval rounds observed live), and the raw id already left the
adapter pre-arc as `result_id` — so the arc types an existing exposure
rather than inventing one. What is new is a read-by-raw-id primitive over
sequential media/prompt PKs once a user clicks "always allow". The task
file's own Description warned against "a THIRD overlapping surface"; this
overlap is discussed nowhere.

**(4) Finding 15 — the conversation fetch loads image BLOBs it never uses.**
`document_expansion_tool._fetch_conversation` takes
`get_messages_for_conversation`'s default `include_image_data=True`, so up
to 500 message images are read into memory to render text that uses only
`sender`/`content`. One-word fix (`include_image_data=False`, the task-260
precedent); left out of the fix wave because it is a performance change with
no test asserting the read cost.

**(5) Finding 16 — the live run never exercised the window/continuation
half of the contract.** Every target document in the Phase E run is
289–4,924 chars against `DEFAULT_MAX_CHARS = 8000`, so all 8 expansions
recorded `expand_truncated: false` and every call returned a whole document.
The measured claim is "expansion opens a label", not "expansion navigates a
long document"; the latter is unit-tested only (including, since the fix
wave, the `HARD_MAX_CHARS` cap and the >500-message conversation).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The tool's and the policy's source-type allowlists can no longer drift apart: one is derived from the other, or a test fails when they differ (finding 5); the same is done or explicitly declined, with a reason, for PROMPT_BODY_COLUMNS vs rag_service.PROMPT_DOCUMENT_COLUMNS
- [ ] #2 A row whose raw provenance source_type is a canonicalization VARIANT (media_chunk, chat, or a plural spelling) either gets a hint and identity like its singular twin, or the deliberate exclusion is recorded with the reason and pinned by a test (finding 6)
- [ ] #3 The relationship between expand_document's gate and [console] direct_library_tools is recorded in one place a reader will find (task trade-offs plus Docs/User_Guide/mcp.md): whether expansion defers to that toggle, and why the raw-id read is acceptable under the ask floor (finding 13)
- [ ] #4 The conversation fetch no longer reads image BLOBs it does not render, or the cost is measured and the default is kept deliberately (finding 15)
- [ ] #5 The window/continuation half of the contract is exercised outside unit tests at least once — a document larger than the budget, expanded and continued via next_offset — and the result recorded (finding 16)
- [ ] #6 Any behaviour change here re-runs the gated retrieval suite and it still reads "PASSED: No regression. 105 metric(s)" with every cell at (+0.000)
<!-- AC:END -->
