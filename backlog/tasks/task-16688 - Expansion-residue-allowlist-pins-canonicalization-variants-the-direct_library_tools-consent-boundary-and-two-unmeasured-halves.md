---
id: TASK-16688
title: >-
  Expansion residue: allowlist pins, canonicalization variants, the
  direct_library_tools consent boundary, and two unmeasured halves
status: Done
assignee: []
created_date: '2026-08-15 20:20'
updated_date: '2026-08-16 16:59'
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
- [x] #1 The tool's and the policy's source-type allowlists can no longer drift apart: one is derived from the other, or a test fails when they differ (finding 5); the same is done or explicitly declined, with a reason, for PROMPT_BODY_COLUMNS vs rag_service.PROMPT_DOCUMENT_COLUMNS
- [x] #2 A row whose raw provenance source_type is a canonicalization VARIANT (media_chunk, chat, or a plural spelling) either gets a hint and identity like its singular twin, or the deliberate exclusion is recorded with the reason and pinned by a test (finding 6)
- [x] #3 The relationship between expand_document's gate and [console] direct_library_tools is recorded in one place a reader will find (task trade-offs plus Docs/User_Guide/mcp.md): whether expansion defers to that toggle, and why the raw-id read is acceptable under the ask floor (finding 13)
- [x] #4 The conversation fetch no longer reads image BLOBs it does not render, or the cost is measured and the default is kept deliberately (finding 15)
- [x] #5 The window/continuation half of the contract is exercised outside unit tests at least once — a document larger than the budget, expanded and continued via next_offset — and the result recorded (finding 16)
- [x] #6 Any behaviour change here re-runs the gated retrieval suite and it still reads "PASSED: No regression. 105 metric(s)" with every cell at (+0.000)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Twin-literal equality pins (`Tests/Tools/test_expansion_twin_literals.py`), RED by mutation on BOTH sides of both pairs.
2. Variant-exclusion pin + docstring note citing TASK-16588's measured zero; one sentence in the probe README.
3. `include_image_data=False` in `_fetch_conversation`, RED-first behind a kwargs-recording pin.
4. A QA continuation walk (`Docs/superpowers/qa/2026-08-16-expansion-residue/`) over a >20,000-char note, report committed.
5. The consent-boundary recording in `Docs/User_Guide/mcp.md` + these notes.
6. Gate re-run (a fetch path changed) + `Tests/Tools` / `Tests/Agents` / `Tests/Library` batteries + ruff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Five 16174-review residue findings closed; every decision was pre-registered
in `Docs/superpowers/specs/2026-08-16-expansion-residue-design.md`, so this
was execution, not re-litigation. One behaviour change (finding 15); the
other four are pins and recordings.

**Finding 5 — twin literals (AC#1): equality pins, not imports.**
`Tests/Tools/test_expansion_twin_literals.py` asserts
`set(SUPPORTED_SOURCE_TYPES) == set(EXPANDABLE_SOURCE_TYPES)` (membership on
both sides, so order is deliberately not pinned) and
`tuple(PROMPT_BODY_COLUMNS) == tuple(PROMPT_DOCUMENT_COLUMNS)` (order IS
pinned — the rendering is those columns joined in that order). Deriving one
from the other was declined: the tool is constructed by the Settings-side
gate enumerator purely to read its description, and importing `rag_service`
there would drag the embeddings stack into that path; the policy is a pure
`Library/` helper. Each pin was proved failable by mutating BOTH sides
(bogus entry in the policy allowlist, then in the tool's; a reordering of
the tool's columns, then a bogus column in `rag_service`'s), each reverted
by Edit.

**Finding 6 — canonicalization variants (AC#2): the exclusion is RECORDED
and PINNED, not fixed.** TASK-16588's route probe measured variant rows at
**0 across all 340 rows** on four (index × route) arms with a committed
positive control (its detector fires on every variant, on no singular), and
today's indexer stamps only singulars. Broadening the allowlist on a
measured zero would be speculative agent-facing surface with no producer —
the inert-knob lesson in miniature. Shipped instead: a module docstring note
on `EXPANDABLE_SOURCE_TYPES` carrying the count, the citation and the
revisit condition; parametrized pins over all five variant spellings (no
hint) with their four singular twins as the control (hint); and one sentence
in `Tests/RAG_Eval/README.md` tying the probe's last table column to this
decision. RED proved by temporarily admitting `chat` to the allowlist.

**Finding 13 — the consent boundary (AC#3): a recording obligation, no
behaviour change.** `Docs/User_Guide/mcp.md` gains
"`expand_document` and the Library consent boundary", and the trade-off is
this: **expansion does not defer to `[console] direct_library_tools`**
(default **on**; off means agents get bounded `search_library_rag` excerpts
instead of direct Library reads). It is governed by its own registration
gate, `[tools] expand_document_enabled`, which is **off by default**, plus
the per-call **Ask** floor its `risk_tags=("reads",)` forces. What that
means once a user enables the gate and clicks "always allow" is a
**read-by-raw-id primitive**: `source_type` + the row's backing database id
returns the whole note/media/conversation/prompt in bounded windows,
duplicating the 18 direct Library get-tools while bypassing their opaque
`type:<base64url>` ID codec, which normally lets a get-tool open only a row
some earlier search returned. Why it is accepted: the gate is off until
turned on; the risk tag floors an inherited Allow back to Ask (one approval
card per call — 7 rounds observed live in the 16174 run); and the raw
backing id already left the Library RAG adapter as each row's `result_id`
before expansion existed, so the arc TYPES an existing exposure rather than
inventing one. The stricter posture is the default one — leaving
`expand_document_enabled` off, or answering Ask per call; turning
`direct_library_tools` off does **not** disable expansion, and the docs now
say so where a reader will look.

**Finding 15 — image BLOBs (AC#4): fixed.** `_fetch_conversation` now passes
`include_image_data=False` (the task-260 precedent); the transcript renders
`sender`/`content` only, so the reader's default was pulling up to 500 image
BLOBs into memory for text that cannot use them. Pinned RED-first by
wrapping the REAL `get_messages_for_conversation` on a seeded conversation
whose first message carries a BLOB, recording the kwargs, and asserting the
flag — with two controls in the same test (the flag really does null the
column; the default really does return it) and a byte-identical transcript
assertion.

**Finding 16 — the continuation half (AC#5): walked.**
`Docs/superpowers/qa/2026-08-16-expansion-residue/continuation_walk.py` seeds
ONE 25,228-char note through the production writer in a scratch profile
(env-isolated before any `tldw_chatbook` import; real config sha256
unchanged), expands it at the DEFAULT 8,000-char budget and follows
`next_offset` to exhaustion: **4 calls, windows 0-8000 / 8000-16000 /
16000-24000 / 24000-25228, 25,228 of 25,228 chars = 100% coverage**, the
concatenation compared to the stored body with `==`. Failability is what
makes it a reading: windows must be contiguous (no gap, no re-served
overlap) and two unique markers — one in the first budget, one in the last —
must each appear in exactly one window, which a head-re-serving tool fails
on all three counts. `report.md` is committed beside it.

**AC#6 — gate.** `RAG_EVAL=1 pytest Tests/RAG_Eval/ -q -p no:randomly`:
`[rag-eval baselines] PASSED: No regression. 105 metric(s) within 0.05 of
baseline.`, all 105 cells at (+0.000), 307 passed.

Batteries: `Tests/Tools` 577 passed / 15 skipped, `Tests/Agents` 1456
passed, `Tests/Library` 1994 passed / 1 skipped with ONE failure —
`test_library_skills_state.py::test_shadow_name_set_stays_in_sync_with_real_sources`
(`ConsoleCommandRegistry` name `research` missing from
`_SHADOWED_BUILTIN_NAMES`), which arrived on dev with `e1f3a4424`, an
ancestor of this branch's base, and touches no file this task modifies. It
is a dev-baseline red, not this task's, and its own message says it must not
be accepted as a baseline. It is not a new finding: **TASK-13214** already
owns this guard (filed 2026-08-10 on generate-video/stream-video), and its
AC#3 predicted exactly this — the assertion short-circuits on the first
uncovered subset, so fixing one gap reveals the next. The sighting, the new
name and its introducing commit are recorded there rather than duplicated
into a second task.

**Modified/added:** `tldw_chatbook/Tools/document_expansion_tool.py`,
`tldw_chatbook/Library/library_expand_policy.py`,
`Tests/Tools/test_expansion_twin_literals.py` (new),
`Tests/Tools/test_document_expansion_tool.py`,
`Tests/Library/test_library_expand_policy.py`, `Tests/RAG_Eval/README.md`,
`Docs/User_Guide/mcp.md`,
`Docs/superpowers/qa/2026-08-16-expansion-residue/continuation_walk.py` +
`report.md` (new).
<!-- SECTION:NOTES:END -->
