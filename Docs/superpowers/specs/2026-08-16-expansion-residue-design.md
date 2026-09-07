# Expansion residue batch: TASK-16688 + TASK-16788 (one PR, two small tasks)

Date: 2026-08-16
Status: draft-pending-user-review
Programme: RAG server-port (twelve merged; last: 16588 #1729 → dev `faa8ba393`)
Worktree: `.worktrees/rag-16688-residue`, branch `chore/rag-16688-16788-residue`, off dev `c2f30862c`.

Both tasks are 16174-family residue with every finding already
evidence-backed; this arc's job is decisions + pins, not discovery. All
decisions are pre-registered here so the implementation is mechanical.

## TASK-16688 — five findings, five decisions

**AC#1 (twin literals): equality PINS, not imports.** Two tests assert
`set(SUPPORTED_SOURCE_TYPES) == set(EXPANDABLE_SOURCE_TYPES)` and
`PROMPT_BODY_COLUMNS == rag_service.PROMPT_DOCUMENT_COLUMNS` (content AND
order for the columns — the join/strip rule depends on order). No
cross-layer import: the tool (Tools/) and the policy (Library/) stay
decoupled; a pin that reds on drift is the whole requirement, and the
existing in-file reasons for not importing stay.

**AC#2 (canonicalization variants): record + pin the EXCLUSION.** The
16588 route probe measured variant rows at ZERO across 340 rows with a
committed positive control, and today's indexer stamps only singulars.
Broadening the allowlist would be speculative surface for a case with a
measured zero — the inert-knob lesson in miniature. Ship: a test pinning
that a `media_chunk`/`chat`/plural row gets no hint (the DELIBERATE
behaviour), a docstring note in `library_expand_policy.py` stating why
(with the probe citation), and one sentence in the 16588-probe README
already carrying the count.

**AC#3 (consent boundary): record in the two places a reader looks.**
`Docs/User_Guide/mcp.md`'s expand_document entry gains the relationship:
expansion does NOT defer to `[console] direct_library_tools` — it is
gated by its own `[tools]` key (default OFF) plus the per-call ask floor;
what that means (a raw-id whole-document read exists once a user sets
Always-allow, duplicating the direct get-tools while bypassing their
opaque-ID codec); and why it is accepted (OFF by default, risk-floored,
the raw id already left the adapter as `result_id` pre-arc). The same
trade-off paragraph lands in TASK-16688's Implementation Notes. No
behaviour change — this AC is a recording obligation.

**AC#4 (image BLOBs): the one-word fix + a kwargs pin.**
`_fetch_conversation` passes `include_image_data=False` (task-260
precedent). Pin: a test wrapping the real
`get_messages_for_conversation` to record kwargs, asserting the flag —
plus the existing transcript tests staying green (text output is
byte-identical; the flag only skips BLOB reads).

**AC#5 (continuation half, outside unit tests): a 30-line QA walk.**
`Docs/superpowers/qa/2026-08-16-expansion-residue/continuation_walk.py`:
scratch profile, ONE seeded >20,000-char note (16588's corpus-builder
pattern), expand at default budget → assert `truncated: true` +
`next_offset`; walk `next_offset` until exhaustion; record windows
observed, coverage (concatenated windows == the document), and the
call count in a short committed report. Env-isolate before any tldw
import; no TUI; no LLM.

**AC#6:** gate re-run (AC#4 changes a fetch path): verbatim
`PASSED: No regression. 105 metric(s)`, cells at (+0.000).

## TASK-16788 — the decision, pre-registered: DOCUMENT, don't filter

`search_run_log`/`run_log_slice`/`run_log_stats` are appended to
`runtime_schemas` (`agent_service.py:1564-1580`) — the same family as
`spawn_subagent`, `find_tools`/`load_tools`, the skill-file tool — ALL of
which bypass `config.allowed_tools` by design: that parameter filters the
CATALOG (`:1501` and the Q7 guards), not the runtime layer. Filtering
only the run-log tools would make one runtime tool behave unlike its
family; filtering the whole family would break skills and sub-agents.
So: **the docstring arm.** `allowed_tools`' documentation states
explicitly that runtime tools (enumerated, with their own gates) are
always offered regardless; a test pins that the run-log schemas are
offered under `log_active` even with an empty allow-list (the documented
behaviour, red if someone later "fixes" it silently); and the docstring
references the 16174 oracle report's confound note
(`Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/report.md`) so the
experiment-isolation trap stays discoverable. Plan-phase verification:
how run-log CALLS are dispatched (the `:1041` invoke_tool guard rejects
non-allowed catalog names — confirm runtime calls take a different path,
and say where, in the docstring).

## Out of scope
- Broadening the variant allowlist (AC#2 records the exclusion instead).
- Any change to the runtime-tool offering logic (16788 documents it).
- The 16450 shielded-drain loop; 3502; anything retrieval-affecting.

## Testing
- The pins above; the gated suite (AC#6); `Tests/Tools` +
  `Tests/Agents` + `Tests/Library` batteries, counts read.
- The QA walk's committed report is AC#5's evidence.
