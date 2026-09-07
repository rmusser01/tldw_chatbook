# Agentic document expansion: one gated tool, a wired policy, an honest measurement (TASK-16174)

Date: 2026-08-15
Status: draft-pending-user-review
Programme: RAG server-port (ten resolved; last: 15810 via PR #1640/#1672)
Worktree: `.worktrees/rag-16174-expansion`, branch `feat/rag-16174-agentic-expansion`, off dev `8727a2861`.

## The gap, restated from the task's measured record

An agent can already RETRIEVE (`RAGSearchTool`, and the default Console
Library tool `Agents/library_rag_tool_provider.py`), but it cannot FOLLOW
a hit into its document — and since TASK-16071's rank-fair merge, **54%
of the rows a top-M consumer is fed are label-only** (113/211; media rows
say `Matched media · {type}`, conversation rows say `Matched conversation
· N messages` — `library_local_rag_search_service.py:1187/:1202`, both
with `chunk_id: ""`). The majority of what an agent sees is a label it
cannot see behind. Meanwhile three shipped profiles switch ON a
parent-inclusion surface that is wired to nothing
(`include_parent_docs`/`parent_size_threshold`/`parent_inclusion_strategy`,
`RAG_Search/simplified/config.py:559-561`; set by `config_profiles.py:310-312`,
`:347-349`, `:524-526`; **grep re-verified at this HEAD: 4 occurrences
total = 1 definition + 3 profile writes, 0 reads**).

## Phase K — knob truth first (AC#7): RETIRE, with the decision pre-registered

**Decision rule, stated before any code:** the knobs get WIRED only if
this arc's capability lands engine-side (inclusion during retrieval).
It does not — the capability is an AGENT TOOL: pull-based, budgeted,
gated, invoked per-hit after retrieval. Engine-side inclusion would
change what retrieval returns and this arc's gate is 105/105 (+0.000),
so wiring them here is out of scope by construction. Therefore: **retire
all three knobs** from `SearchConfig` and from the three profiles, so no
user-switchable dead surface survives alongside a tool that does the job.

Compat obligations (plan-phase verification items): how `SearchConfig`
construction treats unknown keys from a user's saved TOML (a removed
field must degrade to an ignored key with a logged notice, not a crash);
whether any Settings/UI surface renders these fields; whether any test
fixture sets them.

## Phase T — the tool (AC#1, #2, #4)

One new gateable builtin, `expand_document` (final name checked against
catalog conventions at plan time), following CLAUDE.md's recipe: `Tool`
subclass + `GateableTool` row in `_GATEABLE_BUILTINS` + `[tools]` gate
key, OFF by default. It reads user data (notes, media, conversations,
prompts), so it carries `risk_tags` and is floored to `ask` — per-call
approval cards until the user sets Allow in the MCP screen's permission
layer. That friction is a fact to disclose, not design around.

**Contract (AC#2), in terms of what rows actually carry:**
- Input: `source_type` (the row's `provenance.source_type`) + `source_id`,
  optional `chunk_id` (semantic note rows have one; media/conversation
  rows ship `""`), optional `offset` for continuation.
- Returns per type: note → note body; media → media content; conversation
  → rendered transcript (role-prefixed messages); prompt → prompt body.
  Every payload states `total_size`, the window returned, and whether it
  was truncated.
- Over budget: a window centred on the matched chunk when `chunk_id` is
  known, else the head — plus a continuation `offset`, so navigation
  within one document is possible without re-querying (the owner's
  "starting points for additional information" ask).
- Budget: a `max_chars` parameter with a config-backed default; the tool
  never returns more than the budget regardless of what is asked.
- **AC#4 is closed by construction**: the contract works from exactly the
  fields label-only rows carry (`source_id` + `source_type`, empty
  `chunk_id`), and the media/conversation branches are the acceptance
  tests' first-class cases, not afterthoughts.

## Phase P — the policy, WIRED so it cannot be inert (AC#3)

A policy nothing consumes would be the same sin AC#7 condemns. So the
policy lives where the agent actually looks:
1. The tool's `description` states the decision rule in agent-actionable
   terms (expand when a high-ranked hit is label-only or visibly
   truncated and the answer needs its content; re-query instead when the
   hit itself looks wrong; never re-expand the same document).
2. The Library retrieval tool's result payload gains a per-row
   `expand_hint` (e.g. `{"expandable": true, "reason": "label_only"}`)
   computed by a pure, unit-tested helper — the agent is TOLD which rows
   are labels rather than left to infer it. This is a payload ADDITION
   (no row removed or reordered): the plan verifies consumers tolerate
   the extra key.
3. Tests exercise each policy branch (label-only high rank → expand;
   text-bearing full snippet → no; budget exhausted → no; repeat
   expansion of the same source → no).

## Phase E — the measurement, decided not assumed (AC#5, #6)

Retrieval-level scoring is NOT sufficient — the golden set scores
whether the document arrives, and expansion's value begins after it
arrives. So AC#5's first arm: define AND RUN a small answer-level check.
Shape (pre-registered, mechanical, no LLM grader): N fixed questions
whose evidence windows are label-only-dominated on the eval corpus, each
with a FACT ORACLE — a string/regex fact that appears ONLY inside the
media/conversation content, never in labels or note chunks. Run the real
agent loop tool-OFF vs tool-ON (small N, low spend, keys from the repo
root per standing rule; costs recorded); score = oracle-fact inclusion in
the answer. Disclosed limits: N is small, inclusion ≠ answer quality,
one corpus. That is an honest first answer-level instrument, not a P3
grader programme.

AC#6 recorded now: the tool is INDEPENDENT of reranking — it consumes
final result rows whatever produced them; nothing presumes TASK-3502's
unimplemented `cross_encoder`. The relationship is one sentence in the
tool's docs and the task file.

## Out of scope (declared)
- Engine-side parent/sibling inclusion (P2's "granularity router" line) —
  the knobs retire; the feature, if ever wanted, is its own measured arc.
- Reranking implementation or measurement (TASK-3502).
- Any retrieval-behaviour change: the gate must read **105/105 (+0.000)**.
- Changing the label-only rows themselves (their snippet text is the
  15700-era contract; the tool sees behind them instead).

## Testing
- Phase K: config-compat tests (saved TOML with retired keys loads with a
  notice); profile tests; grep-zero assertion that no read appears.
- Phase T: per-type contract tests incl. both label-only types; budget/
  truncation/offset tests; catalog gate tests (OFF by default, Settings
  switch derived, risk-tag floor to ask).
- Phase P: one test per policy branch; payload-compat test for the added
  `expand_hint` key.
- Phase E: the oracle run's artifact committed (questions, oracles,
  per-run answers, costs); the always-on Library/RAG batteries; the gated
  suite at 105/105 (+0.000).

## Plan-phase verification (before tasks are cut)
1. `SearchConfig` unknown-key behaviour on load (crash vs ignore) and
   every constructor/serialisation site of the three knobs.
2. Note/prompt fetch APIs by id (media/conversation verified:
   `get_media_by_id` `Client_Media_DB_v2.py:6158`,
   `get_conversation_by_id` `ChaChaNotes_DB.py:7575`) — and what
   `source_id` actually contains per seam for SEMANTIC rows (doc_id vs
   note_id vs media id; the chroma metadata carries `note_id`/`doc_id`).
3. The tool-catalog recipe end-to-end at this HEAD (gate key naming,
   Settings derivation, risk-tag flooring — CLAUDE.md's caveat about
   non-`_GATEABLE_BUILTINS` LocalToolSpec tools does not apply here).
4. Consumers of the Library tool payload (Console agent rendering) to
   confirm an added per-row key is tolerated.
5. Where the eval corpus's media/conversation content lives, to author
   oracle facts that genuinely appear only there.
