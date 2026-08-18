# Residual zero-row queries: what keyword construction can and cannot reach (TASK-17855)

Date: 2026-08-18 · gated fixture (172 docs, 60 golden queries), plain path,
current dev (post-TASK-17755). No network, no spend.
Reproduce: `reachability_census.py` in this directory.

## The test, and why it is decisive

A construction can only rearrange the terms the user typed. So the question
is lexical, not algorithmic: **does the relevant document contain ANY content
word from the query?** If not, no AND / OR / prefix / stopword-trim
arrangement of those terms can ever retrieve it, and the query belongs to the
semantic leg permanently.

## The census — 30 residual zero-row queries with ground truth

| | queries |
|---|---|
| **UNREACHABLE** — target shares NO content word | **19** (63%) |
| lexically reachable — some or all words present | **11** (37%) |

The 11 break down as one query with **all** content words present, and ten
with partial overlap (from 1/8 up to 7/8).

## CORRECTION (2026-08-18, before any work on TASK-18255)

**The prompt-seam conclusion below is WRONG, and is retained only so the
error stays legible.** The plain path returns zero rows for prompt queries
because the EVAL HARNESS does not wire a prompts seam: its fake app sets
`prompt_scope_service=None`, so `_search_prompts` returns `(False, [])` —
the seam reporting itself **unavailable**, which is not the same as matching
nothing. Production wires it (`app.py:5682`).

The harness states this in its own comment, which I did not read before
concluding: *"Leaving it None means the harness's plain column reports 0.000
for prompts while the shipped app's plain mode does find them."*

So `plain`'s `category.prompt.*` cells are **vacuous by construction**, and
reading them as a retrieval failure — as the section below does — is the
mistake.

**Stated precisely, so this correction does not overclaim in the other
direction:** what is established is that the measurement was vacuous, not
that the seam works. Production wires the service; whether it then retrieves
`prompt-vendor-chaser` is **untested here** — no arc has ever exercised the
plain prompts sub-leg against a real `PromptScopeService`. The defect claim
is withdrawn as unsupported, not disproven. Establishing which it is, is
exactly what TASK-18255 now exists to do.

**The rest of this report stands**: the 19-unreachable / 11-reachable split
is a lexical property of the corpus and does not depend on any seam being
wired.

## The finding that changes the task's premise (SUPERSEDED — see above)

The task assumed the residual was a *recall* problem to be attacked by
broadening. Half of that is right — the 19 unreachable ones are the semantic
leg's job by construction, and no keyword work will ever move them.

But the reachable 11 are dominated by one category, and it is not behaving
like a construction problem:

| mode | `prompt` queries returning rows | finding the target |
|---|---|---|
| `plain` | **0 of 5** | 0 |
| `semantic` | 5 of 5 | 0 |
| `hybrid` | 5 of 5 | 1 |

**The plain path returns nothing at all for any prompt query**, including
`"saved prompt for chasing a supplier about a late order"` — whose target is
named *"Saved prompt: chasing a late order"* and contains **every content
word of the query**. `prompts_fts` indexes five columns including `name`, so
the terms are indexed. A sub-leg that returns zero rows when every term is
present is a **defect**, not a construction that needs widening.

## Decision (AC#3)

**Keyword construction has NOT reached its ceiling — but the remaining
headroom is not in broadening, and pursuing broadening would paper over a
defect.**

- The **19 unreachable** queries are closed as the semantic leg's business.
  No construction can reach them; this is recorded so no future arc
  re-derives it.
- The **11 reachable** queries are pursued as **TASK-18255**, a seam defect
  in the plain prompts sub-leg — filed rather than fixed here, since this
  task's scope is characterisation.
- **Broadening is explicitly not recommended.** Its cost is measured:
  TASK-3997 found pure OR/prefix cut MRR by 34% (0.396 → 0.261). Paying that
  to reach queries whose target contains the words — when the seam should
  already be reaching them — would be buying with the wrong currency.

## AC#2's candidate, with its cost

| candidate | reaches | precision cost |
|---|---|---|
| **fix the prompts sub-leg** (TASK-18255) | up to 5 of the 11 | **none by construction** — the terms are already present and indexed |
| broaden the construction | some of the remaining 6 | **−34% MRR**, measured (TASK-3997) |
| semantic/vocabulary work | the 19 unreachable | out of scope here; the only route for them |
