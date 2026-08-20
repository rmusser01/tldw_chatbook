# TASK-18903 — a seam that failed must not read as a seam that found nothing

## The defect, measured

Every one of the Library's four keyword seams ends:

```python
except Exception:
    logger.opt(exception=True).warning("Library keyword search: <seam> failed.")
    return True, []
```

`True` means **available**. So a seam whose backend threw reports itself
healthy and empty. The merge site gates on `any(available)`, then discards
per-seam availability.

**Measured on dev (`692ed00d9`), all four backends raising:**

```
notes          -> available=True  rows=0   <-- claims AVAILABLE while broken
media          -> available=True  rows=0
conversations  -> available=True  rows=0
prompts        -> available=True  rows=0

caller receives: {"results": [], "runtime_backend": "local-fts"}
```

**A total backend failure is presented to the user as a successful search
that matched nothing.** No status, no recovery state, no indication anything
went wrong. The only trace is a log line nothing reads.

This is not only an instrument problem. It is the same collapse that produced
TASK-17855 (a wrong production-defect filing), TASK-18255, and two bugs inside
this programme's own censuses — but here it is **user-facing**.

## Why the obvious one-line fix is not enough

Changing `return True, []` to `return False, []` makes the all-fail case
report "blocked", which is right. But it leaves the **partial** case silently
wrong: if notes succeeds and prompts throws, the user gets notes results and
no indication that prompts was never searched. Silence still reads as absence.

## Design: three states, surfaced twice

**1. Replace the boolean with an explicit state.** The seams return
`(SeamState, rows)` where `SeamState` is `AVAILABLE` / `UNAVAILABLE` /
`FAILED`:

| state | meaning | today |
|---|---|---|
| `AVAILABLE` | the seam ran and its rows are its answer | `True` |
| `UNAVAILABLE` | not configured — `service is None` | `False` |
| `FAILED` | configured, ran, and **threw** | **`True` — the bug** |

**2. The merge site distinguishes total from partial.**

- No seam `AVAILABLE` and at least one `FAILED` → **`status="blocked"`** with
  a recovery state naming the failure. Today: a silent empty result.
- No seam `AVAILABLE`, none `FAILED` (all `UNAVAILABLE`) → `status="blocked"`
  with the existing no-backend recovery state. **Unchanged.**
- Some `AVAILABLE`, some `FAILED` → results are returned **and** the failed
  seams are recorded in `diagnostics` under a new keyed slot, so the caller
  can say "showing media results; notes and prompts failed" rather than
  implying the corpus was searched.

`diagnostics` is already a `dict[str, Any]` threaded through every outcome
with keyed slots (`SCOPE_DIAGNOSTICS_KEY`, `LIBRARY_RAG_ROUTE_NOTES_KEY`), so
this follows the established pattern rather than inventing a channel.

**3. Nothing else changes.** Row shape, the rank-fair merge (TASK-16071),
scope handling, and the `_empty_scoped_seam` sentinel (which is deliberately
`AVAILABLE`-with-no-rows, and stays that way) are untouched.

## Scope

**In:** `library_local_rag_search_service.py` — the four seams, the merge-site
gate, one new diagnostics slot; tests pinning each state and each combination.

**Out:** the UI copy that would render the partial-failure diagnostic (a
follow-up once the data exists); the RAG-eval metrics layer (it consumes rows,
not seam state — it benefits indirectly because a future census can finally
see the difference); the engine's own hybrid keyword leg.

## Acceptance criteria

- [ ] A seam whose backend throws reports `FAILED`, not available
- [ ] **All seams failing yields `blocked` with a recovery state, never a
      zero-row success** — pinned by a test that reds on today's code
- [ ] All seams merely unconfigured still yields the existing no-backend
      blocked outcome, byte-identical
- [ ] A partial failure returns the surviving seams' results **and** names the
      failed seams in `diagnostics`
- [ ] Every existing caller of the seams still type-checks and behaves
      identically for the `AVAILABLE` path — this must not become a
      three-state refactor that changes healthy behaviour
- [ ] `Tests/Library` stays green; the RAG-eval gate reads
      `PASSED: No regression. 105 metric(s)` (this touches no retrieval logic,
      so any movement is a defect in the change)

## Risks, all checked against dev rather than guessed

**1. The change could introduce the very collapse it fixes.** An `Enum`
member is **truthy**, including `UNAVAILABLE` and `FAILED`:

```
bool(SeamState.UNAVAILABLE) = True
```

The existing gate reads `if not any(available for available, _rows in ...)`.
Swap the boolean for an enum and that line silently becomes *never blocked* —
a working guard turning inert with no error, which is precisely this
programme's recurring defect. **Mitigation:** the all-unavailable case gets
its own test *before* the type changes, so it reds if the gate goes inert;
the gate is rewritten to an explicit `is SeamState.AVAILABLE` comparison.

**2. Callers: checked, and the seams are module-private.** Every hit outside
`library_local_rag_search_service.py` is a comment or a differently-named
method (`_search_conversations_sync` in CCP, `_tool_search_notes` in the MCP
delegate). Two tests unpack the tuple
(`Tests/Library/test_library_keyword_and_then_prefix.py`) and both discard the
flag, so they are unaffected.

**3. One committed QA script would go silently inert.**
`Docs/superpowers/qa/2026-08-18-prompts-seam/seam_effect.py` (merged in
TASK-18255) does `if not available:` to refuse a verdict on an unavailable
seam. Under an enum that check becomes always-False — the guard dies quietly.
It must be updated in the same change.

**4. `_empty_scoped_seam` returns `(True, [])` deliberately** — scope excluded
the seam, which is a genuine "searched, nothing in scope" answer. It maps to
`AVAILABLE`, never `FAILED`, or scoped searches start reporting failures.

**5. Hot path.** These seams run on every Library search. The change is
state-only: no added awaits, no extra queries, no row-shape change.


---

# Design review (2026-08-18, before implementation)

Six findings. Two change the design.

## R1 — SEVERITY: the defect creates a hallucination surface (worse than specced)

Traced end to end on dev `692ed00d9`:

1. all four seams throw → each returns `(True, [])`
2. `any(available)` passes the gate
3. rows empty, unscoped → returns a plain `{"results": [], …}` dict
4. `_outcome_from_service_result`: `if not rows: status="empty"`
5. `LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES = frozenset({"ready", "empty"})`
6. `library_screen.py:30880` — `if outcome.status not in …: return` — **so
   "empty" PROCEEDS**

**A total backend outage therefore reaches the RAG answer path, which
generates an answer with zero retrieved context and presents it as
Library-grounded.** The failure is not merely "user told nothing matched"; it
is "user given a confident answer built on nothing". This raises the priority
of the fix and is the strongest argument for the total-failure status being
NON-answerable.

## R2 — DESIGN CHANGE: a new status is probably redundant; `failed` exists

`run_library_rag_search`'s except branch already returns:

```python
return LibraryRagSearchOutcome(
    status="failed",
    recovery_state=_retrieval_failed_recovery_state(),
)
```

It already means "retrieval did not happen", already has a recovery state,
and is already fail-closed (absent from the answerable allowlist).

**DECIDED (owner, 2026-08-18): reuse `failed`.** Recommendation was to reuse
`failed` rather than mint a second failure status, and that is the ruling. Two statuses that both mean "retrieval did not happen" is an
ambiguity every future reader must resolve, and the only difference would be
recovery-state copy — which `recovery_state` already carries independently of
`status`. If a distinct status is still wanted, it must be added to no
allowlist and its copy must explain the difference from `failed`, or the
distinction decays into folklore.

## R3 — IMPROVEMENT: mirror the semantic leg; do not invent a scheme

The semantic leg **already implements this exact design**:
`SEMANTIC_DIAGNOSTICS_KEY` carries `{"status", "message"}` with
`SEMANTIC_STATUS_UNAVAILABLE` / `SEMANTIC_STATUS_EMPTY_INDEX`, and
`chat_rag_events.py` notifies the user — with the **same partial/total split
this task needs**:

```python
if results:
    notification = f"RAG context is keyword-only (FTS): {message}"
else:
    notification = f"Semantic retrieval returned no context: {message}"
```

The keyword-seam diagnostic should mirror that shape and that handler, and
follow the append-to-a-LIST convention the scope slot already uses (a recorded
task-9 review finding: assignment lets one entry silently overwrite another).

## R4 — SCOPE CHANGE: the UI notice belongs IN this task, not after it

My original scope deferred the rendering. Given R3, the notify site already
exists and extending it is small — and shipping a diagnostics slot that
nothing reads is the "declared but inert" trap this repo has hit before
(TASK-16174's retired knobs, the eight unimplemented middleware names).
**Pulled into scope.**

## R5 — the change can introduce the very collapse it fixes

`Enum` members are truthy, so `if not any(available for …)` silently becomes
*never blocked*. Pin the all-unavailable case BEFORE the type changes;
rewrite the gate to an explicit `is` comparison. (Already in Risks.)

## R6 — pin the inside/outside-`try` asymmetry

An exception **inside** a seam's `try` is swallowed to `(True, [])`; one
**outside** it propagates through `asyncio.gather` (no `return_exceptions`)
and becomes `status="failed"`. The fix makes both paths report failure, but
the asymmetry should be pinned so it cannot silently return.

## Revised acceptance criteria (superseding the list above where they differ)

- [ ] A seam whose backend throws reports `FAILED`, not available
- [ ] **Total failure is NON-answerable**: the outcome's status is excluded
      from `LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES`, pinned by a test that
      asserts the answer path does not run — this is R1, the real defect
- [ ] All seams merely unconfigured still yields the existing no-backend
      blocked outcome, byte-identical
- [ ] Partial failure returns surviving results AND records the failed seams
      under a keyed diagnostics slot, appended to a list, mirroring
      `SCOPE_DIAGNOSTICS_KEY`/`SEMANTIC_DIAGNOSTICS_KEY`
- [ ] **The user is notified** on both partial and total failure, mirroring the
      semantic leg's existing copy split
- [ ] `_empty_scoped_seam` maps to `AVAILABLE`; scoped searches report no
      failures
- [ ] `seam_effect.py`'s `if not available:` guard is updated — under an enum
      it would go silently inert
- [ ] `Tests/Library` green; RAG-eval gate `PASSED: No regression. 105
      metric(s)`
