# Evals Rebuild PR 3b — Results Grid

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a word bench run as a pivotable grid, and finish the screen.

**Architecture:** The grid is a view over `word_bench.storage.load_grid` and `word_bench.analysis`. **The engine supplies every number; this PR adds no arithmetic beyond formatting.** Any calculation in `results_grid.py` past that is a defect.

**This is PR 3b of two.** PR 3a delivered the runner's preflight results, retired the card hub, and built the shell plus the bench and snippet editors. This PR adds the grid, its lenses, empty states, the sample bench, export, and the stylesheet cleanup.

**Tech Stack:** Python 3.11+, Textual, pytest. No new dependencies.

## Global Constraints

- Base branch: `origin/dev` **after PR 3a merges**. This PR builds on 3a's shell, selection state, and inspector.
- **A git worktree has no `.venv`.** Use the primary checkout's interpreter with cwd set to the worktree:
  ```bash
  cd <worktree> && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ...
  ```
  Verify `python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"` resolves **inside** the worktree before the first test run, or tests verify the wrong tree.
- **`pytest Tests/UI` cannot run in one call** — 5,250+ tests, ~51 minutes, exceeds a hard 10-minute per-call cap. Per-task gate:
  ```bash
  python -m pytest Tests/UI --collect-only -q          # 0 collection errors
  python -m pytest Tests/UI/test_evals_screen.py Tests/UI/test_evals_deletion_guard.py -q
  python -m pytest Tests/Evals -q                       # engine must stay green
  python -c "import tldw_chatbook.app"
  ```
  Collection alone is not sufficient — some UI tests read source files off disk by path and fail only at runtime. Full-suite runs are the controller's job.
- **Do not modify `tldw_chatbook/Evals/word_bench/` except in Task 1.** The engine is merged and reviewed.
- **`Tests/UI/test_evals_deletion_guard.py` and its 19-entry tuples must stay green.** PR 3 extends them (Task 2), never rewrites them.
- Design-system contract: use `.ds-*` shared classes and `$ds-*` tokens; assert **readable status text**, never colours; support `.density-compact` and `.density-comfortable`; `ds-status-badge` colour lives in app-tier CSS, never widget `DEFAULT_CSS`.
- The `timeout` command is not available. Do not push or open a PR without explicit authorization.

## Facts this plan is built on

**The hub is already broken.** Verified during PR 1 by capturing the screen on the branch and on a baseline worktree: `DestinationHeader` and `LabModeStrip` render, the body is **empty**, on both. `EvalsWindowV3` mounts fine in isolation (`EvalNavigationScreen` plus 8 buttons), so the failure is shell integration — Textual `Screen` objects mounted inside a `Container`. **There is no working behaviour to preserve parity with.**

**Top-1 is an unstable reading.** Two identical requests seconds apart returned the top two tokens in opposite rank order, magnitudes stable to ~0.002. The Top-1 lens must mark near-ties rather than presenting a bare winner.

**Divergence is not a bound.** PR 2's whole-branch review disproved the original claim. The number is comparable and reproducible; the grid must **not** render it with a leading `≥`.

**The mode-strip slot is taken.** `LabModeStrip` (Models | Speech | Evals) occupies it, so Evals-internal navigation is the library rail, not a second strip.

## File Structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Evals/word_bench/runner.py` | **Modified** (Task 1): return preflight results |
| `tldw_chatbook/UI/Screens/evals_screen.py` | **Rewritten**: three-pane shell, selection state |
| `tldw_chatbook/UI/Evals/library_rail.py` | Benches / Datasets / Runs, collapsible, counts |
| `tldw_chatbook/UI/Evals/bench_editor.py` | Bench detail + target table + readiness |
| `tldw_chatbook/UI/Evals/snippet_editor.py` | Snippet table, whitespace flags, import |
| `tldw_chatbook/UI/Evals/results_grid.py` | The grid, lenses, baseline |
| `tldw_chatbook/UI/Evals/inspector.py` | Readiness / stats / run meta / focused-cell detail |
| `tldw_chatbook/UI/Evals/evals_state.py` | Selection state and the screen's view model |
| `tldw_chatbook/css/features/_evals.tcss` | New sheet, `$ds-*` only |

**Deleted:** `UI/Evals/navigation/`, `UI/Evals/screens/`, `UI/Evals/widgets/`, `evals_window_v3.py`, `UI/evals_window_v2.py` (~2,700 lines), and the 12 Evals-only selectors in `_evaluation_unified.tcss` (verified still unused by surviving code).

---

### Task 1: Results grid and lenses

**Files:** create `results_grid.py`; test `Tests/UI/test_evals_results_grid.py`

**Interfaces:** consumes `word_bench.storage.load_grid` and every public function in `word_bench.analysis`. Produces `ResultsGrid`, `#evals-lens-selector`, `#evals-baseline-selector`.

This is the screen's centrepiece and the most likely place to misrepresent the engine's numbers.

**Requirements the tests must pin:**

- Five lenses: **Top-1**, **Entropy**, **Probe**, **Coverage**, **Δ baseline**.
- **Top-1 marks near-ties.** Two identical requests were observed returning the top two tokens in opposite rank order at magnitudes stable to ~0.002. When rank 1 and rank 2 are within a stated threshold, the cell must show the tie rather than a bare winner. A grid that hides this shows spurious differences between statistically identical cells.
- **Δ baseline never renders a leading `≥`.** PR 2's review disproved the lower-bound claim: crediting an absent token with 0 pulls the value up against the lumping approximation that pulls it down, and neither dominates. Render the number plainly; mark high-truncation cells with `!` and explain in the inspector.
- Baseline is explicit and switchable between a **column** and a **row**, and the header always states which is active. A divergence with an unstated reference point is the easiest way to mislead yourself.
- **Entropy passes a shared `k`** via `analysis.effective_k(...)`, and the header states the effective K. Otherwise a K=5 column reads as "more confident" than a K=20 column with no behavioural difference.
- A failed cell renders `—` with its reason in the inspector; an unrun cell is blank. Never `0` for either — both read as "measured and found nothing".
- A **warned** column (degenerate canary) carries its warning into the grid, so a divergence caused by an out-of-distribution target is never read as a finding about content.
- Focusing a cell updates the inspector with its full top-K and probe table; arrow keys move focus. No modal.
- Keys `l` (lens), `b` (baseline), `s` (sort), `e` (export) register through `ShortcutContext` so the footer stays truthful.

---

### Task 2: Empty states, sample bench, export

**Files:** modify `library_rail.py`, `bench_editor.py`, `results_grid.py`; create `sample_bench.py`; test `Tests/UI/test_evals_empty_states.py`

The screen's most common initial condition is zero benches, zero datasets, zero runs — and possibly zero configured providers. The current screen's core failure is looking functional while not being so.

**Requirements:**

- No providers configured → empty state routes to Settings; no target list, no wall of preflight failures.
- No benches → a **one-click sample bench** (the loaded-nouns snippet set, prewired to a configured target). This is the only way the screen's value is legible before a user invests in authoring anything.
- No datasets → authoring and import offered side by side.
- Export writes CSV for the active lens and JSON for the whole run group — snapshot, every cell's top-K, and resolved probe readings.

---

### Task 3: Retire the Evals-only stylesheet selectors

**Files:** modify `tldw_chatbook/css/features/_evaluation_unified.tcss`, regenerate the bundle

PR 1 deliberately left this sheet alone: its rules are **unscoped**, and 21 of its 33 selectors have surviving consumers across Chat, Logs, MCP, RAG search, Chatbooks and more. Only these 12 are Evals-only:

```
.advanced-config-form  .config-grid             .config-toggles
.cost-display          .dataset-management-form .empty-message
.model-management-form .quick-start-bar         .results-dashboard
.suggestion-text       .system-prompt-editor    .template-editor
```

- [ ] **Step 1: Re-verify each is still unused** by surviving code before deleting — this plan's survey was taken at branch time and other work lands continuously.
- [ ] **Step 2:** Delete only those 12 rules. **Leave the other 21 selectors untouched.** Deleting the file wholesale would silently restyle unrelated screens.
- [ ] **Step 3:** Regenerate the bundle via `build_css.py`; never hand-edit it.
- [ ] **Step 4:** `Tests/UI/test_non_obscuring_focus_contract.py` reads `EVALUATION_UNIFIED` off disk — confirm its 9 pre-existing failures are still exactly 9.

---

### Task 4: Live verification

No code. Uses the `verify` skill.

- [ ] Launch the app in this worktree with a scratch `TLDW_CONFIG_PATH`, navigate Lab → Evals.
- [ ] **Capture the screen and diff it against the same capture on `origin/dev`.** On dev the body is empty; here it must show the three-pane workbench. This is the inverse of PR 1's gate, where identical captures proved nothing broke — here a *difference* is the proof.
- [ ] Create the sample bench, run it against the live llama.cpp on `127.0.0.1:9099`, and confirm: readiness badges render, the degenerate canary produces a visible warning, the grid fills row-major, and lens switching works.
- [ ] Confirm no CSS warnings about missing selectors in the log.
- [ ] Note: `Ctrl+1`..`Ctrl+0` **cannot** be verified through tmux — `send-keys` has no ASCII encoding for ctrl+digit. Assert those in a unit test; never conclude from a tmux probe.

---


## Notes for the reviewer

- **The engine's numbers are not this PR's to change.** Any arithmetic in `results_grid.py` beyond formatting is a defect; the grid calls `analysis.py`.
- **Three renderings would misrepresent the engine, and each is pinned by a test:** a bare Top-1 winner on a near-tie (rank order was observed flipping between identical requests), a `≥` prefix on divergence (PR 2's review disproved the lower-bound claim), and entropy without a shared `k` (a K=5 column would read as more confident than a K=20 one with no behavioural difference).
- **Do not delete `_evaluation_unified.tcss`.** Only 12 of its 33 selectors are Evals-only; the other 21 have surviving consumers across Chat, Logs, MCP, RAG search and more. Re-verify the 12 at implementation time — this survey was taken at branch time.
