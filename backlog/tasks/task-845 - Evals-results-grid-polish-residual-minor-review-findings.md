---
id: TASK-845
title: >-
  Evals results-grid polish: dedupe scans, unused params, and two unconfirmed robustness gaps
status: To Do
assignee: []
created_date: '2026-07-27 03:00'
labels:
  - evals
  - polish
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual Minor findings from the PR 3b whole-branch review. The Critical and all five Important findings were fixed in that PR, along with five cheap Minors (two weak test assertions, an un-awaited-coroutine warning, a stale CSS comment, a stray decorator). These nine were deferred deliberately — none affects correctness of the shipped path — and are recorded here so they are not silently dropped.

**Duplication**
- `_ever_observed_first_probe` / `_ever_observed_all_probes` (`results_grid.py:987`, `:1016`) are the same nested scan written twice; the second one's docstring already says so. Fold into one helper taking a probe tuple.
- `_notify` is byte-identical in three places: `results_grid.py:543`, `library_rail.py:492`, and `snippet_editor.py`. Promote to a shared helper in the `UI/Evals` package.

**Dead or unexercised**
- `analysis.combined_truncation`'s `k` parameter (`analysis.py:197`) has no caller — every call site passes `None`. Either use it from `_delta_reading`, which already knows the shared k, or drop it.
- `EvalsScreen._sample_bench_cancel_token` (`evals_screen.py:93`) is assigned and cleared but never read to cancel anything. Acceptable as a documented seam for PR 3c, but dead today.
- Redundant guard `if len(top) > 1 and analysis.near_tie(cap)` (`results_grid.py:849`) — `near_tie` already returns `False` for `len(top_k) < 2` and is tested for it.

**Robustness (both flagged unconfirmed by the reviewer — verify before acting)**
- `analysis.effective_k` (`analysis.py:119`) takes `min(k_returned)` without clamping to `len(top_k)`, whereas `divergence` clamps with both (`analysis.py:160`). If a provider ever reported `k_returned` larger than the tokens it actually returned, entropy would read that cell's full list while its neighbour is truncated. Depends on whether the normalizer guarantees `k_returned == len(top_k)`; that was not verified.
- `ResultsGrid.compose` (`results_grid.py:304`) catches only `ValueError` from `load_grid`. A `sqlite3`/`CharactersRAGDBError`, or a `KeyError` on a malformed stored payload, would escape `compose()` and become a `MountError`. Reachability unconfirmed; broadening to `except Exception` with a logged reason costs nothing.

**Pre-existing, in this PR's blast radius**
- `EvalsInspector.compose` (`inspector.py:149-150`) has a bare `except Exception: return` around `load_bench`, silently rendering an empty inspector pane with no message and no log line. Introduced in PR 3a.

**Deliberate choice needed**
- `_export_csv_text` (`results_grid.py:553`) writes through `csv.writer` into a `StringIO` without `newline=""`, then `write_text`, producing `\r\n` terminators. RFC-correct, but it should be a decision rather than a default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The duplicated probe scan and the three `_notify` copies are each single-sourced
- [ ] `combined_truncation`'s `k` parameter is either used or removed
- [ ] The two unconfirmed robustness gaps are each verified, then fixed or closed with the evidence
- [ ] The inspector's bare except either reports a reason or is narrowed
- [ ] CSV line terminator is a deliberate, documented choice
<!-- AC:END -->
