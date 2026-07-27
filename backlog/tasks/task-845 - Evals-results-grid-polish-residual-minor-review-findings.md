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

## Investigation (2026-07-27)

Each item was verified by reading the code and running probes, not by inference. One item is a severity upgrade and one is closed.

**UPGRADE — `ResultsGrid.compose` catching only `ValueError` is not Minor; it can exit the whole app.** Textual's `Widget._compose` (`textual/widget.py:4716-4725`) catches `Exception` and hands it to `App._handle_exception`, whose docstring states "Always results in the app exiting" and which sets `_return_code = 1`. So an escaping exception is not a localized `MountError` as the review supposed — the process terminates.

Reproduced, with the caveat that not every corruption shape triggers it:
- a missing run group raises `ValueError` and *is* correctly caught;
- a stored snapshot with a dropped key was tolerated, no error;
- a DB-level failure after the run group resolves (probe: `DROP TABLE eval_results`) raises `sqlite3.OperationalError`, which escapes.

`Evals_DB.list_runs`/`get_run_results` have no exception wrapping, so any sqlite-level fault — a locked database, a disk error, a schema mismatch from opening an older profile — reaches `compose()` unconverted. Treat as Important. Fix: catch broadly, log the reason, render the existing error state. Size S.

**CLOSED — the `effective_k` clamp is not reachable.** `capture_client.py:132` sets `k_returned=len(tokens)` from the same list that becomes `top_k`, so `k_returned == len(top_k)` holds by construction on every live path; it is a tautology, not an accident. It can only be broken by corrupting the stored JSON, since `storage.py:233` reads the two fields independently. When broken that way the code does not crash — `entropy()` silently falls back to the cell's own unclamped entropy while the header still advertises a shared K. Adding the same clamp `divergence()` already applies (`analysis.py:160`) is XS and purely defensive; the original correctness worry does not stand.

**Confirmed real, unchanged severity:**
- The duplicated probe scan is genuine (one has since been renamed to `_ever_observed_active_probe`, and a stale docstring cross-reference came with it). Foldable into one helper over target x probe sequences without losing either performance profile. S.
- The three `_notify` copies are byte-identical and self-acknowledged in their own docstrings. No shared base class exists; a small `NotifyMixin` in `UI/Evals` is the natural home. S.
- `combined_truncation`'s `k` is 100% dead. It is NOT a latent correctness gap: `_delta_reading` never computes k itself, and the recomputation it would replace is provably identical to `divergence()`'s. Drop the parameter or wire it; either is fine. S.
- The redundant `len(top) > 1` guard is a provable no-op — `near_tie` already returns `False` below two tokens and is tested for it. XS.
- `EvalsInspector.compose`'s bare `except Exception: return` yields zero widgets from a generator, so the user gets a blank pane with no message and no log line — nothing to diagnose from. S.
- `_sample_bench_cancel_token` is dead today but self-documented as a seam for the next PR. Reasonable to keep as-is.

**Partly unresolved:** the CSV terminator was probed directly rather than reasoned about — on POSIX the bytes on disk are correct RFC-4180 `\r\n` with no doubling. The Windows `\r\r\n` risk follows from documented `io.TextIOWrapper` semantics but was NOT executed, as no Windows host was available. Passing `newline=""` is XS and removes the question either way.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The duplicated probe scan and the three `_notify` copies are each single-sourced
- [ ] `combined_truncation`'s `k` parameter is either used or removed
- [ ] The two unconfirmed robustness gaps are each verified, then fixed or closed with the evidence
- [ ] The inspector's bare except either reports a reason or is narrowed
- [ ] CSV line terminator is a deliberate, documented choice
<!-- AC:END -->
