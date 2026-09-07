---
id: TASK-861
title: >-
  Evals results-grid polish: dedupe scans, unused params, and two unconfirmed
  robustness gaps
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 03:00'
updated_date: '2026-07-27 05:53'
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
- [x] #1 The duplicated probe scan and the three `_notify` copies are each single-sourced
- [x] #2 `combined_truncation`'s `k` parameter is either used or removed
- [x] #3 The two unconfirmed robustness gaps are each verified, then fixed or closed with the evidence
- [x] #4 The inspector's bare except either reports a reason or is narrowed
- [x] #5 CSV line terminator is a deliberate, documented choice
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cleared all 8 in-scope items from the investigation (item 2, the compose() broad-catch, was already merged and skipped per instruction).

1. Folded probe scan: `_ever_observed_active_probe`/`_ever_observed_all_probes` now both call a new module-level `_probe_observed_in_target(snippets, cells, target_id, probe)` in results_grid.py, each still holding its own axis fixed (one probe/every target vs. one target/every probe), so neither call site pays for the other's cross-product. Test: `test_ever_observed_helpers_share_one_scan_and_hold_the_right_axis_fixed` monkeypatches the shared helper and asserts exact call counts/args per axis; reverted the fold and confirmed it fails (AssertionError on the dict content), then restored.

2. `_notify` de-duplicated into `tldw_chatbook/UI/Evals/notify_mixin.py::NotifyMixin`, a plain mixin (no `@on` handlers, so no metaclass concerns) mixed into `ResultsGrid`, `LibraryRail`, `SnippetEditor`.

3. Dropped `combined_truncation`'s dead `k` param rather than wiring it: its only caller (`_delta_reading`) never computes a shared k of its own before calling it (it gets `(jsd, is_bounded)` back from `divergence()`, no k), so wiring would mean re-deriving the same `min(...)` one level up for no benefit. Function now always recomputes internally, unchanged behaviour.

4. Removed the redundant `len(top) > 1 and` half of the near-tie guard in `_render_top1`; `near_tie` already returns False below two tokens (tested).

5. `EvalsInspector.compose`'s bare `except Exception: return` now logs via `logger.opt(exception=True).error` and yields a visible `Static(id="evals-inspector-error")`, mirroring ResultsGrid.compose; still narrowly `except Exception`, so CancelledError keeps propagating. Test: `test_inspector_reports_an_unexpected_load_bench_failure_instead_of_going_blank` (test_evals_screen.py) monkeypatches `load_bench` to raise `sqlite3.OperationalError` and asserts both the visible error Static and a captured loguru ERROR record naming the bench. Reverted to the bare except and confirmed the test fails (NoMatches on `#evals-inspector-error`), then restored.

6. Added the defensive clamp to `analysis.effective_k` (`min(cap.k_returned, len(cap.top_k))` per cell), mirroring `divergence()`'s own two-way clamp. Purely defensive per the investigation -- not reachable on any live capture path. This broke 3 existing tests because several fixtures (`clean_run_group`'s s2/s3 rows in particular) deliberately abbreviate `top_k` below their claimed `k_returned` as a pre-existing test-writing shorthand, not corruption. Fixed WITHOUT touching `clean_run_group` (many other tests hard-code divergence/spread/group-mean values derived from its s2/s3/s4 cells, which padding would have shifted): padded the self-contained fixture in `test_effective_k_is_the_minimum_k_returned_across_cells`, and added a new dedicated single-snippet fixture `k_depth_matched_run_group` (a verbatim copy of clean_run_group's already-well-formed s1 cells) for the two UI tests that only ever asserted on s1. Test: new `test_effective_k_clamps_a_k_returned_that_exceeds_the_actual_top_k_length` constructs a CellCapture with k_returned=99 but 2 real tokens and asserts effective_k clamps to 2. Reverted the clamp and confirmed the test fails (99 != 2), then restored.

7. CSV newline: `_export_csv_text`'s `io.StringIO()` -> `io.StringIO(newline="")` (confirmed a no-op byte-for-byte on this host -- StringIO's own default newline='\n' already means "no translation on write", same as ""). The change that actually matters for the documented Windows \r\r\n risk is in `_write_export_file`: the .csv branch's `write_text(...)` call now also passes `newline=""`, since Path.write_text's own default (newline=None) performs os.linesep translation-on-write, which would double an already-embedded \r\n into \r\r\n on a platform where os.linesep == "\r\n". POSIX output confirmed unchanged.

8. `_sample_bench_cancel_token`'s comment in evals_screen.py updated to state plainly "NOTHING READS THIS TODAY" and name PR 3c as the PR expected to wire a Cancel affordance, matching this file's own PR-numbering convention used elsewhere.

Modified: tldw_chatbook/UI/Evals/results_grid.py, tldw_chatbook/UI/Evals/inspector.py, tldw_chatbook/UI/Evals/library_rail.py, tldw_chatbook/UI/Evals/snippet_editor.py, tldw_chatbook/UI/Screens/evals_screen.py, tldw_chatbook/Evals/word_bench/analysis.py, Tests/UI/test_evals_results_grid.py, Tests/UI/test_evals_screen.py, Tests/Evals/word_bench/test_analysis.py. Added: tldw_chatbook/UI/Evals/notify_mixin.py.

Full specified suite green: 207 passed (Tests/UI/test_evals_results_grid.py, test_evals_empty_states.py, test_evals_screen.py, Tests/Evals/word_bench). ruff clean on all touched files.
<!-- SECTION:NOTES:END -->
