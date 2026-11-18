---
id: TASK-41770
title: "The decomposition recipe has no record of the media body-move series' rulings and traps"
status: To Do
created_date: 2026-11-18 16:14
labels:
- follow-up
- architecture-review-2026-10-25
- library
- docs
priority: low
references:
- backlog/docs/library-decomposition-recipe.md
- backlog/docs/lessons-testing-evidence.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The earlier Library series each have an "as landed" section in `backlog/docs/library-decomposition-recipe.md` (§15, §18-§23 and §26). §22, "The media series, as landed", is wave 7's media extraction, not this series. TASK-36952.3 is the eight-slice series that moved 142 media bodies from LibraryScreen into the media controller, and it merged on 2026-11-18. The recipe does not mention it: 0 hits for 36952.

Its binding rulings and the traps it found live in two places only. One is the coordinator's working ledger, outside the repository. The other is two entries in `lessons-testing-evidence.md`. The next Phase A2 series (Notes or Prompts without the screen) needs them:
- **R1, asserts:** only the receiver may change inside an assert. The proof undoes each substitution and compares the def's AST with the base.
- **R2, body tokens:** an enumerated list of body tokens may change. Each `self in <widget>.ancestors` check (Form D) gets a red-first test, because that shape fails silently.
- **Patch census:** it covers every name a moved body calls that now resolves on the controller, not only the moved names.
- **A/B sets:** they come from Pilot selectors and key presses as well as name references.
- **Path-keyed guards and log filters:** source guards (`__file__` reads, `getsource(<module>)`, frame-file attribution) and log-record name filters keyed to `library_screen` are widened to the controller, and each widening is shown red on a planted violation.
- **Patched module globals:** a test that patches a `library_screen` module global misses a moved body, which now reads the controller module's global.
- **Duck-typed seams:** binding a name on the controller can switch on a `getattr(host, name)` seam for code that did not move. In 3.3 this was `_focus_library_list_entry_if_current`.
- **Incoming dev tests:** after every rebase, census dev's incoming test lines for reaches of moved names. In rb4 this found a dev test that called a pruned delegator.
- **RecoveryRequired ids:** ids blocked by `RecoveryRequired` run under the private-profile marker on both sides.
- **Prune census:** count the lambdas and attribute reads inside the screen's other controller constructors too. In 3.3 the browse controller's `sync_view` lambda kept alive a delegator the plan had marked for pruning.
- **Tokens inside a broad `except`:** an R2 token that fails to resolve inside a body's own `except Exception` is swallowed and shown as an ordinary failure. In 3.4 only the real-database restore tests would have caught a broken token in the restore worker.
- **Varying probe rows:** teardown-probe rows that vary run to run are judged by modal tuple at n=6, extended to n=12 when a mode splits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The recipe has a section for the TASK-36952.3 series as landed, titled so it cannot be confused with §22's wave-7 media series: its rulings, its measured totals and the traps above, each with the slice that found it
- [ ] #2 The section says which of these steps a future Phase A2 series must repeat
<!-- AC:END -->
