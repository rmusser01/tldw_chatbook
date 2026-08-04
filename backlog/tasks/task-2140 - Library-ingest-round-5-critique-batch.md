---
id: TASK-2140
title: >-
  Library ingest round-5 critique batch (commit-summary regression, failure-aftermath P1s, confirm at scale)
status: Done
assignee: []
created_date: '2026-08-04 03:00'
labels:
  - library
  - ingest
  - ux
priority: high
dependencies: []
---

## Description (the why)

The round-5 dual-agent critique (snapshot `2026-08-04T02-23-46Z…`, 26/40 —
trend 21 → 24 → 29 → 25 → 26; every round-4 fix verified holding by both
agents) found the remaining defects concentrated in the failure
aftermath, plus one self-inflicted round-4 regression. Owner: fix
everything now.

1. **[P2, regression] The commit-summary line is unreliable** — it is a
   conditionally-composed canvas-level Static (the round-3
   empty-Recent bug class, reintroduced by task-2130): a text-only
   pre-flight applies via the non-structural in-place path, which never
   mounts it (deterministic: PDF selections render "1 will import",
   plain-text selections never do), and after Clear it goes STALE
   ("0 will import · 1 will match" above an empty field).
2. **[P1] "Show details" adds no substance and the advisory misleads:**
   the "Underlying:" line repeats the row's error verbatim (the worker
   dedups chain entries against the message STRING, but entries carry a
   `ClassName: ` prefix, so equality never fires), and the retry
   advisory suggests "a network hiccup" for a local parse failure.
3. **[P1] Dismiss destroys the only record of a failure with zero
   friction** — after Dismiss the failure is gone from the queue, the
   summary, AND Recent ingests, while the less-destructive Clear
   finished has a two-press confirm.
4. **[P2] The armed clear-finished confirm can hide at queue scale:**
   with a ~9-row queue the arming press scroll-jumped the pane leaving
   "Press again…" below the fold (a 2-row queue passes) — the queue
   panel recompose resets scroll before/despite the `scroll_visible`
   call.
5. **[P2, suspected] One unreproduced dead Start click** at the commit
   moment — pin the type-then-immediately-click-Start path with a
   regression test and record the investigation.

## Acceptance Criteria (the what)

- [x] The commit-summary line renders for ANY valid selection (plain
      text, PDF, folder) regardless of which update path applied the
      pre-flight, and clears when the selection clears — always-mounted,
      display-managed, owned by the in-place updater.
- [x] No expanded detail line repeats the row's error text (prefix
      differences do not defeat the comparison); the retry advisory for
      parse errors talks about corrupt files, not network hiccups —
      network/transient copy is reserved for non-parse categories.
- [x] Dismissing a failed row preserves it in Recent ingests (marked
      dismissed where representable); the failure record survives on at
      least one surface.
- [x] The armed clear-finished confirm is visible after the arming press
      with a tall queue (scroll ordered after layout settles).
- [x] A Start click immediately after typing submits on the first click
      (regression-pinned); the dead-click observation is recorded with
      the investigation outcome.

## Implementation Plan (the how)

Five fixes in one pass (owner: "fix everything now"): (1) commit-summary
Static → always-mounted/display-managed/updater-owned; (2) prefix-aware
chain dedup + advisory-by-category; (3) dismissed failures → Recent
ledger with "(dismissed)" suffix; (4) call_after_refresh for the armed
confirm scroll; (5) type-then-click-Start regression pin.

## Implementation Notes

- **Commit summary (regression):** the conditional compose was the
  round-3 empty-Recent bug class reintroduced — the non-structural
  pre-flight apply path never mounts a conditionally-composed
  canvas-level element. Now always mounted; the in-place updater owns
  content + visibility. Live-verified: "1 will import" renders for a
  plain-text selection (the deterministic FAIL case) and clears with
  the field.
- **Details substance:** chain entries are compared after stripping the
  "ClassName: " prefix and unwrapping, against both the structured
  message and the job's own error — a prefixed restatement never
  renders. Parse-error advisory now: "If the file is corrupt or
  truncated, repair or re-export it, then Retry." (network/transient
  copy reserved for non-parse categories).
- **Dismiss ledger:** `registry.dismiss()` returns the job; the handler
  prepends it to `_library_ingest_recent_ledger` (dedup, cap 10) and
  Recent renders a " (dismissed)" suffix off the job's own flag.
- **Confirm at scale:** the arming branch defers `scroll_visible` via
  `call_after_refresh`, querying the post-recompose button inside the
  callback (immediate scrolls aimed at pre-refresh geometry on tall
  queues).
- **Dead Start click:** not reproducible in the harness; pinned by
  `test_start_click_immediately_after_typing_submits` (click within the
  debounce window, before any pre-flight lands → job submitted). If the
  live observation recurs, the pin separates app defect from driving
  noise.

**Verification.** 243 core + 71 shell-subset targeted tests green;
29,363 collect cleanly. Live on a fresh isolated profile: commit line
for text + clears on Clear.

**Qodo round (PR #1307):** one finding — Google-style docstrings on the
new test functions — declined on file idiom (276 tests in the file use
prose narrative docstrings; 39 share the "(task-NNN)" provenance
convention these follow), consistent with every prior round of the arc.
