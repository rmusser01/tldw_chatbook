---
id: TASK-2311
title: Briefing failures surface their reason proactively
status: Done
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: medium
---

## Description (the why)

UAT: Generate with no provider configured yields a bare "failed" row —
no toast, no reason. The actual cause ("OpenAI API Key is required but not
found") only appears after clicking the row and reading the detail region,
and the provider defaulted to openai silently. A user who skipped first-run
setup hits their first configuration cliff with zero guidance.

UAT finding F39.

## Acceptance Criteria (the what)

- [x] A failed generation surfaces its reason at failure time (toast or
      inline, markup=False) without requiring row selection.
- [x] Configuration-class failures point the user at where to fix it
      (Settings), naming the provider that was attempted.
- [x] The provider used for generation is visible before generating.

## Implementation Plan (the how)

1. Trace the Generate flow end to end: `handle_generate_briefing_
   requested` -> `_generate_briefing` -> `briefing_service.generate_
   briefing` -> `_invoke_chat`, to find exactly where a provider failure
   becomes a `failed` row and confirm no toast fires on that path today.
2. Add a toast at the point `_generate_briefing` observes a `failed`
   row, naming the provider actually attempted (from the row's own
   `model_used`) and the provider's own error text, pointing at
   Settings -- reusing the screen's existing `_notify_watchlists` helper
   (`markup=False`, matching every sibling failure toast on this path).
3. Make the provider Generate will use computable and visible BEFORE
   the press: mirror `generate_briefing`'s own resolution order (preset
   provider, else the app default) into a screen helper, and render it
   on the Artifacts pane's always-visible scope line.
4. Tests (a real provider exception, not a database error) + mutation
   verification; live verification in tmux (no provider configured).

## Implementation Notes

**The pipeline was never silently swallowing information -- it was
recording it and nobody read it.** `generate_briefing` already turns a
provider exception into a `failed` row carrying `error` (the provider's
own message) and `model_used` (the endpoint actually attempted) --
`_finish_failure`'s existing contract. `_generate_briefing`'s worker
awaited the row, read `generated_id`, and moved on to its `finally`
repaint with no branch on the row's own status at all -- the failure was
on screen (as a table row) but never proactively surfaced.

**The fix is one new branch plus one new helper.** `_generate_briefing`
now checks the returned row's status and, on `failed`, calls the new
`_notify_briefing_failure(row)`, which reads `model_used`/`error`
straight off that row (not recomputed) -- so it always names the
provider that was ACTUALLY attempted, immune to a preset change racing
the toast. `default_briefing_provider()` (renamed from the module-private
`_default_provider`, its one other caller in `briefing_cast.py` updated
too) is the same resolution `generate_briefing` itself uses, now public
so the UI can call it prospectively.

**Provider visibility "before" is the Artifacts pane's existing
always-visible scope line, not a new row.** `ArtifactsPane.compose()`'s
toolbars are already at their one-row-strip height budget (TASK-995), so
the provider name is appended to `#artifacts-scope-note` -- the Static
that already reads "Briefings are written on this device..." -- rather
than adding a new control. `default_provider_display` is screen-computed
(mirrors preset-provider-else-app-default) and re-seeded at every point
that can change it: pane construction/rebuild, the in-place `_load_
briefings` refresh, and `_write_briefing_default_preset` (a NEW default
preset can carry a different provider than the one just displayed).

### Verification

* New tests in `Tests/Watchlists/test_watchlists_artifacts_pane.py`:
  a real provider exception (not the existing database-error test, which
  exercises a DIFFERENT code path -- `generate_briefing` itself raising,
  never reaching the `failed`-row branch) proving the toast fires without
  selecting the row, names the provider ("OpenAI") and its own error
  text, and points at Settings; a second test proving the provider is
  visible on the pane BEFORE Generate is ever pressed.
* Mutation-verified: 3 mutations (the failure-branch dispatch, the
  Settings-pointer text, the provider-visible-before-generating concat),
  each reverted individually -> RED -> restored byte-exact (md5).
* Gates: `Tests/Subscriptions/` **667 passed**, plus one pre-existing
  failure (`test_briefing_selection.py::test_overflow_and_watermark_
  stay_exact_over_a_backlog_larger_than_the_cap` -- reproduces
  standalone with zero diff against `origin/dev` in that module or its
  test, confirmed unrelated). `Tests/Watchlists/test_watchlists_
  artifacts_pane.py` **130 passed**.

### Follow-up (UAT batch-5 whole-branch review, finding m3)

Making the failure reason fire in an UNCONDITIONAL toast (rather than
requiring a click into the row's own detail region, its behavior before
this task) widened the DEFAULT exposure of whatever text a provider
handler's exception carries -- not a new leak (the row already stored and
displayed this same, already-1000-char-capped, never-a-traceback text
before this task; `briefing_service._error_text` is unmodified), but a
toast is a one-line surface and firing automatically is a lower bar than
requiring a deliberate click. Added `_bounded_briefing_failure_reason`
(`watchlists_collections_screen.py`): collapses embedded newlines/
whitespace to one line first (a raw HTTP response body or header dump
takes that shape; a curated one-line provider message like "OpenAI API
Key is required but not found." does not and passes through unchanged),
then caps the result at 160 chars with a truncation marker. The bound is
on SIZE and SHAPE, not content-pattern redaction -- it does not attempt to
detect or strip specific "looks like a payload" substrings, which would
be fragile pattern-matching; it guarantees a multi-line or very long
reason can never reach the toast as anything but one short, bounded line.
The full (server-side-capped, still not a traceback) text remains
reachable by clicking into the row's own detail region, exactly as
before this task. Mutation-verified (Edit-tool revert -> RED -> restored
byte-exact, md5).

### Files

* `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` --
  `_briefing_provider_display`, `_notify_briefing_failure`, the status
  check in `_generate_briefing`, seeding at all reseed points. Follow-up
  (m3): `_bounded_briefing_failure_reason`.
* `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` --
  `default_provider_display` reactive, the scope-note text.
* `tldw_chatbook/Subscriptions/briefing_service.py` (public rename),
  `briefing_cast.py` (updated caller).
* `Tests/Watchlists/test_watchlists_artifacts_pane.py`.
