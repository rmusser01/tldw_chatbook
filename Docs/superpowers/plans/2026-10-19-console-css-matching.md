# Console CSS matching implementation plan

**Goal:** Reduce measured matching work during Console return and investigate the
previous latency regression without changing storage checks or styling behavior.

**Architecture:** Strengthen the existing candidate-only rejection filter. Use
Textual's parsed selectors and live type/class/id names; retain its original
matcher, cascade, pseudo-state processing and full resume refresh.

**Tech stack:** Python 3.12, Textual 8, native Windows profile diagnostics.

**Spec:** User's Console return/CSS follow-up; TASK-35611 acceptance criteria.

ADR required: no. ADR path: N/A. This extends an existing conservative presentation
optimization, retaining ADR-126/231 storage and runtime boundaries.

## Constraints and review focus

- Native storage admission, source ownership and physical retirement stay intact.
- No cached matching result or computed style; live pseudo-state/order still match.
- Split subject compounds from ancestors at the final child/descendant boundary.
- Comma alternatives are OR; requirements within one alternative are AND.
- Inherited types follow the matcher's CSS inheritance names; inherit_css=False
  excludes base classes, unlike Python's full MRO.
- Keep unknown selectors conservatively; source replacement invalidates candidates.
- Retain existing cache bounds and no-node ownership.

## Steps

- [x] Repeat unchanged return profile; separately count actual matcher calls.
- [x] Add tests for absent type scopes, mixed alternatives, compounds, inherited
  types and reparenting in `Tests/Performance/test_css_static_requirements.py`.
  Verify failure on original code before implementation.
- [x] Update `Utils/textual_css_fastpath.py` to require known ancestor and subject
  names per alternative. Keep upstream matching for all surviving rules.
- [x] Run new tests, candidate reuse tests, existing parser/contract tests and
  real Console/Library/Settings style fidelity. Record baseline-only failures.
- [x] Measure four interleaved same-screen CSS pairs with style comparisons, then
  separate original/current six-return Windows profiles without concurrent tests.
- [x] Obtain independent review, record timings and limitations, update PR3166.

## Investigation notes

The previous outlier was first return (2523 ms); an unchanged repeat's first
return was 444 ms and slowest was last (1749 ms). This does not reproduce a fixed
first-return regression. Main-thread samples consistently identify Textual's
ScreenResume full stylesheet refresh. A separate observer counts 16960 rule checks
and 2059 matching checks in one update, including hundreds of rules whose leading
type does not exist in the Console ancestry. Diagnostic timings are excluded from
end-to-end comparisons. The initial direct pytest invocation was refused by the
shared TEMP-directory safety guard; use the genuine user-owned fixture runner.
