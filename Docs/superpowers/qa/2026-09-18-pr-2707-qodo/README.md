# PR2707 Qodo follow-up

Qodo review [5737187131](https://github.com/rmusser01/tldw_chatbook/pull/2707#issuecomment-5737187131) on 4d0201e812 identified nine public-documentation gaps and one unbounded Persona-picker finding. All ten are addressed in this closeout patch (TASK32824).

Both canonical Settings and the shared creation/default picker now browse 100 Personas per page using the existing service limit/offset contract. The selected or staged identity remains included when outside the page. Previous/Next use existing compact Button styles; keyboard focus returns to the picker. Browsing does not stage a new identity or grant memory consent. The local service sorts existing record references and normalizes only the requested page. Catalog sorting still depends on catalog size; this change bounds record-copy/normalization and option construction, not the underlying in-memory store. No schema, provider boundary, or identity/consent contract changes (ADR079/139, presentation ADR150/161).

The nine documentation groups add accurate Google-style arguments, return values, yields and relevant exceptions, including queue intent and deferred layout behavior. AST comparison after removing docstrings confirms no executable changes in those nine modules.

## Targeted evidence

- Three regression cases fail on the old implementation and pass after the fix: bounded modal options, bounded Settings options with complete three-page reach, and page-only record normalization. Their original failures are retained.
- All 46 selected identity, memory-confirmation, assistant-default and local-persona-service tests pass, serially in fresh private profiles. No full repository sweep.
- Scoped Ruff comparison introduces zero diagnostics; existing baselines are recorded per file. Modified small files pass formatting; larger legacy files receive scoped formatting.
- Synthetic 25,000-record view/option construction: seven-run median 26.79 ms before, 1.32 ms after. Returned records fall from 25,000 to 101 (one lookahead); constructed catalog options from 25,000 to 100. This excludes startup, disk and painting.

The first native run found that paging controls pushed the compact default modal's Apply action outside its clip. A new visible-region assertion reproduces it. The form now uses the existing fill-height token, keeping the actions visible while form contents scroll. Validation errors sit outside the scroller immediately above the actions; the existing unavailable-Persona test caught their initial loss of visibility. The behavioral regression passes; the historical CSS-migration comparison correctly reports the intentional auto-to-fill change, so final verification uses ordinary private-profile tests rather than claiming migration parity. Failed-run evidence records normal shutdown, eleven healthy databases and unchanged default fingerprints.

Final verification passes all 22 selected Persona/visual-governance/bundle cases, including the added Apply visibility assertion and existing error-paint check. The original 46-case run plus these 13 governance cases cover 59 distinct targeted cases; overlaps are not added twice. All seven preflight guards passed; the final stylesheet was then rebuilt and its reproduction check passed in the 22-case run.

## Native qualification

The real TldwCli/LinuxDriver passed dark/light × 80×24/170×48 using 207 real local Personas. Each journey created a workspace through Settings, preserved its exact `none`/`auto` identity across folder recomposition, traversed the default modal to page three without changing that identity, required explicit read-write confirmation, preserved defaults on Cancel, then selected the oldest Persona on Settings page three and retained it while paging back before Apply. The private workspace database was reopened after each mutation.

All 20 SVG captures were rendered and visually inspected, including readable page controls, focused saved/staged selections, visible validation and action buttons. The modal form scrolls within its capped dialog. This is a focused activation journey, not an exhaustive keyboard traversal of every control.

[Native result](native-result.json), [capture hashes](capture-manifest.json) and [lifecycle](lifecycle.json) record exit 0, process absent, lock reacquired, eleven healthy databases, zero conversations/messages, no error or faulthandler output, matching source hashes and unchanged default configuration/UI/policy fingerprints. The runner is preserved exactly as executed.

| State | Dark compact | Light compact |
| --- | --- | --- |
| Saved Persona on page three | [Capture](native/textual-dark-80x24-saved.svg) | [Capture](native/textual-light-80x24-saved.svg) |
| Explicit memory confirmation | [Capture](native/textual-dark-80x24-confirmation.svg) | [Capture](native/textual-light-80x24-confirmation.svg) |
| Oldest catalog choice staged | [Capture](native/textual-dark-80x24-catalog.svg) | [Capture](native/textual-light-80x24-catalog.svg) |
| Exact choice persisted after paging back | [Capture](native/textual-dark-80x24-applied.svg) | [Capture](native/textual-light-80x24-applied.svg) |

The user authorized merging after Qodo findings and current-head CI are handled. This evidence closes local qualification; the PR records final remote review and CI status before merge.
