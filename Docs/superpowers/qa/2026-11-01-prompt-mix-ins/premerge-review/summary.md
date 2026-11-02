# PR #3346 pre-merge review and corrections

Initial independent review range: dev
`88af9f937908b46e8c5f9f7fddd9e8748c917d46` through
`b1108874cab28d23f660aa22c11c863466a84a8e`. The original head's required CI
was green before this fresh review. The corrected source is identified by the
frozen-file proofs and strict manifests in the commit containing this report.
Historical reports retain their original head qualification.

## Findings closed

| Finding | Correction and evidence |
| --- | --- |
| Supported Mix-in retained versions could not be restored | History recognizes the literal editor, retains dirty/read-only/source/version gates, and restores the exact retained type, body, placement and metadata as a new current version. Native tests cover v1 → v2 → restored v3 and permitted retained type changes. |
| Browse → Mix-ins blocked its modal message pump | A modal-owned worker admits the transition/read. Actual Button-route held-read tests prove Close and Back complete before release, and stale completion cannot replace the newer view. |
| The public legacy CSV/Markdown API flattened Mix-ins | The actual filtered selection refuses Mix-ins before keywords, temporary files or output. Normal Prompt/Recipe bytes and filters remain compatible; typed Library/Chatbook exports remain available. |
| Inspection omitted required retention information | Original mix-in inspection explains conversation-owned exact Retry retention with Capture Off and independence from trace purge. Existing explicit body visibility is preserved. |
| Persisted Info mode hid Mix-in Author/Keywords without a mode control | Literal Mix-ins retain the existing metadata presentation regardless of the saved ordinary editor mode. Prompt/Recipe preferences are unchanged; a production-CSS test edits and saves metadata through this route. |
| Separate size guards caught four owner overruns | Existing save/ack, Study handoff and command-history owners receive cohesive code; the redundant Browse worker accessor is removed without changing live lookup. No size ceiling increases. |
| Static guards missed the extraction's literal dependency syntax | Literal logger dictionary lookups remain counted, and literal lookup-exception dictionary keys retain recognition. Unknown/dynamic/unrelated keys and lookups outside the protected try body remain negative controls. No worker-census allowance was added. |

[Core review](reviews/mixins-pr3346-premerge-core-review.md.json),
[UI review](reviews/mixins-pr3346-premerge-ui-review.md.json), and
[integration review](reviews/mixins-pr3346-premerge-integration-review.md.json)
record the original findings and scope. The
[final combined independent review](reviews/mixins-pr3346-premerge-fix-review.md)
finds no remaining Critical, Important or Minor issues in the corrected technical
work. Qodo was intentionally bypassed under the owner's explicit instruction.

## Current source and guard ownership

| Owner | Corrected measurement | Previous ceiling |
| --- | ---: | ---: |
| Legacy export helper | 241 lines | 242 |
| Library Prompts controller | 5065 lines | 5100 |
| Library Browse controller | 281 lines | 281 |
| ChatScreen | 24590 lines / 741 methods | 24605 / 741 |
| LibraryScreen | 38177 lines / 1373 methods | 38242 / 1374 |

Existing receiving helpers have new exact ceilings: save/ack 548 lines, Study
screen helpers 401, and command handoff 322. Original public save arguments and
documentation remain at its facade; executable submission/acknowledgement and
live defining-module lookups remain equivalent. Screen moves retain registry,
cache, task, action and event ownership. Ten body/predicate comparisons and
caller inventories qualify those moves.

The state-field census advances from 43 to 45 for the two actual private fields
`mixin_draft` and `mixin_source_identity`, both excluded from representations.
This is a field-ownership census, not an increased code-size allowance.

Required CI previously selected 90 individual general-size nodes but omitted
the independent affected Library/screen guards. The existing fast-lane rider
list now uses the whole general guard, canonical screen guards, affected Library
ceiling/slack and discovery nodes, the export regression, wiring checks and new
literal-dependency controls. Every old general node remains covered and the
other 149 old targets retain their order. The target tuple changes 239 → 161.
The UI census retains its exact 1333-entry prefix and appends two exercised
extraction controls, with floor 1335. No lane, timeout, dependency set, required
aggregate or UI-census path rule changed.

Diagnostic review reads all three moved statements. A 132-call AST multiset
comparison restores only their original logger receiver spelling and proves
identical messages, arguments, levels and options. The inventory changes only
the four expected owner rows: Prompts controller −2, save/ack +2, ChatScreen −1,
command handoff +1. There are 728 owners and the same 16 sink files; no logger
disappears behind an unrecognized receiver or a new waiver.

## Verification

These are separate, sometimes overlapping cohorts; counts are not added together.

| Cohort | Result and qualification |
| --- | --- |
| Export refusal and ordinary compatibility | 27 passed; 116 typed portability/size controls passed separately. Exact old-helper injection produced 16 expected failures and 3 ordinary passes. |
| Three affected UI files | 119 passed in 157.52 seconds, including the new restore, navigation, disclosure and metadata behavior. |
| Named original native app controls | 28 passed in 107.57 seconds with a collection-only private-profile marker on the exact listed app cases; original bodies, assertions and admission remain unchanged. |
| New screen derivations and ordinary Browse worker node | Three helper tests and one exact ordinary CI node passed. Only the new Browse case has its required private-profile marker. |
| Affected architecture, ownership and size guards | 244 passed in 17.40 seconds, including all general/canonical-screen and the affected Library guards, wiring, layers, lookup census and import footprint. |
| CI contract and UI-census checks | 254 passed in 34.34 seconds; selectors resolve without overlap. |
| Worker static checker tests | 375 passed in 44.98 seconds. Current preflight has no new W001/W002/W003 findings and no census edits. |
| New literal-dependency controls | Before correction: 6 failed / 9 passed. Final corrected bytes: 15 passed. |
| Diagnostic inventory tests | 80 passed / 1 skipped in 195.52 seconds. The skip requires unavailable historical commits; current-source assertions remain enforced. |
| Diagnostic and markup preflights | Current inventory verified; markup census unchanged and passing. |
| Scoped lint/format | No new diagnostic multiplicities in each owned scope; new tests/helpers are clean. Existing formatter debt is separately recorded rather than called repository-wide clean. |

The five unchanged startup guards passed in isolation: **5 passed, 3 existing
headroom warnings in 29.20 seconds**. UI-ready remains 1031/1031 modules;
preimport is 554/558 modules and 426108/426199 LOC; largest Library entry is
132417/135111 LOC; warmup remains 93/94. The original boot-worker allowlist and
CSS ≤268 check pass. Source manifests for all 3450 product/performance/bootstrap
paths match before and after the run, and all owned processes have stopped.
This is startup qualification, not a new first-popup latency measurement.

## Evidence and limits

`artifact-map.json` retains **128 explicit source references**. Producer
inventories list exact owned files and known SVGs; no shared temporary-directory
prefix search supplies evidence. Relative paths and collision checks prevent
overwrites. Text wrappers and deterministic gzip preserve original bytes, and
every decoded retained artifact is checked against its source hash. A
credential-shaped-text precheck found no values requiring redaction in this
new set. Earlier phase redactions remain explicitly qualified and untouched.

The export, screen-extraction, UI and startup subdirectories retain exact
commands, source proofs, failures and results. Root command vectors and guard
proofs live in `root-checks/`. Earlier failed exploratory imports and fixture
setup/settlement mistakes are preserved as such, not described as product
failures. In particular, a root `--noconftest` architecture invocation could not
collect the UI-wiring module; its corrected run uses the repository's normal
private-profile bootstrap. The older unmarked native test invocation's four
passes and pre-body source-selection refusal remain separate from the 28
properly profile-owned original controls.

Unchanged baseline limitations remain explicit: the unrelated Skills controller
is 29 lines over its older cap; four historical Chat projection guards retain
older already-exceeded caps; and five unrelated quit-policy census entries match
the base exactly. Those files/assertions were not suppressed or relaxed, and no
full-suite or all-historical-guards-green claim is made. No new live-provider
inference/cache billing, cross-OS local execution or PTY/font tour was run.

Existing ADR-286/040/103 and the decomposition contracts govern these direct
corrections; no new ADR is required. Fresh hosted confirmation on the published
corrected head is tracked on PR #3346 before the owner-authorized ready/merge
operation completes.
