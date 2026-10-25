# PR #3244 — current-dev rebase and integration review

Rebased the 24 portrait commits onto fetched dev
`2f0b142d42cfff88ad4cd44aa04cd9afe068fdf4`. The original published head is
preserved locally at `codex/portrait-pre-rebase-20261024`. Re-fetched dev before
publication; it remained at this commit. No merge commit, auto-merge or change to
the unrelated primary checkout is part of this work.

ADR required: no
ADR path: N/A — direct integration of existing ADR-242/243.
Reason: Preserve approved contracts while allocating migrations after the existing
Canvas predecessor; no new storage policy, provider, dependency or speech boundary.

## Resolutions and self-review

- Preserve dev's Canvas v81→v82 method and SQL exactly. Portrait storage is now
  v82→v83; regions/provenance are v83→v84. Register both steps in initialization,
  retain every predecessor recovery catalog, and expose the exact new catalogs and
  fixed native migration steps. Adapt real historical migration fixtures, failure
  rollback controls, policy counts and Canvas current-version assertions.
- Preserve dev's Personas `_wait_at_most` / `_whole_count` helpers and paging/save
  methods while retaining the portrait thumbnail extraction. AST review found no
  lost upstream-only definitions or duplicate class methods. Existing markup
  handling on dev already covered the added `markup=False` arguments; remove the
  duplicate automatic-merge arguments instead of replacing either implementation.
- Regenerate CSS through the builder, including dev's new file-notes modal sheet.
  The pair-review test now inherits dev's already theme-aware consolidated App
  directly. No product styling or authority rule was loosened to pass a test.
- Compile all 85 changed Python files. Strict F811 checks cover merged screen/DB
  class bodies. Three revised portrait fixtures pass Ruff; six touched small files
  pass format checks. Five legacy DB/recovery/Canvas files have no added diagnostics
  against the exact fetched-dev sources (618→618, 3→2, 1→1, 44→43, 0→0).
- Original composition/controller/probe implementations, including all three review
  fixes and the queued-cancellation correction, remain byte-identical to the
  pre-rebase head. All 15 historical capture hashes remain unchanged. Earlier
  evidence keeps its original source/schema label; it is not relabeled as v84.

## Fresh targeted evidence

| Gate | Passed | Baseline failures |
|---|---:|---:|
| Portrait/Canvas/response-rule migrations and recovery ownership | 89 | 0 |
| Compositor, repository, exports, SQL allowlist and packaging | 122 | 2 |
| Speech presentation, sinks, TTS ownership and renderer | 323 | 0 |
| Regions, generation, pair review and manual authoring | 80 | 0 |
| Console portrait/partial resize, pending saves and private hands-free journey | 53 | 6 |
| Inventory/interruption probe controls | 10 | 0 |
| CSS/token guards | 13 | 0 |
| Prompt boundaries (3 nodes repeat the core gate) | 15 | 0 |
| Fresh installed authoring/Console journey | 1 | 0 |
| Original native F9 restore/fresh open/component edit | 1 | 0 |

These selected feature/integration gates contain **704 distinct passing tests**;
repeated/diagnostic runs and extra legacy-wiring passes are excluded from that total.
Commands, exact counts and result identities are retained in
[verification.json](../qa/portrait-compositing/2026-10-24/rebase/verification.json).

Both successful wheels match **all 50 changed production files** and the actual
Canvas/portrait/region SQL chain. Native restore verifies v84 admission, exact source,
regions/provenance/frame hashes, equivalent PNG reuse, changed source/pixel refusal
and another one-request edit. The installed journey verifies real authoring/pair
review/Console controls with deterministic external image/TTS/device boundaries.
No direct test publication substitutes for the speech/animation owners.

The first combined native run timed out in its fresh child after restore validation
and reaching Personas. Test-only step evidence completed a follow-up run in 54.35s;
the instrumentation was removed byte-for-byte and the original isolated test passed
in 51.19s with unchanged timeouts. This does not establish a cause for the first
intermittent timeout or erase its failure. Both attempts remain distinguished from
the successful final result; no recovery/product workaround was added.

## Baseline failures and limits

All **51 failed node IDs** from the broader selected legacy UI/Actor Pack runs
also fail on an unmodified export of fetched dev. That baseline has 52 failing
nodes; the changed hands-free journey uses its existing private-profile wrapper and
passes on this branch. There is no branch-only failed node in those runs.

The branch failures are 48 legacy UI config/admission fences
(`raw_source_selection_changed`), one stale acoustic-barge-in config-call
expectation, and two Actor Pack golden-byte mismatches. Golden production sources,
package version and fixtures are byte-identical to dev. These targeted commands do
not establish that full-suite/CI execution fails the same way. We retain failure
node sets and source identities; no guard bypass, expected-byte regeneration,
fixture-wide reset or unrelated product change was used to hide them. An initial
mixed run was interrupted after failures; it is excluded from the passing totals.

No full test suite, new live provider call, new hardware/DAC or live TTS
qualification was run. This is local integration evidence, not a CI-green claim.

## Live-provider qualification remains open

TASK-36555 stays **In Progress**, AC5 unchecked. Fresh inventory still finds SwarmUI
as the default and only enabled backend; this app's reference-image path does not
support it. **Zero image requests**, no fallback, `live_qualified=false`.

Historical external ChatGPT patches qualify local compositing/import and review,
not the configured in-app adapter. Real three-edit generation, selected retries and
semantic/alignment/lighting/seam review still require a reachable supported adapter.
The explicit five-request probe remains opt-in and cannot qualify artwork without
review. The PR stays draft with this gap prominent.

Evidence: [rebase/](../qa/portrait-compositing/2026-10-24/rebase/),
[ADR-242](../../../backlog/decisions/242-console-radio-portrait-performance.md),
[ADR-243](../../../backlog/decisions/243-source-bound-portrait-compositing.md).

Self-review outcome: the merged migration lineage, preserved upstream helpers and
three original fixes meet their approved contracts. No unresolved branch-only defect
was identified by this review; the baseline failures, initial timeout and live-provider
qualification limit remain explicitly documented above.
