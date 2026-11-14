# PR #3244 — final hosted CI corrections

The complete run on `fda2a3ee96` finished with 27 passing checks, two skipped
checks and three failures: UI shards 11/15 and their required summary. There
were no submitted reviews or inline threads. CodeRabbit skipped the draft;
its successful status does not represent review approval.

CI tested merge `3963966e`, including newer dev `6fcbded875` and its newly
added contrast sweep. Rebased onto that exact dev revision without conflicts.
The final production tree differs from that tested merge in only the pair-viewer
CSS declaration and its app-tier/generated stylesheet owners; speech, storage
and provider behavior remain the already-tested implementation.

ADR required: no
ADR path: N/A — existing ADR-150/161/242/243 apply.
Reason: Correct existing focus and test-gesture contracts, without new boundaries.

## Corrections and review

- Shard 15 found the pair-canvas focus outline under 3:1 on 29 themes.
  The existing `$ds-active-fg` token clears the floor, but changing it only in
  widget-default CSS did not fix mounted output: the app's global `*:focus`
  still won and pastel_dreams stayed at 1.03:1. Move the one targeted outline
  into app-tier `components/_dialogs.tcss`, remove the shadowed declaration,
  and rebuild both generated sheets. No palette, shared token, contrast floor
  or debt exclusion was changed. All 89 themes pass; mounted light/dark tests
  assert actual focus ownership, tall outline and computed contrast while
  retaining pan, resize, original-frame and keyboard-close assertions.
- The new standalone static contrast guard needs the established
  `bootstrap_profile` ownership marker. Without it the targeted invocation
  stopped at config admission instead of testing contrast. Add that marker
  to this function, preserving every assertion and admission fence.
- Shard 11 clicked an offscreen Library receipt. A controlled real recompose
  after finding the receipt reproduces this on both the branch and unchanged
  dev production: scrolling the detached button leaves its replacement offscreen.
  Re-query, focus and immediately scroll the current button; wait for its center
  to be in the viewport and hit-test to it before the actual pointer click.
  Keep both ordinary/replaced variants and all real-ingest, Reader, selected-id
  and immediate Read later assertions. No Library production code, fixed delay,
  timeout increase, force-click or event bypass was added. An intermediate
  receipt/census attempt clicked without opening the Reader; explicit focus
  readiness fixes that repeated invocation and remains recorded in evidence.

## Targeted verification

- 114 contrast, discovery/debt, pair-viewer, CSS/token and boot-CSS checks pass.
- All nine Library opened-item tests pass. The final receipt/census invocation
  also passes all 74 cases, including both receipt variants.
- Fresh installed authoring/Console journey passes in 41.83s. Its wheel matches
  all 53 production files changed relative to the rebased dev source.
- Focused Ruff/format and syntax checks pass; two legacy test files add no Ruff
  diagnostics versus their exact pre-change baseline. CSS was rebuilt from source.

[Bounded evidence](../qa/portrait-compositing/2026-10-24/final-ci-corrections/)
retains the completed failed run, red/green controls, intermediate failures, wheel
identity and fresh zero-request inventory. No full suite was run. The published
replacement head must pass its own complete hosted run before a green-CI claim.

## Subsequent receipt-identity correction

On `f442237200`, the contrast shard is green. Shard 11 instead stops at the
extra `assert await pilot.click(button, ...)` added during the first correction.
Textual captures a target before its own pre-mouse pauses and returns an object
identity comparison; a recomposed receipt can be successfully clicked while
that comparison returns False. A real recomposition during the first mouse pause
reproduces the False result deterministically. Remove that incidental assertion,
retain the actual pointer gesture and all Reader/canonical-id/saved-item outcome
assertions, and cover ordinary, before-scroll and during-click replacement.

The complete Library/census selection passes **82 tests**. The controlled late
case starts red at the identity assertion and passes with the correct Reader
and DB result after its removal. No production code, timing budget, mouse-event
implementation or storage/admission guard changes in this correction; installed
production-source identity from the prior checkpoint remains applicable.
[Late-click evidence](../qa/portrait-compositing/2026-10-24/final-ci-corrections/late-receipt-identity.json)
keeps that additional failed attempt distinct from the final assertions.

## Qualification remains explicit

TASK-36555 remains In Progress, AC5 unchecked. Fresh inventory is SwarmUI-only,
unsupported by the reference-image route: **zero live image requests, no fallback,
`live_qualified=false`**. Earlier Pocket TTS 3.3.0/CoreAudio qualification is recorded
separately against its original source; these corrections do not repeat live TTS,
physical loopback or microphone/STT hardware checks. The PR remains draft.
