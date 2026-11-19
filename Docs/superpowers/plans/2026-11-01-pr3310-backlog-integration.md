# PR3310 current-dev integration

Integrate TASK-33620.15.26 onto dev b1e19a1dba935a599186d8a252cb01a5ecf54abf, preserving its accepted composer-height and view-only echo-stand-in behavior.

ADR required: no new ADR.
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md (existing Console lifetime contract).
Reason: preserve the existing send/custody and presentation boundaries; no new storage, runtime or UX policy.

1. Rebase the thirteen incoming commits; retain current-dev captured-input custody, acknowledgement-first receipt, guard warning, Terminal focus cue and typed attachment blocker. Keep all1263 base census entries and append the five incoming selectors once.
2. Requalify both mounted routes against current acknowledgement-first behavior. Tighten the stock oracle to require the echo and reply placeholder in the same reflow, matching the current route rather than the historical source-base split.
3. Run only the changed acknowledgement/layout tests, receipt/custody suite, selected draft/attachment/Terminal controls and census contracts in private profiles on frozen source. Compare any failure with the exact dev baseline before classification. Size limits and hosted CI are waived by the user; do not weaken behavior checks.
4. Obtain independent requesting-code-review review of the complete integrated delta and evidence; fix Critical/Important findings. Keep tracked .36 timing-oracle and .37 quiet-release limitations explicit unless addressed and verified.
5. Record current evidence separately from historical native/performance claims, refresh live non-draft state/head/threads, publish with exact force-with-lease, merge and verify actual merge commit and restored protection. Never merge drafts, act on3267/3297, or perform release work.

## Review repair: acknowledgement-owned geometry

The independent integrated review traced Queue full, chainless run hold and a second edited press during another turn's Preparing state: unconditional `_schedule` marking can attach the height clamp to a refused capture. Reproduce with real acknowledgement scheduling and a mounted composer. Mark only a capture that receives its own acknowledgement token, before painting; leave an older capture marker intact on refused repeats. Narrow the compositor hold to the canonical Preparing reason so other queue refusals re-wrap normally. This is a correction of the existing exact-send presentation contract, using ADR094/230 ownership; no new authority, state owner or ADR.

The two initial mounted layout tests timed out before the provider during cold warm-up: diagnostics show durable acceptance occurring after the 15-second harness wait. Diagnose separately from the measured third send; retain the layout assertions and measured-send deadline. Record any setup qualification changes and compare legacy failures against exact dev.
