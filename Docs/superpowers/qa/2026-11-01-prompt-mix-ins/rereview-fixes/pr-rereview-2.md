# PR3346 second scoped re-review

Fix base: `1a58f0af2cbc5129f9c8f01595d67c330a7defda`.
Reviewed product/test head: `a443d71c5ceca74feca6d2fc5af1d6e29c682ccc`.
Parent reports evidence-only follow-up `e7be3ebfba3a057057964b7787e0b0b0e49c250d`; product/tests remain unchanged.

Scope: R1/R2 from `pr-rereview.md` and new breakage in their fix diff. Original I1/M1 and approved CI corrections were not broadly re-reviewed. No subagents, product/test/index/HEAD edits, Qodo, full suite, or redundant test reruns.

## Strengths

The changes address both reproductions in their existing owners. Placement publication distinguishes a current observation from a frozen execution snapshot; artifact opening retains worker responsiveness while checking ownership after the async manual-target boundary. Tests exercise the native controller/SQLite and actual mounted event route rather than only asserting helper values.

## Finding verdicts

### R1 — closed

`MixinCaptureWitness` carries the captured preference revision without granting frozen configurations fresh observation authority. `publish_accepted_usage` retains accepted placement/order only while that revision still matches, and advances the intent revision if layout changes. Used-version-only facts remain independent. Reconciliation preserves current names, skips removed/unavailable selections, reads no source, and cannot overwrite a newer user intent.

Current manual/queue and Regenerate/Edit/Continue paths supply observation provenance at their acceptance boundary. Received manual publication commits the captured draft before publishing layout; the queued branch's draft commit uses its existing draft-specific identity check. Original Retry, queue drain and old supplied complete configurations do not acquire layout authority. Pre-acceptance witnesses and old Workbench intents become stale after accepted layout changes; accepted execution is already past that fence.

The added native regression reproduces the exact successive A/B case and asserts `second / body B / body A`. Its real SQLite flush/reopen variant hydrates the existing Library policy before a new Send, rather than bypassing that policy gate. Additional controls cover delayed user edits, received manual/queue acceptance, frozen supplied requests, one-time queue observation, current replay routes, refusal and stale intents. The ADR now describes retained accepted observations rather than the previous Workbench-only layout restriction.

### R2 — closed

`open_artifact` captures mode-generation, detail-token and source ownership. It checks that ownership after detail/capability reads and passes it through the manual-target await for both ordinary structured Prompts and selected Recipes. The target remains local until that check succeeds; stale completion returns before publishing target, artifact, editor or navigation state. Authorized mode transitions occur afterward, so the open does not invalidate itself.

The worker dispatch remains, preserving nested confirmation/Cancel responsiveness. Mounted actual-event regressions hold the target callback, navigate to Mix-ins, edit Instructions, release the callback, and assert the same dirty editor and text survive. Source-change and actual removal cases also assert no stale manual target. Normal Prompt/Recipe/legacy and saved-Recipe chooser controls cover the successful paths; existing real confirmation tests remain green.

## New issues

- Critical: none.
- Important: none.
- Minor: none.

Original I1's authoritative checkbox/count restoration and M1's provisional capture remain intact. No new SQL/index/recovery, diagnostic inventory, markup census, or worker census weakening appears in this diff.

## Evidence assessed

Read the committed follow-up report, source/test diff, targeted command/results and bounded log summaries. The recorded results are 6 passing corrected controls, 22 passing replay controls, 18 passing final controls plus one test API typo subsequently corrected, and 3 passing corrected Edit/received controls. Counts overlap and are not summed. The real reopen setup correction is supported by the reported native Library-policy refusal and uses production hydration. Textual removal is asserted via actual screen-stack absence and unmount generation rather than relying on sticky `is_mounted` state.

The three affected preflight guards pass without repins. The static record shows no new duplicate methods across nine touched Python files and no newly failing formatter file; per-file Ruff remains 200/200 with zero additions. The reviewed tree was clean. No additional test run was necessary because no specific unanswered code doubt remained after inspecting these controls.

The source-head shallow-clone receipt verifies 756 current records, one reachable commit, no alternates, and explicit absence/UNAVAILABLE for all three historical source commits. The parent reports 757 current records for the subsequent evidence-only head. Historical actual-terminal captures remain pinned to their captured revisions and are not claimed as current fix execution.

## Assessment

**Ready to merge? Yes, for the reviewed feature/fix scope, subject to the repository's normal PR checks.**

R1 and R2 are closed and no new actionable issue was found in the fix diff. Original I1/M1 and the previously approved CI corrections remain closed. This review does not convert the disclosed target-dev failures or broad-run FD warning into green results.

Skipped full-suite, external-provider/cache, and new actual-terminal whole-journey runs. Existing disclosed baseline/FD/live-qualification limits remain; no new risk identified in this scoped pass.
