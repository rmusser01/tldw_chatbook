# PR3301 backlog integration — final targeted evidence

Source6e5c370c6e57d16f6d0b1c892434e3cc638218e5, initial rebase6b0dac4d238da6de9ea749c94ad2f52d03dd35db on db7e04cf484b4c0b803366b00f2bbae1b5550042. Final integration now rests on PR3300 merge59b8fe02499ffdfb7498999313d0f39e585c17e4; the completed evidence is below. Exact committed-head approval is recorded in the external independent review. Hosted CI and size limits are waived; drafts and PR3267/3297 are excluded, and this session performs no release.

Initial independent review found no Critical/Important issues and two Minor issues. The clear-all image action now says **Remove images**, and scoped hover uses the existing shared hover tokens while focus remains distinct. No attachment-clearing or provider authority behavior changed. Existing ADR150/161/031 and blocker precedence ADR098 apply; no new ADR.

## Targeted evidence

All receipts are in the PR3301 preparation parent, with frozen before/after source hashes and command/log/JUnit/result artifacts.

- `r-9130e827`:2 causal failures,0errors/skips,35.493s pytest. A real two-image staged row still said Remove image; actual production CSS masked the shared hover background.
- `r-2a047b06`:29passed,2failed,0errors/skips,284.965s pytest. All capability cases and shared-hover/focus, narrow/wide painted labels and other adjacent controls passed. The two failures observed stale control-bar rendering after clearing and in the missing-model setup state. This is not an all-green31-case run.
- `r-c584ff12`:2failed,0errors/skips,36.381s pytest. Diagnostic staging observed the actual synchronous refresh returningFalse while attach had not reconciled; the no-model failure shifted to the post-selection row visibility. Source inspection confirms that this current-dev API can defer to its existing0.2s coalesced fresh-state refresh. The tests previously waited only one Pilot pause. The two failed cases now await their actual rendered outcome within5s; no manual refresh after button action, guard replacement, source mutation, product timer or timeout change. The follow-up outcome is qualified below.

The initial reviewed implementation retains explicit user text-only capability overrides and models.dev context metadata. Historical provider API/image calls in the original tasks are author evidence, not new local provider execution. No full suite or native terminal paint is claimed.

## Current-base integration and review

Rebased onto PR3299 merge63b764c858cc5aedeba7f0c7ba28b57500e90568 at7f9771d62f0af09c77a932cef70caaf2e290e3ef. Production merged automatically; metadata retained the complete1246-node dev census prefix plus3 distinct new nodes (1249 total). Exact current measurements are ChatScreen24517lines/741methods, composer7154lines, display2908lines. This was the intermediate base; the completed final rebase is recorded below.

Independent integration review found no Critical/Important source issues. Both Minor documentation findings are fixed: the wrapped Remove label is plural, and the provider architecture describes explicit overrides first, then the Claude4+ vision overlay preserving models.dev metadata. The reviewer verified both corrections and the outcome-based render waits.

`r-4a833b59`:3 passing test bodies,1teardown error,0failures/skips;36.636s pytest,42.16s wall, frozen sources. Removal/draft retention and shared hover/focus passed. The no-model case's assertions passed, but default-executor shutdown exceeded its3s limit and config retirement refused uncertain workers. This is not a clean3-test result. No shutdown, admission, retirement or timeout guard was changed. The later exact-node result is recorded below.

`r-e83d8616`: the exact no-model case passes with clean teardown (1PASS,0failures/errors/skips;22.491s pytest,27.86s wall; frozen sources). The earlier teardown error remains recorded above. That was prior-base evidence; final-base integration follows below.

## Final base and qualification

The 26 commits rebased onto actual dev `59b8fe02499ffdfb7498999313d0f39e585c17e4` at preparation head `1919ab439efe8276a67b0593040f12d21d9508dd`. Only screen-size metadata conflicted; both historical comment blocks survived. The independent reviewer mapped 25 patches identically and the remaining metadata-context change. Terminal composer hiding, input ownership, focus/cues and guarded return restoration remain intact. Exact final measurements: ChatScreen 24,602 lines / 741 methods, composer 7,154 lines, display 2,908 lines.

- `r-6cd4fd69`: 14 passed, 1 failed, no errors/skips; 109.247s pytest, 114.14s wall. Removal/draft retention, no-model precedence, switch-focus restoration, narrow/wide paint, hover/focus, inherited Terminal cues, six exact size checks and generated CSS reproduction passed. The generic vision test read the old disabled strip before the existing deferred refresh completed; the real classifier already returned the correct refusal.
- The shared `_stage_image` helper now observes the expected real strip within the existing five-second bound. No-model and long-model cases supply their explicit expected text. This replaces single-pause assumptions without forcing publication or changing a product guard. Post-action removal and model-transition waits still observe real outcomes.
- `r-c1f4b3c4`: 10 passed, 1 failed, no errors/skips; 137.836s pytest, 142.52s wall. The generic strip case and all other attachment cases passed; Switch model did not open because the new predicate could already be true and skip the original layout yield before clicking. Restoring that unconditional Pilot layout yield preserves the old fixture behavior in addition to waiting for the async state. The test now also asserts that its actual button click hits.
- `r-f33e7f70`: the two affected generic-strip and Switch model cases pass with clean teardown (2 passed, no failures/errors/skips; 28.050s pytest, 32.16s wall). This completes passing evidence for the affected module; it is not presented as one clean eleven-case invocation.

All receipts froze their sources. No production change followed the final-base tests. Production hash continuity across these three receipts is recorded in `final-production-receipt-parity.json` beside the worktree. The 1,249-entry UI gate inventory validates at its unchanged floor; the full current-dev prefix plus the three incoming entries is retained. Focused Ruff comparison of all six changed production files finds no introduced diagnostic against actual dev59b8fe0249, recorded in `lint-parity-final-base.json`; the modified attachment UI test passes Ruff and `git diff --check` passes.

The 18 capability cases and two vision-reason controls passed in the earlier targeted31-case invocation, whose two UI failures are explicitly preserved above. Capability implementation is unchanged by the final rebase. Historical author provider/live evidence remains historical. No full suite, release, or main promotion occurred; drafts and PR3267/3297 remain excluded.
