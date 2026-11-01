# PR3301 backlog integration — verification in progress

Source6e5c370c6e57d16f6d0b1c892434e3cc638218e5, initial rebase6b0dac4d238da6de9ea749c94ad2f52d03dd35db on db7e04cf484b4c0b803366b00f2bbae1b5550042. Current-base rebase and final exact-head review remain required. Hosted CI and size limits are waived; drafts and PR3267/3297 are excluded, and this session performs no release.

Initial independent review found no Critical/Important issues and two Minor issues. The clear-all image action now says **Remove images**, and scoped hover uses the existing shared hover tokens while focus remains distinct. No attachment-clearing or provider authority behavior changed. Existing ADR150/161/031 and blocker precedence ADR098 apply; no new ADR.

## Targeted evidence

All receipts are in the PR3301 preparation parent, with frozen before/after source hashes and command/log/JUnit/result artifacts.

- `r-9130e827`:2 causal failures,0errors/skips,35.493s pytest. A real two-image staged row still said Remove image; actual production CSS masked the shared hover background.
- `r-2a047b06`:29passed,2failed,0errors/skips,284.965s pytest. All capability cases and shared-hover/focus, narrow/wide painted labels and other adjacent controls passed. The two failures observed stale control-bar rendering after clearing and in the missing-model setup state. This is not an all-green31-case run.
- `r-c584ff12`:2failed,0errors/skips,36.381s pytest. Diagnostic staging observed the actual synchronous refresh returningFalse while attach had not reconciled; the no-model failure shifted to the post-selection row visibility. Source inspection confirms that this current-dev API can defer to its existing0.2s coalesced fresh-state refresh. The tests previously waited only one Pilot pause. The two failed cases now await their actual rendered outcome within5s; no manual refresh after button action, guard replacement, source mutation, product timer or timeout change. The follow-up outcome is qualified below.

The initial reviewed implementation retains explicit user text-only capability overrides and models.dev context metadata. Historical provider API/image calls in the original tasks are author evidence, not new local provider execution. No full suite or native terminal paint is claimed.

## Current-base integration and review

Rebased onto PR3299 merge63b764c858cc5aedeba7f0c7ba28b57500e90568 at7f9771d62f0af09c77a932cef70caaf2e290e3ef. Production merged automatically; metadata retained the complete1246-node dev census prefix plus3 distinct new nodes (1249 total). Exact current measurements are ChatScreen24517lines/741methods, composer7154lines, display2908lines. Final rebase onto PR3300 remains pending.

Independent integration review found no Critical/Important source issues. Both Minor documentation findings are fixed: the wrapped Remove label is plural, and the provider architecture describes explicit overrides first, then the Claude4+ vision overlay preserving models.dev metadata. The reviewer verified both corrections and the outcome-based render waits.

`r-4a833b59`:3 passing test bodies,1teardown error,0failures/skips;36.636s pytest,42.16s wall, frozen sources. Removal/draft retention and shared hover/focus passed. The no-model case's assertions passed, but default-executor shutdown exceeded its3s limit and config retirement refused uncertain workers. This is not a clean3-test result. No shutdown, admission, retirement or timeout guard was changed. A fresh exact-node run remains required.

`r-e83d8616`: the exact no-model case passes with clean teardown (1PASS,0failures/errors/skips;22.491s pytest,27.86s wall; frozen sources). The earlier teardown error remains recorded above. Final-base integration after PR3300 is still required.
