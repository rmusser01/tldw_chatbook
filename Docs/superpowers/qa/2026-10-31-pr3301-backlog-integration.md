# PR3301 backlog integration — verification in progress

Source6e5c370c6e57d16f6d0b1c892434e3cc638218e5, initial rebase6b0dac4d238da6de9ea749c94ad2f52d03dd35db on db7e04cf484b4c0b803366b00f2bbae1b5550042. Current-base rebase and final exact-head review remain required. Hosted CI and size limits are waived; drafts and PR3267/3297 are excluded, and this session performs no release.

Initial independent review found no Critical/Important issues and two Minor issues. The clear-all image action now says **Remove images**, and scoped hover uses the existing shared hover tokens while focus remains distinct. No attachment-clearing or provider authority behavior changed. Existing ADR150/161/031 and blocker precedence ADR098 apply; no new ADR.

## Targeted evidence

All receipts are in the PR3301 preparation parent, with frozen before/after source hashes and command/log/JUnit/result artifacts.

- `r-9130e827`:2 causal failures,0errors/skips,35.493s pytest. A real two-image staged row still said Remove image; actual production CSS masked the shared hover background.
- `r-2a047b06`:29passed,2failed,0errors/skips,284.965s pytest. All capability cases and shared-hover/focus, narrow/wide painted labels and other adjacent controls passed. The two failures observed stale control-bar rendering after clearing and in the missing-model setup state. This is not an all-green31-case run.
- `r-c584ff12`:2failed,0errors/skips,36.381s pytest. Diagnostic staging observed the actual synchronous refresh returningFalse while attach had not reconciled; the no-model failure shifted to the post-selection row visibility. Source inspection confirms that this current-dev API can defer to its existing0.2s coalesced fresh-state refresh. The tests previously waited only one Pilot pause. The two failed cases now await their actual rendered outcome within5s; no manual refresh after button action, guard replacement, source mutation, product timer or timeout change. This correction is not yet verified.

The initial reviewed implementation retains explicit user text-only capability overrides and models.dev context metadata. Historical provider API/image calls in the original tasks are author evidence, not new local provider execution. No full suite or native terminal paint is claimed.
