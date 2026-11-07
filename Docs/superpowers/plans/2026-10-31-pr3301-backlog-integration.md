# PR3301 backlog integration

Rebase and qualify the existing Claude image-capability and typed attachment-block change. Merge only after PR3299 and PR3300 and a fresh independent Superpowers review. Drafts and PR3267/3297 are excluded. This session does no release or main promotion. Hosted CI and size limits are waived; targeted local evidence remains required.

ADR required: no new ADR.
ADR paths: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/098-visible-bounded-console-prompt-queue.md (existing blocker/queue precedence).
Reason: preserve the existing provider/capability precedence and attachment authority, and apply the existing button-state contract; no storage, permission or provider boundary changes.

Source: 6e5c370c6e57d16f6d0b1c892434e3cc638218e5. Rebased preparation: 6b0dac4d238da6de9ea749c94ad2f52d03dd35db on db7e04cf484b4c0b803366b00f2bbae1b5550042. Rebase again after earlier PRs merge.

1. Preserve current dev's ownership/readiness behavior, shared composer changes, task identities and complete census prefix. Re-measure affected size metadata under the explicit waiver.
2. Address initial review's two Minor issues: the action clears all staged images, so label it Remove images; preserve the shared hover colors under the feature row's more specific selectors. Existing focused style wins over hover. Keep clear-all behavior and draft retention unchanged.
3. Extend the existing mounted removal test to two images and accurate label; add a computed production-CSS hover/focus comparison. Observe causal failures before the small source edits, then rebuild generated CSS and verify the affected controls.
4. Run targeted capability, attachment, composer, narrow/wide UI and census/CSS contracts. Do not run a full sweep or claim historical provider/live-terminal evidence as newly reproduced.
5. Record actual results and limitations, finish task notes, commit, obtain exact-head independent review and merge in order after confirming current remote state.
