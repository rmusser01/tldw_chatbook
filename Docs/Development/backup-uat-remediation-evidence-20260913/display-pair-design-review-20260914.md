# Adjacent provider display-pair design review

**Conditionally suitable for one focused native TDD experiment.** No source edits, app runs or tests performed. The observed Windows timeout/config counts do not prove these two pairs dominate remaining time; do not implement solely from an assumed wall-time gain.

## Permissible boundary

In `_build_console_workbench_state:14912`, the blocker projection is immediately followed by the recovery-action projection. After the first returned readiness pair, those helpers only inspect immutable settings/readiness and build fixed display strings. There is no intervening native write, callback, await or UI mutation. Acquire through the unchanged `_active_console_settings_readiness()` at the original blocker-acquisition point and pass that local pair through the existing keyword-only API to both helpers.

In `_build_console_inspector_state:14274`, the same pair is separated only by `if setup_blocker_copy`. Preserve that conditional: when the blocker is empty, do not call the recovery-action helper. Use a distinct local such as `setup_settings_readiness`; do not overwrite or reuse the earlier inspector `settings_readiness` acquisition at14207. The substantial intervening inspector construction and its own reads must remain unchanged.

The full native/session/config derivation still finishes once before the shared presentation result is used, including any existing cold-session/default convergence. No larger derivation scope, retained screen attribute or operation lifetime is introduced. External changes after this point can be reflected by subsequent independent readers; the reused pair describes only these adjacent presentation values.

## Prefer explicit locals, not another combined helper

Two small explicit call sites use the already-reviewed optional-pair API without introducing another abstraction. The two sites differ: workbench always obtains recovery-action copy, inspector does so only when blocked. A combined helper either calculates a previously skipped action or needs a branching parameter/extra return protocol. Neither is simpler or more faithful than the local pair plus existing calls.

Keep the later workbench composer/draft reads, controller/run/Canvas-image state, `_console_setup_blocked_reason()` and send eligibility expression in their current order. In particular, do not pass the pair into that later send blocker. Preserve all independent helper defaults and all later inspector rows. Pair-level readiness count is2→1; the entire workbench may still do another readiness call for a nonempty draft, and inspector retains its earlier readiness read. Ready/active-run inspector paths that already skip action remain1→1 for this pair.

## Required native proof

- Extend an existing fixed-profile native fixture to call the actual workbench and inspector builders with real config/session/readiness helpers. Delegating counters should prove a reduction of exactly one completed native derive on blocked pairs; retain actual output assertions and count the unchanged earlier inspector/later populated-draft checks separately. Run the prepatch version to establish RED rather than only asserting a mock call count.
- Exercise credential-blocked, saved-ready and active-run states. Check workbench labels/copy/send eligibility and inspector Setup/Blocked impact/Next action rows. Inspector ready/active-run action remains uncalled; do not infer this from total counts alone.
- Perform a real config save/readback between complete builder calls; verify the next pair is fresh. Cold eligible convergence and explicit-user preservation can reuse the established guidance test machinery, distinguishing in-memory session state from durable config. Default helper calls elsewhere remain independently fresh.
- Ensure failure from acquisition and each projection propagates by identity, no UI result is partially published, and the next builder acquires afresh. Preserve callback/read ordering after the pair. A controlled real settings change at a later boundary must still reach the unchanged fresh send blocker rather than use the display pair as authority.
- Check the existing keyword-aware fixture lambdas from the guidance correction; do not add compatibility exception swallowing. AST comparison should show only local acquisition/keyword forwarding at these two sites; no change to whole-method predicates, earlier reads or conditional action ordering.

Native Windows qualification and before/after measurement remain separate from these correctness tests. No NTFS-query factoring or filesystem guard changes are part of this proposal.

Reviewed revision b7f51bdc20e0355dce8ac630e0c1ddbe4fcdb8f3; ChatScreen SHA256 `851c53a1299068a8e430f8307bdc5f56a28a73c99b9d40bf85fc1179273ed0ea`.

## Superseding workbench simplification

Full review of `Widgets/Console/console_workbench_state.py:16–153` confirms `provider_action_label` is a reserved signature/docstring parameter with **no expression/name read anywhere in the function body** (also checked by AST). Workbench recovery is always None and its actual Settings action uses fixed copy. Thus the preferred workbench proposal is now simply to remove `_console_provider_recovery_action()` and omit the `provider_action_label=action_label` keyword at ChatScreen's call. Keep the public/reserved callee parameter intact; deleting it would unnecessarily change unrelated callers.

This is smaller than the workbench pair plumbing proposed above. Preserve its original blocker call, then composer/controller/image state and independently fresh send blocker exactly. The first readiness derive still performs native checks and any eligible convergence. The removed call only re-derived readiness to produce a value this callee discards. There is no additional presentation use of its target/tooltip either. Removing that derive has the same display-snapshot qualification as sharing the pair: do not claim identical transient failure opportunities for the removed read.

Native proof should expect **zero workbench recovery-action calls**, not one, and compare the complete resulting WorkbenchState across blocked/ready/active-run cases. Empty-draft display derives become2→1; a populated draft retains its separate send-readiness derive. Verify config-save/next-builder freshness and unchanged send gating. Existing workbench contract callsites remain compatible because the callee default is retained. No callee/source edits or tests were performed by this reviewer.

The inspector proposal is unchanged: one distinct local at its original blocker point, action only if blocker is nonempty, earlier inspector readiness and subsequent row builders untouched. Do not introduce a shared helper now that workbench needs only deletion.
