# Native workflow PR integration

Goal: publish the native phased shared-worktree workflow feature and coroutine fix against current origin/dev, preserving all unrelated checkout/index bytes.

ADR required: no new ADR expected
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: integrate the approved feature with newer upstream owners; amend this existing decision before changing migration/compatibility mappings.

## Constraints

The original checkout remains on its existing branch and retains its working files and index. Use a temporary Git index and ignored in-repository integration directory, not a registered worktree or stash/reset/clean. Preserve current upstream changes, shipped migration25 and original61 staged support files. Keep only the intended feature/bugfix scope; exclude unrelated local radio/mobile/prompt-mixin design work. No plain force push, merge or auto-merge. A necessary topic-only rebase uses an exact force-with-lease as documented below. Run targeted affected checks; no full suite. Root controls scope/governance/review and publishing.

## Execution

- [x] Capture current protected checkout/index baseline and fetch current upstream.
- [x] Confirm exact local feature ownership and current upstream dependency/schema/runtime compatibility.
- [x] Build scoped feature tree using a temporary index, three-way integrate with current upstream, materialize source in the ignored task directory, and preserve upstream behavior when resolving conflicts.
- [x] Update actual migration/catalog/recovery mappings under ADR285 before schema edits; retain known legacy data conservatively and test the approved path.
- [x] Publish the scoped integration checkpoint as a draft PR; keep it draft while behavioral/static fixes and the independent review remain pending.
- [x] Run affected clean-source contract, DB/recovery, runtime/ownership and Console checks with exact source receipts; fix actual integration regressions.
- [x] Obtain independent integration code review, address blocking findings and reverify current tree.
- [x] Push final reviewed corrections, mark PR ready only after the gates pass, and report actual validation/CI state.
- [x] Reverify original branch/index/protected bytes; retain scoped integration evidence.

### Dev advancement before readiness

Initial integration and earlier qualification use dev3c39215749a96d5c1b8c05972fdd2aeb7588b002. During final validation dev advanced to29592ecc9bf3856957a8d9b70a270795b8f4348d and the draft became conflicting. Retain the immutable old-base checkpoint and all qualified receipts, inspect the72-path upstream delta with a read-only audit, rebase only the owned topic branch through Git plumbing in the same ignored source export, preserve upstream changes and resolve actual intersections through the sole writer. Review the precise new delta and rerun affected UI/coroutine/static checks before readiness. Lease-protect the remote topic update and preserve original dev/HEAD, staged index and all292 protected files. No merge or auto-merge. ADR required: no additional ADR; this is integration workflow under existing ADR285 and the repository’s rebase policy.
