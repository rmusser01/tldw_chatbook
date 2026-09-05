# Dev test review checkpoint — 2026-09-05

This is an **in-progress review**, not a green full-suite or merge-ready claim.
The user requested a saved draft PR and another rebase to absorb dev churn.
All work is isolated from the original dirty checkout.

## Checkpoint scope

- Console and Library controller decomposition, ownership/AST ratchets, and
  first-use import repairs, with targeted behavioral and import-closure evidence.
- Runtime repairs include Buddy teardown without a screen, retained Notes focus
  and editor identity during Files handoffs, and three invalid splash effects.
- Test repairs restore current provider/persistence/authority contracts, real
  styled controls, attached-widget readiness, isolated resource measurements,
  and diagnostic privacy fixtures. Security/resource limits were not relaxed.

## Evidence before the new rebase

Independent complete affected-file selections include:

| Selection | Result |
| --- | --- |
| Diagnostic inventory and privacy | 327 passed |
| Buddy and Models adoption | 171 passed |
| MCP gateway tools and prompts | 143 passed |
| Audio.cpp handoff | 122 passed |
| vLLM workflow and Console provider apply | 140 passed |
| Raw CLI processes | 51 passed, 1 Windows-only skip |
| Persona publication | 53 passed; parent-descriptor pressure probe also passed |
| Historical migration, SQLite privacy, workspace roots | 159 passed |
| Console settings | 416 passed |
| Console transcript | 165 passed |
| Console exchanges | 47 passed |
| Console state and generation actions | 83 passed |
| Splash, Cast, Watchlists pagination and rebuilds | 269 passed |
| Scheduler, TTS ownership, Watchlists busy runs | 125 passed |
| Evals and interoperability | 352 passed, 6 existing unfinished-feature skips |

These selections overlap earlier sweeps and must not be added into a unique-test
total. Three staged non-UI sweeps reached 16,203, 12,440, and 5,376 passes before
stopping on distinct failure families. Two staged UI sweeps reached 2,900 and
2,397 passes. Their original failures were retained as a diagnosis ledger, not
silently treated as passing after code changes. The remaining unexecuted cases
and post-rebase integration still need completion.

## Open at the checkpoint

- TASK-31717: final integrated Console/decomposition verification and closeout.
- TASK-31707: oversized trace boundary inputs and cold reserved-call clock setup;
  diagnosis recorded, no implementation yet.
- TASK-31708: agent gateway/gate fixture signatures and regeneration failure
  reporting; diagnosis recorded, no implementation yet.
- TASK-31710: Console journey phase synchronization has six targeted passes;
  its complete-file run was intentionally stopped after 48 passes for rebasing.
- TASK-31711: Files-to-Notes browse scroll restores 6 instead of logical offset 7.
  The interrupted Notes workspace run reached 105 passes and this one failure.
- TASK-31712: thread-start fault injection mutates shared stdlib threading and
  can cause test-runner teardown warnings; diagnosis only.
- Unallocated: Notes Save-failed contrast in the light theme, pinned sync-history
  paging geometry, and load-sensitive Qwen retry/MCP child cleanup failures.
- Re-run the architecture, diagnostic, screen-size, preimport and UI-ready
  ratchets after the new rebase; their pre-rebase results do not qualify new dev.

## Environment qualification

Subprocess tests use an isolated installed review environment so `python -I`
children resolve this checkout, not the original workspace. Writable Notes
fixtures use the per-user macOS temporary directory with correct UID/GID;
`/private/tmp` inherited `wheel` and correctly failed metadata guards. The
workspace tool executor's nested-environment tests were separately qualified
with the native project environment. Platform, configured-service, and
unfinished-feature skips are not proof of executed coverage.

No full-suite completion or merge readiness is asserted by this checkpoint.

## Rebase and draft PR

[Draft PR #2427](https://github.com/rmusser01/tldw_chatbook/pull/2427) preserves
this progress. The review was rebased onto
`da2fbdbc212d16030bb2802a91944527c5db43e7`; a second fetch confirmed that dev tip
before publishing. This incorporated 73 upstream commits since the previous
review base and replayed 109 review commits. The local backup branch
`codex/dev-test-review-before-rebase-20260905` preserves the prior checkpoint.

Conflicts retained upstream last-good Scheduling display and async reachability
checks, alongside unmounted-screen guards and first-use imports. Console timer
tracking/cancellation was retained with the settings-navigation controller.
The diagnostic inventory was rebuilt from the merged owners, preserving the
upstream additions and reviewed controller movements. Review-only Backlog ID
collisions are renumbered; upstream task identities are preserved.

Post-rebase evidence:

- All 195 changed Python files parsed; branch whitespace checks passed.
- Scheduling, Library reuse, import closures, migration and workspace roots:
  241 passed, 2 failed. Both failures exposed the same newly added Console
  suspend caller still targeting a helper moved to the settings-navigation
  controller. Integration plan: retarget that call to the existing owner and
  rerun the complete reuse and settings-return selection. ADR required: no;
  this preserves an existing owner boundary, not a new lifecycle policy.
- Architecture/preimport selection: 44 passed, 3 failed. Console is 17,541
  lines against 16,873; Library is 41,651 against 41,324; preimport adds 504
  modules against 500. The ceilings remain unchanged. Upstream growth needs
  further decomposition/import work before this draft can be merge-ready.
- The suspend caller was retargeted to the existing settings-navigation owner.
  Both complete Console/Library reuse files now pass: 8 tests in 32.25 seconds.
  Full-screen Ruff and changed-line formatting also pass.
- The broader Console reuse/settings-return selection produced 32 passes and
  3 failures. The failures still expect navigation to create a fresh Console or
  cancel an unmount worker; upstream now reuses/suspends the screen. The final
  failure's cancellation-suppressing fixture required interrupting teardown.
  These test journeys need adaptation; their existing handoff assertions have
  not been relaxed for this checkpoint.
- Diagnostic inventory verification reports no drift: 584 owners, 1,336
  TASK-492 calls, 7,615 TASK-494 calls and 11 sink files.
