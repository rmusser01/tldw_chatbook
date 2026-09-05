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

- TASK-31642: final integrated Console/decomposition verification and closeout.
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
