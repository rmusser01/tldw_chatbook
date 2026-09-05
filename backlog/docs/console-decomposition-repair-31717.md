# Console decomposition repair — TASK-31717

The rebased development screen measured 21,882 physical lines and 677 direct
AST methods. The repair measures 16,873 lines and 559 methods, below the
unchanged 16,966 / 563 ceilings. Shared ratchet reconciliation remains owned
by the coordinating repair task.

TASK-31750 continuation: the approved 64 private forwarding methods have been
removed, and real callers and fault-injection receivers target their existing
controllers. No new owner boundary was introduced. The measured pre-second-rebase
screen is 16,818 lines / 505 methods; the ratchet now uses those tighter counts.
Public/framework actions, agent-bridge injection and the transcript ephemeral
probe remain on the screen. Two new callback tests preserve delayed lookup and
replacement-owner behavior. The 145-test architecture/import/inventory selection
passed. The affected-file run is incomplete (1,749 passed, seven failures before
interruption); see the main checkpoint report. This is not final closeout evidence.

This implements DESIGN.md §7 and the approved screen-decomposition design.
No new ADR is required: ADR-033's application-session ownership remains intact.

## Ownership

| Controller | Responsibility moved from the screen |
| --- | --- |
| `settings_durability.py` | Coordinated settings submissions, default-write admission, durable completion |
| `settings_navigation.py` | Suspended settings snapshots, exact return claims, reopen/navigation orchestration |
| `provider_selection.py` | Provider/model selection, readiness, option warnings, pending provider intents |
| `commands.py` | Typed command routing, prefill/research commands, rewind choice orchestration |
| `context_cost.py` | Context/spend projections, estimate/fingerprint caches, inspector factories/loaders |
| `submission.py` | Draft admission, per-session send stashes, acceptance barriers and recovery |
| `row_actions.py` | Conversation/workspace action targets, action dispatch, Markdown export |

All construction stays in `Console_Modules/wiring.py`. Dependencies are named
late-bound callables; sibling controllers are resolved at use time, not captured
at construction. New controllers do not receive a screen, generic screen proxy,
or DOM root. Composer-dependent controllers receive the specific composer lookup.
Menu placement, mounting, registries, focus, timers, and Textual event hooks remain
on the screen; event hooks stop the event before passing its action and target to
the row-action owner. The application-owned chat runtime and durability lifetime
remain unchanged.

The method-count reduction comprises 64 net ownership moves and 54 removed
getter/setter methods. The latter were exactly 27 redundant forwarding pairs,
replaced by the existing writable `_ControllerState` descriptor; they are not
counted as new ownership extractions. Wired reads and writes retain their owner.
Missing owners deliberately follow the existing fail-loud descriptor contract.
Neither `ChatScreen.__init__` before wiring nor `BaseAppScreen.__init__` reads
these fields. Bare test shells must wire their actual owner explicitly.

Tests that patch imported helpers or inspect method source now target the defining
controller module. Thin screen APIs remain where lifecycle/event callers or
existing late-bound patch points require them. Source-worker checks follow both
branches of the existing conditional summary-worker alias and reject unresolved
aliases instead of dropping the original guard targets.

## Focused evidence

Sequential gates were run with the shared Python environment and isolated pytest
temporary roots; these are separate runs, not an aggregate unique-test count:

- Settings navigation/return ownership: 144 passed.
- Provider/readiness/model coverage: 181 passed; provider/default files: 37 passed.
- Commands: 42 rewind tests and 16 action/research/permission tests passed.
- Exact state-descriptor forwarding: 246 hands-free/workspace/dictation tests passed.
- Context/cost/inspector: 81 passed, with two failures reproduced on development
  baseline 93388ba69b and assigned to the coordinating repair.
- Submission: 141 command/raw/question tests, 54 parallel-run/draft-snapshot tests,
  and two vision-gating tests passed.
- Row actions, persistence and export: 48 passed.
- Final size, worker inventory, bare-shell guard and worker-group checks: 42 passed.
- Ruff check passed for the seven new controllers, wiring, screen and affected
  row/worker tests; the ten-file production/worker formatter check passed.

The star-write journey initially raced a queued production repaint after directly
painting a fixture tray. Its fixture now supplies the row at the real workspace
state-derivation boundary, preserving the durable-write assertions. The rewind
guard's worker spy similarly now observes only summary dispatches and allows
unrelated queued workspace refreshes through the real scheduler.

The coordinating repair owns the full suite, baseline style/cost-cache fixtures,
shared architecture inventories and final task closure. TASK-31659 separately
records first-use deferral of the command-only style and rewind modal imports.

The integrated sweep also found detached native-state and generation fixtures
that skipped constructor wiring. They now call the focused settings or commands
builder before reaching that owner's state; both complete State/Chat files pass
(83 tests). The native snapshot helper follows the same settings-owner contract.
The global-name refresh test patches the context/cost estimator's defining owner
and verifies its current provider-equivalent seeded-greeting projection. The full
settings file initially passed 415 tests with one unchanged wall-clock teardown
overrun, which passed isolated; a final full-file rerun remains in progress.
The native flow file passed 345 tests with three timing-sensitive journey fixture
failures, all passing isolated; TASK-31769 owns their explicit phase synchronization.
These results do not claim that the full sweep is green.
