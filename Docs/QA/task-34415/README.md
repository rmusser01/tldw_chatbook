# Console busy configuration entry — TASK-34415

Owner-approved bounded deferral, on `codex/console-busy-config-deferral`, from
dev `76d5d157aa6a628584ead573976c9adee72f5412`. Separate from PR3029's Library
test-only correction. No merge or automated-review waiver.

## Repair

Both incumbent refresh callers share the checked owner in
`UI/Console_Modules/config_sync.py`. It takes the unchanged native REBUILD then
FILE RLocks nonblocking and retains them across unchanged `operation(config)`.
Contention returns False and requests the existing coalesced 0.2s timer. Partial
acquisition is released; no selected-source read/render runs while busy. Replay
computes current state. Source checks, admission, native retirement, maintenance,
body/cleanup error precedence, whole-worker replay and teardown are unchanged.
No new config API, state/cache, lock, global policy or await.

The screen shrinks from 25231 lines/760 methods to 25204/759. The published
25218/759 ceiling becomes 25204/759; nothing rises. Removing the redundant
private wrapper repairs the inherited method excess within this extraction.
The affected worker fixture supplies its incumbent session acknowledgement
callback without changing assertions. Formatting is mechanical; the ratchet's
unbounded `lru_cache` becomes stdlib `cache` with identical semantics.

ADR required: no — a routine scheduling repair under
[ADR126](../../../backlog/decisions/126-complete-local-backup-and-recovery.md) and
[ADR120](../../../backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md).

## Evidence

Committed terminal receipts normalize trailing whitespace and final blank lines
only. Original captures remain in the `/tmp/task34415-*` directories; no message,
failure, warning, timing or assertion was removed.

- `red.log`: six real-thread held-lock scenarios fail with the old owner;
  both callers wait about 2.01–2.02s for the test watchdog, then render. GREEN
  asserts return while still held, not an arbitrary latency threshold. This run
  retains housekeeping warnings for old unrelated shared-temp debris; no cleanup
  or warning suppression was performed.
- `baseline.log`: current-config smoke passes; base line ratchet fails at
  25231 against 25218. Independent AST measurement finds 760 methods versus 759.
- `green.log`: six cases pass in 7.26s: both locks/callers, no read/publication,
  one retry, FILE-case partial REBUILD release proven on another thread, fresh
  checked config after save and source-drift refusal before replay publication.
- `first-combined.log`: 56 pass/five fail in 104.95s: four stale worker fixture
  callback failures and inherited method excess. Two inherited Splash
  SyntaxWarnings are retained, not fixed or waived.
- `base-worker-counterfactual.log`: the exact base checked owner reaches the
  same missing-callback error. Shared-owner counterfactual, not whole-repo BASE.
  The first driver attempt failed before fixture creation on a missing root;
  it remains in `/tmp` and is not regression evidence.
- `final-tests.log`: **61 pass in 113.84s, no pytest warnings**. Targeted config
  sync lifetime, native lock order, sync maintenance, mounted character activation,
  reduced motion and both Console ratchets. No full suite ran; earlier warnings
  remain recorded above.
- `initial-preflight.log` and `final-preflight.log`: all eleven derived-artifact
  guards pass, including the final source/receipt closeout rerun.
- `screen-lint-comparison.json`: full-screen Ruff has 192 diagnostics on base and
  work, none added/removed after accounting for shifted line references. Helper
  and both modified test paths are Ruff clean; four Python paths format clean.
  This is not a clean whole-screen lint claim.
- `design-detector.log`: Impeccable detector invoked once on two changed Python
  UI targets; exit 0/no output. It is not native terminal paint verification.

Independent read-only scoped review: no Critical/Important/Minor issues;
unchanged checked-scope/render AST, partial cleanup, fresh replay and final
test/size receipts confirmed. Review is not CI or Qodo acceptance.

## Still open

Busy-lock responsiveness is not full unchanged 50ms activation qualification.
Native viewport/ordinary quit, actual Windows Terminal, three unfamiliar
participants, whole-app retirement, aggregate FD warnings and the separate
baseline closed-cursor finding remain follow-up. TASK31966/TASK31245 remain
In Progress; their missing criteria are neither checked nor waived. No semantic
implementation, global cache/GC/lifetime expansion or Terminal control workaround.
Exact-head PR checks and external review remain pending.
