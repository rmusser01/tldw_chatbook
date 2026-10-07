# Console busy configuration entry — TASK-34415

Owner-approved bounded deferral, on `codex/console-busy-config-deferral`, originally
from dev `76d5d157aa6a628584ead573976c9adee72f5412`. Separate from PR3029's Library
test-only correction. The owner subsequently authorized latest-dev rebase,
scoped corrections and merge, accepting a fresh independent review instead of
credit-blocked Qodo for **PR3034 only**. This does not waive failed evidence or
unfinished qualification. [Approval recorded on the PR](https://github.com/rmusser01/tldw_chatbook/pull/3034#issuecomment-6029504207).

## Repair

Both incumbent refresh callers share the checked owner in
`UI/Console_Modules/config_sync.py`. It takes the unchanged native REBUILD then
FILE RLocks nonblocking and retains them across unchanged `operation(config)`.
Contention returns False and requests the existing coalesced 0.2s timer. Partial
acquisition is released; no selected-source read/render runs while busy. Replay
computes current state. Source checks, admission, native retirement, maintenance,
body/cleanup error precedence, whole-worker replay and teardown are unchanged.
No new config API, state/cache, lock, global policy or await.

The original branch shrank from 25231 lines/760 methods to 25204/759. Its published
25218/759 ceiling becomes 25204/759; nothing rises. Removing the redundant
private wrapper repairs the inherited method excess within this extraction.
The affected worker fixture supplies its incumbent session acknowledgement
callback without changing assertions. Formatting is mechanical; the ratchet's
unbounded `lru_cache` becomes stdlib `cache` with identical semantics.

ADR required: no — a routine scheduling repair under
[ADR126](../../../backlog/decisions/126-complete-local-backup-and-recovery.md) and
[ADR120](../../../backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md).

## Initial branch evidence

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

## Latest-dev rebase

Rebased onto `cddc89d3e780b27392549b58ff9134b64e7d5907`. Both appended lessons
were retained in the sole conflict; the original repair replay is unchanged.
Upstream added 27 screen lines. The unchanged 25204 ceiling correctly went RED.
Caller census found no code/test/script users of `_get_shell_bar` or
`_collapse_console_hidden_control_bar`, and one user of `_get_compact_model_bar`.
Delete the unused helpers and inline that exact query/QueryError fallback;
remove only the unused `_summary_row_value` screen import, not its module-owned
implementation. Combined screen: **25204 lines/756 methods**; ceilings
25204/756. Both ratchets pass; neither ceiling rises.

All six existing compact-control/model-default/provider-mirror cases pass in
separate private pytest processes using the existing `bootstrap_profile` fixture
mode. Their assertions and production source checks are unchanged. The initial
ordinary five-case run failed before mounting on `raw_source_selection_changed`:
its fixtures reselected an already-bound config source. It is not a production
regression or passing UI evidence. The selected-profile driver adds only the
existing marker at collection, asserts exactly one original node and calls
`pytest.main`; it changes no test body, config getter or recovery admission.

The first combined rebase run recorded **60 passed/one overlay checkpoint
timeout/two inherited Splash SyntaxWarnings in 222.70s**. The unchanged six-case
activation-fault parametrization then passed **6/6 in 47.46s**, including overlay,
with its original deadlines and no warnings. Earlier failure/warnings remain
recorded; the passing rerun does not turn them into a warning-free qualification
claim. The final serial affected-suite rerun passed **61/61 in 235.40s, no pytest
warnings**, with unchanged assertions, deadlines and source guards. Fresh
independent review found no actionable code regressions; its historical-count
documentation finding was corrected before publication.

All eleven artifact guards pass on the combined source. Full-screen Ruff:
192 inherited diagnostics on latest dev, 189 on work; no additions after
normalizing shifted F811 line references. The three removals belong to deleted
code. Four Python paths format clean; helper and both changed tests Ruff clean.
Raw captures remain under `/private/tmp/pr3034-rebase-*`,
`/private/tmp/pr3034-rebased-*`, `/private/tmp/pr3034-compact-*` and
`/private/tmp/pr3034-overlay-recheck-SE88H7`; no warning or failure was suppressed.

Committed rebase receipts: `rebase-ratchet-red.log`, `rebase-first-affected.log`,
`rebase-compact-controls.log`, `rebase-fault-recheck.log`, `rebase-preflight.log`,
`rebase-screen-lint-comparison.json`, `rebase-final-affected.log`.
Only trailing whitespace was normalized. The compact-case driver below is run
once per original node, not on the whole module (configuration writes must not
leak between cases):

```python
import sys
import pytest


class SelectedBootstrapProfile:
    def pytest_collection_modifyitems(self, items):
        assert len(items) == 1
        items[0].add_marker(pytest.mark.bootstrap_profile)


raise SystemExit(pytest.main(sys.argv[1:], plugins=[SelectedBootstrapProfile()]))
```

## Still open

Busy-lock responsiveness is not full unchanged 50ms activation qualification.
Native viewport/ordinary quit, actual Windows Terminal, three unfamiliar
participants, whole-app retirement, aggregate FD warnings and the separate
baseline closed-cursor finding remain follow-up. TASK31966/TASK31245 remain
In Progress; their missing criteria are neither checked nor waived. No semantic
implementation, global cache/GC/lifetime expansion or Terminal control workaround.
Exact-head PR checks and the newly approved independent review remain pending.
