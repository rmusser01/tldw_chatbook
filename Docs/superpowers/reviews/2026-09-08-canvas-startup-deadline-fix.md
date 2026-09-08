# Canvas shared startup deadline — 2026-09-08

Status: complete and independently reviewed in `235641b38026f747112f55899f6dd07994e7e6fd`.
This is Task13 under TASK-31942, not Canvas V2 admission.

## Approved correction

The initial actual-Console browser test previously combined an implicit
5000ms Playwright first-output assertion with an explicit 45,000ms Composer
assertion. A page default timeout does not set Playwright's assertion default.
The user explicitly approved replacing those two independent budgets with one
45,000ms monotonic deadline immediately after login.

Both original conditions remain required, in order. Each assertion receives
only positive remaining time; expiry prevents dispatch of the next assertion.
There are no retries, clock resets or background tasks. Original assertion
failures are re-raised with fixed stage notes; cancellation propagates.
All later workflow assertions, F12 behavior and timeouts remain unchanged.
Only two test files changed; no production code or runtime/helper/import/UI
budget changed. Existing ADR125 applies unchanged; no new ADR is required.

This deliberately removes the earlier 5-second first-output sub-limit. It is a
test-contract correction, not evidence of a production startup optimization or
an explanation of the original run's latency. The earlier observed 4.883-second
first-output and 12.773-second Composer timings include scheduling and polling;
the marker itself means a qualifying output chunk, not full application readiness.

## Verification

All commands used the existing linked `canvas-v1` worktree and its
`../../.venv/bin/python`. Browser launches used scoped permission for owned
Chromium/Chatbook children and a loopback listener. No diagnostic environment,
host/dependency repair, broad-suite run or blind actual-browser retry was used.

| Evidence | Result |
| --- | --- |
| Implementer controlled RED, before helper | 4 failed, 1 deselected, 1 inherited warning, 2.85s; missing helper/monotonic seam |
| Expiry-before-dispatch sensitivity check | 1 failed, 1 warning, 3.14s; caught premature second locator construction |
| Implementer final focused file, including local DOM | 5 passed, 1 warning, 7.02s |
| Root exact actual Chatbook browser case, once | 1 passed, 1 warning, 46.86s; process exit 0 |
| Root committed focused file | 5 passed, 1 warning, 2.37s; process exit 0 |
| Root two-file Ruff / new-file format / whitespace | All pass |

The initial full-file RED attempt also hit sandbox Chromium launch denial
(5 failed, 1 warning, 3.22s). That environmental failure is preserved, not counted
as behavioral RED. The controlled RED selection isolates the absent helper.

Root actual-browser command:

```text
../../.venv/bin/python -m pytest 'Tests/Canvas/browser/test_canvas_served_flow.py::test_actual_chatbook_console_finalizes_canvas_create_and_update[read-publication-False]' -q -ra --tb=short --show-capture=no
```

Root committed focused command:

```text
../../.venv/bin/python -m pytest Tests/Canvas/browser/test_canvas_startup_readiness.py -q -ra --tb=short
```

The focused tests cover initial and consumed budgets, expiry before second-stage
dispatch, assertion failure without retry, cancellation, and real Playwright DOM
enforcement of both readiness conditions. The actual case retains creation,
update, restored-card selection, pin/metadata, provider-count and owned-cleanup
assertions. These are scoped results, not a new aggregate qualification run.

All pytest runs retain the existing Requests dependency-version warning. The
served-flow whole-file formatter check remains nonzero for pre-existing,
unrelated spans; the new helper is not in its reported formatting spans. Root
compared baseline/current formatter edits after excluding line numbers: all
67 edit lines have identical content. The nonzero whole-file gate is not waived.
No bulk formatting or warning suppression was introduced.

Full implementation/RED/GREEN details and root logs remain in the ignored
`.superpowers/sdd/2026-09-07-sqlite-lock-safe-private-validation-implementation/`
workspace as `task-13-report.md`, `task-13-live-run.log` and
`task-13-root-focused.log`. The old lifecycle capture was preserved before
overwrite as `output/playwright/mermaid-release/task-13-pre-fix-lifecycle.json`;
parsed JSON equality was checked, not byte identity.

## Review and remaining gates

Independent Task13 spec compliance and quality review approved with no Critical
or Important findings. Its sole Minor is the inherited Requests warning, verified
against root output and deferred to dependency-owned work. The complete review
is preserved as `task-13-review.md` in the same SDD workspace; no execution gap
was raised. The completed broad SQLite-correction review was not repeated.

Earlier failures and evidence remain intact. The historical 5-second startup
failure is addressed by an explicitly revised test contract, not a demonstrated
runtime speedup. Eleven host-blocked cases, platform/optional coverage and
nonzero aggregate static gates remain as recorded in
[final local qualification](2026-09-08-sqlite-final-local-qualification.md).
TASK-31942 stays In Progress, original final ACs remain unchecked, and V2 stays
disabled. No PR, push, rebase, merge or evidence cleanup occurred.
