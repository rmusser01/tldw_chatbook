---
id: TASK-1270
title: Run log in a bound workspace folder is readable by sub-agents
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 00:00'
updated_date: '2026-07-28 21:32'
labels:
  - agents
  - run-log
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent run log picks its directory name based on where the log lands:

- **sandbox fallback** → `.agent-runs` (dotted, so `_is_hidden_within` excludes it from
  `glob_files`/`grep_files`)
- **bound workspace folder** → `agent-runs` (undotted)

The dotting exists to stop a spawned sub-agent — which inherits its parent's allow-list —
from reading the parent's entire run log through the generic file tools. That was
reproduced during review of PR #1066 by extracting a planted `PARENT_SECRET_API_KEY`, and
fixed for the sandbox case.

The undotted workspace name was justified by a premise recorded in the design spec §9.4 and
in `run_log.py`'s own comments: that `glob_files`/`grep_files` glob `_tool_sandbox_root()`
alone and cannot reach workspace folder roots at all.

**That premise is no longer true.** TASK-850 ("Scope glob_files and grep_files to workspace
folder roots") is Done, and both tools now resolve their roots through `allowed_file_roots`:

```
GlobFiles  (class at file_operation_tools.py:897)  → :959   allowed_file_roots(write=False, ...)
GrepFiles  (class at file_operation_tools.py:1433) → :1562  allowed_file_roots(write=False, ...)
```

So when a workspace folder is bound, the run log is written undotted into a root those tools
now search, and a sub-agent can `grep_files` its parent's history — the same disclosure the
dotting was introduced to prevent, reopened for the workspace case by an unrelated change
landing on dev.

The existing isolation tests do not catch this because they only exercise the
sandbox-fallback configuration.

Note the trade-off behind the original undotted choice: the log was meant to be a
user-visible artifact in the user's own project. A dotted directory remains fully visible to
the *user* (it is an ordinary directory, listed by `ls -a` and shown by editors); it is
hidden only from the app's own sandboxed file tools, which is precisely the intent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A sub-agent cannot read its parent's run log via `glob_files` or `grep_files` when a workspace folder is bound, proven by a test that plants a distinctive secret and asserts it is not recoverable
- [x] #2 The same protection holds in the sandbox-fallback configuration (no regression of the existing behaviour)
- [x] #3 The remedy does not depend on which branch `resolve_log_root` took, so a future change to root resolution cannot silently reopen it
- [x] #4 The app's own reader (`search_run_log` / `load_records`) still reads the log in every configuration, since it deliberately does not route through `validate_path`
- [x] #5 The log remains reachable by the user as an ordinary file, and the design spec §9.1/§9.4 and `run_log.py`'s comments are corrected to state the current premise rather than the stale one
- [x] #6 A regression test pins the invariant against the specific change that caused this — i.e. it fails if `glob_files`/`grep_files` can reach the log directory, regardless of how their root resolution is implemented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the disclosure first: add xfail(strict) tests in Tests/Agents/test_run_log_workspace_isolation.py that plant a secret in a genuinely bound (non-nested) workspace folder and confirm grep_files/glob_files can recover it against current code.
2. Evaluate the direct fix (dot RunLogWriter.bind()'s directory name unconditionally, delete the is_sandbox_fallback conditional and the _root_kind side channel) and measure its blast radius against the full pre-existing suite before committing to it.
3. Get an explicit ruling on the 22 pre-existing tests the fix flips (including two "must not dot every workspace folder" PR #1066 regression guards) before editing any of them, per the standing rule against silently editing pre-existing tests to make them pass.
4. On authorization: apply the fix, un-xfail the two reproduction tests, update the 22 flipped tests (naming-only ones get the new dotted string; the two regression guards get rewritten to assert the security invariant instead of the superseded naming choice), and correct the design spec (§9.1/§9.4/§11) and run_log.py's comments to state the current premise.
5. Prove search_run_log/load_records still read the log in both configurations with a dedicated test.
6. Run the full Tests/Agents suite and confirm no test outside the predicted 22 broke.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: reproduced the disclosure first (TDD), got an explicit ruling before touching any pre-existing test, then applied the authorized fix.

`RunLogWriter.bind()` used to dot the log directory name only when `resolve_log_root()` resolved via the sandbox fallback, on the premise (correct when written) that `glob_files`/`grep_files` could not reach a workspace folder root. TASK-850 invalidated that premise. Reproduced with a planted `PARENT_SECRET_API_KEY` in `Tests/Agents/test_run_log_workspace_isolation.py` (direct grep/glob check + a full spawn_subagent run), confirmed failing against unfixed code, then paused: the fix (dot unconditionally, delete the conditional) flips the literal directory-name string asserted by 22 pre-existing tests, two of which are explicit PR #1066 "must not dot every workspace folder" regression guards. Reported BLOCKED with the exact diff and failure list rather than silently editing them.

On explicit authorization, applied the fix: `bind()` now dots the directory name unconditionally in every configuration; deleted the `is_sandbox_fallback` conditional, the F8 sandbox-containment check it guarded, and the now-dead `_root_kind` thread-local side channel entirely (nothing else referenced it). `resolve_log_root()` no longer has a naming side effect.

Un-xfailed both reproduction tests (now ordinary passing tests). Updated the 22 flipped tests: ~20 were pure test-harness naming/fixture-path assertions (`root / "agent-runs"` -> `root / ".agent-runs"`, including two custom-`dir_name` tests where the configured name is now also dotted); the two regression guards (`test_bound_workspace_folder_keeps_the_undotted_name` in test_run_log_sandbox_isolation.py, `test_workspace_folder_outside_the_sandbox_keeps_the_undotted_name` in test_run_log_writer.py) were rewritten rather than just re-pointed: they now plant a secret and assert `grep_files` cannot recover it, pinning the current security invariant instead of the superseded naming choice. Also fixed a latent gap found along the way: `test_existing_gitignore_is_never_overwritten` was pre-creating the OLD undotted path, which `bind()` no longer touches at all -- it was passing vacuously without exercising the behavior it claims to; fixed to pre-create the actual (dotted) path the writer binds to.

Added two passing (non-xfail) tests proving `search_run_log`/`load_records` still read the log in both the sandbox-fallback and bound-workspace configurations, since that reader intentionally never routes through `validate_path`.

Corrected the design spec (§9.1, §9.4, §11) and run_log.py's own comments to state the current (TASK-850-changed) premise instead of the one TASK-1270 disproved, so a future maintainer isn't talked back into the conditional.

Verification: full Tests/Agents suite went from 529 passed (baseline) to 533 passed, 0 failed -- no test outside the predicted 22 broke.

Modified files: tldw_chatbook/Agents/run_log.py; Tests/Agents/test_run_log_workspace_isolation.py (new); Tests/Agents/test_run_log_sandbox_isolation.py; Tests/Agents/test_run_log_service_wiring.py; Tests/Agents/test_run_log_writer.py; Docs/superpowers/specs/2026-07-27-agent-programmatic-run-memory-design.md; task-1270-report.md.
<!-- SECTION:NOTES:END -->
