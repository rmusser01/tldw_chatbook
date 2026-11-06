---
id: TASK-38991
title: "Main TUI entrypoints paint a screen instead of refusing a pending recovery and the recovery-entrypoint inventory is out of date on dev"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- backup
- recovery
- startup
priority: high
references:
- tldw_chatbook/cli.py
- tldw_chatbook/__main__.py
- Tests/Architecture
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. All 7 failing nodes are in the recovery-entrypoint Architecture file. That file had 7 failed and 25 passed (108 s), with XDG_* both unset and set, so the failures do not depend on the environment. TASK-34100.4 (In Progress) recorded the same 7 as pre-existing on its own base but does not own them. Its 87aa6d1d03 (2026-10-03) added `startup_unlock` to this route. No lane runs the file, and no branch due to merge before this one touches it.

**The main routes do not refuse (5 nodes).**
- `test_real_route_refuses_before_side_effect_imports` fails for the `main_cli_runner()` route and the `runpy tldw_chatbook __main__` route, with the parameter both True and False (4 nodes).
- In each, the child should print "Recovery required: recovery_pending". Instead it paints a Textual screen (alt-screen escapes and a `^p palette` footer), and the test kills it at the 20 s timeout.
- The other routes refuse correctly.
- `test_console_enrolls_…` fails because `ready.select(15)` returns `[]`.
- The cause has not been traced. Either the expectation is stale because the route now opens an unlock screen first, or the app really starts while a recovery is pending. The second would be a recovery-safety gap.

**The inventories are out of date (2 nodes).**
- New bootstrap leaf imports: `effective_roots`, `Utils.platform_files` and `collections.abc`.
- About 20 package-main routes are unclassified, for example `Chat/console_trace_regex_worker.py`, `Chunking/lab_runner.py` and `Backup_Recovery/detached_worker.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is recorded: either the main routes now open an unlock or recovery screen before refusing, or the app itself starts while a recovery is pending
- [ ] #2 Both main routes exit non-zero with "Recovery required: recovery_pending" and import nothing with side effects before the fence, or the test is updated to a reviewed new behaviour that still never starts the app over a pending recovery
- [ ] #3 The enrollment node and the two inventory nodes pass, with each new import and route classified in the notes
- [ ] #4 All 7 nodes pass on dev (red before: 7 of 7 at 29592ecc9b), and the refusal test still fails when the recovery fence is removed from one main route (negative control)
<!-- AC:END -->
