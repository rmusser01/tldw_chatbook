---
id: TASK-38991
title: "Recovery-entrypoint architecture tests expect a stderr refusal, but the CLI routes open the recovery-only host"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- backup
- recovery
- tests
priority: medium
references:
- tldw_chatbook/cli.py
- tldw_chatbook/Backup_Recovery/launcher.py
- tldw_chatbook/Backup_Recovery/bootstrap.py
- Tests/Architecture/test_recovery_entrypoints.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. The cause was traced in review of the filing.

All 7 failing nodes are in `Tests/Architecture/test_recovery_entrypoints.py`. The file had 7 failed and 25 passed (108 s), whether XDG_* was set or unset. TASK-34100.4 (In Progress) recorded the same 7 as pre-existing on its base but does not own them. No lane runs the file, and none of the branches due to merge before this one touches it.

**The route tests (5 nodes): the expectation is stale; the product is safe.**
- `test_real_route_refuses_before_side_effect_imports` fails for the `main_cli_runner()` route and the `runpy tldw_chatbook __main__` route, each with the parameter True and False. That is 4 nodes.
  - The test expects the child to print "Recovery required: recovery_pending" and exit.
  - Instead the child paints a Textual screen and is killed at the 20 s timeout.
- That paint is the recovery-only host, not the app:
  - `cli.py:84` calls `launcher.startup_unlock()`;
  - `startup_preflight()` returns `recovery_pending` (`tldw_chatbook/Backup_Recovery/bootstrap.py:514-517`);
  - `startup_unlock` then returns `minimal_recovery(reason)` (`tldw_chatbook/Backup_Recovery/launcher.py:448-449`), which runs `recovery_app(reason).run()` (`:620-623`).
- `RecoveryApp` never imports the normal app. On `recovery_pending`, its `on_mount` opens Backup & Restore. A review probe at base confirmed that the route calls `minimal_recovery('recovery_pending')` and that `tldw_chatbook.app` is never imported.
- The routing predates the test (b5251e9a6e, 2026-09-16). TASK-34100.4's 87aa6d1d03 (2026-10-03) added `startup_unlock` to this route.
- The 5th node, `test_console_enrolls_…`, fails because `ready.select(15)` returns `[]`. Its fixture writes `config.write_text("preferences")` (`:151`), which is not valid TOML, so startup routes to `configuration_invalid` instead of the state the test means to set up.

**The inventories are out of date (2 nodes).**
- New bootstrap leaf imports: `effective_roots`, `Utils.platform_files` and `collections.abc`.
- About 20 package-main routes have no classification, for example `Chat/console_trace_regex_worker.py`, `Chunking/lab_runner.py` and `Backup_Recovery/detached_worker.py`.

Making `tldw-cli` exit with a stderr refusal instead of opening the recovery host would be a product change and needs an owner decision. This task updates the tests to match the shipped behaviour.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The route test asserts the shipped behaviour for both main routes and both parameter values: on a pending recovery, the recovery-only host opens (with minimal_recovery stubbed so the child exits) and tldw_chatbook.app is never imported
- [ ] #2 The enrollment test's fixture writes a valid TOML config, so the test exercises the state it means to, and the node passes
- [ ] #3 The two inventory nodes pass, with each new bootstrap import and package-main route classified in the notes
- [ ] #4 All 7 nodes pass on dev (red before: 7 of 7 at 29592ecc9b), and the route test still fails when a main route imports tldw_chatbook.app before the recovery fence (negative control)
<!-- AC:END -->
