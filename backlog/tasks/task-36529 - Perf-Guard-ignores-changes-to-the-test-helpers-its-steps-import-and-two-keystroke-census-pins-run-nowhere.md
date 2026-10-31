---
id: TASK-36529
title: Perf Guard ignores changes to the test helpers its steps import, and two keystroke census pins run nowhere
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- ci
- tests
- performance
priority: low
references:
- .github/workflows/perf-guard.yml
- Tests/Performance/test_console_keystroke_work_census.py
- Tests/private_profile.py
- Tests/UI/console_controller_stubs.py
- Tests/UI/app_factory.py
dependencies:
- TASK-36390
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36390's review (PR #3230, 2026-10-23). Perf Guard is the only PR run of the latency and keystroke guards.
- **Trigger paths.** perf-guard.yml's `pull_request` paths are `tldw_chatbook/**.py`, `tldw_chatbook/css/**`, `Tests/Performance/**`, one Tests/Utils file, `pyproject.toml` and the workflow itself; PR #3230 adds `Packaging/speculative_voice_*.py`. The files its steps run import helpers outside those paths:
  - `Tests/private_profile.py`: five of them;
  - `Tests/UI/console_controller_stubs.py`: the keystroke census;
  - `Tests/UI/app_factory.py`: the CSS fast-path test;
  - and every test loads the root `Tests/conftest.py`.

  A PR that changes only these files merges without a Perf Guard run.
- **Two passing pins that nothing runs.** `test_settled_400_message_typing_traverses_no_history` (:1090) and `test_typing_does_not_rebuild_the_provider_derivation` (:1185) pass on dev (2 passed). PR #3230 adds only the census's other two behaviour nodes to Perf Guard.
- **Detection runs through the avatar.** The census sees a full Console sync during typing by counting `messages_for_session` calls. TASK-36390 found that a full sync costs 3 such calls: 2 from the avatar path (`UI/Console_Modules/character.py:368`) and 1 from the transcript. The avatar call runs only when expression reactions are on (`if react and store is not None`, character.py:366). So with reactions off, or a quieter avatar path, a full sync inside the burst counts 1 or 0, and the guard passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A PR that changes only a file a Perf Guard step imports (including Tests/private_profile.py, Tests/UI/console_controller_stubs.py, Tests/UI/app_factory.py and Tests/conftest.py) triggers Perf Guard
- [ ] #2 test_settled_400_message_typing_traverses_no_history and test_typing_does_not_rebuild_the_provider_derivation run on a PR lane
- [ ] #3 The keystroke census fails when a full Console sync runs inside the typing burst with avatar expression reactions off, shown by a negative control
<!-- AC:END -->
