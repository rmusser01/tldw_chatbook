# Phase 1.1 Protocol Smoke Walkthrough

## Environment

- Date: 2026-05-03
- Branch: `codex/unified-shell-phase1-qa-harness`
- Commit: pending
- Python version: repo venv Python 3.12.11
- Runtime source: repository checkout
- Config/home directory: not exercised by this smoke

## Task Or Phase

- Backlog task: `TASK-2.1`
- Phase: Phase 1.1
- Destination or workflow: shell QA protocol and evidence template setup

## Entry Path

- Focused protocol regression: `python3 -m pytest Tests/UI/test_unified_shell_qa_protocol.py -q`
- Supporting mounted navigation check: `python3 -m pytest Tests/UI/test_master_shell_navigation.py -q`

## Steps Attempted

1. Confirmed the Phase 1 protocol identifies `python3 -m tldw_chatbook.app` as the manual running-app entry point.
2. Confirmed the protocol references focused mounted tests for navigation support.
3. Confirmed the template includes visual, keyboard, mouse/click, functional, severity, evidence, and residual-risk sections.
4. Confirmed the roadmap links the protocol, template, and smoke summary.

## Visual Usability Notes

- Layout: not a product UI walkthrough.
- Clipping: not tested.
- Focus indication: template requires this field for future walkthroughs.
- Labels: template requires this field for future walkthroughs.
- Disabled or blocked states: protocol requires owner, reason, and next action to be recorded.
- Information hierarchy: protocol requires destination purpose, source authority, and current status to be recorded.

## Keyboard Path Result

- Not tested in product UI during this protocol smoke.

## Mouse/Click Path Result

- Not tested in product UI during this protocol smoke.

## Functional Result

- Completed workflow: Phase 1.1 protocol/template wiring.
- Honest blocked state: product destination workflows are outside this smoke.
- Failed workflow: none in protocol wiring.
- Recovery path: use `walkthrough-template.md` for destination-level failures found in `TASK-2.2`.

## Defect Severity

- No product defect severity assigned. This is not a full destination workflow audit.

## Evidence

- Test commands:
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_unified_shell_qa_protocol.py -q`
  - Result: 5 passed in 0.13s.
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_master_shell_navigation.py -q`
  - Result: 3 passed, 3 warnings in 10.28s.
- Related task: `TASK-2.1`

## Residual Risk

- Live app launch through `python3 -m tldw_chatbook.app` still needs manual walkthrough evidence in `TASK-2.2`.
- Optional dependency, server, auth, runtime, and policy states are not exercised by this smoke.
- This smoke confirms the QA harness is trackable and runnable; it is not a full destination workflow audit.
- System Python 3.9.6 cannot run the mounted app import path because the project requires Python 3.11+ syntax; use the repo venv or another Python 3.11+ runtime for Phase 1 QA.

## Product QA Boundary

This summary validates the Phase 1 QA protocol and template. It does not verify that Home, Console, Library, Artifacts, Personas, W+C, Schedules, Workflows, MCP, ACP, Skills, or Settings are usable end-to-end.
