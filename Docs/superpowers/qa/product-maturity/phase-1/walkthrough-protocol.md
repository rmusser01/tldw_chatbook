# Product Maturity Phase 1 Walkthrough Protocol

Status: active for Phase 1.1 harness setup

Use this protocol before marking product-maturity workflows usable. Run it against the actual Textual app when product behavior is in scope. Render-only checks and click-event-only checks do not prove workflow completion.

## Clean-Run Setup

Use a fresh `HOME` and `XDG_*` directory set when validating first-run or setup behavior:

- Fresh HOME
- XDG_CONFIG_HOME
- XDG_DATA_HOME
- XDG_CACHE_HOME

Record the exact directories in the QA summary. Do not use a developer's normal app state for first-run claims.

## Entry Commands

Manual app entry:

```bash
python3 -m tldw_chatbook.app
```

Focused harness contract:

```bash
python3 -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q
```

Existing shell support checks:

```bash
python3 -m pytest Tests/UI/test_master_shell_navigation.py -q
python3 -m pytest Tests/UI/test_shell_destinations.py -q
python3 -m pytest Tests/UI/test_unified_shell_qa_protocol.py -q
```

## Terminal-Size Matrix

Record the terminal size used for every visual/focus walkthrough:

- minimum supported compact
- common laptop terminal
- large power-user workspace

The implementation task that performs a visual sweep must replace these labels with exact dimensions after confirming supported sizes.

## Required Walkthrough Scope

For each product workflow or harness gate, record:

- Navigation entry path.
- Whether status and source authority are visible.
- Whether primary actions are reachable by keyboard and mouse.
- Whether disabled, blocked, unavailable, pending approval, and recovery states explain owner, reason, and next action.
- Whether the workflow completes, is honestly blocked with recovery, or fails.
- Whether repeated use remains fast enough for power users.

## Severity Labels

Use exactly one taxonomy label when a defect is found:

- `blocker` - P0; prevents basic use, traps the user, corrupts or loses user work, or makes a required workflow impossible.
- `workflow-degradation` - P1; breaks or seriously slows a core workflow but leaves a workaround.
- `recoverability` - P2; blocked/error state exists but recovery copy, ownership, or next action is unclear.
- `polish` - P3; visual or wording issue that does not block completion.

## Evidence Rules

- Do not count render-only checks as workflow completion.
- Do not count click-event-only checks as workflow completion.
- If a workflow is blocked, record the recovery path and severity instead of treating the blocked state as a pass.
- Record screenshots only when they materially clarify layout, focus, or visual defects.

## Output

Create one markdown summary per gate using `walkthrough-template.md`. Store summaries in this directory and link them from `Docs/superpowers/trackers/product-maturity-roadmap.md` and the relevant Backlog task.
