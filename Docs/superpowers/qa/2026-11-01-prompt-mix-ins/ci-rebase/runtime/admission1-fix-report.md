# PR3346 admission1 handoff

## Findings and changes

The two exact chat-create bridge tests pass unchanged after the earlier memory receipt fix. `_bridged_controller` uses `persisted_console_store()` whose default real SQLite DB is :memory:. Their original CI diagnostics show accepted native commit followed by pre-provider dispatch-settlement refusal, matching the independently proven receipt-worker connection failure. The fixed run reaches the bridge and retains every existing non-None chat-create callback assertion. No callback, permission, fixture or product edits were needed.

The exact request field inventory in `Tests/UI/test_console_runtime_ownership.py::test_runtime_owned_custody_tracks_only_lifetime_handles` omitted three existing fields: `_pressed_inputs`, `_pressed_stash`, `_pressed_attachment_generation`. These fields are present unchanged in pinned dev62a and used by `console_received_dispatch.project_received_input` to project only the original exact input/attachment-generation owner. They are not added run-state or queue stores. Updated the exact inventory, retained its runtime record/input field equality and no-run-state assertions, and added checks that all three witnesses default to None and are repr=False.

Only new repository edit: `Tests/UI/test_console_runtime_ownership.py` (+13). Parent authorized confirmed stale inventory correction; no real product fix was made in this follow-up.

## Verification

Original exact three-node run on current working tree:1 failed,2 passed in4.66s. Both chat-create callback nodes PASSED with original assertions. Only the stale inventory failed.

After test inventory correction:exact ownership node1 passed in2.24s. Subsequent formatting-only line split adds no behavior change. Scoped git diff --check passes. Whole file retains12 baseline Ruff findings and22 baseline formatter edits:0 additions to either. No unrelated formatting cleanup or full suite.

## Owned artifacts

- Repository: Tests/UI/test_console_runtime_ownership.py
- Report: this file
- /private/tmp/mixins-admission1-red.log
- /private/tmp/mixins-admission1-green.log
- /private/tmp/mixins-admission1-static-comparison.json
- /private/tmp/mixins-admission1-format.diff (initial formatter observation; its one new set-layout issue was fixed)
- pytest basetemp: /private/tmp/mixins-admission1-red
- pytest basetemp: /private/tmp/mixins-admission1-green

The input admission1 CI log is parent-owned. Earlier reports list six prior owned repository files, now seven total. No manifest, commit, push or unrelated file edits.
