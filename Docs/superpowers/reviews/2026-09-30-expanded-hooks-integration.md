# Expanded hooks: current-dev integration

Base: `origin/dev` at `10b34ffdd5698a64de9c285ef899e8c610055ff0`.
Branch: `codex/expanded-hooks`. The clean original `codex/managed-plugins`
checkpoint `cdeb1687b6` is preserved and has no published PR. Its completed
implementation tasks were still marked To Do on dev; this stream reuses and
qualifies each implementation before adding the remaining capability work.

## TASK-32676 — v2 validation

Reused H1 checkpoint `7e99298cf2`. The current legacy inventory, row enablement,
shared validation and persistent exact-definition consent are retained. Explicit
v2 declarations have a separate lazy projection, including bounded invalid
required/event-policy metadata. No new v2 execution entry is activated by H1.

The unchanged current-dev baseline returned **81 passed, 1 failed**. The failure
was the old real-config test changing its profile after source-bound recovery
admission. The test now uses the existing selected private profile fixture.
Public-loader RED returned **2 failed, 2 passed**: missing v2 definitions and
required failure metadata, with disabled-legacy and real-save controls passing.

Final covering command:

```sh
.venv/bin/python -m pytest Tests/Agents/test_hooks_v2_validation.py Tests/Agents/test_run_hooks.py Tests/Chat/test_run_hooks_metadata.py Tests/Agents/test_hook_permissions.py Tests/Agents/test_hook_config_inventory.py Tests/UI/test_settings_hooks.py -q --timeout=120 --basetemp=/private/tmp/expanded-hooks-h1-qualified --junitxml=/private/tmp/expanded-hooks-h1-qualified.xml
```

**302 passed in 50.69s**, no skips. Final fixture alias/import cleanup was followed
by the exact metadata file: **6 passed**, separately recorded and overlapping.
The actual imported package is this worktree, Python 3.12.11/Pydantic 2.12.5.
Eight authored Python files pass full Ruff and formatting; all nine changed/new
Python files parse. Shared config.py retains exactly the same 168 existing Ruff
diagnostics. `git diff --check` passes.

Existing [ADR-163](../../../backlog/decisions/163-expanded-console-hook-runtime.md),
[ADR-148](../../../backlog/decisions/148-console-run-hooks.md), and
[ADR-197](../../../backlog/decisions/197-console-hook-configuration-review.md)
apply. H1 self-review covers all four ACs. No full suite or live-provider run,
new storage schema, dependency or permission owner is claimed. Command execution,
tool pipelines, lifecycle events and plugin/MCP composition remain separate tasks.

## Remaining requested order

- TASK-32677: bounded command execution.
- TASK-32678: input transformations and post-event barriers.
- TASK-32679: session/child/compaction events.
- TASK-32680: Stop continuations and teardown.
- TASK-32685: MCP hooks, including required MCP/plugin prerequisites.
- TASK-32686: native capabilities, required by the requested TASK-32687.
- TASK-32687: Cursor/Codex package and hook adapter qualification.
