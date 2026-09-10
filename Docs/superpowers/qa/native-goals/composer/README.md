# Composer goal entry point qualification

TASK-32194 adds `/goal [task description]` to the native Console composer.
The command opens existing goal setup with the description prefilled. It retains
the provider, resource, enablement, finite-policy and explicit Review/Start
owners governed by [ADR-141](../../../../../backlog/decisions/141-native-console-goal-runs.md).
The change is relative to the completed goal branch at `4cd0e6f3d7`.

The mounted tests type `/go`, accept `/goal ` completion with Tab, enter the
description and press Enter. The real setup requires review before Start and
persists the exact objective through the real coordinator and SQLite service.
Other cases cover case-insensitive commands, multiline descriptions, bare
`/goal`, cancellation, repeated disabled-goal and missing-binding refusals.
These cases inspect the preserved composer draft and verify zero provider calls
before confirmation. Existing composer tests retain literal/paste behavior.
Only provider resolution/transport is synthetic; no live model is called.

The initial failing run stopped after two expected missing-feature failures:
no `/go` completion and no goal setup from the command; import provenance
passed. The first implementation run recorded 37 passes and one test-fixture
mistake: body-free `GoalHistoryEntry` has no `request`. The test now reads
`service.get(entry.id)` to inspect the actual saved launch.

The [affected gate](affected.txt) passed **118 tests**:

```bash
python -m pytest Tests/test_probe_import_provenance.py Tests/UI/test_console_goal_command.py Tests/Chat/test_console_command_grammar.py Tests/Chat/test_console_command_suggestions.py Tests/UI/test_console_command_popup.py Tests/UI/test_console_command_composer.py Tests/UI/test_console_goal_setup.py Tests/UI/test_console_goal_navigation.py -q --tb=short
```

After mechanical lint cleanup, the [final gate](final.txt) passed **38 tests**:

```bash
python -m pytest Tests/test_probe_import_provenance.py Tests/UI/test_console_goal_command.py Tests/Chat/test_console_command_grammar.py Tests/Chat/test_console_command_suggestions.py -q --tb=short
```

These scopes overlap. The only production change between them was importing
`Callable` from `collections.abc`; test cleanup sorted imports, removed an
unnecessary `return None` and bound a callback's loop value. Six-file Ruff and
format checks pass. The existing large ChatScreen owner has
[150 baseline/current lint diagnostics, with no additions](legacy-lint.json).
Whitespace checks pass. Both runs emit the existing Requests dependency warning.

This qualifies the composer entry point. The prior whole-feature
[qualification report](../../../reviews/2026-09-09-goal-runs-qualification.md)
still records four separate pre-existing test failures, unsuccessful configured
local-model trials, and prerequisite reconciliation before merging into `dev`.
No full-suite sweep, new live-model attempt, schema change or permission grant
was introduced by this follow-up.

The independent [scoped review](review.md) approved the final addition with no actionable findings. Full-branch integration with `dev` remains separate.
