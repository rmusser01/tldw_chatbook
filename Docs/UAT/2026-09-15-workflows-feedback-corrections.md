# Workflows feedback corrections — 2026-09-15

## Result and scope

**The approved UAT corrections pass.** Step labels and required-field summaries
update during editing; validation notices are separate from saved/draft status;
Retry is limited to failed loading or draft persistence; successful context
changes clear transient errors. Validate does not erase an unrelated import error.
Removing the selected step through Advanced JSON also safely returns to Overview
and allows the next field edit.

Tested the working-tree correction over `15b31e4210a919e65060b9b3224c3390b44c3814`
on `codex/workflows-authoring-dev`, under TASK-32601 / ADR-138 and ADR-150.
Only the screen, editor, navigator, regression tests and documentation changed.
No SQLite/schema/runtime/provider changes, new dependencies, commits, pushes or
merge verification. This is scoped authoring qualification, not execution,
server-sync or whole-release sign-off.

## Automated evidence

Final command, from the authoring-dev worktree with the existing Python 3.12 venv:

```sh
PYTHONPATH=. python -m pytest Tests/UI/test_workflows_editor.py Tests/Workflows/test_authoring.py Tests/UI/test_app_quit_guard.py Tests/UI/test_destination_shells.py -k workflow -q --timeout=60 --tb=short --show-capture=no -p no:randomly
```

**93 passed, 128 deselected, 1 existing dependency warning, 112.03 seconds.**
This covers 65 editor cases plus 28 authoring/lifecycle/destination cases. No full
repository sweep was run. Ruff check and format check pass for all four changed
Python files; `git diff --check` passes. Retained branch-wide baseline static debt
is unchanged, under the previously approved no-new-debt gate.

RED evidence: the initial selection produced nine expected failures (stale
heading/summaries and misleading Retry) while existing load/write retry passed.
A separate four-case run reproduced stale errors after workflow, inspection,
return and create transitions. Review then reproduced two correction regressions:
the missing selected-step lookup raised `RuntimeError: coroutine raised
StopIteration`, and Validate erased an unrelated import error. Both have
failing-then-passing regression tests.

Two test synchronization gaps were corrected rather than loosening assertions:
scroll comparisons now await scheduled animations; the modal helper waits for
the actual choices widget within five seconds. The pre-correction final selection
had 92 passes and one premature missing-modal query; the same 93-case command
passed after the helper change. The reviewer repeated the affected recovery test
successfully in three separate runs. Existing RequestsDependencyWarning and
pytest cleanup warnings concerning unrelated old Kokoro temporary directories
remain visible; they were not suppressed or cleaned up.

## Live acceptance evidence

Reused only the disposable `.uat-workflows-9NUT5t` profile. Normal app entry point
via the existing read-only observer, real services/SQLite/file picker, keyboard
and mouse input through attached tmux clients; no app factories or mocked UI.
Cloud catalog refresh was disabled and no credentials or inference calls were
used. The detected local endpoint remains llama.cpp at localhost:9099.

The final-build process was PID 56380. Captures below are relative to the retained
local UAT directory; each `paint-*` has raw compositor text, SVG and widget-state
JSON. Original UAT failures and earlier correction captures were not overwritten.

| Scenario | Final evidence |
| --- | --- |
| Step rename immediately updates field, heading and navigator; Validate preserves draft status and does not show Retry | `paint-1789490307410606000`, `paint-1789490308176590000` |
| Empty required provider adds issue/summary; filling it removes both without changing field focus or collapsing Inputs | `paint-1789490337503749000`, `paint-1789490339102431000` |
| Medium layout and compact live name edit; compact label, populated value and focus visible together | `paint-1789490340602458000`, `paint-1789490403767499000` |
| Real malformed import, then Validate: import error remains, local validation notice is separate, Retry hidden | `paint-1789490408925199000` |
| Switch workflow clears import error; saved inspection is read-only; return restores draft | `paint-1789490445883654000`, `paint-1789490449464216000`, `paint-1789490451597980000` |
| Remove selected step in Advanced JSON, edit Overview name, restore original definition and save | `paint-1789490547962443000`, `paint-1789490578603973000`, `paint-1789490582461215000` |
| Empty foreign SQLite writer transaction causes persistence failure; attempted navigation is vetoed; text and Retry stay visible | `paint-1789490644509686000` |
| Release writer, Retry, then Save: exact edits persist; Retry disappears | `paint-1789490671179496000`, `paint-1789490672605809000` |

Ten post-walk checks of the actual captures and saved database passed. They
checked status/Retry/Run, required summaries/focus, compact editing, unrelated
error preservation, context changes, read-only inspection, edit after raw step
removal, write refusal, Retry recovery, and final saved content / `quick_check`.
The final saved test workflow is `UAT Research Summary recovered`, with step name
`UAT final label!`; `PRAGMA quick_check` returned `ok`. Converted Menlo PNGs were
visually inspected alongside raw captures; font substitution is capture-only.

Both correction-session app processes exited normally with status 0. Both empty
writer transactions were released and the dedicated tmux sessions closed. The
disposable profile and evidence remain local. No personal profile was changed.

## Review and limitations

Independent read-only reviewer `Faraday` (agent
`01a0a5e7-1b97-7b62-9863-46c65ffdfbec`) found and verified the two edge cases above,
then reported no remaining Critical, Important or Minor findings. Its final
focused run passed 18 cases, with separate 3/3 recovery repeats after the modal
helper correction. The coordinator separately ran the final 93 cases and live UAT.

Run remains disabled. No execution, publication, synchronization, branching,
parallelism, current-dev merge, adversarial file-replacement race, or new
server-contract qualification was attempted. ADR-138's approved stable-file
limitation remains unchanged.

Startup logged the existing RichLog widget setup error and a Console sidebar-state
timer attribute error. Their emitting code is outside this correction and exists
at its parent HEAD; this pass did not diagnose or fix them. No `CRITICAL` or
`unhandled_exception` entries appeared in the live-app log. They are reported
separately, not represented as a perfectly clean application log.
