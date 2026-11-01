# Task7 same-screen exact Open correction

Date: 2026-09-27. Branch: `codex/agent-archive-recovery`.
Base: `2f28e6d6d6b7c5000b57881a4e859292d5f563de`.
Commit: `f83f8975c88b762eb61e927247d76a20c466ea2f`.
Worktree: `/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook`.

## Delivered

The shared `request_conversation_resume` now supplies the existing `NavigateToScreen.on_completion` callback. After navigation commits successful target stack ownership, a still-current source request whose Console remains the visible screen schedules that Console's existing exact-ID consumer. Textual's same-screen switch does not emit `ScreenResume`; this callback closes that specific gap. Ordinary cross-screen navigation retains the existing lifecycle consumer. No synthetic lifecycle events, new hooks, screen/CSS changes or new action framework.

Capture uses the existing screen stack and tolerates an empty stack, preserving callers whose factory app instance delegates presentation to a separate mounted test host. The callback retains `request_current()`/receipt `request_if`, current screen identity and the existing consumer's claim/revision, durable lifecycle and activation guards. No restore-write or confirmation guard changed.

Regression: `Tests/UI/test_console_conversation_archive_flow.py::test_exact_resume_dispatches_on_the_already_current_production_console`. It runs a fresh full production `TldwCli` in a subprocess with isolated HOME/XDG/config/data paths before imports and real scratch SQLite. It does not replace `post_message`, manually consume a handoff or synthesize `ScreenResume`. It follows real production dispatch on an already-current reusable Console, then ordinary Home-to-Console navigation, waiting for exact handoff revisions to be acknowledged. Assertions cover original persisted ID, active leaf with an inactive sibling, source draft, empty destination composer, retained message count and zero agent run/step rows. Provider points to `http://127.0.0.1:1`; no send occurs.

ADR required: no new ADR. Direct correction of ADR-147/166's approved original-ID resume contract, under existing TASK-33020 AC2 and plan step12. Read the task, full brief, diagnosis/reproduction, spec sections7/10, ADR-147/166, design-language governance and relevant evidence/live-navigation lessons. Applied systematic-debugging, test-driven-development and verification-before-completion.

## RED and baseline controls

Final regression bytes were run with only `archive.py` temporarily replaced by `git show 2f28e6d6d6b7c5000b57881a4e859292d5f563de:tldw_chatbook/UI/Console_Modules/archive.py`. A `try/finally` restored the exact fixed bytes. The regression failed at the first post-request settle: `AssertionError: exact conversation resume did not reach its mounted destination`. The exact-ID handoff could not complete on the current Console. Earlier test-first RED established the same boundary before the production fix.

The same final baseline control included both `test_restore_resume_then_send_retains_original_history_and_unrelated_draft[False/True]` siblings. Both reproduced `AttributeError: 'ChatScreen' object has no attribute '_submit_console_native_draft'` at line323 after successful resume/draft assertions. They remain outside this correction; no forwarding/send/trace fix was made.

Baseline control command used the Python executable below and pytest arguments: the new regression node plus both restore-to-send nodes, `-q -p no:cacheprovider --basetemp=/private/tmp/live-open-baseline-control-pytest`. Result: **3 failed in23.00s**, exit1 (expected new regression RED plus two proven baseline send-harness failures). Full output: `/private/tmp/live-open-baseline-control.log`. Earlier final-regression-only base control: **1 failed in14.15s**, `/private/tmp/live-open-final-red.log`.

During scoped verification the first callback capture accessed `app.screen` with an empty stack in separate-host harnesses. This was attributable to the first fix, corrected with stack-based capture before final verification. Initial scoped result: 27 passed/6 failed; affected rerun after correction: 5 passed/2 failed, leaving only the two baseline failures above. No such failure is relabeled as green.

## Final GREEN and static checks

Executed from the worktree, with final source and regression hashes below:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_console_conversation_archive_flow.py::test_exact_resume_dispatches_on_the_already_current_production_console \
  Tests/UI/test_console_archive_recovery_boundaries.py \
  Tests/UI/test_console_archive_resume_errors.py \
  'Tests/UI/test_console_archive_recovery_flow.py::test_focused_native_recovery_actions_activate_from_keyboard[enter-open]' \
  'Tests/UI/test_console_archive_recovery_flow.py::test_focused_native_recovery_actions_activate_from_keyboard[space-open]' \
  Tests/UI/test_library_crit8_conversation_handoff.py::test_stale_list_resume_dispatches_original_and_checks_current_storage \
  -q -p no:cacheprovider --basetemp=/private/tmp/live-open-final-green-pytest
```

Result: **31 passed in28.22s**, exit0; no warnings reported. Full output: `/private/tmp/live-open-final-green.log`. The earlier standalone production GREEN was 1 passed in8.92s; the final selection supersedes it and tests acknowledged revision completion.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --no-cache tldw_chatbook/UI/Console_Modules/archive.py Tests/UI/test_console_conversation_archive_flow.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check --no-cache tldw_chatbook/UI/Console_Modules/archive.py Tests/UI/test_console_conversation_archive_flow.py
git diff --check -- tldw_chatbook/UI/Console_Modules/archive.py Tests/UI/test_console_conversation_archive_flow.py
git diff --cached --check
```

Results: `All checks passed!`, `2 files already formatted`, whitespace exit0. No already-large application/screen file was modified or reformatted. Self-review traced every producer/consumer caller and navigation completion ownership and inspected the final scoped diff. Only source/regression were staged and committed.

Final SHA256:

| File | SHA256 |
| --- | --- |
| `tldw_chatbook/UI/Console_Modules/archive.py` | `70abb1223697df3fca7a6f572fc95ed86539ea5172c8223fac52371c38f11c1e` |
| `Tests/UI/test_console_conversation_archive_flow.py` | `55ac426a4a601bf84a1ad58dabb9640d889467fcf53b0bc15603bef60ba53c3f` |

## Limits and handoff

Full production regression starts at the shared downstream resume-request boundary with an available target; native keyboard receipt Open producer and archived Library launch remain covered by closest mounted siblings. The cross-screen full production control uses Home, because the separately documented Library constructor issue is outside scope. This report does not claim a new real-model/native receipt live qualification; root owns the fresh native run and independent scoped review.

No full sweep, subagents, model inference, root tmux/server use, original-checkout edits, task/plan/QA edits, push/PR/merge, trace-capture/retry change or Library constructor repair. Root's concurrent task/plan/QA/live-artifact changes remain dirty and outside the two-file commit. This ignored handoff report is not committed. Git commit emitted existing GC/loose-object warnings; no repository housekeeping was performed.
