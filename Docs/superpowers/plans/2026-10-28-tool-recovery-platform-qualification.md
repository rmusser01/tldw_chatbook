# Tool recovery platform qualification

Task: TASK-37204. Follow-up to PR #3284 and TASK-37146.

ADR required: no
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/252-tool-availability-refusals-before-hook-preparation.md
Reason: test portability and bounded CI qualification under the accepted Windows refusal and preparation contracts. No new command backend, permission authority or runtime policy.

1. Reproduce the complete review/pipeline batch on native Windows, including the explicit unsupported-platform cleanup control. Attribute the teardown error rather than suppressing fixture checks.
2. Mark only actual command-execution cases as requiring the POSIX backend. Keep unavailable/disabled handlers, validation, pre-dispatch refusals, empty-engine hook recovery and typed refusal guards active on Windows. Also exclude invalid-output cases whose Windows refusal could otherwise produce a false pass.
3. Make the Windows refusal control prove no child launch, owner/environment access or leaked tickets, and repeat it on the same engine. Keep successful post-refusal command execution as a POSIX-only recovery control.
4. Add a narrow Ubuntu/Python 3.12 PR workflow using existing dependency conventions. Run actual native/fenced discovery recovery, command execution/cleanup, pipeline, working-set replacement and skill gates. Retain platform identity and JUnit results.
5. Run the Windows target, inspect the Linux run, repair observed failures at their cause, and rerun only affected checks. Run lint/format/task/ADR/whitespace checks; record exact results and any skips.
6. Review the final patch, update task and PR evidence, commit and push. No full suite is authorized.

Progress: native Windows RED reproduced the original 19 pipeline failures and
fleet teardown error, plus the refusal control's host-restoration assertion.
GREEN: 72 passed, 20 explicit POSIX skips, one existing Pydantic warning, no errors
(141.21s). The extra skip corrects an invalid-output negative control that was
otherwise falsely passing on unsupported_platform. Independent review found no
actionable issue. Linux execution remains pending on the committed PR head.
