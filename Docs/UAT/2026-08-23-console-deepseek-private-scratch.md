# Console DeepSeek private-scratch UAT

Date: 2026-08-23
Task: TASK-21161
Provider: DeepSeek (`deepseek-chat`)

## Execution metadata

- Live terminal: 180 columns × 55 rows.
- Live branch: `codex/console-per-chat-sandbox`.
- Live code: `6c6844290b498c05f88bfdfd65cd72fcab2badcb`, based on the
  then-current `origin/dev` commit
  `ae817fefed519921d7da5047e22634756337fc34`.
- Final integration base: the branch was subsequently rebased cleanly onto
  latest `origin/dev` at `be0a1694696b7b2296dcb79017696dd79c56f677`.
  The intervening upstream changes did not overlap the Console implementation;
  fresh post-rebase targeted gates are recorded below.

## Claim under test

Ordinary Console Chats must send without a folder, receive only one private
temporary file sandbox per live chat, and never recover another chat's files.
Named Workspaces retain the same private scratch space and may additionally
reach only folders the user explicitly binds. Closing and reopening a saved
conversation must create fresh scratch authority. The run must not mutate the
real user configuration.

## Isolation and safety

The app ran with a disposable owner-only profile, an explicit disposable data
directory, and a mode-0600 copy of the configured DeepSeek credentials. The
named Workspace bound a separate disposable owner-only folder. No credential,
scratch locator, session identifier, or copied config content is recorded in
this report.

The real config SHA-256 was captured before and after the mounted app lifecycle:

```text
before 4b4a5c250ad439952eea04e41041b2ca576ceb18505ce72ea228bd967ec8315b
after  4b4a5c250ad439952eea04e41041b2ca576ceb18505ce72ea228bd967ec8315b
```

The two disposable profiles and bound-folder fixture were deleted after the
hash comparison.

## Observed scenarios

| Scenario | Live observation | Result |
| --- | --- | --- |
| New ordinary Chat | Default Chat mounted and sent a plain DeepSeek prompt without selecting a folder or opening folder setup. | Pass |
| Private scratch read/write | Chat A used approved `fs_write` and `fs_read` calls against a relative marker in its private scratch directory. | Pass |
| Concurrent-chat isolation | Chat B received approval for `fs_read` of Chat A's relative marker; the tool reported it missing and DeepSeek returned the expected isolation marker. | Pass |
| Close and reopen | Chat A was closed, the saved conversation was reopened, and the former relative marker could no longer be read. | Pass |
| Named Workspace access | A Workspace with one explicit read/write folder binding used approved built-in `read_file` and `write_file` calls; the seeded content and exact written canary were independently verified on disk. | Pass |
| Chat cannot read Workspace | Back in Default Chat, the approval card warned that the bound-folder path was outside allowed folders; approving once still produced a failed `read_file`, and DeepSeek returned the expected denial marker. | Pass |
| Visible authority copy | The Details section identified Default Chat file authority as private scratch; named Workspace selection remained distinct from Chats. | Pass with polish fix |

## Issues found and addressed

1. **Folderless sends were blocked by optional project-instruction setup.** An
   unselected session with no eligible binding entered a folder-recovery modal
   after send. The controller now treats that state as a valid scratch-only
   session while preserving fail-closed recovery for a previously selected
   binding that disappears or changes.
2. **Workspace creation copy contradicted the scratch model.** It said an agent
   had no file access without a bound folder. The explainer now states that
   every chat has private scratch and that Workspace folder access is optional.
3. **The narrow Details rail clipped the security-relevant value.** Live output
   painted `Private scratch` as `Priva…`. The rendered label is now the concise
   `Local files`, leaving the full authority value visible; the underlying
   detailed status remains unchanged.
4. **Private scratch locators could enter tool results and run logs.** An
   independent code review traced absolute scratch-owned paths from built-in
   success results and local `fs_*` errors into model history and persisted
   tool-result records. Both provider boundaries now convert only the private
   scratch locator to relative text before those shared sinks. Explicit
   Workspace folder paths retain their existing behavior. Success, error, and
   real run-log regressions pin the boundary.
5. **The Workspace-create modal's legacy test harness omitted production
   CSS.** A broad UI pass found ten off-screen Pilot clicks. Untouched latest
   `dev` reproduced the same ten failures. Widening the screen moved the
   controls farther out, identifying missing layout CSS rather than a narrow
   terminal defect. Mounting `WorkspaceCreateModal.BUNDLED_CSS` restored the
   real geometry and all 23 modal tests pass.

The first issue was fixed before the complete scenario matrix was rerun. The
copy and narrow-rail polish are also pinned by targeted mounted-widget tests.

## Automated evidence

On the final latest-`dev` base:

- 388 authority, lifecycle, provider, project-instruction, run-log, and
  retained-skill tests passed.
- The broad affected UI-module diagnostic reached 576 passes and the ten
  pre-existing modal-harness failures described above. Untouched latest `dev`
  reproduced those failures at 10 failed / 12 passed; after loading the
  production modal CSS, the complete branch modal module passed 23 / 23.
- The final focused mounted Console/Workspace UI gate passed 56 / 56.
- Ruff and Python compile checks passed for every changed Python file;
  `git diff --check` and the diff secret-pattern scan were clean.

No full repository sweep was run, in accordance with repository policy.
