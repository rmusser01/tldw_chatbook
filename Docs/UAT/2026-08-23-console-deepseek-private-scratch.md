# Console DeepSeek private-scratch UAT

Date: 2026-08-23  
Task: TASK-21161  
Provider: DeepSeek (`deepseek-chat`)

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

The first issue was fixed before the complete scenario matrix was rerun. The
copy and narrow-rail polish are also pinned by targeted mounted-widget tests.

## Automated evidence

The implementation was verified with targeted authority, lifecycle, retained
artifact, Console UI, and project-instruction suites. The final command set and
pass counts are recorded in TASK-21161's implementation notes. No full
repository sweep was run, in accordance with repository policy.

