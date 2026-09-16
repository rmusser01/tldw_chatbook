# VoiceAEC Windows checkout failure — bounded review

Reviewed PR2642 HEAD `6b20175799c47f44cf287065207630ab1ad86e93`, saved run34926584884/job104245831599 log, and locally available Git objects. No edits, fetching, tests, app boots, or workflow changes.

**This is a checkout failure inherited through the PR merge tree, not an executed VoiceAEC build/test failure.** Log lines100–109 fetch and check out `144d416943ab5d2039f8f0492639235162a6896e` as `refs/remotes/pull/2642/merge`. Line171 fails creating TASK32540's long filename; line217 identifies the merge as 9dff9dd1b2 into base `77eb2601a63ba473318b8ec1e4edb53f8ac5899e`. The workflow's checkout at `.github/workflows/voice-aec-wheels.yml:81` has no head-ref override; it uses the normal PR merge checkout. No startup/security/backup product error is implicated.

The earlier `3f0dd2038aef0120abfd0135db88b19e82647459` performs two 100% renames, plus documentation. It is already an ancestor of current HEAD. Current HEAD contains both short paths, so cherry-picking/replaying that fix is not the missing action.

| Task | Current HEAD | New dev base77eb |
|---|---|---|
|32540|`backlog/tasks/task-32540 - Library-Notes-import-once-keyboard-navigation.md` (75 characters); blob `d627445345ca8137cdc2671f3282e61163387b1d`|Long path (236 characters); blob `8737503d2d436ae272e84d4d2282bfab3ebdf8a4`|
|32555|`backlog/tasks/task-32555 - Library-Notes-first-run-hand-off-affordances.md`; blob `4d31d13bbc78946f0365a3f449d646923193ba15`|Identical short path and blob|

The offending exact relative path is:

`backlog/tasks/task-32540 - Library-Notes-Import-once-cannot-be-completed-by-keyboard-—-the-picker-opens-with-the-tree-focused-its-buttons-highlight-by-colour-only-and-the-selection-panes-Tab-marks-nothing-before-leaking-into-the-rail.md`

Base32540 is not merely the old bytes under the old name: it changes To Do→Done, marks the four acceptance criteria complete, and adds implementation/validation notes (33 insertions, five deletions against HEAD's short-path blob). Its newer content must be preserved. The base commit is PR2687's merge, but its parents are absent from this shallow local object set; this review does not identify the precise ancestor that recreated the old filename. The synthetic merge object144d is also not locally available. The log proves its failed path and parent identities, not its complete tree or rename-resolution details.

**Smallest legitimate correction:** a single filename-only rename on the current dev/base lineage, from the existing long32540 path to the established short32540 path, preserving blob `8737503d2d436ae272e84d4d2282bfab3ebdf8a4` exactly. TASK32555 needs no change. This can be a narrowly scoped upstream correction; it does not require importing broad dev changes into the backup branch or altering VoiceAEC checkout behavior. After integration, inspect the newly generated real PR merge tree for one short32540 entry with the current task content, then obtain fresh checkout evidence. The historical failure remains recorded until that succeeds.

**Backup-branch-only disposition:** no filename-only correction is currently available in HEAD because the offending long path is already absent and the short rename is already integrated. Renaming the short path again, replaying the existing rename, or copying only upstream task content does not establish that the base-introduced long path will disappear from a new synthetic merge. Do not manufacture ancestry with an ours merge or switch the workflow to test the head to conceal the merge failure. If upstream correction is unavailable, resolving a genuine integration would be a separately reviewed integration task; it is not necessary to broaden this investigation now.
