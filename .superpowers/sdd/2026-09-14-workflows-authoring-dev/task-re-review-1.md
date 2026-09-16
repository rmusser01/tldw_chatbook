## Finding verdict

Prevent database aliases from reaching file exchange — NOT ADDRESSED (Important).

Existing aliases are rejected before generic I/O, and the new foreign-writer
regression covers the original stable-path case. A regular JSON file can still
pass metadata inspection (authoring.py:150), then be replaced with a hard link to
the live DB before open_private_binary opens it (authoring.py:169). The helper
opens, rejects and closes that inode, permitting the same SQLite lock loss.
The documented limitation does not satisfy the binding prohibition. Required:
make the actual exchange boundary safe against substitution and add a
deterministic replacement-between-check-and-open regression.

## New breakage

None beyond the remaining original finding. Export metadata inspection correctly
runs off the UI thread (authoring.py:195).

## Out-of-scope

713 baseline Ruff findings, five formatter failures and Requests/Kokoro noise
remain deferred and unwaived (implementation report:600).

## Verdict

Findings remain open — database-alias protection at the actual file-open boundary.
Complete six-file fix package and appended RED/GREEN evidence reviewed, including
final 35 passes. No mutations, app boots, subagents or test reruns.
