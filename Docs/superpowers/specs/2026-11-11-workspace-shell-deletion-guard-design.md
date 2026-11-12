# Workspace shell deletion guard

## Intent

A Chatbook workspace chat can edit project files and delete specific files or subfolders. The model's raw shell tool refuses recognizable commands that delete any local folder bound to the chat's workspace or wipe its contents. The refusal applies even with a shell session grant.

## Approaches

| Approach | Tradeoff |
| --- | --- |
| Workspace-aware command guard (selected) | Extends the existing safety floor and preserves direct shell workflows. Coverage depends on which command forms the guard understands. |
| Route mutations through structured tools | Can enforce the policy on actual writes if all other execution paths are isolated. Programs lose direct write access to the real project. |
| Execute against an isolated copy and validate publication | Preserves more script workflows. Requires isolation, synchronization, and a guarded publication mechanism. |

## Behavior

- Protect every local filesystem binding of the chat's owning workspace, including read-only bindings and folders outside the current project-instruction selection. A model-supplied working directory only affects path interpretation. A workspace switch elsewhere in the UI does not retarget the guard.
- For recognizable Bash, PowerShell, and cmd deletion forms, refuse a target that is a bound root, a recursive target that contains one, or a root-wide deletion selection. Cover both direct root-wide wildcards and common native shell forms that enumerate the root's children for deletion. Refuse a recognizable deletion of explicit paths when the combined targets would remove every current child of a root, including a single remaining child. Evaluate all protected roots before launching any part of the request.
- Use the shell's path and command rules to compare resolved targets. Cover absolute and working-directory-relative operands, quoted paths, and straightforward directory changes within a command. Respect path component boundaries and the host's filesystem semantics. Quoted text passed as data should not trigger a refusal.
- Permit targeted deletion such as removing a build folder while other project contents remain. Existing file edits and the ordinary shell permission flow continue to work.

## Enforcement

Extend the model raw shell request validation used during approval review and again before execution. Feed it trusted binding information from the owning workspace, separate from the model's arguments and the narrower set of roots admitted to structured file tools. Recheck the binding information before execution. If workspace protection data cannot be determined, refuse the model shell request instead of treating the workspace as unbound. Use the existing safety refusal path, with a reason that does not expose host paths. User-authored `!` commands and the user-only terminal retain their existing authority.

This is command detection for accidental wipes. Arbitrary scripts, programs, dynamic shell expressions, and sequences of operations whose combined effects cannot be inferred from the request are outside its detection guarantee. SSH bindings and external tools are outside this local raw shell guard. Preserve ADR-094's description of the raw shell as host-authority execution.

## Verification and decision record

Use targeted tests with temporary folders. Cover direct root and parent deletion, whole-root selections, explicit deletion of the last remaining child, sibling roots, paths with spaces, shell working directories, and permitted targeted deletion. Verify that a blocked command never reaches the executor, including when a session grant already exists. Exercise the actual owning-workspace wiring with multiple bindings and a different active workspace.

Amend `backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md` before implementation to record the new model shell refusal policy and its limits. Keep the shell parsing and path comparison implementation choices for the plan.
