# Workspace Shell Deletion Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refuse recognizable model shell commands that delete or empty any bound local folder of the chat's workspace, including under an existing session grant.

**Architecture:** The Console supplies the model shell provider with trusted binding locators from the owning workspace. The provider checks before approval and checks again with refreshed bindings at dispatch. The raw CLI executor checks the dispatched snapshot with the dialect of the exact argv it hands to the worker.

**Tech Stack:** Python 3.12+, standard library, pytest, Bash/sh, PowerShell, and cmd.

**Spec:** [Workspace shell deletion guard](../specs/2026-11-11-workspace-shell-deletion-guard-design.md)

## Global Constraints

- Protect every local binding, regardless of access mode or project-instruction selection. The active tab and model-selected working directory do not determine protection scope.
- Refuse before a shell is started. Neither approval nor a session grant overrides a refusal. Keep model-facing refusals free of raw protected paths.
- Keep targeted deletes, file edits, user-authored `!` commands, and the existing system-wide safety floor available under their current rules.
- Document coverage as static detection of recognizable local shell commands. Arbitrary scripts, dynamic shell expressions, SSH bindings, and external tools are outside this guard.
- Each dispatch uses the latest binding snapshot taken for that request and the executor inspects the filesystem before launch. Concurrent requests, registry changes after dispatch, and filesystem changes after inspection can race the checks. The guard provides no atomic filesystem transaction.
- Use the standard library for recognition and path handling. Keep the supported command forms bounded instead of implementing a general shell interpreter.
- Use temporary test directories and launch probes. Run targeted suites only, per repository instructions.

ADR required: yes

ADR path: `backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md`

Reason: This amends the existing model raw shell authority policy with a refusal that cannot be overridden by approval.

## Review Focus

- A root with a single hidden child still refuses deletion of its last child. A targeted delete with another child remaining is allowed (Task 1).
- Two explicit deletion commands that together remove every immediate child of one root are refused as a unit (Task 1).
- Quoted prose and a directory with a similar prefix do not count as deletion of a bound root (Task 1).
- Filtered enumeration and mutually exclusive branches must not be counted as a single whole-root selection (Task 1).
- A Windows `auto` request must be evaluated against cmd when PowerShell is unavailable, and the worker must launch the checked argv (Task 2).
- A binding added after review or a failed registry lookup cannot be bypassed by an existing session grant (Task 2).
- The binding read must retire any SQLite connection opened just for the lookup on a transient worker (Task 2).

## File Map

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/Tools/workspace_shell_guard.py` | Bounded shell parsing and protected target evaluation |
| `tldw_chatbook/Tools/raw_cli_executor.py` | Shared shell selection, trusted request metadata, and final validation against launch argv |
| `tldw_chatbook/Agents/raw_shell_tool_provider.py` | Model-only binding lookup and approval or dispatch refusal |
| `tldw_chatbook/Chat/console_provider_composition.py` | Wire the owning workspace and the live binding registry into the provider |
| `Tests/Tools/test_workspace_shell_guard.py` | Recognition, path semantics, and allowed operations |
| `Tests/Tools/test_raw_cli_executor_contract.py`, `Tests/Tools/test_raw_cli_executor_process.py`, `Tests/Agents/test_raw_shell_tool_provider.py` | Dialect, prelaunch refusal, approvals, and production provider composition |
| `backlog/decisions/094-raw-and-virtual-cli-execution-boundaries.md`, `Docs/Architecture/console-file-authority.md` | Decision record and runtime behavior |

## Task 1. Implement the protected target check

**Interfaces:** Add `workspace_deletion_violation(command: str, *, dialect: Literal["bash", "powershell", "cmd"], cwd: Path, protected_roots: tuple[Path, ...]) -> Literal["workspace-folder-delete", "workspace-folder-wipe"] | None` in `Tools/workspace_shell_guard.py`. The function inspects paths and directory entries and never executes command text. Propagate filesystem errors that prevent determining protection, for Task 2 to turn into a refusal.

- [x] **Record the task and decision.** Create a Backlog task linked to the approved spec, move it to In Progress, and link this plan. Use Task 25905 as prior art. Amend ADR-094 with the model-only safety floor and its detection limits before writing product code.
- [x] **Write the guard regressions.** In `Tests/Tools/test_workspace_shell_guard.py`, parameterize `test_protected_workspace_deletions` over root deletion, recursive ancestor deletion, root-wide wildcard, native child enumeration, and exact deletion sets. Assert `workspace-folder-delete` for root or recursive ancestor deletion and `workspace-folder-wipe` for bulk or complete child deletion. Use temp roots with spaces, a second bound root, a last hidden child, and a two-command deletion of all children. Add `test_targeted_deletions_and_quoted_data_are_allowed` with a build directory alongside source, a similarly prefixed sibling, a quoted command example passed as data, and a literal quoted wildcard filename on POSIX. Assert `None` and leave fixtures intact. Include absolute and relative targets, a literal directory change before deletion, PowerShell `-LiteralPath` versus `-Path`, and Windows case, separators, and a junction on the native platform when available.

  Representative Bash assertions with `root` containing `src` and `.keep`:

  ```python
  opts = dict(dialect="bash", cwd=root, protected_roots=(root, other_root))
  assert workspace_deletion_violation("rm -r src", **opts) is None
  assert workspace_deletion_violation("rm -r src; rm .keep", **opts) == "workspace-folder-wipe"
  assert workspace_deletion_violation("find . -mindepth 1 -delete", **opts) == "workspace-folder-wipe"
  ```
- [x] **Add false-positive controls.** Assert a filtered Bash `find` and a PowerShell `Get-ChildItem -Filter` pipeline that select a subset pass. Assert two single-target deletion alternatives joined by `||` are not summed into a wipe. Include traversal through a link or junction to a bound root and deletion of just a link entry with other project contents remaining.
- [x] **Run the guard regressions before implementation.** Run `python -m pytest Tests/Tools/test_workspace_shell_guard.py -q`. Confirm the new cases fail because the guard interface is absent or its behavior is missing.
- [x] **Implement the guard interface.** Recognize Bash/sh `rm`, `rmdir`, and unfiltered `find … -delete`, PowerShell `Remove-Item` and built-in aliases plus unfiltered `Get-ChildItem | Remove-Item`, and cmd `del`/`erase` and `rd`/`rmdir`. Account for filters and depth or type restrictions before classifying enumeration as root-wide. Parse option operands, end-of-options markers, and direct root-wide wildcard selections in the correct dialect. Preserve quoting and distinguish command position from argument text. A quoted wildcard is literal in Bash, while PowerShell `-Path` can expand it even when quoted and `-LiteralPath` cannot.
- [x] **Evaluate recognized paths and command sequences.** Track straightforward literal `cd` or `Set-Location` with each shell's separators. Compare component paths and canonical targets under native filesystem semantics. Distinguish traversal through a link or junction from a command that only removes the link entry itself. Combine covered immediate root children across straightforward sequences of `;`, newline, cmd `&`, and `&&` as supported by the dialect. Inspect individual recognized commands across alternatives, with no combined coverage inferred across `||` or complex control flow. Check an exact deletion set against current immediate entries, including hidden entries, and stop on the first untouched entry. A targeted delete inside a surviving child directory does not count as removing the child. A direct root-wide selection is blocked even when the shell's wildcard would skip hidden files.
- [x] **Verify and commit the guard.** Run `python -m pytest Tests/Tools/test_workspace_shell_guard.py -q` and confirm all cases pass. Review the scoped diff and commit the detector, tests, and ADR amendment together.

## Task 2. Enforce the check on model shell calls

**Interfaces:** Add `protected_roots: tuple[Path, ...] = field(default=(), repr=False)` to `RawCliRequest` as internal metadata. Add keyword-only `protected_roots: Callable[[], tuple[Path, ...]] = tuple` to `RawShellToolProvider.__init__`. Add `resolve_checked_workspace_shell_argv(request: RawCliRequest) -> tuple[str, ...]` in `raw_cli_executor.py`. Factor the existing shell selection so this helper can obtain the dialect and argv together and `resolve_shell_argv` keeps its public return type and candidate order.

- [x] **Write the approval and authority regressions.** In `Tests/Agents/test_raw_shell_tool_provider.py`, add `test_workspace_wipe_never_reaches_runtime_even_with_session_grant` using a recording runtime. Assert no approval row with the grant absent, then assert a blocked result with the grant active, a safety rule with no protected locator, and zero runtime calls. Assert that a targeted delete reaches the runtime under the grant. Add `test_workspace_guard_refreshes_roots_after_review` with a binding added between review and invocation and `test_workspace_guard_refuses_when_binding_lookup_fails` with a raising resolver. Assert both refuse before dispatch. Add `test_console_raw_shell_protects_all_bindings_of_owning_workspace`, invoking the real controller composition with an unrelated active workspace, a selected project root, a second read-only local binding, and an SSH row. Assert the second local root is protected and only the owning workspace is queried. Cover the known Default/global no-binding case and the user-authored request control. Exercise a real registry read on a worker and verify that the lookup retires a connection opened for that operation.
- [x] **Write the executor regressions.** In `Tests/Tools/test_raw_cli_executor_contract.py`, add `test_guard_uses_the_resolved_auto_shell` with a Windows executable lookup that resolves only `cmd.exe`. Assert `auto` catches cmd syntax. In `Tests/Tools/test_raw_cli_executor_process.py`, add `test_worker_uses_checked_shell_argv` using a launch probe, asserting the worker uses the executor's argv without a second selection. Add `test_guarded_wipe_refuses_before_worker_setup` and `test_missing_shell_keeps_guarded_request_outcome`. Use a real executor with a launch probe to prove a refused request does not launch. Use a harmless child command to verify successful launch and a missing shell to verify the `shell_unavailable` outcome.
- [x] **Cover stale approval after a guard refusal.** In `Tests/Agents/test_raw_shell_tool_provider.py`, add `test_workspace_guard_discards_one_use_approval_on_refusal`. With a real `approve_once` stamp, add a protected binding after review and invoke the command. Assert the call is blocked and the stamp is unavailable for a later dispatch even if the binding changes again.
- [x] **Run the new regressions before wiring.** Run `python -m pytest Tests/Agents/test_raw_shell_tool_provider.py Tests/Tools/test_raw_cli_executor_contract.py Tests/Tools/test_raw_cli_executor_process.py -q -k 'workspace_guard or workspace_wipe or protects_all_bindings or resolved_auto_shell or checked_shell_argv or guarded_wipe or missing_shell'`. Confirm the new cases fail because protected requests are not yet enforced.
- [x] **Wire the guard into raw execution.** Keep `validate_raw_cli_request` responsible for the existing shared checks. Put workspace inspection in `resolve_checked_workspace_shell_argv`. Raise `RawCliHardlineViolation` for either deletion rule or `workspace-protection-unavailable` on an inspection failure. For a guarded model request, the executor calls the helper before worker setup and passes the checked argv as an internal worker argument. Pass `None` to keep existing worker-side selection on unguarded requests. If the guarded lookup finds no shell, use an empty tuple to settle as `shell_unavailable` without a fallback selection. This avoids repeating directory enumeration in both runtime and executor validation.
- [x] **Wire the owning workspace into approval and dispatch.** In the provider composition called by `ConsoleChatController._compose_raw_shell_provider`, capture `turn_context.project_authority.workspace_id` when present and otherwise use `self.store.session_workspace_id(session_id)`. In the provider's worker-side review and dispatch paths, read `workspace_registry_service.list_runtime_bindings(workspace_id)` inside `operation_owned_connection(registry.db)` and filter to local filesystem records with their bound locators. Keep read-only and missing local bindings in the protection set. Missing local paths retain their lexical protection and have no current children to count. Only known Default/global chats or successful named-workspace lookups with no local bindings return an empty tuple. `pending_gate_for` runs the checked shell helper before constructing a card. If the shell is unavailable, retain the normal permission flow and let the executor settle the existing unavailable outcome. `invoke` reads the bindings again, checks the request even with a grant, and hands the snapshot to the executor for its launch check. Route lookup errors and late executor safety refusals through the existing blocked result. Preserve model and approval argument schemas.
- [x] **Retire the call's one-use approval on workspace refusal.** The provider currently returns from validation failures before reaching `_pop_stamp_detail`. Consume the refused call's approval stamp on the new workspace refusal path, using the existing stamp ownership rules. Keep session grants subject to the guard on every call.
- [x] **Update the architecture notes.** Describe the guard and its limits in `Docs/Architecture/console-file-authority.md`.
- [x] **Verify the integrated guard.** Run `python -m pytest Tests/Tools/test_workspace_shell_guard.py Tests/Tools/test_raw_cli_hardline.py Tests/Tools/test_raw_cli_executor_contract.py Tests/Tools/test_raw_cli_executor_process.py Tests/Agents/test_raw_shell_tool_provider.py Tests/Chat/test_console_raw_shell_approval.py -q`. Check Ruff lint and format on changed Python files with the repository configuration. Confirm the touched tests pass and compare any unrelated failures with the base checkout before reporting them.
- [x] **Commit the integration.** Review the scoped diff, record actual results and any platform limits in the Backlog task, and commit the integration with its tests and documentation.
