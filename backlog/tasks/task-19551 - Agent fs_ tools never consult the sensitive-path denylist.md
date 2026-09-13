---
id: TASK-19551
title: >-
  Agent fs_* tools never consult the sensitive-path denylist
status: Done
assignee: []
created_date: '2026-08-21 20:01'
labels:
  - security
  - agents
  - tools
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — its **Tier 1
#1**. CONFIRMED by the lane, and **independently re-confirmed by the review
controller** and again at this branch base.

The `fs_*` agent file-tool family never calls `is_sensitive_path`. The entire
guard on the agent-supplied path is:

```
tldw_chatbook/Tools/local_tool_impls.py:53
    return validate_path(path, workspace_root, allow_hidden=True)
```

`resolve_workspace_path` is the single choke point the seven `fs_*` tools
funnel through (dispatched from `tldw_chatbook/Agents/local_tool_provider.py`).
A `grep is_sensitive_path` across `Tools/` and `Agents/` returns **9 hits in
`Tools/file_operation_tools.py` and 1 in `Agents/run_log.py` — and ZERO in
`local_tool_impls.py`, `patch_tool_impls.py`, or `local_tool_provider.py`**.
`Utils/sensitive_paths.py` names only the five `file_operation_tools`
enforcers as the contract's enforcement points; the `fs_*` family postdates
that contract and never joined it.

Paths the lane proved are accepted without ever asking — **all of them already
in the denylist**: `~/.ssh/id_rsa`, `~/.config/tldw_cli/config.toml`,
`~/.aws/credentials`, `mcp_permissions.json`, `chachanotes.db`.

Two consequences, and the second is the severe one:

1. **Credential read.** Keys are read into the agent transcript, which is then
   sent to a model provider.
2. **One-step permission-gate bypass.** `fs_write`/`fs_patch` can rewrite
   `mcp_permissions.json`, flipping tools from `ask` to `allow`. The gate that
   is supposed to be fail-closed can be disarmed by a single tool call that the
   gate itself permits.

Reachability is worse than "a malicious user": this is reachable **by prompt
injection from fetched web content**, since the agent loop will act on
instructions embedded in pages it retrieves.

Aggravating configuration: `config.py:2952` ships `workspace_root` commented
out, so launching the app from `$HOME` confines the tools to `$HOME` — which
contains every path listed above.

The lane's bottom line for its whole tier applies here exactly: this is a
**seam that never adopted an existing correct primitive**, not a wrong
primitive. The denylist is correct; these tools just do not call it.

## Acceptance Criteria

- [x] `is_sensitive_path` is consulted inside `resolve_workspace_path`, so that
      all seven `fs_*` tools inherit the check from the one choke point rather
      than each re-implementing it
- [x] Reading, writing, or patching any denylisted path through an `fs_*` tool
      is refused or routed to an explicit approval, matching the behaviour the
      `file_operation_tools` enforcers already have
- [x] Specifically pinned by test: `~/.ssh/id_rsa`, `~/.aws/credentials`, the
      app's own `config.toml`, `mcp_permissions.json`, and `chachanotes.db` are
      not silently readable or writable via `fs_read`/`fs_write`/`fs_patch`
- [x] A regression test pins that `fs_write`/`fs_patch` cannot rewrite the
      permission store — the gate cannot be disarmed by a gated tool
- [x] `Utils/sensitive_paths.py`'s enforcer list is updated so the contract
      names its real enforcement points, and a test fails if a new tool reaches
      the filesystem without passing through the choke point
- [x] The shipped `workspace_root` default is reviewed: launching from `$HOME`
      must not silently confine the agent to a root containing the user's
      credentials

## Implementation Plan

1. Read both file-tool families end to end and mirror the sibling
   (`file_operation_tools.py`) exactly — error shape, context reuse, the
   `refuses_new_directory_chain` write guard.
2. Write the born-red tests FIRST and run them at a pristine `origin/dev`
   worktree, so the accepted credential read / permission-store rewrite is
   recorded verbatim before any fix exists.
3. Enforce the denylist inside `resolve_workspace_path` (the single choke
   point), threading a read/write/list intent rather than weakening the check.
4. Cover what the choke point structurally cannot: the three enumerating tools
   walk their own candidates, so each filters entries against the same denylist
   with one shared `SensitivePathContext`.
5. Update the `Utils/sensitive_paths.py` contract to name the second family,
   add the drift pin and the AST tripwire, re-run the affected suites, and
   baseline every failure against `origin/dev` before attributing it.

## Implementation Notes

**Approach.** The denylist is now enforced inside
`Tools/local_tool_impls.py::resolve_workspace_path` — the one choke point every
`fs_*` core function, `patch_tool_impls.patch_files` and `git_tool_impls`
already funnelled through — so all seven tools inherit it from a single change
instead of seven copies that can drift (that drift is exactly how this bug
happened: the contract's enforcer list was a docstring, and the `fs_*` family
shipped later without ever joining it).

The function grew an `intent: "read" | "write" | "list"` keyword and an optional
pre-resolved `SensitivePathContext`:

* `intent` selects the refusal verb, so refusals read the same as the sibling
  family's (`Refused: '<path>' is a protected path and cannot be read/written/
  listed`), and for `"write"` it additionally applies
  `refuses_new_directory_chain` to the target's parent chain — the same guard
  `WriteFileTool` consults before `mkdir(parents=True)`. No tool in this family
  creates directories today (a write target's parent must already exist), so it
  short-circuits on the first existing ancestor; it is wired now so a future
  `create_directories`-style option cannot silently reintroduce TASK-849. A
  test pins that no `mkdir`/`makedirs` call exists in these modules.
* `context` lets a caller resolve the ~11 config accessors behind the denylist
  once per CALL rather than once per path (`patch_files`' multi-file loop and
  the three enumerating tools do this). Deliberately not a module-level cache —
  see `resolve_sensitive_context`'s own docstring for why.

**What the choke point structurally cannot cover.** `list_directory`,
`glob_files` and `grep_files` resolve only the workspace ROOT through it, then
walk their own candidates, so each now filters entries against the same
denylist with the shared context. `grep_files` is the sharpest — it READS every
file it walks and prints matching lines, so its check runs *before* the read;
at base, a home-rooted workspace and a pattern as bland as `KEY` dumped
`~/.ssh/id_rsa` into the transcript.

**`allow_hidden=True` is KEPT, deliberately — with one correction to the
argument.** ADR-032 adopted it for this family precisely because real
workspaces need `.git`/`.github`/dotfile configs, and a coding agent that
cannot read `.github/workflows/ci.yml` is useless. Dotted names are how
`~/.ssh` and `~/.aws` are spelled, but "starts with a dot" is a name heuristic,
not a security boundary — `is_sensitive_path` answers that question by RESOLVED
ancestry, so `~/.sshfoo` is not mistaken for `~/.ssh` and a symlink cannot
smuggle a path past it. Reversing the flag would contradict a signed ADR, break
the family's core use case, and still not cover non-dotted credentials.

The correction (review round, re-measured while filing): resolved-ancestry
matching is only as strong as what the denylist ENUMERATES, and `~/.netrc`,
`~/.git-credentials`, `~/.npmrc`, `~/.pypirc`, `~/.cargo/credentials.toml` and
`~/.config/gh/hosts.yml` are in none of its entries — `fs_read` returns all six
bodies, while the sibling family refuses them at confinement because its
`allow_hidden` defaults False. So for DOTTED credential paths this family is
strictly the weaker of the two, and the drift pin's comment calling that
asymmetry "an honest distinction" is only partly right: it is partly residue.
Filed as **TASK-19633** (denylist CONTENT gap, shared with the primitive); the
decision to keep `allow_hidden=True` stands, since flipping it would fix none
of the non-dotted cases. Both halves of the decision are pinned:
`test_fs_read_refuses_dotfile_component_credential_path` (a dotted path IS
refused, and only the denylist can be what refuses it) and
`test_benign_workspace_dotfiles_stay_readable` (`.github/`, `.gitignore` stay
readable — if that test ever has to change, ADR-032 needs amending).

**Born-red evidence** (final tests, run against a pristine `origin/dev`
worktree): 14 failed / 4 passed, each failure printing the exact accept —
`fs_read(~/.ssh/id_rsa)` returning the key body, `fs_read(config.toml)`
returning `api_key = '…'`, `fs_write/fs_edit/fs_patch(mcp_permissions.json)`
returning `wrote 15 characters` / `made 1 replacement` / `patched
mcp_permissions.json`, `fs_grep` emitting the key contents, `fs_glob`/`fs_list`
emitting `.ssh/id_rsa`. After the fix: 18 passed. The 4 that pass on both sides
are the controls (the allow_hidden pin, the AST tripwire, the no-mkdir pin, and
the unchanged confinement contract).

**Drift pin.** `test_both_file_tool_families_refuse_the_same_denylisted_paths`
runs six denylisted paths through BOTH families and requires the shared oracle
(`is_sensitive_path`) plus both tools to refuse, with no content leaked. It
keeps an honest distinction: for a candidate whose parent directory is dotted
(`~/.ssh/id_rsa`), the second family refuses at *confinement* — its
`validate_path_multi` rejects a dotted base directory outright — so only the
non-dotted candidates assert its refusal is denylist-sourced.

**A confirmed live leak found during this work, in the neighbouring `git_*`
family — filed as TASK-19632.** The choke point governs a path the model
NAMES. `path` is optional on `git_status`/`git_log`/`git_diff`, and when it is
omitted `Agents/local_tool_provider.py:444-445` returns the repo root and
stops, so nothing reaches the denylist; git then enumerates the repository and
the tool returns its output verbatim. Measured on a `$HOME`-rooted workspace
containing a synthetic `~/.ssh/id_rsa`: `git_diff` returns the file's CONTENT,
and — this is worse than my first report — it needs **no write primitive and no
dirty worktree**, because `commit_range="HEAD~1..HEAD"` reads it out of
history. `git_diff(stat=True)` and a dirty-tree `git_status` return the NAME;
`git_log` leaks nothing (commit metadata only), so my original "three git
tools" framing was wrong and is corrected in the filing. Not fixed here: it is
a different tool family, and the fix (pathspec exclusion vs. parsing diff
output) is a design decision that must not ship as a hurried diff filter.

**The contract text was rewritten in the review round**, because the first
version said the choke point covered "every `fs_*` and `git_*` core function
... so the check lives in ONE place for all of them". A reader could have used
that to conclude `git_*` was safe — which is precisely the failure this task
exists to fix (an enforcer list that describes only the implementations
existing when it was written). `Utils/sensitive_paths.py`,
`Tools/local_tool_impls.py`'s module docstring and `resolve_workspace_path`'s
now state the path-argument/output distinction explicitly and name TASK-19632
and TASK-19633 as open exceptions.

**AC5's tripwire covered 2 of the family's 3 modules** in the first round
(`local_tool_impls`, `patch_tool_impls`), so a new `git_*` tool would have had
zero coverage. Fixed: `git_tool_impls` is in the module tuple, with the three
documented indirect resolvers (`prepare_repository`, `_prepare_for_path`,
`_repo_relative_path`) accepted — each of which is itself required to call the
choke point DIRECTLY, so the indirection is verified rather than assumed.
Mutation-tested four ways: without the accept-set the four repo-scoped git
tools are reported (reproducing the gap exactly); as shipped all three modules
are clean; a synthetic `git_show` that resolves its own path IS reported; and
an indirect resolver that stops calling the choke point IS reported.

**`workspace_root` default (AC6), reviewed — kept, documented.** The default
(app cwd at startup) is an explicit, signed ADR-032 trade-off, and changing it
would break every existing coding-agent workflow; it needs an ADR amendment,
not a side effect of a security fix. What made it dangerous was that the root
was the ONLY check — that is now false: credential/gate-state/app-state paths
are refused regardless of the root. The shipped `config.py` comment no longer
states the default without stating its consequence (it names the `$HOME`
launch, points at `Utils/sensitive_paths.py`, and says plainly that the
denylist is a guardrail, not a substitute for pointing the root at one project
directory).

**Cost.** The denylist resolution is the same one the sibling family already
pays per call: measured on this branch vs base, `fs_read` ~0.1ms → ~7ms per
call, a 300-entry `fs_list` ~1ms → ~50ms, `fs_grep` over 300 files ~18ms →
~67ms. Parity with the existing enforcing family was preferred over caching the
security primitive.

**Modified/added files.** `tldw_chatbook/Tools/local_tool_impls.py` (choke
point + three enumerating filters), `tldw_chatbook/Tools/patch_tool_impls.py`
(shared context, write intent), `tldw_chatbook/Agents/local_tool_provider.py`
(`path_targets` intents), `tldw_chatbook/Utils/sensitive_paths.py` (contract now
names BOTH enforcing families), `tldw_chatbook/config.py` (shipped
`workspace_root` comment), new `Tests/Tools/test_local_tool_sensitive_paths.py`.
