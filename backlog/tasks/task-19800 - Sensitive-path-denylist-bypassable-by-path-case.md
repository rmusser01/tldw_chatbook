---
id: TASK-19800
title: 'Security: sensitive-path denylist bypassable by changing path case'
status: Done
assignee: []
created_date: '2026-08-22'
labels:
  - security
  - tools
dependencies:
  - TASK-19551
  - TASK-19700
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The credential/gate-state denylist (`Utils/sensitive_paths.py`, added by
TASK-19551) compared resolved paths with exact, case-sensitive equality. On
macOS and Windows — where filesystems are case-insensitive by default, and
where `Path.resolve()` does NOT canonicalise case — a case-variant spelling
reaches the same file while comparing unequal to every denylist entry.

Found while fixing TASK-19700's `.git` write guard for exactly this
weakness, which Qodo flagged on PR #1934; checking whether the older
denylist shared it showed that it did.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A case-variant spelling of a denylisted path receives the same verdict as the canonical spelling, for denied files, denied directories, database sidecars, and the direct-child container rule
- [x] #2 Ancestry and lookalike behaviour are unchanged: `~/.sshfoo` is still not `~/.ssh`
- [x] #3 The end-to-end tool path (`fs_read`) refuses a case-variant of a denylisted file, not just the predicate
- [x] #4 Confinement checks are explicitly NOT case-folded, with the reasoning recorded, since folding those would admit paths rather than refuse them
<!-- AC:END -->

## Implementation Notes

Routes every denylist comparison through `_compare_key`, which casefolds
path components; `_same_path` and `_is_within` replace the previous `==` and
`in .parents` checks for denied files, DB paths and their sidecars, denied
directories, and the direct-child container rule.

**Evidence this was live, not theoretical.** Before the fix, end-to-end
through `fs_read` against the app's own denylisted config:

```
tldw_cli/config.toml   -> refused (Refused: '...' is a protected path...)
TLDW_CLI/config.toml   -> ALLOWED, returned 32782 chars
```

That is the real config file — the one holding provider API keys — read
through the denylist by changing the case of one path component. The same
shape reaches `~/.ssh`, `~/.aws`, and `mcp_permissions.json`; bypassing that
last one turns every `ask` into `allow`, which is the permission-gate bypass
TASK-19551 exists to prevent. Reachable by prompt injection on the two
platforms where case-insensitive filesystems are the default.

**Casefolding is unconditional, not platform-gated or probed.** Platform is
only a proxy for the real question (macOS can be configured case-sensitive,
Linux can mount a case-insensitive volume), a per-path probe would add I/O
to a check that runs on every candidate, and the two error directions are
not symmetric: over-refusing a genuinely distinct `~/.SSH` on a
case-sensitive filesystem costs one explained refusal of a very unusual
path, while under-refusing leaks a credential.

**Confinement is deliberately left case-sensitive** (AC #4). The two checks
fail in opposite directions: for a denylist, folding produces extra
refusals and fails safe; for confinement ("is this inside the allowed
root?"), folding produces extra ADMISSIONS — on a case-sensitive filesystem
`/Root/evil` would begin counting as inside `/root`. `is_within` in
`Tools/file_operation_tools.py` is left alone for that reason, and the
asymmetry is documented in `_compare_key` so a later "consistency" pass does
not apply it there.

**Not addressed, deliberately:** Windows also normalises trailing dots and
spaces (`.ssh.` → `.ssh`). That is a real second normalization, but it is
Windows-only — on POSIX `.ssh.` is a genuinely different directory, so
stripping there would over-refuse for no security gain — and it wants a
platform-gated rule of its own rather than being folded into the
case-comparison change.

**Files:** `tldw_chatbook/Utils/sensitive_paths.py`,
`Tests/Tools/test_local_tool_sensitive_paths.py`.
