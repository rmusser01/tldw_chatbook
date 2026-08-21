---
id: TASK-19633
title: >-
  Denylist does not enumerate common dotfile credentials
status: To Do
assignee: []
created_date: '2026-08-21 12:05'
labels:
  - security
  - agents
  - tools
priority: medium
dependencies: []
---

## Description

Demonstrated by TASK-19551's reviewer and independently re-measured while
filing this. TASK-19551 made the `fs_*` family consult
`Utils/sensitive_paths.py` and kept `allow_hidden=True` (ADR-032) on the
argument that "starts with a dot" is a name heuristic while `is_sensitive_path`
answers the question properly, by resolved ancestry. The decision is right; the
argument is overstated in one specific way, and this task records the gap it
hides.

Resolved-ancestry matching is only as strong as what the denylist ENUMERATES.
`_SENSITIVE_DIRS` lists `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gcloud`,
`~/.docker`, `~/.kube`, `~/.local/share/keyrings` — and nothing else. Common
credential files outside those trees are simply not refused.

Measured on a `$HOME`-rooted workspace with synthetic marker files (isolated
HOME; "family A" = the workspace-confined `fs_*` tools, "family B" = the
sandbox-confined `Tools/file_operation_tools.py` tools):

| path | `is_sensitive_path` | `fs_read` (A) | `read_file` (B) |
| --- | --- | --- | --- |
| `~/.netrc` | False | **returns body** | refused (confinement) |
| `~/.git-credentials` | False | **returns body** | refused (confinement) |
| `~/.npmrc` | False | **returns body** | refused (confinement) |
| `~/.pypirc` | False | **returns body** | refused (confinement) |
| `~/.cargo/credentials.toml` | False | **returns body** | refused (confinement) |
| `~/.config/gh/hosts.yml` | False | **returns body** | refused (confinement) |

Two consequences:

1. **A denylist-CONTENT gap**, shared with the primitive itself: family B is
   only protected here by accident — its `validate_path_multi` defaults
   `allow_hidden=False`, so a dotted component is rejected at confinement
   before the denylist is ever consulted. Widening a sandbox root to a
   non-dotted directory that contains one of these (or binding such a folder)
   would expose family B too.
2. **For dotted credential paths, family A is strictly weaker than family B.**
   That asymmetry deserves a decision, not silence. TASK-19551's own drift-pin
   comment describes it as "an honest distinction" between the two families'
   refusal reasons; that is true as far as it goes, but part of what it is
   describing is residue, not design.

Scope note: this is not an argument to reverse `allow_hidden=True`. Doing so
would contradict ADR-032, break the coding-agent use case (`.github/`,
`.gitignore`, `.pre-commit-config.yaml`), and still not refuse a non-dotted
credential file. The fix belongs in the denylist's CONTENT and in whatever
supplementary rule the team wants for high-signal credential filenames.

## Acceptance Criteria

- [ ] The paths measured above are refused by BOTH file-tool families, in a
      configuration where confinement alone does not explain the refusal (root
      set so no dotted component appears in the relative portion)
- [ ] The additions are expressed the way the rest of the module already
      prefers — a rule or accessor where one exists, a literal only where the
      location genuinely is fixed — and the module docstring says which
      choice was made and why
- [ ] A born-red test per added path shape, showing the accept before and the
      refusal after
- [ ] The asymmetry is resolved deliberately and recorded: either both families
      apply the same hidden-component policy, or the module docstring states
      which family is stricter and why that is acceptable
- [ ] `Utils/sensitive_paths.py`'s docstring paragraph naming this task
      (added by TASK-19551) is updated or removed to match the new state
- [ ] TASK-19551's drift-pin comment in
      `Tests/Tools/test_local_tool_sensitive_paths.py` is re-read: the phrase
      "an honest distinction" must still be true after this change

## Notes

Worth deciding at the same time: whether the denylist should grow a
NAME-based rule for a small set of unambiguous credential filenames
(`.netrc`, `.git-credentials`, `credentials.toml`, ...) in addition to
location-based rules. The module deliberately prefers rules over
enumerations because enumerations trail reality — that reasoning applies to
this gap too, and an enumeration of six filenames will itself be stale within
a year.
