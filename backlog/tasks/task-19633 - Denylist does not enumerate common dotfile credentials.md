---
id: TASK-19633
title: >-
  Denylist does not enumerate common dotfile credentials
status: Done
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

- [x] The paths measured above are refused by BOTH file-tool families, in a
      configuration where confinement alone does not explain the refusal (root
      set so no dotted component appears in the relative portion)
- [x] The additions are expressed the way the rest of the module already
      prefers — a rule or accessor where one exists, a literal only where the
      location genuinely is fixed — and the module docstring says which
      choice was made and why
- [x] A born-red test per added path shape, showing the accept before and the
      refusal after
- [x] The asymmetry is resolved deliberately and recorded: either both families
      apply the same hidden-component policy, or the module docstring states
      which family is stricter and why that is acceptable
- [x] `Utils/sensitive_paths.py`'s docstring paragraph naming this task
      (added by TASK-19551) is updated or removed to match the new state
- [x] TASK-19551's drift-pin comment in
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


## Implementation Plan

1. Re-measure the six paths on this branch, with an isolated `$HOME`.
2. Decide rule vs enumeration for each of them and record the reasoning in the
   module docstring rather than in a commit message.
3. Add the chosen instrument(s), composed with -- not duplicating -- the
   module's existing comparison normalization.
4. Construct, deliberately, a configuration in which each family's refusal
   cannot be explained by confinement, and pin it.
5. Bound the cost: pin the near-misses and container shapes that must stay
   readable.
6. Update the docstring paragraph naming this task, and re-read TASK-19551's
   drift-pin comment against the result.

## Implementation Notes

**Decision: BOTH instruments, chosen per case, and the docstring says so.**

* A **name rule** (`_SENSITIVE_FILE_NAMES`) for filenames that identify a
  credential store wherever they appear: `.netrc`, `_netrc` (the Windows
  spelling -- also the one non-dotted probe that makes the AC's "confinement
  cannot explain it" configuration constructible for that family of names),
  `.git-credentials`, `.npmrc`, `.pypirc`, `credentials`, `credentials.toml`.
* A **location rule** for `~/.config/gh`, added to `_SENSITIVE_DIRS`, because
  there the LOCATION is the unambiguous part and the filename is not --
  `hosts.yml` is just as often an Ansible inventory.

The argument, in short: both instruments are enumerations and both trail
reality, names no less than locations. The name rule wins where it applies
because one entry covers unbounded locations while one location entry covers
exactly one, and because a tool's config DIRECTORY migrates between XDG/legacy/
OS conventions far more often than its credential FILENAME ever changes. A
location rule could not have covered most of this set at all: a project-local
`.npmrc`/`.pypirc` carries an auth token exactly like the home one, and
refusing all of `~/.cargo` or `~/.config/git` would take down things an agent
legitimately reads.

The cost is real and is stated rather than hidden: an agent cannot read a test
fixture named `credentials`, and the refusal names it as protected. `.env` is
the deliberate omission -- as often build configuration as secrets, and
refusing it would break the ADR-032 coding-agent use case this module exists to
keep working. Directory-shaped matches are exempted by the same `is_dir()` gate
the direct-child-file rule uses, so a container named `credentials/` stays
listable.

**The asymmetry, resolved deliberately (AC4).** The two families'
hidden-component policies still differ and that difference is now design
throughout: family 1 (`file_operation_tools`) refuses any dotted component at
confinement; family 2 (`fs_*`) passes `allow_hidden=True` per ADR-032. Family 1
is stricter for dotted NAMES, which is acceptable because the roots are
different kinds of place -- family 1's sandbox root is app-owned storage where
a dotfile has no legitimate purpose, family 2's is a user source tree where
dotfiles are the point. What is no longer allowed to differ is the shared
oracle's ANSWER, and it does not: all six measured paths are refused by the
denylist itself under either family. The residue is gone; the policy split
remains.

**Constructing "confinement alone does not explain it".** Family 2 is easy
(`allow_hidden=True`, root at the file's own directory). Family 1 took care,
and is done three ways: the two name-rule paths that can be spelled without a
dot are planted at a NON-dotted location under a NON-dotted root
(`projects/repo/_netrc`, `credentials.toml`, `credentials`) -- which is also
the name rule's whole point, that a moved credential is still a credential; the
location-rule path keeps its location but is rooted at `~/.config/gh`, whose
own basename is not dotted and whose relative portion is a plain `hosts.yml`;
and the four inherently-dotted names go through `ListDirectoryTool` with
`include_hidden=True`, the one family-1 seam that lists dotted entries by
request, so `is_sensitive_path` is the only thing that can withhold them
(`.gitignore` is the control proving the listing works).

**Composed with TASK-19800, which merged mid-task.** 19800 made every denylist
comparison go through one folded key, `_compare_key`. The name rule reads
through it via a new `_name_key` rather than growing its own `.casefold()` --
a security primitive with two independently-normalized comparison paths is how
they drift. `test_the_name_rule_uses_the_modules_one_folding_rule` pins that by
mutation: with folding removed from `_compare_key` and nowhere else, a
case-variant credential name must stop being refused. For the record, 19800
did NOT already refuse any of the six -- re-measured at `d4f3f9776`, all six
still returned their body through `fs_read`; 19800 fixed how the paths it
already knew were compared, not which paths it knew.

**Evidence.** Born-red at `origin/dev` (`d4f3f9776`): 12 failures -- the six
parametrized shapes, the rule-attribution test, case-insensitivity, the
both-families test, the family-1 listing test, `fs_grep`, and the folding
composition pin. The over-refusal bound
(`test_the_name_rule_does_not_refuse_near_misses_or_containers`) passes at base
AND after, deliberately. All pass on this branch. Blast radius checked across
`Tests/Tools/`, `Tests/Agents/`, `Tests/Utils/`, `Tests/Workspaces/`,
`Tests/MCP/`, `Tests/Notes/` git suites and `Tests/UI/` navigation.

**Housekeeping.** The module docstring's "the denylist is an ENUMERATION ...
(TASK-19633, open)" paragraph is replaced by the two-instruments decision and
the asymmetry record. TASK-19551's drift-pin comment in
`Tests/Tools/test_local_tool_sensitive_paths.py` was re-read and rewritten: "an
honest distinction" is now TRUE without its caveat, and the comment says why it
used to carry one.

Modified: `tldw_chatbook/Utils/sensitive_paths.py`,
`Tests/Tools/test_local_tool_sensitive_paths.py`.
