# ADR-174: Per-workspace binding exclusions for agent file access

Status: Accepted
Date: 2026-09-20
Related Task: [TASK-32833](../tasks/task-32833%20-%20Workspace-binding-exclusions.md)
Design: [Workspace binding exclusions](../../Docs/superpowers/specs/2026-09-20-workspace-binding-exclusions-design.md)
Related: ADR-028, ADR-069, ADR-079, ADR-101, ADR-102

## Decision

A workspace folder binding may carry user-managed exclusions: exact
binding-relative paths (files or directories, non-existent allowed, capped
at 200, stored in `workspace_runtime_bindings.metadata_json["exclusions"]`)
that are fully invisible to agent file tools — reads, writes, edits,
patches, stat, and enumeration across BOTH agent-facing file-tool families
(the local provider with its one-shot pinned worker, and the builtin file
tools), plus Git pathspec exclusion and project-instruction activation.

Enforcement is a single injection point: user exclusions fold into the
per-call `SensitivePathContext` (`merge_sensitive_context`), so the
existing sensitive-path machinery (choke point, worker serialization,
pathspec rendering, directory-chain guard) enforces them with refusal copy
byte-identical to the system denylist's. Exclusions are therefore enforced
in BOTH agent-facing file-tool families and in project-instruction
discovery, and the model cannot distinguish user exclusions or learn that
the paths exist. No worker-protocol change.

Run semantics follow ADR-102's discipline, as shipped. Admission snapshots
the exclusion set; a per-root provider live-reads the registry on every
tool call and keeps a high-water mark, wired into both the one-shot
executor and the local provider's preflight, so mid-run additions refuse
the next tool call and removals take effect only for new runs. Registry
read failures degrade to the last-known effective set. The builtin
file-tool family merges the effective set per invocation through the same
fold. Project-instruction activation freezes the excluded set at ledger
construction (ADR-069 snapshot discipline): already-activated instructions
are not retracted mid-run, and candidates under an exclusion are skipped
at activation. In the files modal, unexclude removes the deepest covering
layer first and reports remaining parent coverage truthfully instead of
claiming inclusion.

The user's own surfaces (Console file inspector, Settings) remain
direct-user authority (ADR-079): excluded entries stay visible, badged,
and un-excludable there.

## Alternatives considered

- Permission-store deny rules: cannot hide entries from listings; the
  pinned worker never consults the store.
- Negative bindings in root resolution: binding validation forbids
  nesting/overlap; would need a second implementation seam.
- Gitignore-style patterns: a pattern engine in the security path; exact
  literals cover the v1 need.

## Links

- [ADR-028](028-settings-workspaces-category-and-folder-roots.md)
- [ADR-069](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-079](079-workspace-file-inspector-direct-user-authority-and-save-publication.md)
- [ADR-101](101-one-shot-pinned-workspace-tool-execution.md)
- [ADR-102](102-console-run-admitted-local-path-authority.md)
