# 028 - Settings owns workspace management; folders are file-tool access roots

Date: 2026-07-26
Status: accepted
Relates to: spec `2026-07-26-settings-workspaces-category-design.md`; supersedes the Stage-5 read-only boundary in `Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md`

## Context

The tldw_chatbook workspace architecture requires a management surface and a coherent enforcement model for folder access. Prior to this decision, workspace state was read-only in Settings, and folder bindings existed only as a data model without enforcement or UI.

## Decision

### (a) Ownership supersession with Console/Library quick actions retained

Settings now owns the complete workspace lifecycle — create, rename, archive/unarchive, set active, and add/remove folder bindings. The Console workspace switcher (Alt+W) and Library quick actions retain their convenience surfaces: the switcher keeps switch/rename/archive, and Library keeps create as in-context actions. This coexistence avoids centralizing every operation in Settings while making Settings the authoritative management home.

### (b) Folder semantics: access roots, read-only default, call-time validation, run-bound resolution

Folders bound to a workspace define file-tool access roots — the boundaries where agent file tools (`read_file`, `list_directory`, `write_file`) may operate. Each folder binding specifies:
- **Access mode:** read-only is the default; write access is per-folder opt-in via an explicit toggle.
- **Existence validation:** folders are validated at call time, not at binding time. A deleted folder drops out of the allowed set immediately and does not block tool operations.
- **Workspace resolution:** the tool catalog injects a roots-provider closure bound to each run's workspace id, so the same tool instance cannot silently retarget if the active workspace changes mid-run. Only when a run has no workspace context does the provider fall back to `get_active_workspace()` at call time.
- **Default workspace:** the built-in Default workspace cannot have folder bindings. It remains tool-less by design, preserving the existing sandbox-only behavior for everyday chats that do not opt into workspace separation.

### (c) Deliberate divergence from Codex prior art — reads are confined too

OpenAI Codex's `workspace-write` sandbox allows reads anywhere and confines only writes (via `workspace + sandbox_workspace_write.writable_roots`). tldw deliberately confines both reads and writes to the sandbox plus bound folders, because the existing sandbox architecture already does so and the app is privacy-first. Do not "fix" this divergence to match Codex's model; the asymmetry is intentional.

### (d) Canonical locators as the external-runtime export surface

Canonical folder locators (resolved paths with symlinks dereferenced) are the roots handed to external agent runtimes — the shape that Codex's app-server takes as `runtimeWorkspaceRoots` at thread/turn start. The future ACP/app-server handoff will consume workspace bindings as-is via this export surface; no separate roots model is needed.

## Consequences

- Settings becomes the single authoritative surface for workspace and folder management, reducing cognitive load and surfacing management actions consistently.
- File-tool access is scoped to workspace-bound folders (plus the global sandbox), enabling privacy-preserving agent isolation without sacrificing usability.
- Run-bound workspace resolution prevents mid-run context switches from silently retargeting agent file operations.
- Folder validation at call time keeps the system robust to filesystem changes without requiring background monitoring.
- The export surface enables future external runtimes (ACP, app-server) to enforce the same workspace semantics.
