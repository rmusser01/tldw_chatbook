# Codex CLI source reference for project context

Inspected: 2026-10-20, America/Los_Angeles
Repository: `openai/codex`
Revision: `c7234c1174b27d9c55199fba0d546dc27d1b0415` (main at inspection)
Method: GitHub commit/tree APIs followed by raw source reads at that revision.
No downloaded source was executed or copied into the application.

This reference informs the approved [project-context design](2026-10-20-workspace-aware-project-context-design.md)
and its three implementation plans. The findings below come from the CLI source,
not a product-documentation description. They describe the inspected paths,
not an exhaustive claim about every Codex distribution, extension, or model.

## Findings and application

| Inspected source | Observed behavior | Application to Chatbook |
| --- | --- | --- |
| [Fallback model prompt](https://github.com/openai/codex/blob/c7234c1174b27d9c55199fba0d546dc27d1b0415/codex-rs/models-manager/prompt.md#L260-L266) | The fallback prompt recommends `rg` for content search and `rg --files` for file discovery. | Guide the agent toward the offered `fs_list`, `fs_glob`, `fs_grep`, and `fs_read` tools. The instruction is capability-dependent; do not add a shell requirement. |
| [Repository instruction loader](https://github.com/openai/codex/blob/c7234c1174b27d9c55199fba0d546dc27d1b0415/codex-rs/core/src/agents_md.rs#L1-L114) | Discovery follows project-root markers, loads ancestor guidance under a shared byte budget, observes trust and executor permissions, and iterates the turn's environments. | Preserve the existing bounded project-instruction system and owning-run authority. Folder access and instruction discovery remain separate from loading project content. |
| [Instruction refresh manager](https://github.com/openai/codex/blob/c7234c1174b27d9c55199fba0d546dc27d1b0415/codex-rs/core/src/agents_md_manager.rs#L68-L145) | Selection/trust changes invalidate cached repository instructions before asynchronous rediscovery; the source explicitly addresses tightened read permission. | Revoke index query/publication eligibility before refresh or cleanup. A cancelled or failed refresh must not retain inaccessible old content. |
| [User instruction fragment](https://github.com/openai/codex/blob/c7234c1174b27d9c55199fba0d546dc27d1b0415/codex-rs/core/src/context/user_instructions.rs#L8-L42) | Repository instruction text is a typed user-role fragment with an instruction-specific origin. | Keep indexed source text as evidence; do not promote it into system guidance or bypass the instruction activation ledger. |
| [Shell execution handler](https://github.com/openai/codex/blob/c7234c1174b27d9c55199fba0d546dc27d1b0415/codex-rs/core/src/tools/handlers/unified_exec/exec_command.rs#L180-L198) | The working directory resolves from the selected turn environment, with explicit workdir handling. | Resolve folder aliases from the owning run. Switching the visible workspace must not retarget a running query or job. |
| [File-search options and session](https://github.com/openai/codex/blob/c7234c1174b27d9c55199fba0d546dc27d1b0415/codex-rs/file-search/src/lib.rs#L121-L175) | Fuzzy path search has configurable result/thread limits, ignore handling, query updates, and session shutdown. Defaults include 20 results and two threads. | Bound result payloads independently of corpus size. Reuse job/query state and cancellation; avoid rescanning synchronously for each keystroke or model turn. Chatbook keeps its approved limits of 10 results/16 KiB. |
| [Path walker](https://github.com/openai/codex/blob/c7234c1174b27d9c55199fba0d546dc27d1b0415/codex-rs/file-search/src/lib.rs#L420-L520) | A parallel ignore-aware walker injects relative path entries into a fuzzy matcher, checks cancellation, and signals traversal completion. It follows symlinks and uses git-aware ignore semantics. | Distinguish discovered paths from indexed text and completed coverage from partial results. Keep Chatbook's stricter no-symlink indexing policy and exact exclusions; do not transplant this traversal policy. |
| [TUI file-search manager](https://github.com/openai/codex/blob/c7234c1174b27d9c55199fba0d546dc27d1b0415/codex-rs/tui/src/file_search.rs#L1-L122) | The `@` picker owns a search session, drops it when the search root changes, and uses a session token when publishing results. | Fence status callbacks against replaced bindings, jobs, and index epochs. A finished background callback cannot establish current ownership. |

## What this establishes

The inspected CLI separates agent content exploration, repository instructions,
turn environment/permissions, and interactive fuzzy path discovery. Its `@`
file-search subsystem indexes path strings in a session; this is not evidence
of a persistent semantic index over file bodies. A bounded result list also
does not establish that the in-memory path catalog is bounded independently of
tree size.

Our optional durable keyword/content and semantic caches therefore remain
Chatbook-specific additions justified by the user's large-tree requirement.
Do not claim feature parity or copy Codex's memory, symlink, ignore, trust, or
approval behavior into Chatbook. The approved binding and Library contracts win.

## Checks carried into the plans

- Guidance works with no index and an unrelated Library bundle present.
- Instruction-like content inside a repository file stays untrusted evidence.
- Delayed callbacks from replaced roots/jobs do not publish into the new view.
- Revocation remains effective when refresh fails or is cancelled.
- Discovery covers deep trees and a very wide directory with bounded queues,
  independently of the model-visible result limit.

No Claude Code implementation-source claims are made here; the implementation
reference requested for this plan is the pinned Codex CLI code above.
