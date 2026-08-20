# ADR-069: Keep Console project-instruction state local and prepare tool context before security review

Status: Accepted
Date: 2026-08-20
Related Task: Implementation tasks will be created from the approved design during planning.
Supersedes: ADR-068

## Decision

Chatbook Console agents will support repository-authored `AGENTS.md` and
`AGENTS.override.md` as untrusted, ephemeral user-level project context within
one selected workspace folder binding. The selected binding root is both the
authority boundary and working directory in v1. Its effective root instruction
file loads at dispatch start; narrower files activate lazily before local
filesystem/git/patch operations target their scopes.

For enabled sessions, the run's single-root `LocalToolProvider` is composed at
that selected binding root rather than the global `[console] workspace_root`
fallback. The built-in `read_file`, `list_directory`, and `write_file` tools
also participate in path-aware activation when they target the selected root;
their existing authorization for other workspace bindings remains intact but
produces an outside-instruction-scope warning. Disabled and legacy-disabled
sessions retain current provider-root behavior.

Binding access is not widened by provider composition. A read-only selected
binding loads instructions and exposes read-only local filesystem/git tools,
but omits `fs_write`, `fs_edit`, and `fs_patch` for that run. V1 omits patching
entirely on a read-only binding instead of introducing a dry-run-only variant.

The four durable control fields—enabled state, binding ID, canonical-locator
fingerprint, and opaque first-use-notice key—live in a versioned,
**local-only** `conversations.console_project_context_json` column. A dedicated
write path changes neither conversation version/sync timestamps nor
`sync_log`; the column is absent from every conversation sync trigger and
payload. Its JSON envelope contains one schema-version discriminator plus only
those four control fields; it stores no raw locator or instruction-derived
data. Null, invalid, forward-versioned, or legacy screen state is disabled.
Temporary sessions remain in live/screen state and write the column only if
they become durable conversations.

Chatbook currently has no inbound conversation-sync/apply service;
`DB.Sync_Client.ClientSyncEngine` is media-only. V1 therefore excludes this
column from outbound conversation triggers and payloads and requires all
existing conversation mutation and Chatbook import paths to preserve it.
Ordinary updates and soft delete/restore change explicit synchronized columns.
The current importer does not overwrite an existing conversation: `SKIP`
leaves it untouched, while every non-skip resolution creates a new row whose
local state starts null. Any future inbound conversation-sync/apply service
must use a synchronized-column allowlist and preserve this local column through
create/update/delete/undelete/replay and conflict resolution; that future
service is outside this decision's v1 implementation scope.

The notice key is a domain-separated SHA-256 value derived from the locator
fingerprint plus the resolved provider destination identity (provider adapter
and canonical endpoint identity, excluding credentials). A provider or custom
endpoint change therefore requires renewed consent; a model-only change at the
same destination does not. No raw endpoint is stored. The notice may show only
a sanitized destination label, removing URL credentials and paths.

At dispatch, Chatbook resolves the binding ID and compares the SHA-256
fingerprint of its current canonical locator with the stored selection
fingerprint. Missing, unauthorized, unavailable, or retargeted bindings require
explicit re-selection. Binding identity alone is insufficient because the
registry permits an existing binding ID's locator to be updated.

`LoopDeps` gains an optional typed `prepare_tool_calls` hook, threaded through
`AgentService` and invoked immediately before the existing `review_tool_calls`
hook. Preparation owns path-aware instruction discovery and may return
a typed proceed/retry status plus tagged ephemeral context. `AgentRuntime`, not
the hook, synthesizes one protocol-safe deferral result per original call so
call IDs, ordering, and cardinality retain one owner. The existing string-map
review hook remains unchanged as the permission and change-review boundary and
runs only after preparation proceeds. This keeps optional guidance failures
separate from security verdicts.

Preflight resolves each call through `ToolCatalogRegistry`'s same cached,
first-registrant-wins owner mapping used by dispatch, then asks only that owner
for path targets. Preparation exceptions emit a content-free ephemeral UI
warning and code-only log, never exception text, traceback, `AgentStep`, or
durable state, before proceeding to unchanged security review.

The parent and subagents share one run-local activation ledger and byte budget,
while each model conversation tracks what it has received. Automatic
instruction bodies never enter tool results, durable captures, summaries,
agent steps, or logs. Explicit file reads and model-authored quotations retain
ordinary persistence semantics.

## Context

ADR-068 established the untrusted project-context boundary, Codex filename
precedence, lazy nested activation, and nonpersistence requirement. A later
readiness audit found three implementation-breaking assumptions:

1. `conversations.metadata` is included in `conversations_sync_*` triggers, so
   storing local binding IDs there would synchronize meaningless device-local
   state and could block sends on another device.
2. `WorkspaceRegistryService` can update the locator under an existing binding
   ID, so persisting only the ID cannot enforce the promised no-silent-retarget
   behavior.
3. `review_tool_calls` already owns permission review, provider stamps, and
   change-review gating. Extending its string verdicts with ephemeral content
   would mix optional repository guidance with security policy and its distinct
   failure posture.

The audit also confirmed that current `/rewind` compaction happens before the
agent bridge constructs run-local messages. Automatic riders never reach its
summarizer, so adding mid-agent compaction/rebuild machinery would solve a
nonexistent problem.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Store control state in `conversations.metadata` | That column synchronizes; local binding identity would leak across devices and restore without local authority. |
| Persist only the binding ID | The same ID can be retargeted, making silent folder changes undetectable. |
| Persist the canonical raw path | Unnecessary local-path disclosure; a fingerprint is enough to detect retargeting. |
| Keep a relative working-directory field | The selected folder is already the tool-relative working directory, and lazy activation covers descendants; no current UX or runtime needs a second cwd. |
| Extend `review_tool_calls` with a union return type | Couples optional context preparation to permission and change-review contracts and their failure policies. |
| Add mid-agent compaction support | No such runtime compaction exists; `/rewind` already operates before ephemeral riders are added. |
| Add an inbound conversation-sync service solely for this feature | No such conversation service exists today; outbound exclusion and preservation in real mutation/import paths satisfy v1 without inventing unrelated sync architecture. |
| Infer `fs_glob` scope from a static pattern prefix | The tool has no search-root argument, and pattern inference would create a second ambiguous path grammar. |

## Consequences

- A schema migration adds one local-only JSON column and must be numbered from
  the actual schema head at implementation time. Tests must prove writes create
  no sync-log entry or conversation version bump.
- New sessions explicitly enable project instructions; a null/missing/invalid
  local state keeps legacy restored conversations disabled.
- Binding selection captures both ID and locator fingerprint. Retargeting,
  removal, or mismatch blocks silent use and asks the user to select again.
- The first-use notice appears before the first provider request even when no
  root instruction file exists, because deeper guidance may activate later;
  changing provider destinations requires renewed consent.
- Root startup discovery is O(1); nested discovery is O(depth). `fs_patch`
  preflights every parsed create/modify target, while `fs_glob` and `fs_grep`
  use only the binding root under their current schemas.
- Candidate bodies are never read beyond the configured per-source cap; an
  oversized override is omitted and still suppresses same-directory fallback.
- POSIX uses no-follow plus descriptor identity checks. Windows rejects
  symlink/junction/reparse ancestors via `st_file_attributes` and uses the same
  pre/post descriptor identity checks. Missing platform primitives fail only
  the source closed with a content-free warning.
- Both local file-tool families participate: the single-root `fs_*`/`git_*`
  provider uses the selected root, while built-in file tools preserve their
  multi-binding authorization and warn rather than loading a second hierarchy.
- Opaque process and skill-script tools receive startup guidance but cannot
  trigger guessed nested scopes from free-form command text.
- A selected read-only binding remains read-only; provider composition cannot
  bypass workspace access metadata.
- The runtime adds a preparation phase but leaves existing security-review
  call sites and string verdicts unchanged.
- Existing conversation updates, soft delete/restore, and Chatbook import
  conflict handling preserve device-local control state. There is no current
  inbound conversation apply owner; any future owner inherits the same
  preservation invariant and must prove it at its apply boundary.
- Current `/rewind` behavior needs a boundary regression test, not new
  compaction machinery. Any future mid-agent compactor must make a new design
  decision that preserves automatic-context ephemerality.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-20-agents-md-support-design.md)
- [Superseded ADR-068](068-console-project-instruction-context-boundary.md)
- [ADR-005: Console workspace/server readiness](005-console-workspace-server-readiness.md)
- [ADR-028: Settings workspaces and folder roots](028-settings-workspaces-category-and-folder-roots.md)
- [ADR-032: Local agent tool permission boundary](032-local-agent-tool-permission-boundary.md)
- [Codex AGENTS.md guide](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [Claude Code memory guide](https://code.claude.com/docs/en/memory)
