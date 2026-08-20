# Console AGENTS.md Project Instructions — Design

- **Date:** 2026-08-20
- **Status:** Owner-approved amendments; pending post-amendment review
- **Scope:** Console agent runs only
- **Decision record:** Proposed [ADR-069](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md), which will supersede ADR-068 if accepted

## 1. Problem

Chatbook's Console agents can work inside workspace-bound folders, but they do
not read repository-authored operating guidance. Projects that already explain
their conventions in `AGENTS.md` therefore behave differently in Chatbook than
they do in Codex, and agents can miss narrower instructions when they move into
a subdirectory.

The feature must add that compatibility without turning repository text into
privileged policy, leaking instruction contents into Chatbook persistence, or
weakening the existing workspace and tool-approval boundaries.

## 2. Ecosystem model and chosen direction

Chatbook will use a deliberate hybrid of the two established behaviors:

- [Codex](https://learn.chatgpt.com/docs/agent-configuration/agents-md) supplies
  the filename and precedence model: `AGENTS.override.md` before `AGENTS.md`,
  one effective file per directory, and broad-to-specific composition.
- [Claude Code](https://code.claude.com/docs/en/memory) supplies the useful
  path-sensitive behavior: nested project guidance becomes relevant when the
  agent works in that part of the tree. Claude Code does not natively use
  `AGENTS.md`; it uses `CLAUDE.md`, with `AGENTS.md` available only through an
  import or symlink.

Chatbook differs from both where its own trust model requires it:

- The selected workspace folder binding, not a Git root or process cwd, is the
  discovery and authority boundary.
- Repository instructions are untrusted user-level project context. They are
  never system policy and cannot override sandboxing, path confinement,
  approvals, provider safety, or Chatbook's operating prompt.
- Instruction contents are ephemeral provider context and never durable
  conversation or agent-run data.
- Symlinked instruction files and symlinked directory traversal are refused in
  v1.

The rejected alternatives were:

1. **Codex parity only:** load root-to-cwd once and ignore deeper scopes. This
   is simpler but fails the requested path-aware behavior.
2. **Prompt-only convention:** tell the model to search for `AGENTS.md`. This
   makes correctness depend on model initiative, cannot guarantee discovery
   before a tool action, and is difficult to audit.
3. **Universal interception of every tool provider:** this would overreach
   into remote MCP semantics. V1 limits path awareness to Chatbook's local
   filesystem/git/patch provider contract.

## 3. Goals

1. Give each Console agent run one explicit working folder selected from its
   authorized workspace bindings.
2. Resolve `AGENTS.override.md` / `AGENTS.md` at that working-folder root, then
   activate narrower instructions before local operations in deeper paths.
3. Apply the same rules to parent agents and subagents, including concurrent
   subagents.
4. Keep automatically loaded instruction contents out of the conversation
   transcript, run database, agent steps, persisted logs, diagnostic logs,
   compaction summaries, and durable outbound exchange captures.
5. Make the feature visible and controllable through a per-session toggle,
   folder selection, compact rail state, pre-send notice, and Context
   diagnostics.
6. Continue safely, with explicit warnings, when optional project guidance is
   missing, unreadable, invalid, stale, or over budget.

## 4. Non-goals

- Global or personal instruction files outside the selected workspace folder.
- Codex fallback filenames configured through `project_doc_fallback_filenames`.
- `CLAUDE.md`, `.cursorrules`, or other repository-memory conventions.
- Project-instruction discovery for ordinary chat, non-Console surfaces, or
  non-agent provider calls.
- Interpreting MCP or other remote tool paths as local workspace paths.
- Inferring nested scopes from opaque script/process execution, including
  `run_skill_script`; such tools do not structurally declare every path they
  may touch and receive only the startup guidance already in their model chain.
- A dedicated source-file viewer/editor. The Context surface reports metadata
  and the existing file tools remain the way to open files.
- Enforcement of repository prose as a security policy. It is guidance to the
  model; existing runtime controls remain enforcement.
- Persisting automatically loaded instruction bodies, digests, absolute paths,
  timestamps, or a history of prior instruction versions.
- Data-loss-prevention filtering of ordinary model output or explicit file-tool
  results. If an agent deliberately calls `fs_read` on `AGENTS.md`, or the model
  quotes project guidance in its answer, that model-authored/tool-authored data
  follows the existing persistence rules. The ephemerality guarantee applies
  to Chatbook's automatic instruction-loading channel.

## 5. Session working context and rollout defaults

`ConsoleChatSession` owns project-instruction state because it is conversation
working context, not a provider preference. The session adds:

- `project_instructions_enabled: bool`
- `working_folder_binding_id: str | None`
- `working_folder_locator_fingerprint: str | None`, the SHA-256 fingerprint of
  the canonical locator when the binding was selected
- `project_instruction_notice_key: str | None`, an opaque SHA-256 key recording
  consent for the canonical working folder **and resolved provider destination**

The selected binding root is the working directory in v1. There is no second
relative-cwd setting or UI: the selected-root `fs_*`/`git_*` paths are relative
to that binding root, while lazy activation covers descendants. A future
independent working-directory feature can add that distinction if users need
it.

For an enabled, validated session, `_compose_local_provider` constructs the
run's `LocalToolProvider` with that selected binding root instead of the global
`[console] workspace_root` fallback. This makes `fs_*` and `git_*` paths and
project-instruction paths share one canonical origin. When project instructions
are disabled or legacy-disabled, local-provider composition remains exactly as
it works today, including the configured/cwd fallback.

Binding access remains authoritative. A read-only selected binding still
supports instruction loading and the read-only `fs_*`/`git_*` operations, but
the run does not advertise `fs_write`, `fs_edit`, or `fs_patch`. Selecting a
binding must never turn its workspace access metadata into a broader writable
root. V1 omits `fs_patch` entirely for read-only bindings rather than adding a
special dry-run-only schema variant.

These fields round-trip through Console screen-state restoration and a new
local-only `conversations.console_project_context_json` column. The versioned
JSON envelope stores one schema-version discriminator plus only the four
control fields above; it never stores a raw locator, instruction path,
instruction digest, or instruction body. A dedicated local read/write method
updates the column without changing conversation sync metadata, and the column
is omitted from all `conversations_sync_*` trigger conditions and payloads.
This requires a schema migration whose version must be allocated against the
actual schema head at implementation time. Temporary or ephemeral sessions
keep the same fields only in live/screen state; if a session becomes a durable
conversation, its current control state is then written through the local-only
column.

Chatbook has no inbound conversation-sync/apply service in v1;
`DB.Sync_Client.ClientSyncEngine` is media-only and is not part of this design.
The implemented boundary is therefore precise: the column is omitted from
outbound `conversations_sync_*` trigger conditions and payloads, and every
existing conversation mutation path updates explicit synchronized columns
without replacing or clearing `console_project_context_json`. This includes
ordinary updates, soft delete/restore, and Chatbook import conflict handling.
The current importer creates a new conversation for non-skip resolutions
(including its current `REPLACE` behavior), so it leaves an existing row's
local state untouched; every genuinely new imported row starts null/legacy-
disabled until Console writes its own state. A future inbound conversation-
sync/apply service must preserve this local-only column through create,
update, delete, undelete, replay, and conflict resolution, but adding that
service is outside v1. The preservation invariant applies across restart and
must be enforced at any such future apply boundary rather than inferred only
from outbound triggers.

At every agent dispatch, Chatbook resolves `working_folder_binding_id` through
the workspace registry, revalidates the canonical locator, recomputes its
fingerprint, and compares it with the stored selection fingerprint. A removed,
retargeted, unavailable, mismatched, or unauthorized binding enters recovery;
Chatbook never silently follows the same binding ID to a different folder or
selects a replacement binding.

Defaults are migration-safe:

- Sessions created after this feature ships explicitly set project
  instructions enabled.
- Restored conversations whose local-only project-context column is null,
  missing, malformed, or forward-versioned—and restored legacy screen-state
  payloads with no project-context fields—are treated as legacy and default
  disabled. Merely upgrading, restoring, or synchronizing Chatbook cannot block
  an old send or transmit newly discovered repository text.
- When an enabled session has exactly one eligible folder binding—belonging to
  the session workspace and currently authorized, available, canonical, and
  non-symlinked—it is selected automatically. With several eligible bindings
  and no selection, the agent send is held at a chooser. With none, the user
  may disable project instructions or cancel the send; stale/ineligible entries
  may be shown only as disabled recovery information.
- Enabling the feature later uses the same sole-binding/chooser behavior.
- Selecting or changing a binding captures its current locator fingerprint and
  clears the notice acknowledgement.

Other folders bound to the same workspace remain available through the
existing built-in `read_file`, `list_directory`, and `write_file` authorization
rules. They are outside the selected project's instruction scope and generate
one explicit warning per run when targeted. They do not acquire their own
instruction hierarchy in v1. The single-root `fs_*`/`git_*` provider is
confined to the selected binding for enabled sessions.

## 6. Core types and ownership

The implementation adds one focused module,
`tldw_chatbook/Agents/project_instructions.py`, containing pure resolution,
selection, and context-building logic.

### 6.1 Immutable source and turn snapshot

`InstructionSource` is immutable and contains only what the resolver and
context builder need:

- canonical source path, retained in memory only
- workspace-relative display path
- workspace-relative scope directory
- kind (`override` or `standard`)
- decoded content
- raw byte count
- content digest, retained in memory only for deduplication and delivery

`InstructionSnapshot` is immutable for one user-initiated Console agent
dispatch. In this design, submit, retry, regenerate, and continue each start a
fresh dispatch and therefore a fresh snapshot. It contains:

- selected binding identity, validated canonical root, and matching locator
  fingerprint
- dispatch wall-clock cutoff in nanoseconds, used only for filesystem-mtime
  staleness checks (performance timing uses a separate monotonic clock)
- the effective source selected at the binding root for the startup budget
- byte/token budget outcomes and content-free warnings

Once captured, base instruction content is pinned for the dispatch. File edits
apply to the next user-initiated dispatch, not midway through the current one.

### 6.2 Mutable activation ledger

`InstructionActivationLedger` is a run-local, concurrency-safe object shared
by the parent and all subagents. It contains:

- activated nested sources and a monotonic activation revision
- the remaining global nested byte budget for the dispatch
- warning/deduplication state
- per-model-chain delivery cursors identifying which source revisions that
  parent or subagent has actually received
- terminal requirement outcomes, globally or per model chain as appropriate:
  `delivered`, `omitted_byte_budget`, `omitted_token_budget`, `stale`,
  `invalid`, or `resolution_failed`

The delivery cursor is essential: one subagent loading a source does not mean
another model conversation has seen it. A tool batch may proceed only when
every instruction requirement for its paths is either delivered to that model
chain or has a terminal no-content outcome that the chain has received as an
ephemeral warning. New subagents receive the currently active snapshot in
their spawn context and begin with the matching delivery cursor. If another
chain activates guidance concurrently, each affected chain must receive its
own ephemeral context update before executing under it. A globally selected
source can still have a chain-specific `omitted_token_budget` outcome when that
chain's provider payload has less headroom.

The ledger and snapshot die with the dispatch. Neither is serialized.

### 6.3 Path-aware provider contract

Path awareness is an optional structural protocol on tool providers, named
`PathAwareToolProvider`. Its operation maps a validated tool call to zero or
more workspace-relative path targets and identifies whether each target is an
exact path, directory/search root, or outside the selected instruction root.
`LocalToolProvider` implements it for its `fs_*`/`git_*` tools, and
`BuiltinToolProvider` implements it for `read_file`, `list_directory`, and
`write_file`. Existing providers and tools that do not implement it retain
current behavior and cause no project-instruction discovery.

The mapping must reuse the local tool provider's own argument parsing and path
validation; the instruction layer must not maintain a second interpretation of
tool arguments. MCP is explicitly excluded because its paths can be remote or
provider-defined.

Ownership resolution is registry-owned. For each model call,
`ToolCatalogRegistry` resolves the LLM-facing name to the same cached
first-registrant-wins `(tool_id, provider)` owner used by `invoke_by_name`.
Preflight asks only that resolved owner for path targets; it never scans every
provider or consults a shadowed same-name tool. Runtime-special calls with no
catalog owner and owners without `PathAwareToolProvider` yield no targets. The
registry exposes one internal owner-resolution operation so preflight and
dispatch cannot grow separate collision rules.

The contract is intentionally limited to tools whose complete path targets can
be derived before execution. Shell-like, process, skill-script, network, todo,
and spawn tools return no path targets; the feature never guesses from command
strings, prompts, or free-form arguments.

## 7. Resolution rules

### 7.1 Authority and traversal

The canonical selected binding locator is the maximum boundary. The resolver
never ascends above it and never substitutes a Git root. Because the selected
binding root is the v1 working directory, startup examines only that directory.

For every directory in a chain:

1. Examine `AGENTS.override.md`.
2. If it is a regular, non-symlinked, non-empty file, select it and do not
   examine `AGENTS.md` for that directory.
3. If the override is absent or whitespace-only, examine `AGENTS.md`.
4. If the selected candidate is unreadable, unstable during the read, invalid
   UTF-8, or a symlink, warn and skip that directory. An invalid override does
   not fall back to `AGENTS.md`; otherwise an attacker or broken override could
   silently expose instructions the override was meant to replace.

Candidate reads are bounded. Before decoding or testing for whitespace, the
resolver checks descriptor metadata against the applicable configured
per-source cap (startup cap for the root source, nested cap for a nested
source). A larger candidate is selected but receives an `omitted_byte_budget`
outcome without reading its body. In particular, an oversized override is not
treated as empty and does not fall back to same-directory `AGENTS.md`.

Files decode as strict UTF-8 with an optional UTF-8 BOM. Reads use standard
library descriptor identity checks on every platform. On POSIX, the resolver
uses `os.open` with `O_NOFOLLOW` where available, then compares pre-open
`lstat`, descriptor `fstat`, bounded-read, post-read `fstat`, and post-read
`lstat` identity/size/mtime. It records every ancestor directory's `lstat`
identity before opening and rechecks the chain after reading. On Windows, it
also rejects any file or ancestor whose `st_file_attributes` contains
`stat.FILE_ATTRIBUTE_REPARSE_POINT`, then applies the same pre/post identity
checks; an open that followed a raced-in reparse target cannot match the
pre-open file/ancestor identities. Every read is capped at the configured limit
plus one byte so growth cannot allocate unbounded memory. If a platform cannot
expose the required identity or reparse metadata, that source fails closed with
a content-free platform warning rather than disabling the whole run. Directory
symlinks and reparse points are not traversed for instruction discovery. This
refusal does not independently change whether the existing local tool boundary
allows the requested operation; it only prevents guidance from being inferred
through the linked route and produces a warning.

Selected files are composed in broad-to-specific order. A more specific file
supersedes conflicting broader guidance for paths in its scope. Sources from
different sibling directories are labeled as separate scopes and never imply
that one sibling overrides another.

### 7.2 Lazy nested activation

The startup snapshot resolves only the binding-root source, so cost is O(1),
not a recursive repository walk. A path-aware tool batch computes the
additional root-to-target directory chains required by its targets:

- `fs_read`, `fs_write`, and `fs_edit` use the exact target's parent chain.
- Built-in `read_file` and `write_file` use the validated exact target's parent
  chain; built-in `list_directory` uses the validated directory chain. If that
  target belongs to another authorized workspace binding, the call remains
  eligible under existing authorization but receives the outside-scope warning
  instead of activating another hierarchy.
- `fs_patch` reuses the patch tool's parser, extracts every supported `+++`
  create/modify target, and activates the union of their parent chains before
  either a real or `dry_run` invocation. Invalid, delete, or rename forms remain
  the patch tool's existing errors; the instruction layer does not invent a
  second patch grammar.
- `fs_list` uses the listed directory chain.
- `fs_glob` and `fs_grep` activate only the selected binding root because their
  current schemas do not accept a search-root argument. A static prefix in a
  glob pattern is not inferred as a scope. Matching or mentioning a deep file
  does not activate every descendant instruction file.
- `git_branches`, unfiltered `git_diff`, and unfiltered `git_log` activate only
  the discovered repository-root chain.
- `git_status`, including `git_status(path)`, activates only the discovered
  repository-root chain because `path` selects which repository to inspect but
  the command reports status for that whole repository.
- Path-filtered `git_diff(path)` and `git_log(path)` activate the repository
  root through the validated target scope: the target itself when it is a
  directory, otherwise its lexical parent (which also covers deleted paths).
  `commit_range`, `staged`, and `stat` do not change that path scope.
- `git_blame(path)` activates the repository root through the blamed file's
  parent chain.
- A later concrete action on a matched file activates that file's directory
  chain.

When a nested source is first encountered, the resolver performs a stable
read and pins the content for the dispatch. A candidate whose metadata shows
it was created or changed after dispatch start is marked stale, omitted, and
deferred to the next dispatch. An already pinned source remains in force even
if the file is subsequently edited or deleted.

## 8. Provider-context construction

Project instructions are a clearly labeled user-level context rider, separate
from `compose_agent_system_prompt`. The wrapper states that the files are
untrusted project guidance, gives only workspace-relative paths and scopes,
and reminds the model that system instructions and runtime controls remain
authoritative. It excludes absolute paths, timestamps, hashes, and other host
metadata.

Every automatic rider and nested update carries an internal ephemeral-origin
tag. That tag is not sent to the model, but every persistence, diagnostic,
and **durable historical exchange-capture** boundary uses it to omit the body.
A future or concurrently developed historical payload inspector must replace
these bodies with a content-free marker such as
`[ephemeral project instructions omitted]` and may record only relative source
metadata. This ADR-backed exception takes priority over durable exact
historical payload capture. It does not redact the existing user-invoked,
nonpersistent Next Send preview, which may show the exact rider about to be
sent and is discarded when the Context surface closes.

One pure context-rider builder handles text, multimodal submit, retry,
regenerate, and continue. It produces provider-safe message ordering and a
synthetic user row where a run path has no natural new user row. It must not
leave a user context message between an assistant `tool_calls` row and the
required tool-result rows.

Before the first provider request in a dispatch, the run-local message copy
receives the startup rider. The session transcript is not modified. Context
preview uses the same pure builder against a copy; previewing must not activate
sources, consume budgets, acknowledge notices, or mutate the live ledger.

The first time an enabled session uses a selected binding with a resolved
provider destination, Chatbook resolves the immutable startup snapshot, then
displays a pre-dispatch notice derived from that exact snapshot before the
first provider request. The notice names the provider, shows a sanitized
destination label (scheme/host/port for custom endpoints, with credentials and
paths removed), names any binding-root source, states that its content will be
sent, and explains that deeper
`AGENTS.md` files may be sent later when local tools target their scopes. It is
shown even when no binding-root file exists, so a nested-only repository cannot
transmit guidance before consent. It contains no file bodies. Proceed reuses
the captured snapshot without rereading; cancel or disable discards it.
Acceptance stores a domain-separated SHA-256 key derived
from the locator fingerprint and the resolved provider destination identity
(provider adapter plus canonical endpoint identity, without credentials).
Changing or retargeting the binding, switching provider destinations, or
changing a custom endpoint causes a new notice; changing only the model at the
same destination does not. The raw endpoint is never persisted in project
context state.

### 8.1 Existing `/rewind` compaction boundary

Chatbook's current `/rewind` summary is created from the stored conversation
before `run_reply` constructs its run-local message copy. Automatic project
riders are never part of that stored input, and the startup rider is inserted
only after the controller has applied the persisted summary. Therefore v1
needs no rider filtering, delivery-cursor reset, or post-compaction rebuild.

Mid-agent automatic compaction does not exist today and is not added by this
feature. If such a mechanism is introduced later, it must honor ADR-069's
ephemeral automatic-context boundary. Explicit file-tool results and
model-authored quotations continue to follow ordinary `/rewind` semantics.

## 9. Atomic tool-batch preflight

`LoopDeps` gains a separate optional `prepare_tool_calls` hook, threaded through
`AgentService` and invoked immediately before the existing `review_tool_calls`
hook. Its contract is
`Callable[[list[ToolCall]], ToolBatchPreparation] | None`, where the frozen
result carries a status plus zero or more internally tagged ephemeral context
rows. The only statuses are:

- `proceed`
- `retry_with_context`

The preparation hook owns optional project-context discovery only. The
existing string-map `review_tool_calls` contract remains unchanged as the
permission/change-review boundary, including its existing fail-closed policy
where applicable. A preparation exception follows the feature's explicit
fail-open-with-warning policy, produces `proceed`, and cannot silently change
an approval verdict. Binding/setup recovery happens before runtime construction
and therefore needs no third preparation outcome.

`LoopDeps` also gains a separate optional `on_ephemeral_runtime_warning`
callback for nonpersistent UI/runtime warnings. If preparation raises,
`AgentRuntime` emits only the enum-like code
`project_instruction_preparation_failed` plus tool names/count through this
callback, logs only the same sanitized code/metadata without exception text or
traceback, and continues with `proceed` into the unchanged security review.
This callback is not `on_step`: it cannot create an `AgentStep`, transcript
row, run-log record, or database write. Callback failure is swallowed after a
code-only log entry.

`proceed` carries no context rows. `retry_with_context` carries only the
ephemeral instruction/warning rows; it never carries review verdicts or tool
results. `AgentRuntime` itself synthesizes the fixed deferral result for every
original call, which keeps call IDs, ordering, cardinality, and persistence
behavior under one owner.

Preparation occurs before permission prompts and before any tool in the batch
is dispatched. It asks path-aware providers for every target, resolves the
union of required instruction chains, and acquires the activation ledger's
mutation guard. The complete batch is deferred if its calling model chain has
not received any required source. No earlier call in the same batch may
execute, and `review_tool_calls` is not invoked for the discarded batch.

`retry_with_context` produces two deliberately separate channels:

1. **Runtime-generated persistable protocol stubs** for every deferred tool
   call, such as
   “Deferred because project instructions were loaded; reconsider and retry.”
   These satisfy provider tool-call grammar and may appear in existing agent
   logs.
2. **Ephemeral context updates** containing the actual newly required
   instructions. These are appended only to the run-local provider message
   copy after all tool-result stubs, then the model loop continues normally.

The runtime's canonical representation is unambiguous: it appends one tool
result stub for each original tool call, preserving call ID, tool name, order,
and cardinality, followed by one distinct, ephemeral user-context row.
`AgentRuntime` owns canonical result cardinality and ordering;
`ConsoleProviderGateway` plus the existing provider adapter owns transport
serialization. That boundary translates the canonical block without allowing
project text inside an individual tool result:

| Transport | Required serialization |
| --- | --- |
| OpenAI-compatible and Gemini native tool messages | Emit every required tool-response row, then a separate user project-context row. |
| Anthropic native tool use | Emit one user turn containing all required `tool_result` blocks first, followed by a distinct text block containing project context. |
| Fenced/local transports | Close the complete tool-results fence/section, then emit a separately labeled project-context section in the synthetic user text. |

Provider adapters may coalesce rows only as required by their wire grammar;
they may not change call IDs, omit results, or blend instruction text into a
tool-result block. The canonical ephemeral-origin tag survives until request
capture has omitted the body and wire serialization has consumed it.

Automatically loaded instruction contents must never appear in a review
verdict string, tool result, `AgentStep`, run log, transcript marker, exception,
or database row. User-visible activation events name only relative sources and
scopes. This restriction does not rewrite a later, explicit file-tool result.

The ledger marks a source delivered to a model chain only when that chain's
provider payload receives the context update. Byte-budget omissions and stale,
invalid, or failed sources become global terminal no-content outcomes;
token-budget omissions are per chain. Each new terminal outcome is delivered
as a content-free ephemeral warning and can defer that chain once. After the
warning is delivered, the outcome satisfies preflight for that source, so an
identical retry proceeds and cannot loop forever.

Nested budget admission is serialized by the ledger guard and intentionally
first-lock-wins across concurrent parent/subagent batches. Within the winning
batch, selection is deterministic and deepest-first. Later batches use the
remaining global budget and receive explicit omission outcomes if it is
exhausted. A global cross-agent scheduler would add complexity without making
the eventual model-call ordering deterministic.

After a `proceed` preparation outcome, the unchanged `review_tool_calls` hook
runs normally. Permission review therefore happens on the model's reconsidered
tool call, not the discarded pre-activation call. This preserves the guarantee
that approval UI describes the exact call that may execute and avoids coupling
repository-file failures to security-review failures.

## 10. Budgets and model limits

There are two independent configurable raw-content byte budgets:

- startup binding-root budget: 32 KiB per dispatch
- cumulative lazy nested-activation budget: 32 KiB per dispatch

The maximum selected instruction content is therefore 64 KiB before model
limits. Wrapper labels count toward token estimates but not raw-content byte
budgets. The UTF-8 BOM, when present, counts as file bytes.

Budget selection prioritizes more specific sources. Within startup or one new
target batch, candidates are admitted deepest-first; admitted sources are then
rendered broad-to-specific. A source is included whole or omitted whole—never
silently truncated. Omitted sources generate explicit relative-path warnings.

Byte limits are only the outer cap. Before the initial rider and every newly
activated nested update, Chatbook uses the existing model-limit resolver and
token estimator to assemble the ordinary system prompt, conversation payload,
tool schemas, staged context, and response reserve, then computes the remaining
safe input allowance. New project instructions must fit both their byte budget
and that remaining token allowance. Already delivered riders are not silently
evicted. Later tool-history growth retains the agent runtime's existing budget
and terminal-error behavior; this feature does not introduce mid-run
compaction. If the resolver provides no positive safe allowance, Chatbook
records the applicable terminal omission, warns once, and continues rather
than overflowing the context window.

Configuration belongs under `[console]` and exposes only
`project_instructions_startup_max_bytes` and
`project_instructions_nested_max_bytes`, each defaulting to 32768. The token
allowance reuses existing model context and response-reserve configuration
instead of adding parallel knobs.

## 11. Failures, warnings, and security behavior

Project files are untrusted inputs. They can guide model behavior but do not
grant capabilities. Existing workspace authorization, local-tool confinement,
risk tags, permission prompts, provider policy, and Chatbook system prompts are
unchanged and remain authoritative.

Failure behavior is fail-open for optional guidance but explicit to the user:

| Condition | Behavior |
| --- | --- |
| Feature disabled | No instruction-file reads; ordinary agent run |
| No instruction files | First-use notice still explains possible nested loading; run normally; rail shows `None` |
| Empty override | Treat as absent and try `AGENTS.md` |
| Unreadable/invalid/symlinked override | Warn; skip directory; do not fall back |
| Resolver exception | Warn prominently; continue without unresolved guidance |
| Preparation-hook exception | Emit content-free ephemeral UI warning and code-only log; proceed to unchanged security review |
| Stale nested candidate | Defer it to next dispatch and warn |
| Byte/token budget omission | Continue with admitted, more-specific sources and warn |
| Target in another authorized binding | Tool remains eligible; warn that it is outside instruction scope |
| Target outside all authorized roots | Existing tool boundary rejects it |
| Binding missing or locator fingerprint changed | Hold send for re-selection; never retarget silently |
| Selected binding is read-only | Load instructions and expose read-only local operations; omit `fs_write`, `fs_edit`, and `fs_patch` |
| Local project-context write fails | Keep the in-memory choice, warn that it may not survive restart, and do not write through synchronized metadata |

Warnings are aggregated by category and source and shown once per run to avoid
toast storms. Persistent and diagnostic logs receive only content-free codes,
relative paths where safe, counts, and sizes from the automatic loading
channel. They never receive automatic rider bodies. An ordinary explicit
`fs_read` of an instruction file is not reclassified: its tool result follows
existing review, logging, and persistence behavior, as does any model-authored
quotation.

## 12. UX

The Console rail adds one compact project-instruction status row:

- `Off`
- `Choose folder`
- `None`
- `<N> loaded`
- `Warning`

The row opens the existing Context surface. Its Project Instructions section
shows enabled state, selected binding, locator-match status, source precedence,
relative source paths, scopes, byte counts, active/omitted state, and warnings.
It does not add a bespoke file viewer or editor. The existing exact next-send
payload view may naturally show the context rider when the user explicitly
inspects that payload.

Nested activation posts a normal context event naming the newly active
relative sources and scopes. It is not styled as an error. Blocked setup offers
the directly relevant actions: select a folder, disable project instructions
for the session, or cancel the send.

## 13. Testing and verification

### 13.1 Pure resolver tests

Temporary directory trees cover:

- binding-root selection and root-to-target nested ordering
- override precedence, empty override fallback, and invalid override no-fallback
- bounded candidate reads and oversized-override no-fallback behavior
- strict UTF-8 and BOM handling
- no global/personal discovery and no ascent above the binding root
- refusal of symlinked files and directory traversal
- POSIX no-follow/identity checks, Windows symlink/junction/reparse refusal,
  unsupported-metadata fail-closed behavior, and bounded concurrent file growth
- stable-read and changed-after-dispatch behavior
- sibling-scope isolation
- `fs_patch` multi-file target extraction, invalid/delete/rename parity with the
  existing parser, and identical preflight for `dry_run`
- binding-root-only semantics for `fs_glob` and `fs_grep`, including glob
  patterns with static-looking prefixes
- selected-binding activation and other-binding warning behavior for built-in
  `read_file`, `list_directory`, and `write_file`
- read-only selected bindings never advertise `fs_write`, `fs_edit`, or
  `fs_patch`, while instruction discovery and read-only tools still work
- unfiltered and path-filtered semantics for every `git_*` tool, including
  `git_status(path)` as a repository selector rather than a result filter
- deepest-first admission with broad-to-specific rendering
- whole-source omissions under startup/nested byte budgets

Property tests cover canonical path confinement, deterministic precedence, and
budget invariants. Discovery tests assert O(depth) behavior and prohibit a
recursive repository walk.

### 13.2 Runtime integration tests

With a deterministic fake provider, verify:

- startup guidance is present exactly once in each model chain's initial
  provider context and absent from stored transcript state
- text, multimodal, retry, regenerate, and continue use the same rider builder
- a multi-call batch is atomically deferred before approval or execution when
  any target needs new guidance
- protocol stubs precede the ephemeral context update, after which the model
  may issue a reconsidered call that follows the normal approval path
- resolution failure defers once and cannot loop forever
- parent and concurrent subagents share one activation budget while each must
  receive unseen revisions before execution
- byte, token, stale, invalid, and failed outcomes each defer once per affected
  chain, become terminal, and cannot produce a retry loop
- first-lock-wins concurrent admission and deterministic deepest-first
  selection within each admitted batch
- targets in other authorized bindings warn without activating instructions
- non-path-aware providers and disabled sessions retain existing behavior
- opaque process/skill-script calls receive startup context but do not trigger
  guessed nested activation
- `prepare_tool_calls` runs before the unchanged `review_tool_calls`; discarded
  batches never prompt for approval, and preparation failures never rewrite a
  security verdict
- registry first-wins collisions across built-in/local/skill/MCP providers are
  inspected only through the exact owner that dispatch would invoke
- preparation exceptions reach only the ephemeral warning callback and
  sanitized code-only log, then proceed to review without raw exception text,
  instruction bodies, `AgentStep`, transcript, or database leakage
- `/rewind` constructs its summary before startup-rider injection and receives
  no automatic project context

Provider grammar tests assert valid assistant-tool-call/tool-result/context
ordering for OpenAI-compatible, Anthropic, Gemini-style, and fenced/local
transports.

### 13.3 Automatic-channel persistence-leak sentinel audit

Every instruction fixture contains a unique secret-like sentinel, and the fake
provider is configured not to echo it. Across success, cancellation, provider
failure, resolver failure, `/rewind`, and application restart, assert the
automatic loading channel never copies the sentinel into:

- synchronized conversation columns, generic conversation metadata, or
  `sync_log`
- the local `console_project_context_json` column, which may contain only the
  versioned control fields
- agent-run and step databases
- run logs and ordinary application logs
- tool review/result records
- transcript/context event markers
- exception text and error reports
- persisted `/rewind` summaries
- outbound exchange-capture records

Only the test harness's in-memory outbound provider-request spy may contain it.

A separate boundary test explicitly calls `fs_read` on `AGENTS.md` and asserts
that its result retains normal tool logging/persistence behavior. Another fake
provider test returns the sentinel in an assistant response and asserts that
the normal response is not silently redacted. These tests prevent the
ephemeral automatic channel from becoming an accidental data-loss-prevention
filter.

### 13.4 Session, UI, and compatibility tests

- New-session, legacy-session, sole-binding, multi-binding, removed-binding,
  binding-retarget, locator-fingerprint mismatch, and read-only-binding state
  transitions.
- Screen-state and local-only `console_project_context_json` round trips,
  explicit schema-version handling, malformed/forward-version and legacy
  screen-state fallback, temporary-to-durable promotion, migration from the
  actual schema head, and proof that local-only writes create no sync-log row
  or conversation version bump.
- Existing conversation update, soft-delete, restore, and Chatbook import
  conflict paths preserve an existing local project-context column across
  restart. Import `SKIP` leaves the row untouched; non-skip resolutions create
  a new null/disabled row and do not overwrite the existing row. Tests also
  prove outbound conversation triggers and payloads omit the local column.
- A contract test or documented extension point records that any future
  inbound conversation-sync/apply service must use a synchronized-column
  allowlist and preserve this column; implementing such a service is outside
  v1 because none exists today.
- First-use notice proceed/cancel/disable behavior with and without a root
  source, including disclosure that nested sources may load later and
  re-consent on provider/custom-endpoint changes but not model-only changes.
- Rail states, metadata-only Context section, warnings, and nested activation
  events.
- Ordinary chat, non-agent Console paths, disabled project instructions,
  default workspace behavior, the disabled-session `[console] workspace_root`
  fallback, local tool approvals, MCP tools, and existing context preview
  remain regression-covered.
- Small and unknown model context windows, large tool schemas, long histories,
  and existing agent budget exhaustion honor token headroom without adding
  mid-run compaction.

### 13.5 Performance and live verification

Record cold startup-resolution and first nested-activation timings against a
deep synthetic tree. Use deterministic synchronization barriers for
parent/subagent races rather than timing sleeps.

Fake-provider tests prove payload and persistence mechanics, not real adapter
interoperability. Before the feature is complete, run optional credentialed
UAT with at least one native cloud provider and one fenced/local-model path;
include multimodal input when the selected provider supports it and exercise
nested activation followed by a successful retry. Credentials and live tests
remain isolated from the default suite, but provider interoperability evidence
is required for completion.

## 14. Delivery sequence

The implementation plan should decompose the work into three independently
testable, PR-sized deliveries while preserving one design and one ADR:

1. **Startup project context:** resolver, session working context, startup
   instructions, locator fingerprinting, local-only schema migration and
   persistence, selected-root local-provider composition with read-only
   capability filtering, byte/token budgets, migration-safe defaults,
   first-use notice, ephemeral-origin tagging, historical exchange-capture
   omission, `/rewind` boundary tests, base-rider provider transport tests, and
   basic rail/Context visibility.
2. **Nested path activation:** path-aware provider contract and every `git_*`
   / `fs_*` / built-in file-tool scope rule, separate typed
   `prepare_tool_calls` preflight, per-chain terminal outcomes and delivery
   cursors, shared subagent ledger, protocol stubs, nested-channel
   persistence-leak tests, and the full multi-tool provider-grammar suite.
3. **Interop and rollout:** complete UX states, real-provider UAT,
   performance/concurrency evidence, and user documentation.

Each delivery includes all safety and transport checks needed by the context it
introduces and is safe if later deliveries do not land. Delivery 1 may ship
with nested activation explicitly unavailable; delivery 2 completes the
requested path-aware semantics; delivery 3 supplies release evidence and
polish rather than correctness prerequisites.

## 15. ADR check

```text
ADR required: yes
ADR path: backlog/decisions/069-console-project-instruction-local-state-and-preflight.md
Reason: This feature establishes a provider/runtime trust boundary, a new
cross-module path-aware tool contract, and local-only durable session state for
repository-authored instructions. Once accepted, ADR-069 will supersede
ADR-068 after the final readiness audit corrected persistence and preflight
ownership.
```
