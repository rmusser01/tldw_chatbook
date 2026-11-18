# Authoring managed plugins

Start with the repository's executable [native review example](../../Tests/Plugins/fixtures/native_review/plugin.json)
and [skill](../../Tests/Plugins/fixtures/native_review/skills/review/SKILL.md).
Copy the complete `native_review` directory to a local folder, inspect it in
Plugins, review installation, review trust, then enable it in the intended
workspace. Installation and enablement do not grant tool permission.

The example has one `skill:review` and an explicit empty tool ceiling. Invoke
its displayed installation-qualified skill name in Console with a change and
acceptance criteria. The completion text is model output, not a host execution
receipt. The qualification test controls the provider response and inspects the
actual final request. [Provenance](../../Tests/Plugins/fixtures/PROVENANCE.md)
records its license and origin.

## Invocation and inspection identity

Use the exact installation-qualified name shown by Plugins. Current native
commands use `$alias:command:id` with strict named JSON arguments; manual rules
use `$alias:rule:id`. Existing `/alias:id` command and `@alias:id` manual-rule
forms resolve only the matching captured typed row. Only the slash form with
one declared argument treats its original remainder as opaque argument text;
zero or multiple arguments still use strict named JSON. Expanded references and
display names cannot select another component or widen its authority.

New inspections record `chatbook-portable/2`, `chatbook-codex/2` or
`chatbook-cursor/2`; this is an interpretation identity, not a new manifest
version. Existing installations retain their authenticated historical
interpretation. Ambiguous or unrecognized retained records refuse instead of
being silently converted. Foreign agent definitions must explicitly declare a
supported tools form; an explicit empty list stays empty.

## Components and constraints

| Component | Authoring contract |
|---|---|
| Skill | `skills/<name>/SKILL.md`, YAML name/description and Markdown body; manual-only and allowed-tool restrictions remain effective. |
| MCP server | Root `mcp.json`, portable `mcpServers`; stdio or Streamable HTTP. Save configuration, then separately review connection/discovery and exact tool mappings. |
| Command | Chatbook extension `commands` paths; name/description, optional ordered arguments; arguments remain data, never shell interpolation. |
| Rule | Extension `rules` paths; name and `mode: always` or `manual`; body enters attributed user context. |
| Agent | Extension `agents` paths; name/description/tools, optional model reference. `tools: []` denies all tools; `inherit` uses the parent's eligible ceiling. Model references need reviewed canonical same-provider mappings. |
| Hook | Extension `hooks` path to version-2 JSON; event/type/effects and declared dependencies follow the hook contract below. |

Use `extensions.io.github.rmusser01.chatbook.version: 1`. Declare required
edges in `requires` using exact typed local IDs, such as `skill:review` and
`hook:protect-writes`. Missing, cyclic or unsupported required constraints block
affected use, including manual invocation; unrelated supported components may
remain available. Unknown recognized-extension versions never erase guards.
Newly supported components still require explicit selection review.

The complete [native format contract](../superpowers/specs/2026-09-15-managed-plugins-design.md#31-portable-core)
and [hook contract](../superpowers/specs/2026-09-15-expanded-hook-runtime-design.md#2-definition-and-result-contracts)
are authoritative for fields and permitted event effects. Hook commands use argv
arrays. Command output is bounded version-2 JSON; MCP hooks preserve typed error
and structured-content semantics. `require_context` requires a nonempty bounded
block; plain required success may return no context. Required final guards see
the final transformed arguments. Approval does not survive changed arguments.
Required post-events block subsequent model admission without replaying already
committed tool, compaction or child outcomes.

## Hook events and permission

| Event | Permitted role |
|---|---|
| SessionStart | Provisional session admission; deny or contribute runtime context. |
| UserPromptSubmit | Validate the explicit submission; deny or contribute turn context. |
| PreToolUse | Transform candidate arguments, guard the final candidate, or contribute context before ordinary approval. |
| ApprovalRequested | Observe a permission request; never approve it. |
| PostToolUse / PostToolUseFailure | Observe settled dispatched outcomes and contribute turn context; required checkpoints block the next model input, never replay effects. |
| SubagentStart | Deny or narrow inherited child limits and context; never widen authority. |
| SubagentStop | Observe a settled child and contribute to an active parent's next input; late context is discarded. |
| PreCompact / PostCompact | Contribute before actual compaction or after its committed summary; a failed post-checkpoint cannot undo that summary. |
| Stop | Propose a bounded scheduler continuation or end continuations; user work and cancellation win. |
| Interrupt / SessionEnd | Bounded observation during cancellation/disposal; cannot veto, reconnect, ask for approval or continue work. |

All 13 events use their [defined timing and effects](../superpowers/specs/2026-09-15-expanded-hook-runtime-design.md#3-event-timing-and-effects).
A command hook uses a controlled process; an MCP hook uses an explicitly connected,
reviewed independent dependency. Missing connections and initialization/guard cycles
refuse dependent use. Successful initialization never reuses another run's stamp.
Legacy six-event configuration keeps its existing version-1 behavior.

## Declared values and command locations

Declare `variables` in the Chatbook extension with `type` (`string`, `integer`
or `boolean`), `required` and `secret`. Optional nonsecret declarations may have
a typed `default`. A command hook consumes `${NAME}` in argv or environment,
or an exact `{"variable": "NAME"}` environment value. Its optional `cwd` is a
literal captured package-relative directory, with no variable substitution.
Substitution in argv and environment happens once;
a value containing another placeholder remains data. Strings are bounded to
16 KiB UTF-8; integers are signed 64-bit and booleans are not integers.

Users save typed values in the host's protected credential backend, then review
reference mappings. Packages contain declarations, never protected values or
reference authority. Unused declarations do not block unrelated components.
An unavailable selected required or controlling hook retains its original event
and matcher: a write guard still blocks writes but does not block reads. A
required initializer blocks its initialization event; dependency-only requirements
narrow their dependents. Child and archived continuation ceilings retain those
constraints without advertising the unavailable hook as executable.

Three reserved names come only from current host capabilities:

- `PLUGIN_ROOT`: the immutable captured package.
- `PLUGIN_DATA`: the single current shared saved-data grant.
- `CHATBOOK_WORKSPACE_DATA`: the single current grant for the captured named
  workspace, or the actual Console scratch capability in an unscoped chat.

Missing or ambiguous grants refuse the consuming launch. Workspace names never
construct paths. Scratch remains leased until the real hook process has settled;
closing the chat revokes new use without deleting a live process's directory.
A package may use an explicit absolute or `./` executable. Bare commands resolve
against the trusted host PATH before applying the child's reviewed environment;
a child PATH override cannot select another executable.

## Limits and portability

Manifest/hooks JSON: 256 KiB each, depth 32. Package: 100 MiB expanded,
10,000 files, 10 MiB per file, depth 32; inventory: 512 components. Plugin
instructions: 8 KiB per block; hook context: 4 KiB per block and 16 KiB per event;
combined attributed send: 32 KiB and the remaining model-context limit. An
oversized selected block is rejected whole. Catalogs: 5 MiB, 10,000 entries,
50 sources. Display truncation does not modify immutable source.

Hook execution defaults to 10 seconds, with a 60-second handler ceiling;
effectful events have a 180-second wall ceiling including waits. MCP approval
has a separate 120-second ceiling. Interrupt/SessionEnd default to one second
and cap at three; host reaping has a separate five-second allowance. Unresolved
cleanup retains ownership. Input envelopes cap at 1 MiB, depth 32 and 16,384 JSON
nodes; structured results cap at 16 KiB and retained stderr at 4 KiB. Effectful
execution caps at 4/runtime and 8/application, with 16/64 outstanding lifetime
tickets. Optional observation queues cap at 64/128 deliveries; required guards
never use the lossy queue. Causal depth is 4; Stop chains cap at 3 turns and 120
seconds. Definition/matcher and continuation limits are in the complete
[budget table](../superpowers/specs/2026-09-15-expanded-hook-runtime-design.md#9-budgets-and-failure-policy).

Do not put secrets in package files, literal headers, prompts or hook output.
Use reviewed host credential references. Generic MCP OAuth is unsupported.
A plugin's prompt, hook or model declaration never supplies filesystem, network
or tool authority. See [operations](operations.md) for cleanup/recovery and
[compatibility](compatibility.md) for the exact limits of exercised behavior.
