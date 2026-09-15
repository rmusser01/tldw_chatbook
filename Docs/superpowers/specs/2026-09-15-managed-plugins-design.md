# Managed plugins and Git marketplaces

Date: 2026-09-15
Status: Draft for written-spec review; six design sections and their review amendments approved.
Task: [TASK-32645](../../../backlog/tasks/task-32645%20-%20Design-managed-plugins-and-expanded-hook-runtime.md)
Decisions: [ADR-162](../../../backlog/decisions/162-managed-agent-plugins.md), [ADR-163](../../../backlog/decisions/163-expanded-console-hook-runtime.md)
Companion: [Expanded hook runtime](2026-09-15-expanded-hook-runtime-design.md)

## 1. Outcome and scope

Users can add a public/private Git or local marketplace, inspect a plugin,
choose supported components, install a managed snapshot, and enable it for
named Chatbook workspaces. They can copy selected existing Cursor/Codex
packages or marketplace definitions into this model. They can explain what
is available, why something is blocked, where it came from, and what an
update or removal affects.

The unit of distribution is a package of agent capabilities:

- Skills and supporting assets.
- Stdio and Streamable HTTP MCP definitions.
- Manual prompt commands.
- Rules/instructions.
- Agent presets.
- Reviewed lifecycle hooks, including the expanded companion runtime.

One installation and selected revision exist per user-data directory.
Workspace activation is independent, with optional user-wide defaults.
Global default activation starts disabled. Importing a catalog does not
install or enable its entries.

Explicit exclusions:

- Arbitrary Python UI/provider extensions, loading Python entry points, and
  arbitrary application monkey-patching.
- Persona/workflow/content-pack distribution; those retain their existing owners.
- Per-workspace package versions, dependency solvers, automatic package
  build/install scripts, and pip/npm package distribution.
- Full vendor-runtime emulation, editor/Tab-completion events, managed
  enterprise-policy imports, or access to vendor-hosted connector accounts.
- Automatic file-glob or model-selected rule activation in this release.
  A user may explicitly select the documented manual adaptation.
- Automatic project-folder discovery that executes or trusts content.

All approved hook additions are in scope. Delivery stages below are ordering,
not deferral of those additions.

## 2. Architectural boundaries

The package layer feeds existing Chatbook runtime services through owned,
qualified component registrations. It does not introduce a parallel agent or
permission engine.

~~~mermaid
flowchart LR
    Sources["Git / local sources / selected app imports"] --> Inspect["Bounded fetch and dialect inspection"]
    Inspect --> Review["Exact snapshot and component review"]
    Review --> Registry["Managed registry and authenticated authority"]
    Registry --> Admission["Workspace and run admission"]
    Admission --> Catalog["Existing tool catalog and permissions"]
    Admission --> Context["Attributed instruction context"]
    Admission --> Hooks["Shared hook runtime"]
    Catalog --> Runtime["Skills / MCP / agent execution"]
~~~

| Owner | Contract |
| --- | --- |
| Acquisition service | Obtain bounded immutable source snapshots; never execute package code. |
| Dialect adapters | Pure validation/normalization, inventory, provenance and compatibility diagnostics. No network, shell or trust mutation. |
| Plugin registry/coordinator | Installation identity, revisions, selections, activation, operation journal, process ownership, recovery and revocation. |
| Trust service | Authenticate reviewed material and authority; verify at use time. |
| Existing Skills/MCP/agent services | Execute owned components through ordinary restrictions, approval and cancellation. |
| Shared hook runtime | Consume normalized owned hook definitions; no dependency on Git, catalogs or plugin discovery. |
| Plugins UI | Review and mutate through the coordinator; projections never grant authority. |

Implementation should introduce a focused Plugins package with separate models,
adapters, acquisition, registry/lifecycle and service boundaries. Extend the
existing tool providers through their public registration seams. Do not move
unrelated code or reuse a destructive standalone skill overwrite as a package
transaction.

Existing [skill trust](../../../backlog/decisions/009-local-skill-trust-boundary.md),
[project authority](../../../backlog/decisions/069-console-project-instruction-local-state-and-preflight.md),
[workspace defaults](../../../backlog/decisions/079-workspace-assistant-defaults.md),
and [Tool-use Packs](../../../backlog/decisions/107-portable-tool-use-packs.md)
remain separate authorities. Package activation grants no filesystem binding,
tool permission, network credential or trusted project status.

## 3. Native authoring and format selection

### 3.1 Portable core

Native packages use Agent Plugins 1.0.0 at the root. The portable loader selects
locally shipped validation rules; it never downloads a schema during inspection.
Core discovery uses immediate skill children and root MCP configuration.
Schema failures use the standard's component-specific failure boundaries,
including its documented nonfatal handling of unknown top-level manifest fields
and malformed extensions. Every ignored item is visible in inspection.
See [loading and discovery](https://agent-plugins.org/client-implementers/loading-and-discovery).

~~~text
review-helper/
  plugin.json
  skills/
    review/
      SKILL.md
  mcp.json
  io.github.rmusser01.chatbook/
    commands/
    rules/
    agents/
    hooks.json
~~~

~~~json
{
  "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
  "name": "review-helper",
  "version": "1.0.0",
  "description": "Review changes using a reusable checklist",
  "extensions": {
    "io.github.rmusser01.chatbook": {
      "version": 1,
      "commands": ["./io.github.rmusser01.chatbook/commands/review.md"],
      "rules": ["./io.github.rmusser01.chatbook/rules/review.md"],
      "agents": ["./io.github.rmusser01.chatbook/agents/reviewer.md"],
      "hooks": "./io.github.rmusser01.chatbook/hooks.json",
      "requires": {
        "skill:review": ["mcp:repository", "hook:protect-writes"]
      }
    }
  }
}
~~~

This example illustrates the proposed Chatbook extension; referenced files must
exist for the package to validate. Reverse-domain extension ownership and the
matching directory follow [Agent Plugins client extensions](https://agent-plugins.org/plugin-authors/client-extensions).
The namespace describes a client contract, not publisher verification.

Extension v1 is closed: version is required; commands/rules/agents are arrays
of contained file paths; hooks is one contained JSON file; requires maps
typed local component IDs to arrays of typed local component IDs.
Absent arrays are empty; there is no extra folder discovery for these fields.
Unknown versions or execution-affecting fields invalidate that extension,
while valid portable components remain inspectable. Duplicate IDs and
dependency cycles invalidate affected components. Unknown namespaces are ignored.
The extension cannot override portable root identity, skills or MCP paths.

An optional variables object maps configuration names to closed declarations:
type (string, integer or boolean), required (boolean), secret (boolean) and
optional default of that type. Defaults are forbidden for secret variables.
Names are 1–64 uppercase ASCII letters/digits/underscores, beginning with a
letter; host-reserved variables cannot be declared. User values live in the
owning config/credential service, not in package files. Required unset values
block affected components. This extension does not change the portable MCP
interpolation rules: remote header/token values use reviewed host mappings.
Vendor variable schemas normalize only their explicitly supported validation
subset; unsupported constraints cannot silently disappear.

Command files use YAML frontmatter with name and description, and a Markdown
body. Optional arguments is an ordered array of unique identifiers. Inputs are
supplied as a separate attributed data block, never evaluated as shell syntax.
Rule frontmatter has name and mode (always or manual); its body is instruction
text. Agent frontmatter has name, description, tools and optional model:
tools is either "inherit" or an explicit array of qualified/local tool
references; an empty array means no tools. A model is a reference requiring a
valid mapping to the parent provider's supported model. Unsupported constraints
block the component rather than falling back to inheritance.

### 3.2 Deterministic dialect selection

The inspection result records format, format version, adapter version, root
manifest and every applied overlay. Detection rules:

1. A valid portable root is the portable candidate. Its Chatbook extension,
   when present, is Chatbook's default interpretation.
2. A portable package can have an OpenAI interpretation. Inline
   extensions.com.openai replaces the compatibility overlay wholesale;
   otherwise .codex-plugin/plugin.json may supply that overlay. Portable
   identity and component locations remain canonical.
3. Standalone .cursor-plugin/plugin.json and .codex-plugin/plugin.json are
   distinct candidates. If more than one viable interpretation remains,
   inspection requires an explicit choice and shows their inventories.
4. A root plugin.json lacking a supported portable schema is not silently
   treated as portable. If a valid vendor candidate exists, offer that
   interpretation with the rejected-root diagnostic. Without a valid candidate,
   reject the package.
5. Never union unrelated vendor manifests. Persist the chosen interpretation
   for updates; a format switch requires a new review.

The OpenAI overlay rule and compatibility layout are based on
[OpenAI packaging documentation](https://developers.openai.com/plugins/build/plugins).
Cursor explicit component locations replace that component's default discovery;
they do not add a second scan. Cursor catalog-supplied definitions are part of
the selected interpretation, with package declarations taking precedence as
documented in the [Cursor plugin reference](https://cursor.com/docs/reference/plugins).

## 4. Identity, persistence and trust

### 4.1 Identity

An installation gets a random internal installation_id. Names, catalog labels
and upstream URLs are not primary keys. Each revision records:

- Original acquisition locator, normalized transport identity and package subdir.
- Selected update source/ref; exact Git commit or local snapshot fingerprint.
- Exact catalog commit/digest, entry identity and inherited definitions, if any.
- Original files, selected dialect/adapter, effective normalized inventory digest.
- Content digest covering relative paths, bytes, executable mode and applicable
  materialized link targets; timestamps are not content identity.
- Reviewed component selection and declared dependency closure.

Do not infer equivalence from a manifest repository field. Two catalogs
supplying identical package bytes but different effective definitions are
different review subjects. Deduplication may offer an existing installation
only when acquisition identity and effective interpretation match; the user
can keep a distinct installation.

Internal component keys are installation_id + kind + local_id. Human command
aliases use a stable installation alias, for example /review-helper:review,
$review-helper:review and @review-helper:reviewer. On collision, assign a
short installation suffix at first install and keep it stable. Native command
names are reserved. Provider tool names use a bounded deterministic encoding
and collision check; display-name equality never resolves a tool.

Structured intra-package references resolve through the installation namespace.
Do not rewrite arbitrary prose. Command/skill argument expansion is one pass;
arguments containing command-looking text remain data.

### 4.2 Storage and concurrency

Use a dedicated versioned SQLite plugin registry through the repository's
private-database seam. Initial plugin schema version is 1; migrations belong
to the plugin store, not an unrelated conversation schema. The registry holds
installations, revisions, selections, workspace overrides, source records,
operation intents and recovery states. Secrets remain references to existing
credential storage.

Under the configured user-data directory:

~~~text
plugins/
  registry.sqlite3
  packages/<installation_id>/<revision_digest>/
  staging/<operation_id>/
  data/<installation_id>/shared/
  data/<installation_id>/workspaces/<workspace_id>/
  receipts/
~~~

Trust snapshots/markers use the trust service's protected store, outside package
content. All paths derive from the resolved profile; a config-only environment
override is not sufficient isolation.

Package snapshots are immutable to supported application operations.
PLUGIN_DATA points to the installation's shared directory for portable/vendor
contracts that expect persistent shared data. The Chatbook extension may use
CHATBOOK_WORKSPACE_DATA, which resolves to the admitted named-workspace directory
or the run's private scratch for unscoped work. No workspace path is synthesized
from a display name. Host-owned workspace configuration is explicitly scoped.
Authors must opt into the workspace path for workspace-specific mutable content;
portable shared data can be visible to the same plugin in multiple workspaces.

V1 has one OS-locked plugin execution/mutation owner per user-data directory.
Other Chatbook instances may read validated registry snapshots, but may not
refresh source caches, mutate packages or execute plugin capabilities until
they acquire ownership. Existing non-plugin functionality continues normally.
Use the existing portalocker dependency and proven lease pattern; do not build
a distributed ownership service. Local filesystems are the supported storage
target; network/synchronized roots need separately qualified lock/publication
semantics and otherwise refuse runtime ownership.

### 4.3 Authenticated authority

Extend ADR-009's passphrase-rooted trust with a separate plugin namespace and
secure generation marker. Do not reset or migrate standalone skill trust into
implicit plugin trust. Review covers all effective instruction/executable
material, catalog overlays, component selection, mappings and configuration
that can change execution or authority. Authenticity includes installation ID,
revision digest, selected interpretation and revocation generation.

The registry controls visibility, but an unauthenticated registry row is never
sufficient to execute or inject content. Authentication failure or a marker
mismatch blocks the affected authority. No recovery path returns an empty
permission store or default Allow. Keyring unavailability and reduced rollback
protection follow ADR-009's explicit posture controls.

Installation can remain untrusted and disabled. "Install and enable" records
activation intent; execution still requires reviewed trust and setup.
No foreign trust, approval, credential or enablement state is imported.
Foreign app configuration contributes only allowlisted source references and
package locations. Per-install env/header overrides and generated credential
files are never copied. If a foreign installation cannot separate those from
its original package, require a clean upstream/local package source instead.
Trust does not sandbox command hooks or stdio processes: they run with host
user privileges. Review must state that fact beside executable components.
Content digests do not pin external interpreters, packages launched through
npx, mutable remote services or network-fetched dependencies.

## 5. Components and runtime behavior

| Component | Chatbook behavior |
| --- | --- |
| Skills | Validate SKILL.md/assets, preserve manual/automatic and inline/fork metadata, advertise only eligible trusted skills. Run-scoped eligible tools can include the installation's MCP tools under parent/workspace restrictions. |
| MCP | Independently configure stdio and Streamable HTTP servers; separate credentials and ordinary permission gates. A failed optional server does not disable unrelated components. |
| Commands | Explicit namespaced prompt entry, supplied arguments treated as data, no shell substitution or recursive expansion. |
| Rules | Always rules apply while active; manual rules apply to the explicitly selected turn. File-triggered and model-selected rules require a reviewed manual adaptation. |
| Agents | Adapt to AgentDefinition with narrowing tool restrictions and validated model mapping. Missing, empty and unmappable tool lists are distinct. |
| Hooks | Use the companion runtime and a tested event/payload/output mapping. Required guards cannot vanish through partial installation. |

Cursor's hyphenated disable-model-invocation and supported vendor spelling
variants normalize explicitly; conflicting declarations are invalid.
File-scoped skills receive the same explicit manual-adaptation treatment as
file-triggered rules. See [Cursor skills](https://cursor.com/docs/skills) and
[rule activation modes](https://cursor.com/docs/rules).

Declared dependencies are mandatory. Deselection or failure propagates to
dependents with an explanation; it never silently broadens their tool access.
For vendor guard hooks whose dependency scope cannot be established, treat
all that installation's executable/automatic capabilities as dependent until
the user reviews an explicit narrower adaptation. Plaintext can describe
dependencies the loader cannot reliably infer; make no completeness claim for
dependency discovery from prose.

### 5.1 Instruction context

Package instructions enter an attributed untrusted instruction-context lane.
They cannot become internal system authority simply because AgentDefinition
normally appends an instruction string. Preserve Chatbook's instruction
hierarchy and source metadata through agents, inline skills and forked skills.

Assembly order within the plugin lane is stable installation_id, then local
component ID: active always rules, explicitly selected manual material, then
accepted hook contributions at their defined boundary. Ordering is deterministic
presentation, not an instruction-priority mechanism. Bodies stay out of
catalog metadata, notifications and ordinary logs. Explicit review/context
views can show their contents through existing privacy controls.

Apply whole-block context limits; never silently truncate a rule or guard
instruction into different semantics. If selected required material does not
fit, block that send/component with a remediation action. Old conversation
messages may still contain prior plugin outputs; disabling cannot erase what
was already sent. It prevents new automatic injection and invocation.

### 5.2 MCP contract

Portable MCP configuration preserves declared transport selection, isolated
entry validation, executable/argv separation and the standard's expansion
rules. Only args, env values and cwd expand portable PLUGIN_ROOT/PLUGIN_DATA;
command tokens, URLs and headers do not. Set host-controlled variables last.
See the [portable MCP runtime contract](https://agent-plugins.org/client-implementers/mcp-runtime).

Direct generic Streamable HTTP is a prerequisite. The existing tldw_server
MCP API wrapper is not evidence of that capability. Implement it behind the
existing MCP client interface using the existing HTTP stack, with versioned
negotiation, JSON/SSE responses, connection lifecycle, cancellation, authentication
challenges, tool discovery and reconnect. Keep transport code separate from
plugin loading. No new mandatory vendor client dependency is assumed.

The initial protocol profiles to qualify are 2026-07-28, 2025-11-25 and the
existing local client's 2025-03-26 baseline. Implement the current per-request
metadata/discovery model and the older initialize/session model as explicit
profiles. Follow the [MCP versioning contract](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
for detection and compatible version selection. Never infer protocol support
from a successful TCP/HTTP connection or blindly reuse the old handshake.
Use side-effect-free discovery for detection; do not probe with tools/call.
Connection readiness is host state even when the wire protocol is stateless.
Unknown versions and optional unimplemented capabilities produce clear
unsupported diagnostics. No arbitrary protocol extensions are advertised.

Use host-managed credentials, explicit header/token mappings and the existing
authentication service for supported OAuth flows. Unsupported authentication
requirements produce Needs configuration with an exact reason. Never infer
access from a vendor app/connector ID. Missing OAuth integration must be
reported as unsupported authentication, not claimed as working transport auth.

Remote endpoints use HTTPS except explicitly selected loopback development
servers. Do not forward credentials across origins or follow an authentication
redirect as an authorization grant. Qualified protocol-era detection stays
within the selected transport; no downgrade to deprecated HTTP+SSE or weaker
TLS/authentication after failure.
Reconnect refreshes definitions and permissions; it never automatically
replays an invocation with an uncertain result.

Vendor interpolation is a versioned adapter behavior. Reserved host variables
cannot be overridden; missing variables keep affected components unready.
Do not add portable-variable expansion to arbitrary vendor fields by analogy.
Configured executable lookup uses the host-approved environment, not package
instructions that rewrite process lookup. Relative package executables must
remain inside the immutable snapshot. User-defined MCP connection mappings
are reviewed by identity and definition digest, never matched by display name.

### 5.3 Compatibility and readiness

Store separate axes:

- Support: supported, adapted, unsupported, invalid.
- Selection: selected or deliberately excluded.
- Availability: disabled, needs trust, needs configuration, ready, failed,
  recovery required.
- Evidence: parsed, connection exercised, behavior exercised, optional
  original-host comparison; include time, revision, adapter, platform and
  configuration identity.

Evidence expires when its relevant inputs change. Ready means eligible for
normal execution gates; it is not a claim of safety or exhaustive testing.
Support added in a later Chatbook version does not auto-select previously
excluded components. Adapter interpretation changes require renewed review.

## 6. Marketplaces and acquisition

Sources are public/private HTTPS or SSH Git repositories, GitHub shorthand,
or an explicitly selected local directory. Git is a checked prerequisite,
with a clear installation diagnostic if missing. Users choose optional ref
and package subdirectory. Branches/tags resolve to commits before review.

Supported catalog locations are .agents/plugins/marketplace.json,
.cursor-plugin/marketplace.json and the Codex-compatible legacy
.claude-plugin/marketplace.json. Legacy catalog support does not promise a
Claude runtime adapter. If several catalogs exist, choose each source
explicitly. A Chatbook-authored catalog uses the supported Codex-compatible
shape, for example:

~~~json
{
  "name": "team-plugins",
  "interface": {"displayName": "Team plugins"},
  "plugins": [
    {
      "name": "review-helper",
      "source": {"source": "local", "path": "./plugins/review-helper"},
      "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"},
      "category": "Productivity"
    }
  ]
}
~~~

Resolve local entry paths relative to the catalog root, not its metadata
directory. Git-backed root/subdir entries honor supported ref/sha selectors;
conflicting selectors reject the entry. npm/unknown source types remain
visible as unsupported, without disabling valid siblings.

Catalog metadata can supply executable component definitions. Pin catalog
and package content together and show the effective source of each definition.
Catalog policies never auto-install, authenticate or grant permission in
Chatbook. Publisher metadata is self-declared; display acquisition provenance
separately. The application includes no supposedly connected marketplace until
the user adds or imports one.

### 6.1 Safe acquisition

Run Git with argv, disabled repository hooks, no checkout filters or recursive
submodules, and a restricted transport set. Read the selected tree and
materialize validated files without invoking repository-controlled checkout
behavior. No Git LFS fetching, submodule execution, install scripts or builds.
If omitted material is needed, affected components are unready with that reason.
Existing user Git credential helpers and SSH agents remain explicit host
authentication mechanisms; package content cannot select new helpers or SSH
commands. Never copy credentials into locators, receipts or logs.

Do not accept local file paths from a remote catalog as authority to read
arbitrary host files. Resolve package files inside the fetched catalog root.
Separate remote Git entries identify their own explicitly reviewed origin.
Disable Git protocol forms such as ext that execute arbitrary helpers.
Transport redirects cannot silently widen approved network access.
Private LAN/self-hosted origins are user-selected source authorities; a
catalog entry does not silently authorize unrelated local network targets.

Apply byte/time/file-count limits while fetching and materializing, not after
an unbounded clone finishes. External Git runs have a staging quota monitor,
deadline and cancellation; terminate and discard staging on exhaustion.
Never use recursive package traversal to follow links outside its root.
Materialize internal regular-file links only after containment checks; reject
directory links, external links, device/special files and platform-ambiguous
paths. Report this stricter link policy as a Chatbook adaptation where needed.
Reject case/Unicode-normalization collisions that cannot coexist faithfully
on the destination platform.

### 6.2 Refresh, updates and imports

- Refresh catalog updates cached listings, with last-success time and error.
- Check updates resolves candidate package/catalog revisions without activation.
- Review update selects and commits a reviewed candidate through the lifecycle.

Search uses bounded cached metadata. Inspection fetches only selected packages.
Failure preserves the last usable source snapshot and labels it stale.
Removing a source preserves installations and pauses updates through that
source; rebind update provenance through explicit review.

Import from Cursor/Codex first selects the app/location and then distinguishes
marketplace definitions from installed package copies. Discovery reads only
known allowlisted non-secret configuration fields and selected package roots.
Do not sweep entire home directories or read credential databases.
Never execute the foreign app or its plugins to discover installation state.
Local copies receive their own fingerprint and installation identity.
Binding a copied package to an upstream update source is a separate review.
Partial import failures are reported per item; successful copies remain
installed but never inherit foreign grants or activation.

## 7. Activation and run ownership

Workspace overrides are Inherit, Enabled or Disabled. Inherit resolves to the
plugin's explicit user-wide default, initially disabled. A valid named
workspace ID is required to persist an override. Unscoped/default work uses
the global default and offers Install disabled or an explicit global-default
edit; it never silently converts a workspace action to global enablement.

Admission captures a run snapshot containing installation/revision IDs,
selected components, effective workspace scope, mappings and authority
generations. Before every injection, child launch, hook dispatch or tool
invocation, recheck current trust, revocation and parent/workspace restrictions.
Already-admitted runs retain their revision; a new enablement does not add
capabilities halfway through a run.

Leases track actual runs, dispatched operations and background jobs. An idle
MCP connection or archived checkpoint is not a run lease. Resuming an archived
run revalidates its required revision, component definitions and current
authority. If the old revision is incompatible with the selected installation
or shared data generation, refuse exact resumption and offer a new run with
the current revision. Never silently substitute revisions.

## 8. Installation, updates and removal

### 8.1 Review and installation

The flow is Inspect → Review → Install → Configure → explicit use/test.
Inspection/review do not execute plugin commands, hooks or MCP processes.
Activation only makes components eligible at a subsequent runtime admission.
Connecting/testing a server is an explicit execution action.

Review is bound to installation/replacement identity, exact package/catalog
digests, adapter interpretation, selection, dependencies, target workspace or
global default, configuration/mappings and relevant authority generations.
Changing these inputs makes the pending action stale. A branch moving does
not invalidate the already-reviewed immutable commit. Navigating elsewhere
cannot retarget the action.

Review explains executable privileges, hooks, new endpoints, adaptations,
blocked dependencies, selected/excluded components and affected workspaces.
Actions are Install disabled and Install and enable in <workspace name>.
Trust/bootstrap and missing configuration remain visible separate gates.

### 8.2 Authoritative commit and recovery

Use one operation_id and an authenticated write-ahead intent with old/new
authority generations and digests. Files and derived registrations are staged
before publication. No derived Skills/MCP entry is independently advertised
without its active, authenticated plugin registry reference.

The commit order is:

1. Acquire mutation ownership; validate the exact reviewed inputs again.
2. Persist prepared immutable material and authenticated recovery intent.
3. Commit registry state and operation phase atomically in SQLite.
4. Advance the external secure trust generation marker.
5. Publish eligible runtime projections and acknowledge completion.

Between steps 3 and 4, affected plugin authority is fenced. The operation has
committed state to reconcile, not permission to execute. An authenticated
intent describes the exact approved transition; an arbitrary journal file is
not recovery authority.

| Observed durable state | Recovery |
| --- | --- |
| Prepared intent; registry and marker still old | Abort unpublished staging; preserve old installation. |
| Registry new; marker old; matching authenticated intent | Keep execution blocked; complete the exact marker transition after trust unlock. |
| Registry and marker new; completion response missing | Rebuild projections and return the recorded committed result for the same operation_id. |
| Marker new; registry old/missing | Fail closed. Recover the exact new state only from an authenticated committed recovery record; otherwise require recovery. |
| Invalid journal, content mismatch, unrelated generations or corrupt registry | Quarantine affected authority; preserve evidence; do not guess defaults or overwrite it. |
| Committed removal; filesystem cleanup failed | Keep revocation effective and report Uninstalled — cleanup pending. |

The journal remains until the registry, secure marker and recoverable
publication agree. Do not prune the only recovery record. File fsync/replace
and directory durability follow platform-qualified storage helpers; a
successful write call alone is not a cross-platform durability claim.

Retries reconcile the same operation. A lost response cannot cause a second
install, recreated grants or blind rollback. An independent process must be
able to recover each interrupted transition without running plugin code.

### 8.3 Applying an update

Download, inspect and review while work continues. Applying an update installs
an admission fence under the same lifecycle lock used by run admission:

- No new old-revision work is admitted. New runs requiring it receive an
  actionable waiting/unavailable result rather than silently omitting it.
- Existing leased work may finish. Pending approvals are visible blockers.
- Stop hooks cannot enqueue continuations that perpetuate the drain.
- Idle MCP connections close after active requests drain.
- The user can keep waiting, cancel the update before commit, or explicitly
  cancel affected work. Waiting does not authorize forced cancellation.

After drain and confirmed local cleanup, publish the new revision through the
commit protocol. The update review includes effective content/configuration
changes, executables, hooks, endpoints, affected workspaces and trust changes.
Preserve deliberate exclusions. New or newly supported capabilities remain
unselected until reviewed. Changing an adapter alone can require the same
review as a package update.

Retain the prior immutable package and historical selection for comparison.
Rollback is another reviewed revision transition through the same drain and
commit gates, reconciled with current explicit disables, mappings and grants.
Historical trust records are evidence, never restoration of permission.
There is no automatic rollback after a runtime startup failure: the new code
may already have changed data or an external service.

Rollback does not undo PLUGIN_DATA, external effects, remote changes or
external dependency updates. Opaque data compatibility is Unknown unless a
host-verifiable declaration establishes it. Report this before rollback.

### 8.4 Disable, uninstall and data

Disable here changes workspace activation. Disable everywhere revokes plugin
availability across scopes without deleting the installation. Uninstall is
user-wide. These actions are distinct from disconnecting one MCP connection
or removing a marketplace.

Seal current admission immediately and durably commit revocation before
cleanup. Pending approvals, continuations and callbacks must observe the
new generation. Plugin-owned Stop, Interrupt and SessionEnd callbacks are
suppressed after revocation; cleanup is performed by the host and cannot
be vetoed. Existing independently configured user hooks retain their own
authority and cannot revive the removed installation.

Cancel already-dispatched work where possible, without claiming its effects
were undone. An uncertain remote outcome remains uncertain and is not
automatically retried. Uninstall revokes installation-owned grants and
registrations; independent MCP connections, shared credentials and unrelated
policy records remain owned by their existing services.

Stale workspace references resolve to Unavailable, never a name-matched
replacement or permissive default. Reinstallation gets fresh authority.
Retained data is tied to the original installation ID, with explicit reviewed
reattachment; a same-name package/fork cannot automatically inherit it.

Keep data by default. Delete saved plugin data is a separate destructive
action naming the owned roots and affected workspaces. Delete only validated
owned paths; never follow a symlink into an unrelated directory. Locked files
or cleanup failures leave a visible cleanup-pending record without undoing
revocation.

### 8.5 Owner death and surviving children

An OS lock releases on owner death; child processes may survive. Persist an
unclean-runtime record before launching plugin-owned local processes and
retain exact process provenance through launch/publication/cleanup. A new
owner must reconcile these records before admitting affected plugins.

Use platform-qualified containment/termination for owned process trees.
Never kill an arbitrary reused PID based solely on a stale number. When the
host cannot prove an owned process stopped, quarantine the affected plugin's
runtime/data reuse as Recovery required. The user can inspect diagnostics and
perform explicit recovery after resolving surviving processes; there is no
automatic force-takeover action.

Process groups/timeouts do not sandbox malicious commands or guarantee
termination of deliberately escaped descendants. Remote side effects also
cannot be fenced by a local lock. The product reports these limits instead
of advertising transactional rollback of arbitrary plugin behavior.

## 9. Plugin management UI

Plugins is a dedicated shell destination with visible navigation, command
palette access and a link from the canonical Settings screen. Global plugin
preferences live in canonical Settings; no additions to deprecated settings
surfaces. Use the shell's registered shortcut labels rather than hardcoding
the historical F9 label. Adding a destination cannot silently reassign
existing shortcut ownership.

| View | Job |
| --- | --- |
| Installed | Components, activation, setup, update/rollback, disable/uninstall, operations and recovery. |
| Browse | Cached source-qualified search and package inspection. |
| Marketplaces | Add/edit source locators, refresh status, authentication remediation and source removal. |

Install from repository and Import from Cursor/Codex are visible entry actions.
An empty installation offers these actions and adding a marketplace; it does
not imply that a remote directory or account is already connected.

### 9.1 Scope and details

Separate Activation in <workspace> from Installation on this device.
Show Inherit/Enabled/Disabled, effective source and component availability,
for example: Enabled in Research via global default · MCP needs configuration.
Show 3 of 4 selected components available instead of collapsing partial
functionality into one misleading badge. Global defaults have a separately
labeled control and cross-workspace impact preview.

Details provide source/catalog identity, exact revision, component inventory,
selection/dependencies, adaptations, trust/setup state and evidence freshness.
Unknown compatibility remains visible in discovery. Display author claims
separately from transport provenance; names never imply verified publication.

Package-owned skills appear in Library → Skills and servers in MCP, with
ownership labels and links back. Definitions are read-only through both UI
and service mutation paths. Supported user settings/credential mappings
remain editable through their existing owner. Removing a package component
routes to Plugins selection, not physical deletion by a legacy action.
Cross-screen handoffs carry installation/component/workspace identity and
restore review, selection and focus on return.

### 9.2 Interaction and recovery

Use the existing terminal list-and-details shell. Wide layouts show both;
narrow layouts provide a full-width details view with Back. Preserve search,
filters, page, stable selection and focus across transitions. Reject obsolete
search/inspection responses. Reordering cannot retarget a pending action.
Distinguish no sources, no matches, failed refresh and stale cached results.

Close/Back means leave the view, not consent or cancellation. Before submission,
retain an in-session draft; an explicit Discard clears it. Draft inputs must
be revalidated on return. Secrets are held only by the owning credential flow.
Unsubmitted drafts are not promised across application restart. Durable
operation progress/outcomes are available after navigation and restart.
Cancel operation is a separate state-aware request; after commit it cannot
pretend to undo the operation. Escape, backdrop clicks and Cancel follow the
existing safe-dismissal grammar, including nested overlays and focus return.

Show active blockers, cleanup pending and Recovery required with specific
actions. A secondary instance explains that another Chatbook instance owns
plugin execution; it never silently takes over or terminates that process.
Configuration save never starts a server. Explicit Test connection/Run hook
test actions disclose execution and use the normal authority path.

External descriptions/readmes/status text are bounded untrusted content:
literal labels, control-sequence sanitization, safe link handling, no
automatic remote images or executable previews. Render source bodies only
in deliberate details/review views. Ordinary logs show stable identifiers,
phases and error codes, not raw credential values, hook payloads or config.

### 9.3 Design language

Follow the [design constitution](../../../backlog/docs/design-language.md).
Use the standard sidebar shell, list/table rows, stacked form components and
status areas. Spacing uses ds-space-inline/stack/section/inset; controls use
ds-control-height variants; separators use ds-grid-line/ds-column-line.
All interactive states use existing rest/hover/focus/disabled tokens.
Readable disabled explanations are visible without hover; color is never
the only status carrier.

At 80×24 and 120×35 terminal sizes, keep scope, review consequences and primary
actions reachable without horizontal scrolling; long names wrap or have an
explicit detail reveal. Pane navigation uses existing global conventions.
No terminal-convention key overrides. Footer hints advertise working actions
only. Add any necessary token before feature CSS and rebuild the generated
bundle through its builder.

## 10. Resource and failure limits

These are initial host defaults and hard caps for package inspection and
management. A package/catalog cannot raise them. Any future user-configurable
increase requires validation and explicit settings; it is outside v1.
Smaller existing runtime/provider limits continue to apply.

| Resource | V1 limit | Exhaustion behavior |
| --- | --- | --- |
| Manifest or hooks definition JSON | 256 KiB each; depth 32 | Reject that document with a bounded diagnostic. |
| Catalog snapshot | 5 MiB; 10,000 entries; 50 sources | Reject oversized refresh; retain previous catalog. |
| Package snapshot | 100 MiB expanded; 10,000 files; 10 MiB/file; path depth 32 | Abort materialization; preserve current installation. |
| Normalized component inventory | 512/package | Reject inspection; do not drop arbitrary tail components. |
| Concurrent acquisitions | 2; 120 s overall per acquisition | Queue visibly or cancel with timeout; no plugin execution. |
| Git staging including object data | 500 MiB/operation | Terminate fetch at quota check; discard staging. This is host acquisition control, not an OS disk quota. |
| Managed package/cache storage | 2 GiB total; require estimated new bytes plus 100 MiB free reserve | Prune eligible cache or refuse before commit. |
| Inactive revisions | 2 most recent per installation; 30-day age target | Prune only unleased/unreferenced revisions; current and recovery records are protected. |
| Abandoned staging | 24 hours | Remove only after journal reconciliation and ownership checks. |
| Plugin instruction blocks | 8 KiB/block, 32 KiB combined per send, also bounded by remaining model context | Reject oversized selected material whole; explain affected components. |
| Listing page | 50 rows | Paginate; search remains over cached metadata. |
| Display metadata | 256 characters/name; 2,000/summary; 64 KiB README preview | Sanitize and mark display truncation; preserve immutable source for explicit file review. |
| Operation receipts | 1,000 terminal receipts or 30 days | Drop oldest eligible terminal receipts; never delete recovery authority. |

Plugin data and an external program's arbitrary writes are excluded from the
managed-cache quota; show data usage separately and never auto-delete it.
Long-running readers/writers may temporarily prevent retention targets; show
protected usage and refuse new acquisitions rather than pruning live material.
Hook-specific limits are defined in the companion spec.

No catalog scan/fetch or MCP launch blocks app startup. Initialize metadata
services lazily and use workers for work over 100 ms. Verify existing startup
and screen-preload budgets, plus responsive cancellation under worst-case
bounded input. A fetch completing only after unbounded work fails this contract.

## 11. Verification and acceptance

Each implementation task must identify the corresponding contract below.
Use controlled Git repos, stdio/HTTP servers and deterministic barriers for
repeatable CI. External live checks are separate, explicitly scoped evidence.
Do not execute third-party plugin code just to establish parser compatibility.

1. **Formats:** version-pinned upstream fixtures and independent expected
   inventories exercise portable, Cursor and Codex candidates, overlays,
   explicit paths, unknown fields, malformed components and ambiguities.
   Record provenance/license and fixture revision.
2. **Behavior:** a native skill installs, activates, invokes its own MCP
   tool through normal Console review, and becomes unavailable after disable.
   Manual-only skills, empty allowlists, model mappings, rule lifetimes and
   namespace collisions hold across composer/catalog/agent paths.
3. **Partial support:** unsupported optional items do not disable unrelated
   components; required guards/dependencies block affected behavior; new
   adapter support never auto-selects excluded items.
4. **Authority:** tamper, locked trust, stale approvals, workspace changes,
   registry rollback and same-byte reinstall cannot revive permissions.
   Pair negative tests with successful controls on the same production entry.
5. **Acquisition:** traversal, escaping links/reparse points, duplicate paths,
   hostile Git configuration, oversized/deep trees, moving refs and source
   authentication failure cannot widen file/network authority or run package code.
6. **Publication:** real child-process termination before/after each durable
   commit boundary recovers in a different process. Repeat operation IDs,
   missing responses, full disks and cleanup failures yield the recorded state.
7. **Concurrency:** competing app processes, launch-before-owner-death,
   surviving children, pending approvals, busy updates and late callbacks
   preserve the admission/revocation contract. Observe completion, not a
   transient idle counter.
8. **Privacy/resources:** credential sentinels stay out of ordinary logs,
   receipts, listings and errors; guard payload integrity is tested separately.
   Bounded inputs, loops, retention, worker/process/file-handle cleanup and
   cancellation are checked at the actual resource boundary.
9. **UI:** mounted and live isolated-profile checks cover browse → inspect →
   review → install → configure → use → update → disable → uninstall; include
   keyboard-only 80×24/120×35, stale responses, changed/deleted workspace,
   narrow details, configuration return and cleanup-pending recovery.
10. **Regression:** standalone Skills/MCP and legacy hook configs keep their
    documented behavior; no automatic import/enablement; historical registry/
    trust migrations reopen through production paths without lost grants or
    revived grants. Package-owned legacy edit/delete paths remain guarded.
11. **Platform:** Linux/macOS/Windows runs qualify locks, publication, path
    normalization and process cleanup. Local evidence/skipped tests cannot
    establish all-platform compatibility.
12. **Governance:** targeted functional suites plus applicable private-DB,
    configuration, tool, route, modal, startup/preload and CSS inventories;
    changed-file lint/format and document-link checks. Full suite only on
    explicit user request.

Run UI/live verification with isolated config, data, credential and process
roots and prove the isolation. Follow the repository's
[testing evidence](../../../backlog/docs/lessons-testing-evidence.md) and
[live verification](../../../backlog/docs/lessons-live-verification.md) lessons.
The release compatibility matrix distinguishes parsing, exercised Chatbook
behavior and direct original-host comparison by dialect/adapter/platform.
Never publish a blanket Cursor/Codex-compatible claim from parser tests alone.

## 12. Delivery and handoff

1. Registry/trust/recovery and minimal native skill integration sufficient
   for local install → activate → use → disable.
2. Native expanded hook events and structured effects through existing
   dispatch/scheduler interfaces.
3. Direct MCP transport/authentication and package-aware permission-gated
   invocation.
4. MCP-backed hooks and full portable/vendor component adapters.
5. Git acquisition, catalogs, selected app imports and management UI in
   individually tested vertical slices.
6. Integrated qualification, authoring examples and compatibility report.

Every stage includes its own documentation, targeted verification and
review; final qualification does not postpone integration testing.
Implementation plans and atomic Backlog tasks are created after written-spec
approval, with foundation dependencies pointing only to already-created tasks.

ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md;
backlog/decisions/163-expanded-console-hook-runtime.md
Reason: New storage/trust/ownership, runtime contracts, compatibility boundaries
and a long-lived management destination.

## 13. Design review record

The written design incorporates all accepted section reviews:

- Package identity/overlay ambiguity, mutable-data separation, run revisions,
  independent trust and owned runtime registration.
- Metadata normalization, MCP-capable skill runs, rule modes/context bounds,
  namespaces, tested hooks, transport prerequisites and persistent exclusions.
- Catalog/package provenance, explicit imports and separate refresh/update.
- Admission drain, revocation before callbacks, journal recovery, single
  runtime owner, exact review targets, current-policy rollback and cleanup.
- Scope clarity, partial readiness, service-level ownership, safe source
  display, stable async selection, dismissal and accessible navigation.
- Surviving children, platform evidence, dependency-aware milestones,
  independently derived fixtures, regressions/migrations and concrete limits.

This is a proposed implementation contract, not a report of implemented
features or passing runtime tests. The two written specs receive a final user
review before implementation planning.
