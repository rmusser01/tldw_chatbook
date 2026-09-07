# User settings

The F9 Settings destination is the canonical place to configure Chatbook. The
legacy Tools Settings window and enhanced Chat sidebar are deprecated and do
not receive new settings. Code comments marked `# USER-SETTING` identify values
that should be represented in this surface rather than requiring users to edit
configuration files.

Settings are grouped into searchable categories. A category owns presentation,
validation, and draft state; application services continue to own persistent
data and security-sensitive mutations. Long-running filesystem, database, or
network work runs outside the Textual event loop, and a result is applied only
while its captured category and identity are still current.

## Tool Profiles

Tool Profiles are named Allow/Ask/Deny policies for built-in, local, Virtual
CLI, raw-shell, and external MCP tools. Settings owns their lifecycle UI:
listing, portable import and export, selecting a workspace bind, and removal.
The MCP Permissions destination remains the only editor for individual global,
server, and tool rules.

Each `.tldw-tool-pack` V1 archive is a deterministic, policy-only package. Its
review shows the suggested profile id, safe display name, producer, content and
policy digests, Ask/Deny fallbacks, Allow/Ask/Deny counts, omitted rules, and
excluded authority categories. It carries stable server keys and raw tool names
plus hashes of tool descriptions, schemas, and policy-relevant risk tags. The
description and schema text themselves are not included.

Export never includes executable tools, skills, plugins, server configuration,
credentials, commands, arguments, environment variables, endpoints, approval
history, session grants, global kill-switch state, workspace or Persona data,
project-instruction bindings, or import receipts. The review captures an exact
profile digest/revision and the publication destination is captured separately;
if either changes, the operation stops instead of silently retargeting.

Import is review-first and initially unbound. Automatic matches require the
same authority, exact server key, raw tool name, and contract hash. External MCP
servers may be mapped only through explicit one-to-one source/destination
mappings shown in the review. Changed or missing Allow/Ask rules are omitted;
safe Deny rules can remain pending. Importing a profile does not install tools,
change an existing workspace, or make the profile active.

The first workspace bind requires a second confirmation showing the exact
workspace, current defaults, profile id/revision/digest, effective posture, and
Allow/Ask/Deny details. Persona and memory acknowledgement remains the existing
independent workspace control. Later policy edits are made in MCP Permissions
and stale pending actions are rejected.

Imported profiles retain a private local receipt for provenance and uncertain
outcome reconciliation. If that receipt is missing or invalid, Settings reports
degraded receipt health; policy remains authoritative, the first-bind marker is
not bypassed, and destructive lifecycle actions fail closed where proof is
required. A removable imported profile must have no active or archived
workspace references and no active runtime lease. Removal replaces it with a
hidden permanent-Deny tombstone, so its id cannot fall through to another
profile later.

Errors use bounded stable categories. A result such as
`durability_uncertain`, `activation_uncertain`, or `outcome_uncertain` means the
application reconciled what it could but will not guess whether an ambiguous
replace completed. Refresh the profile list before retrying. Native Windows
publication is reported separately as `publication_unsupported`; this does not
mean the V1 archive schema or import contract is unsupported on Windows.

Tool Packs V1 never install executable content. A future combined Tools+Skills
pack or runtime plugin installation needs a new package schema, security review,
ADR, dependency and permission model, signature/provenance policy, and explicit
installation UX. It is not an extension field that V1 readers may ignore.

The governing architecture decision is
[ADR-107](../../backlog/decisions/107-portable-tool-use-packs.md).

