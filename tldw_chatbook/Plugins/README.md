# Native package inspection

`inspect_package(root: Path, *, dialect=None)` reads an existing canonical absolute
package directory. It does not execute commands, launch MCP, fetch schemas,
install a package, or grant trust. Use a worker for this bounded filesystem work.
`PackageInspection` is an immutable record; component definitions are canonical
JSON strings and collections cannot be mutated in place after hashing.

## Authoring

Use Agent Plugins 1.0.0 `plugin.json`, immediate `skills/<name>/SKILL.md` children,
and optional root `mcp.json`. Unknown manifest fields are diagnosed and ignored.
Invalid skills and MCP entries are isolated from valid siblings. A missing or
unsupported portable schema rejects the portable interpretation.

`extensions.io.github.rmusser01.chatbook` requires integer `version: 1`. Its only
optional fields are `commands`, `rules`, `agents`, `hooks`, `requires`, and
`variables`. Paths are contained package files; there is no implicit extension
folder scan. Commands require `name`/`description`, rules `name`/`mode`, and agents
`name`/`description`/`tools` YAML frontmatter. An empty agent tools array means no
tools. Hook files use the closed v2 envelope from ADR-163. Tool/model mappings
are explicit blockers where they still need host configuration.

Dependencies use typed local IDs such as `skill:review` and `hook:protect-writes`.
Missing dependencies, duplicate IDs, and cycles block affected components.
Unrecoverable recognized extension constraints block activation conservatively;
unrelated unknown extension namespaces do not. Native variables require closed,
typed declarations, cannot override `PLUGIN_ROOT`/`PLUGIN_DATA`, and secret
variables cannot supply package defaults. Required unset variables block use.

Every parsed component starts disabled, with parsed-only evidence. Support is a
format interpretation result, not runtime compatibility or permission. Hook
transformer/guard effects and MCP transport declarations remain explicit for
later runtime owners. No publication into live skills/tools occurs in this
foundation module.

## Snapshot operation and identity

`materialize_package(source, destination)` requires an absent destination under
an existing canonical parent. It rejects all unsafe or incomplete source captures
before writing, creates a private destination, writes regular files, and verifies
its resulting content digest. Existing destinations are never overwritten.
Internal regular-file links become regular files. Directory/external/broken
links, special files, reparse points, and ambiguous platform spellings are refused.
Case-folding and Unicode normalization collisions are conservatively rejected.

- `content_digest` binds sorted relative file paths, exact bytes, and executable
  bits of the materialized representation. It reproduces on destination inspection.
- `source_digest` additionally binds normalized internal link targets.
  `link_targets` retains that provenance after links become regular files.
- `effective_digest` binds native normalized identity, component definitions,
  instruction-file hashes, dependencies, variables, blockers, and interpretation.
  Ignored portable fields alone do not change it. Both content and effective
  digests are review inputs; neither grants trust.
- `source_identity` names the captured source; `materialized_identity` names the
  verified destination when materialization succeeded. Evidence timestamps are
  deliberately outside deterministic identity.

Inspection retains safe portable components after narrow path errors, but an
incomplete capture has no content/source digest and cannot be materialized.
Source and destination mutations are checked through anchored descriptors and
identity comparisons. This is a package acquisition boundary, not a sandbox
against arbitrary same-user code or a substitute for the later trust/lease owner.

Limits: 256 KiB/depth 32 per definition JSON; 100 MiB expanded, 10,000 files,
10 MiB/file, path depth 32; 512 normalized components; 64 native hooks. Traversal
also caps directories at `10,000 * 32`, bounding empty-directory trees. Every
visited entry consumes its file/directory budget before filename validation;
unreadable entry metadata consumes the file budget. Retained filesystem diagnostics
share a 10,000-entry cap. Files are read in 64 KiB chunks and counted before retention. Frontmatter is bounded to
256 KiB and rejects aliases, duplicate fields, and non-JSON values. JSON rejects
nonfinite numbers (including exponent overflow) and non-UTF-8-representable strings
or keys before normalization. Numeric timeout bounds precede float conversion.

## Qualification boundaries

POSIX descriptor APIs and no-follow support are required. Automated filesystem
checks currently run on macOS; Windows fails closed as `platform_capture_unqualified`.
Windows reparse behavior is not claimed qualified. Vendor markers retain candidate
identity and ambiguity but are explicitly unsupported/unqualified until a vendor
adapter is delivered. A valid native root defaults to its native interpretation;
independent vendor roots require an explicit choice. An inline OpenAI overlay
replaces the compatibility-file overlay wholesale.

See [ADR-162](../../backlog/decisions/162-managed-agent-plugins.md),
[ADR-163](../../backlog/decisions/163-expanded-console-hook-runtime.md), and
[fixture provenance](../../Tests/Plugins/fixtures/PROVENANCE.md).

Portable remote MCP definitions reject userinfo or fragment delimiters in URLs,
case-insensitive duplicate headers, invalid HTTP field characters, and leading or
trailing field whitespace. Loopback HTTP and ordinary HTTPS definitions retain
the same inspection boundary; syntax validation never grants network access.
