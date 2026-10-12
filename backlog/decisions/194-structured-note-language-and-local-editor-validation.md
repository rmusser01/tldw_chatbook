# ADR-194: Portable note language and local structured editing

Date: 2026-09-27
Status: Accepted by the user with written-spec approval on 2026-09-27
Task: [TASK-33096](../tasks/task-33096%20-%20Design-structured-document-validation-and-formatting-across-Chatbook-and-server-Notes.md)
Spec: [Structured document editing](../../Docs/superpowers/specs/2026-09-27-structured-document-editing-design.md)
Extends: ADR-027, ADR-029, ADR-073 without changing their save-authority boundaries

## Context

Chatbook and tldw_server Notes need live YAML, JSON, and JSONL syntax feedback and
explicit formatting. Their different runtimes and existing source transformations
make a parser added directly to each widget insufficient. The user chose local
engines, whole-document validation, continued saving of invalid drafts, and a
language selection that travels with the note.

The review found extension-independent frontmatter hiding, a lossy Chatbook Notes
sync adapter, strict server sync payload fields, source-wrapping exports, and no
demonstrated native undo integration for web formatting. Empty note persistence is
also currently rejected by the server. These are design constraints, not evidence
that runtime integration has been qualified.

## Decision

1. Treat nullable `content_language` as canonical note metadata, separate from
   title-generation language. Preserve it through note persistence, conflicts,
   drafts, duplication, backup, and versioned portable export/import.
2. Add a negotiated Notes sync payload version. Omitted fields in old updates
   preserve stored language; explicit null resets the mode. Mixed-version paths
   unable to preserve metadata expose unsupported/pending state rather than
   dropping it. Private note language remains protected with its note content.
3. Keep validation and formatting local: a browser engine and a Python engine
   with shared versioned fixtures, bounded terminable workers, and no save or
   network authority. No server validation endpoint or language-server runtime.
4. Validation observes immutable source snapshots. Only results matching the
   current document, session, language, and draft revision can affect the UI.
5. Formatting proposes a source-preserving edit, admitted under the editor's
   current authority and applied as one normal undoable transaction. Preserve
   numeric/string tokens and YAML semantic/source constructs or decline.
6. Keep source content exact across accepted storage/transport operations;
   reject unsupported content explicitly. Escape at rendering boundaries.
   Raw source downloads and portable note envelopes are different outputs.
7. Syntax errors never add save vetoes. Existing persistence failures remain
   visible; retain empty drafts locally when the server cannot persist them.
8. File Notes retains its disk authority and representation rules. Structured
   files expose full source and cannot inherit Markdown frontmatter stripping.

## Alternatives

| Alternative | Reason not chosen |
| --- | --- |
| One server validation API | Adds connectivity, latency, and disclosure requirements for local drafts. |
| Language-server integration | Adds runtime/install complexity beyond syntax checks and explicit formatting. |
| Device-local language preference | Does not meet the user's portable selection requirement. |
| Parse objects and serialize them to format | Can lose numeric precision, duplicate entries, comments, or scalar representation. |
| Block saving on invalid syntax | Prevents preserving normal unfinished work. |
| Extend the existing sync payload without negotiation | Existing validators reject unknown fields and old writers may erase metadata. |

## Consequences

This is coordinated cross-repository work, with migrations, capability negotiation,
and transport qualification preceding editor adoption. Existing note/file save
owners remain separate. Engine, history, and source-preservation tests are release
gates; dependency selection follows those experiments. This ADR does not approve
new editing surfaces, external commands, or a wholesale editor replacement.

The complete language rules, bounds, compatibility behavior, and acceptance matrix
are in the linked spec. Written-spec approval was received on 2026-09-27;
implementation planning may proceed. No existing accepted ADR is superseded.

### Immutable publication origin scope (F4)

Existing local publication-intent payloads carry the exact originating principal and workspace, including explicit null. Drain compares both before copying source into a scoped outbox. Missing historical proof remains pending; an active profile cannot supply historical authority. This local metadata never enters wire payloads and adds no schema or transport.

### Authenticated scope and complete materialization (F4 review correction)

Notes publication/application captures the existing runtime target/client and authenticates the exact requested principal through Sync profile user_id and the exact dataset personal/workspace scope through its restore manifest. Unknown/null principals, absent target evidence and missing/ambiguous manifests retain pending intent. Reads and mutation use the same pinned client with current-context checks; this adds authenticated reads rather than a new authority owner.

Complete backlink fields must be validated and retained through the existing upsert materializer. A target without backlink storage refuses non-null values before writes or mirror advancement; null remains lossless. No canonical schema or metadata side store is added. Supplied organization receipt origins bind immutable request fingerprints; ready saves capture the same existing intent in their transaction. Unproved historical origins retain source and expose a bounded scoped profile reason/count, which unrelated successful sync cannot clear.

### Device-private draft recovery (F5)

Database Note drafts reuse the runtime-owned NotesDeviceStateStore through a narrow
recovery facade and a dedicated schema-v3 table. They are not filesystem sync
journal entries and never enter portable backups, diagnostics or sync payloads.
Rows hold exact source and language at one revision, with the original optimistic
base version. Matching acknowledgements or explicit discard remove them; older
acknowledgements cannot remove newer drafts. Storage failure fences navigation.
The local Library port scopes drafts to its captured canonical database path
identity, local principal and explicit global (null) workspace, never the selected
remote profile. Moving that database requires explicit future recovery mapping.
Facade construction does no I/O and neither starts watchers nor creates sync roots.

### Portable source and note import contracts (F6, Rulings 35/36)

The complete portable note marker is `format_version: "tldw.note.v1"`; title and
content are strings and `content_language` is required, including explicit null.
Unknown baseline language cannot be exported as portable metadata. Source remains
a string; document values are never parsed to manufacture a note body.
The existing server import discriminator adds `portable_note` and `source`, so
older import schemas refuse declared contracts before mutation. Versioned batches
are validated before any writes. Legacy JSON/YAML note containers retain their
existing reader paths. C raw import is explicit `source_mode=False` by default;
JSONL, which has no legacy container reader, is source-only. Filename extensions
supply mode; filename-less source uses a safe title and makes no mode promise.
Empty or storage-rejected bodies fail explicitly without title substitution.

New Notes Chatbooks use verified archive 4.0 and identity-based JSON carriers,
including null-language Notes. Combined archives retain the Canvas extension;
conversation-only Canvas remains 3.0. Readers 1.0–3.0 remain available, without
claiming their Markdown carriers retain recoverable original boundary whitespace.
Approval/effects fingerprints include supplied mode, preserving the historical
fingerprint for legacy omission. No new transport, store or import service.

### Python source-engine qualification (E2 / TASK-34654)

Accept `ruamel.yaml==0.19.1` (MIT, pure Python, no default transitive dependencies)
as a base-install parser for the structured editor. Qualification is recorded in
[the Python engine report](../../Docs/superpowers/reviews/2026-09-27-document-engine-python.md)
before packaging adoption. Use only lazily imported instance scan/parse APIs,
without application constructors, alias expansion, or load/dump formatting.
The candidate's generic round-trip output demonstrably changed negative zero,
overflow exponents, and high-precision decimal spelling, so it is not adopted.

JSON uses bounded source lexing and strict stdlib validation. YAML edits source
whitespace and verifies raw tokens, comments, and the symbolic event graph.
Default YAML 1.2 Core resolution and explicit consistent 1.1/1.2 versions govern
scalar-key identity. Mixed versions, custom/unqualified tags, complex/alias keys,
indentless sequences, unqualified merge targets, and multiline-scalar layouts
requiring scalar-token edits explicitly decline. Nontrivial commented streams,
anchors/merges, block scalars, and indented sequences format successfully.

The pure engine enforces source/depth/diagnostic bounds and cooperative deadline
checks only; it has no heap ceiling or hard interruption. The existing planned
worker boundary still owns process termination and lifecycle. No UI, persistence,
network, filesystem authority, or new store is introduced by this dependency.

### Isolated worker launch (E4 / TASK-34655, Ruling39)

Use a fixed package-owned Python script under `sys.executable -I`, with an
explicit empty environment, neutral cwd and closed inherited descriptors. The
bootstrap imports the pure `Document_Editing` package from its installed parent,
without importing the application/package initializer. Exchange bounded JSON
frames for one snapshot per child; terminate, reap and close before replacement.
No document path, credentials, arbitrary launcher option or parent-environment
mutation is admitted. This separates application authority, not OS user privileges.

Actual module-entry mounted TldwCli qualification failed spawn at captured stderr
fileno=-1. A test-only fd-backed control inherited a synthetic secret and replayed
app.py/config as __mp_main__ (1782 modules,1.949s). Therefore multiprocessing spawn
cannot satisfy this boundary. No production stderr workaround or worker pool is
adopted. Installed package paths, copy/startup costs, cancellation and actual
termination remain required E4 qualification, separately from editor adoption.

### Secondary Research Quick Notes source preview (I2, Ruling43)

The ResearchQuickNote read contract and its local/server row mappings omit mode
presence. Its secondary preview therefore displays exact markup-disabled source
for every note and is labeled Source preview. The raw TextArea and canonical save
contract stay unchanged; omission is never manufactured into null metadata.
This narrowly guarantees source-safe preview without adding a second metadata
owner. It costs rendered Markdown preview in this secondary surface until its
read contract carries qualified mode presence; the main Library Notes retains
its existing Markdown preview for null/Markdown modes.

### File editor representation mapping (I2, Ruling44)

FileNotesService already presents homogeneous CRLF disk source as LF in its
incumbent TextArea and serializes it back with the captured BOM/newline profile.
The File-only editor adapter maps a qualified CRLF format proposal to that LF
view before native admission. A mapped equality is an unchanged result and makes
no edit, dirty revision, autosave or history entry. Captured original request
facts, result identity/check qualification and synchronous source guards remain;
Database Notes, parser, generic bridge and file serializer are unchanged. Mixed
newlines and lone CR keep their existing read-only representation restrictions.

### Database native representation fence (I2, Ruling45)

If Textual's native document cannot round-trip the canonical Database body
exactly, the body view is restricted to escaped source and native body edits,
checks and formatting are unavailable. Title, mode and save continue through
the canonical session using its original body. Queued Changed and pre-recompose
flushes cannot import the normalized widget string. Existing copy/export read
that same canonical body. Mode selection never removes this fence. This costs
body editing for unsupported representations until a qualified adapter exists;
it adds neither persisted body ownership nor Database newline normalization.
Homogeneous LF/CRLF and lone CR remain usable when native round-trip is exact.
