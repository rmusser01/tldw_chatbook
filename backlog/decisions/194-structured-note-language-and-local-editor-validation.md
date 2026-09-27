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
