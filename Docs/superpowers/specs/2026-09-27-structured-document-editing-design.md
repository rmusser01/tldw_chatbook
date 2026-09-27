# Structured document validation and formatting

Date: 2026-09-27
Status: Revised design; awaiting written-spec approval
Task: [TASK-33096](../../../backlog/tasks/task-33096%20-%20Design-structured-document-validation-and-formatting-across-Chatbook-and-server-Notes.md)
ADR: [ADR-194](../../../backlog/decisions/194-structured-note-language-and-local-editor-validation.md) (proposed)
Applications: tldw_chatbook and tldw_server web Notes

## Purpose and approved scope

Users editing YAML, JSON, or JSONL receive syntax feedback without losing
incomplete work. Formatting is an explicit, reversible document edit. Validation
runs locally in each application, with common conformance fixtures and behavior.
The selected language of a database note travels with that note.

The user approved whole-document checks, syntax validation rather than style or
application-schema linting, normal save/autosave behavior in the presence of
syntax errors, local engines, and portable language metadata. On review, the user
approved incorporating the source-preservation, compatibility, and undo findings
below into release acceptance criteria before selecting formatter libraries.

Initial editing surfaces are Chatbook Library Database Notes and File Notes, and
the server's main web Notes source editor. Other Notes editing entry points must
preserve language metadata and raw source; integration planning must inventory
them, including the Notes dock. A structured note must not silently enter a
Markdown conversion path through a secondary editor. The Console Workspace Files
modal inspected in this checkout is read-only; this feature does not grant it
editing authority. Prompt/configuration-specific editors and new file browsers
are separate adoption work, not an implicit expansion of this release.

Excluded: Markdown fenced blocks/frontmatter linting, style-rule configuration,
application schemas, AI repair, format-on-save, language servers, external linter
commands, and a server validation API. Markdown and plain-text editing retain
their existing behavior.

## Source review findings and required corrections

These are source-level findings, not claims of completed runtime qualification.

| Finding | Required design response |
| --- | --- |
| Chatbook `Notes/file_notes_service.py::_parse_opened` strips leading frontmatter without considering the extension. | YAML/JSON/JSONL expose the complete source. Preserve existing Markdown behavior; changing the editor language cannot strip or restore bytes. |
| Chatbook `Sync_Interop/domain_adapters/notes_m1.py::_validated_note_payload` truncates and HTML-escapes content. | Qualify and repair participating source transport paths so they preserve exact content or explicitly reject it, never silently truncate, trim, or entity-encode it. Escape at rendering boundaries. |
| Server `core/Sync/v2/models.py::validate_notes_note_upsert_payload` rejects unknown fields. | Introduce a versioned Notes payload extension with negotiated support; do not append a field to the locked existing payload. |
| Web Notes updates content through React state; AI undo restores a previous snapshot. | Prove normal undo/redo around formatting. A snapshot restoration button does not satisfy editor history requirements. |
| Existing note exports wrap content in Markdown or a JSON note envelope. | Distinguish raw source download from portable note export. Neither silently substitutes for the other. |
| Server create/update schemas and sync reject empty content. | Preserve recoverable empty drafts and expose unsaved status. This release does not broaden the persisted-empty-note contract. |

Relevant Chatbook governance: ADR-027 Database Note session ownership, ADR-029
File Notes disk authority, ADR-073 representation-preserving sync, and
`backlog/docs/design-language.md`. This feature preserves their ownership and
conflict boundaries. ADR-194 adds the language/validation contract rather than
merging the different save authorities.

## Editing behavior

The editor exposes a language selector, independent save and validation status,
a Problems list, and Format document. File extensions `.yaml`/`.yml`, `.json`,
and `.jsonl` supply the initial language. A per-open-file override is temporary
and clearly labeled; it neither renames the file nor writes metadata into it.
Database notes use their persisted `content_language`; null retains the existing
default. Selecting a language never changes the source text.

Check after 400 ms of typing inactivity and on initial open or language change.
This debounce is scheduler behavior, not an animation token. Immediately after
an edit, retire the previous success result and mark the current revision pending.
Expose pending/checking, checked, unsupported, resource-limited, and failed states.
Only a completed check for the current revision can say "No syntax errors".
Save status remains independent: "Saved · JSON · 1 error" is valid feedback.

Diagnostics include severity, stable category, brief message, and source range.
The Problems list can move the caret to the affected source without taking focus
during background updates. It announces that further diagnostics were omitted if
the display cap is reached; it never claims an exact total when checking stopped
early. Provide a manual Check action using the same resource limits.

Format is unavailable for syntax errors, duplicate-key warnings, unsupported
constructs, incomplete checks, excerpts, or read-only documents. Explain the
reason inline. A successful format is one edit in normal undo/redo history and
then follows existing autosave behavior. No-op formatting creates neither a dirty
revision nor an undo entry. Preserve selection and viewport where possible, with
caret mapping into the formatted source rather than resetting to the beginning.

Structured notes use source editing and an escaped code preview. Disable
Markdown/WYSIWYG conversion and Markdown insertion controls for those documents.
Restoring an old per-device WYSIWYG preference must not run a conversion before
the note's language is loaded. Secondary Notes entry points either use the same
source-safe editor behavior or offer opening the main source editor without
transforming the content.

Use the Chatbook design tokens and existing component patterns for statuses,
controls, focus, and spacing. Problems must remain keyboard-accessible at compact
sizes, have meaningful labels, and not depend on color alone. Do not add screen
bindings that shadow terminal conventions or global actions.

## Language contract

| Language | Validation and formatting rules |
| --- | --- |
| JSON | Strict JSON grammar; reject comments, trailing commas, NaN, and Infinity. Accept any top-level JSON value. Duplicate object names are warnings that block formatting. Preserve original number and string tokens and member order when reindenting. |
| JSONL | Apply the JSON rules independently to each physical record line. Accept any JSON value per line. Blank/whitespace-only records are errors; a single final line terminator is not an extra blank record. The empty file is zero records. Never wrap records in an array, join records, or expand one record across lines. |
| YAML | Default to YAML 1.2 Core rules. Honor explicit supported 1.1/1.2 directives, keeping their spelling and semantics. Support multiple documents, comments, anchors, aliases, scalar styles, and document markers. Duplicate mapping keys are errors. Parse custom tags without executing constructors; constructs whose preservation/interpretation cannot be qualified are unsupported for formatting, not automatically malformed syntax. |

Duplicate checks compare decoded JSON key strings within the same object,
including escaped spellings of the same name. YAML duplicate detection follows
the selected version's key semantics. Do not mistake merge-key overrides for
ordinary duplicate declarations. Qualify complex YAML keys or explicitly report
the check as unsupported rather than guessing equality.

An empty JSON draft is invalid; empty YAML is a valid empty stream. Neither
result determines whether the existing note persistence API accepts that draft.
An indented YAML document can be syntactically valid while expressing the wrong
structure: the UI says "No syntax errors", never "Correct configuration".

Use two-space block indentation for JSON and supported YAML layouts. JSONL uses
compact values per line. Preserve strings, numeric lexemes, key order, comments,
YAML tags/anchors/aliases/directives, and scalar content. Never expand aliases
or flatten merges as a formatting technique. A formatter that cannot preserve a
construct declines with a reason. Formatting is deterministic and idempotent
within each engine; identical whitespace between the two engines is not required.

Validation observes source-representation metadata as well as visible text.
File Notes currently hides a UTF-8 BOM and normalizes CRLF for editing, so the
diagnostic adapter must not report a BOM-containing JSONL file as fully valid.
Preserve original newline convention and final-newline presence on write under
existing file rules. JSONL's forbidden BOM is reported; formatting does not
silently remove it. Existing mixed-newline and unsafe-file restrictions remain.

## Source preservation and formatting admission

Text validation is read-only. Keep literal `<`, `>`, `&`, Unicode, whitespace,
and escape sequences through save, reopen, synchronization, recovery, duplication,
and portable export/import. Where an existing storage boundary forbids a value
or size, reject it explicitly and preserve the draft rather than normalize or
truncate it. Content must not pass through HTML sanitization on its way to storage;
rendering and preview boundaries retain escaping/sanitization appropriate to them.

Do not implement formatting as a generic object parse followed by serialization.
Numbers such as `9007199254740993`, `1e400`, `-0`, and high-precision decimals must
retain their original tokens. Likewise YAML block scalar content/chomping and
quoted strings must not change value. Maintain source-aware representations and
verify format invariants before applying the result. A failed invariant leaves
the original draft untouched and produces an actionable formatter failure.

Formatting admission captures document identity, authority/profile, session
generation, draft revision, language revision, and editor editability. Revalidate
all of them on completion. Switching language, typing, changing notes, entering
a file conflict, or losing write authority makes the result obsolete. Never
rerun or apply an obsolete format automatically. Autosave and external-file hash
checks retain their current authority; formatting does not write files directly.

## Local engines and resource ownership

Each application owns a small engine with no UI, database, network, or filesystem
write dependencies. Input is source text, language/version, representation facts,
and request identity. Output is a typed check result or a proposed replacement
plus diagnostics. A separate editor adapter owns scheduling, display, navigation,
and revision-safe edit application. Do not create a shared cross-language runtime
package merely to share the interface; share versioned fixtures and rules.

Use browser workers in the web client and a lazily started, isolated Python
worker process in Chatbook. A running parser must be terminable on a deadline;
ignoring a future or cancelling a UI worker alone does not provide that guarantee.
Workers receive document text only, no credentials or document-path authority,
and never interpret executable YAML constructors. Do not log source snippets or
persist diagnostics. Worker failure must not affect saving or draft recovery.

Initial common limits are 1 MiB of UTF-8 source, nesting depth 100, 100 displayed
diagnostics, and a 2-second processing deadline per request. Enforce size before
dispatch and depth during parsing; do not fully construct an unbounded object
graph before checking depth. Do not expand aliases. Inputs over limits get a
resource-limited result; manual Check and Format obey the same limits.

Keep at most one executing request and the latest pending request per active
editor. Retire superseded pending snapshots and terminate expired workers before
replacement. Account for inactive mounted editors so they do not each maintain
an idle process. Profile document copy/encoding and worker startup costs in both
apps; source transfer itself must not cause typing stalls. Limits are initial
product choices to qualify with the adversarial fixtures, not measured guarantees.

Diagnostic positions use zero-based Unicode scalar offsets into the exact editor
snapshot, with end-exclusive ranges. Adapters translate JavaScript UTF-16 offsets
and parser line/column coordinates into this representation. The UI displays
one-based logical line/column values; tabs are characters, not terminal cells.
Test emoji, combining characters, CRLF, and EOF positions. Engine messages may
differ; shared fixtures assert validity, stable categories, and relevant spans.

## Portable language metadata and compatibility

Add nullable `content_language` to the canonical note model with known values
`text`, `markdown`, `yaml`, `json`, and `jsonl`. Null means existing default
behavior, not content sniffing. This field is unrelated to the existing title
generation `language` hint. Add migrations and carry the field through note
reads/writes, revisions, drafts/recovery, duplicates, conflicts, backups,
portable note exports/imports, and every participating sync producer/materializer.

Changing language is a note mutation under existing optimistic version control.
It must mark metadata dirty without rewriting content and be captured with the
associated draft when saving. API updates distinguish omitted (preserve) from
explicit null (reset). Never send a stale cached default alongside an unrelated
body edit. Conflict resolution includes language; choosing source text and a
language from different versions must be an explicit user choice.

Extend the existing Notes sync contract with a newly negotiated payload version.
Reserve its exact number only after inspecting both repositories during planning;
the rule is new-version negotiation, not a speculative fixed version number.
Supported new-version snapshots carry language, including explicit null. Existing
version payloads keep their current field set. New receivers preserve existing
language when applying old-version updates; old receivers must not silently
accept and discard new-version fields. Language belongs inside the note's
protected payload for private/encrypted sync, with no new cleartext disclosure.

Do not assume unknown-field preservation for older encrypted/full-snapshot
writers. If the negotiated path cannot maintain language end to end, show
metadata sync as unavailable and retain local metadata and its pending state;
never describe it as synchronized or strip it to make a request succeed. Existing
content-only synchronization may continue only if it provably preserves the
local pending language and cannot clear a remote value. Otherwise pause that
note's affected publication with an actionable compatibility explanation while
keeping its local draft/save usable.

For a directly server-backed note whose server cannot persist the field, allow
a clearly labeled temporary language selection, scoped to that note/session.
This does not claim portable persistence. Existing recognized language metadata
must not be reset simply because another endpoint omits it. Preserve future
unknown language values through read/edit/write without interpreting them as
known modes; display unsupported mode and disable structured formatting. Only
an explicit user selection replaces that value.

## Empty drafts, source downloads, and note exports

Syntax errors never add a save veto. Existing authority, conflict, size, and
persistence errors remain real save errors. In particular, the existing server
does not persist an empty note body. Retain that empty draft through navigation
and restart using the existing scoped recovery mechanism, and show "Draft kept
locally; empty content cannot be saved to this server" rather than "Saved".
Keep the server baseline and do not publish a deletion, replace it with whitespace,
or discard the draft. Once content is nonempty, normal save can resume. Local
File Notes follows its existing empty-file save capability.

"Download source" exports only the document content with the corresponding
extension. It adds no title, frontmatter, note envelope, or language marker.
"Export note" is a portable envelope/archive carrying title, content, language,
and existing note metadata. Importing that envelope restores language and exact
content. Raw source imports infer language from the extension. Plain raw text
without a filename cannot carry metadata by itself; do not promise otherwise.

Portable archive schema changes must be versioned and old readers must reject
unsupported versions clearly rather than silently discard the language. Existing
Markdown export remains an explicit conversion/export choice for Markdown notes,
not the automatic container for structured source.

## Release acceptance criteria

These checkboxes describe future implementation evidence, not completed work.

- [ ] AC1: All initial surfaces provide whole-document YAML/JSON/JSONL checks, language selection, separate save/check status, and keyboard-accessible problem navigation.
- [ ] AC2: Invalid syntax never adds a save veto; empty and otherwise rejected drafts remain recoverable with truthful unsaved status across navigation/restart.
- [ ] AC3: Exact-source fixtures survive each participating local/server save, sync direction, duplication, recovery, and portable export/import path; rejected values fail explicitly without modifying the draft.
- [ ] AC4: YAML files retain visible document markers and every document; Markdown frontmatter handling and file authority/hash-conflict checks do not regress.
- [ ] AC5: Formatting preserves numeric/string values and required YAML source constructs, is idempotent, refuses duplicates/unsupported constructs, and is a no-op when already formatted.
- [ ] AC6: In each real editor, type → format → type → undo → undo → redo preserves intervening edits and prior history; stale or read-only results never apply.
- [ ] AC7: Language survives all supported metadata paths; language-only updates persist; omitted/null and conflict semantics are verified against real stores and APIs.
- [ ] AC8: Mixed-version API/sync/archive peers either preserve metadata or expose the exact unsupported/pending operation. No silent dropping, clearing, downgrade, or false synchronized state occurs, including private sync.
- [ ] AC9: Both engines pass the same versioned language fixtures, including strict JSON, escaped duplicate names, JSONL blank/final lines, YAML directives/streams/merges, and unsupported custom constructs.
- [ ] AC10: Diagnostic jumps land on the intended logical source range for Unicode, tabs, CRLF, EOF, and language switches; partial/failed checks cannot display success.
- [ ] AC11: Oversized/deep/alias-heavy input, timeout, worker crash, rapid typing, and repeated navigation do not freeze the UI, accumulate work/processes, or lose drafts.
- [ ] AC12: Structured notes cannot be transformed by Markdown/WYSIWYG paths, including restored preferences and secondary note entry points.
- [ ] AC13: Source downloads contain only exact source; note exports preserve language metadata and import it correctly; existing file newline/BOM restrictions remain honest.
- [ ] AC14: No source text enters new logs, telemetry, validation network calls, or durable diagnostic stores; rendering stays escaped and custom tags never execute.
- [ ] AC15: Targeted tests, static checks, and live TUI/browser verification qualify the integrations; passing engine tests alone is insufficient.

## Verification and delivery boundaries

Use real temporary SQLite databases and real API serialization/materialization for
metadata and round-trip tests. Exercise both sync paths found in Chatbook: the
server-trusted Notes M1 adapter and private encrypted content publication. Include
the literal string `a&b <tag>`, trailing whitespace, escaped control characters,
large numeric lexemes, non-BMP Unicode, YAML aliases/comments/block scalars, and
CRLF/BOM file profiles. Never infer transport correctness from a mocked repository.

For UI verification use mounted real widgets plus native typing/history tests in
the browser and TUI, not just state setters. Include IME composition (defer checks
until composition commits), narrow layouts, screen-reader announcements, switching
documents mid-check/format, and external edits mid-autosave. Follow
`backlog/docs/lessons-testing-evidence.md` and `lessons-live-verification.md`.
Run targeted checks only unless the user explicitly authorizes a full suite.

The architecture stays one coordinated spec, implemented as independently reviewed
units in dependency order: source transport/metadata compatibility, qualified
engines, editor integrations/history, and cross-application qualification. These
are delivery boundaries, not an implementation plan or newly invented task IDs.
The detailed plan is written only after this spec is approved. Formatter library
and widget choices must earn their place through AC5/AC6/AC9 experiments before
feature integration; no dependency or wholesale editor replacement is preapproved.

## Alternatives and trade-offs

Local engines keep feedback available offline and avoid transmitting drafts for
validation, at the cost of two implementations and shared fixture maintenance.
A server validation API would centralize code but add latency/connectivity and
disclosure requirements. Language servers would add installation and lifecycle
costs without serving the first release's limited syntax/formatting scope.

Portable language metadata costs migrations and compatibility work but prevents
users repeatedly selecting modes across devices. Device-only preferences were
considered and rejected by the user. Library-specific exact error messages and
identical formatter whitespace across runtimes are intentionally not contracts.

## References and review record

- [Python JSON interoperability](https://docs.python.org/3/library/json.html#standard-compliance-and-interoperability)
- [JavaScript JSON parsing and precision](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/parse)
- [JSON Lines specification](https://jsonlines.org/)
- [YAML source-aware parsing and document options](https://eemeli.org/yaml/)
- [ruamel.yaml round-trip behavior](https://yaml.dev/doc/ruamel.yaml/detail/)

2026-09-27 self-review: explicitly separated draft retention from persistence,
source downloads from portable envelopes, syntax validity from schema correctness,
and unsupported checks from success. Qualified old-peer preservation claims,
included both sync modes and source-representation metadata, and made formatter
fidelity and normal undo release gates. No product code has been changed and no
runtime acceptance criterion is claimed complete.
