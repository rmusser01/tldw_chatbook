# ADR-053: Prompt Variable Grammar and Guarded Insertion

Status: Accepted
Date: 2026-08-10
Related Task: [TASK-199](../tasks/task-199%20-%20Templated-variables-in-prompt-bodies.md)
Supersedes: N/A

Allocation note: ADR-053 was allocated after TASK-198 merged at `d5d0b419c`.
ADR-051 was the highest number on remote refs, while an active worktree had
already reserved ADR-052. An all-ref and all-worktree numeric sweep therefore
made 053 the first unreserved number.

## Decision

Use one deterministic, single-pass Prompt variable grammar and one shared
Prompt Variables dialog for exact `/prompt` resolution, the Console Prompt
picker, and Library `Use in Console`. The grammar and application request live
outside Textual. The Console composer and current session remain the only
authorities allowed to mutate the active draft or System prompt.

A variable is a single-braced placeholder whose name matches
`[A-Za-z_][A-Za-z0-9_]*`. Names are exact and case-sensitive. A name may contain
at most 64 characters, and one application may contain at most 64 unique names.
A syntactically valid placeholder that first exceeds either bound makes the
render plan invalid; it is not truncated, discarded, or reclassified as
literal text. The dialog then offers only the unchanged-placeholder path or
Cancel.

Lexing proceeds left to right exactly once:

1. `{{` emits one literal `{` and consumes both characters.
2. `}}` emits one literal `}` and consumes both characters.
3. A remaining `{` begins a variable only when the text through the next `}`
   is a valid variable name. Otherwise that `{` is emitted literally and the
   scan continues with the next character.
4. Any other character, including an unmatched single brace, is literal.

Rendering walks the already-tokenized plan exactly once. It substitutes each
variable token from the ephemeral value map and decodes literal-brace tokens.
Rendered values are appended as opaque text: they are never lexed again.
Escape decoding likewise cannot expose another placeholder.

### Grammar truth table

`X` below is the supplied value for `name`. “Variables” lists the extracted
names, not text that merely resembles a placeholder after escape decoding.

| Input | Variables | Rendered result | Reason |
| --- | --- | --- | --- |
| `{name}` | `name` | `X` | One valid placeholder. |
| `{Name}{name}` | `Name`, `name` | their two exact values | Names are case-sensitive and adjacent variables are independent. |
| `{name}{name}` | `name` once | `XX` | One shared value is reused for duplicate occurrences. |
| `{{name}}` | none | `{name}` | Both brace pairs are escapes; decoded text is not reparsed. |
| `{{{name}}}` | `name` | `{X}` | The outer pairs are literal escapes and the remaining inner pair is a variable. |
| `{{name}` | none | `{name}` | The opening escape consumes both braces; the trailing single brace is literal. |
| `{name}}` | `name` | `X}` | The final unmatched brace is literal. |
| `{}` | none | `{}` | Empty names are not variables. |
| `{1name}` | none | `{1name}` | The first character is invalid. |
| `{first-name}` | none | `{first-name}` | Hyphens are invalid in names. |
| `{ name }` | none | `{ name }` | Whitespace is invalid in names. |
| `{"key": "value"}` | none | unchanged | Ordinary unescaped JSON object braces are invalid variable delimiters and remain literal. |
| `{"key": "{name}"}` | `name` | `{"key": "X"}` | Ordinary JSON structure remains literal while its explicit valid placeholder is active. |
| `{{"key": "{name}"}}` | `name` | `{"key": "X"}` | JSON structure uses escaped literal braces while its explicit placeholder remains active. |
| `<customer>{name}</customer>` | `name` | `<customer>X</customer>` | XML syntax is otherwise literal. |
| `{outer {name}}` | `name` | `{outer X}` | The invalid outer `{` is literal; scanning continues and finds the explicit inner placeholder. |
| `{name` or `name}` | none | unchanged | Unmatched single braces are literal. |
| value `"{other}"` for `{name}` | `name` | `{other}` | Braces introduced by a value are opaque and never reparsed. |

Across active lanes, first occurrence is computed from System text followed by
User text. Each unique name appears once in the dialog with a literal lane-use
label, and one entered value is shared by every occurrence in both lanes.
Blank values are valid. Temporarily disabling a lane removes its variables from
the active list without discarding their mounted ephemeral values, so toggling
that lane back on restores the same entries.

When a System lane exists, the shared dialog displays this exact checkbox:

`Replace the current session System prompt with this System lane`

It is off by default. A User lane is active according to the destination flow.
If System replacement is off and there is no applicable User lane, Apply and
`Use original placeholders` are disabled with `Select a lane to apply`; Cancel
remains enabled. The ordinary Apply action renders active lanes. The secondary
action applies the selected source lanes without interpolation so an existing
literal `{name}` remains expressible.

The dialog describes its destination truthfully. For replacement flows, the
complete segment-aware composer snapshot is captured at `/prompt` dispatch or
picker opening, before any asynchronous resolution or user selection. That
exact snapshot and its one-way fingerprint are threaded through the flow; the
application may replace all of its draft segments only if the live snapshot
still compares equal. Pending attachments, which are not composer draft
segments, remain outside this transaction.

- exact `/prompt` and picker flows replace the entire Console composer snapshot
  captured when the flow opened;
- picker use over ordinary text says `Replace the current Console draft`, not
  that the text is a slash command;
- Library use says `Append to the current Console draft` and captures the
  settled active composer snapshot only when Console consumes the handoff.

The dialog never mutates Console state. It produces a validated,
memory-only `PromptVariableApplication` containing only the selected rendered
or original lane text, lane flags, destination (`replace_snapshot` or
`append_active`), target session identity, optional composer and System
fingerprints, and monotonic creation and expiry data. It contains no raw value
map, separately retained values, or additional copy of the source Prompt body
beyond the final lane payload selected for application. Sensitive lane text is
excluded from representations, feature-owned serialization or persistence
APIs, diagnostics, and logs. The final selected original-or-rendered lane text
is the application payload itself; no separate raw/source copy is retained.

Applications expire when monotonic elapsed time is greater than or equal to
120 seconds. The pending handoff is detached, latest-wins, one-shot, and
owner-thread-only. `PendingHandoffStore.claim()` returns a claim with the
bounded status `ready` or `expired`. For `CONSOLE_PROMPT_INSERT`, the store
computes that status with its injected monotonic clock while moving the exact
revision in flight. An expired claim still reaches the consumer so it can
acknowledge it and show a bounded warning; it is never returned to pending
state. Release always settles the exact in-flight claim. It requeues the
retained payload only when the claim is ready, still unexpired, and its revision
is still the channel's current revision; otherwise it discards that old claim
while preserving any newer pending revision. Apply-time expiry is acknowledged
and warned, not released. A wrong type, wrong session,
stale captured composer, or stale authorized System fingerprint is likewise
acknowledged and discarded without mutation and with a bounded warning.

Library obtains its authorization target only through an owner-thread,
app-owned `ConsolePromptTargetProjection`. The projection contains the target
session ID and a one-way System fingerprint—never System or composer text.
Immediately before navigation away from Console, the app asks the live Console
owner for this sanitized projection and publishes it in `ScreenStateStore`
under the current runtime identity. Restoring a compatible projection returns
a detached value. A runtime/source identity change, Console snapshot discard,
or explicit Console reset invalidates it together with the corresponding
screen snapshot. Library never reads serialized Console sessions directly. If
no prior Console target exists, Library refuses before opening the dialog or
staging content and shows `Open Console once, then retry Use in Console.`; it
does not guess or create a hidden destination session.

The Console applies composer and authorized System changes as one coordinated,
reversible in-memory operation. A subsequent durable conversation/System write
is a separate outcome: if it fails, the UI reports that failure honestly and
does not claim atomic disk rollback. Values are never persisted as reusable
defaults. Text intentionally inserted into the composer or authorized System
lane follows those destinations' ordinary later lifecycle.

Prompts without recognized variables and without a System lane retain the
existing direct safe insertion path through the same guarded application
helper. A System-only Prompt still opens the shared dialog so replacement must
be authorized. Recipes remain non-executable under ADR-040: their existing
selection path must first create an unsaved Prompt working copy before variable
application becomes available.

## Context

The three Prompt insertion entry points currently converge only after making
different assumptions about composer timing and destination behavior. Adding
placeholder rendering independently to each would create incompatible brace
grammars, duplicate sensitive value handling, and allow one path to bypass
System authorization or stale-state checks.

Repeated regular-expression replacement is especially unsafe here. Decoding
`{{` and `}}` before another substitution pass can turn deliberately literal
`{{name}}` into an active `{name}`. Replacing variables in a loop can likewise
interpret braces introduced by user values. A tokenized single pass makes both
behaviors impossible by construction.

The Library-to-Console route crosses a navigation boundary. A bare string
cannot truthfully carry the destination, session, authorization, staleness, or
expiry evidence needed to avoid modifying the wrong draft. The application
therefore needs a typed memory-only handoff, while mutation authority remains
with the destination Console rather than the Library or modal.

## Required Boundaries

- The parser, render plan, fingerprints, and application value are independent
  of Textual and can be property-tested without a mounted app.
- All three entry points use the same grammar and the same shared dialog.
- The raw variable map exists only while the mounted dialog is collecting
  values and is absent from the application request.
- Variable and lane content is never included in `repr`, exception text,
  notifications, structured diagnostics, or logs.
- Composer and System fingerprints are computed from exact current state with
  a one-way digest; no body text is embedded in the fingerprint.
- Slash and picker replacement compare the entire captured composer snapshot,
  not only the command text or cursor-local segment; capture occurs before the
  first awaited resolution or selection step.
- Library reads only an app-owned, sanitized target projection. Publication,
  runtime-compatible restoration, and invalidation remain owner-thread-only and
  follow the corresponding Console screen snapshot lifecycle.
- Library append captures the active composer snapshot at consumption, not at
  Library authorization time.
- System application is opt-in, defaults off, and compares the authorization-
  time System fingerprint immediately before mutation.
- Claim-time and apply-time expiry use an injectable monotonic clock and the
  exact `elapsed >= 120` boundary.
- Claim-time expiry returns an explicit `expired` claim status so the consumer
  can acknowledge once and warn; release never resurrects an expired claim.
- Release always clears the exact in-flight claim. Missing-composer retry
  requeues only a ready, unexpired, still-current revision and never overwrites
  newer pending work or resurrects an expired, stale, or acknowledged request.
- In-memory rollback and durable persistence reporting remain distinct.
- Prompt insertion never turns a Recipe into an executable artifact.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Run a regular expression repeatedly until no placeholders remain | Escape decoding or a value containing braces could create a new placeholder on a later pass. |
| Use separate parsers or dialogs for slash, picker, and Library | Produces grammar/copy drift and lets authorization or stale-state rules diverge. |
| Treat every `{...}` span as a variable | Ordinary JSON, XML-adjacent text, unmatched braces, and invalid names would become errors or unintended inputs. |
| Persist variable definitions or recent values with Prompt records | TASK-199 only needs insertion-time values; persistence adds a sensitive-data lifecycle and changes the Prompt artifact schema without a user requirement. |
| Pass a bare rendered string through the handoff | Cannot express lane authorization, destination, expiry, target session, or stale-state guards. |
| Let Library or the dialog mutate the Console composer | Violates destination ownership and races navigation, session changes, and composer remounts. |
| Let Library inspect serialized Console session state | Couples one screen to another screen's private snapshot and exposes more session data than authorization needs. |
| Guess or create a destination when no Console target was ever published | System authorization would be bound to an undisclosed session; the honest recovery is to initialize Console once and retry. |
| Enable System replacement by default | A Prompt could silently change session behavior while the user intended only to insert User text. |
| Make composer and durable System persistence one advertised atomic transaction | The current durable write is a separate failure domain and cannot truthfully roll back all external effects. |
| Execute Recipe lanes directly | Violates ADR-040's artifact boundary and bypasses the explicit unsaved Prompt-copy step. |

## Consequences

### Benefits

- Literal braces and variables have one documented, property-testable meaning.
- Shared values remain consistent across System and User lanes and entry paths.
- System changes are always explicit and stale state fails closed.
- Navigation handoffs cannot silently apply to a later session or draft.
- Sensitive insertion values have no reusable persisted-default or logging
  surface.

### Accepted trade-offs

- Literal single braces around a valid name must use `{{name}}` to remain
  literal.
- The dialog may temporarily retain up to 64 ephemeral values while mounted.
- A Library handoff may expire or be refused after navigation instead of
  applying optimistically to changed Console state.
- Durable System persistence can fail after a coordinated live update and must
  be reported as a distinct outcome.

## Links

- [TASK-199 implementation plan](../../Docs/superpowers/plans/2026-08-02-task-199-shared-prompt-variables.md)
- [Library Prompt Enhancement Series design](../../Docs/superpowers/specs/2026-08-02-library-prompt-enhancement-series-design.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-033: Application Session State Ownership](033-application-session-state-ownership.md)
- [ADR-040: Versioned Prompt Artifacts and Safe Improvement Transactions](040-versioned-prompt-artifacts-and-safe-improvement-transactions.md)
