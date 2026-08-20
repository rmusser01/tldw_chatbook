# ADR-075: Durable character emote metadata

Status: Proposed
Date: 2026-08-20
Related Task: [TASK-19060](../tasks/task-19060%20-%20Match-server-streaming-emotes-and-persistence.md)
Related Spec: [Actor Pack, Persona Buddy, and Streaming Emote Programme Design](../../Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md)
Amends: [ADR-067](067-bundled-samira-visual-identity-pack.md)
Supersedes: N/A

## Decision

Chatbook will match the character-emote contract in `tldw_server` development commit
`385afa951922c8a9dc2002c675bb6cad65e4ac23` exactly. Compatibility is pinned to that
commit and its frozen cross-language vectors, not to an evolving server branch.

Character prompts will project safe slugs from canonical expression keys in the active
Shared Visual Identity version. A canonical key advertises its safe slug;
`custom:<token>` advertises `<token>` only when the token passes the pinned safe-state
grammar and local expression normalization maps it back to that exact asset key.
Invalid, ambiguous, colliding, or non-round-tripping keys are omitted. Projected slugs
are deduplicated in first stored asset order. The prompt exposes the first 25 states
and, when states remain, appends the exact suffix ` (+N more)`, where `N` is the
hidden-state count. Imported labels and other arbitrary display text are not prompt
inventory.

Streaming and non-streaming responses will parse only assistant-visible text. Only
out-of-fence lines matching the standalone, case-insensitive `Emote: <state>` form are
control lines and stripped; fenced directives and inline prose remain visible. The
state is trimmed, lowercased, and has internal whitespace replaced by hyphens, then
must match `[a-z0-9][a-z0-9_-]{0,39}`. At most five events are accepted, and
consecutive accepted duplicates are suppressed. Invalid, duplicate, and over-cap
matching directive lines are still stripped. Arbitrary chunk boundaries, CRLF, and
unterminated final lines follow the pinned server contract. Reasoning, tool arguments
and results, citations, and provider control events are never parser input.

A session-local manual override remains the highest display choice, but it does not
suppress directive parsing or durable metadata persistence. When at least one valid
explicit event is accepted, the explicit sequence determines the final expression and
the last accepted state determines `mood_label`; the heuristic fallback does not run.
The heuristic may supply the final mood only when no explicit event exists. A valid
explicit state with no resolvable asset still suppresses the heuristic and records a
stable fallback reason while display retains the current or base portrait.

Assistant messages will durably retain bounded `mood_label` and `emote_events`
metadata, with at most five normalized `{state, at_char}` events. Each `at_char` is a
nonnegative integer, event offsets are nondecreasing, and no offset exceeds the
sanitized text length measured in JavaScript-compatible UTF-16 code units. The actor
and resolved immutable pack, version, expression, and asset identities needed for the
final appearance are profile-local references, not server IDs. Optional heuristic
mood fields and fallback reasons remain bounded scalar metadata. Raw directives,
assistant text, prompts, provider payloads, local paths, and manual-display overrides
are not part of this visual metadata contract.

Diagnostics contain fixed categories and identifiers only. They exclude assistant
text, prompts, paths, raw provider output, archive member names, bytes, and cleanup
tokens.

V1 retains activated immutable visual versions and adds no physical version garbage
collection. History therefore restores the exact final immutable expression while
the referenced local data remains intact and reports a deterministic fallback when it
is unavailable. It does not replay intra-message emote beats in V1.

This decision amends ADR-067's session-only message boundary only for bounded durable
character-expression metadata and final-expression history restore. It does not make
manual overrides durable, and it does not merge Persona Visual operational states or
Persona Buddy control into Shared Visual Identity expressions. It authorizes no sync,
server transport, or server implementation.

## Context

ADR-067 deliberately kept manual expression overrides session-local and deferred a
durable message replay contract. Chatbook now needs server-compatible streaming
expression changes without leaking control syntax into the conversation, search, or
exports, and it needs historical messages to recover their final immutable visual
identity after a session ends.

The pinned server already defines the prompt cap, directive grammar, event bound,
UTF-16 offset semantics, and explicit-over-heuristic precedence. Treating those rules
as one compatibility contract avoids divergent behavior between live streaming,
non-streaming completion, persistence, and history reload.

## Alternatives Considered

| Alternative | Why rejected |
| --- | --- |
| Parse only the final emote | It cannot drive live expression changes or retain the bounded event evidence and offsets required for server compatibility. |
| Make tool calls the primary emote protocol | It couples expression control to provider tool support and diverges from the pinned assistant-visible-text contract. |
| Reuse Persona Buddy states | Buddy states represent trusted application operations, not character reactions; reuse would merge the runtimes ADR-067 keeps separate. |
| Run heuristic and explicit emotes together | Competing selectors make the final expression nondeterministic and violate the server's explicit-emote precedence. |
| Persist raw directive lines | Control syntax would leak into conversation content, search, and export and would require reparsing untrusted historic text. |
| Replay historical emote beats in V1 | Timed replay adds lifecycle and rendering semantics not needed to restore a message's final immutable appearance. |

## Consequences

- Streaming and non-streaming character replies share one pinned sanitization and
  metadata contract.
- Explicit emotes can update the live portrait while manual display choice remains
  local and non-durable.
- Messages gain bounded visual metadata and immutable final-expression references;
  malformed or unavailable metadata must fail soft to a deterministic fallback.
- Parser and resolver failures cannot block or corrupt the sanitized assistant reply.
- Cross-language fixtures must be reviewed before adopting behavior from a later
  server commit.
- Historical intra-message beat replay and Persona Buddy state control remain out of
  scope.

## Links

- [ADR-067: Bundle Samira through a local Visual Identity bridge](067-bundled-samira-visual-identity-pack.md)
- [Approved programme design](../../Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md)
- [TASK-19060: Match server streaming emotes and persistence](../tasks/task-19060%20-%20Match-server-streaming-emotes-and-persistence.md)
- [`tldw_server` compatibility commit](https://github.com/rmusser01/tldw_server/commit/385afa951922c8a9dc2002c675bb6cad65e4ac23)
