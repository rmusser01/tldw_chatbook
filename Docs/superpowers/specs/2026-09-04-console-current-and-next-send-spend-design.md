# Console Current and Next-Send Spend Design

## Goal

Make the Console context/spend chip answer two separate money questions at a glance: how much the conversation has cost so far, and the estimated additional request charge on the next send.

## UX Contract

The combined chip keeps context fullness first and gives each spend value an explicit time frame:

- Wide: `Context 45% · Current $0.48 · On next send ~+$0.13`
- Narrow: `Ctx 45% · Now $0.48 · Next ~+$0.13`

`Current` is the conversation spend already accumulated by completed usage plus locally estimated transcript rows whose providers did not return usage. It excludes the current composer draft and staged evidence, because those have not been sent yet. The approximation marker remains attached whenever the current transcript total contains estimates. Unpriced fleet tokens remain disclosed separately in the tooltip and never fabricate a dollar contribution.

`On next send` is the incremental request/input estimate for the provider payload represented by the current context snapshot, including staged evidence and the live composer draft. It excludes the unknowable response/output charge. The tooltip states that output spend is added after completion, cache reads may lower the estimate, and cache writes may raise it.

When either selected-model pricing or the request-token estimate is unavailable, the forecast says `unavailable`; it never fabricates `$0.00`. A genuinely zero-priced local model may show `$0.00`.

Cache warning behavior remains intact. The alert glyph may remain visible on the chip, while cache state, expiry, break reason, and any cache-specific delta stay in the tooltip rather than competing with the next-send estimate as a second unexplained dollar increase.

## Data Flow

The context estimate is extended to include the mounted composer's current draft as the same synthetic user turn used by the real next-send preview. Idle draft edits schedule one short coalesced refresh of the context summary and spend chip, so typing updates the forecast without tokenizing the transcript on every keystroke. The screen then reuses `ConsoleContextControlState.request_tokens`, multiplies that estimated next-request token count by the selected model's uncached input rate, and passes the result into the existing combined context/cost formatter. No second transcript tokenization or provider request is introduced.

Using the uncached input rate makes this an explicitly named **uncached baseline estimate**, not a bound. Cache reads may reduce the actual input charge, while providers whose cache writes cost more than uncached input may raise it. This avoids guessing how each provider will divide the next payload among uncached, cache-read, and cache-write buckets.

## States

- Known current spend and forecast: show both dollar values.
- Estimated current spend: prefix the current value with `~`.
- Unknown historical/current pricing with known selected-model pricing: show the current tokens-only fallback and the next-send dollar forecast.
- Known current pricing with unknown selected-model pricing: show current dollars and an unavailable next-send forecast.
- Both prices unknown: show the current tokens-only fallback and an unavailable next-send forecast.
- Empty conversation with known selected-model pricing: show `Current $0.00 · On next send —` until a text request can be estimated; the tooltip prompts the user to type a message.
- Unknown request size with known current pricing: show current dollars and an unavailable next-send forecast.
- Pending or historical media: show the text-token detail in the tooltip, but mark the overall next-send forecast unavailable because media input cost is not estimated.
- Cache alert: preserve alert styling and explanation without treating the cache-break delta as the full next-send charge.

## Verification

Tests cover exact wide and narrow labels, current versus next-send semantics, input-only forecast math, staged-evidence and composer-draft ownership, an idle draft edit updating the chip through the coalesced refresh, independent current/forecast pricing states, unknown request size, pending and historical media, zero-priced local models, tooltip disclosure, cache-alert copy, keyboard activation, initial narrow mounting, live wide-to-narrow and narrow-to-wide resizing, and mounted status-strip geometry.

## Architecture

ADR required: no.

[`backlog/decisions/052-console-conversation-memory-and-compaction-policy.md`](../../../backlog/decisions/052-console-conversation-memory-and-compaction-policy.md) already owns request capacity and compaction semantics; [`backlog/decisions/095-conversation-owned-console-generation-settings.md`](../../../backlog/decisions/095-conversation-owned-console-generation-settings.md) already owns conversation settings. This is a display and estimation refinement within those boundaries.
