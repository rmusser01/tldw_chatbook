# ADR-042: Design token system and design-language constitution

Status: Accepted
Date: 2026-08-05
Related Tasks: TASK-2511

## Decision

UI styling in tldw_chatbook is governed by a design-token system. Every
visual value that must stay consistent across screens — spacing, control
sizing, motion duration, opacity, typography emphasis, colors, status and
component-state semantics — exists as a `$ds-*` token defined in
`tldw_chatbook/css/core/_variables.tcss`, or as a written rule in
`backlog/docs/design-language.md`. Contributors and agents compose new UI
from these tokens and rules; they do not invent literals per screen.

Token rules:

- All tokens live in one place: `core/_variables.tcss`. Tokens are scalar,
  single-value definitions (Textual TCSS substitutes variables per token).
- Naming is `$ds-<category>-<name>[-<variant>]`; semantic aliases (e.g.
  `$ds-space-stack`) point at scale values, never at raw literals of their
  own.
- A new raw value may only enter a stylesheet by first becoming a token in
  `_variables.tcss`. Feature-scoped one-off geometry may remain literal but
  is expected to migrate when a second consumer appears.
- A governance test fails the build when a `$ds-*` token is referenced but
  not defined.
- Python code does not set ad-hoc `styles.*` literals for values covered by
  tokens; it assigns token-backed CSS classes or uses values that already
  exist as tokens.

The constitution document (`backlog/docs/design-language.md`) is the
human- and agent-readable specification: token catalog, layout laws,
interaction rules, component-state conventions, and the process for adding
tokens. AGENTS.md points at it as required reading before UI work.

## Alternatives considered

- **Status quo (per-screen literals):** rejected — every new screen re-opens
  the same styling debates; ~7,100 hardcoded dimension declarations and ~900
  hex literals have already accumulated, and existing design rules survive
  only as comments.
- **Full migration to tokens in one pass:** rejected for now — rewriting
  thousands of declarations in one change is unreviewable and risks visual
  regressions. The system is introduced additively with exemplar sheets and
  a ratchet: new work must use tokens; legacy literals migrate opportunistically.
- **Tokens in Python (`Constants.py`):** rejected as the primary home — the
  stylesheet is where values are consumed; a single token file keeps one
  source of truth. Python-side consumers reference tokens by name via CSS
  classes, not duplicated constants.
- **Multi-value composite tokens (e.g. `$ds-border-panel: round $border`):**
  deferred — Textual variable substitution semantics for multi-token values
  are not verified in this repo; tokens are scalar until that is tested.

## Consequences

- `core/_variables.tcss` becomes the design vocabulary; it is append-mostly
  and changes to existing token values are visual-breaking changes requiring
  deliberate review.
- New UI PRs are reviewable against the constitution ("which token/rule
  justifies this value?") instead of taste.
- The bundle sync guard and the new governance test enforce the system
  mechanically.
