# ADR-161: Component-pattern layer of the design system

Status: Accepted
Date: 2026-09-13
Related: ADR-150, Docs/superpowers/specs/2026-09-13-component-pattern-library-design.md

## Decision

The design system gains a component-pattern layer above ADR-150's tokens:

1. Form factor is a CSS-class catalog, not Python builders. Patterns are documented
   classes + structure + state contracts; screens keep hand-composing class strings.
   Evidence: two prior builder libraries died of disuse (0 and 2 importers) while the
   raw class vocabulary reached 300+ uses. Winning Python idioms (SafeModalDismissMixin,
   ConfirmationDialog) are documented as part of their pattern, not replaced.
2. `tldw_chatbook/css/patterns.json` is the machine-readable registry (families,
   owning sheets, class lifecycle). Governance tests enforce: single definition per
   Canonical class (bare-selector rules only in the owning sheet; compound/scoped
   selectors are legal composition), deprecated-name and literal ratchets,
   doc/gallery sync.
3. Carve-up rule: classes promote to sheets matching consumption scope. Moves stay
   within the bundle vs per-screen-source track (Textual per-source $variable scope;
   TASK-15993/16811). New per-screen blocks carry no local $ds-* fallbacks.
4. ADR-150 §6's "migrate opportunistically" policy is AMENDED: the ~6,250 legacy
   dimension literals and ~580 Python `styles.*` visual assignments migrate as
   committed, ratcheted, project-owned work. End state: zero raw literals in sheets
   (recurring values → shared tokens; feature geometry → `$ds-<feature>-*` tokens)
   and zero ad-hoc Python visual assignments.

## Alternatives considered

- Python builder revival — contradicts 170 screens of practice; strands existing UI.
- Opportunistic literal migration (status quo per ADR-150 §6) — unowned and
  unbounded; rejected by the project owner for consistency and long-term stability.
- One big-bang migration pass — unreviewable; ADR-150 already rejected this shape.

## Consequences

- `_variables.tcss` grows feature-scoped `$ds-<feature>-*` tokens (append-mostly).
- Bundle bytes grow from tokenization; sequencing coordinates with TASK-31500.
- `_agentic_terminal.tcss` is dissolved (absorbs TASK-24451); ≤2,000
  comment-stripped lines per sheet enforced on completion.
