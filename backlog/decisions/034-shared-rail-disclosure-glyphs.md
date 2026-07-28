# ADR-034: The rail disclosure glyphs are shared widget vocabulary, not Console vocabulary

- Status: Accepted
- Date: 2026-07-27
- Task: task-832

## Context

`Widgets/destination_rail.py` re-declared `GLYPH_EXPANDED` (`▾`) and
`GLYPH_COLLAPSED` (`▸`) rather than importing them from
`Chat/console_glyphs.py`, so the extracted shared widget would carry no
Chat-layer dependency. A test in a third file
(`Tests/UI/test_destination_rail.py`) asserted the two copies stayed equal.

PR #940's final review argued this installs a hidden bidirectional lockstep:
neither module can change its glyphs without a test in an unrelated file
going red, and nothing in either module says so. It proposed inverting the
ownership. The counter-argument recorded in task-832 was that inverting
"makes the Chat layer import from Widgets, which is the worse dependency
direction".

Two measurements decided it.

**The stated dependency concern is backwards for this codebase.** Counting
imports on `origin/dev`:

| Direction | Imports |
| --- | --- |
| `Widgets/` → `Chat/` | 40 |
| `Chat/` → `Widgets/` | 1 |

The direction the counter-argument calls "worse" is the rare one. The
direction it treats as safe is the established tangle. Avoiding a single
`Chat` → `Widgets` import buys nothing while 40 imports run the other way.

**These glyphs are already not Console-only.** `UI/Evals/library_rail.py` —
a Lab destination with no Console involvement — imports
`GLYPH_COLLAPSED, GLYPH_EXPANDED` from `Chat.console_glyphs`. A destination
outside Chat reaching into the Chat layer for a disclosure triangle is the
clearest evidence that these two glyphs are not part of Console's glyph
language. The rest of `console_glyphs` (`GLYPH_ACTIVE`, `GLYPH_IN_PROGRESS`,
`GLYPH_DONE`, `GLYPH_CLOSE`, `GLYPH_COLLAPSE_LEFT`) genuinely is.

## Decision

`Widgets/destination_rail.py` **owns** `GLYPH_EXPANDED` and
`GLYPH_COLLAPSED` — it is the widget that renders them.
`Chat/console_glyphs.py` re-exports both so Console code keeps one import
site for its glyph vocabulary. Destinations that are not Console import
them from `destination_rail` directly.

Each constant is defined exactly once. `destination_rail` still carries no
Chat import, so no cycle is possible.

The equality guard is replaced by an identity assertion: with one
definition and a re-export, `console_glyphs.GLYPH_EXPANDED is
destination_rail.GLYPH_EXPANDED` holds structurally, and drift is no longer
something a test has to watch for.

## Alternatives rejected

**Keep the duplication and the guard test.** Leaves the lockstep the review
objected to, and leaves a non-Console destination importing Console
vocabulary.

**`console_glyphs` owns; `destination_rail` imports it.** Reintroduces the
Chat dependency into a module whose stated purpose is to have none, and
would make every future destination adopting the shared rail depend on the
Chat layer to draw a triangle.

**A neutral third module owning the glyphs.** One definition and no
layer-crossing, but it adds a module for two constants and leaves
`destination_rail` — the only thing that renders them — not owning them.
Reconsider if a third category of shared glyph appears.

## Consequences

- `Chat/console_glyphs.py` imports from `Widgets/destination_rail.py`. This
  is the first deliberate `Chat` → `Widgets` import; it is the healthy
  direction (a feature layer depending on shared widgets) and is recorded
  here so it is not mistaken for an accident.
- `UI/Evals/library_rail.py` no longer imports from the Chat layer.
- Changing a disclosure glyph is now a one-line edit in one file.
