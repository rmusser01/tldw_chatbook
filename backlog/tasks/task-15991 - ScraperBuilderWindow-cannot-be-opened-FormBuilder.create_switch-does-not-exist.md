---
id: TASK-15991
title: 'ScraperBuilderWindow cannot be opened: FormBuilder.create_switch does not exist'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 01:10'
labels:
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`UI/ScraperBuilderWindow.py:322` calls `self.form_builder.create_switch(...)`, but `FormBuilder` (Widgets/form_components.py) has no such method, so opening the window dies in compose with AttributeError. Pre-existing on dev well before the CSS consolidation (confirmed absent at `6b57458b8` too) — the screen is unopenable, which also means any historical evidence that 'measured' it opening measured a crash. Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening ScraperBuilderWindow composes without raising
- [x] #2 A mounted regression test pins the open (born-red against the current crash)
- [x] #3 Notes state whether create_switch was added to FormBuilder or the call sites changed, and why
<!-- AC:END -->

## Implementation Plan

1. Reachability check: confirm whether ScraperBuilderWindow is reachable from live navigation at HEAD (registry, palette, imports) and whether any retirement decision exists — record the verdict.
2. Convention check: read FormBuilder (Widgets/form_components.py) and its other consumers to decide the fix side (add create_switch vs compose Switch at the call sites).
3. Write a born-red mounted test (Tests/UI/test_scraper_builder_window.py): push the screen in a bundle-CSS harness, assert compose succeeds; drive one switch's value into its consumer (_format_options_code). Run at HEAD to capture the exact AttributeError.
4. Fix on the chosen side; re-run the test green; ruff check + format on touched files.
5. Notes + AC tick + Done.

## Implementation Notes

**Fix side: the call sites changed, not FormBuilder.** FormBuilder
(Widgets/form_components.py) deliberately has no per-widget factories at all —
its whole API is `create_form_field(label, widget)` / `create_field_set` /
`add_field`, and its only other consumer, `UI/SiteConfigSettings.py`, composes
`Switch(value=..., id=...)` directly and passes it in (lines 227/311-323, with
the very same ids `remove-scripts`/`remove-styles`/`preserve-links`). The four
`self.form_builder.create_switch(...)` calls in `_compose_options` were changed
to `Switch(value=<default>, id=<id>)` to match that grain; the now-dead
`self.form_builder` attribute and `FormBuilder` import were removed.

**The born-red mounted test surfaced three MORE compose crashes behind the
first** — each was the next AttributeError/MountError once the previous was
fixed, all pre-existing (the window had never composed end to end):

1. `TextArea(language="html")` for the HTML preview raises
   `LanguageDoesNotExist` at construction when tree-sitter is installed but the
   `tree-sitter-html` grammar package is not (true of the project venv: only
   python/javascript grammars resolve). Fixed with a module-level
   `_syntax_text_area(**kwargs)` helper that retries without `language` on
   `LanguageDoesNotExist` — a missing optional grammar costs highlighting, not
   the screen.
2. `Collapsible("Common Selectors", collapsed=False)` passed the title string
   as a *child* (Collapsible's first positional is `*children`) →
   `MountError: Can't mount <class 'str'>`. Fixed to `title=`.
3. Both `Select`s had their option tuples `(value, label)`-reversed (Textual
   expects `(label, value)`), so `value="clean"` raised
   `InvalidSelectValueError` — and the rule-type/text-processing consumers
   (`event.value == "custom"`, `query_one("#text-processing").value`) would
   have received display labels. Swapped to `(label, value)`.

**Reachability**: the window is nav-unreachable at HEAD (zero imports outside
its own file; no screen-registry route, palette entry, or button), but no
retirement decision exists — ADR-020 in
Docs/Development/Subscriptions/Subscriptions-Implementation-1.md records it as
a designed feature that simply was never wired up. Fixed per the ACs;
wiring it into navigation is out of scope here.

**Test**: `Tests/UI/test_scraper_builder_window.py` — bundle-CSS harness pushes
the screen, asserts compose succeeds, asserts all four switches mount with
their defaults, asserts `#text-processing` carries the machine token `clean`,
and drives `#remove-scripts` True→False asserting the value reaches its
consumer (`_format_options_code`) both before and after. Born red at HEAD
`11646bba0` with exactly `AttributeError: 'FormBuilder' object has no
attribute 'create_switch'` at ScraperBuilderWindow.py:322; green after.

**Files**: tldw_chatbook/UI/ScraperBuilderWindow.py,
Tests/UI/test_scraper_builder_window.py (new).

**Known pre-existing red (not this task)**:
`Tests/test_call_from_thread_guard.py` fails on dev/HEAD for
`UI/Screens/chat_screen.py:3029` (bare `self.call_from_thread(` on a Screen) —
present byte-identical on origin/dev, untouched by this change.
