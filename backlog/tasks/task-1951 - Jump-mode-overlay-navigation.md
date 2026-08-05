---
id: TASK-1951
title: Jump mode overlay navigation
status: To Do
assignee: []
created_date: '2026-08-02 20:30'
labels:
  - ux
  - navigation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Borrowed from [Bagels](https://github.com/EnhancedJax/Bagels), which lists "Jump Mode
Navigation" as a headline feature and binds it to `v` in its footer.

**The mechanic**, read from Bagels' `components/jumper.py` (59 lines) and
`components/jump_overlay.py` (92 lines), and confirmed against its own snapshot test
`TestBasic.test_4_jump_screen.svg`:

- `Jumper` walks `screen.walk_children(Widget)`, keeps those whose `id` is in an
  id-to-key map (or that satisfy a `Jumpable` protocol exposing `jump_key`), and returns
  `{Offset: JumpInfo(key, widget)}` using `screen.get_offset(child)`. Widgets not
  currently laid out raise `NoWidget` and are skipped, so hidden panes cost nothing.
- `JumpOverlay` is a `ModalScreen` with a 25%-black background. It paints a `Label` per
  target with `label.styles.offset = offset`, plus a centred "Press a key to jump" and
  "ESC to dismiss". `on_key` dismisses with the chosen widget.
- Its snapshot shows the label drawn as a small box over each pane's **top-left border
  corner**, replacing the first characters of the border title — `╭─a─╮ccounts`,
  `╭─t─╮emplates`, `╭─p─╮eriod`, `╭─r─╮ecords`, `╭─i─╮nsights`. The keys are **mnemonic
  first letters, not arbitrary Vimium labels**, so on a stable layout they become
  muscle memory rather than something to read every time.
- It explicitly stops bubbling for a named set of keys, so a key pressed to jump cannot
  also be handled by the parent after dismissal and move focus a second time.

**Licensing — take it from Posting, not Bagels.** Bagels is GPL-3.0. Its jumper is
derived from [Posting](https://github.com/darrenburns/posting), which is **Apache-2.0**
and ships the same `jumper.py` / `jump_overlay.py`. We are AGPL-3.0-or-later; sourcing
from Posting removes the licence question entirely. Keep the attribution notice either
way.

**Why it fits us specifically.** With 47 screens and 34 `BINDINGS` blocks, our problem is
not too few shortcuts — it is that nobody can hold them. Jump mode replaces "memorise a
binding per control" with "one binding, then look at the screen". It also sidesteps a
defect family our own history keeps producing: a jump target is focused by id, so it
cannot be made unreachable by a layout change that moves a control out from under the
mouse.

**The one adaptation we need.** Bagels assigns jump keys from a static config map for a
single screen. With 47 screens we want keys resolved per *visible* screen at overlay
time — which the `walk_children` design already supports, since it only sees what is
laid out. Depends on nothing, but reads much better once [[task-1950]] exists, because
the two use the same "the key lives on the control" idea.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] One global binding opens a jump overlay over the current screen
- [ ] Every jumpable target on the visible screen shows a single mnemonic key at its own position; hidden or unlaid-out widgets are skipped without error
- [ ] Pressing a target's key dismisses the overlay and focuses that widget; ESC dismisses without moving focus
- [ ] A key pressed to jump does not also reach the underlying screen and move focus a second time
- [ ] Two screens with different layouts are covered, and a test drives a real keypress through `pilot` and asserts which widget ended up focused
- [ ] Attribution and licence provenance recorded for whatever implementation we source
<!-- AC:END -->
