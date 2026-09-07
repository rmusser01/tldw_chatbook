---
id: TASK-1952
title: Config-driven keybindings
status: To Do
assignee: []
created_date: '2026-08-02 20:30'
labels:
  - ux
  - navigation
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Borrowed from [Bagels](https://github.com/EnhancedJax/Bagels), which lists "Customizable
Keybindings and Defaults" as a headline feature.

We have **no user-configurable keybindings at all** — every one of our 34 `BINDINGS`
blocks is a hardcoded literal. Bagels reads its hotkeys from config under a nested
namespace (`CONFIG.hotkeys.new`, `CONFIG.hotkeys.edit`, `CONFIG.hotkeys.delete`,
`CONFIG.hotkeys.home.new_transfer`), which buys two things beyond user preference:

1. **Code can reason about keys by name.** Bagels' jump overlay suppresses key bubbling
   by referring to `CONFIG.hotkeys.new` rather than the literal `"a"`. Any code that must
   know "is this key spoken for?" — a jump overlay, a modal, a text input that must not
   swallow a global — needs exactly this.
2. **Collision detection becomes possible.** With 34 independent hardcoded blocks we
   cannot currently answer "does this new single-letter binding collide with anything?"
   except by reading all of them. That question is a blocker for [[task-1951]].

This is a real migration rather than a weekend: config schema, defaults, a resolution
layer that builds `BINDINGS` from it, per-screen namespacing, and a migration path for
users who have never had this setting. Worth scoping as its own small plan.

Deliberately NOT in scope: remapping keys through the UI. A config file the user edits is
enough for v1; a settings screen for it is a later question.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Keybindings resolve from config with the current hardcoded values as defaults, so an untouched install behaves exactly as it does today
- [ ] The namespace is per-screen, so two screens can bind the same key to different actions without ambiguity
- [ ] Code can refer to a binding by name rather than by literal, and at least one caller does
- [ ] A malformed or unknown key in config fails loudly at startup naming the offending entry, rather than silently dropping the binding
- [ ] Collisions within one screen's resolved set are detected and reported
- [ ] Documented in `DESIGN.md` and in whatever the user-facing config reference is
<!-- AC:END -->
