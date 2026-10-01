---
id: TASK-33623
title: >-
  Console unification: one action registry and glossary generating footer, F1,
  palette, /help and docs
status: To Do
assignee: []
created_date: '2026-09-30 03:03'
labels:
  - console
  - ux-review-2026-09-29
  - unification
dependencies: []
references:
  - qa/console-ux-review-2026-09-29/report.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: the Console's actions, keys and names are described in six hand-maintained lists: the footer hint tuples, the F1 groups, the command-palette provider, the slash-command registry, BINDINGS and the User Guide. They already disagree about which keys exist, what they are called and whether they work. F1 claims the palette 'lists every Alt action' when it covers 18. The footer advertises keys F1 omits. F1 does nothing inside modals and decision cards. The docs describe controls that no longer exist. Vocabulary drifts the same way: one object is called tab, chat, conversation, session and 'agent'; 'Context' and 'Sources' each carry several meanings; Settings has several names; the same letter means different things in adjacent surfaces. Five 'new chat' and six 'find chat' entry points use different verbs.

What unifying means here: one register of actions (ConsoleActionRegistry). Each entry has an id, canonical label, verb, key, slash alias, palette text, scope and focus precondition, priority and doc anchor. The footer, F1, the palette, /help and the User Guide key tables are all generated from it, and an agreement test proves they match. A short Console glossary in DESIGN.md fixes the canonical nouns, and a copy lint enforces them. Duplicate entry points collapse to one visible entry per action, plus its key and palette entry. Stale-guide cleanup builds on TASK-33125. The theme has no P0/P1 findings, so this umbrella owns the unification work directly and parents any child tasks filed for it.

Evidence: qa/console-ux-review-2026-09-29/report.md (theme 'No single register') and qa/console-ux-review-2026-09-29/findings.md / findings.json.

Related P2/P3 in the ledger: G4-17, G4-16, G2-16, G2-35, G3-33, G4-19, G4-21, G4-22, G1-17, G1-18, G3-19, G3-20, G3-21, G3-22, G4-49, G1-41, G4-48, G3-23, G3-24, G3-25, G3-26, G2-38, G4-23, G1-36, G1-48, G4-45, G2-14, G2-19, G3-42, G4-20, G4-61, G1-46.

Source: Console UX review 2026-09-29 — qa/console-ux-review-2026-09-29/report.md (themes) and findings.md (ledger).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The footer hints, F1 help, command palette, /help output and the User Guide key tables are all generated from one Console action register, and an agreement test fails if any of them lists an action, key or label the register does not
- [ ] #2 Every footer hint shown is valid for the currently focused control and current state, and the footer never shows a bare key without its label
- [ ] #3 F1 opens help from every Console context, including modals and pending decision cards
- [ ] #4 DESIGN.md contains a Console glossary with one canonical noun each for the chat object, context, sources and settings, and a copy lint fails on shipped Console labels that use a non-canonical synonym
- [ ] #5 Each Console action has exactly one visible entry point in the default view, plus its key and palette entry
- [ ] #6 All child tasks filed under this umbrella are Done (the theme has no P0/P1 findings, so its children are the unification work itself)
<!-- AC:END -->
