# ADR-004: Personas destination-native workbench and CCP route retirement

Status: Accepted
Date: 2026-06-09
Related Task: [backlog/tasks/task-85 - Personas-destination-native-workbench-Characters-Personas.md](../tasks/task-85%20-%20Personas-destination-native-workbench-Characters-Personas.md)
Supersedes: N/A

## Decision

The `personas` route owns a single destination-native workbench for Characters and
Personas (create/view/edit/manage, import/export, preview, Console attachment). The
legacy `ccp` route and its screen (`ccp_screen.py`), the `conversation_screen.py`
re-export shim, and sidebar-era chrome (`ccp_sidebar_widget.py`,
`ccp_sidebar_handler.py`) are retired; `ccp`, `characters`, and `prompts` legacy
routes resolve to `personas`.

## Amendment (2026-09-03, TASK-31241 — Roleplay conversation browsing)

[ADR-120](120-character-conversation-navigation-and-local-semantic-search.md)
extends the destination-native Roleplay workbench with complete, searchable,
keyset-paginated conversation browsing for one exact resolved local Character
card. Roleplay owns that per-character browse, read-only preview, and exact
Console resume path; it does not edit, delete, or otherwise administer
conversation content. Persona records do not receive a Conversations view.

Library remains the global archive and cross-domain conversation browser and
retains its existing administrative actions, including the explicitly confirmed
repair of unresolved character links. This narrows the earlier consequence that
assigned all full conversation browsing to Library without changing Library's
global/archive ownership or Roleplay's Character-card mutation ownership. This
amendment is owned by
[TASK-31241](../tasks/task-31241%20-%20Align-character-conversation-navigation-decisions.md).

## Context

The Personas destination was split across a thin snapshot shell (`personas` route)
and a half-converted legacy workbench (`ccp` route), forcing two-hop navigation and
duplicating attachment logic. Console, Library, and Notes already follow the
destination workbench grammar defined in
`Docs/Design/agentic-terminal-visual-system.md`. The CCP behavior layer (handlers,
character card/editor widgets, import/validation libraries, scope service) is sound
and is reused; only the route shell, persona stubs, and sidebar chrome are replaced.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Finish converting `ccp_screen.py` in place, flip routes last | Keeps building on an 1,800-line screen with sidebar-era state; two parallel screens persist; hardest path to Notes-level layout quality |
| Clean rebuild including the data/behavior layer | Re-implements an 866-line character editor and battle-tested import flows for no product gain; highest regression risk |
| Separate Characters and Personas top-nav destinations | Violates the top-navigation contract (stable global destinations); New_UI images are layout references, not nav requirements |

## Consequences

- One management surface; Library keeps ownership of full conversation browsing.
- New pane widgets live in `Widgets/Persona_Widgets/`; reused CCP widgets keep their
  import paths and internal IDs (`#ccp-character-card-view`, `#ccp-character-editor-view`).
- Selection/search/preview message classes move out of the retired screen into
  `Widgets/Persona_Widgets/personas_messages.py`.
- Server-backed CRUD is out of scope; authority labels keep the seam visible.

## Links

- Spec: Docs/superpowers/specs/2026-06-09-personas-workbench-design.md
- Plan: Docs/superpowers/plans/2026-06-09-personas-workbench-implementation.md
