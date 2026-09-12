# Fleet history navigation implementation plan

Goal: Complete TASK-15201 with an accessible bounded fleet preview and a paged
history picker that selects a saved child into the existing run inspector.

Architecture: AgentRunsDB supplies narrow cursor pages through the Console
bridge. A read-only modal owns async page loading and navigation. The agent
controller owns the selected-conversation callback. ConsoleLeftRail relays the
View all event through one named callable; the screen retains its existing
drill-in ownership. The reusable Inspector section owns preview layout/scroll.

Spec: backlog/decisions/132-fleet-history-navigation.md
ADR required: yes
ADR path: backlog/decisions/132-fleet-history-navigation.md
Reason: Read contract, pagination, and a lasting fleet navigation surface.

- [x] Add red DB tests: metadata-only pages, scope/kind isolation, superseded
  history, equal timestamp ordering, inserts between pages, invalid limits/cursors.
- [x] Implement list_subagent_run_headers(conversation_id, before=None, limit=51)
  with validated keyset pagination; expose it through the bridge.
- [x] Add red component/UI tests: four-row preview and full counts, expansion
  scroll once, View all opens history, keyboard navigation reaches a later page,
  actual row selection drills into the intended old child, foreign selection is
  refused, error/retry and slow/dismissed loading are safe.
- [x] Add ConsoleAgentHistoryModal with DataTable, Previous/Next/Refresh/Close,
  loading/error/empty copy, worker cancellation and 50-row pages. Reuse semantic
  CSS and SafeModalDismissMixin. No new dependencies or execution controls.
- [x] Wire the existing tail event in ConsoleLeftRail via a named late-bound
  controller callback; do not add controller logic to chat_screen.py.
- [x] Extend Inspector section with an opt-in visible-row limit, keeping its
  full state and summary; scroll only after expansion, not during sync.
- [x] Keep the history entry reachable when the selected conversation's
  latest primary has no children; verify selecting an older saved child.
- [x] Run targeted DB/bridge/section/fleet/history UI checks, inspect painted
  text and hit targets at wide and compact sizes, regenerate CSS, check scoped
  formatting/lint/whitespace and update guide, ledger, task notes and criteria.

No full suite, live provider calls, or commits are included in this repair pass.
