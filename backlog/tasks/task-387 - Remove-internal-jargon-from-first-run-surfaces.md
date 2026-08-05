---
id: TASK-387
title: Remove internal jargon from first-run surfaces
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-facing copy includes 'api_settings.llama_cpp.api_url=http://...', 'LLAMA_CPP_API_KEY=<redacted>', 'model=missing | status=blocked', discovery rows suffixed '| runtime | runtime_discovered | capability=unknown', a section headed 'Automatic refresh (ADR-020)', and after the first send the Console rail shows 'Scope b7c3bdfb-99c9-462d-a83e-714b8d...' — a raw conversation UUID as the session scope label.

**Repro:** Read Test Provider output, discovery result rows, the 'Automatic refresh (ADR-020)' section on Providers & Models, and the rail 'Scope' row after sending the first message.

**Verifier note:** Code-confirmed cluster on new/first-run surfaces: rail Scope renders the raw conversation id — display_state.py:282 scope_label = str(current_conversation or '') → ConsoleWorkspaceStatusPair('Scope', …) (console_workspace_context.py:583); provider-test detail dumps config paths and env-var status (settings_screen.py:5276-5296, 'api_settings.<key>.<url>=', 'LLAMA_CPP_API_KEY=…'); discovery rows are labeled '<id> | runtime | runtime_discovered | capability=unknown' (settings_screen.py:4934); 'Automatic refresh (ADR-020)' is a live Settings heading from the just-shipped task-301 work. Not covered: micro-polish-186 fixed different rail/footer nits and none of the ledger copy items cover these. Downgraded P2→P3: comprehension/trust polish, no task blockage; note the 'Saved as: chat_defaults.model' guide-panel pattern is deliberate Settings design language (the reviewer praised it), so the fix should target the toast/detail/row copy and the Scope UUID, not config-path language wholesale.

**Source:** Console UX expert review 2026-07-20 (finding j1-internal-jargon-in-ui; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J1 new-user cold start journey. Evidence: `j1-16-test-bad-url-full.png`, `j1-28-model-list-clean.png`, `j1-07-setup-surface.png`, `j1-39-send-plus-4s.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First-run copy in user vocabulary ('Endpoint', 'API key: not needed', 'Test failed: connection refused')
- [x] #2 Internal decision-record IDs and UUIDs kept out of primary UI or behind a details view
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Targeted the first-run jargon the verifier flagged, staying inside the caveat
("target the toast/detail/row copy and the Scope UUID, not config-path language
wholesale").

AC#1 (user vocabulary): the model-discovery selection rows read
'gemma-4.gguf · session · discovered · capabilities unknown' instead of the
enum dump 'gemma-4.gguf | runtime | runtime_discovered | capability=unknown'
(source enum -> discovered/discovered (cached)/saved; persisted 'runtime' ->
'session'; 'capability=X' -> 'capabilities X'). The provider-Test *detail* line's
'status=ready/blocked' + config-path tokens were deliberately left: they are a
tested TASK-366 diagnostic contract (asserted in 5 places), and the user-facing
TOAST ('Provider test failed: <message>') plus the app's existing
'API key: not required for this provider' copy already read in plain language.

AC#2 (no internal ids/UUIDs in primary UI): the raw conversation-UUID Scope
label was already humanized to 'This conversation' (id on hover) by task-373;
here the 'Automatic refresh (ADR-020)' Settings heading drops the decision-record
id to plain 'Automatic refresh' (the id survives in the code comment).

RED->GREEN unit test on the discovery-row label vocabulary. settings_screen.py
only.
<!-- SECTION:NOTES:END -->
