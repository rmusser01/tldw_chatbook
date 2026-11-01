# ADR-240: Retire the MCP destination; Settings hosts the Tools & MCP hub

Status: Accepted
Date: 2026-10-16
Related Task: task-35150
Spec: Docs/superpowers/specs/2026-10-15-tools-hub-in-settings-design.md
Amends: ADR-015 (destination set 14 -> 13; fold map gains mcp/tools_settings -> Settings)
Supersedes: the `mcp-defaults` Settings domain contract's "server operations stay
in MCP" stance (the contract object is deleted with the category)
Related: ADR-172 (retirement template reused), ADR-148 (workbench IA preserved
unchanged), ADR-031/ADR-152 (shortcut lockstep for removing ctrl+9)

Renumber note (2026-10-18): originally drafted under number 233; renumbered to
237 on 2026-10-16 because 233-236 are claimed on sibling branches (served-web
work: 233 session persistence/device pairing per `ce8f2d6dcd`, 234 mobile
companion, 235 WebAuthn passkey login, 236 canvas relay message types — spec
amendment A5). 237 was then contested by open PR #3157 (the Kanban work also
claims number 237) and dev has since merged 238 and 239 (canvas ADRs:
`238-offline-authenticated-canvas-build-inputs.md`,
`239-canvas-sync-replica-and-lifecycle-contract.md`), so this ADR was
renumbered again to 240 — the next free number. All references in
spec/plan/task/docs/code were swept to 240; re-swept at merge time 2026-10-18
(full reference sweep across Docs/, backlog/, tldw_chatbook/, Tests/ came back
clean).

## Decision

Retire the MCP shell destination. Settings gains one "Tools & MCP" category
(`SettingsCategoryId.TOOLS = "tools"`) that hosts the existing four-mode
`MCPWorkbench` intact inside a new pane module
(`UI/Screens/settings_tools_hub.py`); `UI/Screens/mcp_screen.py` is deleted
after its shell logic (mode chips, per-mode footer hints, exclusive reload
worker, bounded deep-link context) is ported to the pane.

Route and hotkey compatibility follow the ADR-172 template: `mcp` and
`tools_settings` resolve to Settings (screen-registry alias + shell route
map), carry default context `{"category": "tools"}`, and ctrl+9 survives as
`MCP_COMPATIBILITY_SHORTCUT` deep-linking Settings -> Tools & MCP. The
Settings palette command gains "tools"/"mcp" alias terms.

Settings keybindings follow the screen's own per-category dispatch
precedent: `1-4` (mode switch) are new self-guarded bindings; `a`/`r`/`t`
gain a TOOLS branch (add server / force reload / test tool) while every
other category keeps today's behavior.

## Context

The destination is protocol jargon behind a whole top-level slot; its own
copy says most people never need it. Users looking for tools in Settings
found only Tool Profiles (lifecycle) and a read-only MCP Defaults pointer.
The prior in-code contract warned against "flattening tools into Settings" —
that objection was about flattening tool config into form categories; this
decision re-homes the workbench intact and deletes the pointer card instead.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Config/operations split (Settings takes permissions only) | Splits "where do I manage tools?" across two surfaces and dissects the workbench |
| Hub-and-pointer (link better, change nothing) | That is the status quo; judged not easy enough |
| Keep the destination, add a Settings alias | Two homes for one surface; the jargon destination remains |

## Consequences

- `SHELL_DESTINATION_ORDER` drops to 13 (14 pre-retirement — Research and
  Meetings postdate ADR-015's "13"); ctrl+9 leaves
  `SHELL_DESTINATION_SHORTCUTS` in lockstep (ADR-031/152).
- ADR-148's rail IA, recovery review, and audit drill-down are unchanged;
  only the host screen changes.
- The Tool Profiles "edit policy" handoff becomes an internal Settings
  category switch with the same bounded context.
- Startup configs with `default_tab = "mcp"` still land on Settings -> Tools
  & MCP via the alias + default context.
- Follow-up UX direction (per-gate effect-scope labels, a one-line
  gate-vs-permission explainer, a plain-language pass on the four modes) is
  recorded in the spec's Amendments A7 and is deliberately outside this
  decision's scope.

## Links

- Spec: Docs/superpowers/specs/2026-10-15-tools-hub-in-settings-design.md
- [ADR-015: shell destination IA](015-shell-destination-ia.md)
- [ADR-172: retirement template](172-library-artifacts-browse-and-navigation.md)
- [ADR-148: workbench rail IA](148-mcp-hub-rail-ia-and-responsive-triad.md)
- [ADR-031: keybinding and footer conventions](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-152: nav-layer reservation](152-roleplay-mode-keys-single-letters-nav-layer-reserved.md)
