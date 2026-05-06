# Screen Design Adaptation Audit

Date: 2026-05-06
Status: Design audit and implementation gate
Primary Repo: `tldw_chatbook`
Current Base: `origin/dev` at `6d4389fa` (`Verify Phase 3.3 Library contract layout (#263)`)
Scope: Current top-level destination screens after the Phase 3.3 Library contract merge

## Summary

The current shell has the approved top-level information architecture and a verified global navigation model. It does not yet have fully adapted screen designs for every destination.

`Library` is the only destination that substantially follows the approved destination layout grammar today. `Home` and `Console` have meaningful foundations but still need structural adaptation. Most other destinations are honest wrapper or skeleton screens: they explain ownership, expose some recovery states, and preserve route compatibility, but they are not yet designed as complete usable destination screens.

This document should be used as the screen-design gate before deeper feature work continues. A screen should not be considered product-mature just because it renders, has clickable buttons, or passes shallow navigation tests. It must expose a usable layout, clear state, recoverable failure modes, and correct Console handoff behavior.

## Evidence Reviewed

- Approved master shell UX spec: `Docs/superpowers/specs/2026-05-02-new-user-first-run-shell-ux-design.md`
- Approved design-system spec: `Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md`
- Destination layout contract: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`
- Route inventory: `Docs/Design/master-shell-route-inventory.md`
- Current destination metadata: `tldw_chatbook/UI/Navigation/shell_destinations.py`
- Current screen implementations under `tldw_chatbook/UI/Screens/`
- Mounted Textual harness review at `140x42` for `Home`, `Console`, `Library`, `Artifacts`, `Personas`, `W+C`, `Schedules`, `Workflows`, `MCP`, `ACP`, `Skills`, and `Settings`

## Maturity Labels

| Label | Meaning |
| --- | --- |
| Adapted | The screen substantially follows the destination contract: header, status/authority, local modes/scope, primary object/workspace/inspector regions, recoverable states, and Console handoff where applicable. |
| Partial | The screen has correct product ownership and some real controls or recovery states, but its layout is not yet a complete destination design. |
| Skeleton | The screen mostly renders static orientation, placeholder sections, or disabled controls. It may be honest, but it is not yet usable for its core workflow. |

## Global Findings

1. The shell IA is now coherent, but screen-level IA is uneven. Users can reach the right destination but may not know what to do after arriving.
2. `Console` still behaves visually like legacy Chat plus readiness cards, not yet like the primary agentic control surface.
3. `Library` proves the destination contract can work in Textual, but the same grammar has not been applied across the rest of the product.
4. Recovery copy has improved, but many screens stop at "unavailable" without offering an in-screen setup path, object selection path, or concrete next action.
5. Most destination headers lack a compact status/authority row. This weakens visibility of system status and source/runtime authority.
6. Several screens expose product concepts as static section labels instead of mode bars, lists, detail panes, and inspectors. This passes recognition tests but not workflow completion tests.
7. Current QA is strongest for shell navigation and selected workflows. It needs screen-design gate tests that assert layout regions, actionable states, and first-focus behavior.

## Screen Maturity Matrix

| Destination | Current State | User Impact | Required Design Adaptation |
| --- | --- | --- | --- |
| Home | Partial | High | Convert from vertical summary sections into dashboard regions: attention queue, active work, selected-item inspector, next-best actions, recent/resume. |
| Console | Partial | Critical | Reframe from legacy Chat wrapper into agentic live-work surface with staged context tray, transcript/event stream, composer, run/tool/approval inspector, and artifact controls. |
| Library | Adapted with workflow gaps | High | Keep the three-region shell; make modes actionable and progressively replace outward legacy routing with Library-native source/search/study flows. |
| Artifacts | Skeleton/Partial | High | Add artifact filters, artifact list, preview/detail, provenance inspector, Chatbook reopen/export/bundle actions. |
| Personas | Partial | Medium | Add Personas/Characters/Prompts/Dictionaries/Lore modes, profile/detail editor, import/export, and attachment readiness inspector. |
| W+C | Partial | Medium | Add Watchlists and Collections tabs, object list, item/run detail, status/history/retry inspector, and Console follow/stage actions. |
| Schedules | Skeleton | Medium | Add schedule list, upcoming/detail/history workspace, pause/resume/retry controls, and selected-run Console handoff. |
| Workflows | Skeleton | Medium | Add workflow list, builder/run detail workspace, dry-run/readiness inspector, approvals, and launch/follow in Console. |
| MCP | Partial | Medium | Wrap `UnifiedMCPPanel` in destination contract framing: status/authority row, mode clarity, and readiness/permission/audit hierarchy. |
| ACP | Skeleton | Medium | Add runtime/agent/session modes, runtime setup detail, compatibility inspector, and launch/follow states. |
| Skills | Partial | Medium | Add installed/discover/import/validate/attach modes, skill list, `SKILL.md` detail/file tree, validation inspector, and attachment controls. |
| Settings | Skeleton | Medium | Add category list, setting form/detail, diagnostics/impact inspector, save/revert/test controls. |

## Destination Findings

### Home

Current evidence: `HomeScreen.compose_content()` renders the title, purpose, dashboard sections, primary action, and controls in one vertical stack. The mounted empty/default state shows `Status`, `Attention`, `Active Work`, `Next Best Action`, and `Recent Work`.

Assessment: Partial.

Design issue: Home has the right data model but not the right layout. It should work as a dashboard and control center, but today it reads as a report. Active work controls can exist, but the selected target and its details are not spatially clear enough.

Required adaptation:

- Add a status/authority row after the destination purpose.
- Add a local scope/filter row for `All`, `Needs attention`, `Running`, and `Recent`.
- Split the body into `Attention Queue`, `Active Work`, and `Inspector`.
- Keep `Next Best Actions` and `Recent Work` visible without forcing users through a modal.
- Preserve dedicated Chatbook artifact controls when mixed active work is present.

Acceptance checks:

- A mixed watchlist-run plus Chatbook artifact scenario exposes both targets.
- A selected active item has visible owner, impact, current state, and available controls.
- Empty state tells the user the next safe action without pretending the system is ready.

### Console

Current evidence: `ChatScreen.compose_content()` renders a live-work launch/status card or source-readiness card before mounting `ChatWindowEnhanced`. The mounted default state shows live-work source readiness plus many legacy Chat settings controls.

Assessment: Partial, with critical product risk.

Design issue: Console is the product center, but the current composition still visually prioritizes Chat settings/sidebar complexity. It does not yet clearly separate staged context, transcript/event stream, composer, run inspector, approvals, and artifacts.

Required adaptation:

- Add a Console destination header with readiness, workspace/source authority, and primary action.
- Introduce a stable Console shell grid: staged context tray, transcript/event stream, run inspector, composer/action row.
- Promote live-work source readiness into the inspector or a readiness lane rather than making it compete with the conversation surface.
- Make RAG/source usage visible in Console before send.
- Keep legacy `ChatWindowEnhanced` behavior behind the new shell until it can be incrementally decomposed.

Acceptance checks:

- A user can tell what sources are staged before sending.
- A user can identify whether a response will use RAG, tools, persona, MCP, or ACP context.
- A blocked model/provider state shows cause, impact, and setup route.
- Live-work launches from Home/W+C/Schedules/Workflows/Artifacts visibly preserve provenance.

### Library

Current evidence: `LibraryScreen.compose_content()` now renders `#library-status-row`, `#library-mode-bar`, `#library-contract-grid`, `#library-source-browser`, `#library-source-detail`, and `#library-source-inspector`.

Assessment: Adapted, with workflow gaps.

Design issue: The structural shell is correct, but the modes are not yet fully interactive destination-native flows. Several actions still route to legacy screens rather than presenting source selection, Search/RAG, Import/Export, Workspaces, Study, Flashcards, and Quizzes as Library-owned modes.

Required adaptation:

- Keep the current three-region shell as the template for other destinations.
- Make mode chips actionable where feasible.
- Add selected-source state and make the inspector reflect the selected item.
- Keep outward legacy routes only as compatibility paths while the Library-native modes mature.

Acceptance checks:

- Source selection changes detail and inspector content.
- Search/RAG is usable from Library without knowing the legacy `search` route.
- Import/Export copy stays source-oriented and does not blur with Artifacts exports.
- Flashcards and Quizzes remain visible as Library-owned study paths.

### Artifacts

Current evidence: Artifacts shows the correct destination purpose, an `Open Chatbooks` action, generated-output status text, and Console-launch recovery copy when no local Chatbook artifact exists.

Assessment: Skeleton/Partial.

Design issue: Artifacts is not yet an output hub. It lacks type filters, an artifact list, selected preview, provenance, and export/bundle/reopen actions.

Required adaptation:

- Add type filters: `All`, `Chatbooks`, `Reports`, `Datasets`, `Drafts`, `Exports`.
- Add artifact list, preview/detail, and provenance inspector.
- Keep Chatbooks first-class, but avoid making Artifacts a renamed Chatbooks screen.
- Distinguish raw Library sources from generated/reusable outputs.

Acceptance checks:

- Empty state explains how artifacts are created or imported.
- A Chatbook artifact can reopen in Console with source provenance.
- Export/bundle actions show target and recovery when unavailable.

### Personas

Current evidence: Personas shows purpose, ownership boundary copy, a local snapshot/loading/error state, `Open Personas`, and `Attach to Console`.

Assessment: Partial.

Design issue: Personas correctly owns behavior and identity, but the screen does not yet expose the mode structure needed to distinguish profiles, characters, prompts, dictionaries, and lore.

Required adaptation:

- Add mode bar: `Personas`, `Characters`, `Prompts`, `Dictionaries`, `Lore`, `Import/Export`.
- Add profile/character list, detail/edit preview, and attachment inspector.
- Show attachment targets for Console, Workflows, ACP, and Skills.
- Preserve policy and service recovery taxonomy in status rows.

Acceptance checks:

- A user can tell whether they are editing behavior profile, character data, prompt text, or lore.
- Attach-to-Console shows selected persona summary and disabled reason when unavailable.
- Import/export failures are recoverable.

### W+C

Current evidence: W+C shows Watchlists and Collections explanatory sections, local snapshot/recovery states, staging to Console, `Open current Watchlists`, and follow-in-Console controls.

Assessment: Partial.

Design issue: The destination explains the two domains but does not yet make them operationally distinct as tabs with lists, details, run history, and status/retry controls.

Required adaptation:

- Add true internal tabs for `Watchlists` and `Collections`.
- Add object list, detail/items/runs workspace, and status/history inspector.
- For Watchlists, expose run state, retry/backoff, alert state, latest output, and Console follow.
- For Collections, expose item state, highlights, saved searches, archive state, note links, and Library/RAG/Console handoff.

Acceptance checks:

- Watchlists and Collections are visibly distinct.
- Watchlist run failure shows cause, retry/backoff, and follow/retry target.
- Collection items can feed Library/RAG or Console without becoming Artifacts by default.

### Schedules

Current evidence: Schedules renders purpose, placeholder sections for Next Run/Paused/Failed/Retry, and Console recovery unavailable copy.

Assessment: Skeleton.

Design issue: The screen states that Schedules owns when things run, but it does not yet provide a timing-focused workflow.

Required adaptation:

- Add filters for `Active`, `Paused`, `Failed`, `Upcoming`.
- Add schedule list, schedule detail/upcoming runs, run history, and control inspector.
- Make pause/resume/retry target the selected schedule.
- Keep workflow procedure editing out of Schedules.

Acceptance checks:

- A user can distinguish schedule timing from workflow procedure.
- Failed or missed schedule states explain cause and recovery.
- Live or failed runs open in Console with schedule provenance.

### Workflows

Current evidence: Workflows renders purpose, placeholder section labels for Recipes/Inputs/Steps/Dry Run/Approvals/Outputs, and Console launch recovery copy.

Assessment: Skeleton.

Design issue: The screen states that Workflows owns what procedure runs, but it does not yet support browsing, building, dry-running, or inspecting procedure readiness.

Required adaptation:

- Add mode bar: `Browse`, `Build`, `Runs`, `Templates`.
- Add workflow list, builder/run detail workspace, and run/readiness inspector.
- Expose inputs, tools, skills, persona, approvals, dry-run, and output targets.
- Launch/follow live work only through Console.

Acceptance checks:

- Dry-run/readiness is visible before launch.
- Approval points and output targets are visible.
- A workflow launch opens Console with workflow/run context.

### MCP

Current evidence: MCP embeds `UnifiedMCPPanel` under a destination title and purpose. The mounted state exposes local/server/scope controls and a disabled run action.

Assessment: Partial.

Design issue: MCP has the most real functionality after Library and Console, but the destination wrapper needs clearer contract framing. Some copy still references "Tools & Settings", which weakens the approved boundary that MCP is not global Settings.

Required adaptation:

- Add MCP status/authority row at the wrapper level.
- Ensure mode labels map to `Servers`, `Tools`, `Resources`, `Permissions`, `Audit`.
- Treat readiness, auth, permission, and risk as inspector concepts.
- Remove or revise copy that implies MCP lives under Tools & Settings.

Acceptance checks:

- `tools_settings` resolves as MCP, not Settings.
- Tool readiness and permission status are visible before use.
- Blocked tools show owner, impact, and recovery.

### ACP

Current evidence: ACP shows purpose, static section labels for installed agents/sessions/resume/diffs/terminal, and honest runtime-unconfigured recovery copy with disabled launch/follow controls.

Assessment: Skeleton.

Design issue: ACP is correctly separate from MCP, but it is not yet a runtime/session management screen.

Required adaptation:

- Add modes: `Agents`, `Sessions`, `Runtimes`, `Compatibility`.
- Add agent/session list, runtime setup/detail workspace, and compatibility inspector.
- Show missing runtime setup steps in-screen.
- Launch/follow live ACP sessions through Console only.

Acceptance checks:

- Runtime-unconfigured state is honest and recoverable.
- ACP and MCP purposes remain visibly distinct.
- Follow-in-Console is disabled with a target-specific reason until session payloads exist.

### Skills

Current evidence: Skills shows Agent Skills ownership, local skills directory, static sections for installed/discover/import/validate/scripts/references/assets/attachments, empty state, disabled import, and disabled Console attachment.

Assessment: Partial.

Design issue: The Agent Skills model is clear, but the screen does not yet provide the core skill-management workflow: list, inspect `SKILL.md`, validate, edit/import, and attach.

Required adaptation:

- Add mode bar: `Installed`, `Discover`, `Import`, `Validate`, `Attach`.
- Add skill list, `SKILL.md`/file-tree detail, and validation/compatibility inspector.
- Show frontmatter validity, allowed tools, compatibility, scripts, references, and assets.
- Add recoverable import and attach states.

Acceptance checks:

- User can identify valid vs invalid skills before attachment.
- `SKILL.md` frontmatter and bundled directories are visible.
- Attachment targets are explicit and show compatibility blockers.

### Settings

Current evidence: Settings shows purpose, static category labels, boundary copy that MCP/tool-control settings live under MCP, and an `Open Appearance` action.

Assessment: Skeleton.

Design issue: Settings has the right boundary but no usable settings structure.

Required adaptation:

- Add category list: `Providers`, `Models`, `Storage`, `Privacy`, `Appearance`, `Diagnostics`.
- Add setting form/detail workspace and diagnostics/impact inspector.
- Add save, revert, test, and diagnostics controls.
- Keep MCP, ACP, Skills, Personas, Schedules, and Workflows configuration out of global Settings except for global defaults.

Acceptance checks:

- Validation errors are local to the setting and include recovery.
- Provider readiness explains downstream Console impact.
- Settings does not duplicate task-specific configuration owned by other destinations.

## Recommended Adaptation Gates

### Gate 1: Core Product Loop

Screens: `Console`, `Home`, `Library`.

Why: These define the first-run and repeated-use loop: understand status, stage/ask/control live work, and browse/search/source knowledge.

Done when:

- Home has dashboard regions and selected-item inspector.
- Console has staged context, transcript/event stream, composer, and live-work inspector.
- Library keeps its current contract layout and gains actionable mode/selection behavior for the next Knowledge/Study slice.
- QA walkthrough verifies that the app is usable, not merely renderable.

### Gate 1.5: Console Internals Decomposition

Screen: `Console`.

Why: Gate 1 is allowed to wrap the existing `ChatWindowEnhanced` surface for compatibility. That cannot be the end state. The product center must look and behave like one coherent agentic Console, not a new shell wrapped around an out-of-place legacy Chat implementation.

Done when:

- `ChatWindowEnhanced` has been decomposed or replaced by Console-native components for provider/model controls, staged context, transcript/event stream, composer, tools, approvals, RAG controls, artifact/Chatbook save controls, and recovery states.
- Existing chat behavior remains compatible: basic chat, tabs/session state, provider/model selection, streaming/non-streaming fallback, handoffs, RAG-related controls, tool-call visibility, and persona/character attachment paths still work or have documented replacements.
- Visual and interaction QA verifies that the Console internals fit the agentic terminal design system and no longer look like a legacy embedded screen.

### Gate 1.6: Library-Native Search/RAG Workflow

Screen: `Library`, with Console handoff.

Why: Gate 1 is allowed to make `Search/RAG` mode selectable inside Library. That cannot be the end state. Library must support a real retrieval workflow that can answer inside Library and hand evidence into Console.

Done when:

- Library Search/RAG mode includes source selection, query input, retrieval status, evidence/results list, citations/provenance where available, and clear empty/error/setup recovery states.
- Users can start from Library Search/RAG and continue into Console with staged evidence and source authority preserved.
- Users can start from Console and invoke RAG against Library sources with visible retrieval state and cited evidence.
- QA verifies retrieval usability rather than only selector presence or route navigation.

### Gate 2: Knowledge Inputs, Outputs, And Behavior

Screens: `Artifacts`, `Personas`, `W+C`, `Skills`.

Why: These destinations prepare material, outputs, behavior, and capability packs that feed Console and Library.

Done when:

- Each destination has mode/list/detail/inspector structure.
- Each destination has at least one end-to-end Console handoff or explicit unavailable recovery state.
- Empty/error/setup states are tested in mounted UI.

### Gate 3: Operations And Runtime Control

Screens: `Schedules`, `Workflows`, `MCP`, `ACP`, `Settings`.

Why: These screens manage timing, procedures, tools, protocol runtimes, and global configuration. They need stronger readiness and recovery states before they can be treated as mature.

Done when:

- Schedules and Workflows are visually and functionally distinct.
- MCP and ACP remain visibly separate.
- Settings stays global and does not absorb destination-owned configuration.
- Runtime/provider/tool blockers include owner, impact, and next action.

## Cross-Screen Acceptance Checklist

Every adapted destination must satisfy these checks before being marked product-mature:

- Destination header includes title, purpose, status/readiness, and authority/scope where relevant.
- Local modes or filters are visible when the destination has multiple subflows.
- Primary object list, main workspace/detail, and inspector are present or the spec documents why the screen is intentionally simpler.
- Empty, loading, blocked, error, and missing-runtime/provider states are recoverable where applicable.
- Disabled controls explain why they are disabled and what to do next.
- Console handoff preserves provenance, selected object, source authority, and recovery state.
- Keyboard focus path supports repeated use without mouse-only interaction.
- Mounted UI tests verify visible layout regions and at least one empty/error/setup state.
- QA walkthrough confirms the screen is actually usable, not only clickable.

## Recommended Next Step

Create implementation tasks from this audit in the gate order above. The first implementation plan should focus on `Console`, `Home`, and the next `Library` mode-selection slice, because these screens define the core product loop and will set the reusable pattern for the remaining destinations.
