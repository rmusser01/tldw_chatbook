# Design-system completion ledger — 2026-09-17

The token/component migration is implemented. The user's subsequent request to
review each feature/component remains active. This ledger separates those two
contracts so that another successful bounded review cannot accidentally close
the whole workstream.

Saved work: draft [PR #2704](https://github.com/rmusser01/tldw_chatbook/pull/2704),
`feat/component-pattern-library` → `dev`. The last checked PR state has conflicts.
The earlier integration of `fd30614dcdc` is historical evidence, not evidence of
compatibility with current dev. No merge into dev or full-suite run is authorized
by this ledger.

ADR required: no. Existing ADR-150/161 define the language and migration;
ADR-031 governs interaction and ADR-097 governs startup budgets. This document
records evidence and remaining review work, without changing those boundaries.

## Original migration requirements

Source of requirements: the approved
[spec](../specs/2026-09-13-component-pattern-library-design.md),
[plan](../plans/2026-09-13-component-pattern-library.md), and TASK-32596.
The original [closeout](2026-09-14-component-pattern-library-closeout.md) records
implementation, deviations, baseline failures and native gallery limitations.

| Requirement | Implementation/evidence | Completion gate |
| --- | --- | --- |
| Nine component families share a documented contract | `backlog/docs/component-patterns.md`, `tldw_chatbook/css/patterns.json`, Pattern Gallery; catalog/registry governance | Implemented; recheck governance after integration |
| Canonical ownership and deprecated-pattern ratchets | `Tests/UI/test_component_pattern_governance.py`; zero allowances are not raised to hide debt | Implemented; fresh targeted checks recorded in the current QA receipt |
| Agentic monolith carved into owning sheets, ≤2,000 active lines each | CSS_MODULES inventory and sheet-ceiling check; TASK-24451 superseded by TASK-32596 | Implemented; recheck source/build after integration |
| No numeric source-sheet dimensions or non-token hex colors | Governance covers authoring TCSS plus all active CSS_MODULES inputs, including `stats_screen.css` | Implemented; fresh negative cases and ratchets remain required |
| No ad hoc Python visual values in the approved property set | AST inventory covers assignments, `set_styles`, `setattr` and opaque input | Implemented within approved scope; explicit runtime geometry/user values and `None` releases remain exceptions |
| Dark/light gallery makes component rendering reviewable | `Tests/UI/snapshots/pattern_gallery/{dark,light}.svg`; layout/snapshot tests; historical native gallery evidence | Implemented; not a claim of visual parity for every feature |
| Generated CSS and boot budget remain valid | Bundle sync, selector/comment integrity and boot byte tests; 634,050-byte ceiling, 600,000-byte anti-vacuity floor | Must stay green after every style change and integration |

Fresh verification for this checkpoint: 193 adjacent/governance cases pass in the
[TASK-32746 regression receipt](../qa/2026-09-17-settings-model-catalog/regressions.txt).
Boot CSS is 616,683 bytes, below the unchanged 634,050-byte limit.

The approved Python property set is `background`, `color`, `border*`, `width`,
`height`, `padding*`, `margin*`, and `opacity*`. Min/max dimensions and display/
layout are outside that original migration; the conservative inventory is not
whole-program dataflow analysis. A wider property migration would require an
explicit scope decision, not a misleading zero count.

## Approved component/feature review

The sequence comes from the
[first component audit](2026-09-14-component-first-ui-audit.md). “Reviewed” below
means the recorded workflows and fixture bounds, never every possible backend
or permission configuration.

| Surface | Qualified work | Remaining gate |
| --- | --- | --- |
| Shared navigation, compact forms and gallery | TASK-32592–32595; [repair report](2026-09-14-component-audit-fixes.md) | Regression checks after integration |
| Library: Workspaces, Notes, Media, Conversations, Collections, Search/RAG and ingest journeys | [Workflow audit](2026-09-14-library-workflow-audit.md) links each bounded repair and native receipt | Preserve the individual reports' service/network limits; reconcile open residuals before whole-workstream closeout |
| Settings: provider editing and saved defaults | TASK-32724 and TASK-214: keyboard Save/Revert, ownership, return and real private config persistence | Generation/provider availability are not qualified by these checks |
| Settings: model discovery | TASK-32739: checked-row retention, save/clear feedback, stale completion ownership | External provider availability is outside this UI slice |
| Settings: automatic refresh | TASK-32746: ordered instant saves, validation, retained choices, truthful Retry, compact labels | Qualified by 223 targeted checks and four native journeys; [QA receipt](../qa/2026-09-17-settings-model-catalog/README.md) states fault-injection and backend limits |
| Remaining Settings categories and modal patterns | Prior compact Providers/Network geometry is qualified, not all category behavior | Inventory and review Overview, Web Search, Speech/TTS, Appearance/Theme/Splash, Storage, Workspaces, Tool Profiles, Privacy, Network, Personal Context, Console Behavior, Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP/ACP defaults, Diagnostics/About, Advanced Config, Internal Prompts, Image/Video Generation and Agents |
| Roleplay, Watchlists, Artifacts, Schedules, Workflows, MCP, ACP, Lab, Logs, Research, Meetings | No whole-destination completion claim in this workstream | Bounded keyboard/state/resize/theme review, fix confirmed defects, and record representative native evidence |
| Current dev integration | PR saved; earlier dev reconciliation is documented | Resolve current conflicts, preserve upstream ownership, rebuild, run affected checks, inspect integrated native app |

## How to close the workstream

For each remaining surface, first inspect the existing implementation, tests and
Backlog task. Choose representative journeys covering visible focus, compact
layout, empty/loading/error/recovery, persistence/cancellation and theme changes.
Create an atomic repair task only for a confirmed gap; preserve existing ADRs
unless an actual boundary decision changes. Targeted tests are the default.
A full repository sweep requires explicit user opt-in.

Before declaring completion, reconcile every remaining row with linked evidence,
check that PR integration is reviewable, and report any documented backend limits.
Do not substitute a Done parent task, passing governance, or a single successful
native journey for the broader review the user approved. The next bounded review
after automatic refresh is provider generation defaults; other categories and
destinations remain visible in this ledger until assessed.
