# Design-system completion ledger — 2026-09-17

The token/component migration is implemented. The user's subsequent request to
review each feature/component remains active. This ledger separates those two
contracts so that another successful bounded review cannot accidentally close
the whole workstream.

Saved work: draft [PR #2704](https://github.com/rmusser01/tldw_chatbook/pull/2704),
`feat/component-pattern-library` → `dev`. TASK-32749 reconciles dev
`1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6`; the earlier integration of
`fd30614dcdc` remains historical evidence. No merge into dev or full-suite run
is authorized by this ledger.

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

Fresh integration verification: 280 distinct affected/governance cases pass in the
[TASK-32749 receipt](../qa/2026-09-17-component-current-dev/README.md).
Boot CSS is 620,062 bytes, below the unchanged 634,050-byte limit. Independent
review found no integration-specific blocker; native inspection found the incoming
empty Notes toolbar clipping subsequently repaired by TASK-32752, with 62 targeted
checks and four inspected native captures in its [QA receipt](../qa/2026-09-17-notes-empty-toolbar/README.md).

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
| Settings: generation defaults | TASK-32748: finite validation, retained disclosure/draft, visible compact controls and same-field focus after reflow | Qualified by 173 distinct targeted checks and four final native journeys; [QA receipt](../qa/2026-09-17-settings-generation-defaults/README.md) states provider and failure-injection limits |
| Settings: Overview front door and Web Search | TASK-32753: full compact Overview status/actions, visible keyboard links and resize focus; masked search drafts, Save/Revert, safe failure/retry and saved-only tests | 76 distinct targeted cases, all preflight checks and four final private native journeys; [QA receipt](../qa/2026-09-17-settings-overview-web-search/README.md) records real loopback HTTP and external-service limits |
| Settings: Speech/TTS global defaults and guarded navigation | TASK-32756: visible horizontal/stacked fields and Voice/Browse controls, complete leave-dialog actions, custom IDs and exact local saves | 68 targeted checks and four private native journeys; [QA receipt](../qa/2026-09-17-settings-speech/README.md) records ownership and runtime limits. Provider-specific credential/managed-package dialogs and realtime tuning still need their own bounded review. |
| Settings: Appearance, Theme and Splash | TASK-32757: visible compact values/actions, distinct draft/theme/instant saves, launch-default handoff and pending-write recovery | 130 Settings/governance cases and four native journeys with 28 inspected captures; [QA receipt](../qa/2026-09-17-settings-interface/README.md) records animation/runtime limits and 102 additional integration cases. |
| Settings: Console rail-label control | TASK-23150 confirms and repairs below-fold test interactions; TASK-32759 fits the full checkbox inside the compact pane | 45 distinct targeted cases and four native terminal cells; [QA receipt](../qa/2026-09-17-settings-console-rail/README.md) records exact bounds. Broader Console controls and Storage are addressed in the next bounded review below. |
| Settings: Console controls and Storage defaults | TASK-32761: readable compact fields; ordered immediate-toggle save/recovery across navigation and config reload; staged validation/Revert/retry; non-mutating Storage checks and next-launch saves | 92 distinct targeted cases; [QA receipt](../qa/2026-09-17-settings-console-storage/README.md) records native matrix and limits. Permission summaries, thinking-visibility departure, exchange capture, budgets/background/context workflows remain separate behavioral gates. |
| Remaining Settings categories and modal patterns | Prior compact Providers/Network geometry is qualified, not all category behavior | Review remaining Speech/TTS flows, Workspaces, Tool Profiles, Privacy, Network, Personal Context, Console Behavior, Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP/ACP defaults, Diagnostics/About, Advanced Config, Internal Prompts, Image/Video Generation and Agents; modal workflows including Backup & Restore, source switching and manual sync remain separate gates |
| Roleplay, Watchlists, Artifacts, Schedules, Workflows, MCP, ACP, Lab, Logs, Research, Meetings | No whole-destination completion claim in this workstream | Bounded keyboard/state/resize/theme review, fix confirmed defects, and record representative native evidence |
| Current dev integration | TASK-32749 reconciles 71 commits through `1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6`; generated artifacts, token floors, 280 cases, independent review and four native cells qualified | [Integration report](2026-09-17-component-current-dev-integration.md) records exact bounds; subsequent dev changes require their own review |
| Empty Notes folder actions | TASK-32752 corrects the available-width/chrome budget and repacks only overflowing composed rows; preserves growth and legacy-list identity | 62 targeted checks and four inspected native captures qualify the repair; [QA receipt](../qa/2026-09-17-notes-empty-toolbar/README.md) states exact limits |

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
native journey for the broader review the user approved. Current-dev integration
and the discovered Notes toolbar repair are saved; next review the remaining
Settings categories and remaining Console subflows, then the remaining destinations. Their rows remain visible in this ledger until assessed.

TASK-32757 also reconciles the four incoming dev commits through `c97a64eba5`.
Both Import guide additions are preserved. Select validation now participates in
the retained option update path, with 102 targeted integration cases passing;
no new ingestion native or external-service qualification is claimed.

TASK-32759 also reconciles dev `d8fb4053f9` after the draft PR became conflicting.
Both lesson additions and the incoming parent-side STT diagnostics are preserved;
15 affected STT tests and all seven derived-artifact guards pass. The native rail
captures predate that callback-only merge; the receipt states this boundary.

The [conflict review](2026-09-17-pr-2704-conflict-review.md) reconstructs all four dev integration merges through `96d4ca2b96`: 20 conflicted file entries / 30 blocks, exact parent references, selected outcomes and recorded native captures. The user requires visual review and explicit green light before the PR is merged into dev.
