# Design-system completion ledger — 2026-09-17

The token/component migration is implemented. The user's subsequent request to
review each feature/component remains active. This ledger separates those two
contracts so that another successful bounded review cannot accidentally close
the whole workstream.

Saved work: [PR #2704](https://github.com/rmusser01/tldw_chatbook/pull/2704)
merged into `dev` at `e89f28d751bc8a5b4f4545b8894b87437252c657`
after the owner's visual/conflict approval and passing required checks.
TASK-32749 reconciles dev
`1c0327b3bb3d95b61e3e1b9a83b30e7030453ad6`; the earlier integration of
`fd30614dcdc` remains historical evidence. No merge into dev or full-suite run
is authorized by this ledger alone. That approval covered PR #2704; future
PRs retain their own merge gate. No full-suite run was requested.

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
| Settings: Console controls and Storage defaults | TASK-32761: readable compact fields; ordered immediate-toggle save/recovery across navigation and config reload; staged validation/Revert/retry; non-mutating Storage checks and next-launch saves | 92 distinct targeted cases; [QA receipt](../qa/2026-09-17-settings-console-storage/README.md) records native matrix and limits. Permission-summary and thinking-visibility Settings are addressed below; exchange capture and budgets are addressed below; background/context workflows remain separate behavioral gates. |
| Settings: Permission summaries | TASK-32762: ordered immediate saves, retained failure/retry across screen recreation, runtime config projection and cache-publication recovery | 15 private-profile UI cases plus 52 related checks pass; six older approval-wiring failures reproduce at the saved baseline. Four native cells and 12 inspected captures qualify Settings only; [QA receipt](../qa/2026-09-17-settings-permission-summary/README.md) records exact limits. External approval-summary execution remains unqualified. |
| Settings: Model-thinking visibility | TASK-32763: app-owned immediate-save queue survives Settings departure/recreation; reload rebases the saved value; rollback, retry and cache-warning receipts remain beside the canonical checkbox | 81 distinct targeted checks and four native cells with eight inspected captures; [QA receipt](../qa/2026-09-17-settings-thinking-visibility/README.md) records real private writes, full compact recovery text and clean lifecycle. Mounted Console rows change without altering stored thinking or capture/replay settings; provider execution remains outside this slice. |
| Settings: Full trace-view consent | TASK-32764: real keyboard disclosure, return focus, duplicate/lifetime guards, reload recovery and owner-thread global policy persistence | 59 distinct targeted checks and four final native cells with twelve inspected captures; [QA receipt](../qa/2026-09-17-settings-capture-consent/README.md) records refusal/retry, stale consent, independent PII choice, cancellation settlement and clean lifecycle. Provider execution and live Trace actions remain open. |
| Settings: Console agent-run budgets | TASK-32766: Settings matches the existing Steps cap, rejects over-limit saves without losing the draft, displays the legacy runtime fallback, and paints complete time-unit labels | 121 distinct targeted UI/runtime/governance cases and four native cells with twelve inspected captures pass. [QA receipt](../qa/2026-09-17-settings-agent-budget/README.md) records real private writes and next-run resolution; provider execution remains outside this slice. |
| Settings: Console context and background effects | TASK-32767: complete context label, non-finite FPS fallback, retained validation/drafts/recovery, late-save runtime publication, visible particles below messages, and correct selection/scroll/recomposition/timer lifecycle | 68 distinct targeted cases and four native cells with twelve inspected captures pass. [QA receipt](../qa/2026-09-17-settings-context-effects/README.md) records saved policy agreement and the post-capture attachment-only guard. Provider compaction, streaming with effects and exhaustive animation combinations remain outside this slice. |
| Settings: Workspaces folder controls | TASK-32768: visible local feedback, retained invalid inputs, literal path/access labels, natural workspace-list height and focus on the replacement Add after removal | 37 distinct targeted cases and four final native cells with twelve inspected captures; [QA receipt](../qa/2026-09-17-settings-workspace-folders/README.md) records refusal/retry, registry access, selection, Tab/resize and lifecycle. Workspace assistant defaults and Change Review Settings are qualified below; core lifecycle flows are qualified below. |
| Settings: Workspace assistant field edits | TASK-32769: persona/profile edits preserve the other fields; confirmations remain separate; local receipts follow their action and actual text reflow; bounded pickers retain complete selected text | 87 distinct targeted cases and four native cells with twelve inspected captures; [QA receipt](../qa/2026-09-17-settings-workspace-assistant/README.md) records real private Persona/profile/registry writes, refusal/retry, exact source hashes and clean lifecycle. TASK-32770 now qualifies confirmation lifetime across workspace/category/modal return; native imported-profile review and core lifecycle flows are qualified below; Change Review Settings is qualified below. |
| Library Notes: compact introductory status | TASK-32765: natural text height preserves the complete next action at 80 columns while retaining the work-pane budget | 49 targeted checks, independent review, and eight inspected Console/Notes captures with verified private lifecycle; [refreshed pre-merge visual gallery](../qa/2026-09-17-notes-authority-layout/README.md). Existing optional-prefix timing on resize is unchanged; broader Notes workflows remain separately qualified. |
| Settings: Workspace memory confirmation | TASK-32770: acknowledgements expire on navigation/suspension and reject changed saved/intended defaults; cancellation requires fresh acknowledgement | 95 distinct targeted cases and four native cells with 12 inspected captures; [QA receipt](../qa/2026-09-17-settings-workspace-confirmation/README.md) records exact registry outcomes, visible controls and normal private-profile shutdown. Native imported-profile review and core lifecycle flows are qualified below; Change Review Settings is qualified below. |
| Settings: Workspace Change Review | TASK-32771: pending readiness updates in place, local conflict/retry feedback, preserved drafts/focus, and exact consent intent captured at activation | 73 distinct targeted cases and four native cells with 12 inspected captures; [QA receipt](../qa/2026-09-17-settings-change-review/README.md) records real private shadow Git initialization, bounded retry, retained history, unchanged fixture files and clean lifecycle. Native imported-profile review and core workspace lifecycle flows are qualified below; Console agent-turn diff/review/revert remains separate. |
| Settings: Workspace lifecycle | TASK-32773: local visible Create/Rename/Restore feedback, truthful partial-binding recovery, replacement focus for activation/Undo/Restore, literal labels and a theme-aware Create dialog | 172 distinct targeted cases across final and corrective runs, four native cells and 20 inspected captures; [QA receipt](../qa/2026-09-17-settings-workspace-lifecycle/README.md) records real private registry writes, cancellation, exact-once retry, name-conflict recovery and clean lifecycle. Native imported-profile review is qualified below; Persona auto-creation and project-context interviewing remain separate gates. |
| Settings: Workspace imported-profile review | TASK-32774: Apply intent expires across navigation, suspension and newer staging before delayed review/token publication; failed Clear preserves memory acknowledgement; persona names paint literally | 70 distinct targeted cases across final and corrective runs, four final native cells and 20 inspected captures; [QA receipt](../qa/2026-09-18-settings-workspace-profile-review/README.md) records real unbound imports, cancellation, changed-policy refusal, exact registry persistence and clean lifecycle. Tool Profiles management UI and Persona auto-creation remain separate gates; project-context interviewing is qualified below within its stated limits. |
| Settings: Project-context interview | TASK-32775: readable question/answer and focused review fields, retained failed answers, duplicate-submit guard and edit-preserving Close/reopen | 140 distinct targeted cases and four final native cells with 20 inspected captures; [QA receipt](../qa/2026-09-18-settings-project-interview/README.md) records real workspace cancellation, answer retry, edited selected-only encrypted saves and clean lifecycle. Native interview setup initializes Tool Profiles first; TASK-32777 independently qualifies cold provisioning below. Durable draft resume, adaptive/provider interviewing and broader Personal Context workflows remain open. |
| Settings/Console: Workspace Persona selectors | TASK-32776: distinct saved IDs and control choices, complete catalogs beyond 100 records and accurate independently readable selected labels after list failure | 81 distinct targeted cases and four final native cells with 20 inspected captures; [QA receipt](../qa/2026-09-18-workspace-persona-identity/README.md) records exact saved identities, confirmation, cancellation, reopened registry checks and clean private lifecycle. Cold automatic provisioning is qualified below; broader Persona management remains a separate gate. |
| Settings/Console: Cold workspace provisioning | TASK-32777: automatic Persona provisioning awaits the existing app-owned Tool Profile initialization while preserving cancellation and lazy startup | 115 targeted cases and four native cells; [QA receipt](../qa/2026-09-18-workspace-cold-provisioning/README.md). Historical completed backfill rows are not silently reprovisioned; broader Persona management remains separate. |
| Settings: Tool Profiles management | TASK-32778–32788: originating action authority, shared initialization lifetime, export recovery, review intent, retained focus, compact controls and truthful removal outcomes | Bounded targeted and native evidence is indexed in the [Tool Profiles review ledger](2026-09-18-tool-profiles-review.md). Bind is qualified with four final native cells and sixteen inspected captures; MCP Edit now has 129 targeted cases and four final native cells with 20 inspected captures. Admitted-write overlap now has 173 targeted cases and four final native cells with twelve inspected captures. Screen-destruction/shutdown ownership now has 229 distinct targeted cases, four native recreation cells and a fifth pending-write shutdown journey with seventeen inspected captures (ADR-167). Compact MCP introduction, Source labels and simultaneous Tool/State readability now have 134 distinct targeted cases and four native cells with sixteen inspected captures; broader server/tool/audit/runtime review remains open. TASK-32784 restores the structural CSS ratchet at identical specificity and four-cell mounted paint; its remote guard passed on 1b16329e03. |
| Remaining Settings categories and modal patterns | Prior compact Providers/Network geometry is qualified, not all category behavior | Review remaining Speech/TTS flows, Workspaces, Tool Profiles, Privacy, Network, Personal Context, Console Behavior, Library/RAG, Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, MCP/ACP defaults, Diagnostics/About, Advanced Config, Internal Prompts, Image/Video Generation and Agents; modal workflows including Backup & Restore, source switching and manual sync remain separate gates |
| MCP local-tools master saves | TASK-32793: shared FIFO, activation/config identity, scoped partial receipts and master-only projection | 177 distinct targeted cases and sixteen inspected native captures; [QA receipt](../qa/2026-09-18-mcp-master-settings/README.md). Both entry points, pending refresh/recreation and shutdown are qualified; tool execution and broader Servers/Audit review remain open. |
| MCP Tools controls and catalog access | TASK-32789: full toggle labels, compact stacked filters, scrolling controls and selected-row visibility through resize | 88 distinct targeted cases; four final native theme/size cells and sixteen inspected captures qualify real private toggle persistence, filters and exact row inspection. [MCP review ledger](2026-09-18-mcp-review.md) retains workspace-root guidance/save behavior, refresh/drafts, execution and other modes; Tool/State readability is qualified below. |
| MCP Tools name/state readability | TASK-32790: long names wrap beside complete State; selected tool identity and explicit Enter survive reflow/filter/refresh | 88 final targeted cases, independent review and four native theme/size cells with twelve inspected captures. [QA receipt](../qa/2026-09-18-mcp-tools-readability/README.md) records final source and private shutdown; broader root-save, refresh/focus and execution behavior remains open. |
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

The [conflict review](2026-09-17-pr-2704-conflict-review.md) reconstructs all four dev integration merges through `96d4ca2b96`: 20 conflicted file entries / 30 blocks, exact parent references, selected outcomes and recorded native captures. The user gave the requested green light after the refreshed Notes/Console gallery at `156c06f80b`.

Pre-merge CI then exposed a gallery startup import, four excess broad CSS
rules, and a one-row GGUF layout regression. TASK-32591 was reopened to repair
those failures without raising ratchets or changing the approved conflict
choices. [Repair and native evidence](../qa/2026-09-17-pr-2704-ci-repair/README.md)
records the exact scope and verification; the broader review remains open.

The [fresh PR visual gallery](../qa/2026-09-17-pr-2704-visual-review/README.md)
records eight Console/Notes captures from source commit `7c28855825`, inspected
in dark/light at 80×24 and 170×48 with clean native shutdown. Wide Notes shows
the complete `Remove placement` label. That historical gallery truncates the compact Notes
introductory status; TASK-32765 fixed this residual in the refreshed gallery
linked above. This historical gallery does not
close any whole-destination or provider-execution gate.


TASK-32791 qualifies MCP root draft/save lifetime and corrects the root's actual
local MCP/Hub scope. [Targeted and native evidence](../qa/2026-09-18-mcp-root-settings/README.md)
records 26 root cases, adjacent checks, ten reviewed captures, independent review
and ADR-168. Its related rerun reproduced a preexisting Permissions table
resize clip on both current and saved baseline code. TASK-32792 now qualifies
that repair, plus native selected-row and fresh-Enter continuity, with
[final reflow checks and twelve inspected captures](../qa/2026-09-18-mcp-permission-reflow/README.md).
Master-toggle ordering and remaining MCP workflows stay open in the MCP ledger.
PR2707 remains draft/unmerged. All named applicable checks passed on saved head
fb16c9087c, including Windows GGUF; the earlier 6c0e317ab7 missing SelectOverlay
failure remains historical evidence requiring investigation, rather than a
currently failing check. Neither component review nor integration is complete.
