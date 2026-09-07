---
target: Console Ctrl+K operational session switchboard
total_score: 25
max_score: 40
na_heuristics:
p0_count: 0
p1_count: 3
timestamp: 2026-09-02T13-56-03Z
slug: widgets-console-console-session-switcher-modal-py
---
Method: dual-agent (A: impeccable_design_assessment · B: impeccable_detector_assessment)

## Overall assessment

**25/40 — acceptable, with a strong behavioral foundation and significant presentation/usability work remaining.**

The redesign has made the session switcher substantially more trustworthy and much more product-specific. Its consequence-first model — **Waiting for you → Working → New results → Current → Other open** — matches how an operator actually allocates attention across multiple agents. Exact-query commit, stable selection through live reordering, bounded History loading, degraded-state copy, and exact post-switch receipts are unusually thoughtful.

The primary problem is now representational rather than architectural: the interface does not make its excellent internal model visually obvious enough. Three overlapping concepts — scope, current tab, and candidate destination — compete for the word or treatment “selected.” Rows are center-aligned and metadata-heavy, the modal stays at its maximum height even when mostly empty, and hardcoded colors flatten semantically different states. The result behaves like an operational switchboard but still looks like a generic internal terminal picker.

## Inspection basis

This critique examined the committed Textual implementation, production compositor captures, state/search logic, focused tests, and the QA evidence bundle. The target is a native terminal `ModalScreen`, not an HTML/DOM route, so browser overlay inspection would have produced false confidence. The visual fallback was the production CSS cascade rendered through Textual’s compositor, including:

- [Active switchboard at 120×35](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-session-switcher-trust/Docs/superpowers/qa/task-21351-console-switcher-activity/captures/active-switchboard-120x35.svg)
- [Active switchboard at 72×35](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-session-switcher-trust/Docs/superpowers/qa/task-21351-console-switcher-activity/captures/active-switchboard-72x35.svg)
- [History switchboard at 120×35](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-session-switcher-trust/Docs/superpowers/qa/task-21351-console-switcher-activity/captures/history-switchboard-120x35.svg)
- [Real Ctrl+K candidate selection](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-session-switcher-trust/Docs/superpowers/qa/task-21351-console-switcher-activity/captures/real-ctrl-k-success-selection-160x45.svg)
- [Exact successful landing receipt](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-session-switcher-trust/Docs/superpowers/qa/task-21351-console-switcher-activity/captures/real-success-outcome-notice-160x45.svg)
- [Failure and Mark seen consequence](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-session-switcher-trust/Docs/superpowers/qa/task-21351-console-switcher-activity/captures/real-failure-mark-seen-160x45.svg)

## Workflow walkthrough

### First-time user

1. **Entry is fast and calm.** Ctrl+K opens immediately with the search field focused and useful operational groups already populated; History does not block the modal.
2. **The mental model is not yet self-evident.** “Active — selected,” `CURRENT`, and `Selection: n of n` look like three descriptions of the same thing but refer to scope, current location, and Enter destination. Product vocabulary such as `CONSOLE TAB`, `UNSEEN`, `workspace:<name>`, `is:waiting`, and `+1` is compact but unexplained.
3. **Search teaching is syntax-first.** The placeholder attempts to teach the query grammar, truncates in real captures, and disappears as soon as typing begins. The user must recall syntax before learning why it helps.
4. **Confirmation is behaviorally safe but visually ambiguous.** With blank search focused, Enter targets the most recently used other tab. That fulfills the specified accelerator, but the candidate relationship should be stated as a consequence: `Enter switches to: <title>`.
5. **The landing is the strongest moment.** Exact destination receipts, preserved typing, and explicit failure/Mark seen consequences close the loop with uncommon precision.

**Emotional arc:** reassuring entry → uncertainty at target confirmation → strong, trustworthy landing. The peak-end experience is good; the decision point immediately before Enter needs to catch up.

### Experienced power user managing multiple agents and conversations

1. **Attention triage is excellent.** Waiting, working, and new-result groups let the user allocate attention by consequence rather than remember tab order.
2. **The accelerator path is genuinely fast.** Immediate Active data, live filtering, MRU blank Enter, stable identity through reorder, F2/F3, paging, and exact-query commit support sub-second operation without sacrificing data integrity.
3. **Scanning throughput is the bottleneck.** Centered two-line rows create a moving scan axis. A row can contain state, title, source, workspace, lifecycle, recency, and `+1`; History can mount 50 candidates, but row comparison remains visually serial.
4. **Scope can contradict content.** When an Active query has no match, History results can appear while Active still reads as selected. The source assigns `History — selected`, but the compositor capture clips it to `History`, so the visible UI does not carry the intended state.
5. **Terminal density is inefficient.** The 35-row maximum becomes the default canvas even for a small result set, leaving large unused regions while compact, high-value metadata is crowded into centered subtitles.
6. **The operational landing is precise.** Frozen receipt evidence and exact post-paint acknowledgment preserve trust when a tab owns its own draft, queue, approval, and workspace context.

**Emotional arc:** immediate control → slower-than-necessary visual parsing → confidence at exact activation and recovery. The switchboard’s behavior supports expert use; its composition makes experts work harder than necessary.

## Design specificity

**Authored in information architecture; commodity in composition.**

The switchboard’s state model is distinctly Chatbook and aligned with the product’s “Precision Workbench” intent. The visual treatment is not: hardcoded black/gray/yellow, a generic tall border, centered buttons, and minimally differentiated operational states could belong to almost any terminal picker. The next pass should make the state/authority model visually legible through stable alignment, semantic tokens, adaptive density, and restrained state accents — not through decorative control-room styling.

## Cognitive-load assessment

**Four of eight checks fail, so the current multi-agent case carries high cognitive load for a task that should feel instantaneous.**

| Check | Result | Evidence |
|---|---|---|
| Single focus | Pass | The modal isolates switching and triage cleanly. |
| Chunking | Fail | Active subtitles can carry six or seven centered fragments. |
| Grouping | Pass | Consequence-based operational sections are excellent. |
| Visual hierarchy | Fail | Current, candidate, focus, and selected scope use neighboring treatments. |
| One thing at a time | Pass | Activation is direct; unavailable-session acknowledgment is deliberately staged. |
| Minimal choices | Fail | Six Active destinations or eight visible History rows compete with five shortcut families. |
| Working memory | Fail | Scope, current location, candidate, and filter grammar must be mentally reconciled. |
| Progressive disclosure | Pass | History is lazy, paging is conditional, and degraded states appear contextually. |

## Nielsen heuristic review

| # | Heuristic | Score | Assessment |
|---|---|---:|---|
| 1 | Visibility of system status | 3/4 | Strong grouping, selection status, degraded/error copy, and exact landing receipts; weakened when History content appears under an Active-selected state. |
| 2 | Match with the real world | 2/4 | Human attention groups are strong; query syntax, `CONSOLE TAB`, `UNSEEN`, and `+1` require translation. |
| 3 | User control and freedom | 3/4 | Esc, Cancel, F3, paging, and stale-selection recovery are solid; blank Enter’s MRU consequence is not explicit enough. |
| 4 | Consistency and standards | 2/4 | Shared focus/header patterns exist, but “selected” is overloaded, the History indicator clips, and hardcoded styling bypasses the design system. |
| 5 | Error prevention | 3/4 | Exact-query commit, authority fences, safe stale handling, and two-Enter acknowledgement are rigorous; mouse and keyboard acknowledgement paths need parity. |
| 6 | Recognition rather than recall | 2/4 | Literal state labels help, but syntax teaching truncates and the three selection concepts require interpretation. |
| 7 | Flexibility and efficiency | 3/4 | Ctrl+K, MRU Enter, live filtering, arrows, F2/F3, pointer support, and bounded paging are strong; large pages lack exposed Home/End/Page accelerators. |
| 8 | Aesthetic and minimalist design | 2/4 | Ornament is restrained, but repetitive centered metadata, five-command hints, and unused canvas weaken hierarchy. |
| 9 | Error recognition and recovery | 3/4 | Degraded, stale, unavailable, and selection-moved states are specific and actionable. |
| 10 | Help and documentation | 2/4 | Empty-state teaching and hints help; query help is truncated, syntax-first, and nonpersistent. |
| **Total** |  | **25/40** | **Strong trust mechanics; the presentation layer obscures them.** |

## What is working especially well

1. **Consequence-first grouping is the right product model.** It turns multiple agents from a tab-navigation problem into an attention-allocation problem.
2. **The implementation respects power-user speed and state integrity together.** Stable selection during live reorder, exact-query commit, fast Active data, and bounded History are strong HCI engineering choices.
3. **Recovery language is precise and useful.** “Active agents are still usable,” selection-moved warnings, stale-target recovery, and local-receipt degradation all name both the problem and what remains safe.
4. **Post-switch receipts complete the trust loop.** Exact destination, preserved user input, post-paint success, and explicit failed-result acknowledgment make the end state unusually dependable.
5. **Keyboard and non-color semantics form a solid accessibility base.** State text, candidate markers, visible focus, Escape/Cancel, and shortcut affordances do not rely on color alone.

## Priority issues and solutions

### [P1] Scope, current session, and candidate destination conflict

**Evidence:** `Active — selected` describes the data scope; `Selection: 6 of 6` describes the Enter target; `CURRENT` describes the viewed tab. Active searches can auto-widen into History without visibly changing scope. The code assigns `History — selected`, but the rendered History capture clips it to `History` ([modal source](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-session-switcher-trust/tldw_chatbook/Widgets/Console/console_session_switcher_modal.py:894)).

**Why it matters:** The highest-stakes question in a switcher is “Where will Enter take me?” The UI makes users reconcile three state grammars at precisely that moment.

**Solution:**

- Use conventional Active/History tabs without the word “selected.”
- Reserve `Current tab` for present location and `Switch to` for the candidate.
- Make the status line consequence-first: `Enter switches to: <title>` or `Enter marks seen: <title>`.
- If Active search widens, visibly change the scope to History/All or display an explicit combined-results state.
- Add a compositor assertion for the rendered scope label width, not only its text property.

Suggested follow-up: `$impeccable shape`

### [P1] Fixed-height, centered rows undermine terminal-native scanning

**Evidence:** the modal uses `height: 100%; max-height: 35`, producing a 35-row surface even with only a few results ([modal layout](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-session-switcher-trust/tldw_chatbook/Widgets/Console/console_session_switcher_modal.py:85)). Production captures show substantial empty black space, while row metadata remains centered and dense.

**Why it matters:** Operators compare multiple destinations by state, title, workspace, and time. A shifting centered axis slows comparison and wastes scarce terminal space.

**Solution:**

- Size to content until the 35-row cap, then scroll.
- Left-align a stable row grid: candidate gutter → state → title → workspace/source → recency.
- Keep operational state and title on the primary line; demote lifecycle and source details.
- Add Home/End and Page Up/Page Down when History presents large mounted pages.
- Make the shortcut strip contextual, emphasizing only the action that will occur next.

Suggested follow-up: `$impeccable layout`

### [P1] Hardcoded colors flatten state and bypass user theme authority

**Evidence:** the modal hardcodes black, gray, and yellow in embedded Textual CSS ([modal CSS](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-session-switcher-trust/tldw_chatbook/Widgets/Console/console_session_switcher_modal.py:83)). Ordinary selection status, warnings, and operational conditions consequently occupy nearby visual treatments. The automated detector returned clean because it does not parse this embedded Textual CSS pattern.

**Why it matters:** This weakens high-contrast/theme compatibility and visually undercommunicates the difference between approval, working, unseen completion, failure, and ordinary navigation.

**Solution:**

- Move surface, border, primary/muted text, focus, warning, error, and success styling onto the existing `$ds-*` semantic tokens.
- Use a restrained one-cell marker or state-label accent for waiting, working, new result, and failure.
- Preserve literal labels so color reinforces rather than carries meaning.
- Add a detector rule or focused static test for hardcoded colors inside Textual `DEFAULT_CSS`.

Suggested follow-up: `$impeccable colorize`

### [P2] Search help requires syntax recall and truncates in the real layout

**Evidence:** the placeholder begins with exact `workspace:<name>` and `is:*` grammar but truncates in both narrow and production captures, then disappears after the first typed character ([search field](/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-session-switcher-trust/tldw_chatbook/Widgets/Console/console_session_switcher_modal.py:202)).

**Why it matters:** First-time users encounter implementation grammar before they understand the available search concepts; experienced users cannot recover the complete grammar in place.

**Solution:** Use a plain-language placeholder such as `Search sessions, workspaces, waiting, running, or finished…`. Put exact query syntax behind `?` or F1 contextual help. Keep the main footer focused on the immediate Enter consequence.

Suggested follow-up: `$impeccable clarify`

### [P2] Metadata and unavailable-session actions are overpacked

**Evidence:** a row can read `FINISHED · UNSEEN · CONSOLE TAB · Chats · open session · now · +1`; `+1` does not reveal whether it aggregates another run, receipt, or controller signal. Unavailable results repeat “Mark seen” in title/action treatment and subtitle.

**Why it matters:** The state users most need to triage becomes the hardest row to parse. Duplicated action wording consumes space without adding certainty.

**Solution:** Enforce state → title → workspace/source → time. Expand `+1` to a meaningful `2 updates` on focus or in the row. Give unavailable results one dedicated `Mark seen` action label, with the failure category as metadata. Ensure pointer activation follows the same confirmation contract as keyboard activation.

Suggested follow-up: `$impeccable distill`

## Persona review

### Alex — impatient power user

- Gains genuine speed from Ctrl+K, immediate filter focus, MRU Enter, live selection retention, and F2/F3.
- Loses scanning speed to centered two-line rows and a permanently visible five-command hint strip.
- Needs Home/End/Page navigation for 50-row History pages.
- Needs blank Enter’s exact MRU destination stated without moving focus.

### Jordan — first-time user

- Cannot readily distinguish Active scope, current tab, and candidate selection.
- Encounters unexplained `CONSOLE TAB`, `UNSEEN`, `is:waiting`, and `+1` vocabulary.
- May see History content while Active still appears selected, learning the wrong scope model.
- Benefits from unusually good empty-state and recovery copy once an exceptional state occurs.

### Sam — keyboard/accessibility-dependent user

- Benefits from complete keyboard operation, literal state labels, a non-color candidate marker, focus treatment, and visible Escape/Cancel.
- Faces unnecessary cognitive effort when current, candidate, focus, and scope selection coexist.
- Does not receive proven assistive-announcement behavior when live reconciliation unmounts and remounts result widgets.
- Loses theme/high-contrast authority to embedded hardcoded colors.

### Morgan — solo builder operating several local agents

- Gets an excellent consequence-first attention model.
- Cannot decode `+1` at the moment aggregation matters most.
- Sees workspace and live/saved source but not the smallest safe profile/local authority label carried internally.
- Needs approval/failure category and next action context without exposing transcript content.

## Minor observations

- The automated detector returned exit code 0 and `[]`; it produced no findings or false positives, but missed embedded Textual hardcoded colors.
- Ordinary `Selection: n of n` uses the same hardcoded yellow as warning conditions, diluting warning semantics.
- `#console-switcher-feedback` is always hidden while duplicating status content and appears to be presentation/test residue.
- The empty-state copy is unusually strong: it teaches Ctrl+T, agent tabs, History, and search recovery.
- Existing compositor QA asserts source-label properties but does not assert that `History — selected` is visibly rendered.
- Geometry/focus tests cover 52×20, 60×18, 72×35, and 120×50, including focused-row visibility.
- Native iTerm2 equal-cell parity remains unverified because macOS Accessibility/TCC blocked automation; Windows Terminal parity remains pending behind TASK-20937.6.

## Provocative questions

- Is Ctrl+K primarily a tab switcher, an agent-attention triage board, or global conversation search? Which mental model should visually dominate?
- If Active search silently includes History, should Active/History remain modes, or should the interface use explicit `Open · History · All` scopes?
- Should blank Enter remain a hidden MRU accelerator, or become an explicit consequence line naming the exact destination?
- What is the smallest safe authority label that distinguishes local profile/workspace/live/saved before switching?
- Could the exact landing receipt become Chatbook’s visual signature for trust: precise destination, preserved state, and explicit consequence?

## Run notes

- Target slug resolved successfully as `widgets-console-console-session-switcher-modal-py`.
- No `.impeccable/critique/ignore.md` file was present.
- Assessment A and Assessment B were run by isolated agents. Assessment B was held until Assessment A completed; neither assessment read the other before finishing.
- Detector: exit 0, JSON `[]`, zero rules/locations, zero false positives. Coverage gap: embedded Textual CSS color literals were not detected.
- Browser overlay: not applicable. The target is a native Textual modal with no DOM route; the exact fallback was production SVG compositor output plus Textual source and geometry/focus tests.
- No live server or browser tab was started, so no server/browser cleanup was needed.
- Assessment B removed and verified its temporary directory; Assessment A removed its temporary PNG previews.
