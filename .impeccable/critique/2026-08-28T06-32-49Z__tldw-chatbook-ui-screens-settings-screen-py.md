---
target: Settings screen (Settings + Scheduled Tasks combined run)
total_score: 29
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 1
timestamp: 2026-08-28T06-32-49Z
slug: tldw-chatbook-ui-screens-settings-screen-py
---
# Design Critique — Settings & Scheduled Tasks (combined run, 2026-08-27)

Method: dual-agent (A: live-TUI design review, isolated profile, 235×52 + 235×40 · B: detector + static evidence)
Brief: first-time and power-user workflows — create a repeating scheduled task; change settings. Mode: Operate.

## Design Health Score

### Scheduled Tasks (SchedulesWorkbench) — 24/40 (Acceptable)

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | Disabled task still shows Status "Waiting" + concrete Next Run (live-verified); "Sync completed." toast while bar reads "Last pull: — Last push: —" |
| 2 | Match System / Real World | 2 | Detail humanizes ("Daily at 09:00 UTC") but form demands raw ISO-8601, 5-field cron, IANA tz; reminder/task/schedule noun drift |
| 3 | User Control and Freedom | 3 | Dirty-form discard guard, delete confirm, Esc clears marks; no undo after delete |
| 4 | Consistency and Standards | 3 | Footer 1:1 with bindings (ADR-031); but queue shortcuts advertised on Conflicts tab; create modal breaks workbench idiom |
| 5 | Error Prevention | 2 | Past-time guard, cron/tz validation — undermined by focusable invisible cron field that accepts typing (P0) |
| 6 | Recognition Rather Than Recall | 2 | Empty state teaches `c`; 3 presets; custom recurrence = cron recall; ●/◇ glyphs have no legend |
| 7 | Flexibility and Efficiency | 3 | Single-letter verbs, real bulk ops (x + space/d), run-now, debounced filter; no sort, no palette "create task" |
| 8 | Aesthetic and Minimalist Design | 3 | Clean three-pane layout; sync/owner bar permanently occupies top row even for purely local use |
| 9 | Error Recovery | 2 | Failure toasts name task + next step; "Run now (retry)"; but form errors cluster at bottom and can reference an invisible field |
| 10 | Help and Documentation | 2 | Tooltips with visible text mirrors (UX-073) good; no task-level help; cron helper clips; zero tooltips in workbench queue itself (B) |
| **Total** | | **24/40** | **Acceptable** |

### Settings (SettingsScreen) — 29/40 (Good)

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 4 | Banner + rail asterisk + inspector "Unsaved changes" + "Saved as: …" + live theme apply — genuinely excellent |
| 2 | Match System / Real World | 3 | Mostly plain; leaks: "agent_runs.db (SQLite) — immediate CRUD, no draft", "table-shaped TOML top-level value", roadmap copy in Schedules category |
| 3 | User Control and Freedom | 3 | Revert w/ confirm; drafts survive full screen navigation (live-verified); no undo after save |
| 4 | Consistency and Standards | 2 | Five+ save models coexist; the mitigating State banner renders doubled and self-colliding (P1) |
| 5 | Error Prevention | 3 | Validation-gated saves, "Needs correction | msg", invalid-field→selector mapping, revert confirm |
| 6 | Recognition Rather Than Recall | 3 | `/` search with live match count; grouped rail; search matches categories not fields; Appearance-vs-Theme ambiguity |
| 7 | Flexibility and Efficiency | 3 | s/r/t per-category verbs, contextual accelerators honestly advertised; no cross-category "jump to setting" |
| 8 | Aesthetic and Minimalist Design | 2 | State banner rendered twice; "Configuration: X; Status: X" duplication; collapsed Overview sections burn ~6 rows each |
| 9 | Error Recovery | 3 | Validation message leads banner; atomic save + .bak + "Load Backup"; but 3+ paths surface raw exception text (B) |
| 10 | Help and Documentation | 2 | F1 help for "Settings: Schedules" opened completely empty (live-verified); Appearance help = 3 shortcut keys |
| **Total** | | **29/40** | **Good** |

## Design Specificity Verdict

**LLM assessment (unanchored):** Settings is authored, distinctively so — the save-contract State banner, ownership honesty ("Writes allowed: No - change this in Schedules instead"), credential-source disclosure, and per-category test verbs are direct expressions of the product's local-first identity; almost nothing could be transplanted. Schedules is an authored shell around a generic core: the workbench (sync-ownership bar, Conflicts tab with count, Console handoff, honest late-dispatch copy) is product-native, but the creation flow — the most important interaction — is a category-interchangeable CRUD modal (Title/Body/Kind/cron/ISO-8601/IANA tz) any cron GUI from 2005 could ship.

**Deterministic scan:** unavailable for this stack — detect.mjs scans only web file types (.html/.css/.jsx/…; not .py/.tcss), so both runs exited 0 with zero scannable files. Not a clean bill of health; no findings to false-positive-check. Static fallback evidence instead:
- Theming discipline is excellent: 0 hardcoded colors in owned TCSS vs 1,198 theme-token uses (one stray hex belongs to a Console block). But `task_detail.py:78-91` hardcodes 11 Rich style strings ("bold white on grey50" etc.) and `settings_screen.py:16657` sets a literal "gray".
- **Scheduling has zero screen-specific `:focus` rules** (all of `_scheduling.tcss`, `ReminderForm.DEFAULT_CSS`, widget-default blocks) — corroborates the live P0 (typing into an invisible focused field).
- Tooltip coverage: scheduling detail widgets ~73%, but `schedules_workbench.py` itself 0; settings ~29% with `help_text`/hint counterweights.
- Raw exception text reaches users: `settings_screen.py:10871` ("Model discovery failed: {exc}"), `:11006`, `:14281` ("Backfill failed: {e}" toast).
- Scale: `settings_screen.py` is 23,350 lines, one class of ~21,000 lines, 811 methods, 26 categories.

**Overlays:** n/a — terminal UI, no DOM to inject into. Live evidence came from driving the real app over tmux instead.

**Where A and B disagree productively:** B's static color-only scan found text counterparts everywhere and concluded "no pure color-only signal confirmed"; A's live pass found the real color-only carriers are *dim-contrast disabled buttons* (Enable/Disable pair, sync-bar Clear) — a class static analysis can't see. Conversely, B caught the raw-exception strings and the hardcoded Rich styles that A missed.

## Workflow Walkthroughs (evidence summary)

**Jordan (first-timer), repeating task:** Two modal gates precede work (setup wizard — skippable, well-worded; then a "Check model lists online?" consent dialog that silently swallowed a nav click). Schedules is discoverable; the empty state is a peak: "No scheduled tasks yet. Press c to schedule your first task." Creation lands with toast + queue row + visible "Next Run: 2026-08-28 09:00 UTC". Valleys: the "Run At (ISO-8601):" label; choosing "Custom cron…" makes *nothing visibly appear* (clipped field); disabling a task leaves Status "Waiting" with a live Next Run.

**Alex (power user):** ~19 keystrokes + title for a daily-9:00 repeating task via palette → `c` → presets. Every management verb is one letter with bulk variants (x/space/d). Gaps: no queue sort, no palette command to create/run directly, edit round-trips through the modal, cron typed into an invisible field.

**Settings (both):** `/` filter with live match counts, s/r/t verbs, live theme apply, drafts survive navigating away and back (verified). Gaps: search matches categories not individual settings ("theme" → Theme file-editor vs Appearance coin-flip); F1 help empty for some categories; State banner doubled.

## Overall Impression

Settings is the stronger surface — a genuinely product-specific status/save-contract system undermined by its own presentation bug and by the sheer number of save models it must explain. Schedules nails the workbench loop (discover → create → confirm → manage) but hides its truth at the two moments that matter most: expressing a custom recurrence (clipped form) and knowing whether a task will actually fire (disabled ≠ visibly disabled). The single biggest opportunity: make the recurring-task form as honest and self-explaining as the rest of the product already is.

## What's Working

1. **The save-contract system (Settings):** pinned State banner naming the active save model per category, footer verbs that appear 1:1 with what works, per-category test actions, and text-carried dirty state in three places. Best-in-class TUI status choreography.
2. **Draft durability (Settings):** change a value, walk to another screen, return — category, filter, draft, and dirty state all intact (`settings_screen.py:2864-2933`). "Preserved user work" as a first-class behavior.
3. **Honesty under uncertainty (Schedules):** late-dispatch copy refuses to invent causes ("skipped, not replayed"), "Run now" tooltip states real semantics, disabled buttons carry visible text reasons (UX-073). The product's stated personality executed in copy.

## Priority Issues

1. **[P0] The recurring form's cron field, helper, and live preview clip invisible at common terminal heights — and still receive focus and keystrokes.** At 235×52, choosing "Recurring" shows Frequency, three blank rows, then Timezone; Tab lands on the invisible cron Input; typed garbage silently flips the preset to "Custom cron…". Cause: `ReminderForm` stacks fields in a plain `Vertical` inside a `max-height: 55` container with no scrolling (`forms/reminder_form.py:35-42, 116-186`) while the Body TextArea keeps ~7 rows; B confirms zero `:focus` styling anywhere in scheduling. **Why:** the personas converge here — Jordan concludes the feature is broken, Alex types cron blind, Sam types into a void. The form's best safety feature (live "Runs: …" preview) becomes dead code. **Fix:** make the field column a `VerticalScroll` with scroll-to-focus; cap Body at 3 rows; compress the two long helper Statics; verify at 24 rows. **Command:** /impeccable adapt (heights), then /impeccable harden.

2. **[P1] Disabled schedules are indistinguishable from armed ones.** Toast says "'Morning digest' disabled." but queue Status stays `Waiting` and Next Run still shows a concrete future time (persists across refresh). `_task_status()` returns `last_status`, which disabling never touches (`task_detail.py:186-190`; `schedules_workbench.py:697-700`); the only persistent carrier is the dim Enable/Disable pair — color-only. **Why:** a disabled job displaying a future run time is a false promise discovered only when the digest never arrives; violates Design Principle 8 and the color-never-sole-carrier commitment. **Fix:** derive display status — `enabled is False` → badge `Disabled` (enum + styles already exist, `task_detail.py:52/84`) and Next Run `— (disabled)`. **Command:** /impeccable harden.

3. **[P1] The State banner — sole carrier of the save contract — is duplicated and self-colliding.** Renders "State: Read-only here | State: Active | …" (composition at `settings_screen.py:6413` prepends `State:` to scope strings that embed their own at 6466/6479-6482), and the whole banner appears twice (pinned 16631 + in-card 12405/15116). **Why:** the banner exists because five save models coexist; a stuttering duplicated contract line teaches users to stop reading it — un-fixing the original problem. **Fix:** strip embedded `State:` prefixes from `_category_state_scope_text`; drop the in-card banner where the pinned one is present. **Command:** /impeccable clarify.

4. **[P1] The create form's input vocabulary is expert-only.** "Run At (ISO-8601):" requiring `2026-07-20T14:00:00+00:00`; timezone as free-text IANA name; recurrence beyond 3 presets (daily 9:00 / Monday 9:00 / hourly) requires raw cron — no "every weekday", no time-of-day control on presets. **Why:** Jordan can't express "weekdays at 8" without learning cron; even Alex must type a full RFC timestamp for a one-time run. Violates "plain language, forgiving input." **Fix:** accept forgiving datetimes ("2026-08-28 09:00", default tz = system) with the existing preview disambiguating; Timezone becomes a searchable Select defaulting to system; add "Every weekday at…" + editable preset times; keep cron behind "Custom cron…". **Command:** /impeccable shape (interaction), then /impeccable clarify (labels).

5. **[P2] The sync surface over-promises and mixes signals.** `s` toasts "Sync completed." while the bar reads "Last pull: — Last push: —"; the owner bar permanently shows `Server (http://127.0.0.1:8000)` + `Clear` even when the header chip says "Local schedules"; Clear's disabled state is color-only. **Why:** "completed" with no recorded pull/push is status-requiring-log-reading (an explicit anti-reference); server plumbing is visually first on a local-first screen. **Fix:** local owner → "Synced — nothing to pull or push" or real timestamps; collapse the owner bar to one line; hide Clear until an error exists. **Command:** /impeccable distill.

### Further issues (tracked, lower severity)

- **[P2] Terminology drift:** nav "Schedules" / form "New Scheduled Task" / toast "Reminder created." / guard "Only reminder tasks can be edited here." Standardize on "scheduled task"; where projections differ, say so ("Managed by Watchlists — edit it there").
- **[P2] Bulk-mark is invisible infrastructure:** `x` marks (●) and missed (◇) have no legend, no marked-count, no hint that space/d go bulk. Reuse `#scheduling-pane-notice`: "2 marked — space toggles all · d deletes all · esc clears".
- **[P2] Raw exception text in Settings user paths** (B): `settings_screen.py:10871, 11006, 14281` — wrap in plain-language summaries, keep detail behind Diagnostics.
- **[P2] Settings search matches categories, not settings** — "jump to setting" doesn't exist; "theme" yields a Theme-editor vs Appearance coin flip.
- **[P3] F1 help hollow/empty** for some categories (Schedules category: empty body, live-verified). Feed it the inspector's contract rows + state-scope copy.
- **[P3] Next Run is absolute UTC only** — no relative ("in 14h") or local-time rendering.
- **[P3] Hardcoded Rich status colors** (`task_detail.py:78-91`, fallback `:242`) and one inline `styles.color = "gray"` (`settings_screen.py:16657`) bypass theme tokens.
- **[P3] Overview duplication:** "Configuration: OpenAI / gpt-5.6-terra; Status: OpenAI / gpt-5.6-terra".

## Persona Red Flags

**Jordan (first-timer):** consent dialog silently ate a nav click (no shake/toast); "Run At (ISO-8601):" is the scariest string of the journey; "Custom cron…" appeared to do nothing (clipped); disabled task still reads Waiting; Settings "theme" search forces a coin flip. Saved by: the empty-state CTA, "That time is in the past — pick a future time.", the discard guard.

**Alex (power user):** fast path genuinely fast (~19 keystrokes + title); red flags: cron typed into an invisible field with the validation preview also invisible; no queue sort; no palette command for create/run; edit round-trips through a modal instead of inline pane editing. (Ctrl+digit hotkeys untestable through tmux — environment limitation, not judged.)

**Sam (keyboard-only):** strong baseline — text-carried toggle states, visible-text disabled reasons, words in status badges, Cancel-first delete dialog, focus-help line mirroring tooltips. Breakers: Tab lands on the clipped cron input with no visible focus indicator anywhere; Enable-vs-Disable applicability is dim-contrast only; ● mark announced nowhere but the glyph; Clear's disabled state color-only.

## Minor Observations

- Queue footer keys stay advertised on the Conflicts tab where they act on an invisible selection.
- Empty queue renders full table header + filter before the CTA draws the eye.
- "Agents" sits under Troubleshooting in the Settings rail; "(view)" suffixes are a nice honesty touch.
- Schedules settings category exposes roadmap copy ("Planned: add schedule defaults after…").
- Delete-recovery contracts differ across the app (Schedules: permanent; Notes: trash+recover).
- Width-responsiveness is thoughtful (hidden panes announce themselves, `schedules_workbench.py:790-800`) — better than the form's height handling.
- Setup wizard copy "everything here can be changed later in Settings" is exactly right.

## Questions to Consider

1. If the State banner needs eight badge variants to explain the save models, is the badge the fix — or is the number of save models the bug? What would collapsing Splash/Workspaces/Agents into the draft-save-with-`s` contract cost?
2. Why is creating a schedule a modal at all? Every peer surface edits in the detail pane of a persistent workbench; a pane-based editor gets scrolling, the state banner, and footer-key consistency for free — the P0 disappears structurally.
3. What is the queue *for* — inventory or forecast? A surface promising "when jobs run" arguably leads with a next-24-hours timeline (sorted by next fire, disabled rows visibly parked), not an unsorted CRUD table.
