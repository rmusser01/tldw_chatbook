# Console parallel-agents fleet — expert UX/HCI review (2026-07-28)

Live UAT walkthrough of the complete parallel-agents journey at dev tip
`c22f2a636`, reviewed through a senior UX/HCI lens (NN/g heuristics + TUI
specifics: keyboard reachability, discoverability without hover, copy
accuracy, rendered-coordinates honesty). Two acts on the real TUI (tmux
235×52, SGR clicks/hover/wheel): Act 1 cold first-run (provider configured,
nothing else), Act 2 seeded two-workspace fleet (Red/Blue, read-only bound
folders). Captures under `captures/` (a1-*/a2-*). Severity: Critical /
Major / Minor / Polish, plus Upgrade proposals.

## What works well (credit where due)

- The **park → badge → visit → decide** loop is mechanically excellent: the
  toast copy names session and workspace, ◆ appears on tab and sidebar (and
  aggregates onto collapsed groups), the parked card mounts on visit, and
  the badge honestly persists until the decision. (a2-s7-park)
- The **pinned fleet line** (task-1140) earns its keep: "0 other agents
  running, 1 waiting for approval." sits at the very top of the rail,
  visible regardless of scroll/section state. (a2-s7-park:10)
- **Risk-tiered approvals** feel right at the safe end: `get_current_datetime`
  executed with no card; the first card a user ever sees is for a genuinely
  gateable operation. (a1-s2)
- The **navigation-guard dialog copy** is clear and correctly counts:
  "1 agent run will be cancelled if you leave Console. Leave anyway?"
  (a2-s9-navguard) — when it works (see F1).
- The **confinement denial** is airtight security-wise; every cross-root
  read refused. (a1-s3b)

## Findings

### F1 — Critical: the navigation guard can wedge into a zombie modal that soft-locks the app
Repro (scripted, a2-s9*): busy fleet (1 parked round) → navigate to
Settings → guard dialog → click **Stay** (works, dialog closes) → navigate
again → second guard dialog → click **Leave** at its rendered coordinates →
no effect — and from that point the dialog answers to NOTHING: Leave and
Stay clicks (12-point sweep across the rendered buttons), Escape, Tab,
Enter, and nav-bar clicks are all inert (a2-s9e-deadlock). The user's only
exit is quitting the app (Ctrl+Q). The app log is empty (0 bytes), so the
mechanism is undetermined; hypotheses for the fixer: the Leave click DID
resolve the awaited dialog but navigation failed after confirm, leaving a
painted-but-dead overlay; or a second `push_screen_wait` interleaving that
the existing race test (which queues a second NavigateToScreen *message*,
not a Stay-then-renavigate-then-Leave sequence) does not cover. Note the
test-blindspot rhyme with task-1142: the guard's tests click the button
WIDGET; nothing ever clicked the rendered coordinates.

### F2 — Major: the flagship capability is undiscoverable
At rest, nothing anywhere says agents run per tab, in parallel, under a
cap: not the Console surface (a1-s1), not F1 Help (panes/transcript/composer
shortcuts only — zero occurrences of "agent", "parallel", "approval",
"workspace"; a1-s4-help), not the footer (Alt+W and Alt+1..9 unlisted). The
fleet UX only teaches itself after the user has already, accidentally, run
two agents at once. The Settings guidance and the user-guide section
(task-1143) exist but are pull-only.

### F3 — Major: the first file-tool denial is a dead end that then nags
On an unbound workspace (every fresh install), an approved `read_file`
fails with "Path '…' is outside every allowed root… (+64 chars)" — no
mention of workspaces, folder binding, or Settings ▸ Workspaces, and the
truncation hides whatever the +64 chars say (a1-s3b). A first-run user
literally cannot grant file access from where they stand (Default can't
hold bindings; they must create a workspace, bind a folder in Settings, and
work in a session of that workspace) and nothing on the failure path says
so. The model then retries the identical path and the user is asked to
approve the same doomed request again (a1-s3c) — approval fatigue with
zero learning; the loop guard eventually kills the run with jargon
("loop detected: read_file repeated in a 1-cycle (3x)" — prior UAT).

### F4 — Major: the status-glyph language has no legend
● ◆ ✓ ✗ carry the entire fleet-status story, yet no legend exists anywhere:
not in Help, not in tooltips — the tab tooltip says only "Switch to Console
tab: Blue Chat." even when the tab shows ◆ (a2-s7b-hover). Recognition
over recall fails for the system's core status vocabulary; ✗/✓ color pairs
also carry meaning that colorblind users must infer from shape alone
(fortunately the glyphs differ in shape — keep that).

### F5 — Minor: approval-card ergonomics and copy
For a single decision the card presents a Select (Approve once / Approve
for session / Deny) plus three buttons (Approve all / Submit / Deny all) —
a two-step commit for a one-question dialog (a1-s3-approval). "(high
risk)" on a plain text-file read is technically defensible
(read-exfiltration floor) but unexplained and reads as alarmist; there is
no "why is this high risk?" affordance. Prior agent-driving evidence shows
even automation mis-sequences Approve-all-then-Submit.

### F6 — Minor: below-the-fold guidance in the Settings inspector
Focusing "Max parallel" shows only "Purpose:" in the Scope Inspector; the
Consequences/saved-as/applies-to rows exist but sit below the inspector's
scroll fold with only a thin scrollbar sliver hinting at them
(a2-s10-guidance). Focus does not auto-scroll the inspector to the
Focused-field guide. Same disease task-1140 fixed for the fleet line, in a
second location.

### F7 — Minor: copy nits
- Cap refusal: "1 agents already running (Red Chat)." — number agreement
  ("1 agent already running"); the refusal names busy sessions but offers
  no jump-to-tab affordance (prior UAT capture).
- Tab auto-titles ellipsize mid-string: "What is t…ate an." (a1-s3) —
  garbled; prefer end-truncation ("What is the curre…").
- Stop button says only "Stop" — with parallel runs, nothing communicates
  it stops the viewed tab only (behavior is correct; the label predates the
  fleet).
- "Tools: 0 ready" in the status bar until the first run despite tools
  being enabled (lazy catalog); reads as "no tools available" (a1-s1 vs
  a1-s2).

## Heuristic scorecard (journey-level)

| Heuristic | Verdict |
|---|---|
| Visibility of system status | Strong while busy (markers, fleet line, toasts); weak at rest (Tools: 0, no capability surface) |
| Match with real world | Good copy overall; "(high risk)" and loop-guard message are system-speak |
| User control & freedom | F1 is a hard violation (trapped modal); Stop-scoping uncommunicated |
| Consistency | Glyphs consistent across tab/sidebar/groups; approval flow consistent across bridges |
| Error prevention | Guard dialog (when it works), never-auto-approve, confinement — strong |
| Recognition over recall | F4 (glyph legend), F2 (hidden hotkeys) — weak |
| Flexibility & efficiency | Alt+1..9/Alt+W exist but are undiscoverable; no keyboard path evidenced on cards/dialogs |
| Help & documentation | User guide exists; in-app Help ignores the entire agent feature set |

## Upgrade proposals (beyond fixes)

1. **A one-time "fleet" coach-mark**: on the first occasion a second tab is
   created, a single dismissible line under the tab strip — "Each tab runs
   its own agent; up to 3 in parallel (change in Settings)." Kills F2 for
   the cost of one Static.
2. **Actionable toasts via the fleet line**: the pinned fleet line could be
   a click-target cycling to the next session needing attention —
   "1 waiting for approval →" as a button, not a Static.
3. **Guided unbind recovery**: the outside-every-allowed-root error should
   append the recovery route: "Bind a folder to this session's workspace in
   Settings ▸ Workspaces to allow file access." — and the approval card for
   a path that policy will reject could warn *before* the user approves
   (pre-flight the roots check at card-build time).
4. **Glyph legend line** in F1 Help + marker-aware tab tooltips ("Blue Chat
   — waiting for approval").
5. **Single-round card fast path**: when a card has exactly one row, allow
   one-click "Approve once"/"Deny" without Submit.

## Filed tasks

task-1230 (F1 zombie guard, Critical) · task-1231 (F3 denial teachability +
approval pre-flight) · task-1232 (F2 discoverability: Help/coach-mark/footer)
· task-1233 (F4 glyph legend + marker tooltips) · task-1234 (F5/F6/F7 copy
and ergonomics batch: number agreement, tab ellipsis, Stop tooltip,
inspector auto-scroll, Tools-ready chip, single-row card fast path).

## Verdict

The engineering underneath is genuinely solid — every safety property held
again under adversarial driving, and the previous waves' fixes (pinned
fleet line, coordinate-honest header toggles) visibly improved the surface.
The gap is now experiential: the feature doesn't introduce itself (F2),
doesn't teach recovery at its most common first failure (F3), speaks in
unexplained glyphs (F4) — and has one genuine trap (F1) that must be fixed
before this UX can be called done.
