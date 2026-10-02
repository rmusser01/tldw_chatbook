# Console Close follow-up — native verification, 2026-10-01

TASK-33621.16 passed four native cells in the actual `TldwCli` app: pending
approval and question closes at **235×52** and **80×24**, using `textual-dark`.
[Result](native/result.json) records loaded module origins, source hashes, TTY
streams, the acquired private instance lock, and the `LinuxDriver`.
[Terminal process](terminal-process.json) confirms exit 0; the app returned 0
without an exception. No network connection was attempted.

The tab title `Pending [notes]` appeared literally in each close dialog. The
pending decision consequence and nonzero message/run counts were painted;
zero draft, attachment, delegated-agent and queue categories were absent.
Stay had default focus. SGR mouse input sent through tmux clicked the actual
background tab’s ✕ and then Close. The real approval worker returned deny; the
real question worker returned cancelled. The deterministic registered owning
task was cancelled, the target round and payload disappeared, and the viewed
tab’s question stayed pending until explicit fixture cleanup.

| Native cell | Painted confirmation | Closed background tab |
| --- | --- | --- |
| 235×52 approval | [terminal](native/dark-235x52-approval-confirm.txt), [SVG](native/dark-235x52-approval-confirm.svg) | [terminal](native/dark-235x52-approval-closed.txt) |
| 235×52 question | [terminal](native/dark-235x52-question-confirm.txt), [SVG](native/dark-235x52-question-confirm.svg) | [terminal](native/dark-235x52-question-closed.txt) |
| 80×24 approval | [terminal](native/dark-80x24-approval-confirm.txt), [SVG](native/dark-80x24-approval-confirm.svg) | [terminal](native/dark-80x24-approval-closed.txt) |
| 80×24 question | [terminal](native/dark-80x24-question-confirm.txt), [SVG](native/dark-80x24-question-confirm.svg) | [terminal](native/dark-80x24-question-closed.txt) |

All four terminal captures and rasterized confirmation previews were inspected.
At 80×24 the full title, consequences and both action buttons fit. Every pending
consequence fits on one line in the final copy. PNG previews use local Cairo fonts, which
lack some border/icon glyphs; the terminal text and original SVG are retained.

The final native run also inspected an [all-consequences capture](native/dark-80x24-all-consequences-long-title.txt)
([SVG](native/dark-80x24-all-consequences-long-title.svg)) at 80×24 with a
60-character title, all six nonzero loss counts and all four pending kinds.
The title, Stay and Close were wholly contained in the screen and compositor
clip after default Stay autofocus. This geometry case supplies a synthetic
impact snapshot to the real confirmation method; it does not claim actual work
in those categories. Compact consequence lines and removing the repeated bottom
question kept the complete dialog visible. The focused mounted regression was
RED at 27/29 rows before that copy correction and GREEN for both short and
60-character titles afterwards; receipts are in
`/tmp/console-tool-ux-close/max-risk-{red-qualified,green}.log`.

## Reproduction and scope

[Runner](native_check.py) reuses `../native_runner_args.py` and the
2026-09-19 approval-action-ownership runner pattern. Prepare an unused canonical
profile below `/tmp` with private `home`, `config`, `data` directories and a
`config.toml` containing absolute, contained `[paths].data_dir` and
`[database].USER_DB_BASE_DIR`. Set `first_run.setup_completed = true`,
`model_catalog.auto_refresh_enabled = false`,
`model_catalog.refresh_consent_recorded = true`, and
`console.onboarding.first_send_completed = true` for this fixture. Launch the
runner with `ROOT TMUX_SOCKET SESSION` in an existing 235×52 tmux pane with the
status line disabled, from the profile directory. HOME, USERPROFILE, XDG config
and data paths, and TLDW_CONFIG_PATH must select that profile before imports;
keep stderr attached to the terminal. The runner resizes to 80×24 itself.

This is an actual app/driver/controller/runtime journey with real blocking
human-decision rounds and a deterministic owning asyncio task. It does **not**
exercise a provider, external server or tool dispatch. Actual skill-confirmation cancellation, orphan-question cleanup without an
active turn, and failed-close retry are covered by the separate focused mounted
regressions, not by this native matrix. The extra all-kinds snapshot verifies
geometry only.

[Isolation](isolation.json) confirms that the real config hash and mtimes of
515 files under the real default-user data directory were unchanged. Production
source hashes were identical before and after the qualified run. Native QA
source and evidence changes stayed in this folder and
`/tmp/console-tool-ux-close-native`; production and test source were unchanged.

Earlier attempts are retained under that temporary directory: run1 reached the
fresh-profile setup wizard before Console; run2 hit question-card duplicate IDs
when fixture tabs were synchronously switched during pending repaint; run3’s
merged config was rejected by the shared fresh-profile validator; run4 passed
both wide cells but attempted the narrow tab click while a transient toast
covered it. Run5 qualified the earlier copy; run6 reverified all four real-close cells and
the crowded long-title geometry after the copy correction. Final run7 repeated
those five checks against the combined approval-card, screen and CSS changes;
the exported captures and source hashes are from run7. The qualified fixture
creates/switches tabs before arming the viewed question, uses distinct round
owners, and waits for notices to clear before checking paint and sending
terminal mouse input. All started app
attempts returned normally after cleanup; no production fixes were added for
those fixture issues.
