# Physical Migu follow-up, 2026-09-05

Real foreground Terminal UAT recorded the following interaction results without another production-code change:

- Move: rendered position `(41,31)` → `(69,25)` with real mouse-down, move, and release events. [Receipt](move-evidence.json), [screenshot](migu-native-moved.png).
- Lower-right resize: rendered size `28×15` → `40×21`. [Receipt](resize-evidence.json), [screenshot](migu-native-resized.png).
- Fresh process (PID `46565`) restored rendered geometry `(69,25,40,21)`. [Receipt](restart-evidence.json), [screenshot](migu-native-restored.png).
- All 22 separate terminal-protocol checks passed. [Probe](terminal-probe.json).

The session targeted merged PR #2404, commit `f8cb939e2bd3a111555acc8d87a4b4907ee2268e`. However, the native launcher did not capture a Git revision or dirty-state snapshot in these receipts. That revision is session context, not independently verified tested-commit provenance. The original JSON is preserved without retroactively inserting a `tested_commit` assertion. Rebasing this evidence-only PR does not claim a new native run on newer dev code. TASK-31585 remains In Progress for its separate application-configured OpenAI realtime credential/UAT requirement.

The restart receipt proves restored geometry, not the preceding native process's exit outcome. The surviving runtime `exit.json` reports `app_exception: null`, but contains no PID, timestamp, or process return code; it cannot bind that result to the earlier move/resize PID. A graceful, exception-free native exit is therefore unverified by this published evidence. The separate terminal-protocol probe does not fill that native-process evidence gap.

The durable evidence is this tracked directory, `qa/buddy-uat-2026-09-05/native-followup/`. Two distinct temporary directories were used: `/private/tmp/chatbook-migu-dragging-20260905` is the source checkout (`source_root` in each native receipt); `/private/tmp/migu-dragging-uat-20260905` is the runtime directory containing the launcher, isolated profile, and original receipt files. Temporary paths describe the original session and are not required to read this published record.

The first long-running harness baseline detected normal settings changed since the prior day; do not claim they remained unchanged across that interval. The fresh restart baseline was unchanged. Background per-PID input delivered no mouse events; the explicitly authorized foreground gestures provided the native evidence.

The separate [server UAT report](https://github.com/rmusser01/tldw_server/blob/codex/migu-server-buddy-uat/Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md) covers six subsequent server repairs and the remaining cookie-authentication and stream-outcome feedback gaps.
