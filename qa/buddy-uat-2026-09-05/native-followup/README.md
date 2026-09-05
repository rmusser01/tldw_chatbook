# Physical Migu follow-up, 2026-09-05

Real foreground Terminal UAT on merged PR #2404, commit `f8cb939e2b`, passed without another production-code change:

- Move: rendered position `(41,31)` → `(69,25)` with real mouse-down, move, and release events. [Receipt](move-evidence.json), [screenshot](migu-native-moved.png).
- Lower-right resize: rendered size `28×15` → `40×21`. [Receipt](resize-evidence.json), [screenshot](migu-native-resized.png).
- Graceful exit: no app exception. Fresh process restored rendered geometry `(69,25,40,21)`. [Receipt](restart-evidence.json), [screenshot](migu-native-restored.png).
- All 22 separate terminal-protocol checks passed. [Probe](terminal-probe.json).

The receipts describe that exact tested commit; rebasing this evidence-only PR does not claim a new native run on newer dev code. TASK-31585 remains In Progress for its separate application-configured OpenAI realtime credential/UAT requirement.

The first long-running harness baseline detected normal settings changed since the prior day; do not claim they remained unchanged across that interval. The fresh restart baseline was unchanged. Background per-PID input delivered no mouse events; the explicitly authorized foreground gestures provided the native evidence.

The separate [server UAT report](https://github.com/rmusser01/tldw_server/blob/codex/migu-server-buddy-uat/Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md) covers six subsequent server repairs and the remaining cookie-authentication and stream-outcome feedback gaps.
