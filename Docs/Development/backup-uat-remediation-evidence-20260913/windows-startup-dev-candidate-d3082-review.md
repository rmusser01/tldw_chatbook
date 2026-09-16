# Native Windows startup comparison — d3082 / TASK-32562

Run34831235874, Windows job103934821964, exact candidate d3082ef98211e3ad42273ac2d2dcbbf43b6d93cc. The same workflow passes on macOS and Linux. Windows non-UI selection passes31 cases; its four original full-app cases still fail their unchanged60-second test limits.

The optional failure-only diagnostic runs the same claim-authority case in two fresh interpreters, with each private profile selected before imports. Dev source4631b60f8dd9623fc55bf16f4a37e29fcb1240c7 passes1 case in48.42s under profiling; candidate reaches the existing60-second limit. All8 profile snapshots report source_matches=true and no observer errors. The diagnostic does not alter the original failure or authorize a timing exception.

The final candidate snapshot at59.922s records264 storage acquisitions (27.97 cumulative seconds),112 guarded configuration wrappers (29.07s),80 configuration loads (16.66s),15 get_user_data_dir calls (15.94s), and18 Canvas policy reads (13.39s). The application constructor has completed one call taking22.93s by the44.109s snapshot. Native operations include140541 opens,77792 handle-stat calls and77811 security-descriptor reads. In contrast, the dev case completes; its top product costs include UI/CSS, imports and asynchronous shutdown.

These are overlapping cumulative profiler times and cannot be added. Coroutine/generator call counts may include resumptions; snapshots are partial until completion. Profiling adds overhead. The evidence establishes substantial backup admission/configuration costs during the failing Windows startup; it does not establish that all cost belongs to Canvas, that a proposed optimization is safe, or that the separately reproduced Canvas/config lock cycle caused this run. The macOS whole-case890 Canvas bindings were a different completed execution and must not be substituted for this Windows count.

Next investigation is limited to reusing existing checked configuration lifetimes around concrete synchronous repeated-read boundaries, after resolving the independently reproduced lock-order issue. Constructor-wide grouping is rejected because initialization joins workers needing the same config lock. No native authorization, permission check, policy cache, product deadline or startup timing limit is weakened.

Preserved raw job log: /private/tmp/uat-gguf-windows-d3082.log. Bounded profiles: windows-startup-dev-candidate-d3082-profiles.json. Run: https://github.com/rmusser01/tldw_chatbook/actions/runs/34831235874
