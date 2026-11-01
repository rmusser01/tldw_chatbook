# PR3298 backlog integration evidence

The completed-turn witness selector now checks exact descriptors and the permitted
start sequence before choosing that proof. Its full durable validation remains
mandatory. An unrelated witness no longer masks the separately proven rewritten
tail after dictionary-transformed Edit & resend or Compact now. The existing
256 stored-sequence limit, whole-tail restriction, memory/branch anchors, final
binding, native custody and immutable recorded history remain intact (ADR097).

Initial integration base: `2561efd6587a263de6eb564b4254a09883f7f9b3`.
Prepared head before repairs: `0d646b9daad48dc1fe70b1747dea77b801bbb081`.
Receipts below are in `C:/Users/GDesktop-1/.codex/worktrees/pr3298-resync/`.
Each run used a fresh private Windows fixture root, selected the actual config
path before imports, and recorded matching source manifests before and after.
No live provider, normal user profile, full suite or hosted CI was used.

| Receipt | Result | Qualification |
| --- | --- | --- |
| `r-98c08663` | 4 failed | Planner reached the equal-count proof defect; three controller cases stopped at an outdated dictionary fixture signature and were not causal product evidence. |
| `r-558a3b87` | 3 failed | After correcting that signature, both edit cases failed at `completed_tool_turn_revision`; transformed compaction failed at `completed_tool_turn_range`. Production was unchanged. |
| `r-d467ce29` | 90 passed, 9 failed | All four new regressions, rewritten-tail/history/planner, final values, project-turn and critical-pin controls passed. Six older controls stopped at config admission; three native controls stopped before provider dispatch. |
| `r-75071491` | 2 passed, 7 failed | Existing bootstrap-profile marker restored both dictionary-send controls. Four tamper wrappers still lacked the current call signature; three native controls remained blocked. |
| `r-b0b3f8ec` | 4 passed, 1 failed | Forwarding current substitution arguments restored all four tamper controls. The remaining native positive exposed the denied socket call. |
| `r-0c9b16b4`, `r-8aa80106` | 1 failed each | Diagnostic attempts only. Full stack identified Windows Proactor loop creation through the stdlib socketpair fallback, intercepted by the test's blanket socket patch. |
| `r-6f2876d7` | 2 passed, 1 failed | Stale-custody and rollback negatives passed with the existing repository socket guard. Native positive completed its first turn but a duplicate second-turn socket patch still blocked loop creation. |
| `r-f27d1664` | 1 passed | Removing that duplicate restored the complete two-turn native replay/rounds control. |

The 99 selected cases therefore have passing results across the qualified
receipts; there was no single all-green 99-case rerun. All runs had zero errors
and zero skips. Earlier failures remain recorded, not relabeled as passes.

Fixture changes retain the collection-admitted private profile and forward the
current method signatures. Native controls now retain `Tests.network_guard` in
BLOCKED mode, with explicit assertions that neither unrestricted nor loopback
networking is allowed. That guard admits only its thread-local real socketpair
bootstrap. DNS denial and fake HTTP transport remain, as do all original wire,
immutability, replay and custody assertions. This follows the existing
TASK15458/ADR058 Windows socketpair lesson; no production guard was changed.

Scoped static checks: all four production diagnostic/sink/path inventories
match. Changed Python files retain their baseline Ruff diagnostics (28 existing
exact-type E721 diagnostics and two existing fixture F811 diagnostics); no new
diagnostic was introduced. The measured service pin is 6765 lines, explicitly
updated under the user's size waiver. Current-dev native continuation and replay
integration, census and unrelated production changes were preserved.

The incoming wider-history follow-up collided with the existing SSE TASK37600.
A 300-snapshot filename audit found maximum ID38268; the incoming trace task was
renumbered to TASK38298 and its trace-only references updated. The existing SSE
task and its references remain unchanged. Wider histories, plain compaction,
memory-anchor strengthening, continuation editing and broader race journeys stay
in their existing explicit follow-ups.

Independent prepared-source review found no Critical or Important issues and
matched all 81 receipt hashes. A separate fixture review found no new issue or
guard bypass. Final current-base rebase, targeted integration and exact-head
approval are recorded below when completed. This session performs no release.
