# SDD ledger — plan: Docs/superpowers/plans/2026-09-08-library-phase-c-media-graduation.md
Phase C (media graduation, resident canvas) SDD start — the program's founding motivation, as a designed behavior-change series (NOT pure moves; TDD applies). Branch feat/library-phase-c-resident-canvas off origin/dev (post-program-close 7e81ed55d). TASK-31880 coverage-gate fix in flight on this branch (dispatched before the plan; its closure is the media coverage-density graduation criterion).
Pre-flight conflict scan (4 tasks):
| pair | producer/consumer | finding |
| T1->T2 | mechanism decision + failing acceptance test -> implementation | consistent; T2 is TDD from T1's red test |
| T2->T3 | residency-forced ownership clarity -> handler migration scope | consistent; T3 scope-honest per spec |
| T1..T4 | probe mechanics per recipe §9 (same-location, order-swap, interleave) | consistent |
| plan vs rubric | spike code labeled throwaway; no vacuous tests mandated | clean |
Named risks: TASK-31521 composition seams (per-task rulings required); the canvas_sync/TASK-32089 dispatcher debt adjacency (map, absorb only if required, labeled commits); the probe's location artifact (§9 corrected pairing mandatory). No conflicts requiring pre-execution rulings.
