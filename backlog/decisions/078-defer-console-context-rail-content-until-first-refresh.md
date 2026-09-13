# ADR-078: Defer Console Context rail content until first refresh

Status: Rejected
Date: 2026-08-21
Related Task: [TASK-19505](../tasks/task-19505%20-%20Measure-and-reduce-remaining-Console-first-interactive-mount-cost.md)

## Decision

Do not defer the Console Context rail in production. The candidate improved
median first-interactive time by 18.31% and kept median full-ready within its
budget (+0.13%), but Enter-to-worker p95 regressed 12.39%, above the allowed
10%.

The shipping screen therefore retains eager Context composition and fresh
screen construction on every navigation. The rejected candidate remains
reproducible only inside `Tests/Performance/run_console_mount_profile.py`; no
deferred lifecycle, placeholder, or hydration hook remains in production code.

## Context

TASK-18909 removed the last material synchronous application hotspot and left a
roughly 1.45-second Textual mount/CSS/compositor floor. TASK-19505's isolated
170x48 Pilot probe measured 30 balanced warm navigations per variant:

| Variant | Widgets | First-interactive median | p95 | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Production baseline | 405 | 766.1 ms | 1,188.2 ms | — |
| Empty Inspector content | 331 | 681.1 ms | 1,218.9 ms | 11.10% |
| Empty Context content | 295 | 602.8 ms | 1,038.3 ms | 21.32% |

The Inspector control misses the required 15% threshold. The Context rail is
the only measured secondary subtree that clears it, accounting for 110 of the
405 mounted widgets.

The subsequent real candidate A/B measured 30 eager and 30 deferred fresh
screens in balanced order. Deferred first interaction improved from 774.9 ms
to 633.0 ms median and key p95 improved, but Enter p95 moved from 22.6 ms to
25.4 ms. That 12.39% tail regression rejects the candidate.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Cache and remount `ChatScreen` | Reproduced dead message pumps and stale compositor state; fresh-screen lifecycle gates prohibit it. |
| Defer the Inspector rail | Its measured 11.10% maximum benefit misses the acceptance threshold. |
| Permanently remove Context content | The rail is a primary Console navigation and run-inspection surface; removal is a product regression, not an optimization. |
| Flatten all rail wrappers | Could reduce total mount work but is a broader UI redesign; the measured one-refresh deferral is narrower and reversible. |
| Keep the full eager tree | Selected after the real deferred implementation missed the Enter-to-worker input-p95 budget. |

## Consequences

- Console mount behavior and lifecycle remain unchanged.
- The profiler preserves the rejected A/B so future work can reproduce the
  result without carrying test-only behavior in `ChatScreen` or
  `ConsoleLeftRail`.
- Any future mount optimization must improve first interaction without moving
  the latency into input tails; this candidate is not a safe basis for
  production work as measured.
