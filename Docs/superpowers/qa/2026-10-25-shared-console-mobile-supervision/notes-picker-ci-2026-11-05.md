# Notes picker CI readiness correction

The Windows Eval recovery repair at `e0422a4c6f3a6c0f4788912d73aac4409feb9805`
passed the actual Windows lane on its first automatic run, with the original
45-second restore subprocess limit unchanged: 109 passed, one existing FIFO
skip. The retained journey and all 26 policy regressions ran unskipped.
[Hosted Windows result](https://github.com/rmusser01/tldw_chatbook/actions/runs/41940008548/job/126958086372).
Whole browser UAT397 passed all 15 cases in 508.42 seconds; the original mounted
journey also passed with all nine retained artifact hashes verified. Both were
physically retired with unchanged source and independently reviewed.

That head's later [UI17 shard](https://github.com/rmusser01/tldw_chatbook/actions/runs/41940008359/job/126963140839)
failed one Notes folder-picker case after its single Move click; 94 cases passed.
The log does not identify the actual mouse recipients, so the exact hosted
pointer phase remains unknown. The expected DB=None fixture startup warnings
are not a diagnosis of this failure.

The unchanged local control398 passed. Native pending-layout diagnostic399
then demonstrated a concrete gap in the helper: the same enabled, attached,
unprotected note's button moved from (43,21) to (45,21) during Pilot's first
pause. All three forwarded mouse events hit the toolbar at the old point;
mouse capture was empty. The original ten-second modal assertion failed.
The permanent pending-layout variant also failed that assertion with the old
helper (401 and final arm403); run400 was a retained selector-setup error and
is not regression evidence. Independent review caught and corrected an initial
regression-setup lookup that could precede button composition.

TASK-38901 changes only test readiness: await a Textual refresh within the
existing ten-second readiness budget, reacquire the live enabled/visible button,
and verify its exact origin is the effective current compositor target. Then
issue the original single Pilot click. No click retry, direct handler call,
mouse-capture release, extra timeout, changed folder/filter/title/database
assertion or production UI change is involved. The original unperturbed journey
remains alongside the native pending-layout variant. The exact affected 43-entry UI17 cohort passed all 96 expanded cases locally
in 303.29 seconds (404), including both parameters and the original folder move.
The process was physically retired with unchanged source. All 3,488 production
entries match the Windows-qualified source; original browser/mounted UAT inputs
remain unchanged. Current-head hosted gates and exact-head independent review
remain required before readiness/auto-merge.

ADR required: no; this is test synchronization within existing boundaries.
Hosted logs and retained diagnostic copies are in
`.superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/landing-2026-11-02/ci-repair-2026-11-04/windows-eval-timeout-2026-11-04/ui17-e042-failure`.
Original local logs, source maps and retirement receipts remain in the same
implementation root under `dev-integration-2026-11-01/integrated-uat-2026-11-01`.
