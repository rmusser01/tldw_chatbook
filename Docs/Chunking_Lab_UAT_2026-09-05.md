# Chunking Lab live UAT — 2026-09-05

## Outcome

Initialization is fixed locally. The core single-sample A/B workflow passes the
real-terminal acceptance pass below, with two non-blocking presentation findings.
This is not exhaustive release/platform sign-off. At UAT completion no follow-up
commit, push, PR, or merge had been performed. The user subsequently authorized
publishing this correction as a separate PR against dev; merging is not included.

## Environment and cause

PR2416 merged at `e990738b2812876c2593b91f62d0b2c5b2e3b69d`. Starting HEAD
`3ea9db82743879c44480cb9db21383eeaaefc49d` has the identical Git tree
`88b505689eea0515bfa82b3864dfaa6967492cf6`. Acceptance adds only the local
mount-scheduling correction to production code.

macOS arm64, Python3.12.11, Textual8.2.8; real `python -B -m tldw_chatbook.app`
in a dedicated tmux PTY. Disposable profile and minimal environment selected
before imports; null keyring, no provider configured, catalog refresh disabled.
No user database/config was copied. All authoring actions used terminal input,
not Pilot, widget mutation, or coordinator injection. Read-only SQLite checks
independently inspected the resulting durable state. Main checkout untouched.

Original UAT remained in loading through reopening and process restart. Temporary
tracing in a separate diagnostic launch observed `_load` enter and exit with
`mounted=False`, the message widget present, and no coordinator. Textual can
yield during Mount dispatch before setting `is_mounted`; the worker therefore
mistook initial mounting for teardown. Scheduling the existing lazy worker with
`call_after_refresh` fixes that ordering without removing teardown guards or
changing exclusivity/error handling. Existing
[ADR-118](../backlog/decisions/118-chunking-lab-local-execution-and-recovery.md)
applies; no new ADR, schema, dependency, runtime or global-startup change.

## Live acceptance

| Scenario | Observed result |
| --- | --- |
| Library entry without provider setup | Ready and editable |
| Paste/autosave | Exact26-word synthetic sample; Saved locally |
| Advanced JSON import → controls → Full JSON | max_size changed5→10; name/description/tags, pre/post operations and nested Unicode metadata retained |
| B preview, pin A, edit B, Run both | A six chunks, B three chunks; both completed/local |
| Full configuration execution | Configured prefix applied; preprocessing recorded; changed-text alignment truthfully unavailable |
| Save B | UI success; Media DB independently confirms reusable name/body and preserved metadata |
| Execution inspector | Local backend, engine/execution versions, run IDs and hashes visible |
| Leave/reopen | Comparison, completed results and inspector choice restored |
| Force-kill/restart | Exact sample, candidate drafts/pin and complete result documents equal the pre-crash snapshot; no rerun |
| Invalid JSON/reopen/discard | Raw invalid edit and error restored with previous results; explicit discard restores the exact valid state |
| Viewports | Real shell/configuration observed at80×24,120×40,160×50; automated tests cover full workflows at all three |

Recovered A run `98937114-606b-4d0f-b1a9-7a809e36a067` has six chunks;
B `475efa9e-6876-4e2d-9c44-814f8803bb8f` has three. First chunks start
`UAT 0/6: alpha bravo` and `UAT 0/3: alpha bravo` respectively.
Sample SHA256: `b98a6af0680e8844cac315a8c361a38b978ca714bbf8c3111d568e9127e73d35`.
All retained results record backend `local`. Nested metadata remains
`{"custom":{"keep":[1,"café",{"nested":true}]}}` in drafts, captures and catalog.

Local ignored evidence: `.superpowers/chunking-lab-uat-CdhW74/` in this worktree.
It contains the original failure report, actual numbered terminal frames,
fixture, diagnostic/read-only audit helpers, snapshot and JUnit reports.
Key frames:19–20 initialization/sample;29–31 preview/pin;35 comparison;37 save;
39–40 execution/reopen;42 crash recovery;43 full JSON;44/47–48 invalid draft/error;
52 correction. `audit-before.json`, `audit-after.json`, and `audit-final.json`
record independent checks. Final equality covers complete result bodies and
candidates, not merely their IDs. The disposable profile is retained; only its
tmux session was removed after verifying normal exit0.

## Automated verification and review

The yielding inherited Mount-handler regression failed pre-fix with the expected
readiness timeout. After the correction, the focused regression/teardown pair
passed2. An initial combined run passed65/failed1 because the new test checked
disabled state before a superseding render finished; using the existing bounded
Lab-worker settling helper addresses that assertion timing.

Final command:

```bash
python -m pytest Tests/UI/test_chunking_lab_screen.py Tests/UI/test_chunking_lab_recovery_flow.py Tests/UI/test_chunking_lab_results.py -q --tb=short --show-capture=no
```

**66 passed in70.30s**, one existing Requests compatibility warning. Scoped Ruff,
format checks and whitespace checks pass. Independent read-only review found no
findings in the scheduling/test diff and independently passed both regressions.
No full suite was run.

## Remaining qualifications

Two non-blocking presentation findings remain **unfixed**:

- A prior “Candidate … is not executable” message remains after correcting an
  incomplete control and successfully running both (frame35). Saving later
  replaces it with the success message.
- “Replace A” is sometimes clipped to “Repla”, including on reopen
  (frames31/42/45); other layouts show it fully. No action failure established.

Rapid batched terminal inputs sometimes missed focus or left an incomplete
control; those attempts are not counted as success. Subsequent actions were
verified against visible state and persisted output. The incomplete control was
correctly refused, then deliberately completed with10.

Palette entry, selected-Library-text handoff, every file/excerpt boundary,
template reload/export, failed-save recovery, cancellation and Clear were not
exhaustively repeated in the real terminal. Automated evidence is not live UAT.
No Windows/Linux, crash-during-write, zero-keystroke-loss, memory-boundary or
network-isolation qualification is added. Force-kill occurred after Saved locally.
