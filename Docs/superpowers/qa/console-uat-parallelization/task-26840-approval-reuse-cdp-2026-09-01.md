# TASK-26840 approval prompt repeat-round UAT evidence

Date: 2026-09-01

## Scope

Rendered Textual-web/CDP UAT of the production `ChatApprovalCard` repeated
ordinary one-row path. The test used two native tool calls returned by a live
local llama.cpp provider, clicked the production fast-decision controls, and
checked the row/control identity contract that implements the render-speed
change.

This is scoped component UAT, not a claim that the complete Console run
orchestration passed end to end. A complete Console attempt was performed first
and encountered unrelated pre-existing project-inspector and provider-preflight
failures recorded under **Full-Console blockers** below.

## Environment

- Branch: `codex/approval-card-fast-path`
- Evidence capture HEAD: `6fb7ae170`
- Browser viewport: `2048x1220`
- Textual-web URL: `http://127.0.0.1:19137/`
- Local provider: llama.cpp OpenAI-compatible endpoint at
  `http://127.0.0.1:19099/v1/chat/completions`
- Model: `Qwen3.5-4B-UAT`
- Isolated HOME: `/private/tmp/tldw-approval-uat-26836.XHn8sn/home`
- Isolated config:
  `/private/tmp/tldw-approval-uat-26836.XHn8sn/home/.config/tldw_cli/config.toml`
- Isolated data: `/private/tmp/tldw-approval-uat-26836.XHn8sn/data`
- Isolated cache: `/private/tmp/tldw-approval-uat-26836.XHn8sn/cache`
- Production component source:
  `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`
- Production styling: consolidated application stylesheet via
  `Tests.UI.consolidated_css.ConsolidatedCSSApp`

The temporary host contained no mocked provider response or seeded assistant
transcript. It requested both tool calls from the live provider during app
startup and converted the returned native call IDs, function names, and JSON
arguments into the same pending-call mapping accepted by
`ChatApprovalCard.set_batch`.

## UAT procedure and result

| Step | Expected | Observed |
|---|---|---|
| Open round 1 | Production card shows the first live calculator call | `{"expression":"(6 * 7) + 1"}` rendered with the real card controls |
| Approve round 1 | `Approve once` records the first call and presents the next round | First call recorded as `approve_once`; round 2 appeared with no intermediate blank/error state |
| Inspect round 2 | Mounted details row is reused, commit controls are fresh, and all visible content changes | `row_reused=True`, `fresh_controls=True`; expression changed to `{"expression":"(8 * 9) + 2"}` |
| Measure update | Repeated one-row `set_batch` completes without visible lag | Instrumented synchronous call completed in **9.37 ms** in the captured run |
| Deny round 2 | Deny resolves only the second live call | Final state recorded `YC2XHI4N3jnhuVnNCoSESvO0HNblwEyY=approve_once` and `5FRo72l67x25NJdoUkdNUqGfLhHNCz3H=deny` |

Result: **passed for the scoped production-card flow**.

The single captured `9.37 ms` call is a UAT observation, not the statistical
performance claim. The task's 60-update mounted probe remains the quantitative
evidence: settled median/p95 changed from `8.492/14.305 ms` on the exact base to
`7.563/8.795 ms` on the feature tree (about 11% lower median and 39% lower p95).

## Evidence

- `task-26840-approval-reuse-cdp-2026-09-01.png`
  - Shows round 2 rendered from the second live provider tool call, with
    `row_reused=True`, `fresh_controls=True`, the `9.37 ms` observation, the
    changed expression, and the production decision controls.
- `task-26840-approval-reuse-decision-result-cdp-2026-09-01.png`
  - Shows the terminal PASS state with separate `approve_once` and `deny`
    decisions keyed by the two live call IDs.

Both PNGs were visually inspected after capture.

## Full-Console blockers

The isolated full application was launched first through Textual-web at
`http://127.0.0.1:19136/` and configured against the same live llama.cpp model.
It did not reach an approval card because of behavior outside TASK-26840's
rendering seam:

1. The Default workspace correctly exposes no filesystem authority, so a
   writable workspace binding was created through the real UI.
2. Opening/closing or recovering the Console conversation inspector repeatedly
   raised `NoMatches` for
   `#console-inspector-next-send-loading` after the inspector had detached.
3. First-send project binding selection also raised `NoActiveWorker` from a
   `push_screen(..., wait_for_dismiss=True)` call outside a worker.
4. After resuming an explicit project-off conversation to avoid those paths, a
   built-in calculator request remained at provider validation/preparation for
   more than the configured bounded preflight interval and sent no request to
   the already-healthy llama.cpp server.

No full-Console end-to-end approval is claimed. These failures were not changed
on this branch because they are outside the approval-card render optimization.

## Isolation verification

After both UAT paths and all local services were stopped, the real profile
fingerprints still matched their pre-UAT values:

- `~/.config/tldw_cli/config.toml` SHA-256:
  `455566bd8947f02354deacdfd65da0608c22b3ffc3e1bbdfdd94f52e01a840fb`
- Aggregate SHA-256 for files under `~/.local/share/tldw_cli`:
  `78af5cea5aa50d2835286b4c981d729a0d4e2189928eede1fab9347a98f065b2`

## ADR check

- ADR required: no
- ADR path: N/A
- Reason: this evidence verifies an implementation-local rendering
  optimization; it does not change permission policy, persistence, ownership,
  provider contracts, security boundaries, or long-lived UI structure.
