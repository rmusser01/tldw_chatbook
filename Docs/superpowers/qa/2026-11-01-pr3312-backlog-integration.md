# PR3312 integration evidence

Base: `820ea70cc5b14d2809764abf7a15fc132db34430`. Incoming head: `9fabd0f9aca6a244545aea112641e8dce236ae19`. The user waived hosted CI and size limits and substituted independent requesting-code-review for Qodo. Drafts and PR3267/3297 are excluded; this session performs no release work.

## Repairs and preserved contracts

- Rebased provider-failure presentation and trace-blocked recovery onto current dev, retaining native continuation custody, operation-owned SQLite access, and each stream/fallback call's admission, trace and usage ownership.
- Count each consumed adapter admission. Record an empty HTTP200 before fallback, so a later rejection cannot turn an answered call into refusal-only replay authority. Recovery still requires exact accepted ownership, a single-use proof, and equality between dispatched calls and proven refusals.
- A suggested exception status is not HTTP evidence. A real adapter ReadTimeout was projected as 503 and incorrectly authorized replay before repair. Refusal proof now requires an actual response plus qualified transport retry history. Requests redirects, missing history, any earlier transport failure, suppressed exception chains and unqualified SDK/manual retries stay uncertain. Direct local HTTP calls also reject redirected histories. No response bodies or exception objects are retained in the proof.
- A newer dispatch clears the previous rejection as the latest verdict. Unknown delivery remains unmatched and does not manufacture the card's provider-contacted fact.
- The exhausted-error UI test now waits for actual painted HTTP copy. A captured failing frame showed the old Waiting for a reply row even after the run and stored row were failed. The original visible status/meaning, three-POST, failed-row and Inspector assertions remain.

ADR required: no new ADR. Existing ADR097 semantic trace, ADR211 provider-failure presentation, ADR230 native ownership and ADR029 private diagnostics apply. Plan: [PR3312 integration](../plans/2026-11-01-pr3312-backlog-integration.md).

## Focused verification

Private receipts are retained under the session's `pr3298-resync` parent. Every completed receipt listed below records unchanged source hashes during execution. No full suite was run.

| Receipt | Result |
| --- | --- |
| `r-6a02558f` | Four pre-repair failures through real vLLM/llama.cpp adapters: empty 200 then refusal and refusal then refusal, followed by a reservation failure. The generic empty 200 case incorrectly granted recovery proof. |
| `r-499ebb83` | Pre-repair real-adapter transport timeout incorrectly allowed Capture-Off replay; the expected refusal was not raised. |
| `r-c9e3ff3c` | Four repaired cases passed; one expectation incorrectly assumed an exhausted vLLM 429 retained a response. It is deliberately unproven and the expectation was corrected. |
| `r-274429aa` | Recovery, stream fallback and usage accounting: 78 passed, two stale expectations failed. Both expected an old rejection to remain after a newer unknown dispatch; these were corrected. |
| `r-6577a308` | 117 passed, three visible exhausted-error assertions failed. Includes the corrected recovery proof, HTTP retry-history controls, SQLite reservation lifecycle, all mounted recovery actions, contact copy, and adapter admission/cancellation guards. |
| `r-c26caeae` | Single 429 diagnostic reproduced the stale Waiting row in the actual frame after FAILED was published. |
| `r-f2cacbb8` | All three exhausted 429/500/504 UI cases passed after waiting for the actual painted status; zero failures/errors/skips. |

Earlier harness attempts exposed the required loopback marker, typed HTTP exception family, and the adapter's internal429 retries; those attempts are not used as causal product evidence. The mounted test results establish functional completion, not latency or native-terminal performance. The final UI change is test-only; production source is unchanged from the 117-pass run.

All 27 changed Python files parse. Scoped Ruff comparison against the exact base reports 109 existing diagnostics versus 108 current, with zero introduced diagnostics. The repaired tests pass Ruff directly; exact-type diagnostic/transport guards retain explicit narrow lint explanations. Diff whitespace checks pass.

Diagnostic inventory review found exactly two added controller statements, both constant cancellation messages with no payload interpolation. Regeneration verifies 716 owners, 1065 TASK-492 calls, 56 TASK-31551 calls, 7790 TASK-494 calls and 16 sink files. Under the explicit size waiver, only affected controller/gateway/session pins were updated to measured sizes 29312/8528/6991.

## Remaining limits

TASK-33621.70-.73 remain open: disabled Retry anyway for unreconstructable composer sends, pre-transaction refusals still classified unknown, temporary-chat contact uncertainty, and Cancel's empty-composer/unwritable-database limits. TASK-33621.6 AC5 remains unchecked and the task stays In Progress. A borrowed/nested transaction is not certified by the outermost production reservation rollback proof. Hosted/suppressed/exhausted adapter errors without positive evidence intentionally retain conservative recovery; they are not claimed to be proven refusals. Historical live-provider evidence in the original tasks is not a new live-provider qualification.

The independent final-head review is retained as `pr3312-integration-review.md` outside the checkout before publication.
