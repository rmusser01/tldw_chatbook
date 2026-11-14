# Final extraction review addendum — PR3162

Date: 2026-10-18. Same single read-only reviewer, applying requesting-code-review and Ponytail full. This addendum reviews the mechanical size-ratchet repairs after `review-first-use.md`; both earlier reports remain immutable.

## Reviewed identity

- Feature base: `5eb9a37b34a1b0af3e91ad360c0b997497ee7894`.
- Published/reviewed commit before local follow-ups: `566fcbe6ea56629fa8aedce64494a119cb40b327`.
- Earlier first-use source snapshot: `eec2090dd0ec897f1f9742badd96f6f16b9c30076a3c65c959d87382a1041511`.
- **Final frozen feature source: 104 files, SHA256 `22b25f2b59e4400f52ff78779ca699cf3d7fa13b411e1d9cba972f494816d86a`.** The reviewer independently reproduced this value.
- Eleven files differ from the earlier first-use manifest. Their sorted `path + NUL + file-SHA256 + newline` aggregate is `0117142bbc8c0a9342799e23f077878f54ef824bd6f198329ba7cf150ad3e4fd`. The full snapshot uses the original report's source-path filtering and algorithm.

This final snapshot supersedes earlier working-source identities for publication. The eventual immutable commit must reproduce it. No application/test source, index, branch or HEAD was changed by the reviewer; only the requested reports and isolated temporary review artifacts were written. No new agent, live/model request or full-suite run was made.

## Strengths and fidelity

- `tldw_chatbook/Chat/console_prepared_request.py:1806` and `:1839` receive the existing text-transform predicate and archive trace reseal method. The controller retains its static/bound class aliases. Resealing still reads the defining controller module at call time, preserving exact prepared-request/provenance classes, source ownership, original patch points and controlled trace-persistence refusal. The owner annotations, transaction, current-user identity and derived-artifact proof are unchanged.
- `tldw_chatbook/Chat/archive_recovery/runtime.py:677` receives the rider body, retaining persona, kill, native-tool and provider gates before shared request fitting/capture. The controller's surrounding admission and provider-finalization custody remain in place. Its `pending_recovery_reviews` helper retains exact provider/run/primary checks and the supplied `MCPPendingCall` reader; the existing approval merge and decisions remain unchanged.
- `tldw_chatbook/Agents/approval_provenance.py:137` receives the existing unanswered-decision helper, retaining supplied decision readers and its interrupt-module public re-export. `tldw_chatbook/UI/Console_Modules/wiring.py:3030` receives the complete hook provider through a normal bound-method alias, retaining lazy recovery hooks and inventory clearing. `tldw_chatbook/Widgets/Console/console_selection.py:266` receives the pure line-boundary helper with its transcript import seam preserved.
- The reviewer independently compared the moved execution bodies. Differences are limited to the reviewed static/bound descriptors, docstring indentation/forward annotation spelling, live controller-name qualification, parameter naming and pure local imports. Rider operations and the review collector match. The implementer's `extraction-fidelity.json` and `extraction-rider-review-fidelity.json` retain the same comparisons.
- Prepared-request, approval-provenance, wiring and selection destinations were already present in the reviewer's immutable-dev warm census. Recovery helper imports remain at their guarded use sites; no new eager recovery graph or module was introduced by these moves.

## Issues and ratchets

**Critical: none. Important: no unresolved finding. Minor: none in this extraction delta.**

Targeted checks exposed attributable growth in controller, interrupt, screen and transcript files. The repair moves coherent code; it does not compact comments or raise limits.

| File | Immutable dev lines | Final lines | Ceiling |
| --- | ---: | ---: | --- |
| Console controller | 29182 | 29181 | lowered 29184 → 29181 |
| Interrupt rounds | 6453 | 6449 | lowered 6453 → 6449 |
| Chat screen | 24380 | 24304 | lowered 24369 → 24304 |
| Transcript | 8649 | 8641 | unchanged 8353 |

The Chat screen method ceiling also falls from 740 to 737. The transcript remains above its inherited ceiling, but is eight lines smaller than dev after moving the existing pure selection helper. The broader diagnostic still has seven inherited failures: six identical to dev and this reduced transcript failure. This is explicit attribution, not a green whole-diagnostic claim or a hidden budget exception.

## Verification and assessment

Evidence is under `/private/tmp/archive-recovery-pr3162-merge-20261018`.

- `extracted-ratchets-green.log`: **6 passed, 62 deselected** for the changed controller/interrupt/screen limit nodes.
- `extract-ratchets.log`: **7 failed, 61 passed**, with the inherited failures attributed above. `first-use-ratchet-attribution.json` retains the dev comparison.
- `extracted-static-final.json`: all twenty follow-up Python files have no introduced per-file Ruff finding; syntax qualification is clean. The prior worker/publication gate remains clear.
- Before extraction, the closing dispatch/ownership/lifetime run passed **77 tests** and native qualification passed **12 tests**. Those establish the first-use repair, but are not represented as post-extraction results.
- `extracted-native.log`: **19 passed in 104.83 seconds**, covering the mounted history/draft button and agent journeys at both sizes, Preview/Open and selection-core qualification.
- At this report freeze, `extracted-contracts.log` was still running and the final isolated boot rerun was pending. Their closed outcomes must be added to the implementer's qualification note before publication; partial progress is not a passing claim.

**Final local code ready: yes, for the frozen 104-file source; no unresolved review finding remains.** Publication/merge still requires successful closure of the remaining post-extraction contracts and final isolated boot, immutable-commit source-hash verification, current-head required PR CI, branch currency and zero unresolved review threads. The human-authorized local-review substitution for Qodo remains specific to PR3162. The original full review and first-use addendum continue to supply the unchanged feature and lifecycle findings and their scope limits.
