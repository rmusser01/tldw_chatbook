# TASK-35582: independent final live outcome review

**Outcome: incomplete qualification; keep AC24 unchecked and the PR draft.** The single final invocation reached both reviewers and a fresh verifier on native [2,1], but R2 and V1 did not seal. Runner exit 2 correctly stops at `live_report_incomplete_R2`. This outcome does not authorize another live attempt.

Reviewed the retained final manifest, runner refusal, bounded normal-tool observations, accepted submissions, first refusals and child prepared-request feedback for `qualification/attempt-bmrxalk1`, plus its referenced preflight manifests and read-only native accounting rows. Source pin remains `47093a3175c59db846a746c70b6f45d8b70cc50c`. No production source review or execution was performed.

## First completion failure: R2's corrected submission did not clear its earlier rejection

R2 accepted two findings at revision 1. Its next coverage submission (`sub-r2-f8e9021c-2`) was rejected `invalid_record`: `inspected_refs[0]` and `[1]` contained objects instead of strings (`string_type`), and `notes` was forbidden (`extra_forbidden`). `CoverageGap.source_ref` is optional; its absence did not cause the rejection. The boundary correctly rejected that malformed record.

R2 then corrected the coverage under a **different submission ID**, `...-3`; the host accepted it at revision 2. `reviewer_complete` under `...-4` was also accepted at revision 3. Nevertheless the final host report remains `incomplete`, with `rejected_submissions: ["invalid_record"]` and no seal receipt. Thus “completion submission accepted” did not mean “report sealed.” The earliest demonstrated failure relevant to the final gate is this rejection/correction/sealing boundary. The live artifacts establish the sequence and retained rejection; the exact policy for retiring a corrected rejection belongs in the root's narrow offline reproduction, not another live run.

R1 sealed an empty report, but missed the seeded empty-input regression. Sealed empty output is a valid report shape, not evidence that this reviewer found the seeded bug.

## Later independent failure: verifier acknowledgment/state recovery

V1's first **accepted** dispositions used `v1-batch1-dispositions`, expected revision 0. The normal tool observation records success, both accepted record IDs, revision 1 and acknowledgment cursor. Its accepted checkpoint has revision 1 and **no pending units**. The verdicts are substantively correct: the removed empty-input guard is confirmed, and the claim that strings are silently accepted is refuted by the retained type check.

The verifier then lost the usable acceptance feedback/state trajectory:

- V1 prepared request 4 contains the earlier generic project-instruction deferrals. Its observation timestamp precedes the successful disposition-submit timestamp. This is a **pre-dispatch deferral**; these artifacts do not prove that a generic notice replaced an already committed acknowledgment.
- Prepared request 5, after the observed compaction boundary, contains four messages and **no tool feedback**, whereas request 4 had twelve messages and feedback. The successful-submit observation precedes request 5. The durable accepted checkpoint still says revision 1/no pending units.
- The next batch read returns the same 3,654-byte body/hash as the original batch (`f06c90df9d2457ff9bad051430b4462c7b7c0c88fc969d6e5d7846574724f1b8`). The verifier sees revision 0/pending units again. Later repeated reads have that same hash. This is direct evidence of stale-page replay alongside newer accepted checkpoint state; it does not by itself locate the exact cache owner.
- The next submission uses a new ID and changed disposition prose at expected revision 1, and returns `disposition_conflict`; retrying at revision 0 returns `submission_revision_conflict`. No `verifier_complete` is observed. V1 ends `incomplete`, with no seal receipt. Its raw claim that the dispositions were never accepted contradicts the host's retained successful observation.

Accepted decisions were retained. The demonstrated gap is recovering a coherent current checkpoint/page/acknowledgment after history reduction, not missing historical evidence or a proven loss of the committed dispositions. Restrict the next offline reproduction to this exact boundary and the R2 rejection/correction boundary.

## What is present and what this run proves

The final evidence contains originals **R2-F1 and R2-F2**, exact expansion of each once, their confirmed/refuted dispositions, and **two readable unified-report pages**. It would be incorrect to claim the original mapping or unified pages are absent. Those pages truthfully retain the incomplete R2/V1 statuses; they do not establish a successfully sealed workflow.

Both reviewer reports were presented before the fresh verifier continuation. Three actual child receipts are attributed to two primaries with launches **[2,1]**, the default per-primary limit remains 2, and all runs share one causal chain. All 54 physical final requests have HTTP 200. The second primary eventually reports `stuck` after empty local-provider responses; neither HTTP success nor native child status `done` overrides the incomplete report gates. This final run does not newly establish duplicate coalescing or every fault control; prior deterministic evidence retains its original limits.

## Accounting and cleanup

The first narrow preflight consumed **2 requests** and failed because the **private synthetic instruction** omitted required `schema_version`; this was not a demonstrated production schema defect. The one authorized manual correction/retry consumed **3 requests** and passed the actual native tool/result round trip with native usage settled at **9,638 tokens and zero reserved**. Its scope remains a synthetic admitted verifier fixture, not completion of AC24.

The final invocation consumed **54 requests and 3 accepted children**. New totals are **59 requests / 3 children**; cumulative totals are **314 requests / 21 children** (255 / 18 before this grant). The approved aggregate financial ceiling 321 therefore has **7 numerically unused requests**, but both preflight attempts, the one final invocation and all three child slots are consumed. There is **no remaining live execution authorization**. The final manifest's zero-unused scope markers must not be mistaken for different physical-spend arithmetic or permission to replay.

The native automatic wake is `completed`. Its accounting rows contain one committed automatic generation, one committed child launch, 23 committed model calls and 23 settled token rows totaling **335,466 actual tokens**, with no reserved/uncertain rows. These automatic-continuation counters are narrower than the 54 physical requests and must not replace the transport spending count. The manifest records completed native shutdown, no running run IDs or owned fleet threads, zero Workspace connections/leases, zero registered Chat connections, and `retirement_uncertain=false`; source and temporary team are unchanged. Cleanup/accounting success does not seal the incomplete review.

**Next authorized work:** preserve this failed outcome honestly, reproduce only the two exact boundaries offline, run the one affected CI-failed test, and update the draft/evidence. No full rerun, new provider call, budget question or broader requalification follows from this review.

Not checked: no tests, source re-review, providers, network calls, credentials, new agents or new spending. This report assesses the retained outcome; offline reproductions have not been run by this reviewer.


## Narrow offline evidence added after the outcome review

Root's retained `qualification/final-boundary-control-39teze3_/result.json`, at unchanged source `47093a3175c59db846a746c70b6f45d8b70cc50c`, reproduces exactly the three coverage-validation errors stated above. It also confirms both bounded recovery observations:

- Correcting rejected coverage with a **new submission ID** leaves the report `incomplete`; the **same-ID correction control seals**.
- After accepted dispositions, the checkpoint is revision 1 with zero pending units, but a `cursor=None` read returns the **identical cached page at revision 0**. Resubmitting changed disposition prose returns `disposition_conflict`.

Both controls used **zero physical requests and zero product children**. They reproduce existing contracts and the observed recovery mismatch; no product behavior was changed and no later live success is inferred.

The first private control fixture mistakenly prepared twice under one bound run and received `run_binding_conflict`, also at zero requests. Root corrected only fixture isolation and the narrow controls passed; this was not a product repair. Root also reports the exact CI-failed composer test now passing independently: **1 passed in 5.46s**. This evidence completes the previously proposed offline/test checks; the remaining authorized work is the honest draft/evidence update, with no new live execution.

This addendum changes the factual rejection explanation and records root's supplied reproduction/test evidence only. This reviewer did not rerun tests or controls, revisit production source, make network/provider calls, or spend further requests.
