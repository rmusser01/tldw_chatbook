# Windows e4f8 default replacement/Evals evidence review

**Verified success for this exact native run:** `34880766232`, revision `e4f8aa7a62ea9800e3867d47036fafb4a1ec6b5f`. JUnit reports **42 native passed in 1.374s**, then **10 product passed in 1658.204s**, no errors, failures, skips or missing outcomes. The default replacement/later-rollback case took **1541.985s**; the other nine are the Evals retention/provenance cases. This partition does not cover the separately failing startup or mounted-support cases.

Artifact root: `/private/tmp/uat-windows-e4f8-default/backup-platform-windows-2022-py3.12-replacement-default-evals-e4f8aa7a62ea9800e3867d47036fafb4a1ec6b5f`.

I independently rehashed all **37** `artifact-sha256.json` entries successfully and read both JUnit files. The source receipt names that exact Git revision, a clean status and `private_tracked_head_copy` execution. Existing root verification and Git-blob comparison record one immutable installed-package receipt, **2867 files / 2475 matching Python source blobs** (2443 Windows-CRLF comparisons and 32 exact), zero mismatches. Wheel SHA256: `5b41e80b3857302988b1820aae48e58490e1d96f6c2e1c6d7891722f6285deb5`. Child drivers assert the imported package path equals the installed package; they use repository test helpers alongside it. This is native installed-product automated/headless UI evidence, not manual keyboard acceptance or a claim that every non-Python source asset is a Git-blob match.

## Completed behavior

`test_default_profile_service_r0/f9-child.log` records actual F9 restart into fresh recovery, archive inspection, required Persona safety selection, first credential-review stop with unchecked omissions, untouched Abort, a fresh acknowledged review, and the second replacement terminal `succeeded / restoration_validated` with no issues. The parent verifies that terminal result and separately reopens the ordinary installed app.

`later-child.log` progresses beyond the earlier ambiguous cutoff through: stale preview discarded; review; credential stop; untouched Abort; explicit omissions and changed-acknowledgement re-review; **later rollback succeeded with restoration_validated**; recovery-copy catalog completion; next-copy selection; new safety-copy inspection; body completion; app close; loop close. Exact tested `_LATER` source requires both old and newly created copies to be verified, stale plan/acknowledgements cleared on selection, new archive verification succeeded, and its sealed credential policy equals `rollback`. Successful JUnit and child return code therefore cover those final assertions, not merely a printed success checkpoint.

Three separate ordinary-reopen phase files reach saved-note read and `ORDINARY_REOPEN_COMPLETE`: Abort **86.250s**, replacement **70.250s**, later rollback **75.875s** from each recorded child entry. Their exact native CLI driver checks the expected saved note title/content and mounted UI readiness, then exits the headless app normally. The outer default test finally checks that profile-owned roots exclude the config control parent and recovery controls. It passed after the later ordinary reopen.

## Measured timing and limits

| Interval from fixed later-child checkpoints | Seconds |
|---|---:|
| Begin → final changed-acknowledgement review |464.578|
| Final review → validated rollback |308.407|
| Validated rollback → loop closed |8.031|
| Begin → loop closed |781.016|

The absolute monotonic values are not durations. The 781.016s interval excludes earlier imports/observer setup and trailing process shutdown; it is not the exact `subprocess.run` duration. Successful parent return independently proves the child finished under its unchanged **900s** observation limit. The full default case completed under its unchanged **2400s** outer test bound. Ordinary reopen retains its separate135s Windows bound; all product operation waits/assertions remain in force. Direct comparison of exact ec139 and e4f8 test source confirms the900/2400 expressions were not increased.

Preserved ec139 failed at the900s parent limit while later publication/validation was still advancing and never produced the terminal assertions. Its begin→final review was555.531s, versus464.578s here; a comparison of these two completed checkpoints does not identify why their executions differed. Earlier9aff completed its observed later interval in896.593s; this newer pass is another actual completion, not a controlled performance experiment or proof of a specific fix for ec139.

The current evidence supports retiring an otherwise unnecessary proposal to expand this harness budget: the required actual case has now passed with its existing bound. Preserve prior failures and variability; do not describe this as a guaranteed performance margin, a resolved startup60s regression, or a reason to weaken runtime checks. No source/test/fixture changes or new runs were performed for this review.

Bounded timing/hash summary: `/private/tmp/uat-windows-e4f8-default-review-summary.json`. Existing provenance verifiers: `/private/tmp/uat-windows-e4f8-default/verification.json` and `git-blob-verification.json`.
