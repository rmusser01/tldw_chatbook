# Linux Persona child-budget design review

**A bounded test-only redistribution is justified as a proposal, not implemented or accepted as a product fix.** The supplied approved metadata shows the150s later child being cut off while executing the actual second rollback and then installed-copy validation. It does not prove successful validation, completion, a particular blocking owner or a permanently hung operation. No remote reads, raw logs, new runs, source edits or deadline changes were performed for this review.

Read-only inputs: `/private/tmp/uat-linux-f104-persona-last-frames.json`, `/private/tmp/uat-linux-persona-approved-metadata2.jsonl`, `/private/tmp/uat-linux-persona-trace-numeric.py`, and current exact `f104d61f5bae0ad1c70793fe60497811a158814e` test source.

## Exact sequence and trace mapping

The positive test performs two synthetic native Persona seeds, source capture, target replacement, ordinary target reopen with saved-note verification/new note write, then `_LATER`. Only after `_LATER` returns its completion marker does it run restored ordinary reopen and a fresh restored capture, checking both markers.

`_LATER` itself contains initial current inventory; two equal rollback reviews and semantic-parent checks; first execution; the credential-review exception path with untouched Abort, fresh target/review and acknowledged second execution; committed-journal verification; verified new safety-copy/pack-file hashes/new-note readback; original Persona bytes/dependency restoration; final Complete inventory and completion marker. It is considerably more than a single publication call.

The diagnostic driver inserts five lines before the original embedded script and indents it under try/finally. Reconstructing that wrapper from the exact local source maps **wrapper line56 to original `_LATER` line51**, the second `recovery_copies.rollback(...)` inside the credential-review handler. It is not the final assertion or an arbitrary sleep. The first approved sample is in `_complete_move → journal._flush_records/_records`; the later sample has main waiting at `publication._validate_installed:2334` and a helper thread in `_validate_installed_copies → sqlite_validation → open_recovery_validation → private_sqlite_process._exchange`. This shows phase progress from actual publication into helper-backed installed validation. It does not prove the helper returned, journal committed, archive readback passed or the later children ran.

The86ad/10cc approved summaries contain parent `TimeoutExpired` at the positive test's `_child` call. They list setup/replacement/reopen/later log availability but no restored-reopen/capture artifacts. Availability and absence of captured exception types are not independent success receipts; the sequential source plus parent reaching `_LATER` explains which earlier child calls returned. No final positive completion is claimed.

## Existing independent bounds

Linux/non-Windows child ceilings in this positive test are:

| Stage | Current seconds |
|---|---:|
| Source seed + target seed |70 +70|
| Source capture |120|
| Replacement |150|
| Ordinary replaced reopen/new write |90|
| Later review/execution/validation/readback |150|
| Restored ordinary reopen |90|
| Restored capture |120|

The child maxima sum to **860s**, beneath the existing **900s** pytest marker, leaving an implicit40s envelope for test/fixture/controller overhead. This sum is budget arithmetic, not measured run duration. The approved metadata includes no elapsed per-stage measurements and cannot establish how much time was actually left at the150s child cutoff.

The diagnostic launcher supplies `--timeout=2400` and has a2400s outer process bound. That does not establish a2400s per-case acceptance allowance: the existing positive test has an explicit900s marker; pytest-timeout's item-setting resolution prefers the marker over the command default. Neither outer2400 limit should be used to expand this proposal. The observer wrapper leaves the child timeout expressions unchanged.

## Smallest concrete proposal for approval

Keep the Linux test's900s outer marker, both seed/capture/replacement/reopen ceilings, final90s/120s observations, all product waits and all assertions unchanged. At the positive test-body entry, establish one monotonic deadline for its **existing860s aggregate child envelope**. Just before launching Linux `_LATER`, give it only the unspent envelope **minus the210s already reserved for restored reopen and capture**:

`later_timeout = (test_body_start + 860) - monotonic_now - (90 + 120)`

Require this to be positive before spawning; an exhausted budget must fail immediately rather than start an unbounded process. This is an explicit redistribution of unused earlier-stage allowance to this multi-phase child, not a new independent ceiling or a restarted900s window. If setup used its whole500s allowance, `_LATER` still has at most its original150s. Faster earlier children can leave it more, while retaining the two final observations. Charge all intervening test-body work against that same clock. Leave Windows and macOS branches unchanged for this narrowly evidenced Linux issue.

The outer pytest900s timer remains authoritative, including fixture/setup/teardown time; starting a body clock cannot guarantee the40s allowance covers arbitrary fixture overhead. That is already a constraint of the current sum-of-child-ceilings design and should be disclosed, not bypassed. Do not remove the outer marker or inflate it to the launcher's2400s. Any eventual implementation must use a single monotonic origin and retain the post-child reserves, not calculate a fresh full budget at each phase.

Before implementation/native repeat, focused helper/call-sequence tests should verify: exact unchanged earlier/final/other-platform bounds; actual elapsed setup subtraction; a later timeout greater than150 only when earlier budget remains; exhausted budget spawns no child; monotonic advancement cannot replenish time; failure/TimeoutExpired and final completion assertions remain intact. Retain fixed metadata phase timings if already available so the repeat can distinguish publication, validation and postvalidation completion. No test may replace real recovery results or turn a timeout into success.

This proposal is evidence-supported because an inner observation cutoff prevents determining the outcome of a progressing multi-phase lifecycle while the accepted test already allocates a larger finite envelope. It still needs explicit approval and a real completed native run. A subsequent pass would close that attempt's lifecycle evidence; it would not identify why prior validation was slower, establish a performance improvement or fix a production deadline/guard. The separately retired Windows default-replacement budget proposal is not reopened, and the startup60s regression criterion is unrelated and unchanged.
