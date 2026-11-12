# K2 fix-round1 scoped re-review

Base: db10df9a25f305ea225620a08361803654a52f40
Head: 6cbee38fe0c283ff50b06e424c7bcd6d052c0874
Scope: I1 from task-2-review.md and new breakage introduced by this fix only. M1 remains a deferred minor; this is not a task-wide or merge review.

## Per-finding verdict

I1: ADDRESSED.

At tldw_chatbook/Tools/project_index_scan_protocol.py:30, SOURCE_BUFFER_HEADROOM now reserves two full 1 MiB scratch bodies plus the worst-case 64 KiB UTF-8 sender slice. PAGE_TEXT_BYTES at :31 is correspondingly 2,031,616 bytes. The existing worker admission check consumes that reduced page constant before starting another file. The parent independently enforces retained_bytes + part_bytes <= PAGE_TEXT_BYTES at tldw_chatbook/Workspaces/project_index/scanner.py:206, and its metric adds the same conservative reserve at :211. Directory and private Capture receipts include that headroom at tldw_chatbook/Tools/project_index_scan_worker.py:520 and :373. No ceiling was increased or narrowed to writer payload alone.

For the demonstrated final-flush overlap, let R be prior parent captures, S the current source size, F=1 MiB and T=64 KiB. The parent can retain R+2S while the worker retains S+T. Since R+S <= P=2MiB-T and S<=F, R+3S+T <= P+2F+T=4MiB. Parent UTF-8 digest-validation replaces cleared parts with the encoded source copy; it does not add a third full current-file parent copy. The worker only starts another file while previously sent text <=P-F, leaving room for parent reconstruction of the preceding file plus worker raw/decode or validation buffers. The report's distinction between extracted source buffers and serialized wire/object/RSS overhead is retained.

Tests/Workspaces/test_project_index_scanner.py:632 adds deterministic actual-buffer evidence, rather than another assertion of the production formula. It keeps actual descriptor capture, real framed pipes, admission and containment; only scheduling is held after the final File flush. The parent samples distinct source objects after join and before parts.clear. At :747, observed parent plus worker sizes must fit both the literal 4 MiB ceiling and the reported metric. ASCII/four-byte UTF-8 and full/near-boundary first files cover the missing 16/64 KiB slice and the reduced page boundary. The test restores tracing and retires its stderr/process resources in finally.

The retained GREEN observations support the fix: full-file ASCII actual/accounted 3,162,112/3,211,264; full-file UTF-8 3,211,264/3,211,264; near-boundary ASCII 4,145,152/4,194,304; near-boundary UTF-8 4,194,304/4,194,304. Unlike the old saturated-body assertion, these observations directly exercise the lifetime overlap I1 identified.

## New breakage assessment

Critical: none found.
Important: none found.
Minor: none newly introduced requiring a separate finding.

The lower budget changes large-file page shape: a full 1 MiB file ends its page before an exhaustion check, so an empty complete receipt may follow. The sole covering-run failure was an old fixture indexing captures[0] on that valid receipt. Tests/Workspaces/test_project_index_scanner.py:184 now accepts only complete empty receipts in this fixture; :191 checks last-capture release before every subsequent admission, and the added body-count assertion still requires both files. This fixes the fixture assumption without weakening the resource-lifetime check or changing production code after the covering run.

## Verification receipts reviewed

Read the supplied fix package once (commits/stat/-U10 diff) and the appended fix report. No git rederivation, changed-file crawl, suite rerun, new focused execution, profile operation, or product/test mutation was performed.

Commands exist in task-2-report.md and the corresponding retained output files exist and were read:

- python -m pytest Tests/Workspaces/test_project_index_scanner.py::test_live_source_buffers_include_sender_slice_at_parent_join Tests/Workspaces/test_project_index_scanner.py::test_megabyte_captures_apply_backpressure_before_next_read -q -s; k2-fix1-green.log: 5 passed,1 warning in3.27s, with the four actual/accounted observations above.
- python -m pytest Tests/Tools/test_project_index_scan_worker.py Tests/Workspaces/test_project_index_scanner.py -q; k2-fix1-covering.log: 29 passed,3 skipped,1 failed,1 warning in126.78s. Failure is the empty-receipt fixture assumption described above.
- python -m pytest Tests/Workspaces/test_project_index_scanner.py::test_page_handoff_does_not_retain_previous_directory_text -q; k2-fix1-handoff-green.log: 1 passed,1 warning in1.43s after the fixture-only correction.

This supports 30 runnable cases with passing final evidence across the two final receipts, not a claim of one all-green covering invocation. The report correctly says no production change followed the covering run. The reported scoped Ruff/format/diff checks were not independently rerun. The controller reports a report-heading encoding normalization only; no claim or code changed.

## Out-of-scope observations and limits

M1 inherited warnings remain visible and deferred. Windows-only native qualification and the three POSIX fixture skips remain unchanged; this fix is not POSIX qualification. The previous 20k RSS/timing receipts predate the fix and their queued-byte counters are historical, not proof of the repaired source ceiling. The appended report says so explicitly. Tiny-file corpus page shapes remain below the reduced threshold; large-file page changes are covered by the new overlap and existing seven-file/escaped-file cases. No new timing/RSS claim is made.

All K1/K3/K4 integration gates from the original review remain out of this scoped re-review. The simultaneous extracted-source bound is distinct from immutable writer payload and whole-process RSS.

At controller request, preserved the original already-run baseline synthetic probe as k2-review-overlap-probe.py and sent its path to the implementer. It is marked baseline db10df9a25-only because its trace line is fixed; it was not rerun. The new source-derived real-worker regression supersedes that historical probe.

## Final verdicts

Spec compliance for this fix: APPROVED; I1 ADDRESSED while retaining the stronger simultaneous-source ceiling.
Code quality for this fix: APPROVED; no new Critical or Important issue found in the four-file fix diff.
