# Exact a5237 Windows startup audit

Read-only review of local CI log for run34936472021/job104275363447, checkout `a5237449584d5b6d0090c15adabb4b12846a4aaa`. No new tests/apps, network access, or repository edits.

**Acceptance remains failed:**31non-UI tests pass in9.23s; all four original UI cases hit the existing60s pytest timeout. These are claim-authority/recompose, external-copy geometry, cpp keyboard, and llamafile keyboard. The primary step exits1. The failure-only dev/candidate diagnostic also exits1; continue-on-error does not turn its candidate failures into passes.

| Diagnostic keyboard case | Dev session / call | Candidate result at last sample |
|---|---|---|
| llama.cpp |47.32s /38.73s, passed with2warnings |Timeout; test line1404, `await pilot.press("enter")` |
| llamafile |39.37s /36.49s, passed with2warnings |Timeout; test line1367 inside `_mount_models:116`, `await pilot.pause()` |

Dev is the pinned `4631b60f8dd9623fc55bf16f4a37e29fcb1240c7`. The identical observer runs both sources with `--no-profile`; cProfile is disabled and largest_total/largest_self are empty. Extracted22 monitoring records (5dev/6candidate/5dev/6candidate), all source_matches=true, no observer/config errors, and0dropped-active maximum. Git observer bytes are identical to63756, SHA256 `973e6d8bc973b29ac52f62a8da40328bbf45b33cbd1462ff632fc92a16207c48`. This is exact-code monitoring evidence with observation overhead; it is not an uninstrumented speed benchmark.

The cpp candidate's mount actually returned:55.376927s wall/39.015625s threadCPU, versus dev12.377183s/8.03125s. It completed four observed settle calls and reached Enter on source mode. The last sample60.047s has the main thread in the event loop's IO poll; no `test_focus` helper was entered. This establishes progress beyond mounting, not completed keyboard behavior. Llamafile has no mount completion at60.032s, while dev mount returned10.750267s/5.5625s; the candidate's current sampled main-thread path is Windows handle info→open_handle→parent/stat→bootstrap control records→config companion guard/raw scope. That stack is a sample, not a cumulative native-cost attribution.

Candidate config_operation counts at final samples are759/798. The slowest *individual observed completed* config operations are1.413001s/0.765625threadCPU (cpp) and1.718771s/0.78125threadCPU (file), each on another thread. Their acquisition/body-and-release splits are0.586634/0.826368 and0.591765/1.127005seconds; acquisition includes validation/admission and must not be called pure mutex contention. Counts may be nested; these durations cannot be added or extrapolated into total backup overhead. This no-cProfile observer does not provide cumulative GetSecurityInfo/native-open counts or whole-process CPU. No native-call total should be invented from sampled frames.

Against63756, whose two candidate comparisons were both pending at `_mount_models:114` (push_screen), a5237's cpp comparison reaches Enter and the file comparison reaches mount's post-push pause. Both still exceed60s. Different observed progress/counts do not independently prove that the tray correction caused a timing improvement, nor that any single sampled native operation causes the remaining delay. No further product optimization is justified by this audit alone.

Backup native-close run34936466840 is a separate still-active qualification at assignment time; this startup log cannot qualify it. Compact records, code/log hashes, exact counts and bounded CPU values are retained in `/private/tmp/uat-a5237-startup-independent-summary.json`. No broader PR-success claim.
