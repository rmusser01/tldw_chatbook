# K2 implementation and qualification report



Status: DONE_WITH_CONCERNS. Implemented and qualified within K2; Windows-only native qualification and sampled-RSS limits are stated below. The exact-path commit receipt is recorded below.

Base: 10d738ccc84a5ef0f16bf42dcfa54bce28f55b88; shared K1 types/store from 8bdb4be36df66222dc5f13ed6257ff86eb539b14.

Branch/worktree: codex/workspace-project-keyword-index at the assigned managed worktree.

ADR required: yes, existing ADR240 plus ADR101/102/174/181, as already recorded in Backlog36003; no new architecture decision invented by K2.



## Implementation and interfaces



Owned paths (only these seven are included in the implementation commit):

- tldw_chatbook/Tools/project_index_scan_protocol.py

- tldw_chatbook/Tools/project_index_scan_worker.py

- tldw_chatbook/Workspaces/project_index/scanner.py

- tldw_chatbook/Workspaces/project_index/models.py (ScanRunResult only)

- Tests/Tools/test_project_index_scan_worker.py

- Tests/Workspaces/test_project_index_scanner.py

- Tests/Performance/run_project_index_profile.py



`ProjectScanner.scan(authority, scope, job_id, frontier, emit, stop, *, admit_unit, policy)` consumes a lazy caller-owned iterator. It owns no durable claim/frontier database, service, tool, UI, semantic stack or Library state. K3 persists discovered directories and supplies subsequent incomplete-directory work. Restart re-enumerates a directory from its beginning. A directory completion bit is emitted only after actual exhaustion with unchanged retained metadata and ancestor links; mutation, inaccessible/disappeared or newly excluded directories emit incomplete receipts. False from emit stops further admission, returning paused. Scope/key mismatch refuses execution. Cancellation/authority loss returns interrupted; enumeration limits and incomplete receipts produce partial. Root/protocol failures return error. ScanRunResult validates its closed five-status domain.



`policy` is required and immutable per operation; include prefixes allow their ancestors for traversal, text suffixes filter reads, fixed processing exclusions skip node_modules/.venv/venv/__pycache__/dist/build/.git, archive extensions remain excluded, and strict UTF-8 plus NUL detection skips binary bodies. Authority contains the complete mandatory plus binding exclusion set from K1's registry adapter. The parent and worker accumulate a narrowing floor; removal never widens the running job. The worker additionally applies the existing mandatory sensitive filename rule. No config/runtime is imported into the worker.



`capture_file(authority, relative_path, *, stop)` uses private Capture only in the NEW protocol. The same descriptor capture helper supplies scan and live validation identity, SHA256, byte size, mtime and strict UTF-8. It checks ancestors, no-follow/reparse metadata, regular-file type, pre/open/post identity/size/mtime, same-API ctime, and at most1MiB+1 input bytes. It has no scan frontier or writer claim. The caller must already have selected a policy-eligible candidate and performed permission review; K4 owns that composition.



The closed protocol is v1, four-byte big-endian length plus JSON, <=2MiB total wire frame including its four-byte header (maximum payload is 2MiB minus four bytes). Exact request/response fields and primitive types, duplicate JSON keys, revisions/ack ordering, file-part ordinals/final flag, per-file digest/size and page/capture consistency are validated. Init is one trusted root/policy initialization. Directory requests double as acknowledgements for the preceding bounded unit. Started acknowledges the exact revision before project enumeration/read. Entries/File/DirectoryDone/Error carry bounded data. Text is split at16Ki Unicode code points (<=64KiB UTF-8) before JSON escaping, preserving Unicode/newline data fields. Root locators and bodies appear only on pipes.



The worker uses os.scandir iterators and<=256 raw discovered entries per request, so excluded-only directories still yield bounded admission units. Only one directory iterator is retained. It reads files sequentially (maximum1) and sends captures as they are produced. After review fix I1, retained page bodies are capped at2,031,616bytes (2MiB minus64KiB), reserving two1MiB slots plus a64KiB UTF-8 sender slice under the4MiB accounted ceiling. The initial commit omitted that slice and did not satisfy its stronger simultaneous-source-text accounting claim; see the correction below. Parent clears source parts before capture validation and releases each handoff before both continuation and next-directory admission. Instrumentation reports conservative source-text high-water accounting, total wire-frame bytes, and active reads; these are not claims that whole-process RSS is4MiB. The worker stops at250,000 eligible discovered entries with an explicit incomplete/partial receipt. It creates no cache/staging files; K1/K3 own the total1GiB persisted storage cap.



## Containment and retirement seam for K3



ProjectScanner._worker owns Popen, the existing _PopenProcessAdapter and ExecutorProcessTree. It calls tree.admit before writing Init, so the worker can only wait on stdin until containment exists. A watcher observes stop and the300-second per-unit timeout; cleanup terminates/proves the full tree, joins the watcher, closes stdin/stdout and checks process returncode. `retired`, `worker_pid`, `process_returncode`, `pipes_closed` are observations for the most recent operation, not durable restart authority and not proof for a separately started job. A cleanup failure raises worker_retirement_unproven rather than returning a successful page/result. K3 still owns native lifetime leases/inheritance and hard-parent-death recovery.



Parent-approved NEW-worker-only bootstrap launches sys._base_executable with -I -S and a fixed trusted application-code path derived from scanner.py's installed location. It loads no dependency-site/.pth files. Model/binding/root data never chooses that code path. The ordinary workspace executor, one-shot protocol/results and remote bundle are untouched. Actual scan/capture imports run successfully under this stdlib-only bootstrap; import closure tests forbid ML/UI/DB/config/runtime modules.



A real Windows probe found private venv python.exe PID112828 at3,551,232RSS bytes with actual interpreter child119324 at28,459,008RSS. Both exited after cleanup in that run. This demonstrated a launcher topology/profiling gap and pre-assignment risk; no escaping child was observed. k2-descendant-probe.log retains it. The added native-PID regression failed against the launcher (k2-launcher-red.log) and passed after the approved bootstrap. The final profile samples the native PID and any descendants, records all observed PIDs and checks their death. The native-PID test also verifies no launcher descendant, no project locator in argv, closed pipes, actual process absence, and stable native handle counts after three additional captures.



## TDD and fixes



All commands use worktree .venv/Scripts/python.exe with TEMP/TMP/TMPDIR set to .superpowers/sdd/2026-10-20-workspace-project-keyword-index/native-temp. New files were created using escalated shells for owner/private DACL inheritance. No installs, original-checkout/profile/Library changes, full-suite runs, push or merge.



Initial command: `python -m pytest Tests/Tools/test_project_index_scan_worker.py Tests/Workspaces/test_project_index_scanner.py -q`.

k2-red.log:8failed,1skipped,1warning in14.01s, all missing new scanner/protocol behavior (ModuleNotFoundError inside test calls, not unrelated fixture startup failure).



k2-green-1.log:3failed,5passed,1skipped. Real Windows stable files showed path-stat versus CRT-fstat ctime differences. k2-stat-probe.py/.log retain explicit same-file pre/open/post dev/inode/mode/size/mtime/ctime measurements and platform. Final evidence audit found the earlier probe used the subsequently corrected _state helper and its log was empty; the retained probe was corrected to print native fields directly and rerun, reproducing the cross-API ctime mismatch with stable per-API values. This evidence repair changed no product code. Full dev/inode/mode/size/mtime cross-API checks remain; ctime is compared path pre/post and descriptor open/post, not between APIs. Windows handles use OPEN_REPARSE_POINT and omit FILE_SHARE_DELETE for files and each ancestor. POSIX uses no-follow directory-relative descriptors and post-read lexical parent-child identity checks. Mutation during an actual descriptor read is independently rejected by regression, rather than only matching benign timestamps.



k2-green-2.log:1failed,7passed,1skipped in97.17s. The sole failure was a fixture counting10003 when conftest created an additional test_data directory inside tmp_path; the scanner truthfully discovered10004. The corpus now lives in its own child directory.



k2-handoff-red.log caught a prior page retained across a directory transition. The corrected implementation releases page/entries/captures before the next admission. The test's getrefcount was moved outside pytest assertion rewriting (which itself retains an extra reference). Final handoff test passed.



k2-wire-registry.log:6passed,1warning in20.17s. This includes real K1 registry admission while a separate sqlite writer process changes binding exclusions/revision between acknowledgements. Exclusion removal leaves the old job's floor narrowed. A preliminary private-profile setup attempt omitted its required request fixture; this was corrected and is not counted as product RED.



k2-final-covering.log (before native bootstrap): `python -m pytest Tests/Tools/test_project_index_scan_worker.py Tests/Workspaces/test_project_index_scanner.py Tests/Tools/test_workspace_root_pin.py -q`:37passed,3skipped,1warning in166.88s. This includes all existing root-pin tests. k2-native-bootstrap-green.log:8passed,1warning in3.52s for native PID/handles, saturated bodies and new wire tests after bootstrap change.



Final native-bootstrap command: `python -m pytest Tests/Tools/test_project_index_scan_worker.py Tests/Workspaces/test_project_index_scanner.py -q`. k2-final-native.log: **23passed,3skipped,1warning in149.43s**, exit0. After that completed, two narrowly added coverage cases were run independently: k2-fixed-protection.log **1passed,1warning in2.97s**, and k2-windows-acl.log **1passed,1warning in2.60s**. These cover mandatory filename refusal without scan-policy masking and actual Windows ACL list-denial/incomplete receipt. The only subsequent production change made the literal 2MiB frame limit include its four-byte header and changed frame metrics to total wire bytes. The exact-boundary test first failed (k2-total-frame-red.log: 1failed,1warning in1.73s). The entire wire file plus escaped-megabyte, saturated-body, shared capture/filter and native PID/retirement scanner cases then passed (k2-total-frame-green.log: 11passed,1warning in4.91s). The final 20k profile had already loaded the preceding scanner metric implementation: its recorded frame sizes are payload bytes, so the table below adds four bytes. The measured corpora are far below the new boundary. Parent approved no repeated corpus run solely for this arithmetic change. Published source is qualified by the later focused run; large-corpus timing/RSS receipts describe the preceding source with identical scan/capture/bootstrap behavior.



Inherited pytest-asyncio loop-scope and Pydantic Field metadata deprecations are unsuppressed. The three final-selection skips are stated below; they are not passing platform qualification.



Final static command: shared installed Ruff `check --no-cache` and `format --check --no-cache` on exactly the seven owned Python paths. k2-final-lint.log: All checks passed;7files already formatted. No install or rule exclusions. `git diff --check` passes; root-owned files are outside the staged set.



## Platform limits and honest scope



Actual qualification is Windows10 build19045, AMD64, Python3.12.10,12logical CPUs,77,172,310,016physical RAM bytes. Windows junction escape, changed root,1050-level path without Python recursion, disappeared directory I/O receipt, changed directory/restart, concurrent registry exclusion update, mutation during read, strict UTF-8/oversize/binary skip, escaped1MiB text, saturated bodies and real process/pipe/handle retirement are exercised.



Three explicit test skips on Windows: POSIX newline-on-filesystem filename (wire round-trip still runs), POSIX chmod unreadable-directory fixture, POSIX ancestor rename during capture (Windows no-delete-share handles prevent it). POSIX code exists but has not been executed on this host; no POSIX qualification claim. Windows ACL list denial is separately qualified: icacls denies only RD on a newly generated private fixture directory, the real worker returns an incomplete directory_io receipt with no entries, and finally removes only that new deny entry. No product permission checks or private ancestry protections were weakened. Missing-frontier I/O failure also has a real incomplete receipt. The synthetic profile uses generated private local corpora, not real user folders or SSH. RSS is sampled every10ms, not a proof of exact OS peak. Cancellation metrics begin when stop is set after a real process PID exists; sustained kernel/network filesystem stalls are not simulated.



Preliminary k2-profile.log is not final qualification: initial RSS missed the launcher child, and the harness used sqlite3's transaction context as though it closed the connection. Cleanup correctly exposed the open frontier handle. The final harness uses contextlib.closing and aggregates native/descendant RSS. The initial run also overlapped targeted tests; only the final native run is used for final resource numbers. It starts after the full targeted scanner selection. The additional2.97s/2.60s/4.91s focused cases overlapped parts of its corpus generation/scanning; timings describe observed machine load, not a controlled unloaded performance baseline.



Root owns task/plan/ADR/Architecture integration and final QA records; none are staged by K2. Suggested Architecture handoff: document scanner policy/admit_unit/raw-page boundary and the private Capture/native-bootstrap seams when K3/K4 wires the subsystem. No new user-facing settings or tools were added by this task.





## Final generated-corpus resource receipts



Command: `python Tests/Performance/run_project_index_profile.py --files 20000 --directories 20000 --wide-files 20000`, private native TEMP as above. Retained output: `k2-final-profile.log`; exit0 including TemporaryDirectory cleanup. All three returned complete with no issues. The harness used an on-disk SQLite frontier and fetched one path at a time; it did not hold a whole-tree Python path list.



| Corpus | Discovered / completed directories | Pages / maximum entries | Seconds / entries per second | Sampled parent / worker-tree / simultaneous aggregate RSS bytes | Accounted queued source-text bytes / reads | Maximum total wire frame bytes | Cancellation seconds |

|---|---:|---:|---:|---:|---:|---:|---:|

| 20,000 text files across100directories | 20,100 /101 |101 /200 |194.187 /103.51 |43,913,216 /26,771,456 /70,672,384 |2,103,641 /1 |35,046 |0.032 |

| 20,000 empty directories |20,000 /20,001 |20,079 /256 |93.453 /214.01 |44,756,992 /25,636,864 /70,361,088 |0 /0 |43,822 |0.031 |

| 20,000 files in one wide directory |20,000 /1 |79 /256 |192.469 /103.91 |43,433,984 /26,308,608 /69,730,304 |2,102,016 /1 |43,566 |0.031 |



RSS columns are maxima of10ms samples, hence lower bounds on actual OS peaks, not OS-recorded peak working set. Worker-tree RSS sums the actual native worker plus all descendants observed at each sample. Aggregate is the maximum simultaneous parent-plus-worker-tree sample, not the sum of separate maxima. It includes the Python parent/profile/SQLite frontier and worker; no aggregate sample exceeded71MiB. The sampling method cannot prove unobserved transient maxima; no exact RSS ceiling is claimed. Actual worker PID sets were [106552], [95300], [118316]; all observed processes were absent after return, with retired=true and pipes_closed=true. Parent handle before/after counts were321/313,312/313,312/313. Cancellation returned interrupted with retired=true, worker_gone=true and pipes_closed=true in every corpus. These are same-process actual native containment/retirement receipts, not mocked process claims or durable restart proof.



The small-file corpora do not saturate the text ceiling. The initial seven1MiB-file test observed the old accounting formula exactly4,194,304; independent review I1 showed that this was not proof of the stronger simultaneous-source-text bound. It is superseded by the deterministic actual-buffer observations below. The seven-file test still verifies complete capture with bounded handoffs and one active read after the fix. A separate page-release regression checks the previous handoff is released before the next directory admission. The escaped1MiB case round-trips through real framed pipes without overflowing the wire cap. These source-text bounds do not count metadata/JSON/Python-object overhead as text; whole-process overhead is included by the separate sampled RSS measurement.



Final self-review covered exact owned diff, strict framing, native-only bootstrap/import boundary, ancestor/file metadata checks, narrowing admission floors, bounded iterator/text lifetime and failure cleanup. No service, tool, UI, semantic or ordinary protocol files changed. Existing root-pin evidence remains retained; it was not rerun without a root-pin source change. Independent review is owned by the parent after this commit.



## Commit receipt



Implementation commit: db10df9a25f305ea225620a08361803654a52f40

Message: feat(workspaces): scan project trees in bounded resumable pages

Exact seven owned paths only; commit exited0. Parent-owned plan/ADR edits remain unstaged. Final report and generated receipts remain in the ignored task execution folder for the parent review ledger. No push/merge or hook bypass occurred.





## Review fix round1 - I1 source-buffer headroom



Fix base: db10df9a25f305ea225620a08361803654a52f40. Accepted I1 after independently reproducing it. The writer received at most2MiB of immutable capture payload before this fix; the defect was the stronger live-source-buffer accounting/qualification, not a demonstrated writer-payload overflow or RSS limit. M1 inherited warnings remain deferred and unsuppressed.



The smallest production fix reserves the worst64KiB UTF-8 sender slice in addition to two1MiB full scratch buffers. Shared constants derive PAGE_TEXT_BYTES=4,194,304-(2*1,048,576+65,536)=2,031,616. The worker uses the reduced budget before admitting a file. The parent independently refuses accumulated capture text above that page budget. Both metrics include SOURCE_BUFFER_HEADROOM=2,162,688; no cap was raised or reinterpreted as writer payload only. Four owned paths changed: protocol, worker, scanner and scanner tests.



The new parameterized regression uses the actual isolated contained worker, descriptor capture, framed pipes, page controller and retirement. A test-only wrapper pauses the worker immediately after flushing each final File frame. The parent trace samples the line after the actual join and before parts.clear, counts distinct live source objects by identity (existing captures, file parts, joined text and final-frame text), and adds the sender's actual full capture and distinct slice. It then releases the child through a test-only control byte. No production timing hooks or mock process proof were added. Worker/process/stderr cleanup runs in finally. ASCII and four-byte UTF-8 exercise16KiB and64KiB actual slices, with both full1MiB and near-boundary first files.



RED command: `python -m pytest Tests/Workspaces/test_project_index_scanner.py::test_live_source_buffers_include_sender_slice_at_parent_join -q`. k2-fix1-red.log:4failed,1warning in3.08s. Two full1MiB files produced4,210,688 simultaneous source bytes with ASCII and4,259,840 with four-byte UTF-8. A983,040byte first file produced4,145,152/4,194,304 actual bytes while the old metric reported4,128,768, independently catching the undercount even where the ceiling itself was not exceeded.



Initial GREEN command: `python -m pytest Tests/Workspaces/test_project_index_scanner.py::test_live_source_buffers_include_sender_slice_at_parent_join Tests/Workspaces/test_project_index_scanner.py::test_megabyte_captures_apply_backpressure_before_next_read -q -s`. k2-fix1-green.log:5passed,1warning in3.27s. Actual/accounted bytes: full-file ASCII3,162,112/3,211,264; full-file UTF-8 3,211,264/3,211,264; near-boundary ASCII4,145,152/4,194,304; near-boundary UTF-8 exactly4,194,304/4,194,304. Two full files now occupy separate pages. The literal4MiB assertions derive from independent live object measurements, not the production accounting formula.



Buffer-lifetime audit (F=1MiB, T=64KiB, P=2MiB-T):

- Final-flush overlap: prior parent captures R, current source S, parent parts plus newly joined text2S, sender full text S and slice<=T; R+S<=P, so R+3S+T<=P+2F+T=4MiB. Parent FileCapture UTF-8 digest validation replaces the already-cleared parts with its encoded source buffer, so it has the same bound rather than an additional full copy.

- Worker reading the next file: it starts only when previously sent page text<=P-F. Even if the parent is still joining/validating the prior file, that parent's source buffers are<=P. The next worker read/decode or FileCapture validation holds at most2F full source bytes, so their overlap<=P+2F=4MiB-T. The bounded os.read chunk is<=T while accumulating raw bytes; the terminal read is empty before successful decode, and raw is deleted before FileCapture validation. Invalid/oversized input returns without the text/decode stage.

- Part slicing/validation: forming the next slice may briefly retain old and new slices, and UTF-8 validation encodes a source part<=T. These precede that part's flush; the receiver cannot yet hold the complete current file plus its joined copy. The already-held parent parts plus current worker body and bounded slices remain within the larger full-body reserve above. The old parent frame's final text aliases a part and is counted once; parts are cleared before full-file validation. No previous page survives the next admission.

- The accounting includes raw source reads, decoded bodies, source slices, reconstruction and UTF-8 digest-validation copies. It excludes serialized JSON/wire buffers, metadata, Python object/allocator overhead and RSS, just as I1's measurement does. Those are distinct from extracted source-text bytes; RSS remains separately sampled with its stated limitations.



The three20k resource receipts above predate this fix as well as the noted total-frame metric adjustment. Their queued-byte metrics are historical and are not evidence of the corrected live-source ceiling. Their tiny files never approach even the reduced admission threshold (largest retained source total is under7KiB/page), so page counts, file reads and I/O topology are unchanged. No corpus rerun was requested/needed for this constant/headroom change; no new timing or RSS numbers are claimed. The corrected bound is qualified by actual-buffer overlap tests and the complete two-file targeted selection. Large-file page shape does change and is covered by seven1MiB files, escaped1MiB input and deterministic overlap cases.


Final covering command: `python -m pytest Tests/Tools/test_project_index_scan_worker.py Tests/Workspaces/test_project_index_scanner.py -q`. k2-fix1-covering.log:29passed,3skipped,1failed,1warning in126.78s. The sole failure was test_page_handoff_does_not_retain_previous_directory_text indexing captures[0] on the now-necessary empty exhaustion receipt after a1MiB capture page. This is the expected changed page shape, not a missing body: the fixture now accepts empty complete receipts, asserts both file bodies were received, and checks the last body has no scanner references before every subsequent admission (continuation and next directory). No production code changed after this covering run.

Final affected-node command: `python -m pytest Tests/Workspaces/test_project_index_scanner.py::test_page_handoff_does_not_retain_previous_directory_text -q`. k2-fix1-handoff-green.log:1passed,1warning in1.43s. Thus all30 runnable cases in the targeted selection have passing final evidence, with three explicit Windows skips; this is a combined receipt, not a falsely claimed single all-green full-file run. Passing unrelated cases were not repeated after a test-only correction. Scoped Ruff check and format --check on the four changed Python paths passed (4files already formatted), and git diff --check passed with only a parent-owned ADR line-ending notice. Final self-review rechecked the admission arithmetic, both-side enforcement, lifetime proof, actual-buffer instrumentation and strict task ownership. All inherited warnings remain visible.


Fix-round1 commit: 6cbee38fe0c283ff50b06e424c7bcd6d052c0874 — fix(workspaces): reserve live scan sender buffer headroom. Exact four changed owned paths only; parent task/plan/ADR metadata remains unstaged. No push/merge, install, full-suite run or hook bypass.



Retention note: named command logs and native probes are retained as keyed text plus original-byte hashes in [k2-evidence.json](k2-evidence.json). Historical runs are explicitly distinguished above from corrected source qualification.
