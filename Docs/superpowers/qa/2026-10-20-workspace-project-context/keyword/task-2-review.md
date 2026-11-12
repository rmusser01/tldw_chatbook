# K2 task review: 10d738ccc84a5ef0f16bf42dcfa54bce28f55b88..db10df9a25f305ea225620a08361803654a52f40

## Spec compliance

Issues found in resource-bound qualification; functional K2 scope otherwise matches the brief and its post-dispatch rulings. All seven requested paths have corresponding hunks. Literal total-wire frame sizing, bounded raw enumeration, sequential reads, required immutable policy, live admission revisions, narrowing floors, private Capture, native isolated bootstrap, and physical retirement seams are implemented. No K3 service/restart ownership or K4 permission integration was incorrectly added.

The <=4 MiB writer-handoff payload requirement is satisfied by the current <=2 MiB page capture payload and synchronous release before next admission. The stronger claim that all simultaneous source-text buffers are conservatively accounted within 4 MiB is false (Important finding I1). Do not conflate this with a measured writer queue overflow or an RSS limit: neither was established.

## Strengths

- Tools/project_index_scan_protocol.py:21 and :244, Tests/Tools/test_project_index_scan_worker.py:86: the header-inclusive ceiling rejects oversized declared payloads before allocation; an exact-boundary test covers the clarification. Exact fields, duplicate keys, primitive types and part ordering are checked at both ends.
- Tools/project_index_scan_worker.py:136 and :198: scan and Capture share descriptor-based capture, retained ancestors, no-follow/reparse checks, same-API ctime checks and pre/open/post identity/size/mtime checks. Windows mutation and junction tests exercise actual behavior; POSIX rename handling is present but unqualified here.
- Workspaces/project_index/scanner.py:120: native interpreter containment precedes Init. The finally path terminates/proves the tree, joins the watcher, closes pipes, and refuses successful retirement when proof fails. Tests/Workspaces/test_project_index_scanner.py:554 checks real PID topology, absence, pipes and repeat-capture handle counts.
- Workspaces/project_index/scanner.py:358: prior page/capture references are dropped before both continuation and next-directory admission. Iterator scheduling stays caller-owned; no tree-sized path list is introduced. Directory completion depends on exhaustion and stable retained metadata/links.
- Tests/Workspaces/test_project_index_scanner.py:225 and :435 onward: coverage includes real descriptor mutation, registry changes from another process, narrowing after exclusion removal, incomplete I/O receipts and explicit restart from the beginning.

## Issues

### Important / P2 — I1: live source-text accounting omits the sender slice

Locations: tldw_chatbook/Tools/project_index_scan_worker.py:294, :422, :515; tldw_chatbook/Workspaces/project_index/scanner.py:205, :209, :214; Tests/Workspaces/test_project_index_scanner.py:413.

With two 1 MiB ASCII files in a page, a permitted scheduling overlap has the parent retaining the first capture (1,048,576 bytes), the second file's parts (1,048,576), and the newly joined second text (1,048,576), while the worker still retains the second full capture (1,048,576) and its final distinct source-text slice (16,384). Total: 4,210,688 bytes, exceeding 4,194,304 by 16,384. The sender can be preempted immediately after flushing the final File frame; nothing acknowledges or releases its full capture/slice before the parent joins. UTF-8 source slices can be larger than the ASCII slice used in this probe.

This is source text only; the calculation excludes JSON/wire buffers, metadata, Python object overhead and RSS. The immutable page eventually delivered to emit contains only two file bodies (2 MiB). Thus this proves the report's conservative simultaneous source-text claim and the associated metric are inaccurate, not that the writer receives >4 MiB of payload. The saturated test merely asserts the same accounting formula and cannot detect this overlap.

Reserve explicit headroom for the sender slice (and audit other live source buffers), then add a deterministic overlap regression observing actual buffer lifetimes. Alternatively, the controller must explicitly narrow the contract to writer payload alone and rename/document the metric so it cannot be used as proof of all live source-text bounds. The current report should not claim the stronger ceiling.

### Minor / P3 — M1: inherited verification noise remains

Evidence: .superpowers/sdd/2026-10-20-workspace-project-keyword-index/k2-total-frame-green.log:1 and k2-final-native.log:1. Pytest-asyncio loop-scope and Pydantic Field metadata deprecations are visible. These are disclosed, inherited warnings rather than scanner failures; do not suppress them or expand K2 merely to obtain a clean transcript. Track their existing owner and preserve the warning-qualified receipt.

## Focused review checks and conclusions

Read the complete packaged diff in four bounded sections, without rereading changed files or git commands. Initial combined brief/report/tool output truncated; recovered the missing brief rulings and reviewed the full diff from its package. Consulted unchanged portions of models.py only for FileCapture re-encoding, ScanPage lifetime/consistency and authority contracts. Consulted ExecutorProcessTree and _PopenProcessAdapter/workspace_worker_environment only for admission-failure, concurrent retirement, returncode and native-bootstrap ownership risks. No other codebase crawl.

Named focused check: retained_source_at_second_capture_join. Used an in-memory synthetic two-file producer in a native -I -S -B child, actual protocol pipes, the production _send_capture and parent _unit, and a deterministic hold immediately after the second final File frame flush. At the trace event immediately after join and before parts.clear, observed parent_source=3145728, worker_source=1048576, worker_slice=16384, simultaneous_source_total=4210688, limit=4194304; scanner.max_queued_text_bytes still reported4194304. The child and both data pipes/stderr were retired in finally. This probe reads no project files and writes no fixture/product files. Its first setup attempt traced the line before join and raised KeyError('text'); cleanup ran, the trace point was corrected once, and the second attempt produced the stated result. No reported suites were rerun.

Capture-response conclusion: one public capture_file operation performs one private Capture and returns only after the worker context retires. Worker-local capture retention until the next request/process exit does not by itself establish a separate public-API leak or a second blocking finding. Parent reconstructs and validates digest/size and matches the returned candidate path. K4 still must supply current authority and permission fences; K2 does not authorize disclosure.

Directory-race conclusion: each capture independently validates its retained lexical ancestor chain and file identity; the directory iterator additionally checks metadata and retained links before certifying completion. An interrupted/mutated/inaccessible directory yields an incomplete receipt. Root identity is checked/pinned, with checks before admitted units; later publication/disclosure must still be fenced by K3/K4. I found no additional K2 defect in this division of responsibility. POSIX post-read rename protection remains source-inspected, not natively qualified.

## Evidence and cannot-verify gates

Read retained k2-final-native.log (23 passed,3 skipped,1 warning), k2-total-frame-green.log (11 passed,1 warning), and the final three-corpus profile receipts; these substantiate the report's stated results without rerunning them. Profile receipts show native single-PID observations, absent workers after return, closed pipes, and ~31-32ms cancellation. Actual Windows native qualification is supported. No POSIX execution was possible on this host; the three explicit skips are not passes. The separate actual Windows ACL fixture is visible in the diff and its passing result is reported; its log was not independently replayed.

The final 20k profiles ran the immediately preceding frame-accounting source; reported raw frame maxima are payload bytes, and adding four produces total-wire values. Later focused tests qualify the header arithmetic. Profile timing/RSS is not evidence for a byte-for-byte identical final source run, and this approved source-timing distinction must remain in final notes. RSS is a 10ms sample, not an OS peak guarantee; generated local corpora and briefly overlapping focused checks do not establish unloaded, remote or sustained kernel-stall performance. The focused overlap probe likewise measures source buffers, not total RSS.

Cannot verify from this K2 diff: persisted cumulative entry/storage enforcement across restart, owner registration/production opens, hard-parent-death recovery and lifetime leases, disable/clear ordering across services, exact-scope Ready publication, effective-run ranking exclusions, permission-reviewed fresh disclosure, and search output/run budgets. These are K1/K3/K4 integration gates, not missing K2 scope. Root-owned Backlog/ADR/architecture updates are excluded from this committed diff and remain controller work.

## Assessment

Task quality: Needs fixes / corrected qualification for I1 before trusting the advertised live-buffer ceiling. The bounded writer handoff, closed wire, scanner authority checks and Windows native cleanup design are otherwise coherent and well covered for this foundation. No Critical finding and no additional Important finding was established.
