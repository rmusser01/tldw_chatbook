# Windows 4becc sibling ACL failure review

Read-only review of run 34817971803, revision 4becc2872cd020ef8f2b28669d2e51cc9c7c1f62. Input verification: /private/tmp/uat-windows-4becc-support/verification.json (234 artifact hashes and four installed receipts verified by parent).

Exact failed nodes, all executed with no skips:
- Tests/Backup_Recovery/test_bound_config_siblings.py::test_sibling_guard_rechecks_exact_config_anchor_and_foreign_owner[unsafe_parent-emoji]
- Tests/Backup_Recovery/test_bound_config_siblings.py::test_sibling_guard_rechecks_exact_config_anchor_and_foreign_owner[unsafe_parent-runtime]
- Tests/Backup_Recovery/test_bound_config_siblings.py::test_sibling_guard_rechecks_exact_config_anchor_and_foreign_owner[unsafe_parent-sidebar]

The other 16 cases in this file passed: all three actual absent/read/write owner APIs, twelve other guard cases, and the runtime wrong-parent case. Relevant test/helper/platform source has no diff from 4becc.

Causal limit: the artifact has no child logs or stack/phase/error metadata for any of these three cases. _run in test_home_citation_retirement.py captures child stdout/stderr in memory and puts it only in a final pytest assertion. The outer product termination (return code 124, 80 minutes) prevented final failure tracebacks and product JUnit. pytest-output.log lines 404–406 preserve only FAILED outcomes. Therefore there is no evidence identifying child timeout versus ACL setup/restore failure, native posture assertion, actual guard failure, or final byte/identity assertion. Do not attribute a product defect or change any budget from these outcomes alone.

Source checks: platform_files.os selects WindowsOS; stat/fstat query native owner SID/DACL and conservatively project public read grants into 044. raw._check rechecks pinned companion parent privacy on each IO. The fixture grants actual Everyone:R, expects this exposure before and after refusal, restores the saved DACL, compares the saved/re-saved DACL bytes and parent identity/posture, then verifies config/control file bytes/identities. Existing native opens use share-all, so no deterministic ACL-sharing blocker was found by source inspection. This source trace is not proof that the native fixture completed any particular phase.

Smallest next measurement: select only the exact three nodes above on native Windows (or this single file with -k unsafe_parent), keeping each existing 35-second child ceiling and all assertions. Use the existing source/package provenance workflow; no new job or broad support rerun is needed to measure these failures. Durable bounded test-local *.json.log metadata should record fixed phase names: child entry, imports/binding ready, scope entered, original posture read, ACL saved, grant completed, public posture observed, guard refusal, ACL restored, restoration verified, scope retired, final preservation verified. Retain only monotonic elapsed time, boolean posture/identity/equality facts, fixed command label and return code, error class/errno/winerror and traceback module/line; no ACL contents, config values, child locals or arbitrary exception text. A parent timeout record must be written if the original 35-second subprocess deadline expires. Metadata writes must be best-effort and preserve original exception and ACL cleanup behavior. The existing collector already retains *.json.log via its *.log pattern.

No source edits, reruns, remote mutations or deadline changes performed by this reviewer. The five GUI failures are owned by the separate reviewer and are not conflated with these cases.
