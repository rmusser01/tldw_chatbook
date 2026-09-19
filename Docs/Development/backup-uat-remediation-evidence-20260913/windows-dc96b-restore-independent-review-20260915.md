# Exactdc96 Windows restore acceptance audit

Read-only local artifact/source review of run34942614500/job104294555043, exactdc96b849dc4f05e64843572016c63f645753b150. No apps/tests, remote actions, or repository edits.

**SUCCESS:59native +16selected product cases pass, with0failures/errors/skips/missing terminal outcomes.** Native suite1.264s; product suite367.634s. Rehashed all22 indexed artifacts. Source receipt identifies clean exact Git and private tracked-head copy,16,708files. Eight relevant test/driver/helper digests independently match exact Git, allowing only Git CRLF conversion. One installed receipt contains2,867files;2,475package Python files match source and Git (2,443CRLF/32exact), zero mismatches. WheelSHA2561053772fc0b6f7c6dbfd59a38e0a69e27a54653e4b3af291c92af2093a89bf2c. This verifies retained receipts rather than a fresh remote installation inspection.

| Selected behavior | Evidence |
|---|---|
|Installed plaintext F9 capture→isolated restore→Open|Passed220.386s|
|Native guidance/current config save/readback|Passed30.100s; all7emitted phases|
|Native display pair/fresh later send guard|Passed36.243s; all7emitted phases|
|Tray layout covered/transparent/hidden/visible/unmount|All5passed|
|Bounded layout wrapped/native across covered/transparent/hidden/unmount|All8passed; formerly failing covered-native6.257s|

The F9 test executes real native console controls and the installed package, verified in parent and opened child. It requires verified complete/coherent plaintext archive with one profile, excluded credentials/redacted config history, unchanged source config, native pause released, validated isolated restoration, and subsequent successful Open receipt. The opened child asserts its installed package origin and exact selected config/core paths, actual mounted/UI-ready status, restored pre-capture note content, absence of the post-capture note, and absent synthetic secret. The original profile still reads both its original and resumed-written notes; no blocked network attempts are allowed. The retained console receipt says completed and has actual console-capable stdin/stdout. The Open test intercepts the subprocess invocation only to execute its bounded verification child using the real launch environment/selected profile arguments and normal CLI/app checks; it is not a byte-for-byte test of an unmodified interactive child executable command. No durability/launch claims beyond those assertions are added.

Both actual guidance-phases.log files contain import_begin,import_complete,construct_begin,construct_complete,context_ready,assertions_complete,shutdown_complete. These are file emissions, not embedded command-source matches. The reviewed fixture uses a real TldwCli application context with an unmounted ChatScreen and native config/session/readiness calls. Thus both proof bodies and cleanup completed; this is **not** a full-app mounted startup benchmark. Covered-native also completes the finite2s quiescence observation, unchanged geometry/scroll/hint/focus/allocator assertions, earlier covered callback stability, and final .04s callback-count stability.

Independent Git comparison confirms product/native/Packaging/pyproject sources atdc96 are identical to2031d; intervening changes are reviewed test fixtures, CI and documentation. This closes the specifically selected Windows guidance and covered-layout gaps with current native evidence while retaining the positive installed plaintext restore result. It does not run encrypted/credentials/replacement/later rollback permutations or all backup tests. The run uses exact feature-source private copy, not a synthetic PR merge checkout. Do not conflate it with the separately verified Voice genuine-merge job.

The separate Models fixture proposal was rejected by automatic approval review as outside the original scope and remains unapplied/pending user scope approval. This audit adds no approval for it and makes no new startup or whole-PR acceptance claim. Exact case identities/times, source/hash receipts and emitted phases are recorded in `/private/tmp/uat-dc96b-windows-restore-independent-summary.json`.
