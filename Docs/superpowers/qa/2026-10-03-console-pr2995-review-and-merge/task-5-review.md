### Spec Compliance

- ✅ Spec compliant for the owned test repair and diagnostic work: reviewed BASE c0b251a71422536a3da2be421dd60f79e2cb230a → HEAD e7cc5337617781e9a4b08b9e324297b537fcad07, one owned test file, 159 insertions/23 deletions. The local pre-mount factory fixes the diagnosed harness competition without changing production ownership or startup budgets (Tests/UI/test_console_runtime_ownership.py:1994).
- ✅ The real-startup/public-navigation control remains independent of the manual factory; it awaits the retained startup task, verifies one Console and reconciled current generation, retained mounted screen, shared runtime/store/controller/bridge and idle/active poll behavior (Tests/UI/test_console_runtime_ownership.py:2031).
- ✅ Original successor, late-old-screen refusal, wake polling and dual-fence receipt/composer requirements remain exercised. The self-review AST inventory preserves every original assertion/decorator across 70 original functions; the patch changes setup rather than those expectations (task-5-evidence/self-review.json:1; Tests/UI/test_console_runtime_ownership.py:2333, :2385, :3412, :3543).
- ⚠️ The fourth historical CI failure is independently unproved. Initial local diagnosis already passes native unchanged, so neither the three lifecycle cause nor local GREEN proves its historical readiness timeout repaired. Current-head Ubuntu CI must qualify it before merge (pr-initial-fastlane-job.log:1006, :1026; task-5-evidence/four-original-diag.log:569).
- ⚠️ Final latest-dev integration, queue/dispatch recovery owners, final startup/import ratchets and external merge gates remain controller-owned and outside this source gate (task-5-brief.md:22).

### Strengths

- The repair controls the causal boundary directly: claim startup before run_test can schedule its deferred callback. Its regression awaits that exact task before manual mount and explicitly invokes a late startup callback, then asserts the exact stack, view and generation; no added timing grace can hide a competing startup Console (Tests/UI/test_console_runtime_ownership.py:1994–2027).
- Causal evidence distinguishes unsupported duplicate setup from public navigation: the forced RED has default/Chat/Chat, while the real-startup control passes before the repair (task-5-evidence/causal-red.log:21; task-5-evidence/startup-red-control.log:17, :280). BASE trace records initial Console generation 1 before manual generation 2, retained initial Console on return and reconciliation=false with no current view (task-5-evidence/four-diag.jsonl:27, :35, :219, :217).
- Same-target navigation still requires a distinct successor, current runtime view, live hook and unpoisoned visit; outgoing stale claim refusals remain visible in GREEN diagnostics rather than being suppressed (Tests/UI/test_console_runtime_ownership.py:2348; task-5-evidence/green-diag.jsonl:349, :403).
- Both manual and actual-startup controls use public navigation and the production active-delivery poll admission path. The active wake is a deliberate _WakeDelivery precondition; the tests verify timer admission/reconciliation, not provider execution or transcript delivery end to end (Tests/UI/test_console_runtime_ownership.py:2087, :2410).
- The readiness diagnostic races the exact start task against the Event with the original five-second bound, reports early outcome or pending frames, and cancels/joins its waiter. The unchanged outer finally releases readiness and drains start/controller resources. Consumption, painted composer, durable receipt, exactly one user message/provider call/generation and no lingering claim remain asserted (Tests/UI/test_console_runtime_ownership.py:3380–3411, :3463–3570).

### Issues

#### Critical (Must Fix)

- None found in the owned patch.

#### Important (Should Fix)

- None found in the owned patch. The native historical timeout is a required external qualification, not evidence for a source defect or permission to increase timeouts.

#### Minor (Nice to Have)

- Inherited test-output noise remains: four invalid-escape SyntaxWarnings from the construction census and FD growth of 523 (start 14/end 537; limit 200). This prevents a pristine-output claim; the FD cause is not established by this task. Preserve the warnings and investigate under a separate cleanup task if needed; no suppression belongs in this repair (task-5-evidence/batch.log:73, :77, :80; Tests/conftest.py:609).
- The disposable recorder reads the wrong retirement attribute, _console_view_retired, so its null retirement field cannot establish retired-state behavior. The report correctly discloses this; stack/current-generation/reconciliation evidence and the causal regression independently support the repair. Correct that field before reusing this diagnostic script (task-5-evidence/ownership_diag.py:22; task-5-report.md:38).

### Checked Evidence

- Read review-c0b251a714..e7cc533761.diff once in full; SHA256 3a325e52ffbb5afc4253ae44b1860e5ca21d2aee54985a2ca2c75a2b19fc89f9. No Git commands, suite reruns, helpers or subagents were dispatched for this review.
- Checked original CI tree receipt: 7a15e3e9b4998dba8d8726bbdb722ff60a3df83d and merge 112298c452082b71c9a21572e7982f5bfd6bc9c9 share tree 9d2ff375673c6b46af8a85babf0d4960f68a271d (ci-initial-merge-tree-receipt.json:1); read immutable CI failures and task triage.
- Verified initial four-node order and 3 failed/1 passed, forced causal RED, public-startup RED/control, and six-node GREEN (task-5-evidence/four-original-diag.log:17, :569; causal-red.log:21, :306; startup-red-control.log:280; six-green-diag.log:48). The initial private-profile launch failure ran no test body and is reported separately.
- Verified startup identities/stack, rejected stale attach/current claim and reconciliation outcomes for all three lifecycle nodes in four-diag.jsonl and green-diag.jsonl. Native original start Task:12946b580, coordinator :127a0be00, controller :128ed3710 and physical task :12946be80 identify attempt d30f63f017db4c6d93e212a3a656b06f with successful preparation/durable commit/started outcome; GREEN has independent identified attempt 149cb6bd246942658d6594b4de7c60a5 (four-diag.jsonl:955, :960, :1111, :1112; green-diag.jsonl:829, :844, :989, :990). This does not identify the CI attempt cancelled after 63,502 ms.
- Parsed complete batch.xml: 156 collected, 155 pass, one inherited XFAIL, zero failures/errors. Per-file: ownership 80 pass + 1 XFAIL; viewless hooks 18; skill install runtime 6; chat creation integration 51. All six native consumption variants pass. batch.log reports 343.84 s and five warnings; native unchanged call is 14.75 s (JUnit total case time is 15.618 s, including fixtures) (batch.log:10, :89, :113; batch.xml:1).
- Inspected batch-receipt.json and runner: exact four-file argv, unique private profile/basetemp, no diagnostic plugin, exit 0, 353.424 s wall duration; all four tests and eight production fingerprints match before/after. The precommit run correctly records HEAD=BASE with dirty final source, then commit-receipt.json binds qualified source to HEAD e7cc533761 (batch-receipt.json:1; commit-receipt.json:1).
- Independently hashed final owned source: 91c7698d342e412bf8796575c7a72b4a2fe42985bc8ff1169a98018694619521, equal to self-review, batch before/after and commit receipts. Inspected self-review AST/decorator preservation method and commit receipt's one owned path, exact parent, committed-blob equality and excluded root metadata.
- Inspected fatal Ruff GREEN, formatter baseline (56 inherited units) and initial formatter RED (99 provisional units); format-final.log and format-commit.log are silent as expected. Root supplied task5-format-proof.json:1 with the exact committed --head e7cc533761 command, exit 0 in 0.71 s, equal before/after fingerprints and unchanged HEAD; that resolves the original silent-log exit-provenance gap. Source whitespace GREEN is stated by the implementer; no re-run was performed.
- Named cross-file check: risk that early _initial_screen_pushed changes the tested production lifecycle rather than only preventing duplicate setup. Read the unchanged shared factory contract and app._push_initial_screen guard/retained-screen publication, confirming the local helper leaves the factory intact and deliberately bypasses initial routing/offers only for tests supplying their own initial content screen (Tests/UI/app_factory.py:245; tldw_chatbook/app.py:4174, :4216, :4250).
- Changed-file context extensions were limited to cutoff successor/poll bodies and native post-barrier/assertion/finally tail, needed to judge preserved behavior and readiness cleanup (Tests/UI/test_console_runtime_ownership.py:2330–2465, :3370–3570). Root plan/Backlog/lesson metadata were excluded from source review.

### Assessment

**Task quality:** Approved, with external qualification pending.

**Reasoning:** The causal evidence supports the minimal local fixture correction, and retained real-startup navigation controls protect the product path. The readiness diagnostic strengthens failure provenance without weakening acceptance; current-head CI remains necessary to qualify the unproved historical native timeout and environment difference.
