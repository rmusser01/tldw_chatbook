from pathlib import Path
import hashlib,json,subprocess,shlex
root=Path(__file__).resolve().parent
head='cde073ed62d1e8f0d6163bac0b5f667c9fe7670f'
text=f'''# Latest-dev bounded qualification follow-up

## Outcome and source identities

- Original BASE: `ec8eda1d39a5d8ae8ed043b4270da173b95f6652`.
- Frozen dev: `f0ffcf9e819b577bd38c416f38550969c75fb5a0`.
- Exact dev merged by controller: `9b28ce1479efed6bca687cfbb33d261322f83abe`.
- Initial merged HEAD: `149adb53f1d10b090e31a5f458e8c2ffcd3ca458`.
- Controller docs/identity reconciliation predecessor: `62806f75a1ad5e293e25a00123278c5f1307d601`.
- Owned final HEAD: `{head}`.

This follow-up is bounded to the requested baseline21, six telemetry owners, real-profile guard and complete creation owners, plus evidence-backed repairs in six source/test paths. It does not replace the original task report or its 91 command receipts. No app, helper, reviewer, full suite, merge, push or PR was run by this implementer. Controller-owned task/plan/identity updates were committed independently and were not staged here.

## Merge disposition

`latest-merge-source-probe` verifies the exact two parents of 149adb and that chat_screen's only merge delta is `spend.console_rate_limit_line(provider_key)`; removing that exact line restores the source of the first parent. The lessons file preserves every prior line with only disjoint insertions. Fourteen changed upstream Python files match committed bytes and pass scoped fatal Ruff. That initial proof predates the repairs below. Full upstream evidence/doc whitespace is not claimed: controller observed imported QA capture padding. Source-only whitespace covers production, tests and scripts.

## Actual RED and contract repairs

1. Original complete telemetry run: **61 failed, 601 passed**. Keep those 601 successful nodes as a distinct original result, not a manufactured combined green. Actual guarded config source selection refused per-case HOME/config redirects (`RecoveryRequired: raw_source_selection_changed`). Existing native `bootstrap_profile` markers retain the private collection-selected profile in real config/TLS/app consumers. Gateway, egress and cost-screen markers remove that fixture mismatch; a later isolated three-owner run exposed the same UI autouse import in spend projection (**555 passed, 19 setup errors**), so that owner also retains the private bootstrap. Real-profile guard/refusal policy is unchanged.
2. Llama metadata is deliberately asynchronous under TASK33081 AC1. Tests now drain the actual bounded background metadata tasks before asserting exact probe URLs and credentials. Keyless tests retain no-Authorization assertions; stored credentials remain exact on health, both metadata reads and generation. Metadata404 fallback remains tested. No send-path wait or production metadata policy was added.
3. ADR179's default custom endpoint execution is `custom-hosted`; URL/readiness/selected-provider identity remain pinned. Four stale legacy-only execution assertions now name the real default. A paired explicit `custom_endpoints_use_engine=False` control preserves rollback behavior. The all-handler fixture supplies both explicit URL and key for the hosted custom family. Executable `latest_gateway_fixture_probe.py` captures readiness key `custom_hosted` and real resolution; it did not change admission.
4. Mistral's transport moved from LLM_API_Calls to `mistral.py` / `hosted_chat.py`. The old fake intercepted neither the native transport nor native settings owner, producing502. The fixture now intercepts `hosted_chat.create_default_session` and `mistral.get_runtime_config_snapshot`, returns a strict real requests.Response (index0, assistant role, finish_reason stop), and retains exact separate alias endpoints and credentials. The intermediate502 with incomplete index-less response remains RED evidence; the strict normalizer was not relaxed.
5. Empty spend state still asserts Current $0 and no next-send charge; nonempty tracker failure remains unavailable with independent next-send forecast and cache alert. The incidental Context 11% pin was stale against the shared estimator: actual used 1000/safe 8488 rounds to 12%, including the512 tool/schema budget. `latest-spend-context-probe` first failed due the scratch import path; the corrected executable probe preserves that loader failure separately from estimator evidence. No price/estimator production change was made.
6. Mounted waiting gateways now inherit the existing cached-context protocol implemented by their sibling double; the production caller remains strict. No fallback was added to runtime code.
7. Governing spend spec `Docs/superpowers/specs/2026-09-04-console-current-and-next-send-spend-design.md:26` excludes unaccepted owners from Context and Current. The predispatch test now asserts positive draft tokens before admission, real VALIDATING state, retained optimistic USER transcript, request tokens0, and Current $0; accepted/dispatched positive controls remain. The exclusion and original conflicting test were introduced together in dev 144ac2e1083688bf10977d22540acef6943edcb1 and are unchanged at frozen dev. Linked ADR052/095 preserve context/settings ownership.
8. Same spec line36 requires active sends to cancel coalesced idle refresh. This was a genuine baseline defect. The canonical dispatch stops the pending timer; observational trace showed that the composer-clear event then rearmed it while runtime custody was already accepted but controller state remained IDLE. Existing ConsoleRuntime.has_custodied_turns now accepts an optional exact session filter, preserving no-argument behavior. Existing Input.Changed run_active projection includes that exact custody. No authority, extra state or new owner was created. Genuine native runtime acceptance before VALIDATING, other-chat idle edits, no-argument/filter truth, terminal custody cleanup, refused-send draft retention/rearm and keyboard callback never fires controls pass. This await-gap mechanism is the AC7 remediation approved by the controller before code.

## Verification stages and regression controls

- Initial merged source:21/21literal baseline nodes passed; requested six telemetry owners executed with61RED/601success. Other unchanged telemetry owners (rate limits/context controls) remain qualified by that original invocation; no generic repeat was made.
- Intermediate control runs preserve14 fail/47 pass, then4 fail/10 pass, then1 fail/23 pass. Keyboard timer tracing preserves the remaining failure and exact stop/rearm ordering.
- Final custody controls:3 pass. They catch removing dispatch cancellation, removing exact-session custody admission, using global custody to suppress another chat's idle edits, retaining custody after terminal release, and failing refusal rearm.
- Final complete amended owners: **611 passed, 2 warnings**. Amended-production literal baseline21: **21 passed**. Final-byte guard/creation owners: **115 passed, 2 inherited platform skips**, no warnings. Full results are the literal receipts below. Earlier greens are not represented as executing on later bytes. During the complete amended-owner run only the runtime accessor's wrapping was formatted; `latest-runtime-format-correction` records exact before/after hashes and identical Python AST. Subsequent baseline/creation checks and committed-source proof use final bytes.
- Static qualification is scoped fatal Ruff (E9,F63,F7,F82), source compile/equality and immutable changed-code formatter ratchets, not a claim to remove inherited full-style debt. Original64Python plus four amended test owners are checked against committed HEAD. Fourteen upstream provider/bootstrap files have a separate fatal-lint receipt. Obsolete unshipped migration additions remain absent.
- Controller's 11 derived guards passed on its working tree in `controller-preflight.json/.log`; this implementer did not repeat them and does not relabel that run as committed-HEAD evidence.

## Exclusions, warnings and limits

No skip/xfail marker was added or changed. New upstream guard inherited platform exclusions are `Tests/test_real_profile_guard.py::test_every_write_kind_into_the_profile_is_refused_and_recorded[setxattr]` and `[removexattr]`, exact reasons `no os.setxattr here` / `no os.removexattr here`. These APIs are unavailable on the installed Python, so those two paths remain unqualified. Other 55 guard nodes and all 60 creation nodes passed in the initial guard invocation. Frozen timer XFAIL and historical archaeology SKIP remain unchanged and separately limited in task-1-report.md; neither behavior is qualified by its exclusion.

The14 fail/47 pass control run emitted an unawaited Textual Timer coroutine warning and FD growth403(start 14, end 417, limit 200). Every warning from later runs is retained verbatim in the receipt output and log; no warning filter or cleanup threshold was changed. No broad FD cleanup was attempted. The first formatter snapshot used the same output filename as its command receipt, so run_check replaced the snapshot with the receipt; it was immediately recaptured into a distinct immutable baseline path. One intermediate hunk-detail file was reused by the next formatter pass; both outer literal invocations/results survive and final per-hunk details use a distinct file. Final checks qualify the corrected paths; no lost intermediate detail is manufactured.

Controller will merge/qualify any later docs/perf-only upstream delta, perform live Console qualification and handle publication. Those remain controller-owned and are not claimed completed here.

## Literal command receipts

Each JSON below stores exact argv, cwd, selected environment, returncode, elapsed time and full untruncated combined output. The companion log is the byte-preserved output. The command shown uses shell quoting solely for readability; argv JSON is authoritative. All tests use the installed 3.12 interpreter and TLDW_TEST_GC_EVERY=1, native private-profile isolation, fresh owned basetemps, no installs or real-profile writes.

'''
records=[]
for path in root.glob('latest-*.json'):
 if path.name.startswith('latest-dev-fetch'):continue
 value=json.loads(path.read_text())
 if isinstance(value,dict) and {'argv','cwd','env','returncode','output'}<=value.keys():records.append((path.stat().st_mtime,path,value))
for _,path,r in sorted(records):
 log=path.with_suffix('.log');output=r['output']
 summary=[line for line in output.splitlines() if 'passed' in line or line.startswith(('FAILED ','ERROR ','SKIPPED ')) or 'warnings.warn' in line or 'Recorded:' in line][-10:]
 text+='### '+path.stem+'\n\n```text\n'+shlex.join(r['argv'])+'\n```\n\n'
 text+='Environment: `'+json.dumps(r['env'],sort_keys=True)+'`. Cwd: `'+r['cwd']+'`. Exit: `'+str(r['returncode'])+'`. Elapsed: `'+str(round(r['elapsed_s'],3))+'s`.\n\n'
 text+=f'Full literal receipt: [{path.name}]({path}); stdout/stderr: [{log.name}]({log}). SHA256 JSON `{hashlib.sha256(path.read_bytes()).hexdigest()}`; log `{hashlib.sha256(log.read_bytes()).hexdigest()}`.\n\n'
 if summary:text+='Literal output excerpt:\n\n```text\n'+'\n'.join(summary)+'\n```\n\n'
 else:text+='Literal output is preserved in the linked log (including empty successful output).\n\n'
text+=f'\nRecorded {len(records)} owned follow-up command receipts. Executable capture/repair/trace/static/formatter sources and all detail JSON files are direct children of this same plan scratch directory, so the controller byte-exact archiver can retain them.\n'
(root/'latest-dev-followup-report.md').write_text(text)
print('Wrote latest-dev-followup-report.md',len(records),'receipts',head)
