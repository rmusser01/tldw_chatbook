import ast,hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
root=Path('/private/tmp/mixins-f095-perf-contracts');files=['Packaging/speculative_voice_history_gate.py','Tests/Performance/test_console_tab_sync_source_ownership.py','Tests/Performance/test_speculative_voice_latency.py']
proof=[]
for p in files:
 old=ast.parse(subprocess.check_output(['git','show','f09597c:'+p],text=True));new=ast.parse(Path(p).read_text())
 def constants(t):return {n.targets[0].id:ast.dump(n.value,include_attributes=False) for n in t.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id.isupper()}
 assert constants(old)==constants(new),p
 if p.endswith('test_speculative_voice_latency.py'):
  def markers(t):return [ast.dump(ast.Module(n.decorator_list,[])) for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='test_completed_pair_history_gate_meets_large_history_thresholds']
  assert markers(old)==markers(new)
 proof.append({'path':p,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'all_module_constants_unchanged':True})
(root/'source-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
d=json.loads((root/'history-report.json').read_text());assert d['row_counts']=={'baseline':1000,'large':100000};assert d['warmup_count']==10 and d['trial_count']==40 and d['measurement_order']=='ABBA';assert d['scan_count']==14
for corpus in ('baseline','large'):
 rows=[r for r in d['query_plans'][corpus] if r['table']=='console_prompt_mixin_attempts'];assert len(rows)==2 and all(r['uses_search'] and not r['uses_scan'] for r in rows)
(root/'history-assertion-proof.json').write_text(json.dumps({'method':'Evaluate original assertion order against captured unmodified helper return; JUnit omits xfail traceback.','preceding_assertions_pass':True,'first_failing_assertion':'assert report["scan_count"] == 0','actual':14,'mixin_both_corpora_indexed':True},indent=2)+'\n')
(root/'report.md').write_text('''# f095 hosted Perf Guard contract repair

Starting HEAD f09597cdd5a2b5af1b07dea7eabe2eb6d33d9c80. Only three test/harness files changed. Task38544 was reopened/plan updated by parent before work. No product, threshold, schema/index, Qodo, commit or push changes.

## Causes and fixes

- Compiled-source cold FULL fixture omitted upstream TASK37840's temporary-chip callback (also present on origin/dev). Added an explicit recording callback asserting publication completed before the one chip refresh. Existing source compilation/SHA, no-native-import oracle, deferred projection and replay checks stay intact.
- Reaching that path exposed another stale fixture: `lambda: object()` cannot satisfy the current real post-await attachment-visit check. Use a stable actual pure `ConsoleAttachVisit` value; no predicate/fence is mocked or weakened.
- Inventory increased28 to30 solely through console_prompt_mixin_attempts.user_message_id and assistant_message_id, already classified as forbidden by production. Pin their exact names and30 total in both old count oracles; assert26 forbidden distribution entries and100× scaling.
- The old distribution now summed1082, so the existing strict-xfail was masking setup drift. Allocate2checkpoint columns×44 and24others×38=1000, preserving100000 large observations. Same-table columns keep equal row counts. All trial counts, timings, ABBA order, thresholds and strict-xfail remain unchanged.

## Evidence

Exact hosted27-contract cohort:26 passed,1 expectedxfail in3.64s (`final.log/xml`). `run_observed.py` calls original tests/assertions and only records the history helper's unmodified return. Initial bounded run:25passed/1failed/1xfail; its failure revealed the stale attachment fixture (`contracts.log/xml`).

`history-report.json` proves full history construction and40trials/10warmups/ABBA completed, matching pragmas,1000/100000 locator observations and842/84200 physical unrelated rows (multiple columns share rows). The original assertions preceding scan_count all pass against that return; its14scans are the first failing assertion. `history-assertion-proof.json` records this inference explicitly because JUnit emits only the marker message, not the xfail traceback. No new run needed to infer the first failing assertion from the actual returned data.

There are7actual scan plans per corpus (14total), despite old marker prose saying8 unindexed columns; marker left unchanged. Both new Mix-in columns use actual covering indexes idx_console_mixin_attempt_user/assistant in both baseline and large corpora, no scans. No INDEXEDBY, fake plans, index edits or threshold changes.

Scoped default Ruff3->3 versus f095, zero new diagnostics (`lint.json`). Both history/inventory files format-clean; scoped diff whitespace check passes. `source-proof.json` records hashes and confirms all module constants unchanged; strict-xfail decorator AST unchanged. No full suite/budget rerun. All owned processes stopped.

## Owned files

- Packaging/speculative_voice_history_gate.py
- Tests/Performance/test_console_tab_sync_source_ownership.py
- Tests/Performance/test_speculative_voice_latency.py

Independent review is assigned by parent. `artifact-files.json` enumerates exact reports/logs/proofs and hosted input with hashes. Scratch profile databases excluded.
''')
paths=[root/n for n in ['report.md','contracts.log','contracts.xml','final.log','final.xml','history-report.json','run_observed.py','lint.json','patch.diff','source-proof.json','history-assertion-proof.json','write_report.py']]+[Path('/private/tmp/mixins-ci-f095-perf-red.log')]
(root/'artifact-files.json').write_text(json.dumps({'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]},indent=2)+'\n')
print(root/'report.md');print(root/'artifact-files.json')
