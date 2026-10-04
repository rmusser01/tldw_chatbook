from pathlib import Path
import ast,json,hashlib,subprocess
paths=['Tests/UI/test_library_conversation_reader.py','Tests/UI/test_library_conversation_reader_freshness.py','tldw_chatbook/UI/Library_Modules/library_conversation_reader_controller.py','tldw_chatbook/UI/Library_Modules/library_conversations_state.py','tldw_chatbook/UI/Library_Modules/library_skills_controller.py']
before={p:Path(p).read_text() for p in paths};comments={}
p=Path(paths[0]);s=p.read_text();s=s.replace('from Tests.UI.test_library_shell import (','from Tests.private_profile import private_profile_test\nfrom Tests.UI.test_library_shell import (')
s=s.replace('@pytest.mark.asyncio\nasync def test_reader_info_is_explicit_and_truthful() -> None:', '@private_profile_test\n@pytest.mark.asyncio\nasync def test_reader_info_is_explicit_and_truthful(request) -> None:')
s=s.replace('@pytest.mark.asyncio\nasync def test_page_drift_confirms_exact_identity_before_declaring_deletion(\n    monkeypatch: pytest.MonkeyPatch,','@private_profile_test\n@pytest.mark.asyncio\nasync def test_page_drift_confirms_exact_identity_before_declaring_deletion(\n    monkeypatch: pytest.MonkeyPatch,\n    request,');p.write_text(s)
p=Path(paths[2]);s=p.read_text();old='''        # (task-32056) ``LibraryScreen._library_conversation_workspace_block``
        # -- the workspace-registry read behind the reader's inline refusal.
        # Bound like every other cross-cluster dependency: the depth-state
        # cache it consults is shell-wide, not reader-owned.
''';new='''        # task-32056: bind LibraryScreen._library_conversation_workspace_block;
        # its workspace-registry refusal reads the shell-wide depth-state cache.
''';assert old in s;p.write_text(s.replace(old,new));comments[str(p)]={'before':old,'after':new}
p=Path(paths[4]);s=p.read_text();a=s.index('        # task-8 (skills-script-execution) fix: NOT a direct call.');b=s.index('        def _render_then_arm()',a);old=s[a:b];new='''        # task-8: the off-thread grant lookup may finish before editor remount.
        # Rendering early swallows NoMatches/QueryError and never retries,
        # leaving "not granted" visible despite a true in-memory grant.
        # Screen call_after_refresh originally ordered that render, but
        # task-15457 moved recomposition to the canvas's own message pump.
        # A screen callback cannot order work after that canvas recompose.
        # task-15790 measured the resulting race: grant stored True, render
        # ran before the button existed, and no retry repaired the panel.
        # Use the caller's canvas post-recompose hook so `then` runs only
        # after the new children mount, before rendering and arming them.
''';assert len(old.splitlines())==22 and len(new.splitlines())==10;p.write_text(s[:a]+new+s[b:]);comments[str(p)]={'before':old,'after':new}
proc=subprocess.run(['/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python','-m','ruff','format',*paths],capture_output=True,text=True);assert proc.returncode==0;print(proc.stdout)
d=lambda t:ast.dump(t,include_attributes=False);proof={'formatter_argv':proc.args,'formatter_exit':proc.returncode,'formatter_stdout':proc.stdout,'files':[],'comments':comments}
for p in paths:
 old=ast.parse(before[p]);new=ast.parse(Path(p).read_text());strict=d(old)==d(new)
 if p==paths[0]:
  fs=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))};o=fs(old);n=fs(new);selected={'test_reader_info_is_explicit_and_truthful','test_page_drift_confirms_exact_identity_before_declaring_deletion'}
  for name in selected:
   assert d(ast.Module(body=o[name].body,type_ignores=[]))==d(ast.Module(body=n[name].body,type_ignores=[]));n[name].args=o[name].args;n[name].decorator_list=o[name].decorator_list
  assert all(d(v)==d(n[k]) for k,v in o.items());new.body=[v for v in new.body if not isinstance(v,ast.ImportFrom) or v.module!='Tests.private_profile'];assert d(old)==d(new)
 else:assert strict
 # Logger statement source segments must remain exact, including indentation.
 def logs(source):
  tree=ast.parse(source);return [ast.get_source_segment(source,n) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id in {'logger','logging'}]
 assert logs(before[p])==logs(Path(p).read_text())
 proof['files'].append({'path':p,'before_sha256':hashlib.sha256(before[p].encode()).hexdigest(),'after_sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'strict_AST_equal':strict,'only_approved_fixture_prologues_otherwise_exact':p==paths[0],'all_diagnostic_statements_exact':True,'before_lines':len(before[p].splitlines()),'after_lines':len(Path(p).read_text().splitlines())})
assert len(Path(paths[2]).read_text().splitlines())==967;assert len(Path(paths[4]).read_text().splitlines())==3142
Path('.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-6-reader-repair-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print('Proof passed; Reader967 and Skills3142 unchanged caps')
