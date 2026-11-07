"""Write the bounded evidence report; never modifies repository files."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path('/private/tmp/mixins-reader-restore-final')
proof = json.loads((OUT / 'final-frozen-proof.json').read_text())
for name, identity in proof['source'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == identity['sha256'], name
old = json.loads((OUT / 'old-owner-overlay-proof.json').read_text())
published_controller = subprocess.check_output(['git', 'show', proof['published_base'] + ':tldw_chatbook/UI/Library_Modules/library_media_controller.py'], text=True)
published_class = next(n for n in ast.parse(published_controller).body if isinstance(n, ast.ClassDef) and n.name == 'LibraryMediaController')
published_method = next(n for n in published_class.body if isinstance(n, ast.FunctionDef) and n.name == '_restore_library_media_focus')
assert old['old_method_ast_sha256'] == hashlib.sha256(ast.dump(published_method, include_attributes=False).encode()).hexdigest()

assert old['source_sha256'] == proof['source']['tldw_chatbook/UI/Library_Modules/library_media_controller.py']['sha256']
assert old['helper_sha256'] == proof['source']['tldw_chatbook/UI/Library_Modules/canvas_sync.py']['sha256']
assert old['test_sha256'] == proof['source']['Tests/UI/test_library_rail_switch_keeps_reader_and_note.py']['sha256']
assert old['screen_sha256'] == proof['source']['tldw_chatbook/UI/Screens/library_screen.py']['sha256']
assert "assert 'local:media:2' == 'local:media:1'" in (OUT / 'final-approved-old-red.log').read_text()
green = (OUT / 'final-approved-green.log').read_text()
assert '5 passed' in green, green[-1000:]
python = '/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python'
node = 'Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_programmatic_media_row_restore_keeps_reader_until_user_keys'
commands = []
for tag, basetemp, outcome, classification in [
    ('red', 'red-tmp', '1 failed in 10.77s', 'UNQUALIFIED setup error: sketch referenced Find token on screen rather than its current Media controller owner'),
    ('red-valid', 'red-valid-tmp', '1 failed in 41.29s', 'UNQUALIFIED setup timeout: initial resize suppression clears only real input; not a product RED'),
    ('setup-probe', 'setup-probe-tmp', '1 failed in 38.78s', 'UNQUALIFIED setup diagnostic: only resize_settling remained true; observer wrapper replaced with official hook, snapshot diagnostics removed'),
    ('red-native', 'red-native-tmp', '1 failed in 9.89s', 'VALID intermediate old-owner wrong-ID RED; final test bytes later formatted and supplemented'),
    ('green', 'green-tmp', '1 passed in 12.62s', 'VALID intermediate native default-writer + genuine keys GREEN; not final source qualification'),
]:
    tb = 'short' if tag == 'setup-probe' else 'long'
    commands.append({'phase': tag, 'command': f'PYTHONDONTWRITEBYTECODE=1 {python} -B -m pytest {node} -q --tb={tb} -p no:cacheprovider --basetemp={OUT / basetemp} > {OUT / (tag + ".log")} 2>&1', 'result': outcome, 'qualification': classification})
for tag, outcome, classification in [
    ('final-bytes-red', '1 failed in 15.60s', 'VALID intermediate exact old-method import overlay; not final test bytes'),
    ('frozen-old-red', '1 failed in 12.99s', 'VALID intermediate exact old-method import overlay; not final complete test bytes'),
    ('qualification-old-red', '1 failed in 9.74s', 'VALID intermediate exact old-method overlay before permanent failure tests and final approved combined source'),
    ('final-approved-old-red', '1 failed in 9.61s', 'AUTHORITATIVE final four-file identities: exact old restore body compiled in original globals; no repository edits, no runtime handler wrapper; same final test bytes'),
]:
    commands.append({'phase': tag, 'command': f'PYTHONDONTWRITEBYTECODE=1 {python} -B {OUT / "old_restore_owner.py"} > {OUT / (tag + ".log")} 2>&1', 'result': outcome, 'qualification': classification})
commands.extend([
    {'phase':'frozen-green', 'command':f'PYTHONDONTWRITEBYTECODE=1 {python} -B -m pytest Tests/UI/test_library_rail_switch_keeps_reader_and_note.py -q --tb=long -p no:cacheprovider --basetemp={OUT / "frozen-green-tmp"} > {OUT / "frozen-green.log"} 2>&1', 'result':'4 passed in 33.54s', 'qualification':'VALID intermediate original three journeys plus new native case; superseded by final five-case module run'},
    {'phase':'wiring', 'command':f'PYTHONDONTWRITEBYTECODE=1 {python} -B -m pytest Tests/Architecture/test_library_media_wiring.py -q --tb=short -p no:cacheprovider --basetemp={OUT / "wiring-tmp"} > {OUT / "wiring.log"} 2>&1', 'result':'13 passed in 1.51s', 'qualification':'Controller/public wiring; prior to AST-identical helper signature formatting'},
    {'phase':'exact-final-green', 'command':f'PYTHONDONTWRITEBYTECODE=1 {python} -B -m pytest {node} Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_n_in_the_reader_creates_a_note_naming_the_document_and_the_reader_keeps_its_place -q --tb=long -p no:cacheprovider --basetemp={OUT / "exact-final-green-tmp"} > {OUT / "exact-final-green.log"} 2>&1', 'result':'2 passed in 17.79s', 'qualification':'VALID intermediate before explicit expected-lambda binding and permanent marker test; superseded'},
    {'phase':'final-approved-green', 'command':f'PYTHONDONTWRITEBYTECODE=1 {python} -B -m pytest Tests/UI/test_library_rail_switch_keeps_reader_and_note.py -q --tb=long -p no:cacheprovider --basetemp={OUT / "final-approved-green-tmp"} > {OUT / "final-approved-green.log"} 2>&1', 'result':next(line for line in green.splitlines() if '5 passed' in line), 'qualification':'AUTHORITATIVE ordinary native run, final source, normal bootstrap_profile admission; original handlers intact'},
    {'phase':'private-marker-self-check', 'command':f'{python} -B {OUT / "marker_lifetime_self_check.py"} > {OUT / "marker-lifetime.log"} 2>&1', 'result':'10 exact-helper self-check modes passed', 'qualification':'Private branch contract only; four essential modes now permanently covered by final test module'},
    {'phase':'frozen-static-proof', 'command':f'{python} -B {OUT / "verify_frozen.py"} > {OUT / "final-frozen-verification.log"} 2>&1', 'result':'Passed; exact final source hashes, AST reversal, original test byte prefix, facade signature/doc, expected single Screen delta, new-block formatting and published lint parity'},
])
(OUT / 'commands.json').write_text(json.dumps({'cwd':str(Path.cwd()), 'published_base':proof['published_base'], 'commands':commands, 'note':'Intermediate RED overlay scripts used their then-owned basetemp values recorded in logs; the current saved script is exactly the final-approved overlay. Only final-approved-old-red/green + final-frozen-proof qualify final source.'}, indent=2) + '\n')
summary = f'''# Final reader restore repair — bounded evidence

Published base: `{proof['published_base']}`. Repository source/test files are frozen at the hashes in `final-frozen-proof.json`; `write_report.py` asserts all four identities still match and the final RED overlay used the same identities. No commit/index/branch/push or unrelated checkout changes were made.

## Scope and cause

Two focused mechanisms were repaired. A retained Reader's Media rail refresh no longer queues an independent generic first-row focus (`None if keep_media_reader else row0`); fresh list entry retains row0. Separately, the default synchronous recompose focus writer now arms the existing exact-target programmatic marker before `set_focus(..., scroll_visible=False)`, so its queued native DescendantFocus cannot select another media item. The common writer skips same-focused no-event calls and clears only its still-owned marker after failed/no-op focus. Later owners retain their marker. Genuine user keys remain authoritative. The generic Library focus helper is unchanged; this is not a claim that every programmatic focus path is fixed.

To preserve every existing size ceiling, only the existing restore method's executable body moved into the existing canvas_sync owner. The controller keeps its original signature and full docstring, and passes live defining globals for the original Widget/NoMatches/QueryError/grip dependencies. Reverse AST normalization proves the original gates, fallbacks and scroll=False call unchanged apart from the approved exact-marker/same-target/failure-cleanup delta. No new module, dependency, global suppression flag, alias or size allowance was added.

Sizes: Media controller5281→5246; canvas_sync1256→1314; LibraryScreen38177→38177 with only the approved argument conditional. Root owns exact ratchet rows, docs/task updates and publication. ADR required:no new ADR; routine focus repair and mechanical move within existing Library controller/canvas ownership.

## Authoritative verification

- `final-approved-old-red.log`:1 failed in9.61s at selected Media2 != Media1, after exact row event/dispatch and detail retirement. The private import overlay compiles only the exact pre-fix restore method in its original module globals; final repository source/test bytes stay untouched. All four overlay identities match final proof.
- `final-approved-green.log`: {next(line for line in green.splitlines() if '5 passed' in line)}. Ordinary final module run; no product callable replacement. Original three native Media/Notes journeys remain byte-identical as the entire published test file is an unchanged prefix. The original n→note→Media identity, source line, caret, scroll and loaded-row assertions/time budgets remain intact.
- New native case uses official App.run_test(message_hook=...), exact active-message-pump/target/focus matching and a queued call_later witness after the original synchronous handler. It waits for neutral owners and subsequent timer/request/detail-worker retirement; selected/loaded Media1 must survive system focus loss/restore to Media2's row. Real Down/Up/Down then checks DOM focus, selected ID, Reader selected ID, loaded ID and retirement for Media1/Media2/Media1.
- New permanent no-app facade test covers same-focused no-event, silent no-op, raised focus failure and a later owner's marker. It is one pytest node with four internal scenarios. The private10-mode self-check additionally retains missing/disabled/channel/None coverage as limited helper evidence.
- `wiring.log`:13 passed in1.51s; original controller/public wiring. This wiring run preceded helper signature formatting and restoring the approved Screen argument conditional; the final five-case module run qualifies the combined behavior. Root retains consolidated guard checks.
- `final-frozen-proof.json`: all original test/helper ASTs unchanged; original entire source byte prefix unchanged; facade signature/doc unchanged; reverse move AST true; sole approved Screen delta exact; all3new blocks formatted. Ruff vs exact published5a8:224 findings before and224 after, added0/removed0. Existing full-file lint/format debt was preserved, not represented as clean files. `git diff --check` passed.

Native deterministic contract is system-event qualification, not a reproduction of a hosted wall-clock/storage schedule. Initial resize suppression remains until real input; one normal Tab in setup clears it through the existing owner, then focus returns to the real Reader. No authority/resize/selection flag is written by the test. Early invalid-field and resize-precondition failures are harness-only, preserved in commands.json; intermediate proof/lint/RED/GREEN records are superseded by final-approved logs and final-frozen proof.

## Root handoff

Replace old fragile late-refresh census tail with:
`Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_programmatic_media_row_restore_keeps_reader_until_user_keys`

Add permanent marker unit node:
`Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_programmatic_media_restore_releases_only_its_failed_focus_marker`

Both are in the normally bootstrap_profile-marked module. Root chose floor1338. Final native slot is released after final-approved-green completion; no process is left running by this task. Root retains consolidated guards/performance/fresh hosted CI/review/commit/push/merge responsibility.

No full sweep, startup/performance rerun, live server test or historical cold keyboard baseline repeat was performed. Fresh hosted CI is still required before merge.
'''
(OUT / 'REPORT.md').write_text(summary)
entries = []
for path in sorted(OUT.iterdir()):
    if path.name == 'artifact-inventory.json':
        continue
    if path.is_file():
        entries.append({'path':path.name,'kind':'owned evidence file','bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    elif path.is_dir():
        files = [p for p in path.rglob('*') if p.is_file()]
        entries.append({'path':path.name,'kind':'owned pytest fixture/profile scratch; not source evidence','files':len(files),'bytes':sum(p.stat().st_size for p in files)})
(OUT / 'artifact-inventory.json').write_text(json.dumps({'root':str(OUT),'ownership':'Only artifacts created by this reader-restore task; prior diagnosis artifacts outside this directory are referenced separately and untouched. Inventory excludes itself.','entries':entries},indent=2) + '\n')
print('Final evidence report, exact command ledger and bounded owned inventory written.')
