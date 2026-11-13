from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import zipfile

REPO = Path.cwd().resolve()
SDD = REPO / '.superpowers/sdd/2026-10-20-workspace-project-semantic-index'
QA = REPO / 'Docs/superpowers/qa/2026-10-20-workspace-project-context/semantic'
DEST = QA / 'release-evidence'
WHITELIST = SDD / 'task4-release-commit-whitelist.json'
GIT = ['git', '-c', 'safe.directory=' + REPO.as_posix()]

def facts(data):
    return {'bytes': len(data), 'sha256': sha256(data).hexdigest()}

def git(*args):
    return subprocess.run([*GIT, *args], check=True, capture_output=True).stdout

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def jbytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True) + '\n').encode('utf-8')

def verify(ref):
    rows = json.loads(WHITELIST.read_text(encoding='utf-8'))
    all_names = set(rows)
    if ref == '':
        staged = set(git('diff', '--cached', '--name-only').decode('utf-8').splitlines())
        assert staged == all_names, (sorted(staged - all_names), sorted(all_names - staged))
    for name, expected in rows.items():
        assert facts(git('cat-file', 'blob', ref + ':' + name)) == expected, name
    manifest_name = (DEST / 'manifest.json').relative_to(REPO).as_posix()
    manifest = json.loads(git('cat-file', 'blob', ref + ':' + manifest_name))
    archive_name = (DEST / 'original-logs.zip').relative_to(REPO).as_posix()
    import io
    archive_data = git('cat-file', 'blob', ref + ':' + archive_name)
    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        expected_members = {entry['storage']['member'] for entry in manifest['originals'] if entry['storage']['kind'] == 'zip'}
        assert set(archive.namelist()) == expected_members
        for entry in manifest['originals']:
            storage = entry['storage']
            original = SDD / entry['sdd_relative_path']
            if storage['kind'] == 'zip':
                retained = archive.read(storage['member'])
            elif storage['kind'] == 'direct':
                name = (DEST / storage['path']).relative_to(REPO).as_posix()
                retained = git('cat-file', 'blob', ref + ':' + name)
            else:
                retained = git('cat-file', 'blob', ref + ':' + storage['repo_path'])
            assert facts(retained) == {'bytes': entry['bytes'], 'sha256': entry['sha256']}, entry['sdd_relative_path']
            assert retained == original.read_bytes(), entry['sdd_relative_path']
    print(('INDEX' if ref == '' else ref) + ' BLOB PASS:', len(rows), 'exact blobs;', len(manifest['originals']), 'originals;', len(expected_members), 'ZIP members')
    assert facts((QA / 'receipt-manifest.json').read_bytes()) == manifest['existing_271_manifest']
    print('Original 271 receipt manifest unchanged: True')


def prepare():
    assert git('rev-parse', 'HEAD').decode().strip() == 'd0124ba2d1630fa8b04eb3369b4abb7da8d9e9aa'
    assert not DEST.exists(), DEST
    old_manifest = facts((QA / 'receipt-manifest.json').read_bytes())
    prefixes = (
        'task4-final-ci-census-01-', 'task4-final-core12-01-',
        'task4-final-real-app-01-', 'task4-final-console4-01-',
        'task4-final-settings-light170-01-', 'task4-final-qual-trust-01-',
        'task4-final-model-trust-01-', 'task4-final-console-trust-01-',
        'task4-final-light-trust-01-', 'task4-final-read-trust-01-',
        'task4-final-read-trust-02-', 'task4-final-scale-trust-01-',
        'task4-hosted-fixture-core12-01-', 'task4-hosted-fixture-casefold-02-',
        'task4-hosted-fixture-core12-03-', 'task4-hosted-fixture-trust-01-',
        'task4-hosted-fixture-trust-02-', 'task4-hosted-fixture-trust-03-',
        'task4-hosted-fixture-trust-04-', 'task4-coverage-red-01-',
        'task4-coverage-fixture-green-01-', 'task4-coverage-settings-green-01-',
        'task4-coverage-red-trust-01-', 'task4-coverage-green-trust-01-',
        'task4-coverage-settings-trust-01-', 'task4-nested-ready-read-01-',
        'task4-semantic-scale-final-01-', 'task4-semantic-scale-continuation-01-',
        'task4-semantic-scale-continuation-02-',
        'task4-scale-continuation-trust-round4-01-',
        'task4-scale-continuation-trust-round4-02-',
    )
    exact = (
        'task-4-report.md', 'task-4-final-qualification-report.md',
        'task-4-fix-round-3-review-report.md',
        'task4-continuation-harness-independent-review.md',
        'task4-continuation-v2-independent-review.md',
        'task4-continuation-v2-preparation.diff',
        'task4_semantic_scale_final.py', 'run-task4-semantic-scale-final.ps1',
        'task4_semantic_scale_continuation.py', 'run-task4-semantic-scale-continuation.ps1',
        'task4_semantic_scale_continuation_v2.py', 'run-task4-semantic-scale-continuation-v2.ps1',
        'task4_nested_ready_read_01.py', 'run-task4-nested-ready-read-01.ps1',
        'run-task4-hosted-fixture-targets.ps1', 'run-qualified-targeted.ps1',
        'repair-sdd-test-trust.ps1', 'task4-hosted-windows-120735971047-raw.log',
        'task4-hosted-windows-diagnosis.md', 'task4-settings-semantic-textual-light-170.svg',
        'root-rebase-production-mapping.json', 'root-rebase-range-diff.txt',
        'root-pr3237-status-after-local-scale.json', 'retain-task4-release-evidence.py',
    )
    candidates = [p for folder in (SDD, SDD / 'task2-receipts') for p in folder.iterdir() if p.is_file() and p.name.startswith(prefixes)]
    for name in exact:
        p = SDD / name
        assert p.is_file(), name
        candidates.append(p)
    assert len(set(candidates)) == len(candidates)
    for prefix in prefixes:
        assert any(p.name.startswith(prefix) for p in candidates), prefix
    forbidden = {'.sqlite', '.db', '.safetensors', '.bin', '.pth', '.dll', '.pyd'}
    assert not any(p.suffix.lower() in forbidden for p in candidates)
    existing = {}
    for p in QA.rglob('*'):
        if p.is_file():
            existing.setdefault(p.name, []).append(p)
    selected = []
    reused = []
    for p in sorted(candidates):
        data = p.read_bytes()
        matches = [q for q in existing.get(p.name, []) if q.read_bytes() == data]
        if matches:
            reused.append((p, matches[0]))
        else:
            selected.append(p)
    by_group = Counter('task2-receipts' if p.parent.name == 'task2-receipts' else 'sdd' for p in selected)
    print('WHITELIST:', len(candidates), 'requested originals;', len(selected), 'new;', len(reused), 'existing exact-name/byte references;', dict(by_group))
    print('RAW BYTES:', sum(p.stat().st_size for p in selected))

    report = SDD / 'task-4-report.md'
    report_text = report.read_bytes().decode('utf-8')
    report_text = report_text.replace('compares **all 18** original row fields.', 'compares **all 17** original row fields (correcting the earlier count).')
    report_text = report_text.replace('returned `SHAPE PASS: 18 receipt fields, namespace from retained key, flattened generation exactly equals original SQLite row`.', 'was described with an incorrect field count in the earlier notes; the independent focused review confirms 17 actual fields and preserves their exact comparison. That earlier reported shape-output text is not claimed as a retained execution receipt.')
    append = '''

## Round 4 release qualification and exact-byte retention (2026-10-24)

Production source stayed at `d0124ba2d1630fa8b04eb3369b4abb7da8d9e9aa` throughout both continuations. The earlier integrated `a2b56b98773e7edfe9da82f2bb973d135860c7b2` final local groups are retained with their original pre/post hashes, child logs/XML and exact invocation records: contract1/census664, core33 (no SentenceTransformers), actual cached-model app1 with durable Library assertions, Console4, and settled light170 Settings1. The original final-head report and controller-generated historical Git mapping preserve attribution: 16 production blobs were identical through rebase, and the later DB Ready-coverage projection is explicitly separate. No final-source equivalence is inferred for missing historical source bytes.

Round3 retained local source-fix evidence includes the first fixture attempt (31 pass/2 fail), corrected casefold2/2 and core33/33, coverage RED (`None == 1`), coverage/fixture8/8 and final settings-mismatch1/1. The original hosted job120735971047 log is 462,968 bytes with SHA256262284534DFD6F8509949BC9B52765D28CB24A7A8FBA4C8BDFC71D26DFB6DABD; it remains adverse. The private round3 review remains named literally. These are separate evidence sources, not a hosted pass.

Approved invocation1: `repair-sdd-test-trust.ps1 -RunName task4-scale-continuation-trust-round4-01`, then `run-task4-semantic-scale-continuation.ps1 -RunName task4-semantic-scale-continuation-01`. Literal trust output for both SDD and native-temp: `trusted_directory trusted_existing_directory`. The reviewed executed pair hashes are Python89985FF2451D6DAC99728FA860A0CD2602BCB1262083EF7DA57FCDE70349EE71 and PowerShell3C748E4693155FAEC51F5E6E3CE182863FF92912D5CD5CB923CFD594C1A350EF. Child59032 exited1 without timeout or forced termination; source/harness and both old input JSON hashes matched. Exact nested Ready generation/revision3/manifest329/count20000/dimension384 was reused, then the combined query proof failed at line655 before cancellation or wide work. No HF constructor/encode event was emitted; the original ToolResult was not saved, so its exact refusal/fallback is unknown. The report's `adverse_cleanup_or_source_unproved` also covers incomplete functional phases; it does not independently prove cleanup failure. Positive final app/service/sampler/owner receipts and absent PID/native-temp were observed. No second nested build occurred.

Source inspection found a harness lifetime defect: raw preflight registry lookup preceded a finite service registry owner and could leave a cached caller handle on a pooled thread. The old failure's exact runtime cause remains unproved. The separately reviewed v2 wraps the entire preflight in the existing service registry scope, preserves safe `_store` ownership in the generation reader, records original ToolResult and fresh metrics before parsing/assertion, and emits individual proof booleans plus observation-only issuer/registry-retirement reason/thread events. Production calls/returns/re-raises remain unchanged. Preparation used standard-library-only source copying plus `ast.parse` and PowerShell `Parser.ParseFile`: literal `AST PASS; originals unchanged; retained_ready_generation unchanged; no execution/import` and `POWERSHELL PARSE PASS; no launcher execution`. No test/model/DB was run during preparation. Actual generation/retained raw-row comparison preserves all17 fields. The first retention preparation stopped at its explicit assertion on the old18-field report wording before any release/docs writes; only the private retention helper had been created. This factual narrative was corrected against the retained independent review before retrying retention; no receipt or query was rerun.

Approved invocation2: `repair-sdd-test-trust.ps1 -RunName task4-scale-continuation-trust-round4-02`, then `run-task4-semantic-scale-continuation-v2.ps1 -RunName task4-semantic-scale-continuation-02`. Literal trust output again reports both paths trusted. Frozen v2 pair: Python5A47F12E4664C6C617A56318F90DD438AA1E83126E74E5F2C49AF299F04DFFF4, PowerShellA7628E6809D144CCC4110DED12BE069AE192A7131C800B4969CAF0EF60B8CF9C; scoped diff76C944D9BDACC65CCB4EA697E52244C5FA68F7B377C3743B53BB1A06A1C2EA04. Root's actual focused reviews and separate run approvals preceded launch; harness approval was not itself execution evidence.

| Shape | Actual result | Limits |
| --- | --- | --- |
| Nested retained20k | `qualified_continuation`; exact generationb5bc9b0b2ceb4e0c92d41b631662f5aa reused; selected whole-tool query73.906s; actual post-encode held128x384 producer; durableOff while worker alive/Disable pending; release then positive native/wrapper/carrier retirement in12.812s | No rebuild or replacement of the historical1510.984s final-01 build timing. Final-01 remains adverse because its full keyword-status comparison failed before query/cancel. |
| First wide20k | `qualified_shape`; 157 actual HF calls (156x128 plus32 tail) produced20000x384; Ready/native retirement build1591.578s; selected whole-tool query61.828s; held128x384/Off-before-release, physical cancellation12.672s | First wide semantic build; no favorable rerun. Query latency is materially slow despite meeting the original300s qualification bound. |

Both selected queries had all six proof booleans true: successful tool result, semanticReady, selectedf19999 path, fresh actual HF query encode, correct phase/dimension and fresh wrapper constructor. Their exact synthetic ToolResults and fresh constructor/encode/close metrics are retained. Both corpora remained20000 files/728900 bytes with identical before/after path-stat-content SHA256 (nestedf6efe371c180b0d84490799d34b4143ae746f4f7795650e5a91b6302ad6d1a8a; wide23c4bd88ff9cfeb6a1cf1e37be0cc9f453fed0eba7fdd710e0545c8696b3a30f). Both keyword states/policies remained Ready/indexed20000; comparisons preserve every keyword-status field except sharedcache_bytes. Final semantic states are Off after deliberate cancellation, not Ready. No later DB/status query was made to manufacture a receipt.

Five-second absolute sampled RSS peaks were882114560 bytes for nested continuation and923041792 bytes for wide; sampled whole-binding regular-file peaks74072173/54546539 bytes respectively. Wide baseline11186176 bytes; final RSS nested841551872/wide874565632 bytes. Samples do not establish an OS quota, an unsampled peak, or the independent analytical allocation bound. Existing native/source-conditioned1GiB allocation proof remains separately governed. Runtime: Windows10 build19045, Python3.12.10, actualSQLite3.49.1/source/options from the retained core qualification, 12 logical CPUs, cached MiniLM snapshotc9745ed1d9f207416be6d2e6f8de32d1f16199bf, explicitCPU/FP32/dimension384. No HTTP attempts, installer, model download or shared environment mutation. One synthetic selected-marker query per shape does not establish universal relevance; no POSIX/VFS or other-machine qualification follows.

Continuation2 child77376/launcher session1979 exited0; `whole_process_timed_out=false`, `forced_exit=false`, `source_and_harness_unchanged=true` across15 tracked source/harness/old-input paths. Events show no pending preflight owners, successful service close and completed runtime drain; build/cancel native owners physically retired, no pendingto_thread/sampler. Elevated read-only CIM literal outputs: `original_child_live=0`, `continuation_v2_python_live=0`, `native_temp_entries=0`. Original01 scripts/adverse bytes were rechecked unchanged. No executor shutdown warning or traceback occurred; literal factory startup Prompts-absent warning/ChaChaNotesDB-absent ERROR remain in raw stderr and are not suppressed.

Bounds were unchanged: build3600s, query/native join300s, actual-encode hold90s, whole process9600s. Original adverse attempts stay immutable. The current PR snapshot is an actual controller capture with hosted checks QUEUED; AC6 is not hosted-qualified and Task36423 remains In Progress. Root owns PR/CI, final review, task closure and any merge decision. No full suite or new covering/model test was run for retention.

Tracked delivery is `Docs/superpowers/qa/2026-10-20-workspace-project-context/semantic/release-evidence/`: exact whitelist, SHA256/byte manifest, lossless original log/XML ZIP, direct source/hash/invocation/visual/review receipts and limitations README. Existing271 receipt manifest/foundation/reviews remain unchanged. The retention helper verifies every new actual index/HEAD blob and every ZIP member against both original bytes and the explicit manifest. Postcommit literal verification output and commit identity are appended below in the private report after the immutable tracked report snapshot; no old receipt is reconstructed.
'''
    write(report, (report_text + append).encode('utf-8'))
    # Read the report after its authorized append; all other originals stay untouched.
    selected = [p for p in selected]

    arch = REPO / 'Docs/Architecture/project-indexing.md'
    a = arch.read_bytes().decode('utf-8')
    old = 'One captured keyword manifest fences immutable members, <=32-record/1MiB source/vector pages, durable claim/checkpoint and final Ready CAS.'
    new = 'One captured keyword manifest fences immutable members, durable claim/checkpoint and final Ready CAS. Only local Hugging Face source/provider pages may contain up to128 contiguous chunks and1MiB encoded source text; hosted/other providers remain at most32 rows/1MiB. Validate the complete provider result count and dimensions before append. Each native append transaction remains at most32 rows/1MiB with fresh source/manifest/scope/consent checks and checkpoint advancement after its successful slice. Query vector pages remain at most32 rows/1MiB; exact cosine ranking keeps its unchanged single outer `store.sql_unit()`/registry fence across all pages, rather than a transaction per page.'
    assert a.count(old) == 1
    write(arch, a.replace(old, new).encode('utf-8'))
    top_readme = QA / 'README.md'
    t = top_readme.read_bytes().decode('utf-8')
    marker = '# Task 4 semantic consent and qualification evidence'
    assert t.count(marker) == 1
    t = t.replace(marker, marker + '\n\nCurrent local release evidence is in [release-evidence](release-evidence/README.md). It retains the final integrated local checks, round3 fixes, original adverse scale/continuation attempts and positive continuation02. The tables below describe earlier historical snapshots; the release record supersedes their pending local-qualification statements. Hosted AC6 is still queued/unqualified in the retained controller snapshot.', 1)
    write(top_readme, t.encode('utf-8'))
    lesson = REPO / 'backlog/docs/lessons-testing-evidence.md'
    l = lesson.read_bytes().decode('utf-8')
    title = '## A qualification preflight needs finite registry ownership and pre-assert payload retention'
    assert title not in l
    l += '\n\n' + title + '\n\n**Incident (TASK-36423, 2026-10-24).** Continuation01 reused a real20k Ready generation but failed a compound semantic-query assertion without saving ToolResult or individual predicates. Its actual fallback/refusal is unknown. Source inspection found raw preflight `get_runtime_binding` outside finite service ownership; WorkspaceDB retains per-thread handles, and a later semantic operation correctly refuses a borrowed caller connection. Thread reuse was not recorded, so that defect cannot be declared the old run\'s proven cause. Separately reviewed continuation02 used the existing service registry owner around preflight, saved actual query result/fresh metrics/booleans before assertions and recorded thread/issuer/retirement exceptions without changing production catch behavior; both corpus journeys passed. Keep the adverse receipt, use the actual finite owner before registry lookup, and save diagnostic facts before a compound assertion. Do not close borrowed connections or weaken model admission to make a qualification pass.\n'
    write(lesson, l.encode('utf-8'))

    DEST.mkdir()
    write(DEST / '.gitattributes', b'# All retained original and generated evidence bytes are manifest-pinned.\n* -text\n')
    originals = []
    archive_path = DEST / 'original-logs.zip'
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p in selected:
            data = p.read_bytes()
            rel = p.relative_to(SDD).as_posix()
            target = 'originals/' + rel
            if p.suffix in ('.log', '.xml'):
                info = zipfile.ZipInfo(target, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
                storage = {'kind': 'zip', 'archive': 'original-logs.zip', 'member': target}
            else:
                write(DEST / target, data)
                storage = {'kind': 'direct', 'path': target}
            originals.append({'sdd_relative_path': rel, **facts(data), 'storage': storage})
    for p, existing_path in reused:
        data = p.read_bytes()
        originals.append({'sdd_relative_path': p.relative_to(SDD).as_posix(), **facts(data), 'storage': {'kind': 'existing', 'repo_path': existing_path.relative_to(REPO).as_posix()}})
    readme = '''# Local release evidence for TASK-36423

This adds exact original receipts after the existing271-file set. [manifest.json](manifest.json) is the explicit source whitelist, original SHA256/byte inventory and storage map. Log/XML originals are losslessly archived in [original-logs.zip](original-logs.zip); direct JSON, source scripts, reviews and visuals remain under `originals/`. Matching already-retained originals are referenced, not copied. Each ZIP member and direct file was compared with the private original. The scoped `-text` attribute preserves bytes across Git checkouts; the retention helper verifies actual index/HEAD blobs and all archive members, not merely working-tree hashes. No model, profile or SQLite payload is included.

| Evidence group | Actual result and attribution |
| --- | --- |
| Integrated a2 local checks | Contract1/census664; core33 without SentenceTransformers; cached actual-app1 including durable Library policy/item scope; Console4; settled light170 Settings1. Original pre/post/invocation/childXML/log sources are retained. The original final-head report describes those checks at a2; no whole-model run was repeated for hashes. |
| Historical source mapping | `root-rebase-production-mapping.json` and `root-rebase-range-diff.txt` are explicitly controller-generated Git evidence, not an invented implementer handoff. All16 production blobs match old/rebased heads; the later current DB Ready-coverage projection is the declared difference. Original executed source hashes remain attached to their own runs. |
| Round3 fixture and coverage | First fixture run31pass/2fail is adverse; corrected casefold2/2 and core33/33 pass. Coverage RED fails `None == 1`, GREEN8/8 and settings-mismatch1/1 pass. Literal independent review is retained. Original hosted failure job120735971047 log remains462968 bytes/hash26228453...DFB6DABD, not a hosted success. |
| Original final01 scale | Nested actual MiniLM build1510.984s reached Ready and native retirement, then invalid full keyword-status comparison stopped query/cancel; wide never ran in that attempt. Original adverse/log/pre/post/events/scripts stay intact. It recorded no complete prebuild keyword status, so do not claim onlycache_bytes changed historically. |
| Read-only nested receipt | Reconfirmed exact generationb5bc9b0b2ceb4e0c92d41b631662f5aa/rev3/manifest329/20000members/384D with unchanged four-file inventory. Raw generation row has17 fields; projected scope flattening preserves all17. |
| Continuation01 | Expected nested Ready reused, but compound query proof failed. No actual HF constructor/encode event and no persisted ToolResult: exact fallback/refusal remains unknown. Raw preflight registry lifetime defect is source-backed; old runtime causality is unproved. Child/service/app/sampler retired, no timeout/force, no HTTP, source/corpus unchanged. Wide and cancellation not reached. |
| Continuation02 | Nested reuses Ready: query73.906s, held actual128x384 cancellation12.812s. First wide build1591.578s, query61.828s, held cancellation12.672s. Both selected queries satisfy every semantic proof; both held producers reach durableOff while worker alive/Disable pending, then release and physically retire. Both keyword states remain Ready/indexed20000 with saved policy unchanged. Final semantics deliberately Off. |

Continuation02 child77376 and launcher exited0; no timeout or forced exit; no pending owners, sampler or native threads; app/service/runtime drain completed. Read-only physical checks returned `original_child_live=0`, `continuation_v2_python_live=0`, `native_temp_entries=0`. All15 frozen source/harness/consumed-input hashes match. Both retained20k corpora are unchanged (728900 bytes each); no rebuild of nested, downloads, installs, HTTP or shared-environment mutation. The selected model is cached MiniLM snapshotc9745ed1d9f207416be6d2e6f8de32d1f16199bf, CPU FP32, dimension384.

Measured selected whole-tool queries62-74 seconds are a material performance limitation, even though they meet the original300-second qualification deadline. Five-second absolute sampled RSS peaks are882114560 B nested and923041792 B wide; sampled whole-binding regular-file peaks74072173/54546539 B. Wide final build throughput is20000/1591.578 (about12.57 vectors/s); historical nested build is20000/1510.984 (about13.24 vectors/s), with original attribution retained. These are single local runs, not universal relevance/latency claims or unsampled peak/OS-quota proof. Existing analytical source-conditioned allocation proof remains separate. Qualified runtime is Windows10 build19045, Python3.12.10 and SQLite3.49.1 with source/options printed in the core receipts; no POSIX/VFS/other-machine claim follows. Raw startup warnings and all adverse attempts are retained without suppression.

The actual controller `root-pr3237-status-after-local-scale.json` snapshot still has hosted checks QUEUED. Local evidence does not satisfy hostedAC6. Task36423 remains In Progress; root owns final review, external PR/CI and closure. No semantic auto-merge is authorized. [The literal Task4 report snapshot](originals/task-4-report.md) contains actual invocations, preparation hashes, output counts and limitations; later postcommit verification is appended to that same private report without changing the archived original snapshot.
'''
    write(DEST / 'README.md', readme.encode('utf-8'))
    manifest = {
        'kind': 'original-byte task4 release whitelist; local qualification, not hosted approval',
        'production_head': 'd0124ba2d1630fa8b04eb3369b4abb7da8d9e9aa',
        'existing_271_manifest': old_manifest,
        'requested_prefixes': prefixes,
        'requested_exact_names': exact,
        'originals': originals,
        'generated_artifacts': [{'path': p.relative_to(DEST).as_posix(), **facts(p.read_bytes())} for p in (DEST / '.gitattributes', DEST / 'README.md', archive_path)],
    }
    write(DEST / 'manifest.json', jbytes(manifest))
    assert facts((QA / 'receipt-manifest.json').read_bytes()) == old_manifest
    changed = [arch, top_readme, lesson, *sorted(p for p in DEST.rglob('*') if p.is_file())]
    # Ordinary prose uses Git's canonical LF; evidence has scoped -text.
    rows = {p.relative_to(REPO).as_posix(): facts(p.read_bytes() if p.is_relative_to(DEST) else p.read_bytes().replace(b'\r\n', b'\n')) for p in changed}
    write(WHITELIST, jbytes(rows))
    # Fresh byte verification before Git staging.
    with zipfile.ZipFile(archive_path) as archive:
        for entry in originals:
            storage = entry['storage']
            if storage['kind'] == 'zip': data = archive.read(storage['member'])
            elif storage['kind'] == 'direct': data = (DEST / storage['path']).read_bytes()
            else: data = (REPO / storage['repo_path']).read_bytes()
            assert data == (SDD / entry['sdd_relative_path']).read_bytes()
            assert facts(data) == {'bytes': entry['bytes'], 'sha256': entry['sha256']}
    print('PREPARED:', len(rows), 'explicit commit paths;', len(originals), 'original receipts;', sum(e['storage']['kind'] == 'zip' for e in originals), 'ZIP members')
    print('ARCHIVE:', archive_path.stat().st_size, 'bytes;', facts(archive_path.read_bytes())['sha256'])
    print('ORIGINAL BYTE PASS; existing271 manifest unchanged; root Task36423 excluded')

parser = argparse.ArgumentParser()
parser.add_argument('--verify-index', action='store_true')
parser.add_argument('--verify-head', action='store_true')
args = parser.parse_args()
if args.verify_index: verify('')
elif args.verify_head: verify('HEAD')
else: prepare()
