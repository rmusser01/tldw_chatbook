import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path.cwd()
evidence = Path('/private/tmp/mixins-97a-archive-fix')
source_paths = [
    'tldw_chatbook/Widgets/Chat_Widgets/archive_recovery_cards.py',
    'tldw_chatbook/UI/Console_Modules/archive_recovery.py',
    'Tests/UI/test_console_archive_recovery_cards.py',
    'Tests/UI/test_console_archive_recovery_flow.py',
]

def identity(path):
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

assert (evidence / 'source-frozen-before.sha256').read_bytes() == (evidence / 'source-frozen-after.sha256').read_bytes()
run = json.loads((evidence / 'frozen-native-result.json').read_text())
assert run['exit_code'] == 0
suite = ET.parse(evidence / 'frozen-native.xml').getroot().find('testsuite')
assert suite is not None
assert {key: suite.attrib[key] for key in ('tests', 'failures', 'errors', 'skipped')} == {'tests': '34', 'failures': '0', 'errors': '0', 'skipped': '0'}
assert json.loads((evidence / 'ruff-frozen.json').read_text()) == []
assert json.loads((evidence / 'ruff-97a-four.json').read_text()) == []
report = {
    'base_head': '97a53c67aefeb64fab42eca1592dc2cf7d6bff46',
    'textual': (evidence / 'textual-version.txt').read_text().strip(),
    'root_cause': 'Native card Unmount remains queryable and is_mounted=True after children disappear. A pruning Select also retains its mounted flag while its internal controls disappear.',
    'repair': 'Adopt the latest model first; project only through all six typed, mounted, non-pruning and non-closing fixed controls. Existing on_mount replays latest state. Return truthful projection completion and let the view refuse deferred acknowledgment. Exact session/request changes reset consumed/scope state; typed decision handlers remain byte-identical.',
    'test_wait_repair': 'Two presence-only prefixes guard the source-remount test predicates during native removal/mount gaps; original clauses, assertions and deadlines are unchanged.',
    'adr': 'Existing backlog/decisions/166-agent-assisted-archive-recovery.md; no new architecture or ADR required.',
    'sources': [identity(root / path) for path in source_paths],
    'runtime': {'passed': 34, 'failed': 0, 'errors': 0, 'skipped': 0, 'pytest_seconds': 50.80, 'exit_code': run['exit_code'], 'argv': run['argv']},
    'coverage': {'card_module': 26, 'original_resize_cases': 5, 'original_native_recovery_journeys': 3, 'new_permanent_cases': 9},
    'static': {'ruff_97a': 0, 'ruff_final': 0, 'new_ranges_formatted': 4, 'diff_check': 'clean', 'source_bytes_frozen_across_final_run': True},
    'preservation': json.loads((evidence / 'source-preservation.json').read_text()),
    'superseded_observations': [
        'Original hosted CI exact None setter NoMatches preserved by root raw log.',
        'Original isolated resize observation failed its final band=None assertion (5.63s), not the card path; all five original cases pass in both combined native runs without changing their test file.',
        'Initial eight permanent cases failed before the repair (1.91s).',
        'First implementation verification failed on unsupported DOMQuery.filter(type); corrected to native query(type).filter(selector), then eight cases passed (1.74s).',
        'First combined card/resize cohort passed 30 cases (23.52s) before the extra Select retirement audit.',
        'Extra native source-remount flow failed its own unguarded query_one predicate while two companion flows passed (26.68s); its original outcome clauses and deadlines are preserved in the final repair.',
        'Select partial-retirement native probe reproduced SelectOverlay NoMatches with all six outer controls still present; permanent case failed before native pruning/closing refusal (1.44s).',
    ],
    'limits': ['No full suite, live provider calls, Git mutations, external mutations, pin raises or dictionary guard changes. Root owns remaining aggregate startup/static checks, task documentation, publication, hosted CI and actual merge.'],
}
(evidence / 'final-report.json').write_text(json.dumps(report, indent=2) + '\n')
artifact_names = [
    '97a-original-card.py', '97a-original-view.py', '97a-original-tests.py', '97a-original-flow.py',
    'native_lifecycle_probe.py', 'native-lifecycle-original.log', 'original-resize.log',
    'regressions-red.log', 'regressions-first-green.log', 'regressions-corrected-green.log',
    'final-widgets-resize.log', 'final-native-flows.log',
    'native_select_retirement_probe.py', 'native-select-retirement-probe.log', 'scope-retirement-red.log',
    'ruff-97a-four.json', 'ruff-frozen.json',
    'check_source_preservation.py', 'source-preservation.json',
    'format-frozen-card.log', 'format-frozen-view.log', 'format-frozen-tests.log', 'format-frozen-flow.log',
    'source-frozen-before.sha256', 'source-frozen-after.sha256', 'textual-version.txt',
    'run_final_cohort.py', 'frozen-native.log', 'frozen-native.xml', 'frozen-native-result.json',
    'diff-check.log', 'diff-numstat.txt', 'source-frozen.diff',
    'pytest-frozen-native/test_compact_confirmation_focu0/compact-confirm-target.svg',
    'pytest-frozen-native/test_compact_confirmation_focu0/compact-confirm-target.txt',
    'final-report.json', 'seal_artifacts.py',
]
artifacts = [identity(evidence / path) for path in artifact_names]
assert len({item['path'] for item in artifacts}) == len(artifacts)
inventory = {'producer': str(evidence), 'count': len(artifacts), 'artifacts': artifacts}
(evidence / 'artifact-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
for item in artifacts:
    assert identity(Path(item['path'])) == item
print(json.dumps({'artifact_count': len(artifacts), 'inventory': identity(evidence / 'artifact-inventory.json'), 'report': identity(evidence / 'final-report.json')}, indent=2))
