"""Finite original-six AC77 Linux runner; no product loader or policy changes."""
import ast
import errno
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['AC77_OUTPUT']).resolve()
PACKAGE = OUT / 'root-ac77-source-gate-r3'
ZIP = ROOT / '.github/ac77/root-ac77-source-gate-r3.zip'
SOURCE_MANIFEST = ROOT / '.github/ac77/source-manifest.json'
ZIP_SHA = '547de6ca3b7f018729ce9d496e33541c8cf647b9fa2e80aa01cd0977a2bf433e'
PACKAGE_SHA = '1fc254f75e87c8b43a4321b27fc464033ba68b65a94a5fff96f3384fd7631af5'
TEST = 'Tests/Backup_Recovery/test_config_companion_missing_record_evidence.py'
RUNTIME = (
    ('contextlib', 'contextlib.py', '8b7a477f978a8532852fd81e241c78182516bc4975d672d580a5848a76e11eb6', []),
    ('_pytest.fixtures', '_pytest/fixtures.py', '41dfd72c1e6da619906bf84945694c1cb9b919bad18303278fac35630c8c7748', []),
    ('_pytest.assertion.rewrite', '_pytest/assertion/rewrite.py', 'f231048ab925ef8585f309a1022470438ae2c77ffab1de0e98693e65547c316c', ['rewrite_asserts']),
)
VERSIONS = {'pytest': '8.4.2', 'pytest-asyncio': '1.2.0', 'pytest-timeout': '2.4.0'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', *args], cwd=ROOT).decode().strip()


def relative(name):
    value = PurePosixPath(name)
    assert not value.is_absolute() and '..' not in value.parts
    return ROOT / value


def source_map():
    names = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split('\0')
    return {name: digest(relative(name).read_bytes()) for name in sorted(set(names))
            if name.endswith('.py') and relative(name).is_file()}


def manifest():
    data = SOURCE_MANIFEST.read_bytes()
    assert re.fullmatch(r'[0-9a-f]{64}', os.environ['AC77_SOURCE_MANIFEST_SHA256'])
    assert digest(data) == os.environ['AC77_SOURCE_MANIFEST_SHA256']
    return json.loads(data)


def source_guard(label):
    frozen = manifest()
    actual = source_map()
    write(label + '-sources.json', actual)
    differences = [p for p in sorted(actual.keys() | frozen['python_sha256'].keys())
                   if actual.get(p) != frozen['python_sha256'].get(p)]
    controls = []
    required = {'.github/scripts/ac77_linux_qualification.py',
                '.github/workflows/console-pause-native-evidence.yml',
                '.github/ac77/root-ac77-source-gate-r3.zip'}
    assert set(frozen['control_sha256']) == required
    for name, expected in frozen['control_sha256'].items():
        observed = digest(relative(name).read_bytes())
        controls.append({'path': name, 'expected': expected, 'actual': observed})
        if observed != expected:
            differences.append(name)
    write(label + '-source-guard.json', {'python_count': len(actual), 'differences': differences,
                                        'controls': controls})
    assert not differences, differences
    return frozen


def package_guard():
    assert digest(ZIP.read_bytes()) == ZIP_SHA
    assert digest((PACKAGE / 'artifacts.json').read_bytes()) == PACKAGE_SHA
    for row in json.loads((PACKAGE / 'artifacts.json').read_text())['files']:
        assert digest((PACKAGE / row['path']).read_bytes()) == row['sha256'], row['path']


def selected_guard(label):
    package_guard()
    document = json.loads((PACKAGE / 'expected-sources.json').read_text())
    rows = document['sources'] + document['data_sources']
    expected = {r['logical_path']: r['sha256'] for r in rows}
    assert manifest()['selected_sha256'] == expected
    receipts, differences = [], []
    for row in rows:
        data = relative(row['logical_path']).read_bytes()
        observed = digest(data)
        receipts.append({'path': row['logical_path'], 'expected': row['sha256'], 'actual': observed})
        target = OUT / 'actual-checkout' / row['logical_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if label == 'prepare':
            target.write_bytes(data)
        if observed != row['sha256']:
            differences.append(row['logical_path'])
    write(label + '-selected.json', {'rows': receipts, 'differences': differences})
    assert not differences, differences


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    assert not OUT.is_relative_to(ROOT)
    write('platform.json', {'system': platform.system(), 'release': platform.release(),
          'machine': platform.machine(), 'python': platform.python_version(),
          'version': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
          'root': str(ROOT), 'temporary_root': tempfile.gettempdir(),
          'runner_os': os.environ.get('RUNNER_OS'), 'scope': 'Actual OS metadata, not native qualification'})
    assert sys.platform == 'linux' and sys.version_info[:3] == (3, 12, 11)
    assert os.environ['GITHUB_EVENT_NAME'] == 'workflow_dispatch'
    head = git('rev-parse', 'HEAD')
    assert re.fullmatch(r'[0-9a-f]{40}', os.environ['AC77_EXPECTED_COMMIT'])
    assert head == os.environ['AC77_EXPECTED_COMMIT'] == os.environ['GITHUB_SHA']
    write('git.json', {'head': head, 'tree': git('rev-parse', 'HEAD^{tree}'),
          'parents': git('show', '-s', '--format=%P', 'HEAD').split(),
          'ref': os.environ['GITHUB_REF'], 'workflow': os.environ.get('GITHUB_WORKFLOW'),
          'run_id': os.environ['GITHUB_RUN_ID'], 'run_attempt': os.environ['GITHUB_RUN_ATTEMPT']})
    frozen = source_guard('prepare')
    for name in frozen['control_sha256']:
        target = OUT / 'actual-checkout' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative(name), target)
    shutil.copyfile(SOURCE_MANIFEST, OUT / 'root-source-manifest.json')
    assert digest(ZIP.read_bytes()) == ZIP_SHA
    shutil.copyfile(ZIP, OUT / 'original-package.zip')
    with zipfile.ZipFile(ZIP) as archive:
        for row in archive.infolist():
            parts = PurePosixPath(row.filename).parts
            assert parts and parts[0] == PACKAGE.name and '..' not in parts
            assert not PurePosixPath(row.filename).is_absolute()
            assert (row.external_attr >> 16) & 0o170000 != 0o120000
        archive.extractall(OUT)
    selected_guard('prepare')
    (OUT / 'linux-mountinfo.txt').write_bytes(Path('/proc/self/mountinfo').read_bytes())


def freeze_runtime():
    rows, errors = [], []
    versions = {}
    for name, expected in VERSIONS.items():
        try:
            versions[name] = importlib.metadata.version(name)
            if versions[name] != expected:
                errors.append({'distribution': name, 'expected': expected, 'actual': versions[name]})
        except BaseException as error:
            errors.append({'distribution': name, 'error': repr(error)})
    for name, leaf, expected, slots in RUNTIME:
        try:
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            data = path.read_bytes()
            target = PACKAGE / 'runtime/linux' / leaf
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            row = {'module': name, 'runtime_path': str(path),
                   'snapshot_path': target.relative_to(PACKAGE).as_posix(),
                   'sha256': digest(data), 'expected_sha256': expected, 'slots': slots,
                   'spec_origin': module.__spec__.origin,
                   'scope': 'Read from this actual Linux runtime; no Darwin file copy'}
            rows.append(row)
            if (module.__spec__.name != name or Path(module.__spec__.origin).resolve() != path
                    or row['sha256'] != expected):
                errors.append({'module': name, 'expected': expected, 'actual': row['sha256']})
        except BaseException as error:
            errors.append({'module': name, 'error': repr(error)})
    write('runtime-freeze.json', {'versions': versions, 'rows': rows, 'errors': errors})
    assert not any(n == 'tldw_chatbook' or n.startswith('tldw_chatbook.') for n in sys.modules)
    assert not errors and len(rows) == 3, 'Actual Linux runtime differs; archive and review before any test'
    path = PACKAGE / 'runtime-sources.linux.json'
    path.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    return path


def runtime_guard(label):
    path = PACKAGE / 'runtime-sources.linux.json'
    if not path.exists():
        return  # A rejected/missing runtime freeze never authorized a test.
    expected = {name: sha for name, _, sha, _ in RUNTIME}
    rows = json.loads(path.read_text())
    assert len(rows) == 3 and {row['module'] for row in rows} == set(expected)
    observed = []
    for row in rows:
        actual = digest(Path(row['runtime_path']).read_bytes())
        snapshot = digest((PACKAGE / row['snapshot_path']).read_bytes())
        observed.append({'module': row['module'], 'runtime_path': row['runtime_path'],
                         'actual': actual, 'snapshot': snapshot, 'expected': expected[row['module']]})
    write(label + '-runtime.json', observed)
    assert all(r['actual'] == r['snapshot'] == r['expected'] for r in observed)


def run_pytest(label, command, environment):
    write(label + '-command.json', {'argv': command, 'cwd': str(ROOT), 'executable': sys.executable,
          'explicit_environment': {k: environment[k] for k in ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE')}})
    process = None
    receipt = {'label': label, 'started_monotonic': time.monotonic()}
    primary = None
    cleanup = []
    try:
        with (OUT / (label + '.log')).open('wb') as output:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT)
            receipt['pid'] = process.pid
            write(label + '-process-start.json', receipt)
            receipt['wait_returncode'] = process.wait()
    except BaseException as error:
        primary = error
    finally:
        if process is not None:
            try:
                receipt['poll_before_cleanup'] = process.poll()
                if process.returncode is None:
                    receipt['cleanup_kill'] = True
                    process.kill()
                receipt['final_wait_returncode'] = process.wait()
                receipt['final_poll_returncode'] = process.poll()
                try:
                    receipt['unexpected_waitpid'] = os.waitpid(process.pid, os.WNOHANG)
                except ChildProcessError as error:
                    receipt['waitpid_errno'] = error.errno
                assert receipt.get('waitpid_errno') == errno.ECHILD
            except BaseException as error:
                cleanup.append(error)
        receipt['elapsed_seconds'] = time.monotonic() - receipt['started_monotonic']
        receipt['primary_error'] = None if primary is None else repr(primary)
        for guard in (source_guard, selected_guard, runtime_guard):
            try:
                guard(label + '-after')
            except BaseException as error:
                cleanup.append(error)
        receipt['cleanup_errors'] = [repr(e) for e in cleanup]
        write(label + '-process.json', receipt)
    errors = ([primary] if primary is not None else []) + cleanup
    if len(errors) > 1:
        raise BaseExceptionGroup('AC77 child and receipt failures', errors)
    if errors:
        raise errors[0]
    return receipt['wait_returncode']


def qualify():
    source_guard('before-runtime')
    selected_guard('before-runtime')
    runtime = freeze_runtime()  # Fixed actual-runtime guards precede both pytest launches.
    cases = next(ast.literal_eval(n.value) for n in ast.parse((PACKAGE / 'ac77_gate_plugin.py').read_text()).body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_CASES' for t in n.targets))
    assert len(cases) == 6
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join((str(PACKAGE), str(ROOT))), PYTHONDONTWRITEBYTECODE='1')
    base = [sys.executable, '-m', 'pytest', '-q', '-rA', '-p', 'no:cacheprovider', '-p', 'ac77_gate_plugin',
            '--ac77-installed-root=' + str(ROOT), '--ac77-archive=' + str(OUT / 'gates'),
            '--ac77-runtime-binding=' + str(runtime), TEST]
    result = run_pytest('collection', base + ['--collect-only'], env)
    assert result == 0, 'Original six collection failed; tests were not launched'
    collected = [line.strip().split('::')[-1] for line in (OUT / 'collection.log').read_text().splitlines()
                 if line.startswith(TEST + '::')]
    write('collection-names.json', collected)
    assert tuple(collected) == cases
    result = run_pytest('test', base + ['--junitxml=' + str(OUT / 'junit.xml')], env)
    xml = ET.parse(OUT / 'junit.xml')
    rows = [{'attributes': c.attrib, 'outcomes': [n.tag for n in c if n.tag in ('failure', 'error', 'skipped')],
             'properties': [p.attrib for p in c.findall('./properties/property')]} for c in xml.iter('testcase')]
    write('xml-summary.json', {'suite_attributes': [s.attrib for s in xml.iter('testsuite')], 'cases': rows})
    assert tuple(dict.fromkeys(r['attributes']['name'] for r in rows)) == cases
    gates = [json.loads(path.read_text()) for path in sorted((OUT / 'gates').glob('*/gate.json'))]
    expected_ids = [TEST + '::' + name for name in cases]
    gate_ids = [row['nodeid'] for row in gates]
    clean_gates = (len(gates) == 6 and set(gate_ids) == set(expected_ids)
                   and all(not row['observer_errors'] for row in gates))
    all_calls = clean_gates and all(
        any(r['when'] == 'call' and r['outcome'] in ('passed', 'failed') for r in row['reports'])
        for row in gates)
    skipped = any('skipped' in row['outcomes'] for row in rows)
    write('disposition.json', {
        'pytest_returncode': result, 'clean_six_gates': clean_gates,
        'all_six_actual_call_outcomes': all_calls, 'any_skip': skipped,
        'bounded_original_execution_credit': bool(all_calls and not skipped),
        'native_identity_values': 'Not separately recorded; original stock native assertions only',
        'scope': 'Original-source Linux regression baseline only; retain all failures; no product selection'})
    assert clean_gates and all_calls and not skipped, 'No original-six qualification credit for skipped/ineligible/setup-only controls'
    return result


def finalize():
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    for label, call in [('source', lambda: source_guard('final')),
                        ('selected', lambda: selected_guard('final')),
                        ('runtime', lambda: runtime_guard('final'))]:
        try:
            call()
        except BaseException as error:
            failures.append({'phase': label, 'error': repr(error)})
    write('finalization.json', {'errors': failures, 'no_source_staging_or_restoration': True})
    assert not failures, failures


def seal_output():
    files = [{'path': str(p.relative_to(OUT)), 'sha256': digest(p.read_bytes()), 'size': p.stat().st_size}
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact-manifest.json']
    write('artifact-manifest.json', {'files': files})


if __name__ == '__main__':
    phase = sys.argv[1]
    try:
        assert phase in ('prepare', 'qualify', 'finalize')
        result = {'prepare': prepare, 'qualify': qualify, 'finalize': finalize}[phase]()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        write(phase + '-failure.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        if phase == 'finalize':
            seal_output()  # Includes finalization failure diagnostics, if any.
    sys.exit(result or 0)
