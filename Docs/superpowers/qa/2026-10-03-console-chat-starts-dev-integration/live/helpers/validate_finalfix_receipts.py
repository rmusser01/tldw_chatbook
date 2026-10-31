from pathlib import Path
import hashlib, json, subprocess

WT = Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
S = WT / '.superpowers/sdd/2026-10-03-console-chat-starts-dev-integration'
checked, auxiliary, errors = [], [], []
for p in sorted(S.glob('finalfix-*.json')):
    raw = p.read_bytes()
    d = json.loads(raw)
    if not isinstance(d, dict):
        auxiliary.append(p.name)
        continue
    output = d.get('stdout', d.get('output'))
    if 'argv' not in d or 'returncode' not in d or not isinstance(output, str):
        auxiliary.append(p.name)
        continue
    if all(k in d for k in ('before_sha256', 'after_sha256', 'ast_equal')):
        assert d['ast_equal'] is True and d['returncode'] == 0, p.name
        assert all(len(d[k]) == 64 for k in ('before_sha256', 'after_sha256')), p.name
        auxiliary.append({'receipt': p.name, 'kind': 'formatter subcommand detail', 'ast_equal': True, 'returncode': 0})
        continue
    log = p.with_suffix('.log')
    if not log.is_file() or log.read_text() != output:
        errors.append(p.name)
        continue
    checked.append({'receipt': p.name, 'returncode': d['returncode'], 'argv': d['argv'], 'receipt_sha256': hashlib.sha256(raw).hexdigest(), 'log_bytes': len(log.read_bytes()), 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
assert not errors, errors
report = S / 'final-review-fix-report.md'
assert report.is_file()
result = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WT, text=True).strip(), 'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(), 'verified_receipt_log_pairs': checked, 'auxiliary_records': auxiliary, 'mismatches': errors, 'qualification': 'Complete byte agreement, not a claim that diagnostic RED command receipts passed.'}
(S / 'controller-finalfix-receipt-validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'head': result['head'], 'verified_pairs': len(checked), 'auxiliary_records': len(auxiliary), 'mismatches': errors}))
