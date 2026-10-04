"""Verify the frozen fix1 handoff and package it without replaying tests."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook')
SDD = ROOT / '.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge'
HEAD = 'd3443b9e4297fa20897cf99e77a3ae2c8c562b10'
BASE = '72c67b80a870308e2425b5c2ad9437c3cfe01bd8'
ORIGINAL = '40b9aacaa21f7e8c6f2b1989efbb8e8021a1fe77'

spec = importlib.util.spec_from_file_location(
    'handoff_helpers', '/private/tmp/pr2995-task6/verify_task8_handoff.py'
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
git, tree, blobs = helpers.git, helpers.tree, helpers.blobs

assert git('rev-parse', 'HEAD').decode().strip() == HEAD
assert not git('status', '--porcelain')
freeze = json.loads((SDD / 'task-8-fix1-freeze-map.json').read_text())
safe = json.loads((SDD / 'task-8-fix1-safe-evidence-manifest.json').read_text())
final_tree, original_tree = tree(HEAD), tree(ORIGINAL)
selected = freeze['source_and_all_selected_test_helper_maps']
tracked = [row for row in selected if row['path'] in final_tree]
paths = {row['path'] for row in tracked}
paths.update(row['path'] for row in freeze['task7_34_function_carry'])
data = blobs(final_tree[path] for path in paths)
for row in selected:
    path = ROOT / row['path']
    assert path.is_file() and not path.is_symlink()
    content = data[final_tree[row['path']]] if row['path'] in final_tree else path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == row['final']['sha256'], row['path']
    assert len(content) == row['final']['bytes'], row['path']
for row in freeze['task7_34_function_carry']:
    module = ast.parse(data[final_tree[row['path']]])
    stem = row['task7_name'].rsplit('#', 1)[0].rsplit('.', 1)[-1]
    assert any(
        hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        == row['ast_sha256']
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == stem
    ), row['path']
for row in freeze['derived_source_carry']:
    assert row['original_blob'] == row['final_blob'] == final_tree[row['path']] == original_tree[row['path']]
prior = json.loads((SDD / 'task-8-final-preservation.json').read_text())
qa = dict(prior['historical_qa']['blobs'])
assert not (set(qa) & set(prior['incoming_qa']['blobs']))
qa.update(prior['incoming_qa']['blobs'])
assert len(qa) == 11792
assert all(final_tree[path] == original_tree[path] == blob for path, blob in qa.items())
for row in safe['files']:
    path = Path(row['copy'])
    assert path.is_relative_to(SDD / 'task-8-fix1-safe-evidence')
    assert path.is_file() and not path.is_symlink() and not path.parent.is_symlink()
    assert path.suffix in {'.json', '.log', '.xml'}
    content = path.read_bytes()
    assert len(content) == row['bytes'] and hashlib.sha256(content).hexdigest() == row['sha256']
for name in ['task-8-fix1-final-validation.json', 'task-8-fix1-final-callers.json']:
    receipt = json.loads((SDD / name).read_text())
    assert receipt['exit'] == 0 and not receipt['terminated']
    assert receipt['source_before'] == receipt['source_after']
    for path, sha in receipt['source_after'].items():
        content = git('show', f'{HEAD}:{path}') if path in final_tree else (ROOT / path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == sha, (name, path)
changed = ['tldw_chatbook/Chat/console_chat_controller.py', 'Tests/Chat/test_console_chat_create_integration.py']
assert all(git('show', f'{BASE}:{path}') == git('show', f'{ORIGINAL}:{path}') for path in changed)
package = SDD / 'task-8-fix1-review-package-d3443b9e4297.diff'
assert not package.exists()
body = (
    f'# Task8 I1 immutable scoped fix package\nFix base: {BASE}\nOriginal review source: {ORIGINAL}\nFrozen head: {HEAD}\nBoth BASE source/test blobs exactly equal the original reviewed source.\n\n'.encode()
    + git('log', '--oneline', f'{BASE}..{HEAD}')
    + b'\n' + git('diff', '--stat', BASE, HEAD, '--', *changed)
    + b'\n' + git('diff', '-U40', BASE, HEAD, '--', *changed)
)
package.write_bytes(body)
result = {
    'head': HEAD, 'fix_base': BASE, 'original_review': ORIGINAL, 'clean': True,
    'selected_source_test_helper_maps_verified': len(selected), 'tracked_maps_verified': len(tracked),
    'task7_ast_verified': len(freeze['task7_34_function_carry']),
    'derived_blob_carry_verified': len(freeze['derived_source_carry']),
    'historical_qa_blobs_verified': len(qa), 'safe_copies_verified': len(safe['files']),
    'final_behavior_receipts_exact': 2, 'tests_replayed': 0,
    'package': str(package), 'package_bytes': len(body),
    'package_sha256': hashlib.sha256(body).hexdigest(),
    'cap_gate': 'unresolved33027/29367; separate structural repair required',
}
(SDD / 'task-8-fix1-root-handoff-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
