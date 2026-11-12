"""Read-only check of the source-preserving PR3346 rebase onto dev2dcf."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook')
SOURCE = Path('/private/tmp/mixins-4b1-rebase')
OUT = Path('/private/tmp/mixins-4b1-uat-review')
PUBLISHED = '4b1e070c972320e1353b6bac7e185035db125571'
BACKUP = '798222b480b587e2f2248de9fbd0edf3a773b34c'
OLD_DEV = '88af9f937908b46e8c5f9f7fddd9e8748c917d46'
NEW_DEV = '2dcf038cb8be0af991f92c43b0c262f02471c51d'
REBASED = 'e407993a916736ba7175164e716c824379081c54'
GUARD = 'Tests/Architecture/test_module_size_ratchet.py'
QA_HASHES = {
    'Tests/QA/test_project_chat_uat.py': 'bff4182da1994ceb691a644f7c458fd3fc321d64bf58df51be60670eac985681',
    'Tests/QA/test_project_chat_uat_support.py': '8f22db83a9ebf3e3c5f2921894f838b1e745b181f07d6fa77c2f48c4c5a93329',
}

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def budgets(data):
    tree = ast.parse(data)
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == '_BUDGETS')

subprocess.run(['git', 'merge-base', '--is-ancestor', NEW_DEV, REBASED], cwd=ROOT, check=True)
incoming = git('diff', '--name-only', OLD_DEV, NEW_DEV).decode().splitlines()
assert len(incoming) == 14 and [p for p in incoming if not p.startswith('backlog/tasks/')] == [GUARD]
rebase_paths = git('diff', '--name-only', BACKUP, REBASED).decode().splitlines()
assert len(rebase_paths) == 13 and all(p.startswith('backlog/tasks/') for p in rebase_paths)
before = json.loads((SOURCE / 'source-before.json').read_text())
after = json.loads((SOURCE / 'source-after.json').read_text())
assert len(before) == 9684 and before == after
for path, digest in after.items():
    assert sha((ROOT / path).read_bytes()) == digest, path
for path, digest in QA_HASHES.items():
    assert after[path] == digest == sha((ROOT / path).read_bytes()), path
refs = {'published': PUBLISHED, 'backup': BACKUP, 'old_dev': OLD_DEV, 'new_dev': NEW_DEV, 'rebased': REBASED}
raw = {name: git('show', f'{ref}:{GUARD}') for name, ref in refs.items()}
assert raw['published'] == raw['backup'] == raw['rebased'] == (ROOT / GUARD).read_bytes()
rows = {name: budgets(data) for name, data in raw.items()}
raised = {}
for name in ('published', 'backup', 'new_dev'):
    common = rows[name].keys() & rows['rebased'].keys()
    raised[name] = {path: [rows[name][path], rows['rebased'][path]] for path in common if rows['rebased'][path] > rows[name][path]}
    assert not raised[name], raised[name]
repins = [path for path in rows['old_dev'].keys() & rows['new_dev'].keys() if rows['old_dev'][path] != rows['new_dev'][path]]
assert len(repins) == 18
size_log = (SOURCE / 'size-guards.log').read_text()
ci_log = (SOURCE / 'ci-contracts.log').read_text()
assert '201 passed in 2.45s' in size_log
assert '254 passed in 53.80s' in ci_log
proof = {
    'refs': refs,
    'new_dev_is_ancestor_of_rebased_head': True,
    'incoming_changed_paths': incoming,
    'rebase_changed_paths': rebase_paths,
    'source_manifest_paths': len(after),
    'before_and_after_manifest_exact': True,
    'all_current_source_hashes_verified': True,
    'qa_hashes': QA_HASHES,
    'general_guard_sha256': sha(raw['rebased']),
    'guard_exact_to_published_and_backup': True,
    'general_guard_rows': len(rows['rebased']),
    'incoming_dev_repins': len(repins),
    'raised_caps_vs_reference': raised,
    'post_rebase_targeted_results': {'size_guard': '201 passed in 2.45s', 'CI_contracts': '254 passed in 53.80s'},
    'evidence_hashes': {name: sha((SOURCE / name).read_bytes()) for name in ('protocol.json', 'rebase-proof.json', 'source-before.json', 'source-after.json', 'size-guards.log', 'ci-contracts.log')},
    'qualification': 'Read-only source/Git/hash/log inspection only; no reviewer app, test, native or performance execution. Exact earlier native evidence remains qualified to these identical runtime/QA bytes; fresh published-head CI pending.',
}
(OUT / 'rebase-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps({key: value for key, value in proof.items() if key not in {'incoming_changed_paths', 'rebase_changed_paths', 'evidence_hashes'}}, indent=2))
