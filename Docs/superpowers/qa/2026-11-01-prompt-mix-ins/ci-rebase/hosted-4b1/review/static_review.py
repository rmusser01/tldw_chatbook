"""Read-only source comparison for the bounded PR3346 UAT readiness follow-up."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook')
OUT = Path('/private/tmp/mixins-4b1-uat-review')
BASE = '4b1e070c972320e1353b6bac7e185035db125571'
UAT = 'Tests/QA/test_project_chat_uat.py'
SUPPORT = 'Tests/QA/test_project_chat_uat_support.py'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def sha(value):
    return hashlib.sha256(value).hexdigest()

def dump(node):
    return ast.dump(node, include_attributes=False)

original = {name: git('show', f'{BASE}:{name}') for name in (UAT, SUPPORT)}
current = {name: (ROOT / name).read_bytes() for name in (UAT, SUPPORT)}
old_uat = ast.parse(original[UAT])
new_uat = ast.parse(current[UAT])
helper = next(node for node in new_uat.body if isinstance(node, ast.FunctionDef) and node.name == 'mounted_approval_rows')
new_uat.body.remove(helper)
old_listing = next(node for node in ast.walk(old_uat) if isinstance(node, ast.FunctionDef) and node.name == 'listing_approval')
new_listing = next(node for node in ast.walk(new_uat) if isinstance(node, ast.FunctionDef) and node.name == 'listing_approval')
assert len(old_listing.body) == len(new_listing.body) == 2
assert dump(old_listing.body[0]) == dump(new_listing.body[0])
assert dump(new_listing.body[1]) == dump(ast.parse('return mounted_approval_rows(console)').body[0])
new_listing.body[1] = old_listing.body[1]
assert dump(old_uat) == dump(new_uat), 'UAT changed beyond helper and delegated return'
expected_helper = ast.parse('''def mounted_approval_rows(console):
    """Rows enter the DOM before their own Compose/Mount publishes children."""
    rows = list(console.query(".approval-row"))
    return rows if all(row.is_mounted for row in rows) else []
''').body[0]
assert dump(helper) == dump(expected_helper), 'Unexpected readiness predicate'
old_support = ast.parse(original[SUPPORT])
new_support = ast.parse(current[SUPPORT])
control = next(node for node in new_support.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'test_approval_readiness_waits_for_children_without_filtering_rows')
new_support.body.remove(control)
assert dump(old_support) == dump(new_support), 'Existing support controls changed'

frozen_paths = [
    'tldw_chatbook', 'Tests/Performance', 'Tests/Architecture', 'Tests/CI',
    'Tests/UI', '.github/workflows', 'scripts',
    'scripts/check_project_chat_uat.py', 'scripts/ui_pr_gate_census.txt',
    'scripts/check_ui_pr_gate_census.py', 'pyproject.toml',
]
changed_frozen = git('diff', '--name-only', BASE, '--', *frozen_paths).decode().splitlines()
assert not changed_frozen, f'Frozen source changed: {changed_frozen}'
changed_paths = git('diff', '--name-only', BASE).decode().splitlines()
changed_non_docs = [path for path in changed_paths if not path.startswith(('Docs/', 'backlog/'))]
assert set(changed_non_docs) == {UAT, SUPPORT}, changed_non_docs
workflow = (ROOT / '.github/workflows/derived-artifacts.yml').read_text()
assert '          Tests/QA/test_project_chat_uat_support.py\n' in workflow
assert '          Tests/QA/test_project_chat_uat.py\n' in workflow
original_assertions = [dump(node) for node in ast.walk(old_uat) if isinstance(node, ast.Assert)]
original_timeouts = [ast.unparse(node) for node in ast.walk(old_uat) if isinstance(node, ast.keyword) and node.arg == 'timeout']
proof = {
    'base': BASE,
    'source': {name: {'base_sha256': sha(original[name]), 'final_sha256': sha(current[name])} for name in (UAT, SUPPORT)},
    'original_uat_module_ast_preserved_except_helper_and_return': True,
    'original_support_module_ast_preserved_except_new_control': True,
    'original_assertion_count': len(original_assertions),
    'original_assertion_sequence_sha256': sha(json.dumps(original_assertions).encode()),
    'original_timeout_keywords': original_timeouts,
    'helper_is_native_mount_gate_only': True,
    'new_control': control.name,
    'new_control_assertions': sum(isinstance(node, ast.Assert) for node in ast.walk(control)),
    'existing_whole_file_derived_enrollment': True,
    'frozen_paths': frozen_paths,
    'changed_frozen_paths': changed_frozen,
    'changed_non_documentation_paths': changed_non_docs,
    'qualification': 'Static read-only comparison; no app, test, native, performance, remote or Git mutation run by reviewer.',
}
(OUT / 'static-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof, indent=2))
