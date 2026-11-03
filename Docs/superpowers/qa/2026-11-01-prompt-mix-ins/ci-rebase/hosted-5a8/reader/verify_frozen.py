"""Static proof for the final reader-restore changes; no application/test run."""
import ast
import collections
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path('/private/tmp/mixins-reader-restore-final')
BASE = '5a8f23eb9dfabeda25d46ed168d8ac675f483217'
CP = Path('tldw_chatbook/UI/Library_Modules/library_media_controller.py')
SP = Path('tldw_chatbook/UI/Library_Modules/canvas_sync.py')
TP = Path('Tests/UI/test_library_rail_switch_keeps_reader_and_note.py')
LP = Path('tldw_chatbook/UI/Screens/library_screen.py')

def published(path):
    return subprocess.check_output(['git', 'show', f'{BASE}:{path}'], text=True)

def method(tree, clsname, name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == clsname)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)

def dump(node):
    return ast.dump(node, include_attributes=False)

old = method(ast.parse(published(CP)), 'LibraryMediaController', '_restore_library_media_focus')
new = method(ast.parse(CP.read_text()), 'LibraryMediaController', '_restore_library_media_focus')
helper = next(n for n in ast.parse(SP.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_restore_library_media_focus')
body = copy.deepcopy(helper.body[1:])
assert ast.unparse(body[-3].test) == 'target is focused'
assert ast.unparse(body[-2]) == 'controller._library_notes_programmatic_focus_target = target'
assert isinstance(body[-1], ast.Try)
body = body[:-3] + body[-1].body

class UndoMove(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'controller':
            return ast.copy_location(ast.Name(id='self', ctx=node.ctx), node)
        return node
    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'dependencies':
            assert isinstance(node.slice, ast.Constant)
            assert node.slice.value in {'Widget', 'NoMatches', 'QueryError', 'LIBRARY_ADAPTIVE_READER_GRIP_CLASS'}
            return ast.copy_location(ast.Name(id=node.slice.value, ctx=node.ctx), node)
        return self.generic_visit(node)

reversed_body = ast.Module(body=[UndoMove().visit(n) for n in body], type_ignores=[])
assert dump(reversed_body) == dump(ast.Module(body=old.body[1:], type_ignores=[]))
assert dump(old.args) == dump(new.args)
assert ast.get_docstring(old) == ast.get_docstring(new)
assert CP.read_text().count('return _restore_library_media_focus(self, previous, globals())') == 1
original_test = published(TP)
assert TP.read_text().startswith(original_test)
original_nodes = {}
for node in ast.parse(original_test).body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        current = next(n for n in ast.parse(TP.read_text()).body if type(n) is type(node) and n.name == node.name)
        assert dump(node) == dump(current)
        original_nodes[node.name] = True
old_screen = published(LP)
old_call = '\n'.join([
    '                self._library_media_browse_controller.mutation_refresh_scope,',
    '                focus_identity="#library-media-row-0",',
    '            )',
    '            self._request_library_media_facets()',
    '            # task-28245',
])
assert old_screen.count(old_call) == 1
expected_screen = old_screen.replace(old_call, old_call.replace('focus_identity="#library-media-row-0",', 'focus_identity=None if keep_media_reader else "#library-media-row-0",'))
assert LP.read_text() == expected_screen

# Format the exact new helper/test only; the original files contain pre-existing debt.
formatted_blocks = {}
for label, source, name in [
    ('new_helper', SP.read_text(), '_restore_library_media_focus'),
    ('new_native_test', TP.read_text(), 'test_programmatic_media_row_restore_keeps_reader_until_user_keys'),
    ('new_marker_test', TP.read_text(), 'test_programmatic_media_restore_releases_only_its_failed_focus_marker'),
]:
    node = next(n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    start = min([node.lineno] + [n.lineno for n in node.decorator_list])
    block = '\n'.join(source.splitlines()[start-1:node.end_lineno]) + '\n'
    result = subprocess.run([sys.executable, '-m', 'ruff', 'format', '--stdin-filename', str(SP if label == 'new_helper' else TP), '-'], input=block, text=True, capture_output=True, check=True)
    assert result.stdout == block, label
    formatted_blocks[label] = True

# Compare lint with the exact published baseline, ignoring relocation line numbers.
old_lint = []
for path in (CP, SP, LP, TP):
    result = subprocess.run([sys.executable, '-m', 'ruff', 'check', '--stdin-filename', str(path), '--output-format', 'json', '-'], input=published(path), text=True, capture_output=True)
    assert result.returncode in {0, 1}
    old_lint.extend(json.loads(result.stdout))
(OUT / 'lint-published-baseline.json').write_text(json.dumps(old_lint, indent=2) + '\n')
new_lint = json.loads((OUT / 'lint-after.json').read_text())
key = lambda row: (row['filename'], row['code'], row['message'])
old_counts = collections.Counter(map(key, old_lint))
new_counts = collections.Counter(map(key, new_lint))
added = [(list(key), value) for key, value in (new_counts-old_counts).items()]
removed = [(list(key), value) for key, value in (old_counts-new_counts).items()]
assert not added, added
lint = {'published_total': len(old_lint), 'final_total': len(new_lint), 'added': added, 'removed': removed}
(OUT / 'final-lint-delta.json').write_text(json.dumps(lint, indent=2) + '\n')
proof = {
    'published_base': BASE,
    'supersedes': ['move-and-original-test-proof.json', 'lint-delta.json'],
    'move_reverse_ast_equal': True,
    'facade_signature_equal': True,
    'facade_docstring_equal': True,
    'live_dependency_keys': ['Widget', 'NoMatches', 'QueryError', 'LIBRARY_ADAPTIVE_READER_GRIP_CLASS'],
    'all_original_test_and_helper_ASTs_equal': original_nodes,
    'entire_published_test_source_is_unchanged_prefix': True,
    'library_screen_has_only_approved_retained_reader_conditional': True,
    'new_blocks_ruff_formatted': formatted_blocks,
    'lint': lint,
    'source': {str(path): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'lines': len(path.read_text().splitlines())} for path in (CP, SP, TP, LP)},
}
(OUT / 'final-frozen-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof, indent=2))
