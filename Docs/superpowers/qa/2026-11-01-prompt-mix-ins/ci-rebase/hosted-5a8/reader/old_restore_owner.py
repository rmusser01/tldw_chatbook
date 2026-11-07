"""Private RED only: restore the exact pre-fix body at normal module import."""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
OUT = Path('/private/tmp/mixins-reader-restore-final')
NAME = 'tldw_chatbook.UI.Library_Modules.library_media_controller'
PATH = ROOT / 'tldw_chatbook/UI/Library_Modules/library_media_controller.py'
OLD = ast.parse((OUT / 'before-library_media_controller.py').read_text())
OLD_CLASS = next(n for n in OLD.body if isinstance(n, ast.ClassDef) and n.name == 'LibraryMediaController')
OLD_METHOD = next(n for n in OLD_CLASS.body if isinstance(n, ast.FunctionDef) and n.name == '_restore_library_media_focus')

class OldRestoreLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        tree = ast.parse(PATH.read_text(), filename=str(PATH))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LibraryMediaController')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_restore_library_media_focus')
        cls.body[cls.body.index(method)] = OLD_METHOD
        proof = {
            'mode': 'RED only; exact pre-fix method compiled in original module globals',
            'source_path': str(PATH),
            'source_sha256': hashlib.sha256(PATH.read_bytes()).hexdigest(),
            'screen_sha256': hashlib.sha256((ROOT / 'tldw_chatbook/UI/Screens/library_screen.py').read_bytes()).hexdigest(),
            'helper_sha256': hashlib.sha256((ROOT / 'tldw_chatbook/UI/Library_Modules/canvas_sync.py').read_bytes()).hexdigest(),
            'test_sha256': hashlib.sha256((ROOT / 'Tests/UI/test_library_rail_switch_keeps_reader_and_note.py').read_bytes()).hexdigest(),
            'old_method_ast_sha256': hashlib.sha256(ast.dump(OLD_METHOD, include_attributes=False).encode()).hexdigest(),
            'only_method_replaced': method.name,
        }
        (OUT / 'old-owner-overlay-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
        return compile(ast.fix_missing_locations(tree), str(PATH), 'exec')

class OldRestoreFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != NAME:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        spec.loader = OldRestoreLoader(fullname, str(PATH))
        return spec

sys.path.insert(0, str(ROOT))
sys.meta_path.insert(0, OldRestoreFinder())
import pytest
raise SystemExit(pytest.main([
    'Tests/UI/test_library_rail_switch_keeps_reader_and_note.py::test_programmatic_media_row_restore_keeps_reader_until_user_keys',
    '-q', '--tb=long', '-p', 'no:cacheprovider',
    '--basetemp=/private/tmp/mixins-reader-restore-final/final-approved-old-red-tmp',
]))
