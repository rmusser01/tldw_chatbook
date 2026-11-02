"""Private exact-helper self-check; not native event/GUI qualification."""
import ast
from pathlib import Path
from types import SimpleNamespace

PATH = Path.cwd() / 'tldw_chatbook/UI/Library_Modules/canvas_sync.py'
helper = next(n for n in ast.parse(PATH.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_restore_library_media_focus')
namespace = {'LibraryMediaController': object, 'Any': object}
exec(compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])), str(PATH), 'exec'), namespace)
restore = namespace['_restore_library_media_focus']
class Missing(Exception):
    pass
class Target:
    focusable = True
    def has_class(self, name):
        return name == 'grip'
target, foreign = Target(), Target()
deps = {'Widget': Target, 'NoMatches': Missing, 'QueryError': Missing, 'LIBRARY_ADAPTIVE_READER_GRIP_CLASS': 'grip'}
checked = []
for behavior in ('success', 'noop', 'raises', 'foreign-raises', 'already-focused', 'missing', 'disabled', 'entry-channel', 'find-channel', 'none-identity'):
    target.focusable = behavior != 'disabled'
    controller = SimpleNamespace(
        _library_pending_list_entry_focus=behavior == 'entry-channel',
        _library_media_find_focus_pending=behavior == 'find-channel',
        _library_notes_programmatic_focus_target=None,
        focused=target if behavior == 'already-focused' else None,
        fallbacks=0, focus_calls=0,
    )
    def query_one(selector, widget_type):
        assert widget_type is Target
        if behavior == 'missing':
            raise Missing()
        return target
    def set_focus(widget, *, scroll_visible):
        controller.focus_calls += 1
        assert widget is target and scroll_visible is False
        assert controller._library_notes_programmatic_focus_target is target
        if behavior == 'noop':
            return
        if behavior == 'foreign-raises':
            controller._library_notes_programmatic_focus_target = foreign
            raise RuntimeError('foreign marker preserved')
        if behavior == 'raises':
            raise RuntimeError('focus failed')
        controller.focused = widget
    def fallback():
        controller.fallbacks += 1
    controller.query_one, controller.set_focus = query_one, set_focus
    controller._focus_library_list_entry = fallback
    try:
        restore(controller, None if behavior == 'none-identity' else '#target', deps)
    except RuntimeError:
        assert behavior in {'raises', 'foreign-raises'}
    expected_marker = target if behavior == 'success' else foreign if behavior == 'foreign-raises' else None
    assert controller._library_notes_programmatic_focus_target is expected_marker, behavior
    assert controller.focus_calls == int(behavior in {'success', 'noop', 'raises', 'foreign-raises'}), behavior
    assert controller.fallbacks == int(behavior in {'missing', 'disabled'}), behavior
    checked.append(behavior)
print('Exact helper marker lifetime checks passed:', ', '.join(checked))
