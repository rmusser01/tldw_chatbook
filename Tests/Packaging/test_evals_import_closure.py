"""Evals route imports keep optional execution and inspector owners deferred."""

from pathlib import Path

from Tests.Packaging.test_chunking_import_closure import _run_isolated_python


def test_evals_route_defers_execution_and_inspector(tmp_path: Path) -> None:
    result = _run_isolated_python(
        tmp_path,
        """
import sys
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen
assert EvalsScreen.__module__ == "tldw_chatbook.UI.Screens.evals_screen"
for name in (
    "UI.Evals.sample_bench", "UI.Evals.inspector", "UI.Evals.skill_eval_launch",
    "Evals.character_probe.runner", "Evals.word_bench.runner",
    "Evals.skill_eval.judge", "Evals.skill_eval.prompts", "Evals.skill_eval.scoring",
    "Evals.skill_eval.simulation", "Evals.skill_eval.static_analyzer", "Evals.skill_eval.subject",
):
    assert "tldw_chatbook." + name not in sys.modules, name
# Original widget owners and their actual decorator-bound messages remain eager.
for name in ("bench_editor", "character_bench_editor", "library_rail", "results_grid", "skill_eval_panel", "snippet_editor"):
    assert "tldw_chatbook.UI.Evals." + name in sys.modules, name
print("EVALS_DEFERRED_OK")
""",
    )
    assert result.returncode == 0, (result.stdout, result.stderr[-4000:])
    assert "EVALS_DEFERRED_OK" in result.stdout


def test_evals_exports_annotations_and_message_bindings_keep_identity(
    tmp_path: Path,
) -> None:
    result = _run_isolated_python(
        tmp_path,
        """
import importlib
import inspect
import typing
from textual.message import Message
from tldw_chatbook.UI.Screens import evals_screen as screen
from tldw_chatbook.UI.Evals import library_rail as rail
from tldw_chatbook.Evals.skill_eval import runner
assert screen._screen is screen
assert rail._rail is rail
assert runner._runner is runner
# Annotation resolution itself must find the original runner callable alias.
hints = typing.get_type_hints(screen._default_character_probe_chat_factory)
from tldw_chatbook.Evals.character_probe.runner import ChatCallable
assert hints["return"] == ChatCallable
for module in (screen, rail, runner):
    star = {}
    exec("from " + module.__name__ + " import *", star)
    for name, (path, attribute) in module._LAZY_EXPORTS.items():
        owner = importlib.import_module(path, module.__package__)
        expected = getattr(owner, attribute) if attribute else owner
        assert getattr(module, name) is expected
        assert name in dir(module)
        assert name in module.__all__
        assert star[name] is expected
        direct = {}
        exec("from " + module.__name__ + " import " + name, direct)
        assert direct[name] is expected
    for name, method in vars(module).items():
        if inspect.isfunction(method) and method.__module__ == module.__name__:
            typing.get_type_hints(method)
from tldw_chatbook.Evals.skill_eval.subject import SubjectError
assert screen.SubjectError is SubjectError
try:
    screen.subject_from_directory("relative/path")
except screen.SubjectError as exc:
    assert type(exc) is SubjectError
else:
    raise AssertionError("invalid subject must preserve its exception type")
# The same class-bound handlers accept subclasses exactly once, and never
# gain unrelated messages that happen to use the same textual handler name.
instance = screen.EvalsScreen.__new__(screen.EvalsScreen)
for base, handler in ((screen.BenchEditor.Saved, screen.EvalsScreen._on_bench_editor_saved),
                      (screen.SkillEvalPanel.CancelRequested, screen.EvalsScreen._on_skill_eval_cancel_requested)):
    class Derived(base):
        pass
    class Collision(Message):
        pass
    Collision.handler_name = base.handler_name
    for cls, expected_count in ((base, 1), (Derived, 1), (Collision, 0)):
        event = cls.__new__(cls)
        Message.__init__(event)
        methods = [method.__func__ for _, method in instance._get_dispatch_methods(event.handler_name, event)]
        assert methods.count(handler) == expected_count
print("EVALS_CONTRACTS_OK")
""",
    )
    assert result.returncode == 0, (result.stdout, result.stderr[-4000:])
    assert "EVALS_CONTRACTS_OK" in result.stdout
