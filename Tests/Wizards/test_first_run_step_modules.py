"""TASK-34100.1 AC#2: each first-run step lives in its own module, still wired.

The steps moved whole out of ``FirstRunSetupWizard.py``. Textual registers
``@on`` handlers per message-pump class when the class is created
(``_MessagePumpMeta``), so a handler survives a move only if it stays on a
``MessagePump`` subclass. A plain mixin would drop it silently (the
TASK-33921 trap). These checks read each module's own source, so a handler
or worker added to a step later is covered without editing this file.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from tldw_chatbook.UI.Wizards import FirstRunSetupWizard as wizard_module
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import SetupStep

_PKG = "tldw_chatbook.UI.Wizards"

#: step class -> the module that owns it.
_STEP_MODULES = {
    "WelcomeStep": "first_run_welcome_step",
    "ProviderStep": "first_run_provider_step",
    "ModelStep": "first_run_model_step",
    "VoiceSetupStep": "first_run_voice_step",
    "RagStep": "first_run_rag_step",
    "SpeechSetupStep": "first_run_speech_step",
    "ToolsStep": "first_run_tools_step",
    "NotesSyncStep": "first_run_notes_step",
    "AppearanceStep": "first_run_appearance_step",
    "ProtectKeysStep": "first_run_protect_step",
    "SummaryStep": "first_run_summary_step",
}

#: PR #2862 moves exactly these, unchanged, into first_run_setup_widgets.py.
_SHARED_WIDGETS = (
    "SetupRadioButton",
    "SetupCheckbox",
    "SetupRadioSet",
    "SetupStepFailure",
    "SetupStep",
    "ProviderChoiceOption",
)


def _decorated_methods(module, class_name: str, decorator: str) -> set[str]:
    """Names of the methods in ``class_name`` decorated with ``@decorator(...)``."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            break
    else:
        raise AssertionError(f"{class_name} is not defined in {module.__name__}")
    found = set()
    for item in node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in item.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            if isinstance(target, ast.Name) and target.id == decorator:
                found.add(item.name)
    return found


@pytest.mark.parametrize("class_name", sorted(_STEP_MODULES))
def test_each_step_class_lives_in_its_own_module(class_name: str) -> None:
    module = importlib.import_module(f"{_PKG}.{_STEP_MODULES[class_name]}")
    cls = getattr(module, class_name)

    assert cls.__module__ == module.__name__
    assert issubclass(cls, SetupStep)
    # The old import path resolves to the very same class object.
    assert getattr(wizard_module, class_name) is cls


@pytest.mark.parametrize("name", _SHARED_WIDGETS)
def test_shared_widgets_live_in_the_widgets_module(name: str) -> None:
    widgets = importlib.import_module(f"{_PKG}.first_run_setup_widgets")

    assert getattr(widgets, name).__module__ == widgets.__name__
    assert getattr(wizard_module, name) is getattr(widgets, name)


def test_the_wizard_module_defines_no_step_class() -> None:
    """New steps go in their own module; the wizard keeps the container."""
    tree = ast.parse(Path(wizard_module.__file__).read_text(encoding="utf-8"))
    defined = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}

    steps_here = sorted(
        name
        for name in defined
        if inspect.isclass(getattr(wizard_module, name, None))
        and issubclass(getattr(wizard_module, name), SetupStep)
    )
    assert steps_here == []


@pytest.mark.parametrize("class_name", sorted(_STEP_MODULES))
def test_every_on_handler_is_registered_on_the_moved_class(class_name: str) -> None:
    module = importlib.import_module(f"{_PKG}.{_STEP_MODULES[class_name]}")
    cls = getattr(module, class_name)
    expected = _decorated_methods(module, class_name, "on")

    registered = {
        handler.__name__
        for handlers in cls.__dict__.get("_decorated_handlers", {}).values()
        for handler, _selectors in handlers
    }
    assert expected <= registered, f"unregistered: {sorted(expected - registered)}"


@pytest.mark.parametrize("class_name", sorted(_STEP_MODULES))
def test_every_worker_method_is_still_a_worker_on_the_moved_class(
    class_name: str,
) -> None:
    module = importlib.import_module(f"{_PKG}.{_STEP_MODULES[class_name]}")
    cls = getattr(module, class_name)
    decorated = _decorated_methods(module, class_name, "work") | _decorated_methods(
        module, class_name, "wizard_work"
    )

    for name in sorted(decorated):
        method = cls.__dict__[name]
        # Both decorators hand back a launcher that wraps the original body.
        assert inspect.unwrap(method) is not method, f"{name} lost its decorator"


#: Every public top-level name FirstRunSetupWizard.py defined before the split
#: (base 1d8fe87659), except its two tunable timeouts (see the test below).
_OLD_PUBLIC_NAMES = (
    "AppearanceStep",
    "EXIT_ROUTE_LIBRARY_NOTES",
    "FirstRunSetupWizard",
    "GENERIC_DISCOVERY_FAILURE_CATEGORY",
    "ModelStep",
    "NotesSyncStep",
    "ProtectKeysStep",
    "ProviderChoiceList",
    "ProviderChoiceOption",
    "ProviderEndpointCandidateList",
    "ProviderEndpointCandidateOption",
    "ProviderStep",
    "REQUIRED_STEP_MANUAL_SETTINGS_CATEGORIES",
    "RagStep",
    "SUMMARY_KEY_HINTS",
    "SetupCheckbox",
    "SetupRadioButton",
    "SetupRadioSet",
    "SetupStep",
    "SetupStepFailure",
    "SetupWizardContainer",
    "SetupWizardNavigation",
    "SetupWizardProgress",
    "SpeechSetupStep",
    "SummaryStep",
    "ToolsStep",
    "WelcomeStep",
    "manual_settings_context_for_required_step",
    "VoiceSetupStep",  # moved earlier, by TASK-33921
)


def test_every_old_public_import_still_resolves() -> None:
    """``from FirstRunSetupWizard import <public name>`` keeps working."""
    missing = [name for name in _OLD_PUBLIC_NAMES if not hasattr(wizard_module, name)]
    assert missing == []


def test_moved_private_helpers_are_not_reexported() -> None:
    """A test patch left on the wizard must fail, not silently miss.

    The two timeouts are public names, but they are tuning knobs that tests
    patch. A re-export would let ``monkeypatch.setattr(wizard, ...)`` succeed
    and change nothing, so they are left out on purpose.
    """
    for name in (
        "MODEL_DISCOVERY_TIMEOUT_SECONDS",
        "CLOUD_PROBE_TIMEOUT_SECONDS",
        "_probe_first_run_provider_connection",
        "run_parakeet_provision",
        "active_managed_parakeet_dir",
    ):
        with pytest.raises(AttributeError):
            getattr(wizard_module, name)
