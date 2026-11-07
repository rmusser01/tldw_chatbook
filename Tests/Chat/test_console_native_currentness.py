"""Native currentness keeps ordinary negatives cold and rejects source substitution."""

from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import sys
from contextvars import ContextVar
from types import ModuleType

from tldw_chatbook.Chat import console_chat_controller, console_runtime
from tldw_chatbook.Chat import console_ordinary_tree as owner
import tldw_chatbook.Chat as package

mode = sys.argv[1]
heavy_name = "tldw_chatbook.Chat.console_workflow_preparation"
assert heavy_name not in sys.modules
from tldw_chatbook.Chat import console_agent_sources as agents
assert agents.current_agent_sources() is None
assert heavy_name not in sys.modules, "ordinary negative imported workflow owners"
# Only after the original receiver/cold assertion: old SOURCE must fail at the
# observed heavy closure, rather than at a new implementation-only attribute.
source = console_runtime._ORIGINAL_CURRENTNESS_SOURCE
assert source is console_chat_controller._ORIGINAL_CURRENTNESS_SOURCE
if mode.startswith("late-"):
    # No issuance exists. Reload the small consumer only after the producer
    # changes, so its module-level alias cannot reclassify the new tuple.
    del sys.modules[agents.__name__]
    del package.console_agent_sources
    del agents
if mode == "empty":
    print("retired and reopened")
    raise SystemExit(0)

calls = []
def foreign(*args, **kwargs):
    calls.append((args, kwargs))
    return None

if mode.startswith("heavy-") or mode in {"empty-forget", "nonempty-forget"}:
    from tldw_chatbook.Chat import console_workflow_preparation as heavy
if mode == "carrier":
    owner._WORKFLOW_PREPARATION_CURRENT = ContextVar("foreign", default=None)
elif mode == "reader-alias":
    owner.current_workflow_preparation = foreign
elif mode == "reader-defaults":
    owner.current_workflow_preparation.__defaults__ = (object(),)
elif mode == "dispatcher-kwdefaults":
    owner._read_native_currentness.__kwdefaults__ = {"foreign": object()}
elif mode in {"producer-record", "late-producer-record", "producer-closure"}:
    replacement = foreign
    if mode == "producer-closure":
        def make_reader():
            observed = calls
            def replaced(*args, **kwargs):
                observed.append((args, kwargs))
            return replaced
        replacement = make_reader()
    owner.current_workflow_preparation = replacement
    records = list(source[4])
    records[0] = (records[0][0], owner._function_record(replacement))
    owner._NATIVE_CURRENTNESS_SOURCE = (*source[:4], tuple(records), *source[5:])
    if mode == "producer-closure":
        replacement.__closure__[0].cell_contents = []
    if mode.startswith("late-"):
        from tldw_chatbook.Chat import console_agent_sources as agents
elif mode == "reader-code":
    owner.current_workflow_preparation.__code__ = foreign.__code__
elif mode == "helper-code":
    owner._currentness_workflow_namespace.__code__ = foreign.__code__
elif mode == "checker-code":
    owner._function_unchanged.__code__ = foreign.__code__
elif mode == "dispatcher-code":
    owner._read_native_currentness.__code__ = foreign.__code__
elif mode == "owner-clone":
    clone = ModuleType(owner.__name__)
    vars(clone).update(vars(owner))
    sys.modules[owner.__name__] = clone
    package.console_ordinary_tree = clone
elif mode == "consumer-clone":
    clone = ModuleType(console_runtime.__name__)
    vars(clone).update(vars(console_runtime))
    sys.modules[console_runtime.__name__] = clone
    package.console_runtime = clone
elif mode in {"consumer-property-runtime", "consumer-property-controller"}:
    class ForeignModule:
        @property
        def _ORIGINAL_CURRENTNESS_SOURCE(self):
            calls.append("foreign module property")
            return source
    setattr(package, "console_runtime" if mode.endswith("runtime") else "console_chat_controller", ForeignModule())
elif mode == "consumer-missing-anchor":
    del console_runtime._ORIGINAL_CURRENTNESS_SOURCE
    def module_getattr(name):
        calls.append(name)
        return source
    console_runtime.__getattr__ = module_getattr
elif mode == "consumer-package-clone":
    clone = ModuleType(console_runtime.__name__)
    vars(clone).update(vars(console_runtime))
    package.console_runtime = clone
elif mode == "heavy-alias":
    heavy.current_workflow_preparation = foreign
elif mode == "heavy-carrier":
    heavy._CURRENT = ContextVar("foreign", default=None)
elif mode == "heavy-clone":
    clone = ModuleType(heavy.__name__)
    vars(clone).update(vars(heavy))
    sys.modules[heavy.__name__] = clone
    package.console_workflow_preparation = clone
elif mode == "heavy-one-slot":
    del package.console_workflow_preparation
elif mode in {"empty-forget", "nonempty-forget"}:
    if mode == "nonempty-forget":
        source[2].set(object())
    del sys.modules[heavy_name]
    del package.console_workflow_preparation
elif mode == "forged-carrier":
    source[2].set(object())
elif mode == "platform-clone":
    from tldw_chatbook.Chat import console_workflow_runs as workflow
    clone = ModuleType(workflow.__name__)
    vars(clone).update(vars(workflow))
    sys.modules[workflow.__name__] = clone
    package.console_workflow_runs = clone
else:
    raise AssertionError("unknown control: " + mode)

if mode == "empty-forget":
    # Both absent EMPTY slots intentionally have the never-loaded semantics.
    assert agents.current_agent_sources() is None
else:
    try:
        agents.current_agent_sources()
    except (RuntimeError, PermissionError) as error:
        assert "original" in str(error), str(error)
    else:
        raise AssertionError("substituted currentness reached ordinary fallback: " + mode)
assert not calls, "foreign currentness callback ran before source refusal"
print("retired and reopened")
"""


@pytest.mark.parametrize(
    "mode",
    [
        "empty",
        "carrier",
        "reader-alias",
        "reader-code",
        "helper-code",
        "checker-code",
        "dispatcher-code",
        "reader-defaults",
        "dispatcher-kwdefaults",
        "producer-record",
        "late-producer-record",
        "producer-closure",
        "owner-clone",
        "consumer-clone",
        "consumer-property-runtime",
        "consumer-property-controller",
        "consumer-missing-anchor",
        "consumer-package-clone",
        "heavy-alias",
        "heavy-carrier",
        "heavy-clone",
        "heavy-one-slot",
        "empty-forget",
        "nonempty-forget",
        "forged-carrier",
        "platform-clone",
    ],
)
def test_native_currentness_original_source_and_cold_negative(tmp_path, mode):
    _run(tmp_path, mode, "success", script=_SCRIPT, timeout=45)


def test_native_currentness_preserves_actual_partial_and_issued_planning(tmp_path):
    # Observe actual original issuance; never manufacture a preparation or owner.
    script = Path(__file__).with_name("workflow_planning_probe.py").read_text()
    prefix = r"""
import sys
import threading
_currentness_before_source = []
def _observe_currentness(frame, event, argument):
    if (
        event == "call"
        and frame.f_code.co_name == "_configure"
        and frame.f_globals.get("__name__") == "tldw_chatbook.Chat.console_workflow_preparation"
    ):
        from tldw_chatbook.Chat import console_ordinary_tree as currentness
        prepared = frame.f_locals["self"]
        if prepared.source is None and currentness._WORKFLOW_PREPARATION_CURRENT.get() is prepared:
            assert prepared._configured_source_observation is None
            assert currentness.current_workflow_preparation() is prepared
            assert currentness.current_workflow_preparation(prepared.controller) is prepared
            assert currentness.current_workflow_preparation(object()) is None
            _currentness_before_source.append(prepared)
sys.setprofile(_observe_currentness)
threading.setprofile(_observe_currentness)
"""
    needle = """        owner = current_execution_owner()
        assert owner is not None and owner.is_admitted and not owner.child
"""
    assert script.count(needle) == 1
    script = script.replace(
        needle,
        needle
        + """        from tldw_chatbook.Chat.console_ordinary_tree import (
            current_workflow_preparation, current_planning_preparation,
        )
        actual = current_workflow_preparation(controller)
        assert actual is not None and actual is prepared
        assert current_planning_preparation(controller) is actual
        assert current_workflow_preparation(object()) is None
        foreign_getters = []
        class MovedController:
            @property
            def _hooks_v2_runtime(self):
                foreign_getters.append(True)
                return actual.runtime
        original_controller = actual.controller
        actual.controller = MovedController()
        try:
            try:
                current_workflow_preparation()
            except (PermissionError, RuntimeError):
                pass
            else:
                raise AssertionError("moved receiver reached metadata fallback")
        finally:
            actual.controller = original_controller
        assert not foreign_getters, "foreign owner property ran before refusal"
        assert current_workflow_preparation(controller) is actual
        original_runtime = actual.runtime
        for field in ("controller", "runtime"):
            cls = type(actual)
            assert field not in vars(cls)
            def field_getter(self):
                foreign_getters.append(field)
                return original_controller if field == "controller" else original_runtime
            setattr(cls, field, property(field_getter))
            try:
                try:
                    current_workflow_preparation()
                except (PermissionError, RuntimeError):
                    pass
                else:
                    raise AssertionError("grafted prepared descriptor reached metadata fallback")
            finally:
                delattr(cls, field)
            assert not foreign_getters, "foreign prepared property ran before refusal"
            assert current_workflow_preparation(controller) is actual
""",
    )
    script = (
        prefix
        + script
        + "\nassert _currentness_before_source, 'original pre-source metadata state was not exercised'\n"
    )
    _run(tmp_path, "primary", "success", script=script, timeout=60)
