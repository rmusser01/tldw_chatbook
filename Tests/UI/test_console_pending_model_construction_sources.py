"""AC27: optional display refuses foreign constructors/descriptors/metadata."""

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import drain_active_service_patches, drain_created_dirs
from Tests.UI.test_console_pending_display_owner_controls import _prepared
from Tests.UI.test_console_pending_facts_cold_readiness import _CheckedNavigationFactory
from tldw_chatbook.Chat import console_display_state as model
from tldw_chatbook import Chat as chat_package
from tldw_chatbook.Chat import console_chat_models, console_runtime
from tldw_chatbook.UI.Screens import chat_screen


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@private_profile_test
async def test_pending_model_construction_and_table_sources_decline_before_dispatch(
    tmp_path, request
):
    factory = _CheckedNavigationFactory()
    app = factory.build(tmp_path)
    allocator_installed = False
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            console, _, _, _, inspector, _ = await _prepared(app, pilot)
            calls = []

            def foreign_new(cls, *args, **kwargs):
                calls.append("new")
                return object.__new__(cls)

            def setter(instance, value):
                calls.append("set")
                vars(instance)["value"] = value

            def getter(instance):
                calls.append("get")
                return vars(instance)["value"]

            original_classes = model._CONSOLE_PENDING_MODEL_CLASSES

            class ForeignIterable:
                def __iter__(self):
                    calls.append("iter")
                    return iter(original_classes)

            class ForeignName(str):
                def __hash__(self):
                    calls.append("hash")
                    return super().__hash__()

            class ForeignMeta(type):
                def __eq__(cls, other):
                    calls.append("metaclass_eq")
                    return False

            class ForeignTable(metaclass=ForeignMeta):
                pass

            class ForeignModule:
                def __init__(self, original):
                    self.original = original

                def __getattr__(self, name):
                    calls.append("module_getattr")
                    return getattr(self.original, name)

                @property
                def __dict__(self):
                    calls.append("module_dict")
                    return vars(self.original)

            changes = (
                (model.ConsoleDisplayRow, "value", property(getter, setter)),
                (model, "_CONSOLE_PENDING_MODEL_CLASSES", ForeignIterable()),
                (
                    model,
                    "_CONSOLE_PENDING_MODEL_CLASSES",
                    (
                        (ForeignName(original_classes[0][0]), original_classes[0][1]),
                        *original_classes[1:],
                    ),
                ),
                (model, "_CONSOLE_PENDING_MODEL_CONSTRUCTION", ForeignIterable()),
                (model, "_CONSOLE_PENDING_MODEL_CONSTRUCTION", (object(),)),
                (
                    chat_screen,
                    "_CONSOLE_PENDING_MODEL_TABLES_ORIGINAL",
                    ForeignIterable(),
                ),
                (chat_screen, "_CONSOLE_PENDING_MODEL_TABLES_ORIGINAL", ForeignTable()),
                (chat_screen, "ConsoleInspectorState", type("ForeignState", (), {})),
                (chat_package, "console_display_state", ForeignModule(model)),
                (
                    chat_package,
                    "console_chat_models",
                    ForeignModule(console_chat_models),
                ),
                (chat_package, "console_runtime", ForeignModule(console_runtime)),
            )
            for target, name, changed in changes:
                before = inspector.state
                prior = vars(target).get(name)
                existed = name in vars(target)
                calls.clear()
                try:
                    setattr(target, name, changed)
                    assert chat_screen._console_pending_display_owner(console) is None
                    assert not chat_screen._sync_console_pending_display(console)
                    assert inspector.state is before
                    assert calls == [], name
                finally:
                    if existed:
                        setattr(target, name, prior)
                    else:
                        delattr(target, name)
                assert chat_screen._console_pending_display_owner(console) is not None
                assert chat_screen._sync_console_pending_display(console)
                assert calls == []
            assert factory.current()
            # CPython cannot restore the original tp_new fast path after this
            # class mutation. Keep the declared custom allocator through normal
            # App/runtime teardown; only selected optional-guard calls assert
            # non-dispatch. Full custom fallback remains supported outside it.
            before = inspector.state
            assert chat_screen._console_pending_display_owner(console) is not None
            calls.clear()
            model.ConsoleInspectorState.__new__ = staticmethod(foreign_new)
            allocator_installed = True
            assert chat_screen._console_pending_display_owner(console) is None
            assert not chat_screen._sync_console_pending_display(console)
            assert inspector.state is before and calls == []
    finally:
        try:
            drain_created_dirs()
            drain_active_service_patches()
        finally:
            if allocator_installed:
                del model.ConsoleInspectorState.__new__
