# compact_model_bar.py
# Description: Compact inline model selector bar shown above the chat log.
# Provides quick access to Provider, Model, Temperature without opening the sidebar.
#
# Imports
import asyncio
import inspect
import sys
import threading
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import (
    CodeType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MemberDescriptorType,
    ModuleType,
)
from typing import TYPE_CHECKING, Any, Callable
from weakref import ReferenceType

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message_pump import MessagePump
from textual.widgets import Button, Select, Input, Label
from textual import on
from textual.css.query import NoMatches

from .. import config as _config
from ..Chat.console_preparation_reads import (
    drain_preparation_reads,
    run_preparation_read,
)
from ..config import get_cli_providers_and_models, resolve_provider_name

if TYPE_CHECKING:
    from ..app import TldwCli

logger = logger.bind(module="CompactModelBar")

#######################################################################################################################


_MISSING = object()
_CONFIG_SOURCE = _config.__dict__.get("_COMPACT_MODEL_CONFIG_SOURCE")


class _CompactSetupChanged(RuntimeError):
    """A selected optional display source no longer owns publication."""


def _function_current(record: tuple) -> bool:
    if type(record) is not tuple or len(record) != 11:
        return False
    (
        namespace,
        name,
        function,
        code,
        defining,
        defaults,
        kwdefaults,
        items,
        closure,
        cells,
        wrapped,
    ) = record
    if (
        type(namespace) not in (dict, MappingProxyType)
        or type(name) is not str  # noqa: E721 -- plain capsule key
        or type(defining) is not dict  # noqa: E721 -- exact namespace
        or type(items) is not tuple
        or type(cells) is not tuple
    ):  # noqa: E721 -- malformed optional capsule declines
        return False
    if type(function) is not FunctionType or type(vars(function)) is not dict:  # noqa: E721 -- no custom function mapping
        return False
    if function.__closure__ is not closure or (
        kwdefaults is not None and type(kwdefaults) is not dict  # noqa: E721 -- no custom defaults mapping
    ):  # noqa: E721 -- real callable metadata only
        return False
    if (
        len(cells) != len(closure or ())
        or any(
            type(row) is not tuple or len(row) != 2 or row[0] is not cell
            for row, cell in zip(cells, closure or ())
        )
        or any(
            type(row) is not tuple or len(row) != 2 or type(row[0]) is not str  # noqa: E721 -- plain keyword
            for row in items
        )
    ):
        return False
    try:
        return (
            namespace.get(name, _MISSING) is function
            and type(function) is FunctionType
            and function.__code__ is code
            and function.__globals__ is defining
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            and len(function.__kwdefaults__ or {}) == len(items)
            and all(
                (function.__kwdefaults__ or {}).get(key) is value
                for key, value in items
            )
            and function.__closure__ is closure
            and all(cell.cell_contents is value for cell, value in cells)
            and vars(function).get("__wrapped__") is wrapped
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _source_current(source: tuple) -> bool:
    if type(source) is not tuple or len(source) != 6:
        return False
    try:
        defining, path, spec, origin, bindings, records = source
        if (
            type(defining) is not dict  # noqa: E721 -- exact namespace
            or type(bindings) is not tuple
            or type(records) is not tuple
        ):  # noqa: E721 -- no custom capsule dispatch
            return False
        if any(
            type(row) is not tuple
            or len(row) != 3
            or type(row[0]) not in (dict, MappingProxyType)
            or type(row[1]) is not str  # noqa: E721 -- plain capsule key
            for row in bindings
        ):
            return False
        module = sys.modules.get(defining.get("__name__"))
        return (
            type(module) is ModuleType
            and vars(module) is defining
            and defining.get("__file__") == path
            and defining.get("__spec__") is spec
            and getattr(spec, "origin", None) == origin
            and all(
                namespace.get(name, _MISSING) is value
                for namespace, name, value in bindings
            )
            and all(_function_current(record) for record in records)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _parent_code_shape(code: CodeType) -> tuple:
    return (
        code.co_code,
        tuple(
            _parent_code_shape(value)
            if type(value) is CodeType
            else (type(value), value)
            if type(value) is str or value is None  # noqa: E721 -- identity only, no custom metaclass equality
            else _MISSING
            for value in code.co_consts
        ),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_name,
        code.co_qualname,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_nlocals,
        code.co_flags,
        code.co_stacksize,
        code.co_exceptiontable,
    )


def _capture_parent_source() -> tuple | None:
    # Consumer-time metadata alone must not bless a preinstalled body change.
    # Compile the original defining file without executing it, then match the
    # getter and setter separately (they deliberately share one qualname).
    try:
        module = sys.modules.get("textual.message_pump")
        if type(module) is not ModuleType:
            return None
        namespace = vars(module)
        path = namespace.get("__file__")
        spec = namespace.get("__spec__")
        origin = getattr(spec, "origin", None)
        descriptor = inspect.getattr_static(MessagePump, "_parent", _MISSING)
        if (
            type(namespace) is not dict  # noqa: E721 -- no custom module namespace
            or namespace.get("MessagePump") is not MessagePump
            or type(path) is not str  # noqa: E721 -- exact source path
            or origin != path
            or type(descriptor) is not property
            or descriptor.fdel is not None
        ):
            return None
        getter, setter = descriptor.fget, descriptor.fset
        if any(
            type(function) is not FunctionType
            or function.__globals__ is not namespace
            or function.__defaults__ is not None
            or function.__kwdefaults__ is not None
            or function.__closure__ is not None
            or type(vars(function)) is not dict  # noqa: E721 -- no custom metadata dispatch
            or vars(function).get("__wrapped__") is not None
            for function in (getter, setter)
        ):
            return None
        compiled = compile(Path(path).read_bytes(), path, "exec", dont_inherit=True)
        classes = tuple(
            value
            for value in compiled.co_consts
            if type(value) is CodeType and value.co_qualname == "MessagePump"
        )
        if len(classes) != 1:
            return None
        declared = tuple(
            value
            for value in classes[0].co_consts
            if type(value) is CodeType and value.co_qualname == "MessagePump._parent"
        )
        if len(declared) != 2:
            return None
        get_codes = tuple(value for value in declared if value.co_argcount == 1)
        set_codes = tuple(value for value in declared if value.co_argcount == 2)
        if (
            len(get_codes) != 1
            or len(set_codes) != 1
            or _parent_code_shape(getter.__code__) != _parent_code_shape(get_codes[0])
            or _parent_code_shape(setter.__code__) != _parent_code_shape(set_codes[0])
            or namespace.get("ref") is not ReferenceType
        ):
            return None
        return (
            namespace,
            path,
            spec,
            origin,
            MessagePump,
            descriptor,
            getter,
            getter.__code__,
            setter,
            setter.__code__,
            ReferenceType,
        )
    except (OSError, SyntaxError, AttributeError, TypeError, ValueError):
        return None


_PARENT_SOURCE = _capture_parent_source()


def _parent_source_current() -> bool:
    source = _PARENT_SOURCE
    if type(source) is not tuple or len(source) != 11:
        return False
    try:
        (
            namespace,
            path,
            spec,
            origin,
            owner,
            descriptor,
            getter,
            get_code,
            setter,
            set_code,
            reference_type,
        ) = source
        module = sys.modules.get("textual.message_pump")
        return (
            type(namespace) is dict  # noqa: E721 -- exact defining namespace
            and type(module) is ModuleType
            and vars(module) is namespace
            and namespace.get("MessagePump") is owner is MessagePump
            and namespace.get("__file__") == path
            and namespace.get("__spec__") is spec
            and getattr(spec, "origin", None) == origin
            and namespace.get("ref") is reference_type is ReferenceType
            and type(descriptor) is property
            and inspect.getattr_static(owner, "_parent", _MISSING) is descriptor
            and inspect.getattr_static(CompactModelBar, "_parent", _MISSING)
            is descriptor
            and descriptor.fget is getter
            and descriptor.fset is setter
            and descriptor.fdel is None
            and all(
                type(function) is FunctionType
                and function.__code__ is code
                and function.__globals__ is namespace
                and function.__defaults__ is None
                and function.__kwdefaults__ is None
                and function.__closure__ is None
                and type(vars(function)) is dict  # noqa: E721 -- exact metadata mapping
                and vars(function).get("__wrapped__") is None
                for function, code in ((getter, get_code), (setter, set_code))
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _parent_snapshot(fields: dict) -> tuple | None:
    if not _parent_source_current() or type(fields) is not dict:  # noqa: E721 -- no custom backing mapping
        return None
    reference = fields.get("_MessagePump__parent")
    if type(reference) is not ReferenceType:
        return None
    parent = ReferenceType.__call__(reference)
    return None if parent is None else (reference, parent)


def _sources_current() -> bool:
    # Qualify the proof functions themselves without invoking a replaced body.
    # The full capsules below can then safely inspect all remaining bindings.
    for name, function, code in _PROOF_HELPERS:
        if (
            globals().get(name) is not function
            or type(function) is not FunctionType
            or function.__code__ is not code
            or function.__globals__ is not globals()
            or function.__defaults__ is not None
            or function.__kwdefaults__ is not None
            or function.__closure__ is not None
            or vars(function).get("__wrapped__") is not None
        ):
            return False
    return (
        _config.__dict__.get("_COMPACT_MODEL_CONFIG_SOURCE") is _CONFIG_SOURCE
        and _source_current(_CONFIG_SOURCE)
        and _source_current(_WIDGET_SOURCE)
        and _parent_source_current()
        and inspect.getattr_static(_CompactSetup, "__getattribute__")
        is object.__getattribute__
        and inspect.getattr_static(_CompactSetup, "__getattr__", _MISSING) is _MISSING
    )


def _plain_fields(receiver: object, owner: type, names: tuple[str, ...]) -> dict | None:
    # Qualify lookup before reading __dict__; custom descriptors stay on the
    # preceding direct route and are never invoked for optional selection.
    if (
        type(receiver) is not owner
        or inspect.getattr_static(owner, "__getattribute__")
        is not object.__getattribute__
        or inspect.getattr_static(owner, "__getattr__", _MISSING) is not _MISSING
        or any(
            inspect.getattr_static(owner, name, _MISSING) is not _MISSING
            for name in names
        )
        or type(inspect.getattr_static(owner, "__dict__", _MISSING))
        not in (GetSetDescriptorType, MemberDescriptorType)
    ):
        return None
    fields = vars(receiver)
    return fields if type(fields) is dict else None  # noqa: E721 -- no custom mapping dispatch


def _setup_inputs(mapping: dict) -> tuple | None:
    # This is only initial display data, not a disk/posture or permission proof.
    provider_input = mapping.get("providers", _MISSING)
    default_input = mapping.get("chat_defaults", _MISSING)
    providers = {} if provider_input is _MISSING else provider_input
    defaults = {} if default_input is _MISSING else default_input
    if type(providers) is not dict or type(defaults) is not dict:  # noqa: E721 -- decline custom coercion
        return None
    if any(
        type(key) is not str  # noqa: E721 -- no custom string
        or type(models) is not list  # noqa: E721 -- no custom list
        or any(type(model) is not str for model in models)  # noqa: E721 -- plain model IDs
        for key, models in providers.items()
    ):  # noqa: E721 -- exact plain options
        return None
    values = tuple(
        defaults.get(name, fallback)
        for name, fallback in (("provider", ""), ("model", ""), ("temperature", 0.7))
    )
    if any(type(value) not in (str, int, float, bool, type(None)) for value in values):
        return None
    return (
        provider_input,
        default_input,
        tuple((key, tuple(models)) for key, models in providers.items()),
        values,
    )


@dataclass(eq=False)
class _CompactSetup:
    widget: Any
    app: Any
    app_type: type
    app_module: ModuleType
    app_namespace: dict
    app_spec: Any
    app_path: str
    app_origin: str
    widget_fields: dict
    app_fields: dict
    mapping: dict
    inputs: tuple
    identity: tuple
    parent_reference: ReferenceType
    parent: Any
    loop: Any
    thread: Any
    reader: Callable
    runtime: Any
    runtime_fields: dict
    runtime_reads: set
    reads: set = field(default_factory=set)
    task: Any = None
    callback: Any = None
    finished: bool = False
    cancelled: bool = False


def _setup_owner_current(request: _CompactSetup) -> bool:
    if not _sources_current() or request.cancelled:
        return False
    module = request.app_module
    if (
        sys.modules.get("tldw_chatbook.app") is not module
        or vars(module) is not request.app_namespace
        or request.app_namespace.get("TldwCli") is not request.app_type
        or request.app_namespace.get("__file__") != request.app_path
        or request.app_namespace.get("__spec__") is not request.app_spec
        or getattr(request.app_spec, "origin", None) != request.app_origin
    ):
        return False
    fields = _plain_fields(
        request.widget,
        CompactModelBar,
        (
            "app_instance",
            "_compact_setup_request",
            "_MessagePump__parent",
            "_closing",
            "_closed",
            "_pruning",
        ),
    )
    app_fields = _plain_fields(
        request.app,
        request.app_type,
        (
            "app_config",
            "console_runtime",
            "_exit",
            "_shutting_down",
            "_console_runtime_shutdown_task",
        ),
    )
    runtime_fields = _runtime_fields(request.runtime)
    current = _setup_inputs(request.mapping)
    parent = _parent_snapshot(fields)
    return (
        fields is request.widget_fields
        and app_fields is request.app_fields
        and runtime_fields is request.runtime_fields
        and runtime_fields is not None
        and runtime_fields.get("_disposed") is False
        and runtime_fields.get("_app") is request.app
        and runtime_fields.get("_preparation_reads") is request.runtime_reads
        and app_fields.get("console_runtime") is request.runtime
        and app_fields.get("_exit") is False
        and app_fields.get("_shutting_down") is False
        and app_fields.get("_console_runtime_shutdown_task") is None
        and fields.get("_compact_setup_request") is request
        and fields.get("app_instance") is request.app
        and parent is not None
        and parent[0] is request.parent_reference
        and parent[1] is request.parent
        and fields.get("_closing") is False
        and fields.get("_closed") is False
        and fields.get("_pruning") is False
        and app_fields.get("app_config") is request.mapping
        and current is not None
        and current[0] is request.inputs[0]
        and current[1] is request.inputs[1]
        and current[2:] == request.inputs[2:]
        and request.reader
        is get_cli_providers_and_models
        is _config.get_cli_providers_and_models
        and _config.current_config_identity() == request.identity
    )


def _read_compact_setup(request: _CompactSetup) -> dict:
    if not _setup_owner_current(request):
        raise _CompactSetupChanged("compact_model_setup_source_changed")
    result = request.reader()
    if not _setup_owner_current(request):
        raise _CompactSetupChanged("compact_model_setup_source_changed")
    if type(result) is not dict or any(  # noqa: E721 -- exact result
        type(provider) is not str  # noqa: E721 -- no custom string
        or type(models) is not list  # noqa: E721 -- no custom list
        or any(type(model) is not str for model in models)  # noqa: E721 -- plain model IDs
        for provider, models in result.items()
    ):  # noqa: E721 -- no custom worker-result coercion
        raise _CompactSetupChanged("compact_model_setup_source_changed")
    # Do not retain the loader's mutable provider lists across publication.
    detached = {provider: list(models) for provider, models in result.items()}
    if not _setup_owner_current(request):
        raise _CompactSetupChanged("compact_model_setup_source_changed")
    return detached


def _capture_setup(widget: Any) -> _CompactSetup | None:
    if not _sources_current():
        return None
    fields = _plain_fields(
        widget,
        CompactModelBar,
        (
            "app_instance",
            "_compact_setup_request",
            "_MessagePump__parent",
            "_closing",
            "_closed",
            "_pruning",
        ),
    )
    module = sys.modules.get("tldw_chatbook.app")
    if fields is None or type(module) is not ModuleType:
        return None
    namespace = vars(module)
    if type(namespace) is not dict:  # noqa: E721 -- no custom module mapping
        return None
    app_type = namespace.get("TldwCli")
    app = fields.get("app_instance")
    if not isinstance(app_type, type):
        return None
    app_fields = _plain_fields(
        app,
        app_type,
        (
            "app_config",
            "console_runtime",
            "_exit",
            "_shutting_down",
            "_console_runtime_shutdown_task",
        ),
    )
    if app_fields is None:
        return None
    runtime = app_fields.get("console_runtime")
    runtime_fields = _runtime_fields(runtime)
    if (
        app_fields.get("_exit") is True
        or app_fields.get("_shutting_down") is True
        or app_fields.get("_console_runtime_shutdown_task") is not None
        or (runtime_fields is not None and runtime_fields.get("_disposed") is True)
    ):
        raise _CompactSetupChanged("compact_model_setup_owner_closing")
    if (
        runtime_fields is None
        or runtime_fields.get("_disposed") is not False
        or runtime_fields.get("_app") is not app
        or type(runtime_fields.get("_preparation_reads")) is not set  # noqa: E721 -- exact observer ownership
        or app_fields.get("_exit") is not False
    ):
        return None
    mapping = app_fields.get("app_config")
    # App input is display data; the original reader owns source/native proof.
    if type(mapping) is not dict:  # noqa: E721 -- exact plain display mapping
        return None
    inputs = _setup_inputs(mapping)
    if inputs is None:
        return None
    parent = _parent_snapshot(fields)
    if parent is None:
        return None
    try:
        loop = asyncio.get_running_loop()
        if asyncio.current_task() is None or widget.app is not app:
            return None
        identity = _config.current_config_identity()
    except RuntimeError:
        return None
    spec = namespace.get("__spec__")
    return _CompactSetup(
        widget,
        app,
        app_type,
        module,
        namespace,
        spec,
        namespace.get("__file__"),
        getattr(spec, "origin", None),
        fields,
        app_fields,
        mapping,
        inputs,
        identity,
        parent[0],
        parent[1],
        loop,
        threading.current_thread(),
        get_cli_providers_and_models,
        runtime,
        runtime_fields,
        runtime_fields["_preparation_reads"],
    )


def _runtime_fields(runtime: Any) -> dict | None:
    module = sys.modules.get("tldw_chatbook.Chat.console_runtime")
    if type(module) is not ModuleType:
        return None
    owner = vars(module).get("ConsoleRuntime")
    if not isinstance(owner, type):
        return None
    return _plain_fields(runtime, owner, ("_disposed", "_preparation_reads", "_app"))


def _require_setup_current(request: _CompactSetup) -> None:
    if not _setup_owner_current(request):
        raise _CompactSetupChanged("compact_model_setup_source_changed")


def _request_default_sync(widget: Any) -> None:
    # Population emits no user intent. Retain the original explicit host sync.
    screen = widget.screen
    request_sync = getattr(screen, "_request_console_control_bar_sync", None)
    if callable(request_sync):
        request_sync()


class CompactModelBar(Horizontal):
    """Compact inline bar showing Provider, Model, Temperature and a sidebar toggle.

    Uses unique IDs (compact-api-provider, compact-api-model) to avoid collision
    with sidebar widgets (chat-api-provider, chat-api-model).
    """

    def __init__(
        self,
        app_instance: "TldwCli",
        on_sidebar_toggle_requested: Callable[[], Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.on_sidebar_toggle_requested = on_sidebar_toggle_requested
        self._compact_setup_request = None
        self._compact_setup_refused = False

    def compose(self) -> ComposeResult:
        """Compose the compact model bar."""
        config = self.app_instance.app_config
        defaults = config.get("chat_defaults", {})
        request = self._compact_setup_request
        # A selected live callback never falls back to a replacement read.
        if request is None or request.finished:
            try:
                request = _capture_setup(self)
            except _CompactSetupChanged:
                self._compact_setup_refused = True
                request = None
            if request is not None:
                self._compact_setup_request = request
        if self._compact_setup_refused:
            providers_models = {}
        elif request is not None:
            providers_models = {
                name: list(models) for name, models in request.inputs[2]
            }
        else:
            providers_models = get_cli_providers_and_models()
        available_providers = list(providers_models.keys())
        # task-16474: no arbitrary first-provider fallback. When
        # chat_defaults.provider is missing or unresolvable the select stays
        # on its prompt until the user chooses -- the first [providers] key
        # in file order is not a selection anyone made.
        default_provider = resolve_provider_name(
            defaults.get("provider", ""),
            providers_models,
        )

        # Provider select. allow_blank=True (task-16474): Textual's Select
        # force-picks options[0] at mount and on set_options when blank is
        # disallowed, which both fabricates a selection nobody made and fires
        # a Changed event the provider mirror would treat as user intent.
        # Blank now means "nothing chosen" and the screen handler already
        # ignores empty values.
        provider_options = [(p, p) for p in available_providers]
        yield Select(
            options=provider_options,
            prompt="Provider",
            allow_blank=True,
            id="compact-api-provider",
        )

        # Model select
        initial_models = providers_models.get(default_provider, [])
        model_options = [(m, m) for m in initial_models]
        yield Select(
            options=model_options,
            prompt="Model",
            allow_blank=True,
            id="compact-api-model",
        )

        # Temperature input
        yield Label("Temp:", classes="compact-bar-label")
        yield Input(
            placeholder="0.7",
            id="compact-temperature",
            value=str(defaults.get("temperature", 0.7)),
            classes="compact-bar-temp",
        )

        # Sidebar toggle button
        yield Button(
            "⚙",
            id="compact-sidebar-toggle",
            classes="compact-bar-toggle",
            tooltip="Toggle settings sidebar (Ctrl+[)",
        )

    async def on_mount(self) -> None:
        """Populate this exact bar after its finite checked settings read."""
        if self._compact_setup_refused:
            return
        config = self.app_instance.app_config
        defaults = config.get("chat_defaults", {})
        request = self._compact_setup_request
        if request is None:
            # Preinstalled/custom readers retain their original direct ABI.
            providers_models = get_cli_providers_and_models()
            _POPULATE_DEFAULTS(self, defaults, providers_models, refresh_options=False)
            _request_default_sync(self)
            return
        # Compose seeds display data before Textual mounts this bar. A normal
        # startup config publication may finish in between; own the checked
        # read from a fresh complete mount-time capture, never a rebased token.
        if (
            type(request) is not _CompactSetup
            or request.widget is not self
            or request.finished
            or request.cancelled
            or request.task is not None
            or request.callback is not None
            or request.reads
            or not _sources_current()
        ):
            return
        try:
            current = _capture_setup(self)
        except _CompactSetupChanged:
            self._compact_setup_refused = True
            current = None
        if self._compact_setup_request is not request:
            return
        request.finished = True
        self._compact_setup_request = current
        if current is None:
            return
        request = current
        config = request.mapping
        defaults = config.get("chat_defaults", {})
        try:
            if not _setup_owner_current(request):
                raise _CompactSetupChanged("compact_model_setup_source_changed")
            request.task = asyncio.current_task()
            request.callback = partial(_read_compact_setup, request)
            provider_select = self.query_one("#compact-api-provider", Select)
            model_select = self.query_one("#compact-api-model", Select)
            temperature = self.query_one("#compact-temperature", Input)
            selection = provider_select.value, model_select.value, temperature.value
            providers_models = await run_preparation_read(
                request.callback,
                creator=self,
                session_id=None,
                reads=request.reads,
                observers=(request.runtime_reads,),
                require_current=partial(_require_setup_current, request),
                source=request,
            )
            if (
                not _setup_owner_current(request)
                or asyncio.get_running_loop() is not request.loop
                or threading.current_thread() is not request.thread
                or self.app is not request.app
                or self.parent is not request.parent
                or self.query_one("#compact-api-provider", Select)
                is not provider_select
                or self.query_one("#compact-api-model", Select) is not model_select
                or self.query_one("#compact-temperature", Input) is not temperature
                or (provider_select.value, model_select.value, temperature.value)
                != selection
            ):
                return
            _POPULATE_DEFAULTS(self, defaults, providers_models, refresh_options=True)
            if not _setup_owner_current(request):
                raise _CompactSetupChanged("compact_model_setup_source_changed")
            _request_default_sync(self)
            if not _setup_owner_current(request):
                raise _CompactSetupChanged("compact_model_setup_source_changed")
        except asyncio.CancelledError:
            # Mount is a Textual message-pump boundary. The shared owner has
            # already drained its physical read; refuse publication without
            # leaving a cancelled child Task for App shutdown to gather.
            request.cancelled = True
            return
        except (_CompactSetupChanged, NoMatches):
            return
        finally:
            # The shared read owner returns only after physical retirement.
            request.finished = True
            if request.widget_fields.get("_compact_setup_request") is request:
                request.widget_fields["_compact_setup_request"] = None

    def _populate_initial_defaults(
        self, defaults: dict, providers_models: dict, *, refresh_options: bool
    ) -> None:
        """Keep original default resolution and programmatic event suppression."""
        if refresh_options:
            provider_select = self.query_one("#compact-api-provider", Select)
            model_select = self.query_one("#compact-api-model", Select)
            with provider_select.prevent(Select.Changed):
                provider_select.set_options(
                    [(provider, provider) for provider in providers_models]
                )
            resolved = resolve_provider_name(
                defaults.get("provider", ""), providers_models
            )
            with model_select.prevent(Select.Changed):
                model_select.set_options(
                    [(model, model) for model in providers_models.get(resolved, [])]
                )
        available_providers = list(providers_models.keys())
        default_provider = resolve_provider_name(
            defaults.get("provider", ""),
            providers_models,
        )
        # Set provider
        try:
            provider_select = self.query_one("#compact-api-provider", Select)
            if default_provider in available_providers:
                with provider_select.prevent(Select.Changed):
                    provider_select.value = default_provider
        except NoMatches:
            pass
        # Set model
        initial_models = providers_models.get(default_provider, [])
        default_model = defaults.get("model", "")
        try:
            model_select = self.query_one("#compact-api-model", Select)
            if default_model in initial_models:
                with model_select.prevent(Select.Changed):
                    model_select.value = default_model
            elif initial_models:
                with model_select.prevent(Select.Changed):
                    model_select.value = initial_models[0]
        except NoMatches:
            pass

    async def on_unmount(self) -> None:
        """Wait for this widget's original native callback before unmount."""
        request = self._compact_setup_request
        if request is None:
            return
        request.cancelled = True
        cancelled = await drain_preparation_reads(request.reads)
        if request.widget_fields.get("_compact_setup_request") is request:
            request.widget_fields["_compact_setup_request"] = None
        if cancelled:
            raise asyncio.CancelledError

    @on(Select.Changed, "#compact-api-provider")
    async def handle_compact_provider_change(self, event: Select.Changed) -> None:
        """Handle provider change in compact bar and sync to sidebar."""
        # task-16474: the provider select allows blank ("nothing chosen");
        # a blank change carries no provider to sync. Covers Textual's
        # BLANK and NULL sentinels the same way the screen's handler does.
        if (
            event.value is None
            or event.value == Select.BLANK
            or str(event.value).startswith("Select.")
        ):
            return
        new_provider = str(event.value)
        logger.info(f"Compact bar: provider changed to {new_provider}")

        providers_models = get_cli_providers_and_models()
        available_models = providers_models.get(new_provider, [])

        # Update compact model select
        try:
            compact_model = self.query_one("#compact-api-model", Select)
            new_options = [(m, m) for m in available_models]
            compact_model.set_options(new_options)
            if available_models:
                compact_model.value = available_models[0]
            else:
                compact_model.value = Select.BLANK
        except NoMatches:
            pass

        # Sync to sidebar provider select
        try:
            sidebar_provider = self.app.query_one("#chat-api-provider", Select)
            sidebar_provider.value = event.value
        except NoMatches:
            logger.debug("Sidebar provider select not found for sync")

    @on(Select.Changed, "#compact-api-model")
    async def handle_compact_model_change(self, event: Select.Changed) -> None:
        """Sync model change to sidebar."""
        try:
            sidebar_model = self.app.query_one("#chat-api-model", Select)
            sidebar_model.value = event.value
        except NoMatches:
            logger.debug("Sidebar model select not found for sync")
        except Exception as e:
            logger.debug(
                f"Sidebar model select could not accept compact model value {event.value!r}: {e}"
            )

    @on(Input.Changed, "#compact-temperature")
    async def handle_compact_temp_change(self, event: Input.Changed) -> None:
        """Sync temperature change to sidebar."""
        try:
            sidebar_temp = self.app.query_one("#chat-temperature", Input)
            sidebar_temp.value = event.value
        except NoMatches:
            logger.debug("Sidebar temperature input not found for sync")

    @on(Button.Pressed, "#compact-sidebar-toggle")
    async def handle_sidebar_toggle(self, event: Button.Pressed) -> None:
        """Toggle the settings sidebar.

        ``ChatWindowEnhanced`` is retired, so the only live host wiring this
        widget always passes ``on_sidebar_toggle_requested`` (the Console
        control bar routes it to ``ChatScreen._toggle_console_chat_sidebar``);
        the callback is the sole toggle path now.

        Args:
            event: The compact-bar sidebar-toggle button press.
        """
        event.stop()
        if self.on_sidebar_toggle_requested:
            result = self.on_sidebar_toggle_requested()
            if hasattr(result, "__await__"):
                await result

    def sync_from_sidebar(
        self, provider: str = None, model: str = None, temperature: str = None
    ) -> None:
        """Sync values from sidebar to compact bar (called when sidebar values change)."""
        try:
            compact_model = None
            providers_models = get_cli_providers_and_models()
            available_models: list[str] | None = None
            if provider is not None:
                compact_provider = self.query_one("#compact-api-provider", Select)
                if compact_provider.value != provider:
                    with compact_provider.prevent(Select.Changed):
                        compact_provider.value = provider
                available_models = providers_models.get(provider, [])
                compact_model = self.query_one("#compact-api-model", Select)
                with compact_model.prevent(Select.Changed):
                    compact_model.set_options([(m, m) for m in available_models])
            if model is not None:
                if compact_model is None:
                    compact_model = self.query_one("#compact-api-model", Select)
                if available_models is None:
                    try:
                        compact_provider = self.query_one(
                            "#compact-api-provider", Select
                        )
                        current_provider = (
                            None
                            if compact_provider.value == Select.BLANK
                            else str(compact_provider.value)
                        )
                    except NoMatches:
                        current_provider = None
                    available_models = (
                        providers_models.get(current_provider, [])
                        if current_provider
                        else []
                    )
                if model not in available_models:
                    available_models = [*available_models, model]
                    with compact_model.prevent(Select.Changed):
                        compact_model.set_options([(m, m) for m in available_models])
                if compact_model.value != model:
                    with compact_model.prevent(Select.Changed):
                        compact_model.value = model
            if temperature is not None:
                compact_temp = self.query_one("#compact-temperature", Input)
                if compact_temp.value != temperature:
                    with compact_temp.prevent(Input.Changed):
                        compact_temp.value = temperature
        except NoMatches:
            pass


#
# End of compact_model_bar.py
#######################################################################################################################

# Definition-time metadata for only the optional finite lifecycle route.
_POPULATE_DEFAULTS = CompactModelBar._populate_initial_defaults
_PROOF_HELPERS = tuple(
    (name, globals()[name], globals()[name].__code__)
    for name in ("_source_current", "_function_current")
)
_WIDGET_SOURCE = (
    globals(),
    __file__,
    __spec__,
    getattr(__spec__, "origin", None),
    tuple(
        (globals(), name, globals()[name])
        for name in (
            "_config",
            "run_preparation_read",
            "drain_preparation_reads",
            "partial",
            "_CONFIG_SOURCE",
            "_PARENT_SOURCE",
            "_PROOF_HELPERS",
            "MessagePump",
            "ReferenceType",
            "CodeType",
            "_parent_code_shape",
            "_capture_parent_source",
            "_parent_source_current",
            "_parent_snapshot",
            "get_cli_providers_and_models",
            "resolve_provider_name",
            "_function_current",
            "_source_current",
            "_sources_current",
            "_plain_fields",
            "_setup_inputs",
            "_setup_owner_current",
            "_read_compact_setup",
            "_capture_setup",
            "_runtime_fields",
            "_require_setup_current",
            "_request_default_sync",
            "_POPULATE_DEFAULTS",
            "_CompactSetup",
            "_CompactSetupChanged",
            "CompactModelBar",
            "Select",
            "Input",
        )
    )
    + tuple(
        (vars(CompactModelBar), name, vars(CompactModelBar)[name])
        for name in ("compose", "on_mount", "on_unmount", "_populate_initial_defaults")
    )
    + ((vars(_CompactSetup), "__init__", _CompactSetup.__init__),),
    tuple(
        (
            namespace,
            name,
            function,
            function.__code__,
            function.__globals__,
            function.__defaults__,
            function.__kwdefaults__,
            tuple((function.__kwdefaults__ or {}).items()),
            function.__closure__,
            tuple((cell, cell.cell_contents) for cell in function.__closure__ or ()),
            vars(function).get("__wrapped__"),
        )
        for namespace, name, function in (
            *(
                (globals(), name, globals()[name])
                for name in (
                    "_parent_code_shape",
                    "_capture_parent_source",
                    "_parent_source_current",
                    "_parent_snapshot",
                    "_function_current",
                    "_source_current",
                    "_sources_current",
                    "_plain_fields",
                    "_setup_inputs",
                    "_setup_owner_current",
                    "_read_compact_setup",
                    "_capture_setup",
                    "_runtime_fields",
                    "_require_setup_current",
                    "_request_default_sync",
                )
            ),
            *(
                (vars(CompactModelBar), name, vars(CompactModelBar)[name])
                for name in (
                    "compose",
                    "on_mount",
                    "on_unmount",
                    "_populate_initial_defaults",
                )
            ),
            (vars(_CompactSetup), "__init__", _CompactSetup.__init__),
        )
    ),
)
