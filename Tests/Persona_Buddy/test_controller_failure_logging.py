"""A buddy that fails to load must say why somewhere -- but only so much.

TASK-32901 (tier-2 S16 P2): ``Persona_Buddy/controller.py`` swallowed eight
exceptions to ``None``/``False`` with zero ``logger`` calls in the whole
1,444-line module -- no loguru or stdlib logging import at all. A repository
bug, a schema mismatch or a corrupted visual pack therefore rendered as
"this persona has no avatar", with no log line, no notification and no
diagnostic field to work from.

The first cut of that fix used ``logger.opt(exception=True)``, which attaches
the full traceback and the exception's own message. That is wrong *here*:
``_read_local_buddy`` is built over ``self._profile_root`` and its handler has
carried ``# noqa: BLE001 - reject invalid private library authority without
paths`` since before this change, and the repo already keeps a bounded-error
discipline at that boundary (``PrivatePathError`` deliberately exposes a
bounded result and drops the original exception -- see
``Tests/Utils/test_private_paths.py``). An OSError or PrivatePathError raised
under that call renders the user's private profile path into its message.

So the contract these tests pin is two-sided: the failure *site* and the
exception *class* are recorded (diagnosable), and the exception's own message
and traceback are not (bounded). Widening it back to ``exception=True`` fails
the canary assertions below.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest
from loguru import logger

from tldw_chatbook.Logging_Config import (
    RedactingFileFormatter,
    _forward_loguru_to_standard,
)
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController
from tldw_chatbook.Persona_Buddy.preferences import BuddySelection


class _ForwardedRecords(logging.Handler):
    """Collect the standard-logging records the production forwarder emits."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture()
def captured():
    """What a user's log shows: loguru -> the real forwarder -> std logging.

    ``_forward_loguru_to_standard`` is the only production loguru sink
    (``configure_application_logging`` removes loguru's default one), so
    these records are the user-visible output (TASK-38989 regression fix).
    """
    handler = _ForwardedRecords()
    root = logging.getLogger()
    # An earlier TldwCli() in the same process sets the package logger to INFO
    # (app.py), which would drop these DEBUG records (review r10a-rev4).
    package = logging.getLogger("tldw_chatbook")
    old_levels = (root.level, package.level)
    root.setLevel(logging.DEBUG)
    package.setLevel(logging.DEBUG)
    root.addHandler(handler)
    sink_id = logger.add(_forward_loguru_to_standard, level="DEBUG")
    try:
        yield handler.records
    finally:
        logger.remove(sink_id)
        root.removeHandler(handler)
        root.setLevel(old_levels[0])
        package.setLevel(old_levels[1])


def _rendered(records) -> str:
    """The forwarded lines as the persistent sink's redacting formatter writes them."""
    formatter = RedactingFileFormatter("%(message)s")
    return "\n".join(formatter.format(record) for record in records)


def _attached_tracebacks(records) -> list:
    """Every forwarded record that carries exception details."""
    return [record for record in records if record.exc_info]


def test_graph_read_failure_is_recorded(captured):
    def _explode(_db):
        raise RuntimeError("visual pack schema mismatch")

    controller = PersonaBuddyController(
        profile_db=object(),
        repository_factory=_explode,
    )

    assert controller._read_graph("persona-1") is None
    rendered = _rendered(captured)
    # Diagnosable: the fixed site category and the failure class both reach
    # the forwarded line (the forwarder renders the bound exception_type).
    assert "persona_buddy_visual_graph_unreadable" in rendered
    assert "RuntimeError" in rendered
    # Bounded: the exception's own text and traceback do not.
    assert "visual pack schema mismatch" not in rendered
    assert _attached_tracebacks(captured) == []


def test_runtime_resolution_failure_is_recorded(captured):
    controller = PersonaBuddyController(
        profile_db=object(),
        profile_root="/nonexistent/profile/root",
        repository_factory=lambda _db: (_ for _ in ()).throw(
            RuntimeError("corrupted pack")
        ),
    )

    assert controller._read_graph(None, buddy_id="buddy-1") is None
    rendered = _rendered(captured)
    assert "RuntimeError" in rendered
    assert "corrupted pack" not in rendered
    assert _attached_tracebacks(captured) == []


def test_private_library_boundary_never_logs_its_exception_payload(
    captured, monkeypatch
):
    """The one handler whose comment promises no paths must keep that promise."""

    def _explode(*_args, **_kwargs):
        raise OSError(2, "No such file", "/Users/someone/private/buddies.db")

    monkeypatch.setattr(
        "tldw_chatbook.Persona_Buddy.library.BuddyLibrary", _explode, raising=True
    )

    controller = PersonaBuddyController(
        profile_db=object(),
        profile_root="/Users/someone/private/profile",
    )

    assert controller._read_local_buddy(BuddySelection(buddy_id="buddy-1")) is None
    rendered = _rendered(captured)
    assert "persona_buddy_library_authority_rejected" in rendered
    assert "/Users/someone/private" not in rendered
    assert _attached_tracebacks(captured) == []


def _is_blanket(handler: ast.ExceptHandler) -> bool:
    """A handler that catches everything, as opposed to a named failure."""
    if handler.type is None:
        return True
    return isinstance(handler.type, ast.Name) and handler.type.id == "Exception"


def test_every_blanket_handler_in_the_module_records_something():
    """No ``except Exception`` body in this module may stay silent.

    The point of the finding is the module-wide silence, not one call site.
    Handlers that hand a failure category back to their caller already report
    it; the rest must log. Named-exception handlers are deliberate control
    flow and are out of scope.
    """
    source = Path(
        *PersonaBuddyController.__module__.replace(".", "/").split("/")
    ).with_suffix(".py")
    tree = ast.parse(source.read_text(encoding="utf-8"))

    silent: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or not _is_blanket(node):
            continue
        dumped = ast.dump(node)
        if "logger" not in dumped and "error_category" not in dumped:
            silent.append(node.lineno)

    assert silent == []


#: TASK-38989 review r10a-rev4: every site that binds ``exception_type``, which
#: the production forwarder appends to the line for every logger. A new site
#: must be reviewed here: the value must be a class name, never data.
EXCEPTION_TYPE_BIND_SITES = frozenset(
    {
        ("tldw_chatbook/Persona_Buddy/controller.py", "load_local_persona_portrait"),
        (
            "tldw_chatbook/Persona_Buddy/controller.py",
            "PersonaBuddyController._current_reduced_motion",
        ),
        (
            "tldw_chatbook/Persona_Buddy/controller.py",
            "PersonaBuddyController._read_local_buddy",
        ),
        (
            "tldw_chatbook/Persona_Buddy/controller.py",
            "PersonaBuddyController._read_local_persona",
        ),
        ("tldw_chatbook/Persona_Buddy/controller.py", "PersonaBuddyController._read_graph"),
        (
            "tldw_chatbook/Persona_Buddy/controller.py",
            "PersonaBuddyController._resolve_runtime",
        ),
        (
            "tldw_chatbook/Persona_Buddy/preferences.py",
            "persist_persona_buddy_preferences",
        ),
    }
)
_PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"


def _is_type_name(node: ast.AST) -> bool:
    """``type(<x>).__name__``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "__name__"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "type"
    )


def _is_class_name_value(node: ast.AST) -> bool:
    """A class name: ``type(x).__name__``, the bounded helper, or a literal."""
    return (
        _is_type_name(node)
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_bounded_exception_type"
        )
        or (isinstance(node, ast.Constant) and node.value == "Exception")
    )


def _bind_problems(keyword: ast.keyword, stack: list[ast.AST]) -> list[str]:
    """Why one ``exception_type`` keyword might bind something other than a class."""
    if keyword.arg is None:
        return ["**spread can carry exception_type"]
    if _is_type_name(keyword.value):
        return []
    if not isinstance(keyword.value, ast.Name):
        return [f"binds {ast.unparse(keyword.value)}"]
    function = next(
        (
            item
            for item in reversed(stack)
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    values = [
        assign.value
        for assign in (ast.walk(function) if function is not None else ())
        if isinstance(assign, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == keyword.value.id
            for target in assign.targets
        )
    ]
    if values and all(_is_class_name_value(value) for value in values):
        return []
    return [f"{keyword.value.id} is not only a class name"]


def _visit_bind_sites(
    node: ast.AST,
    stack: list[ast.AST],
    relative: str,
    sites: dict[tuple[str, str], list[str]],
) -> None:
    scoped = isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    if scoped:
        stack.append(node)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"bind", "contextualize"}
    ):
        for keyword in node.keywords:
            if keyword.arg in ("exception_type", None):
                owner = ".".join(getattr(item, "name", "") for item in stack)
                sites.setdefault((relative, owner), []).extend(
                    _bind_problems(keyword, stack)
                )
    for child in ast.iter_child_nodes(node):
        _visit_bind_sites(child, stack, relative, sites)
    if scoped:
        stack.pop()


def _exception_type_bind_sites(package: Path) -> dict[tuple[str, str], list[str]]:
    """Map each ``bind``/``contextualize`` carrying ``exception_type`` to its problems."""
    sites: dict[tuple[str, str], list[str]] = {}
    for path in sorted(package.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "exception_type" in source:
            relative = path.relative_to(package.parent).as_posix()
            _visit_bind_sites(ast.parse(source), [], relative, sites)
    return sites


def test_exception_type_is_bound_only_at_reviewed_class_name_sites():
    """The forwarder renders ``exception_type`` on every logger's line, so the
    sites that bind it are pinned, and each must bind only a class name."""
    sites = _exception_type_bind_sites(_PACKAGE)
    assert set(sites) == EXCEPTION_TYPE_BIND_SITES
    assert {site: problems for site, problems in sites.items() if problems} == {}


def test_bounded_exception_type_returns_only_a_class_name():
    """The helper the controller's sites bind through."""
    from tldw_chatbook.Persona_Buddy.controller import _bounded_exception_type

    assert _bounded_exception_type(ValueError("private detail")) == "ValueError"
    odd = type("not a / class name", (Exception,), {})
    assert _bounded_exception_type(odd()) == "Exception"

