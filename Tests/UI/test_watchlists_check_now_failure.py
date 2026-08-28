"""A failed watchlist check must be reported and must leave a trace — TASK-1090.

`_check_now_source` wrapped the whole fetch in `except Exception`, logged at
**debug**, and showed a transient toast. That is the swallow that hid
TASK-1100: `Check now` raised `ValueError` on every single press, the entire
feature was dead, and the only evidence was a debug line nobody reads and a
toast that had gone before anyone looked. Three UAT runs and a full test suite
reported the screen as working while it fetched nothing.

Worse than the swallow, and found while fixing it: a check that *ran* and
failed did not raise at all. `LocalWatchlistsService.execute_run` catches the
fetch error, records a `failed` run and returns it — so the screen's `try`
succeeded and it told the user **"Check now started."** over a feed that had
just 404'd.

A fetch is the one Watchlists operation that routinely fails for ordinary
reasons — the feed moved, the host is down, the XML is malformed, the network
is out — so it is exactly the operation that must report.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from loguru import logger
from rich.text import Text
from textual.widgets import Button, DataTable

from Tests.UI.app_factory import _build_test_app
from Tests.UI.full_app_destination_context import (
    FullAppDestinationContext as DestinationHarness,
)
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope


class Notified:
    """Capture what the app told the user, since the toast itself is transient."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, message, *args, severity: str = "information", **kwargs) -> None:
        self.calls.append((str(message), severity))

    @property
    def errors(self) -> list[str]:
        return [message for message, severity in self.calls if severity == "error"]


def _seed_source(app, *, name: str = "Summit Route") -> int:
    db = app.local_watchlists_service._db()
    return db.add_subscription(
        name=name, type="rss", source="https://summitroute.com/blog/feed.xml"
    )


async def _open_sources(pilot, host):
    screen = host.screen_stack[-1]
    screen.active_section = "sources"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    for _ in range(40):
        await pilot.pause()
        if pane.sources:
            break
    return screen, pane


@pytest.mark.asyncio
async def test_a_failed_fetch_is_reported_as_a_failure_and_leaves_a_trace():
    """AC#1, AC#2, AC#4: the durable half.

    The run executor raises the way a dead host does. The service's own
    failure path then has to write `subscriptions.last_error`, record a
    `failed` run, and the screen has to say so instead of "Check now started."
    """
    app = _build_test_app()
    source_id = _seed_source(app)
    notified = Notified()
    app.notify = notified

    async def dead_host(subscription):
        raise ConnectionError("Name or service not known: summitroute.com")

    app.local_watchlists_service.run_executor = dead_host

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, pane = await _open_sources(pilot, host)
        assert pane.sources, "the seeded source must reach the Sources pane"

        pane.select_source_by_id(str(pane.sources[0]["id"]))
        await pilot.pause(0.2)
        pane.query_one("#sources-check-now-button", Button).press()
        # TASK-2309: pressing Check now now ALSO posts an immediate
        # "Checking …" acknowledgment (severity=information) before the
        # worker even starts, so `notified.calls` goes non-empty on that
        # ack alone. Poll for `notified.errors` specifically -- the failure
        # this test is actually about -- so the loop does not exit before
        # the check has even run.
        for _ in range(60):
            await pilot.pause()
            if notified.errors:
                break

        assert notified.errors, (
            "a check that failed must be reported as a failure; it used to "
            f"report success. Got: {notified.calls!r}"
        )
        assert "summitroute.com" in " ".join(notified.errors), (
            "the report must name the reason, not just say it failed"
        )

    db = app.local_watchlists_service._db()
    row = db.get_subscription(source_id)
    assert row["last_error"], "a failed check must write subscriptions.last_error"

    runs = await app.local_watchlists_service.list_runs(source_id=source_id)
    assert runs, "a failed check must leave a run behind, not vanish"
    assert runs[0]["status"] == "failed"
    assert runs[0]["error_msg"], "the recorded run must carry the error"


@pytest.mark.asyncio
async def test_the_sources_status_column_shows_the_failure_after_the_toast_has_gone():
    """AC#1 (the surfaced half) and AC#2.

    `_source_row_cells` read `source.get("status")`, a key no normalizer
    emits -- `normalize_local_subscription_row` publishes `status_summary`.
    So the Status column read `-` whatever had happened, and `Last scraped`
    read `-` even straight after a successful check.
    """
    app = _build_test_app()
    _seed_source(app)
    app.notify = Notified()

    async def dead_host(subscription):
        raise ConnectionError("connection refused")

    app.local_watchlists_service.run_executor = dead_host

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, pane = await _open_sources(pilot, host)
        pane.select_source_by_id(str(pane.sources[0]["id"]))
        await pilot.pause(0.2)
        pane.query_one("#sources-check-now-button", Button).press()

        # The pane recomposes while the source list reloads, so the table is
        # briefly absent; poll rather than grabbing a handle once.
        status_cell = ""
        for _ in range(120):
            await pilot.pause()
            try:
                table = screen.query_one("#sources-table", DataTable)
                if not table.row_count:
                    continue
                status_cell = str(table.get_cell_at((0, 2)))
            except Exception:
                continue
            if "error" in status_cell.lower():
                break

        assert "error" in status_cell.lower(), (
            "after a failed check the Sources table must say so in its Status "
            f"column; it read {status_cell!r}"
        )


@pytest.mark.asyncio
async def test_an_unexpected_exception_in_the_fetch_path_logs_above_debug():
    """AC#3: the level that hid TASK-1100 for three UAT runs."""
    app = _build_test_app()
    _seed_source(app)
    app.notify = Notified()

    records: list[tuple[str, str]] = []
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, pane = await _open_sources(pilot, host)

        # The production app configures Loguru during startup. Observe the
        # boundary only after that lifecycle step, and remove the sink before
        # teardown, so the test neither races nor replaces production startup.
        sink_id = logger.add(
            lambda message: records.append(
                (message.record["level"].name, message.record["message"])
            ),
            level="DEBUG",
        )
        try:

            async def boom(**kwargs):
                raise ValueError("invalid literal for int() with base 10")

            screen._controller.check_now = boom
            pane.select_source_by_id(str(pane.sources[0]["id"]))
            await pilot.pause(0.2)
            records.clear()
            pane.query_one("#sources-check-now-button", Button).press()
            for _ in range(60):
                await pilot.pause()
                if any("check" in message.lower() for _, message in records):
                    break
        finally:
            logger.remove(sink_id)

    check_records = [
        (level, message)
        for level, message in records
        if "invalid literal" in message or "check" in message.lower()
    ]
    assert check_records, "the failure must be logged at all"
    assert any(
        level in ("WARNING", "ERROR", "CRITICAL") for level, _ in check_records
    ), (
        "an unexpected exception in the fetch path must log at warning or "
        f"above, not debug. Got: {check_records!r}"
    )


@pytest.mark.asyncio
async def test_a_check_that_fails_before_execution_still_records_a_run():
    """AC#4's remaining gap.

    `execute_run` records its own failure, but anything that goes wrong
    *around* it -- the namespaced-id `ValueError` of TASK-1100, a bad run id,
    a service fault -- left the queued run sitting at `queued` forever with no
    error on it and nothing else written anywhere.
    """
    app = _build_test_app()
    source_id = _seed_source(app)
    service = app.local_watchlists_service

    async def broken_execute(run_id):
        raise ValueError("invalid literal for int() with base 10: 'local'")

    service.execute_run = broken_execute

    with pytest.raises(ValueError):
        await app.watchlist_scope_service.check_now(
            runtime_backend="local", source_id=f"local:subscription:{source_id}"
        )

    runs = await service.list_runs(source_id=source_id)
    assert runs, "the launched run must not disappear when execution fails"
    assert runs[0]["status"] == "failed", (
        f"a run whose execution raised must be recorded failed, not "
        f"{runs[0]['status']!r}"
    )
    assert "invalid literal" in str(runs[0]["error_msg"] or ""), (
        "the recorded run must carry the error that stopped it"
    )
    row = service._db().get_subscription(source_id)
    assert row["last_error"], (
        "a check that failed before it fetched anything must still mark the "
        "source as errored"
    )


def test_source_row_cells_render_the_normalizer_status_summary():
    """The column mapping, at the unit level (AC#1).

    Pinned separately because the pane's own tests all feed a synthetic
    `status`/`last_scraped` shape that the real normalizer never produces,
    which is how a column reading `-` for every source in every state stayed
    green.

    TASK-2308: this used to assert the column rendered the raw ISO-8601
    string verbatim -- exactly the premise that task exists to remove. The
    column now renders `humane_timestamp` of the same value; asserted here
    by calling that formatter directly (not by pinning one of its outputs),
    since which of "Today HH:MM"/"Jul 28 09:00"/"2025-12-31" it produces for
    a fixed stored value depends on the machine's local zone and the date the
    suite happens to run -- exactly why `humane_time.py` has its own,
    clock-controlled unit tests (`test_humane_time.py`) covering every branch.
    """
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_local_subscription_row,
    )
    from tldw_chatbook.UI.Watchlists_Modules.humane_time import humane_timestamp

    source = normalize_local_subscription_row(
        {
            "id": 1,
            "name": "Summit Route",
            "type": "rss",
            "source": "https://summitroute.com/blog/feed.xml",
            "is_active": 1,
            "last_error": "connection refused",
            "error_count": 2,
            "last_checked": "2026-07-28T09:00:00+00:00",
        }
    )
    cells = SourcesPane._source_row_cells(source, False)
    assert isinstance(cells[2], Text)
    assert "error" in cells[2].plain.lower(), (
        f"Status column rendered {cells[2].plain!r} for an errored source"
    )
    assert cells[3].plain == humane_timestamp("2026-07-28T09:00:00+00:00"), (
        f"Last checked column rendered {cells[3].plain!r}, not through humane_timestamp"
    )


# TASK-1090 AC#6. Every Watchlists action a user can *press*. A swallowed
# failure in one of these means the button did nothing and nothing durable
# said so -- which is exactly how `Check now` stayed dead through three UAT
# runs and a green suite. Load/refresh coroutines are deliberately NOT in this
# list: they are background reads whose failure is already visible as an empty
# region plus a "Failed to load ..." toast, and promoting them would make an
# offline session log a wall of warnings.
#
# That exemption has a price, and TASK-2306's `_load_run_detail` shipped
# taking the exemption without paying it (review wave, Important 2): it logged
# at debug and showed nothing, so a denied `items.list` policy rendered
# byte-identically to "this run produced no items". `LOADERS_THAT_MUST_NOTIFY`
# below turns the price into a contract, so the next loader cannot do the same.
# Item delete gestures now converge on ``_update_item_status``; keep the former
# logical action name so its baseline node identity remains stable.
USER_INITIATED_MUTATIONS = (
    "_start_tree_write",
    "_run_tree_write",
    "_create_source",
    "_cancel_run",
    "_rerun_run",
    "_preview_source",
    "_check_now_source",
    "_on_opml_import_complete",
    "_export_opml",
    "_update_item_status",
    "_save_rule",
    "_delete_source",
    "_delete_run",
    "_delete_rule",
    "handle_delete_requested",
    "_delete_item",
)
USER_INITIATED_MUTATION_OWNERS = {"_delete_item": "_update_item_status"}


def _method_source(module_source: str, name: str) -> str:
    """The body of one method, from its `def` to the next same-indent `def`."""
    import re

    match = re.search(rf"^(\s*)(?:async )?def {re.escape(name)}\(", module_source, re.M)
    assert match, f"{name} is not defined on the Watchlists screen any more"
    indent = match.group(1)
    rest = module_source[match.end() :]
    end = re.search(rf"^{indent}(?:async )?def |^{indent}@", rest, re.M)
    return rest[: end.start()] if end else rest


#: The other half of the exemption above, as a RULE rather than a list
#: (re-review, m5): a background read may log its failure at `debug` ONLY
#: because the user sees a toast instead. The loaders are therefore discovered
#: structurally -- every `_load*` method on the screen -- and each `except`
#: handler guarding an AWAITED read must notify in that SAME handler. A list
#: would have to be remembered; this cannot be forgotten.
#:
#: Why "awaited", and not simply "any handler that logs at debug": the rule
#: the comment above states is about *background reads*, and running the broad
#: form over the real screen showed exactly why the distinction matters --
#: it also catches two things that are not background reads at all:
#:
#:  * `_load_notifications`' trailing handler, which guards a synchronous
#:    widget push (`pane.notifications = ...`), not a fetch. Every sibling
#:    loader wraps that same push in a bare `except Exception: pass`; a toast
#:    per failed repaint is not what the exemption is buying. Its real fetch
#:    handler IS covered, and stays covered -- which a method-level exemption
#:    would have silently stopped doing.
#:  * `_load_source_rows_for_tree`, which is SYNCHRONOUS: the tree calls it
#:    during `compose()`. TASK-2340 now covers this synchronous loader gap via
#:    the debug-handler contract below, alongside `scoped_source_rows`.


def _screen_source() -> str:
    return (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "UI"
        / "Screens"
        / "watchlists_collections_screen.py"
    ).read_text(encoding="utf-8")


SYNCHRONOUS_DEBUG_HANDLER_EXEMPTIONS = {
    (
        "_read_tree_data_snapshot.read_branch",
        "Failed to load watchlists tree branch: {}.",
    ): "Snapshot branch failures are episode-toasted by the tree-load coordinator.",
    (
        "_read_tree_data_snapshot",
        "Failed to load Watchlists tree membership.",
    ): "Snapshot membership failures are episode-toasted by the tree-load coordinator.",
    (
        "_recompute_effective_layout",
        "Workbench not mounted yet; layout applies on compose.",
    ): "Workbench mount lifecycle handling is outside the synchronous loader policy.",
    (
        "_schedule_layout_persist",
        "Could not schedule preferred Watchlists layout persistence.",
    ): "Preference-write scheduling is outside the synchronous loader policy.",
    (
        "_persist_layout_worker",
        "Failed to persist preferred Watchlists pane layout.",
    ): "Preference-write persistence is outside the synchronous loader policy.",
    (
        "_persist_layout_worker",
        "Could not acknowledge preferred Watchlists layout write.",
    ): "Preference-write acknowledgement is outside the synchronous loader policy.",
    (
        "_restore_focus_after_swap",
        "No section tab to restore focus to after the swap.",
    ): "Focus-restoration lifecycle handling is outside the synchronous loader policy.",
}


_SYNCHRONOUS_SCAN_BOUNDARIES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
)
_HANDLER_ATTRIBUTION_BOUNDARIES = (
    ast.ExceptHandler,
    *_SYNCHRONOUS_SCAN_BOUNDARIES,
)


def _nested_synchronous_functions(node: ast.AST):
    """Yield local sync functions without entering any nested scope."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef):
            yield child
            continue
        if isinstance(child, (ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield from _nested_synchronous_functions(child)


def _synchronous_function_handlers(function: ast.FunctionDef):
    """Yield each handler in one sync function, including nested handlers."""

    def walk(node: ast.AST):
        if isinstance(node, _SYNCHRONOUS_SCAN_BOUNDARIES):
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SYNCHRONOUS_SCAN_BOUNDARIES):
                continue
            if isinstance(child, ast.ExceptHandler):
                yield child
            yield from walk(child)

    for statement in function.body:
        yield from walk(statement)


def _synchronous_watchlists_handlers():
    """Yield qualified owners and handlers from class-body sync methods only."""
    module = ast.parse(_screen_source())
    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "WatchlistsCollectionsScreen"
    ]
    assert len(classes) == 1, (
        f"expected exactly one WatchlistsCollectionsScreen class, found {len(classes)}"
    )

    def visit(function: ast.FunctionDef, qualified_owner: str):
        for handler in _synchronous_function_handlers(function):
            yield qualified_owner, handler
        for local_function in _nested_synchronous_functions(function):
            yield from visit(local_function, f"{qualified_owner}.{local_function.name}")

    for node in classes[0].body:
        if isinstance(node, ast.FunctionDef):
            yield from visit(node, node.name)


def _handler_owned_nodes(handler: ast.ExceptHandler):
    """Yield nodes owned by one handler without borrowing nested-scope calls."""
    stack = list(handler.body)
    while stack:
        node = stack.pop()
        if isinstance(node, _HANDLER_ATTRIBUTION_BOUNDARIES):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _call_chain_root(node: ast.AST) -> str | None:
    """Return the root name of an attribute/call chain, if it has one."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _call_chain_root(node.value)
    if isinstance(node, ast.Call):
        return _call_chain_root(node.func)
    return None


def _literal_logger_debug_calls(handler: ast.ExceptHandler):
    """Yield module-logger debug calls and their required literal messages."""
    for node in _handler_owned_nodes(handler):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "debug":
            continue
        if _call_chain_root(node.func) != "logger":
            continue
        assert node.args and isinstance(node.args[0], ast.Constant), (
            "synchronous debug handlers must use a literal first message argument"
        )
        message = node.args[0].value
        assert isinstance(message, str), (
            "synchronous debug handlers must use a string literal first message argument"
        )
        yield node, message


def _is_exact_error_notification(call: ast.Call) -> bool:
    """Whether this is a safe, literal self._notify_watchlists error toast."""
    if not (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "_notify_watchlists"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
    ):
        return False
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    severity = keywords.get("severity")
    markup = keywords.get("markup")
    return (
        isinstance(severity, ast.Constant)
        and severity.value == "error"
        and isinstance(markup, ast.Constant)
        and markup.value is False
    )


def test_synchronous_debug_handlers_notify_or_have_exact_exemptions():
    """Sync debug swallows must toast or carry one exact, reviewed exemption."""
    discovered: set[tuple[str, str]] = set()
    silent: set[tuple[str, str]] = set()

    for qualified_owner, handler in _synchronous_watchlists_handlers():
        debug_calls = list(_literal_logger_debug_calls(handler))
        if not debug_calls:
            continue
        assert len(debug_calls) == 1, (
            f"{qualified_owner} has {len(debug_calls)} literal logger debug calls "
            "in one exception handler; expected exactly one"
        )
        key = (qualified_owner, debug_calls[0][1])
        assert key not in discovered, (
            f"duplicate synchronous debug handler key: {key!r}"
        )
        discovered.add(key)

        has_safe_notification = any(
            isinstance(node, ast.Call) and _is_exact_error_notification(node)
            for node in _handler_owned_nodes(handler)
        )
        if not has_safe_notification:
            silent.add(key)

    assert discovered, "the synchronous debug-handler scan found nothing"
    assert all(SYNCHRONOUS_DEBUG_HANDLER_EXEMPTIONS.values()), (
        "every synchronous debug-handler exemption must explain why it is safe"
    )

    exemption_keys = set(SYNCHRONOUS_DEBUG_HANDLER_EXEMPTIONS)
    unexplained = silent - exemption_keys
    stale = exemption_keys - silent
    assert not unexplained and not stale, (
        f"unexplained silent synchronous debug handlers: {sorted(unexplained)!r}; "
        f"stale exact exemptions: {sorted(stale)!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "expected_message"),
    (
        (
            "_load_source_rows_for_tree",
            "Failed to load sources for this watchlist.",
        ),
        (
            "scoped_source_rows",
            "Failed to resolve sources for the selected scope.",
        ),
    ),
)
async def test_synchronous_source_loader_failure_is_safely_reported(
    method_name, expected_message
):
    """Mounted sync loaders report fixed copy without leaking exception text."""
    app = _build_test_app()
    _seed_source(app)
    notifications: list[tuple[str, tuple, dict]] = []

    def capture_notification(message, *args, **kwargs) -> None:
        notifications.append((str(message), args, kwargs))

    app.notify = capture_notification
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, _pane = await _open_sources(pilot, host)

        sentinel = "service exploded at [/bold]/private/watchlists.db"

        class ExplodingBundleService:
            failing = False

            def list_source_rows(self, watchlist_id):
                if self.failing:
                    raise RuntimeError(sentinel)
                return []

        service = ExplodingBundleService()
        app.watchlist_bundle_service = service
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=73)
        await pilot.pause()

        service.failing = True
        notifications.clear()
        if method_name == "_load_source_rows_for_tree":
            result = screen._load_source_rows_for_tree(73)
        else:
            result = screen.scoped_source_rows()

        assert result == []
        assert len(notifications) == 1
        message, args, kwargs = notifications[0]
        assert message == expected_message
        assert args == ()
        assert kwargs == {"severity": "error", "markup": False}
        assert sentinel not in repr(notifications)


@pytest.mark.asyncio
async def test_scoped_source_failure_notifies_once_per_failure_episode():
    """Repeated mounted reads share one toast until a successful recovery."""
    app = _build_test_app()
    _seed_source(app)
    notifications: list[tuple[str, tuple, dict]] = []

    def capture_notification(message, *args, **kwargs) -> None:
        notifications.append((str(message), args, kwargs))

    class ToggleBundleService:
        failing = True

        def list_source_rows(self, watchlist_id):
            if self.failing:
                raise RuntimeError("private scoped read failure")
            return []

    service = ToggleBundleService()
    app.notify = capture_notification
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen, _pane = await _open_sources(pilot, host)
        app.watchlist_bundle_service = service

        notifications.clear()
        screen.tree_scope = TreeScope(kind="watchlist", watchlist_id=73)
        await pilot.pause()

        expected = "Failed to resolve sources for the selected scope."
        matching = [entry for entry in notifications if entry[0] == expected]
        assert matching == [(expected, (), {"severity": "error", "markup": False})]

        service.failing = False
        assert screen.scoped_source_rows() == []

        service.failing = True
        notifications.clear()
        assert screen.scoped_source_rows() == []
        assert screen.scoped_source_rows() == []
        matching = [entry for entry in notifications if entry[0] == expected]
        assert matching == [(expected, (), {"severity": "error", "markup": False})]


def _own_nodes(node):
    """Every node inside `node`, stopping at any nested function boundary."""
    import ast

    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        stack.extend(ast.iter_child_nodes(child))


def _calls_named(node, name: str):
    """Calls to `name(...)` or `<anything>.name(...)` inside `node`."""
    import ast

    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == name:
            yield child
        elif isinstance(func, ast.Attribute) and func.attr == name:
            yield child


def _awaited_read_handlers():
    """(method, handler) for every `_load*` handler guarding an awaited read."""
    import ast

    tree = ast.parse(_screen_source())
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not func.name.startswith("_load"):
            continue
        for node in _own_nodes(func):
            if not isinstance(node, ast.Try):
                continue
            guards_await = any(
                isinstance(inner, ast.Await)
                for statement in node.body
                for inner in ast.walk(statement)
            )
            if not guards_await:
                continue
            for handler in node.handlers:
                yield func.name, handler


def test_the_loader_contract_actually_finds_loaders():
    """Guard the guard: a scan that matched nothing would pass vacuously."""
    found = {name for name, _ in _awaited_read_handlers()}
    assert len(found) >= 5, f"the structural scan found only {sorted(found)}"
    assert "_load_run_detail" in found, (
        "the loader this contract was written for is not being scanned"
    )


def test_background_loaders_pay_for_their_debug_exemption_with_a_toast():
    """Review wave I2 / re-review m5 -- the exemption's price, enforced.

    A loader that swallows into `debug` with no toast is invisible twice over:
    the region it fills draws exactly as it would for an empty result, and the
    only trace goes to a log the user will never open.

    The notify must live in the SAME handler as the failure, not merely
    somewhere in the method: a loader that notifies on SUCCESS would otherwise
    satisfy a whole-body substring check while telling the user nothing when
    it fails.
    """
    silent = []
    for method_name, handler in _awaited_read_handlers():
        logs_at_debug = any(True for _ in _calls_named(handler, "debug"))
        if not logs_at_debug:
            continue
        notifies = [
            call
            for call in _calls_named(handler, "notify")
            if any(keyword.arg == "severity" for keyword in call.keywords)
        ]
        if not notifies:
            silent.append(method_name)

    assert not silent, (
        f"{sorted(set(silent))} handle their own failure, log it at debug, and "
        "tell the user nothing. Background reads are exempt from the "
        "log-at-warning rule ONLY because their failure surfaces as a toast; "
        "without one the failure renders identically to an empty result."
    )


@pytest.mark.parametrize("method_name", USER_INITIATED_MUTATIONS)
def test_user_initiated_actions_do_not_swallow_failures_into_debug(method_name):
    """AC#6, as a contract rather than a one-time audit."""
    from pathlib import Path

    screen_source = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "UI"
        / "Screens"
        / "watchlists_collections_screen.py"
    ).read_text(encoding="utf-8")

    owner_name = USER_INITIATED_MUTATION_OWNERS.get(method_name, method_name)
    body = _method_source(screen_source, owner_name)
    assert ".debug(" not in body, (
        f"{method_name} logs a failure at debug. It is behind a control the "
        "user pressed, so a swallowed failure there is invisible: the action "
        "appears to do nothing. Log at warning or above."
    )
