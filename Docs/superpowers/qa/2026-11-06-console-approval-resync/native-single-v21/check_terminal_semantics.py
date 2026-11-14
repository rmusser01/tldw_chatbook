"""Pure actual settlement-block controls; metadata/Region only, never pixels/App."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", required=True, type=Path)
parser.add_argument("--output-root", required=True, type=Path)
parser.add_argument("--dependencies", type=Path)
parser.add_argument("--child-source", type=Path)
args = parser.parse_args()
source = args.source_root.resolve()
output = args.output_root.resolve()
if not output.is_relative_to(source / ".superpowers"):
    raise RuntimeError("Pure outputs must stay beneath selectedsource/.superpowers")
output.mkdir(parents=True, exist_ok=True)
sys.dont_write_bytecode = True
if args.dependencies is not None:
    sys.path.insert(0, str(args.dependencies.resolve()))
assert not any(
    n.startswith(("tldw_chatbook", "tldw_profile_core")) for n in sys.modules
)

from textual.geometry import Region  # noqa: E402 - Explicit existing dependency only.

here = Path(__file__).resolve().parent
child = args.child_source or next(
    p
    for p in (here / "native_product_child.py", here / "web_product_child.py")
    if p.exists()
)
child = child.resolve()
if not child.is_relative_to(source):
    raise RuntimeError("Settlement source must belong to selected source/owned QA")
raw = child.read_bytes()
tree = ast.parse(raw)
owner = next(
    n
    for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name == "capture_state"
)
index = next(
    i
    for i, n in enumerate(owner.body)
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "settlement" for t in n.targets)
)
block = copy.deepcopy(owner.body[index : index + 2])
assert isinstance(block[0], ast.Assign) and isinstance(block[1], ast.If)
assert not any(
    isinstance(n, (ast.Import, ast.ImportFrom, ast.Await))
    for n in ast.walk(ast.Module(body=block, type_ignores=[]))
)
function = ast.parse(
    "def observe(app, fixture, driver, full, view, visible):\n    pass\n"
).body[0]
function.body = block + [ast.Return(value=ast.Name(id="settlement", ctx=ast.Load()))]
namespace = {}
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        str(child),
        "exec",
    ),
    namespace,
)
observe = namespace["observe"]


class Widget:
    def __init__(self, content="", **values):
        self.content = content
        self.label = "Send"
        self.disabled = True
        self.display = False
        self.content_region = Region(0, 0, 1, 1)
        self.segment_spans = []
        self.__dict__.update(values)


class StringSubclass(str):
    pass


def fixture():
    correlation = "0123456789abcdef0123456789abcdef"
    widgets = {
        "generation": Widget("Conversation | QA-" + correlation[:12]),
        "header": Widget("Ready"),
        "send": Widget(),
        "provider": Widget(segment_spans=[("provider", 0, 16), ("model", 19, 40)]),
    }
    regions = {
        "generation": Region(10, 2, 40, 1),
        "header": Region(60, 2, 20, 1),
        "send": Region(80, 35, 14, 1),
        "provider": Region(1, 34, 100, 1),
    }
    visible = {}
    for name, widget in widgets.items():
        widget.content_region = regions[name]
        visible[widget] = (regions[name], Region(0, 0, 120, 40))
    stop = Widget()
    scroll = SimpleNamespace(scroll_x=0)
    selectors = {
        "#console-transcript-title": widgets["generation"],
        "#console-workbench-header #workbench-header-status": widgets["header"],
        "#console-send-message": widgets["send"],
        "#console-status-text": widgets["provider"],
        "#console-stop-generation": stop,
        "#console-status-chip-scroll": scroll,
    }
    app = SimpleNamespace(screen=SimpleNamespace(query_one=selectors.__getitem__))
    driver = SimpleNamespace(
        current={"correlation": correlation},
        session=SimpleNamespace(id="session"),
        controller=SimpleNamespace(store=SimpleNamespace(active_session_id="session")),
    )
    full = {
        "run_status": "completed",
        "submission_task": {
            "done": True,
            "result": {"accepted": True, "terminal_status": "completed"},
        },
    }
    view = {"card_display": False, "card_identity": None}
    return SimpleNamespace(
        app=app,
        driver=driver,
        full=full,
        view=view,
        visible=visible,
        widgets=widgets,
        stop=stop,
        scroll=scroll,
    )


results = {}


def check(name, mutate, predicate):
    value = fixture()
    mutate(value)
    observed = observe(
        value.app, value.driver, value.driver, value.full, value.view, value.visible
    )
    results[name] = {"passed": bool(predicate(observed)), "settlement": observed}


def provider_geometry(value, *, bounds=None, content=None, clip=None):
    widget = value.widgets["provider"]
    old_bounds, old_clip = value.visible[widget]
    value.visible[widget] = (bounds or old_bounds, clip or old_clip)
    if content is not None:
        widget.content_region = content


def all_terminal(value):
    return all(
        value[k] is True
        for k in (
            "owner_terminal",
            "session_matches",
            "card_retired",
            "header_terminal",
            "send_terminal",
            "stop_hidden",
            "run_segment_absent",
            "title_generation_matches",
        )
    ) and set(value["rois"]) == {"generation", "header", "send", "provider"}


check("complete_original_Ready_positive", lambda v: None, all_terminal)
check(
    "independent_exact_Ready_not_tested_positive",
    lambda v: setattr(v.widgets["header"], "content", "Ready · not tested"),
    all_terminal,
)
check(
    "independent_clipped_line_visible_provider_positive",
    lambda v: provider_geometry(
        v, bounds=Region(1, 34, 180, 1), content=Region(1, 34, 180, 1)
    ),
    lambda x: all_terminal(x) and x["rois"]["provider"] == [1, 34, 16, 1],
)
for label in (
    "Readyish",
    "Ready · reachable 12:00",
    "Running",
    "",
    None,
    1,
    StringSubclass("Ready"),
):
    check(
        "refuse_header_" + str(len(results)),
        lambda v, label=label: setattr(v.widgets["header"], "content", label),
        lambda x: x["header_terminal"] is False,
    )
for label in ("Ready", "Ready · not tested"):
    check(
        "actual_plain_header_" + str(len(results)),
        lambda v, label=label: setattr(
            v.widgets["header"], "content", SimpleNamespace(plain=label)
        ),
        all_terminal,
    )
for name, spans in (
    ("missing", []),
    ("duplicate", [("provider", 0, 10), ("provider", 12, 16)]),
    ("negative", [("provider", -1, 16)]),
    ("empty", [("provider", 4, 4)]),
    ("reverse", [("provider", 5, 4)]),
    ("outside_content", [("provider", 0, 101)]),
):
    check(
        "provider_span_" + name,
        lambda v, spans=spans: setattr(v.widgets["provider"], "segment_spans", spans),
        lambda x: "provider" not in x["rois"],
    )
for name, clip in (
    ("left", Region(2, 0, 118, 40)),
    ("right", Region(0, 0, 16, 40)),
    ("above", Region(0, 35, 120, 5)),
    ("below", Region(0, 0, 120, 34)),
):
    check(
        "partial_provider_clip_" + name,
        lambda v, clip=clip: provider_geometry(v, clip=clip),
        lambda x: "provider" not in x["rois"],
    )
check(
    "partial_provider_widget",
    lambda v: provider_geometry(
        v, bounds=Region(10, 34, 100, 1), clip=Region(0, 0, 100, 40)
    ),
    lambda x: "provider" not in x["rois"],
)
check(
    "provider_content_not_one_row",
    lambda v: provider_geometry(v, content=Region(1, 34, 100, 2)),
    lambda x: "provider" not in x["rois"],
)
check(
    "provider_widget_not_one_row",
    lambda v: provider_geometry(v, bounds=Region(1, 34, 100, 2)),
    lambda x: "provider" not in x["rois"],
)
check(
    "provider_not_visible",
    lambda v: v.visible.pop(v.widgets["provider"]),
    lambda x: "provider" not in x["rois"],
)
check(
    "provider_scrolled",
    lambda v: setattr(v.scroll, "scroll_x", 1),
    lambda x: "provider" not in x["rois"],
)
for name in ("generation", "header", "send"):

    def clipped(v, name=name):
        widget = v.widgets[name]
        bounds, _clip = v.visible[widget]
        v.visible[widget] = (
            bounds,
            Region(bounds.x + 1, bounds.y, bounds.width, bounds.height),
        )

    check(
        "other_ROI_whole_clip_" + name,
        clipped,
        lambda x, name=name: name not in x["rois"],
    )
for name, mutate, key in (
    (
        "run_not_complete",
        lambda v: v.full.update(run_status="running"),
        "owner_terminal",
    ),
    (
        "task_pending",
        lambda v: v.full["submission_task"].update(done=False),
        "owner_terminal",
    ),
    (
        "not_accepted",
        lambda v: v.full["submission_task"]["result"].update(accepted=False),
        "owner_terminal",
    ),
    (
        "terminal_result_missing",
        lambda v: v.full["submission_task"]["result"].update(terminal_status="running"),
        "owner_terminal",
    ),
    (
        "wrong_session",
        lambda v: setattr(v.driver.controller.store, "active_session_id", "other"),
        "session_matches",
    ),
    ("card_displayed", lambda v: v.view.update(card_display=True), "card_retired"),
    (
        "card_identity_present",
        lambda v: v.view.update(card_identity="other"),
        "card_retired",
    ),
    ("stop_displayed", lambda v: setattr(v.stop, "display", True), "stop_hidden"),
    (
        "run_segment_present",
        lambda v: v.widgets["provider"].segment_spans.append(("run", 45, 50)),
        "run_segment_absent",
    ),
    (
        "wrong_title_generation",
        lambda v: setattr(
            v.widgets["generation"], "content", "Conversation | QA-ffffffffffff"
        ),
        "title_generation_matches",
    ),
    ("missing_current", lambda v: setattr(v.driver, "current", None), "owner_terminal"),
    (
        "missing_session",
        lambda v: setattr(v.driver, "session", None),
        "session_matches",
    ),
):
    check(name, mutate, lambda x, key=key: x[key] is False)
assert not any(
    n.startswith(("tldw_chatbook", "tldw_profile_core")) for n in sys.modules
)
report = {
    "scope": "Actual source-extracted settlement statements; explicit metadata and real Textual Region only. No product import/App/network/model/native input/RGBA/timing claim.",
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "control_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "results": results,
    "passed": sum(v["passed"] for v in results.values()),
    "failed": [k for k, v in results.items() if not v["passed"]],
    "product_samples": 0,
}
target = output / "terminal-semantics-results.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print(
    json.dumps(
        {
            "passed": report["passed"],
            "failed": report["failed"],
            "source_sha256": report["source_sha256"],
            "receipt_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    )
)
raise SystemExit(bool(report["failed"]))
