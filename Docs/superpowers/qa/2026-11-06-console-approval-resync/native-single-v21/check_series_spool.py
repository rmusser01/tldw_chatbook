"""Pure real event-spool cap and metadata controls; no product/UI/native imports."""

import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--source-root", required=True, type=Path)
_parser.add_argument("--output-root", required=True, type=Path)
_parser.add_argument("--dependencies", type=Path)
_args = _parser.parse_args()
OUTPUT = _args.output_root.resolve()
if not OUTPUT.is_relative_to(_args.source_root.resolve() / ".superpowers"):
    raise RuntimeError(
        "Pure control outputs must stay inside selectedsource/.superpowers"
    )
OUTPUT.mkdir(parents=True, exist_ok=True)
if _args.dependencies is not None:
    sys.path.insert(0, str(_args.dependencies.resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_rehearsal_observation import EventSink, EventReader  # noqa: E402 - Selected-source/profile/dependency guard runs first.
from native_series_method import qpc_perf_interval  # noqa: E402 - Selected-source/profile/dependency guard runs first.

HERE = Path(__file__).resolve().parent
results = {}


def sink():
    value = object.__new__(EventSink)
    value.pending = queue.Queue(maxsize=256)
    value.lock = threading.Lock()
    value.loss = False
    value.write_failure = False
    value.observer_failure = False
    value.closed = False
    value.rows = 0
    value.bytes = 0
    value.round_rows = {}
    value.enqueue_max_ns = 0
    value.write_max_ns = 0
    return value


value = sink()
for index in range(64):
    value.emit({"stage": "controlled", "correlation": "0" * 32, "timestamp_ns": index})
assert value.rows == 64 and not value.loss
value.emit({"stage": "controlled", "correlation": "0" * 32, "timestamp_ns": 64})
assert value.rows == 64 and value.loss and value.pending.qsize() == 64
results["actual_sink_64perround_then_loss_without_owner_exception"] = True
value = sink()
for index in range(41):
    value.emit(
        {"stage": "controlled", "correlation": f"{index:032x}", "timestamp_ns": index}
    )
assert value.rows == 41 and not value.loss
value.emit({"stage": "controlled", "correlation": f"{41:032x}", "timestamp_ns": 41})
assert value.rows == 41 and value.loss
results["actual_sink_fixed41correlations_then_loss"] = True
value = sink()
value.emit({"stage": "controlled", "correlation": "x" * 257, "timestamp_ns": 1})
assert value.observer_failure and value.rows == 0
results["unbounded_scalar_never_enters_private_spool"] = True
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "rows.jsonl"
    path.write_text('{"stage":"controlled"}\n' * 2625, encoding="utf-8")
    try:
        EventReader(path).poll()
    except RuntimeError as error:
        assert str(error) == "Event_row_bound"
    else:
        raise AssertionError("2625rows accepted")
    path = Path(directory) / "bytes.jsonl"
    with path.open("wb") as stream:
        stream.seek(10747904)
        stream.write(b"\n")
    reader = EventReader(path)
    reader.offset = 10747904
    try:
        reader.poll()
    except RuntimeError as error:
        assert str(error) == "Event_reader_bound"
    else:
        raise AssertionError("Stage byte bound accepted")
results["actual_reader_mathematical_rows_bytes_caps"] = True
clock1 = {"qpc_frequency": 10000000, "brackets": [[100, 10010, 102]]}
clock2 = {"qpc_frequency": 10000000, "brackets": [[300, 30020, 301]]}
assert qpc_perf_interval(200, [clock1, clock2]) == [19810, 20020]
try:
    qpc_perf_interval(200, [clock1, {"qpc_frequency": 1, "brackets": []}])
except ValueError:
    pass
else:
    raise AssertionError("Frequency mismatch accepted")
results["clock_conversion_keeps_before_after_union_uncertainty"] = True
facts = {
    "status": "GREEN",
    "qualified": False,
    "product_samples": 0,
    "results": results,
    "scope": "Actual private sink/reader with stdlib queues/locks/tempfiles and controlled clock brackets; no App/UI/native/product import, external/service/model traffic or real timing qualification.",
}
(OUTPUT / "series-spool-control-results.json").write_text(
    json.dumps(facts, indent=2), encoding="utf-8"
)
print(json.dumps(facts, indent=2))
assert not any(
    name.startswith(("tldw_chatbook", "tldw_profile_core", "textual", "ctypes"))
    for name in sys.modules
)
