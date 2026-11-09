"""Private single rehearsal observations; never product authority or paint proof."""

import hashlib
import json
from pathlib import Path
import queue
import threading
import time

from native_series_method import LIMITS

SCALARS = {
    "stage",
    "correlation",
    "clock",
    "timestamp_ns",
    "boundary",
    "outcome",
    "shape",
    "provider_matches",
    "model_matches",
    "explicit_base_url_matches",
    "base_url_present",
    "ordinal",
    "request_bytes",
    "call_count",
    "decision_count",
    "kind",
    "endpoint",
    "geometry_cost_ns",
    "host_returned",
    "submitted",
}
VIEW = {
    "visible_send_lifecycle",
    "turn_id",
    "same_screen",
    "attachment_matches",
    "runtime_store_matches",
    "runtime_controller_matches",
    "store_active_session_matches",
    "custody_record_present",
    "custody_record_matches",
    "custody_task_matches",
    "custody_released",
    "task_done",
    "task_cancelled",
    "timer_present",
    "timer_task_present",
    "timer_task_done",
    "timer_active_resident",
    "timer_active_set",
    "poll_asked",
    "settlement",
    "owner_terminal",
    "session_matches",
    "card_retired",
    "header_terminal",
    "send_terminal",
    "stop_hidden",
    "run_segment_absent",
    "title_generation_matches",
    "rois",
    "generation",
    "header",
    "send",
    "provider",
    "session_id",
    "current",
    "correlation",
    "run_id",
    "shape",
    "card_identity",
    "card_identity_source",
    "feature_presentation_revision",
    "phase",
    "card_display",
    "card_attached",
    "information_ready",
    "submitted",
    "card_region",
    "controls",
    "feedback",
    "theme",
    "size",
    "region",
    "clip",
    "fully_visible",
    "id",
    "disabled",
    "display",
    "focused",
    "sequence",
    "decision_state",
    "selected_scope",
    "grant_state",
    "applied_scope",
    "execution_state",
    "terminal_outcome",
    "confirmed_backend_start",
    "snapshot",
    "native",
    "request_summary_cell_region",
    "request_summary_selector",
    "feedback_title_cell_region",
    "observer_health",
    "server_ns",
    "inspect_open",
    "font",
    "columns",
    "rows",
    "geometry",
    "client",
    "origin",
    "host_pid",
    "class",
    "dpi",
    "hwnd",
    "loss",
    "write_failure",
    "observer_failure",
    "rows",
    "bytes",
    "enqueue_max_ns",
    "write_max_ns",
    "rehearsal_scope",
}


def bounded_value(value, depth=0):
    if depth > 6:
        raise ValueError("metadata_depth")
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str and len(value) <= 256:  # noqa: E721 - Exact boundary types reject bool/subclasses.
        return value
    if type(value) in (tuple, list) and len(value) <= 16:
        return [bounded_value(item, depth + 1) for item in value]
    if (
        type(value) is dict  # noqa: E721 - Exact boundary types reject bool/subclasses.
        and len(value) <= 32
        and all(type(key) is str and key in VIEW for key in value)  # noqa: E721 - Exact boundary types reject bool/subclasses.
    ):
        return {key: bounded_value(item, depth + 1) for key, item in value.items()}
    raise ValueError("metadata_not_whitelisted")


class EventSink:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "rehearsal-events.jsonl"
        self.pending = queue.Queue(maxsize=256)
        self.lock = threading.Lock()
        self.loss = False
        self.write_failure = False
        self.observer_failure = False
        self.round_rows = {}
        self.rows = 0
        self.bytes = 0
        self.enqueue_max_ns = 0
        self.write_max_ns = 0
        self.closed = False
        self.thread = threading.Thread(
            target=self._write, name="private-rehearsal-events", daemon=True
        )
        try:
            self.thread.start()
        except BaseException:
            self.write_failure = True
            self.thread = None

    def fail(self):
        with self.lock:
            self.observer_failure = True

    def emit(self, row):
        try:
            self._emit(row)
        except BaseException:
            try:
                self.fail()
            except BaseException:
                pass

    def _emit(self, row):
        started = time.perf_counter_ns()
        try:
            if type(row) is not dict or not set(row).issubset(SCALARS | {"state"}):  # noqa: E721 - Exact boundary types reject bool/subclasses.
                raise ValueError("metadata_keys")
            clean = {
                key: (bounded_value(value) if key == "state" else value)
                for key, value in row.items()
            }
            if any(
                type(value) not in (str, bool, int, type(None))
                or (type(value) is str and len(value) > 256)  # noqa: E721 - Exact boundary types reject bool/subclasses.
                for key, value in clean.items()
                if key != "state"
            ):
                raise ValueError("metadata_scalar")
            raw = (
                json.dumps(clean, ensure_ascii=True, separators=(",", ":")) + "\n"
            ).encode("ascii")
            with self.lock:
                key = clean.get("correlation")
                if (
                    key not in self.round_rows
                    and len(self.round_rows) >= LIMITS["rounds"]
                ):
                    self.loss = True
                    return
                count = self.round_rows.get(key, 0)
                if (
                    self.closed
                    or count >= LIMITS["rows_round"]
                    or self.rows >= LIMITS["rows_process"]
                    or self.bytes + len(raw) > LIMITS["stage_bytes_process"]
                    or len(raw) > 4096
                ):
                    self.loss = True
                    return
                self.round_rows[key] = count + 1
                self.rows += 1
                self.bytes += len(raw)
            self.pending.put_nowait(raw)
        except queue.Full:
            with self.lock:
                self.loss = True
        except BaseException:
            self.fail()
        finally:
            with self.lock:
                self.enqueue_max_ns = max(
                    self.enqueue_max_ns, time.perf_counter_ns() - started
                )

    def _write(self):
        try:
            with self.path.open("ab", buffering=0) as stream:
                while True:
                    raw = self.pending.get()
                    if raw is None:
                        break
                    started = time.perf_counter_ns()
                    written = stream.write(raw)
                    if written != len(raw):
                        with self.lock:
                            self.write_failure = True
                        break
                    with self.lock:
                        self.write_max_ns = max(
                            self.write_max_ns, time.perf_counter_ns() - started
                        )
        except BaseException:
            with self.lock:
                self.write_failure = True

    def health(self):
        with self.lock:
            return {
                key: getattr(self, key)
                for key in (
                    "loss",
                    "write_failure",
                    "observer_failure",
                    "rows",
                    "bytes",
                    "enqueue_max_ns",
                    "write_max_ns",
                )
            }

    def close(self):
        with self.lock:
            self.closed = True
        try:
            self.pending.put_nowait(None)
        except queue.Full:
            with self.lock:
                self.loss = True
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.fail()
        try:
            (self.directory / "rehearsal-observer-health.json").write_text(
                json.dumps(self.health(), indent=2), encoding="utf-8"
            )
        except BaseException:
            with self.lock:
                self.write_failure = True


class EventReader:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.partial = b""
        self.rows = 0
        self.polls = 0
        self.poll_max_ns = 0

    def poll(self):
        started = time.perf_counter_ns()
        self.polls += 1
        try:
            return self._poll()
        finally:
            self.poll_max_ns = max(self.poll_max_ns, time.perf_counter_ns() - started)

    def _poll(self):
        if not self.path.exists():
            return []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            raw = stream.read(65536)
        self.offset += len(raw)
        parts = (self.partial + raw).split(b"\n")
        self.partial = parts.pop()
        if len(self.partial) > 4096 or self.offset > LIMITS["stage_bytes_process"]:
            raise RuntimeError("Event_reader_bound")
        values = []
        for line in parts:
            if line:
                self.rows += 1
                if self.rows > LIMITS["rows_process"]:
                    raise RuntimeError("Event_row_bound")
                values.append(json.loads(line))
        return values


def check_scope(state):
    view = state["snapshot"]
    health = state["observer_health"]
    if any(
        health.get(key) is not False
        for key in ("loss", "write_failure", "observer_failure")
    ):
        raise RuntimeError("Rehearsal observation loss/failure")
    if (
        view.get("theme") != "textual-dark"
        or view.get("size") != [120, 40]
        or view.get("inspect_open") is not False
        or state["native"].get("font") != [8, 16]
    ):
        raise RuntimeError("Unsupported live rehearsal theme/font/grid/Inspect state")


def classify_pixels(image, nonce, action, refs):
    """Hash only supplied NEW actual-image bytes; accept no row/hash authority."""
    if (
        image.mode != "RGBA"
        or len(nonce) != 32
        or any(char not in "0123456789abcdef" for char in nonce)
    ):
        return {
            "nonce_exact": False,
            "action_enabled": False,
            "reason": "unsupported_nonce_or_mode",
        }
    width, height = image.size
    if not (
        0 < width <= 960 and 0 < height <= 640 and width % 8 == 0 and height % 16 == 0
    ):
        return {
            "nonce_exact": False,
            "action_enabled": False,
            "reason": "unsupported_pixel_geometry",
        }
    matches = []
    for y in range(0, height, 16):
        decoded = "".join(
            refs["atlas"].get(
                hashlib.sha256(image.crop((x, y, x + 8, y + 16)).tobytes()).hexdigest(),
                "?",
            )
            for x in range(0, width, 8)
        )
        start = 0
        while True:
            index = decoded.find(nonce, start)
            if index < 0:
                break
            matches.append([index * 8, y])
            start = index + 1
    if (
        type(action) not in (list, tuple)
        or len(action) != 4
        or any(type(value) is not int for value in action)  # noqa: E721 - Exact boundary types reject bool/subclasses.
    ):
        return {
            "nonce_exact": False,
            "action_enabled": False,
            "reason": "unsupported_action_geometry",
        }
    x, y, w, h = action
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
        return {
            "nonce_exact": False,
            "action_enabled": False,
            "reason": "partial_action",
        }
    patch = hashlib.sha256(image.crop((x, y, x + w, y + h)).tobytes()).hexdigest()
    return {
        "nonce_exact": len(matches) == 1,
        "nonce_origins": matches,
        "action_RGBA_sha256": patch,
        "action_enabled": refs["enabled"].get(patch) == "Allow once",
        "source": "recomputed_new_capture_RGBA",
        "paint_claim": False,
    }


class CardEvents:
    def __init__(self, card, fixture, sink, read_state):
        self.card = card
        self.fixture = fixture
        self.sink = sink
        self.read_state = read_state
        self.pending = False
        self.originals = {}

        def wrap(original, name):
            def delegated(*args, **kwargs):
                if name == "_submit_fast_decision":
                    self.record("card_commit_enter")
                answer = original(*args, **kwargs)
                if name == "_submit_fast_decision":
                    self.record("card_commit_return")
                try:
                    self.schedule()
                except BaseException:
                    self.failed()
                return answer

            return delegated

        for name in ("_sync_control_layout", "paint_feedback", "_submit_fast_decision"):
            original = getattr(card, name)
            self.originals[name] = original
            setattr(card, name, wrap(original, name))

    def failed(self):
        try:
            self.sink.fail()
        except BaseException:
            pass

    def record(self, stage):
        try:
            current = self.fixture.current
            if current:
                self.sink.emit(
                    {
                        "stage": stage,
                        "correlation": current["correlation"],
                        "timestamp_ns": time.perf_counter_ns(),
                        "submitted": bool(self.card._batch_submitted),
                    }
                )
        except BaseException:
            self.failed()

    def schedule(self):
        if self.pending:
            return
        self.pending = True

        def after_refresh():
            self.pending = False
            started = time.perf_counter_ns()
            try:
                current = self.fixture.current
                if not current:
                    return
                state = self.read_state()
                self.sink.emit(
                    {
                        "stage": "geometry_after_refresh",
                        "correlation": current["correlation"],
                        "timestamp_ns": time.perf_counter_ns(),
                        "geometry_cost_ns": time.perf_counter_ns() - started,
                        "state": state,
                    }
                )
            except BaseException:
                self.failed()

        self.card.call_after_refresh(after_refresh)

    def close(self):
        for name, original in self.originals.items():
            setattr(self.card, name, original)
