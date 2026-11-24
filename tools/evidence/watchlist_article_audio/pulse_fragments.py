"""Linux PulseAudio monitor fragments. Only stream-frame positions have native custody."""

from contextlib import contextmanager
from dataclasses import dataclass
from queue import Full, Queue
import sys
import threading
import time

import numpy as np


FRAME_BYTES = 8
RATE = 48000
CLOCK_REASON = "pulse_snapshot_age_unqualified"


class PulseError(RuntimeError):
    """A fixed, device-independent reason safe to retain in a receipt."""


@dataclass(frozen=True)
class Fragment:
    frame: int
    data: np.ndarray


@dataclass(frozen=True)
class Route:
    sink: bytes
    sink_index: int
    monitor: bytes
    monitor_index: int


_PULSE_DECLARATIONS = """
struct tldw_timeval { long tv_sec; long tv_usec; };
typedef struct pa_timing_info {
    struct tldw_timeval timestamp;
    int synchronized_clocks;
    pa_usec_t sink_usec, source_usec, transport_usec;
    int playing, write_index_corrupt;
    int64_t write_index;
    int read_index_corrupt;
    int64_t read_index;
    pa_usec_t configured_sink_usec, configured_source_usec;
    int64_t since_underrun;
} pa_timing_info;
const pa_timing_info *pa_stream_get_timing_info(pa_stream *s);
const pa_sample_spec *pa_stream_get_sample_spec(pa_stream *s);
uint32_t pa_stream_get_device_index(const pa_stream *s);
const char *pa_stream_get_device_name(const pa_stream *s);
int pa_stream_is_suspended(const pa_stream *s);
void pa_stream_set_moved_callback(pa_stream *, void (*)(pa_stream *, void *), void *);
void pa_stream_set_suspended_callback(pa_stream *, void (*)(pa_stream *, void *), void *);
void pa_operation_cancel(pa_operation *);
// ABI prefix through proplist from libpulse introspect.h (stable since 0.9.11).
typedef struct pa_sink_input_info {
    uint32_t index;
    const char *name;
    uint32_t owner_module, client, sink;
    pa_sample_spec sample_spec;
    pa_channel_map channel_map;
    pa_cvolume volume;
    pa_usec_t buffer_usec, sink_usec;
    const char *resample_method, *driver;
    int mute;
    pa_proplist *proplist;
} pa_sink_input_info;
typedef void (*pa_sink_input_info_cb_t)(pa_context *, const pa_sink_input_info *, int, void *);
pa_operation *pa_context_get_sink_input_info(pa_context *, uint32_t, pa_sink_input_info_cb_t, void *);
typedef void (*tldw_subscribe_cb_t)(pa_context *, int, uint32_t, void *);
void pa_context_set_subscribe_callback(pa_context *, tldw_subscribe_cb_t, void *);
pa_operation *pa_context_subscribe(pa_context *, int, pa_context_success_cb_t, void *);

"""


def verify_route(default_sink, sink, source):
    """Verify both native name/index links between the default sink and its monitor."""
    name, index, monitor, monitor_index = sink
    if (
        not default_sink
        or not monitor
        or name != default_sink
        or index == 0xFFFFFFFF
        or monitor_index == 0xFFFFFFFF
        or source != (monitor, monitor_index, name, index)
    ):
        raise PulseError("pulse_monitor_mismatch")
    return Route(name, index, monitor, monitor_index)


def _operation(ffi, api, pulse, start):
    """Wait for one native operation outside the lock, cancel it on timeout."""
    with pulse._lock_mainloop():
        operation = start()
    if operation == ffi.NULL:
        raise PulseError("pulse_operation_unavailable")
    deadline = time.monotonic() + 1.0
    try:
        while True:
            with pulse._lock_mainloop():
                state = api.pa_operation_get_state(operation)
                if state == api.PA_OPERATION_DONE:
                    return
                if state != api.PA_OPERATION_RUNNING:
                    raise PulseError("pulse_operation_failed")
                if time.monotonic() >= deadline:
                    api.pa_operation_cancel(operation)
                    raise PulseError("pulse_operation_timeout")
            time.sleep(0.001)
    finally:
        with pulse._lock_mainloop():
            api.pa_operation_unref(operation)


# libpulse def.h subscription event types for the sink-input facility.
SINK_INPUT = 2
NEW = 0
CHANGE = 0x10
REMOVE = 0x20


@dataclass
class InputEvent:
    kind: int
    index: int
    pid: bytes | None = None
    sink: int | None = None
    complete: bool = False
    superseded: bool = False


class PlayerRouteHistory:
    """Private, case-scoped sink-input events. Only the verdict enters a receipt."""

    def __init__(self):
        self.events = []
        self.lost = False

    def verdict(self, pid: int, sink_index: int) -> str:
        unavailable = "pulse_player_route_unavailable"
        target = str(pid).encode()
        indexes = {event.index for event in self.events if event.pid == target}
        if self.lost or pid <= 0 or len(indexes) != 1:
            return unavailable
        owned = [event for event in self.events if event.index in indexes]
        observations = [event for event in owned if event.kind != REMOVE]
        if any(
            event.pid == target and event.sink != sink_index for event in observations
        ):
            return "pulse_player_route_changed"
        if (
            owned[0].kind != NEW
            or owned[-1].kind != REMOVE
            or any(event.kind != CHANGE for event in owned[1:-1])
            or any(
                not event.complete or event.superseded or event.pid != target
                for event in observations
            )
        ):
            return unavailable
        # An unattributed new input might also belong to this player.
        attributed = {event.index for event in self.events if event.pid}
        if any(
            event.kind == NEW and event.index not in attributed for event in self.events
        ):
            return unavailable
        return "verified"


@contextmanager
def watch_player_route(ffi, api, pulse):
    """Subscribe before launch, resolve event snapshots off the Pulse mainloop."""
    history = PlayerRouteHistory()
    pending = Queue(maxsize=128)
    last = {}

    @ffi.callback("tldw_subscribe_cb_t")
    def notify(context, event_type, index, userdata):
        if event_type & 15 != SINK_INPUT or history.lost:
            return
        kind = event_type & 0x30
        if len(history.events) >= 128 or kind not in (NEW, CHANGE, REMOVE):
            history.lost = True
            return
        previous = last.get(index)
        # A later event before our previous snapshot can hide an intermediate route.
        if previous is not None and previous.kind != REMOVE and previous.sink is None:
            previous.superseded = True
        event = InputEvent(kind, index)
        history.events.append(event)
        last[index] = event
        if kind != REMOVE:
            pending.put_nowait(event)

    def snapshot(event):
        @ffi.callback("pa_sink_input_info_cb_t")
        def info(context, entry, eol, userdata):
            if eol:
                event.complete = eol > 0 and event.pid is not None
            elif (
                entry
                and entry.index == event.index
                and entry.proplist
                and event.sink is None
            ):
                prop = api.pa_proplist_gets(entry.proplist, b"application.process.id")
                if prop:
                    event.pid = ffi.string(prop)
                event.sink = int(entry.sink)

        _operation(
            ffi,
            api,
            pulse,
            lambda: api.pa_context_get_sink_input_info(
                pulse.context, event.index, info, ffi.NULL
            ),
        )

    def observe():
        while (event := pending.get()) is not None:
            try:
                snapshot(event)
            except Exception:
                history.lost = True

    def subscribe(mask):
        ok = False

        @ffi.callback("pa_context_success_cb_t")
        def completed(context, success, userdata):
            nonlocal ok
            ok = bool(success)

        _operation(
            ffi,
            api,
            pulse,
            lambda: api.pa_context_subscribe(pulse.context, mask, completed, ffi.NULL),
        )
        if not ok:
            raise PulseError("pulse_player_route_unavailable")

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    with pulse._lock_mainloop():
        api.pa_context_set_subscribe_callback(pulse.context, notify, ffi.NULL)
    try:
        subscribe(1 << SINK_INPUT)
        yield history
        # The server reply fences preceding input events, including ordinary REMOVE.
        subscribe(0)
    finally:
        with pulse._lock_mainloop():
            api.pa_context_set_subscribe_callback(pulse.context, ffi.NULL, ffi.NULL)
        try:
            pending.put(None, timeout=0.2)
            observer.join(timeout=1.1)
        except Full:
            history.lost = True
        if observer.is_alive():
            history.lost = True


def _route(ffi, api, pulse):
    # Read the exact default from the server, bypassing fuzzy device lookup.
    default = []

    @ffi.callback("pa_server_info_cb_t")
    def server(context, info, userdata):
        if info and info.default_sink_name:
            default.append(ffi.string(info.default_sink_name))

    _operation(
        ffi,
        api,
        pulse,
        lambda: api.pa_context_get_server_info(pulse.context, server, ffi.NULL),
    )
    if len(default) != 1:
        raise PulseError("no_default_render")

    def endpoint(device, kind, peer_name, peer_index):
        rows = []
        completed = []

        @ffi.callback(f"pa_{kind}_info_cb_t")
        def callback(context, info, eol, userdata):
            if eol:
                completed.append(eol)
            elif info and info.name and getattr(info, peer_name):
                rows.append(
                    (
                        ffi.string(info.name),
                        int(info.index),
                        ffi.string(getattr(info, peer_name)),
                        int(getattr(info, peer_index)),
                    )
                )

        query = getattr(api, f"pa_context_get_{kind}_info_by_name")
        _operation(
            ffi, api, pulse, lambda: query(pulse.context, device, callback, ffi.NULL)
        )
        if len(rows) != 1 or completed != [1]:
            raise PulseError("pulse_monitor_unavailable")
        return rows[0]

    sink = endpoint(default[0], "sink", "monitor_source_name", "monitor_source")
    source = endpoint(sink[2], "source", "monitor_of_sink_name", "monitor_of_sink")
    return verify_route(default[0], sink, source)


def wait_for_active(api, pulse, stream):
    """Allow an idle sink to wake before opening the measured capture interval."""
    deadline = time.monotonic() + 1
    while True:
        with pulse._lock_mainloop():
            if api.pa_stream_get_state(stream) != api.PA_STREAM_READY:
                raise PulseError("pulse_stream_unavailable")
            suspended = api.pa_stream_is_suspended(stream)
        if suspended == 0:
            return
        if suspended != 1 or time.monotonic() >= deadline:
            raise PulseError("pulse_stream_suspended")
        time.sleep(0.005)


def stream_geometry(ffi, api, stream, route):
    """Read the connected stream geometry and reject unreliable native state (locked)."""
    device = api.pa_stream_get_device_name(stream)
    if (
        api.pa_stream_get_state(stream) != api.PA_STREAM_READY
        or api.pa_stream_is_suspended(stream) != 0
        or not device
        or ffi.string(device) != route.monitor
        or api.pa_stream_get_device_index(stream) != route.monitor_index
    ):
        raise PulseError("pulse_capture_route_unavailable")
    spec = api.pa_stream_get_sample_spec(stream)
    if not spec or (spec.format, spec.rate, spec.channels) != (
        api.PA_SAMPLE_FLOAT32LE,
        RATE,
        2,
    ):
        raise PulseError("pulse_capture_format_mismatch")
    attr = api.pa_stream_get_buffer_attr(stream)
    if (
        not attr
        or not 0 < attr.fragsize <= attr.maxlength < 0xFFFFFFFF
        or attr.fragsize % FRAME_BYTES
        or attr.maxlength % FRAME_BYTES
    ):
        raise PulseError("pulse_capture_geometry_invalid")
    return int(attr.maxlength), int(attr.fragsize)


def read_fragment(ffi, api, stream, readable, capacity, expected_frame):
    """Copy one native peek and verify its actual before/after byte positions (locked)."""

    def read_index():
        info = api.pa_stream_get_timing_info(stream)
        if (
            not info
            or info.read_index_corrupt
            or info.read_index < 0
            or info.read_index % FRAME_BYTES
        ):
            raise PulseError("pulse_read_index_invalid")
        return int(info.read_index)

    if not 0 < readable <= capacity or readable % FRAME_BYTES:
        raise PulseError("pulse_readable_size_invalid")
    before = read_index()
    if expected_frame is not None and expected_frame * FRAME_BYTES != before:
        raise PulseError("pulse_read_index_gap")
    pointer = ffi.new("void **")
    size = ffi.new("size_t *")
    if api.pa_stream_peek(stream, pointer, size) != 0 or not size[0]:
        raise PulseError("pulse_peek_failed")
    count = int(size[0])
    try:
        if count % FRAME_BYTES or count > readable:
            raise PulseError("pulse_fragment_size_invalid")
        if not pointer[0]:
            raise PulseError("pulse_fragment_hole")
        data = (
            np.frombuffer(ffi.buffer(pointer[0], count), dtype="<f4")
            .reshape(-1, 2)
            .copy()
        )
    finally:
        if api.pa_stream_drop(stream) != 0:
            raise PulseError("pulse_drop_failed")
    if read_index() != before + count:
        raise PulseError("pulse_drop_index_mismatch")
    if not np.isfinite(data).all():
        raise PulseError("pulse_nonfinite_samples")
    return Fragment(before // FRAME_BYTES, data)


def _drain(ffi, api, pulse, stream, route, duration_s, block_frames, pending, finish):
    changed = False

    @ffi.callback("void(pa_stream *, void *)")
    def route_changed(stream, userdata):
        nonlocal changed
        changed = True

    @ffi.callback("pa_stream_success_cb_t")
    def updated(stream, success, userdata):
        nonlocal timing_ok
        timing_ok = bool(success)

    wait_for_active(api, pulse, stream)
    now = time.perf_counter()
    end = now + duration_s
    next_frame = None
    total_frames = 0
    timing_ok = False
    with pulse._lock_mainloop():
        capacity, fragsize = stream_geometry(ffi, api, stream, route)
        api.pa_stream_set_moved_callback(stream, route_changed, ffi.NULL)
        api.pa_stream_set_suspended_callback(stream, route_changed, ffi.NULL)
    try:
        pending.put_nowait(
            (
                "ready",
                now,
                {
                    "maxlength_bytes": capacity,
                    "fragsize_bytes": fragsize,
                    "requested_buffer_frames": block_frames,
                    "sample_rate_hz": RATE,
                    "channels": 2,
                    "sample_format": "float32le",
                },
                route.sink.decode(),
            )
        )
        signed_latency_min = signed_latency_max = None
        bracket_max = 0.0
        while time.perf_counter() < end:
            with pulse._lock_mainloop():
                if changed or stream_geometry(ffi, api, stream, route) != (
                    capacity,
                    fragsize,
                ):
                    raise PulseError("pulse_capture_route_changed")
                readable = int(api.pa_stream_readable_size(stream))
            if not readable:
                if finish.is_set():
                    return
                time.sleep(0.001)
                continue
            timing_ok = False
            before = time.perf_counter()
            _operation(
                ffi,
                api,
                pulse,
                lambda: api.pa_stream_update_timing_info(stream, updated, ffi.NULL),
            )
            after = time.perf_counter()
            if not timing_ok:
                raise PulseError("pulse_timing_update_failed")
            with pulse._lock_mainloop():
                if changed or stream_geometry(ffi, api, stream, route) != (
                    capacity,
                    fragsize,
                ):
                    raise PulseError("pulse_capture_route_changed")
                readable = int(api.pa_stream_readable_size(stream))
                fragment = read_fragment(
                    ffi,
                    api,
                    stream,
                    readable,
                    min(capacity, int(duration_s * RATE) * FRAME_BYTES),
                    next_frame,
                )
                usec, negative = ffi.new("pa_usec_t *"), ffi.new("int *")
                if api.pa_stream_get_latency(stream, usec, negative) == 0:
                    latency = int(usec[0]) * (-1 if negative[0] else 1) / 1e6
                    signed_latency_min = (
                        latency
                        if signed_latency_min is None
                        else min(signed_latency_min, latency)
                    )
                    signed_latency_max = (
                        latency
                        if signed_latency_max is None
                        else max(signed_latency_max, latency)
                    )
            total_frames += len(fragment.data)
            if total_frames > duration_s * RATE:
                raise PulseError("capture_bound_exceeded")
            next_frame = fragment.frame + len(fragment.data)
            bracket_max = max(bracket_max, after - before)
            pending.put_nowait(
                (
                    "fragment",
                    time.perf_counter(),
                    fragment,
                    {
                        "signed_latency_min_s": signed_latency_min,
                        "signed_latency_max_s": signed_latency_max,
                        "timing_update_bracket_max_s": bracket_max,
                    },
                )
            )
        raise PulseError("pulse_capture_lifetime_exceeded")
    finally:
        with pulse._lock_mainloop():
            api.pa_stream_set_moved_callback(stream, ffi.NULL, ffi.NULL)
            api.pa_stream_set_suspended_callback(stream, ffi.NULL, ffi.NULL)


def capture_fragments(connection, duration_s, requested_frames, finish, player_pid):
    """Capture in a killable Linux-only process, with one bounded FIFO to the parent."""
    if sys.platform != "linux":
        raise PulseError("pulse_requires_linux")
    from soundcard.pulseaudio import _Recorder, _ffi, _pa, _pulse

    if _ffi.sizeof("long") != 8 or _ffi.sizeof("void *") != 8:
        raise PulseError("pulse_unsupported_timeval_abi")
    _ffi.cdef(_PULSE_DECLARATIONS)
    route = _route(_ffi, _pa, _pulse)
    pending = Queue(maxsize=64)

    def send():
        try:
            while True:
                message = pending.get()
                connection.send(message)
                if message[0] in ("pulse_complete", "error"):
                    return
        except (EOFError, BrokenPipeError, OSError):
            return

    sender = threading.Thread(target=send, daemon=True)
    sender.start()
    try:
        with (
            watch_player_route(_ffi, _pa, _pulse) as player_route,
            _Recorder(route.monitor.decode(), RATE, 2, requested_frames) as recorder,
        ):
            _drain(
                _ffi,
                _pa,
                _pulse,
                recorder.stream,
                route,
                duration_s,
                requested_frames,
                pending,
                finish,
            )
        terminal = (
            "pulse_complete",
            player_route.verdict(player_pid.value, route.sink_index),
        )
    except PulseError as error:
        terminal = ("error", str(error))
    except Full:
        terminal = ("error", "pulse_delivery_backlog")
    except Exception:
        terminal = ("error", "pulse_capture_failed")
    try:
        pending.put(terminal, timeout=0.5)
        sender.join(timeout=0.5)
    except Full:
        # Parent requires the terminal acknowledgement before accepting any samples.
        pass
