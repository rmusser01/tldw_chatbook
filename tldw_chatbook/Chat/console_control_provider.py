"""Finite provider admission ends at the original gateway's adapter handoff."""

from __future__ import annotations

import threading


def begin_provider_dispatch(gateway):
    controls = getattr(gateway, "_console_local_controls", None)
    if controls is None:
        return None
    from .console_control_dispatch import require_control_source

    source = require_control_source(controls._runtime)
    if controls._controller.provider_gateway is not gateway:
        raise RuntimeError("Console provider receiver changed")
    return _ProviderDispatch(controls, source, gateway)


class _ProviderDispatch:
    def __init__(self, controls, source, gateway):
        self.controls, self.gateway = controls, gateway
        self.source = source
        self.trace_owner = None
        self.effect = controls._begin_effect("provider_dispatch", self, source)
        self._lock = threading.Lock()
        self.worker_owned = False
        self.finished = False
        self.admission = None

    def bind(self, gateway, admission):
        with self._lock:
            if (
                self.gateway is not gateway
                or self.admission is not None
                or self.finished
            ):
                raise RuntimeError("Console provider admission changed")
            self.admission = admission
            admission._control_dispatch = self

    def handoff(self, gateway, admission):
        with self._lock:
            if (
                self.gateway is not gateway
                or self.admission is not admission
                or not admission._consumed
                or self.finished
            ):
                raise RuntimeError("Console provider handoff changed")
            self.finished = True
        self.controls._finish_effect(self.effect, retired=True)

    def worker_returned(self, *, retired):
        with self._lock:
            if self.finished:
                return
            self.finished = True
        self.controls._finish_effect(self.effect, retired=retired)

    def finish_unsubmitted(self):
        with self._lock:
            if self.worker_owned or self.finished:
                return
            self.finished = True
        self.controls._finish_effect(self.effect, retired=True)
