#include "duplex_bridge.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace tldw::duplex;

namespace {

std::uint64_t Unsigned(const py::handle &value, const char *name) {
  if (!PyLong_CheckExact(value.ptr()))
    throw py::type_error(std::string(name) + " must be an integer");
  const auto result = PyLong_AsUnsignedLongLong(value.ptr());
  if (PyErr_Occurred()) throw py::error_already_set();
  return result;
}

py::dict Receipt(const RenderReceipt &r) {
  py::dict result;
  result["generation"] = r.generation;
  result["submission_id"] = r.submission_id;
  result["output_epoch"] = r.output_epoch;
  result["reference_sequence"] = r.reference_sequence;
  result["output_dac_time"] = r.output_dac_time;
  result["output_dac_end_time"] = r.output_dac_end_time;
  return result;
}

class NativeDuplexBridge {
public:
  NativeDuplexBridge(const py::handle &capture, const py::handle &render,
                     const py::handle &generation)
      : owner_(std::make_unique<State>(Unsigned(capture, "capture_capacity"),
                   Unsigned(render, "render_capacity"),
                   Unsigned(generation, "generation"))), state_(owner_.get()) {}

  State &state() { return *state_; }
  void RetainForProcessLifetime() {
    // Intentionally leak only native storage, without a Python-rooted registry
    // whose finalization could free a live callback's userdata. Irreversible.
    state_->Deactivate();
    (void)owner_.release();
  }

private:
  std::unique_ptr<State> owner_;
  State *state_;
};

py::object PopCapture(NativeDuplexBridge &bridge) {
  const auto record = bridge.state().PopCapture();
  if (!record) return py::none();
  const auto &r = *record;
  py::dict d;
  d["pcm16"] = py::bytes(r.pcm16.data(), r.pcm16.size());
  d["output_pcm16"] = py::bytes(r.output_pcm16.data(), r.output_pcm16.size());
  d["capture_sequence"] = r.capture_sequence;
  d["callback_ordinal"] = r.callback_ordinal;
  d["generation"] = r.generation;
  d["observed_ns"] = r.observed_ns;
  d["input_adc_time"] = r.time.inputBufferAdcTime;
  d["current_time"] = r.time.currentTime;
  d["output_dac_time"] = r.time.outputBufferDacTime;
  d["status_bits"] = r.status_bits;
  d["fatal_status_bits"] = r.fatal_status_bits;
  d["capture_overflows"] = r.capture_overflows;
  d["invalid_frames"] = r.invalid_frames;
  d["invalid_timing"] = r.invalid_timing;
  d["capture_occupancy"] = r.capture_occupancy;
  d["render_occupancy"] = r.render_occupancy;
  d["submission_id"] = r.rendered ? py::cast(r.render.submission_id) : py::none();
  d["output_epoch"] = r.rendered ? py::cast(r.render.output_epoch) : py::none();
  d["reference_sequence"] = r.rendered ? py::cast(r.render.reference_sequence) : py::none();
  d["startup_discarded_before"] = r.startup_discarded_before;
  d["playback_context_valid"] = r.playback_context_valid;
  d["playback_start_dac_time"] = r.has_playback ? py::cast(r.playback_start_dac_time) : py::none();
  d["playback_end_dac_time"] = r.has_playback ? py::cast(r.playback_end_dac_time) : py::none();
  return std::move(d);
}

} // namespace

void BindDuplex(py::module_ &module) {
  module.attr("DUPLEX_ABI_VERSION") = 1;
  py::class_<NativeDuplexBridge>(module, "NativeDuplexBridge")
      .def(py::init<const py::handle &, const py::handle &, const py::handle &>(),
           py::kw_only(), py::arg("capture_capacity") = 64,
           py::arg("render_capacity") = 64, py::arg("generation") = 0)
      .def_property_readonly("callback_address", [](NativeDuplexBridge &) {
        return State::CallbackAddress();
      })
      .def_property_readonly("userdata_address", [](NativeDuplexBridge &b) {
        return reinterpret_cast<std::uintptr_t>(&b.state());
      })
      .def_property_readonly("output_epoch", [](NativeDuplexBridge &b) {
        return b.state().OutputEpoch();
      })
      .def("monotonic_ns", [](NativeDuplexBridge &) { return State::MonotonicNs(); })
      .def("set_render_admission", [](NativeDuplexBridge &b, bool enabled) {
        b.state().SetRenderAdmission(enabled);
      }, py::arg("enabled"))
      .def("queue_render", [](NativeDuplexBridge &b, const py::handle &pcm,
                              const py::handle &submission, const py::handle &epoch) {
        if (!PyBytes_CheckExact(pcm.ptr())) throw py::type_error("pcm16 must be bytes");
        if (PyBytes_GET_SIZE(pcm.ptr()) != kFrameBytes)
          throw py::value_error("pcm16 must contain exactly 960 bytes");
        return b.state().QueueRender(PyBytes_AS_STRING(pcm.ptr()),
            Unsigned(submission, "submission_id"), Unsigned(epoch, "output_epoch"));
      }, py::arg("pcm16"), py::arg("submission_id"), py::arg("output_epoch"))
      .def("abort_output", [](NativeDuplexBridge &b) { return b.state().AbortOutput(); })
      .def("deactivate", [](NativeDuplexBridge &b) { b.state().Deactivate(); })
      .def("pop_capture", &PopCapture)
      .def("latest_render", [](NativeDuplexBridge &b) -> py::object {
        const auto receipt = b.state().LatestRender();
        if (!receipt) return py::none();
        return Receipt(*receipt);
      })
      .def("snapshot", [](NativeDuplexBridge &b) {
        const auto s = b.state().GetSnapshot();
        py::dict d;
        d["capture_occupancy"] = s.capture_occupancy;
        d["render_occupancy"] = s.render_occupancy;
        d["capture_overflows"] = s.capture_overflows;
        d["invalid_frames"] = s.invalid_frames;
        d["invalid_timing"] = s.invalid_timing;
        d["fatal_status_bits"] = s.fatal_status_bits;
        d["startup_discarded"] = s.startup_discarded;
        d["callback_count"] = s.callback_count;
        d["active"] = s.active;
        d["render_admission"] = s.render_admission;
        d["output_epoch"] = s.output_epoch;
        return d;
      })
      .def("retain_for_process_lifetime", &NativeDuplexBridge::RetainForProcessLifetime);
}
