#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "modules/audio_processing/aec3/echo_canceller3.h"
#include "modules/audio_processing/audio_buffer.h"

namespace py = pybind11;

void BindDuplex(py::module_ &module);

namespace {

constexpr int kSampleRate = 48'000;
constexpr int kChannels = 1;
constexpr std::size_t kSamplesPerFrame = 480;
constexpr std::size_t kBytesPerFrame = kSamplesPerFrame * sizeof(std::int16_t);
constexpr int kMaximumDelayMs = 1'000;

void ValidateDelay(int delay_ms) {
  if (delay_ms < 0 || delay_ms > kMaximumDelayMs) {
    throw py::value_error("delay_ms must be between 0 and 1000 inclusive");
  }
}

std::string RequireFrameBytes(const py::handle &value) {
  if (!PyBytes_CheckExact(value.ptr())) {
    throw py::type_error("pcm16 must be bytes");
  }
  std::string frame = py::cast<std::string>(value);
  if (frame.size() != kBytesPerFrame) {
    throw py::value_error("pcm16 must contain exactly 960 bytes");
  }
  return frame;
}

class AecProcessor {
public:
  AecProcessor(int sample_rate, int channels) {
    if (sample_rate != kSampleRate) {
      throw py::value_error("sample_rate must be 48000");
    }
    if (channels != kChannels) {
      throw py::value_error("channels must be 1");
    }
    Initialize();
  }

  void AnalyzeRender(const py::handle &pcm16, int delay_ms) {
    ValidateDelay(delay_ms);
    const std::string frame = RequireFrameBytes(pcm16);
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    FillBuffer(frame, render_buffer_.get());
    render_buffer_->SplitIntoFrequencyBands();
    aec_->SetAudioBufferDelay(delay_ms);
    aec_->AnalyzeRender(render_buffer_.get());
    ++render_frames_;
  }

  py::bytes ProcessCapture(const py::handle &pcm16, int delay_ms) {
    ValidateDelay(delay_ms);
    const std::string frame = RequireFrameBytes(pcm16);
    std::string cleaned;
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);
      FillBuffer(frame, capture_buffer_.get());
      capture_buffer_->SplitIntoFrequencyBands();
      aec_->SetAudioBufferDelay(delay_ms);
      aec_->AnalyzeCapture(capture_buffer_.get());
      aec_->ProcessCapture(capture_buffer_.get(), false);
      capture_buffer_->MergeFrequencyBands();
      cleaned = ReadBuffer(*capture_buffer_);
      ++capture_frames_;
    }
    return py::bytes(cleaned);
  }

  void Reset() {
    py::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mutex_);
    Initialize();
  }

  std::unordered_map<std::string, double> Metrics() {
    double erle_db = 0.0;
    double delay_ms = 0.0;
    double delay_estimate_available = 0.0;
    double delay_estimate_refined = 0.0;
    double delay_age_blocks = 0.0;
    double clock_drift = 0.0;
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> lock(mutex_);
      const webrtc::EchoControl::Metrics metrics = aec_->GetMetrics();
      if (capture_frames_ > 0 &&
          std::isfinite(metrics.echo_return_loss_enhancement)) {
        erle_db = metrics.echo_return_loss_enhancement;
      }
      if (capture_frames_ > 0) {
        delay_ms = static_cast<double>(metrics.delay_ms);
        delay_estimate_available = metrics.delay_estimate_available ? 1.0 : 0.0;
        delay_estimate_refined =
            metrics.delay_estimate_available && metrics.delay_estimate_refined
                ? 1.0
                : 0.0;
        delay_age_blocks =
            metrics.delay_estimate_available
                ? static_cast<double>(metrics.delay_estimate_age_blocks)
                : 0.0;
        clock_drift = metrics.clock_drift ? 1.0 : 0.0;
      }
    }
    return {{"erle_db", erle_db},
            {"delay_ms", delay_ms},
            {"delay_estimate_available", delay_estimate_available},
            {"delay_estimate_refined", delay_estimate_refined},
            {"delay_age_blocks", delay_age_blocks},
            {"clock_drift", clock_drift}};
  }

private:
  void Initialize() {
    const webrtc::EchoCanceller3Config config =
        webrtc::EchoCanceller3::CreateDefaultConfig(kChannels, kChannels);
    aec_ = std::make_unique<webrtc::EchoCanceller3>(config, kSampleRate,
                                                    kChannels, kChannels);
    render_buffer_ = std::make_unique<webrtc::AudioBuffer>(
        kSampleRate, kChannels, kSampleRate, kChannels, kSampleRate, kChannels);
    capture_buffer_ = std::make_unique<webrtc::AudioBuffer>(
        kSampleRate, kChannels, kSampleRate, kChannels, kSampleRate, kChannels);
    render_frames_ = 0;
    capture_frames_ = 0;
  }

  static void FillBuffer(const std::string &frame,
                         webrtc::AudioBuffer *buffer) {
    float *samples = buffer->channels()[0];
    for (std::size_t index = 0; index < kSamplesPerFrame; ++index) {
      const auto low = static_cast<std::uint8_t>(frame[index * 2]);
      const auto high = static_cast<std::uint8_t>(frame[index * 2 + 1]);
      const auto value =
          static_cast<std::int16_t>(static_cast<std::uint16_t>(low) |
                                    (static_cast<std::uint16_t>(high) << 8));
      samples[index] = static_cast<float>(value);
    }
  }

  static std::string ReadBuffer(const webrtc::AudioBuffer &buffer) {
    std::string output(kBytesPerFrame, '\0');
    const float *samples = buffer.channels_const()[0];
    for (std::size_t index = 0; index < kSamplesPerFrame; ++index) {
      const float bounded = std::clamp(samples[index], -32768.0f, 32767.0f);
      const auto value = static_cast<std::int16_t>(std::lrint(bounded));
      const auto bits = static_cast<std::uint16_t>(value);
      output[index * 2] = static_cast<char>(bits & 0xff);
      output[index * 2 + 1] = static_cast<char>((bits >> 8) & 0xff);
    }
    return output;
  }

  std::mutex mutex_;
  std::unique_ptr<webrtc::EchoCanceller3> aec_;
  std::unique_ptr<webrtc::AudioBuffer> render_buffer_;
  std::unique_ptr<webrtc::AudioBuffer> capture_buffer_;
  std::uint64_t render_frames_ = 0;
  std::uint64_t capture_frames_ = 0;
};

} // namespace

PYBIND11_MODULE(_native, module) {
  module.doc() = "Pinned WebRTC AEC3 native binding";
  BindDuplex(module);
  py::class_<AecProcessor>(module, "AecProcessor")
      .def(py::init<int, int>(), py::kw_only(),
           py::arg("sample_rate") = kSampleRate,
           py::arg("channels") = kChannels)
      .def("analyze_render", &AecProcessor::AnalyzeRender, py::arg("pcm16"),
           py::kw_only(), py::arg("delay_ms"))
      .def("process_capture", &AecProcessor::ProcessCapture, py::arg("pcm16"),
           py::kw_only(), py::arg("delay_ms"))
      .def("reset", &AecProcessor::Reset)
      .def("metrics", &AecProcessor::Metrics);
}
