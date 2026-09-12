#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace tldw::duplex {

constexpr std::size_t kFrameBytes = 960;
using Pcm = std::array<char, kFrameBytes>;

// PortAudio's ABI uses unsigned long even on LLP64 Windows.
struct TimeInfo {
  double inputBufferAdcTime;
  double currentTime;
  double outputBufferDacTime;
};

struct RenderReceipt {
  std::uint64_t generation = 0;
  std::uint64_t submission_id = 0;
  std::uint64_t output_epoch = 0;
  std::uint64_t reference_sequence = 0;
  double output_dac_time = 0;
  double output_dac_end_time = 0;
};

struct CaptureRecord {
  Pcm pcm16{};
  Pcm output_pcm16{};
  std::uint64_t capture_sequence = 0;
  std::uint64_t callback_ordinal = 0;
  std::uint64_t generation = 0;
  std::uint64_t observed_ns = 0;
  TimeInfo time{};
  unsigned long status_bits = 0;
  unsigned long fatal_status_bits = 0;
  std::uint64_t capture_overflows = 0;
  std::uint64_t invalid_frames = 0;
  std::uint64_t invalid_timing = 0;
  std::uint64_t capture_occupancy = 0;
  std::uint64_t render_occupancy = 0;
  bool rendered = false;
  RenderReceipt render{};
  std::uint64_t startup_discarded_before = 0;
  bool playback_context_valid = true;
  bool has_playback = false;
  double playback_start_dac_time = 0;
  double playback_end_dac_time = 0;
};

struct Snapshot {
  std::uint64_t capture_occupancy;
  std::uint64_t render_occupancy;
  std::uint64_t capture_overflows;
  std::uint64_t invalid_frames;
  std::uint64_t invalid_timing;
  unsigned long fatal_status_bits;
  std::uint64_t startup_discarded;
  std::uint64_t callback_count;
  bool active;
  bool render_admission;
  std::uint64_t output_epoch;
};

class State {
public:
  State(std::size_t capture_capacity, std::size_t render_capacity,
        std::uint64_t generation);
  State(const State &) = delete;
  State &operator=(const State &) = delete;

  static std::uint64_t MonotonicNs() noexcept;
  static std::uintptr_t CallbackAddress() noexcept;
  bool QueueRender(const char *pcm, std::uint64_t submission,
                   std::uint64_t epoch);
  void SetRenderAdmission(bool enabled) noexcept;
  std::uint64_t AbortOutput() noexcept;
  void Deactivate() noexcept;
  std::uint64_t OutputEpoch() const noexcept;
  std::optional<CaptureRecord> PopCapture();
  std::optional<RenderReceipt> LatestRender() const noexcept;
  Snapshot GetSnapshot() const noexcept;
  int Callback(const void *input, void *output, unsigned long frames,
               const TimeInfo *time, unsigned long status) noexcept;

private:
  struct RenderSlot {
    Pcm pcm{};
    std::uint64_t submission_id = 0;
    std::uint64_t epoch = 0;
  };
  bool Faulted() const noexcept;
  void PublishReceipt(const RenderReceipt &receipt) noexcept;
  const std::size_t capture_capacity_;
  const std::size_t render_capacity_;
  const std::uint64_t generation_;
  std::unique_ptr<CaptureRecord[]> capture_;
  std::unique_ptr<RenderSlot[]> render_;
  std::atomic<std::uint64_t> capture_write_{0}, capture_read_{0};
  std::atomic<std::uint64_t> render_write_{0}, render_read_{0};
  std::atomic<std::uint64_t> epoch_{0};
  std::atomic<bool> active_{true}, admission_{false};
  std::atomic<std::uint64_t> capture_overflows_{0}, invalid_frames_{0},
      invalid_timing_{0}, startup_discarded_{0}, callback_count_{0};
  std::atomic<unsigned long> fatal_status_{0};

  // All receipt words, INCLUDING payload, are atomic. Sequential consistency
  // gives the version check a single total order; no non-atomic seqlock races.
  std::atomic<std::uint64_t> receipt_version_{0};
  std::array<std::atomic<std::uint64_t>, 5> receipt_words_{};

  // Callback-owned only; consumers access their ring-published copies.
  std::uint64_t capture_sequence_ = 0, reference_sequence_ = 0;
  bool has_playback_ = false;
  double playback_start_ = 0, playback_end_ = 0;
  // Python producer-owned; the application serializes all producer calls.
  bool has_submission_ = false;
  std::uint64_t last_submission_ = 0;
};

} // namespace tldw::duplex
