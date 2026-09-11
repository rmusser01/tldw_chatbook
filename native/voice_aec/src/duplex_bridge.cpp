#include "duplex_bridge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tldw::duplex {
namespace {

std::size_t ValidateCapacity(std::size_t value) {
  if (value < 1 || value > 4096)
    throw std::invalid_argument("capacity must be between 1 and 4096");
  return value;
}

std::uint64_t Bits(double value) noexcept {
  std::uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

double Double(std::uint64_t bits) noexcept {
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint64_t Occupancy(const std::atomic<std::uint64_t> &read,
                        const std::atomic<std::uint64_t> &write,
                        std::size_t capacity) noexcept {
  const auto r = read.load(std::memory_order_acquire);
  const auto w = write.load(std::memory_order_acquire);
  // Diagnostic observation can span consumer progress; never report > capacity.
  return std::min<std::uint64_t>(w - r, capacity);
}

#if defined(_WIN32)
#define PA_CALL __cdecl
#else
#define PA_CALL
#endif
extern "C" int PA_CALL DuplexCallback(const void *input, void *output,
                                     unsigned long frames, const TimeInfo *time,
                                     unsigned long status, void *userdata) noexcept {
  return static_cast<State *>(userdata)->Callback(input, output, frames, time,
                                                  status);
}

} // namespace

State::State(std::size_t capture_capacity, std::size_t render_capacity,
             std::uint64_t generation)
    : capture_capacity_(ValidateCapacity(capture_capacity)),
      render_capacity_(ValidateCapacity(render_capacity)), generation_(generation),
      capture_(std::make_unique<CaptureRecord[]>(capture_capacity_)),
      render_(std::make_unique<RenderSlot[]>(render_capacity_)) {
  if (!capture_write_.is_lock_free() || !active_.is_lock_free() ||
      !fatal_status_.is_lock_free())
    throw std::runtime_error("native duplex requires lock-free atomics");
  for (auto &word : receipt_words_) word.store(0);
}

std::uint64_t State::MonotonicNs() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::uintptr_t State::CallbackAddress() noexcept {
  return reinterpret_cast<std::uintptr_t>(&DuplexCallback);
}

bool State::Faulted() const noexcept {
  return capture_overflows_.load() || invalid_frames_.load() ||
         invalid_timing_.load() || fatal_status_.load();
}

bool State::QueueRender(const char *pcm, std::uint64_t submission,
                        std::uint64_t epoch) {
  if (has_submission_ && submission <= last_submission_)
    throw std::invalid_argument("submission_id must increase within generation");
  if (!active_.load() || !admission_.load() || Faulted() || epoch != epoch_.load())
    return false;
  const auto w = render_write_.load(std::memory_order_relaxed);
  const auto r = render_read_.load(std::memory_order_acquire);
  if (w - r == render_capacity_) return false;
  auto &slot = render_[w % render_capacity_];
  std::memcpy(slot.pcm.data(), pcm, kFrameBytes);
  slot.submission_id = submission;
  slot.epoch = epoch;
  render_write_.store(w + 1, std::memory_order_release);
  last_submission_ = submission;
  has_submission_ = true;
  return true;
}

void State::SetRenderAdmission(bool enabled) noexcept {
  if (!enabled) {
    admission_.store(false);
    // Reopening admission must never resurrect queued frames from before a fence.
    AbortOutput();
  } else if (active_.load() && !Faulted()) {
    admission_.store(true);
  }
}

std::uint64_t State::AbortOutput() noexcept { return epoch_.fetch_add(1) + 1; }
std::uint64_t State::OutputEpoch() const noexcept { return epoch_.load(); }

void State::Deactivate() noexcept {
  active_.store(false);
  admission_.store(false);
  AbortOutput();
}

std::optional<CaptureRecord> State::PopCapture() {
  const auto r = capture_read_.load(std::memory_order_relaxed);
  if (r == capture_write_.load(std::memory_order_acquire)) return std::nullopt;
  auto record = capture_[r % capture_capacity_];
  capture_read_.store(r + 1, std::memory_order_release);
  return record;
}

void State::PublishReceipt(const RenderReceipt &receipt) noexcept {
  receipt_version_.fetch_add(1);
  receipt_words_[0].store(receipt.submission_id);
  receipt_words_[1].store(receipt.output_epoch);
  receipt_words_[2].store(receipt.reference_sequence);
  receipt_words_[3].store(Bits(receipt.output_dac_time));
  receipt_words_[4].store(Bits(receipt.output_dac_end_time));
  receipt_version_.fetch_add(1);
}

std::optional<RenderReceipt> State::LatestRender() const noexcept {
  for (unsigned attempt = 0; attempt < 8; ++attempt) {
    const auto before = receipt_version_.load();
    if (!before) return std::nullopt;
    if (before & 1) continue;
    const RenderReceipt receipt{generation_, receipt_words_[0].load(),
        receipt_words_[1].load(), receipt_words_[2].load(),
        Double(receipt_words_[3].load()), Double(receipt_words_[4].load())};
    if (before == receipt_version_.load()) return receipt;
  }
  return std::nullopt; // Off-callback consumer may retry; never return torn data.
}

Snapshot State::GetSnapshot() const noexcept {
  return {Occupancy(capture_read_, capture_write_, capture_capacity_),
          Occupancy(render_read_, render_write_, render_capacity_),
          capture_overflows_.load(), invalid_frames_.load(), invalid_timing_.load(),
          fatal_status_.load(), startup_discarded_.load(), callback_count_.load(),
          active_.load(), admission_.load(), epoch_.load()};
}

int State::Callback(const void *input, void *output, unsigned long frames,
                     const TimeInfo *time, unsigned long status) noexcept {
  const auto ordinal = callback_count_.fetch_add(1) + 1;
  // For a malformed frame count, touch at most our fixed frame allowance and
  // return paAbort below. PortAudio discards the aborted output buffer.
  if (output)
    std::memset(output, 0, std::min<unsigned long>(frames, 480) * 2);
  if (!active_.load()) return frames == 480 ? 0 : 2;

  // Reserve space before excusing any startup status: overflow is always fatal.
  const auto w = capture_write_.load(std::memory_order_relaxed);
  const auto r = capture_read_.load(std::memory_order_acquire);
  const bool full = w - r == capture_capacity_;
  if (full) capture_overflows_.fetch_add(1);
  const auto observed = MonotonicNs();
  const bool bad_frames = frames != 480 || !input || !output;
  const bool bad_time = !time || !std::isfinite(time->inputBufferAdcTime) ||
      !std::isfinite(time->currentTime) || !std::isfinite(time->outputBufferDacTime) ||
      !std::isfinite(time->outputBufferDacTime + 0.01);
  if (bad_frames) invalid_frames_.fetch_add(1);
  if (bad_time) invalid_timing_.fetch_add(1);
  if (status && ordinal <= 50 && !full && !bad_frames && !bad_time && !Faulted()) {
    startup_discarded_.fetch_add(1);
    return 0;
  }
  if (status) fatal_status_.fetch_or(status);
  const auto sequence = capture_sequence_++;
  if (Faulted()) admission_.store(false);
  if (bad_frames || bad_time || full) return bad_frames ? 2 : 0;

  auto &record = capture_[w % capture_capacity_];
  std::memcpy(record.pcm16.data(), input, kFrameBytes);
  record.capture_sequence = sequence;
  record.callback_ordinal = ordinal;
  record.generation = generation_;
  record.observed_ns = observed;
  record.time = *time;
  record.status_bits = status;
  record.fatal_status_bits = fatal_status_.load();
  record.capture_overflows = capture_overflows_.load();
  record.invalid_frames = invalid_frames_.load();
  record.invalid_timing = invalid_timing_.load();
  record.capture_occupancy = w - r + 1;
  record.startup_discarded_before = startup_discarded_.load();
  record.rendered = false;

  auto rr = render_read_.load(std::memory_order_relaxed);
  const auto rw = render_write_.load(std::memory_order_acquire);
  const auto selected_epoch = epoch_.load();
  // Only the callback ever advances render_read_. A cancellation does not
  // recycle a slot while this callback still owns a pointer to it.
  for (std::size_t scanned = 0; rr != rw && scanned < render_capacity_; ++scanned) {
    const auto &slot = render_[rr % render_capacity_];
    if (slot.epoch != selected_epoch) {
      ++rr;
      continue;
    }
    if (!active_.load() || !admission_.load() || Faulted()) break;
    std::memcpy(output, slot.pcm.data(), kFrameBytes);
    // Output commit linearizes at this final fence observation. Cancellation
    // after it cannot recall this callback or samples already in device buffers.
    if (active_.load() && admission_.load() && !Faulted() &&
        epoch_.load() == selected_epoch) {
      record.rendered = true;
      record.render = {generation_, slot.submission_id, selected_epoch,
                       reference_sequence_++, time->outputBufferDacTime,
                       time->outputBufferDacTime + 0.01};
      if (!has_playback_) playback_start_ = time->outputBufferDacTime;
      has_playback_ = true;
      playback_start_ = std::min(playback_start_, time->outputBufferDacTime);
      playback_end_ = std::max(playback_end_, time->outputBufferDacTime + 0.01);
      PublishReceipt(record.render);
    } else {
      std::memset(output, 0, kFrameBytes);
    }
    ++rr;
    break;
  }
  render_read_.store(rr, std::memory_order_release);
  record.render_occupancy = rw - rr;
  std::memcpy(record.output_pcm16.data(), output, kFrameBytes);
  record.playback_context_valid = !Faulted();
  record.has_playback = has_playback_;
  // Conservative envelope only: it does NOT prove continuous playback across
  // phrase gaps. Consumers refine using their bounded actual-reference history.
  record.playback_start_dac_time = playback_start_;
  record.playback_end_dac_time = playback_end_;
  capture_write_.store(w + 1, std::memory_order_release);
  return 0; // paContinue; empty output is not a fabricated device underflow.
}

} // namespace tldw::duplex
