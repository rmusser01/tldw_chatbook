// Test-only PortAudio ABI caller. No PortAudio library or Python callback.
#include <cstdint>
#include <chrono>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#define PA_CALL __cdecl
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#define PA_CALL
#endif

struct TimeInfo {
  double inputBufferAdcTime;
  double currentTime;
  double outputBufferDacTime;
};
using Callback = int(PA_CALL *)(const void *, void *, unsigned long,
                               const TimeInfo *, unsigned long, void *);

EXPORT int emit_callback(std::uintptr_t callback, std::uintptr_t userdata,
                         const void *input, void *output, unsigned long frames,
                         const TimeInfo *times, unsigned long status) {
  return reinterpret_cast<Callback>(callback)(input, output, frames, times,
                                               status,
                                               reinterpret_cast<void *>(userdata));
}

// Invoke through ctypes.PyDLL: the calling Python thread retains the GIL
// throughout the join, while a distinct native thread completes callbacks.
EXPORT unsigned long progress_with_gil_held(std::uintptr_t callback,
                                           std::uintptr_t userdata,
                                           unsigned long count) {
  unsigned long completed = 0;
  std::thread worker([&] {
    for (unsigned long i = 0; i < count; ++i) {
      std::int16_t input[480];
      std::int16_t output[480];
      for (auto &sample : input) sample = static_cast<std::int16_t>(i + 1);
      const TimeInfo times{10.0 + i * 0.01, 10.02 + i * 0.01,
                           10.04 + i * 0.01};
      emit_callback(callback, userdata, input, output, 480, &times, 0);
      ++completed;
    }
  });
  worker.join();
  return completed;
}

// One paced batch for mounted GC testing. Device clocks use the intended
// 10 ms cadence; callback observation remains the real native clock.
struct Observation {
  std::uint64_t callback_ns;
  std::uint64_t completed_ns;
  std::int16_t output_sample;
};

static unsigned long paced_progress(
    std::uintptr_t callback, std::uintptr_t userdata, unsigned long count,
    std::uint64_t *first_current_ns, Observation *observations) {
  using Clock = std::chrono::steady_clock;
  unsigned long completed = 0;
  std::thread worker([&] {
    const auto first = Clock::now() + std::chrono::milliseconds(10);
    *first_current_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        first.time_since_epoch()).count();
    for (unsigned long i = 0; i < count; ++i) {
      const auto deadline = first + std::chrono::milliseconds(10 * i);
      std::this_thread::sleep_until(deadline);
      std::int16_t input[480];
      std::int16_t output[480];
      for (auto &sample : input) sample = static_cast<std::int16_t>(i + 1);
      const double current = (*first_current_ns + i * 10000000ULL) / 1e9;
      const TimeInfo times{current - .01, current, current + .01};
      if (observations) observations[i].callback_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              Clock::now().time_since_epoch()).count();
      emit_callback(callback, userdata, input, output, 480, &times, 0);
      if (observations) {
        observations[i].output_sample = output[0];
        __atomic_store_n(&observations[i].completed_ns,
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now().time_since_epoch()).count(), __ATOMIC_RELEASE);
      }
      ++completed;
    }
  });
  worker.join();
  return completed;
}

EXPORT unsigned long paced_progress_with_gil_held(
    std::uintptr_t callback, std::uintptr_t userdata, unsigned long count,
    std::uint64_t *first_current_ns) {
  return paced_progress(callback, userdata, count, first_current_ns, nullptr);
}

EXPORT unsigned long observed_paced_progress(
    std::uintptr_t callback, std::uintptr_t userdata, unsigned long count,
    std::uint64_t *first_current_ns, Observation *observations) {
  return paced_progress(callback, userdata, count, first_current_ns, observations);
}

// Test-only finite interpreter hold. No platform audio/device API is called.
EXPORT void hold_gil_ms(unsigned long duration_ms, std::uint64_t *bounds) {
  using Clock = std::chrono::steady_clock;
  bounds[0] = std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now().time_since_epoch()).count();
  std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
  bounds[1] = std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now().time_since_epoch()).count();
}

EXPORT std::uint64_t driver_monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

EXPORT unsigned long completed_observations(Observation *items, unsigned long capacity) {
  unsigned long count = 0;
  while (count < capacity && __atomic_load_n(&items[count].completed_ns, __ATOMIC_ACQUIRE)) ++count;
  return count;
}
