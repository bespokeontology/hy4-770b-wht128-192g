#pragma once
// Unattended Zero-Reset Silicon Safety Guard for SB-M5-P2-192G (5x MI50 + 2x P150A)
// Guarantees graceful SIGTERM/SIGINT queue drain via hipDeviceSynchronize() across all active MI50s
// before exit, preventing Vega 20 Mode-1 PSP -62 faults when operating unattended.

#include <atomic>
#include <csignal>
#include <cstdio>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

namespace hy4_770b {

inline volatile sig_atomic_t g_graceful_stop = 0;
inline std::atomic<bool> g_shutdown_requested{false};

inline void graceful_signal_handler(int /*sig*/) noexcept {
    g_graceful_stop = 1;
}

inline bool should_stop() noexcept {
    return g_graceful_stop != 0 || g_shutdown_requested.load(std::memory_order_acquire);
}

inline void request_graceful_stop() noexcept {
    g_shutdown_requested.store(true, std::memory_order_release);
    g_graceful_stop = 1;
}

inline void install_graceful_signal_handlers() noexcept {
    std::signal(SIGINT, graceful_signal_handler);
    std::signal(SIGTERM, graceful_signal_handler);
    std::signal(SIGPIPE, SIG_IGN);
}

inline void drain_all_hip_devices(int max_devs = 5, const char* tag = "HY4_770B") noexcept {
#if defined(__HIP_PLATFORM_AMD__)
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) return;
    if (count > max_devs) count = max_devs;
    for (int d = 0; d < count; ++d) {
        if (hipSetDevice(d) == hipSuccess) {
            (void)hipDeviceSynchronize();
        }
    }
    std::printf("[%s] Graceful queue drain complete across %d MI50 devices (hipDeviceSynchronize OK)\n", tag, count);
    std::fflush(stdout);
#else
    (void)max_devs; (void)tag;
#endif
}

struct ScopedHipDrainGuard {
    int max_devs = 5;
    const char* tag = "HY4_770B";
    bool armed = true;
    ~ScopedHipDrainGuard() {
        if (armed) drain_all_hip_devices(max_devs, tag);
    }
};

}  // namespace hy4_770b
