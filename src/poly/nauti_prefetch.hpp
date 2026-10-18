#pragma once

#include <cstdint>
#include <atomic>
#include <array>
#include <chrono>
#include <string>
#include <sstream>

namespace nauti {

constexpr size_t kPrefetchRingSize = 1024;

enum class PrefetchOutcome : uint8_t {
    PENDING_IN_FLIGHT     = 0,
    HIT_PREFETCHED        = 1, // Expert arrived in VRAM before forward pass needed it
    HIT_JUST_IN_TIME      = 2, // Arrived within execution grace period
    STALL_IN_FLIGHT       = 3, // Compute arrived while DMA was still copying
    UNUSED_FALSE_POSITIVE = 4, // Prefetched into VRAM but never routed
    ALREADY_RESIDENT      = 5, // Expert was already resident in VRAM
};

inline const char* outcome_to_string(PrefetchOutcome outcome) {
    switch (outcome) {
        case PrefetchOutcome::PENDING_IN_FLIGHT:     return "PENDING_IN_FLIGHT";
        case PrefetchOutcome::HIT_PREFETCHED:        return "HIT_PREFETCHED";
        case PrefetchOutcome::HIT_JUST_IN_TIME:      return "HIT_JUST_IN_TIME";
        case PrefetchOutcome::STALL_IN_FLIGHT:       return "STALL_IN_FLIGHT";
        case PrefetchOutcome::UNUSED_FALSE_POSITIVE: return "UNUSED_FALSE_POSITIVE";
        case PrefetchOutcome::ALREADY_RESIDENT:      return "ALREADY_RESIDENT";
        default:                                     return "UNKNOWN";
    }
}

struct alignas(64) PrefetchEntry {
    uint64_t request_id;
    int32_t  layer_idx;
    int32_t  expert_idx;
    int32_t  lead_layers;
    uint64_t timestamp_us;
    std::atomic<uint8_t> outcome;
    uint8_t  pad[31];
};

static_assert(sizeof(PrefetchEntry) == 64, "PrefetchEntry must be 64-byte cache-line aligned");

class PrefetchTracker {
public:
    PrefetchTracker() : head_(0), next_id_(1) {
        for (auto& entry : entries_) {
            entry.request_id = 0;
            entry.layer_idx = -1;
            entry.expert_idx = -1;
            entry.lead_layers = 0;
            entry.timestamp_us = 0;
            entry.outcome.store(static_cast<uint8_t>(PrefetchOutcome::PENDING_IN_FLIGHT), std::memory_order_relaxed);
        }
        total_requests_.store(0, std::memory_order_relaxed);
        hits_prefetched_.store(0, std::memory_order_relaxed);
        hits_jit_.store(0, std::memory_order_relaxed);
        stalls_in_flight_.store(0, std::memory_order_relaxed);
        wasted_transfers_.store(0, std::memory_order_relaxed);
        already_resident_.store(0, std::memory_order_relaxed);
    }

    static inline uint64_t now_us() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
    }

    uint64_t record_request(int il, int e, int lead) {
        uint64_t req_id = next_id_.fetch_add(1, std::memory_order_relaxed);
        size_t idx = req_id % kPrefetchRingSize;
        PrefetchEntry& entry = entries_[idx];

        entry.request_id = req_id;
        entry.layer_idx = il;
        entry.expert_idx = e;
        entry.lead_layers = lead;
        entry.timestamp_us = now_us();
        entry.outcome.store(static_cast<uint8_t>(PrefetchOutcome::PENDING_IN_FLIGHT), std::memory_order_release);

        total_requests_.fetch_add(1, std::memory_order_relaxed);
        return req_id;
    }

    void record_outcome(uint64_t req_id, PrefetchOutcome outcome) {
        if (req_id == 0) return;
        size_t idx = req_id % kPrefetchRingSize;
        PrefetchEntry& entry = entries_[idx];
        if (entry.request_id == req_id) {
            entry.outcome.store(static_cast<uint8_t>(outcome), std::memory_order_release);
        }

        switch (outcome) {
            case PrefetchOutcome::HIT_PREFETCHED:
                hits_prefetched_.fetch_add(1, std::memory_order_relaxed);
                break;
            case PrefetchOutcome::HIT_JUST_IN_TIME:
                hits_jit_.fetch_add(1, std::memory_order_relaxed);
                break;
            case PrefetchOutcome::STALL_IN_FLIGHT:
                stalls_in_flight_.fetch_add(1, std::memory_order_relaxed);
                break;
            case PrefetchOutcome::UNUSED_FALSE_POSITIVE:
                wasted_transfers_.fetch_add(1, std::memory_order_relaxed);
                break;
            case PrefetchOutcome::ALREADY_RESIDENT:
                already_resident_.fetch_add(1, std::memory_order_relaxed);
                break;
            default:
                break;
        }
    }

    std::string to_json() const {
        uint64_t total = total_requests_.load(std::memory_order_relaxed);
        uint64_t hits = hits_prefetched_.load(std::memory_order_relaxed);
        uint64_t jit = hits_jit_.load(std::memory_order_relaxed);
        uint64_t stalls = stalls_in_flight_.load(std::memory_order_relaxed);
        uint64_t wasted = wasted_transfers_.load(std::memory_order_relaxed);
        uint64_t resident = already_resident_.load(std::memory_order_relaxed);

        double hit_rate = (total > resident) ? (static_cast<double>(hits + jit) / static_cast<double>(total - resident) * 100.0) : 100.0;

        std::ostringstream ss;
        ss << "{\n"
           << "  \"total_prefetch_requests\": " << total << ",\n"
           << "  \"hits_prefetched\": " << hits << ",\n"
           << "  \"hits_just_in_time\": " << jit << ",\n"
           << "  \"stalls_in_flight\": " << stalls << ",\n"
           << "  \"wasted_false_positives\": " << wasted << ",\n"
           << "  \"already_resident\": " << resident << ",\n"
           << "  \"effective_hit_rate_pct\": " << hit_rate << "\n"
           << "}";
        return ss.str();
    }

private:
    std::array<PrefetchEntry, kPrefetchRingSize> entries_;
    std::atomic<uint64_t> head_;
    std::atomic<uint64_t> next_id_;

    std::atomic<uint64_t> total_requests_;
    std::atomic<uint64_t> hits_prefetched_;
    std::atomic<uint64_t> hits_jit_;
    std::atomic<uint64_t> stalls_in_flight_;
    std::atomic<uint64_t> wasted_transfers_;
    std::atomic<uint64_t> already_resident_;
};

inline PrefetchTracker& global_prefetch_tracker() {
    static PrefetchTracker tracker;
    return tracker;
}

} // namespace nauti

// Fast C API for external routers (CubeCL NautiBurn / Lemonade Ori)
extern "C" {
    inline uint64_t nauti_record_prefetch_request(int il, int e, int lead) {
        return nauti::global_prefetch_tracker().record_request(il, e, lead);
    }

    inline void nauti_record_prefetch_outcome(uint64_t req_id, uint8_t outcome) {
        nauti::global_prefetch_tracker().record_outcome(req_id, static_cast<nauti::PrefetchOutcome>(outcome));
    }

    inline const char* nauti_get_prefetch_metrics_json() {
        static std::string json_cache;
        json_cache = nauti::global_prefetch_tracker().to_json();
        return json_cache.c_str();
    }
}
