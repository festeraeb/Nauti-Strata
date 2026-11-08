// src/poly/ram_window.hpp - the places in the expert helper's RAM window (strata-expert-helper --ram-mib).
//
// The window is one pinned, RDMA-registered buffer: registration is per buffer and costs a pin of every page, so
// it is done once at start, and experts take places in it as they are held and give them back when released.
// Experts come in a few sizes (a layer's are all one size; a model mixes a handful of quantizations), so a place
// given back is kept in a list by its size and the next expert of that size takes it; a new size takes the next
// unused bytes (a bump).  When neither has room, a larger free place is taken whole: the waste is the difference
// between two of the model's expert sizes, and the place goes back to its own size's list.  No CUDA, no verbs, so
// expert_wire_test checks it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace strata::poly {

class RamWindow {
public:
    static constexpr uint64_t kAlign = 4096;   // a place starts on a page: the CPU's reads and the NIC's DMA like it

    explicit RamWindow(uint64_t bytes = 0) : bytes_(bytes) {}

    uint64_t bytes() const { return bytes_; }
    uint64_t used() const { return used_; }   ///< bytes in places that are taken (each its place's whole size)

    static uint64_t place_bytes(uint64_t want) { return (want + kAlign - 1) / kAlign * kAlign; }

    /// A place of at least `want` bytes: its offset from the window's start, and in `got` its size (to give back).
    /// False when the window has no room for it.
    bool take(uint64_t want, uint64_t& offset, uint64_t& got) {
        if (want == 0) return false;
        const uint64_t b = place_bytes(want);
        auto it = free_.find(b);
        if (it == free_.end() || it->second.empty()) {
            if (bump_ + b <= bytes_) {
                offset = bump_;
                bump_ += b;
                got = b;
                used_ += b;
                return true;
            }
            it = free_.upper_bound(b);   // the smallest larger size with a free place
            while (it != free_.end() && it->second.empty()) ++it;
            if (it == free_.end()) return false;
        }
        offset = it->second.back();
        it->second.pop_back();
        got = it->first;
        used_ += got;
        return true;
    }

    /// Gives back a place take() gave: its offset and its size as take() said.
    void give(uint64_t offset, uint64_t size) {
        free_[size].push_back(offset);
        used_ -= size;
    }

private:
    uint64_t bytes_ = 0, bump_ = 0, used_ = 0;
    std::map<uint64_t, std::vector<uint64_t>> free_;   ///< by a place's size: the free ones' offsets
};

}  // namespace strata::poly
