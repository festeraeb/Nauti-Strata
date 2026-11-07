// src/poly/expert_tier.hpp - a heat-ordered copy of routed experts on a fast local device.
//
// The engine keeps the hottest experts in VRAM and the next warmest in RAM; the rest it reads from the model's
// file, which may be over the network.  Between RAM and the file sits this tier: one file on a fast local disk
// (the Optane module) that holds the next warmest experts, hottest first, which the engine mmaps and reads an
// expert from as it would from the model's file.  The engine writes the file itself, from the learned expert
// profile, and rewrites it when the profile has moved on (profile_id).  The kernel's page cache alone is not the
// design: the file's order is the heat order, so the bytes that matter are the ones that stay mapped.
//
// Layout: Header, then count Entry records, then the experts' bytes, each expert gate | up | down at its offset
// from the start of the file.  Little-endian, as every machine this runs on is.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::poly {

/// An expert's three matrices as bytes: in the model's file, or in the tier.
struct TierSpan {
    const uint8_t* gate = nullptr;
    const uint8_t* up = nullptr;
    const uint8_t* down = nullptr;
};

class ExpertTier {
public:
    static constexpr char kMagic[8] = {'S', 'T', 'R', 'T', 'I', 'E', 'R', '1'};
    static constexpr uint32_t kVersion = 1;

    struct Header {
        char magic[8];
        uint32_t version;
        uint32_t n_layer;
        uint32_t n_expert;
        uint32_t reserved;
        uint64_t model_id;     ///< ExpertTier::id_of() of what names the model's file: path, size and tensor count
        uint64_t profile_id;   ///< the profile the order came from (id_of of its path and its counts), or 0
        uint64_t count;        ///< Entry records after the header
        uint64_t data_offset;  ///< the first expert's bytes
        uint64_t data_bytes;   ///< all the experts' bytes together
    };
    static_assert(sizeof(Header) == 64, "Header is 64 bytes");

    struct Entry {
        int32_t layer;
        int32_t expert;
        uint64_t offset;     ///< from the file's start; the gate, then the up at offset + gu_bytes, then the down
        uint64_t gu_bytes;   ///< one of the gate and the up
        uint64_t d_bytes;
    };
    static_assert(sizeof(Entry) == 32, "Entry is 32 bytes");

    /// The bytes of a layer's expert in the model: its spans and their sizes.  false: the tier leaves it out.
    using Fetch = std::function<bool(int layer, int expert, TierSpan& span, uint64_t& gu_bytes, uint64_t& d_bytes)>;

    /// What a rewrite of a tier may take on its filesystem.  `free_bytes` is the space free there now and
    /// `old_bytes` the file at the tier's name that the rewrite replaces; `meta_bytes` covers the header, the entries
    /// and the filesystem's own overhead.  The budget shrinks to what fits; when free space alone cannot hold the
    /// new file beside the old one, the old one is removed first (the tier is a cache: losing it costs a reload).
    struct Room {
        uint64_t capacity;   ///< expert bytes the rewrite may write
        bool drop_old;       ///< remove the old file before writing the new one
    };
    static Room room(uint64_t capacity, uint64_t meta_bytes, uint64_t free_bytes, uint64_t old_bytes);

    /// Whether the file `name` in the tier's directory is a tier of `base` (the tier path's file name) that this
    /// load does not use: on a split, "<base>.L<n>" for an n no card starts at, and "<base>" itself; on one card,
    /// every "<base>.L<n>".  A changed split names its files by its own first layers, so the old split's files
    /// would otherwise sit beside them and fill the disk (c4's Optane, 2026-11-06).  Other names are never stale.
    static bool stale(const std::string& name, const std::string& base, const std::vector<int>& starts);
    /// Removes the stale tiers of `path` (stale()), returning how many bytes went.  Errors are skipped: a tier is
    /// a cache, and one that cannot be removed only costs the room it takes.
    static uint64_t drop_stale(const std::string& path, const std::vector<int>& starts);

    /// The tier files of an --expert-tier value: comma-separated paths, one a device, each "path" or
    /// "path@MiB" (the expert bytes it holds; `mib` when not given).  c4's tier is five Optane modules, a 16 GB
    /// one and four on an x16 bifurcation card (2026-11-07): a path each, so that they are read side by side.
    struct Spec {
        std::string path;
        int mib;
    };
    static std::vector<Spec> parse(const std::string& list, int mib);

    /// Deals `order` (warmest first) out to devices that hold `caps` bytes each: an expert goes to the device
    /// least full for its size once it is in, among equals the first, and is left out when none has room.  The
    /// hot experts so spread over every device, and a token's misses read from several at once.
    static std::vector<std::vector<std::pair<int32_t, int32_t>>> deal(
        const std::vector<std::pair<int32_t, int32_t>>& order, const std::function<uint64_t(int32_t layer)>& bytes,
        const std::vector<uint64_t>& caps);

    /// Writes the tier file at `path`: the experts of `order`, hottest first, as many whole ones as fit in
    /// `capacity` bytes of expert data (the header and entries come on top), less if the filesystem holds less (see
    /// room()).  The file is written as path + ".tmp"
    /// and renamed over `path` once complete, so a crash leaves no half tier under the name.  Returns the experts
    /// written.  Throws std::runtime_error naming the file on an I/O error.
    static size_t write(const std::string& path, uint32_t n_layer, uint32_t n_expert, uint64_t model_id,
                        uint64_t profile_id, const std::vector<std::pair<int32_t, int32_t>>& order, const Fetch& fetch,
                        uint64_t capacity);

    /// Maps the tier at `path` for reading.  Throws std::runtime_error when the file cannot be opened, is not a
    /// tier file ("not a tier file"), is of another version ("version"), is for another model ("another model"),
    /// or is shorter than its header says ("truncated").
    void open(const std::string& path, uint64_t model_id);
    void close();
    ~ExpertTier();
    ExpertTier() = default;
    ExpertTier(const ExpertTier&) = delete;
    ExpertTier& operator=(const ExpertTier&) = delete;

    bool mapped() const { return map_ != nullptr; }
    /// True when p points into this tier's mapping (an expert's bytes are read from it).
    bool holds(const void* p) const { return map_ && p >= map_ && p < (const void*) ((const uint8_t*) map_ + map_bytes_); }
    size_t count() const { return entries_.size(); }
    uint64_t data_bytes() const { return header_.data_bytes; }
    uint64_t profile_id() const { return header_.profile_id; }
    const std::vector<Entry>& entries() const { return entries_; }

    /// The expert's bytes in the tier.  false, and null spans, when it is not there.
    bool find(int layer, int expert, TierSpan& span) const;

    /// Tells the kernel the experts' bytes will be read (madvise WILLNEED); no error if it cannot.
    void will_need() const;

    /// FNV-1a 64 of a string: ids for the header.
    static uint64_t id_of(const std::string& s);

private:
    static uint64_t key(int layer, int expert) { return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert; }

    void* map_ = nullptr;
    size_t map_bytes_ = 0;
    Header header_{};
    std::vector<Entry> entries_;
    std::unordered_map<uint64_t, size_t> index_;   ///< key(layer, expert) -> entries_ index
};

}  // namespace strata::poly
