// src/poly/expert_tier.cpp - the heat-ordered expert file: written whole through a temporary, mapped for reading.
#include "expert_tier.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

namespace strata::poly {
namespace {

[[noreturn]] void fail(const std::string& path, const char* what) {
    throw std::runtime_error(path + ": " + what + (errno ? std::string(": ") + std::strerror(errno) : std::string()));
}

// Writes all of `n` bytes at `at`, in as many calls as the kernel takes.
void write_all(int fd, const std::string& path, const void* src, uint64_t n, uint64_t at) {
    const uint8_t* p = (const uint8_t*) src;
    while (n > 0) {
        const ssize_t w = pwrite(fd, p, (size_t) n, (off_t) at);
        if (w < 0) {
            if (errno == EINTR) continue;
            fail(path, "write");
        }
        if (w == 0) {
            errno = 0;
            fail(path, "write wrote nothing");
        }
        p += w;
        n -= (uint64_t) w;
        at += (uint64_t) w;
    }
}

}  // namespace

constexpr char ExpertTier::kMagic[8];

uint64_t ExpertTier::id_of(const std::string& s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    return h;
}

ExpertTier::Room ExpertTier::room(uint64_t capacity, uint64_t meta_bytes, uint64_t free_bytes, uint64_t old_bytes) {
    const uint64_t total = free_bytes + old_bytes;
    Room r{};
    r.capacity = total > meta_bytes ? std::min(capacity, total - meta_bytes) : 0;
    r.drop_old = old_bytes > 0 && r.capacity + meta_bytes > free_bytes;
    return r;
}

size_t ExpertTier::write(const std::string& path, uint32_t n_layer, uint32_t n_expert, uint64_t model_id,
                         uint64_t profile_id, const std::vector<std::pair<int32_t, int32_t>>& order, const Fetch& fetch,
                         uint64_t capacity) {
    // as much as the filesystem holds: its free space plus the old file this one replaces
    {
        const uint64_t meta = sizeof(Header) + (uint64_t) n_layer * n_expert * sizeof(Entry) + (1ull << 20);
        std::filesystem::path dir = std::filesystem::path(path).parent_path();
        if (dir.empty()) dir = ".";
        struct statvfs vs{};
        struct stat old{};
        const bool have_old = ::stat(path.c_str(), &old) == 0;
        if (::statvfs(dir.c_str(), &vs) == 0) {
            const Room r = room(capacity, meta, (uint64_t) vs.f_bavail * vs.f_frsize, have_old ? (uint64_t) old.st_size : 0);
            capacity = r.capacity;
            if (r.drop_old) ::unlink(path.c_str());
        }
    }
    // which experts fit: whole ones, hottest first, each once
    std::vector<Entry> entries;
    std::vector<TierSpan> spans;
    std::unordered_map<uint64_t, size_t> seen;
    uint64_t used = 0;
    for (const auto& [layer, expert] : order) {
        if (layer < 0 || expert < 0 || (uint32_t) layer >= n_layer || (uint32_t) expert >= n_expert) continue;
        if (seen.count(key(layer, expert))) continue;
        TierSpan span;
        uint64_t gu = 0, d = 0;
        if (!fetch(layer, expert, span, gu, d) || !span.gate || !span.up || !span.down) continue;
        const uint64_t bytes = 2 * gu + d;
        if (used + bytes > capacity) continue;   // a smaller one later may still fit
        seen.emplace(key(layer, expert), entries.size());
        entries.push_back(Entry{layer, expert, 0, gu, d});
        spans.push_back(span);
        used += bytes;
    }
    Header h{};
    std::memcpy(h.magic, kMagic, sizeof(kMagic));
    h.version = kVersion;
    h.n_layer = n_layer;
    h.n_expert = n_expert;
    h.model_id = model_id;
    h.profile_id = profile_id;
    h.count = entries.size();
    h.data_offset = sizeof(Header) + entries.size() * sizeof(Entry);
    h.data_bytes = used;
    uint64_t at = h.data_offset;
    for (Entry& e : entries) {
        e.offset = at;
        at += 2 * e.gu_bytes + e.d_bytes;
    }

    const std::string tmp = path + ".tmp";
    errno = 0;
    const int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
    if (fd < 0) fail(tmp, "cannot create");
    try {
        write_all(fd, tmp, &h, sizeof(h), 0);
        if (!entries.empty()) write_all(fd, tmp, entries.data(), entries.size() * sizeof(Entry), sizeof(Header));
        for (size_t i = 0; i < entries.size(); ++i) {
            const Entry& e = entries[i];
            write_all(fd, tmp, spans[i].gate, e.gu_bytes, e.offset);
            write_all(fd, tmp, spans[i].up, e.gu_bytes, e.offset + e.gu_bytes);
            write_all(fd, tmp, spans[i].down, e.d_bytes, e.offset + 2 * e.gu_bytes);
        }
        if (fsync(fd) != 0) fail(tmp, "fsync");
    } catch (...) {
        ::close(fd);
        ::unlink(tmp.c_str());
        throw;
    }
    ::close(fd);
    errno = 0;
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        ::unlink(tmp.c_str());
        fail(path, "rename");
    }
    return entries.size();
}

void ExpertTier::open(const std::string& path, uint64_t model_id) {
    close();
    errno = 0;
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) fail(path, "cannot open");
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        fail(path, "stat");
    }
    const size_t size = (size_t) st.st_size;
    auto refuse = [&](const char* what) {
        ::close(fd);
        errno = 0;
        fail(path, what);
    };
    if (size < sizeof(Header)) refuse("not a tier file (shorter than its header)");
    void* m = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) refuse("mmap");
    ::close(fd);
    Header h{};
    std::memcpy(&h, m, sizeof(h));
    auto drop = [&](const char* what) {
        munmap(m, size);
        errno = 0;
        fail(path, what);
    };
    if (std::memcmp(h.magic, kMagic, sizeof(kMagic)) != 0) drop("not a tier file");
    if (h.version != kVersion) drop("version is not this engine's");
    if (h.model_id != model_id) drop("another model's tier");
    const uint64_t table = sizeof(Header) + h.count * sizeof(Entry);
    if (h.data_offset < table || h.data_offset + h.data_bytes > size) drop("truncated");
    std::vector<Entry> entries((size_t) h.count);
    if (h.count) std::memcpy(entries.data(), (const uint8_t*) m + sizeof(Header), entries.size() * sizeof(Entry));
    std::unordered_map<uint64_t, size_t> index;
    for (size_t i = 0; i < entries.size(); ++i) {
        const Entry& e = entries[i];
        if (e.layer < 0 || e.expert < 0 || (uint32_t) e.layer >= h.n_layer || (uint32_t) e.expert >= h.n_expert ||
            e.offset < h.data_offset || e.offset + 2 * e.gu_bytes + e.d_bytes > size)
            drop("truncated");
        index[key(e.layer, e.expert)] = i;
    }
    map_ = m;
    map_bytes_ = size;
    header_ = h;
    entries_ = std::move(entries);
    index_ = std::move(index);
}

void ExpertTier::close() {
    if (map_) munmap(map_, map_bytes_);
    map_ = nullptr;
    map_bytes_ = 0;
    header_ = Header{};
    entries_.clear();
    index_.clear();
}

ExpertTier::~ExpertTier() { close(); }

bool ExpertTier::find(int layer, int expert, TierSpan& span) const {
    span = TierSpan{};
    if (!map_) return false;
    const auto it = index_.find(key(layer, expert));
    if (it == index_.end()) return false;
    const Entry& e = entries_[it->second];
    const uint8_t* base = (const uint8_t*) map_ + e.offset;
    span.gate = base;
    span.up = base + e.gu_bytes;
    span.down = base + 2 * e.gu_bytes;
    return true;
}

void ExpertTier::will_need() const {
    if (!map_ || header_.data_bytes == 0) return;
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t from = (size_t) header_.data_offset / page * page;
    madvise((uint8_t*) map_ + from, map_bytes_ - from, MADV_WILLNEED);
}

}  // namespace strata::poly
