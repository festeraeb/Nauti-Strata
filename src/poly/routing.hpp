// src/poly/routing.hpp - the experts a request's tokens were routed to, for NautiBurn's scrapes.
//
// Every engine we run writes NautiBurn's scrape files (the operator's rule): the routing a request took, a token's
// experts in each layer, which the fleet's puller collects into the histogram database.  The engine fills this as
// its layers route (moe(), moe_rows()), at the token's position; a split's cards write their own layers into the
// one record.  A guess the answer did not keep is written over when its position is read again, so what is left is
// the answer's.  main.cpp writes the files once a request is done (write_scrape()).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::poly {

class Routing {
public:
    /// Room for `positions` tokens of `layers` layers, `used` experts each; everything not routed yet.
    void reset(int64_t positions, int layers, int used) {
        positions_ = positions;
        layers_ = layers;
        used_ = used;
        ids_.assign((size_t) positions * (size_t) layers * (size_t) used, -1);
    }
    /// Forgets what positions [from, to) were routed to (a new request reads them again).
    void clear(int64_t from, int64_t to) {
        from = std::max<int64_t>(from, 0);
        to = std::min(to, positions_);
        if (to > from)
            std::fill(ids_.begin() + (ptrdiff_t) (from * row()), ids_.begin() + (ptrdiff_t) (to * row()), (int16_t) -1);
    }
    /// Layer il of the token at `pos` went to experts e[0..n).  Out of range is ignored: a draft block's layer,
    /// or a position past the record.
    void put(int64_t pos, int il, const int32_t* e, int n) {
        if (pos < 0 || pos >= positions_ || il < 0 || il >= layers_) return;
        int16_t* d = ids_.data() + (size_t) pos * row() + (size_t) il * used_;
        for (int k = 0; k < used_; ++k) d[k] = k < n ? (int16_t) e[k] : (int16_t) -1;
    }
    /// The experts of layer il at pos, or nullptr when it was not routed.
    const int16_t* at(int64_t pos, int il) const {
        if (pos < 0 || pos >= positions_ || il < 0 || il >= layers_) return nullptr;
        const int16_t* d = ids_.data() + (size_t) pos * row() + (size_t) il * used_;
        return d[0] < 0 ? nullptr : d;
    }
    int layers() const { return layers_; }
    int used() const { return used_; }
    bool on() const { return !ids_.empty(); }

    /// One scrape file, NautiBurn's schema 1: positions [from, to) with their token ids.  Written as a hidden
    /// temporary and renamed, so the puller never takes half a file.  False when nothing was routed there or the
    /// file cannot be written.
    bool write(const std::string& path, const std::string& model_id, const std::string& class_id,
               const std::string& request_id, const char* phase, int64_t from, const std::vector<int32_t>& tokens) const {
        const int64_t to = from + (int64_t) tokens.size();
        bool any = false;
        for (int64_t p = from; p < to && !any; ++p)
            for (int il = 0; il < layers_ && !any; ++il) any = at(p, il) != nullptr;
        if (!any) return false;
        const size_t slash = path.rfind('/');
        const std::string tmp = path.substr(0, slash + 1) + "." + path.substr(slash + 1) + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "w");
        if (!f) return false;
        std::fprintf(f, "{\"schema_version\":1,\"model_id\":\"%s\",\"request_id\":\"%s\",\"class_id\":\"%s\","
                        "\"source\":\"scrape\",\"phase\":\"%s\",\"n_tokens\":%zu,\"dtype\":\"gguf\",\"token_ids\":[",
                     model_id.c_str(), request_id.c_str(), class_id.c_str(), phase, tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) std::fprintf(f, "%s%d", i ? "," : "", (int) tokens[i]);
        std::fprintf(f, "],\"activations\":[");
        bool first = true;
        for (int64_t p = from; p < to; ++p)
            for (int il = 0; il < layers_; ++il) {
                const int16_t* e = at(p, il);
                if (!e) continue;
                std::fprintf(f, "%s{\"layer\":%d,\"token_pos\":%lld,\"expert_ids\":[", first ? "" : ",", il,
                             (long long) (p - from));
                first = false;
                for (int k = 0; k < used_ && e[k] >= 0; ++k) std::fprintf(f, "%s%d", k ? "," : "", (int) e[k]);
                std::fprintf(f, "]}");
            }
        std::fprintf(f, "]}\n");
        const bool ok = std::fclose(f) == 0 && std::rename(tmp.c_str(), path.c_str()) == 0;
        if (!ok) std::remove(tmp.c_str());
        return ok;
    }

private:
    size_t row() const { return (size_t) layers_ * (size_t) used_; }
    int64_t positions_ = 0;
    int layers_ = 0, used_ = 0;
    std::vector<int16_t> ids_;
};

}  // namespace strata::poly
