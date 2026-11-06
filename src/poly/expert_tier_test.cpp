// src/poly/expert_tier_test.cpp - the heat-ordered expert file: written, mapped, found, refused.
#include "expert_tier.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using strata::poly::ExpertTier;
using strata::poly::TierSpan;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

template <class F>
void throws(F&& f, const char* needle, const std::string& what) {
    try {
        f();
        check(false, what + ": did not throw");
    } catch (const std::runtime_error& e) {
        check(std::string(e.what()).find(needle) != std::string::npos,
              what + ": threw '" + e.what() + "', expected it to say '" + needle + "'");
    }
}

// A fake model: layer il's experts are gu(il) bytes a matrix for the gate and the up and d(il) for the down, with
// bytes that name the layer, the expert and the matrix.
constexpr int kLayers = 3, kExperts = 4;
uint64_t gu_of(int il) { return il == 1 ? 50 : 100; }
uint64_t d_of(int il) { return il == 1 ? 30 : 60; }
uint8_t byte_of(int il, int e, int part, uint64_t i) { return (uint8_t) (il * 131 + e * 17 + part * 53 + i); }

struct FakeModel {
    std::vector<std::vector<uint8_t>> blobs;   // (layer, expert, part)
    const std::vector<uint8_t>& blob(int il, int e, int part) const {
        return blobs[(size_t) ((il * kExperts + e) * 3 + part)];
    }
    FakeModel() {
        for (int il = 0; il < kLayers; ++il)
            for (int e = 0; e < kExperts; ++e)
                for (int part = 0; part < 3; ++part) {
                    const uint64_t n = part == 2 ? d_of(il) : gu_of(il);
                    std::vector<uint8_t> b(n);
                    for (uint64_t i = 0; i < n; ++i) b[i] = byte_of(il, e, part, i);
                    blobs.push_back(std::move(b));
                }
    }
    ExpertTier::Fetch fetch(int missing_layer = -1, int missing_expert = -1) const {
        return [this, missing_layer, missing_expert](int il, int e, TierSpan& s, uint64_t& gu, uint64_t& d) {
            if (il == missing_layer && e == missing_expert) return false;
            s.gate = blob(il, e, 0).data();
            s.up = blob(il, e, 1).data();
            s.down = blob(il, e, 2).data();
            gu = gu_of(il);
            d = d_of(il);
            return true;
        };
    }
};

bool same(const uint8_t* p, int il, int e, int part) {
    const uint64_t n = part == 2 ? d_of(il) : gu_of(il);
    for (uint64_t i = 0; i < n; ++i)
        if (p[i] != byte_of(il, e, part, i)) return false;
    return true;
}

std::string tmpdir() {
    std::string base = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
    std::string tmpl = base + "/expert_tier_test.XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (!mkdtemp(buf.data())) throw std::runtime_error("mkdtemp failed");
    return std::string(buf.data());
}

void test_write_and_find(const std::string& dir) {
    const FakeModel model;
    const std::string path = dir + "/tier.bin";
    const uint64_t model_id = ExpertTier::id_of("fake model"), profile_id = ExpertTier::id_of("profile 1");
    // hottest first: (0,1) 260 bytes, (1,2) 130, (2,3) 260, (0,0) 260, (1,0) 130; room for 650 = the first three
    const std::vector<std::pair<int32_t, int32_t>> order = {{0, 1}, {1, 2}, {2, 3}, {0, 0}, {1, 0}};
    const size_t n = ExpertTier::write(path, kLayers, kExperts, model_id, profile_id, order, model.fetch(), 650);
    check(n == 3, "three experts fit in 650 bytes, got " + std::to_string(n));
    check(!std::filesystem::exists(path + ".tmp"), "no .tmp left behind");
    check(std::filesystem::file_size(path) == 64 + 3 * 32 + 650, "file is header + 3 entries + 650 bytes");

    ExpertTier tier;
    tier.open(path, model_id);
    check(tier.mapped() && tier.count() == 3 && tier.data_bytes() == 650 && tier.profile_id() == profile_id,
          "mapped: 3 experts, 650 bytes, the profile's id");
    check(tier.entries()[0].layer == 0 && tier.entries()[0].expert == 1 && tier.entries()[0].offset == 64 + 96,
          "the hottest expert comes first, right after the table");
    check(tier.entries()[1].offset == 64 + 96 + 260 && tier.entries()[2].offset == 64 + 96 + 260 + 130,
          "experts lie one behind the other");
    for (const auto& [il, e] : {std::pair{0, 1}, std::pair{1, 2}, std::pair{2, 3}}) {
        TierSpan s;
        check(tier.find(il, e, s), "find (" + std::to_string(il) + ", " + std::to_string(e) + ")");
        check(s.gate && same(s.gate, il, e, 0), "gate bytes of (" + std::to_string(il) + ", " + std::to_string(e) + ")");
        check(s.up && same(s.up, il, e, 1), "up bytes");
        check(s.down && same(s.down, il, e, 2), "down bytes");
        check(s.up == s.gate + gu_of(il) && s.down == s.gate + 2 * gu_of(il), "gate | up | down lie together");
    }
    TierSpan none;
    check(!tier.find(0, 0, none) && !none.gate && !none.up && !none.down, "(0, 0) did not fit: not found, null spans");
    check(!tier.find(1, 0, none) && !tier.find(5, 0, none) && !tier.find(0, 9, none), "absent and out-of-range experts");
    tier.will_need();   // no error either way
    tier.close();
    check(!tier.mapped() && tier.count() == 0 && !tier.find(0, 1, none), "closed: nothing mapped or found");
}

void test_edges(const std::string& dir) {
    const FakeModel model;
    const uint64_t model_id = ExpertTier::id_of("fake model");
    // a smaller expert later still fits once a larger one does not
    {
        const std::string path = dir + "/skip.bin";
        const std::vector<std::pair<int32_t, int32_t>> order = {{0, 0}, {0, 1}, {1, 0}};
        const size_t n = ExpertTier::write(path, kLayers, kExperts, model_id, 0, order, model.fetch(), 400);
        check(n == 2, "260 + 130 fit in 400 once the second 260 is passed over, got " + std::to_string(n));
        ExpertTier t;
        t.open(path, model_id);
        TierSpan s;
        check(t.find(0, 0, s) && t.find(1, 0, s) && !t.find(0, 1, s), "the two that fit are there");
    }
    // an expert the model cannot give, an expert named twice, and one out of range are left out
    {
        const std::string path = dir + "/missing.bin";
        const std::vector<std::pair<int32_t, int32_t>> order = {{0, 0}, {0, 0}, {7, 0}, {0, -1}, {1, 1}};
        const size_t n = ExpertTier::write(path, kLayers, kExperts, model_id, 0, order, model.fetch(1, 1), 10000);
        check(n == 1, "only (0, 0) is written, got " + std::to_string(n));
    }
    // no room for even one: a valid empty tier
    {
        const std::string path = dir + "/empty.bin";
        const size_t n = ExpertTier::write(path, kLayers, kExperts, model_id, 0, {{0, 0}}, model.fetch(), 100);
        check(n == 0, "nothing fits in 100 bytes");
        ExpertTier t;
        t.open(path, model_id);
        check(t.mapped() && t.count() == 0 && t.data_bytes() == 0, "an empty tier maps");
        t.will_need();
    }
    // refusals
    {
        const std::string path = dir + "/tier2.bin";
        ExpertTier::write(path, kLayers, kExperts, model_id, 0, {{0, 0}}, model.fetch(), 10000);
        ExpertTier t;
        throws([&] { t.open(path, model_id + 1); }, "another model", "another model's id");
        throws([&] { t.open(dir + "/nowhere.bin", model_id); }, "cannot open", "no such file");
        // not a tier file
        const std::string junk = dir + "/junk.bin";
        {
            std::vector<uint8_t> z(200, 0x5a);
            FILE* f = std::fopen(junk.c_str(), "wb");
            std::fwrite(z.data(), 1, z.size(), f);
            std::fclose(f);
        }
        throws([&] { t.open(junk, model_id); }, "not a tier file", "junk bytes");
        // too short for a header
        const std::string shorty = dir + "/short.bin";
        {
            FILE* f = std::fopen(shorty.c_str(), "wb");
            std::fwrite("STRTIER1", 1, 8, f);
            std::fclose(f);
        }
        throws([&] { t.open(shorty, model_id); }, "not a tier file", "shorter than a header");
        // another version
        const std::string v2 = dir + "/v2.bin";
        std::filesystem::copy_file(path, v2);
        {
            const uint32_t version = 2;
            const int fd = ::open(v2.c_str(), O_WRONLY);
            if (pwrite(fd, &version, sizeof(version), 8) != (ssize_t) sizeof(version)) check(false, "pwrite v2");
            ::close(fd);
        }
        throws([&] { t.open(v2, model_id); }, "version", "another version");
        // cut short: the data the table promises is not there
        const std::string cut = dir + "/cut.bin";
        std::filesystem::copy_file(path, cut);
        std::filesystem::resize_file(cut, 64 + 32 + 100);
        throws([&] { t.open(cut, model_id); }, "truncated", "cut short");
        check(!t.mapped(), "a refused open leaves nothing mapped");
        // after a refusal the same object still opens a good file
        t.open(path, model_id);
        check(t.mapped() && t.count() == 1, "opens after refusals");
    }
    // write replaces an older tier under the same name
    {
        const std::string path = dir + "/replace.bin";
        ExpertTier::write(path, kLayers, kExperts, model_id, 1, {{0, 0}, {0, 1}}, model.fetch(), 10000);
        ExpertTier::write(path, kLayers, kExperts, model_id, 2, {{2, 2}}, model.fetch(), 10000);
        ExpertTier t;
        t.open(path, model_id);
        TierSpan s;
        check(t.count() == 1 && t.profile_id() == 2 && t.find(2, 2, s) && !t.find(0, 0, s), "the newer tier replaced the older");
    }
    // a path that cannot be written names the file
    throws([&] { ExpertTier::write(dir + "/no/such/dir/t.bin", kLayers, kExperts, model_id, 0, {{0, 0}}, model.fetch(), 1000); },
           "cannot create", "unwritable path");
}

void test_ids() {
    check(ExpertTier::id_of("") == 0xcbf29ce484222325ULL, "FNV-1a offset basis for the empty string");
    check(ExpertTier::id_of("a") == 0xaf63dc4c8601ec8cULL, "FNV-1a 64 of 'a'");
    check(ExpertTier::id_of("abc") != ExpertTier::id_of("abd"), "ids differ");
}

}  // namespace

int main() {
    const std::string dir = tmpdir();
    try {
        test_ids();
        test_write_and_find(dir);
        test_edges(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "unexpected: %s\n", e.what());
        ++failures;
    }
    std::filesystem::remove_all(dir);
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("expert_tier_test: ok\n");
    return 0;
}
