// src/poly/routing_test.cpp - the routing record and the scrape file it writes (no model, no GPU).
#include "routing.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace strata::poly;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

int main() {
    Routing r;
    r.reset(16, 3, 2);
    const int32_t a[2] = {5, 9}, b[2] = {1, 2}, c[2] = {7, 7};
    r.put(4, 1, a, 2);
    check(r.at(4, 1) && r.at(4, 1)[0] == 5 && r.at(4, 1)[1] == 9, "a routed layer reads back");
    check(!r.at(4, 0) && !r.at(3, 1), "a layer or position not routed reads as nothing");
    r.put(4, 1, b, 2);
    check(r.at(4, 1)[0] == 1, "a position read again (a guess not kept) is written over");
    r.put(99, 1, a, 2);
    r.put(4, 7, a, 2);
    check(true, "out of range is ignored");
    r.clear(4, 5);
    check(!r.at(4, 1), "clear forgets a position");

    const std::string dir = (std::filesystem::temp_directory_path() / "routing_test").string();
    std::filesystem::create_directories(dir);
    r.put(10, 1, a, 2);   // the prompt: positions 10 and 11, layer 0 dense (not routed)
    r.put(10, 2, b, 2);
    r.put(11, 1, c, 2);
    const std::string f = dir + "/run-1794000000-1.json";
    check(r.write(f, "mistral-large-4", "live-mistral-large-4", "nb-poly-1794000000-1", "prefill", 10, {100, 101}),
          "a file is written");
    std::ifstream in(f);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    check(s.find("\"phase\":\"prefill\",\"n_tokens\":2") != std::string::npos, "phase and token count");
    check(s.find("\"token_ids\":[100,101]") != std::string::npos, "token ids");
    check(s.find("{\"layer\":1,\"token_pos\":0,\"expert_ids\":[5,9]}") != std::string::npos,
          "positions relative to the file, layer and experts");
    check(s.find("{\"layer\":1,\"token_pos\":1,\"expert_ids\":[7,7]}") != std::string::npos, "the second token");
    check(s.find("\"layer\":0") == std::string::npos, "a layer not routed is left out");
    check(!std::filesystem::exists(dir + "/.run-1794000000-1.json.tmp"), "no temporary is left");
    check(!r.write(dir + "/run-1794000000-2.json", "m", "c", "id", "decode", 12, {5, 6}),
          "nothing routed there: no file");
    check(!std::filesystem::exists(dir + "/run-1794000000-2.json"), "and none appears");
    std::filesystem::remove_all(dir);
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("routing_test: ok\n");
    return 0;
}
