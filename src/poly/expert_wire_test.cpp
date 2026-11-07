// src/poly/expert_wire_test.cpp - the expert helper's messages over a loopback TCP connection.
#include "expert_wire.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace strata::poly::wire;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

int main() {
    try {
        check(model_key(218000000000ull, 128, 6144, 2048) == model_key(218000000000ull, 128, 6144, 2048),
              "the model key is the same for the same model");
        check(model_key(218000000000ull, 128, 6144, 2048) != model_key(218000000001ull, 128, 6144, 2048),
              "the model key differs by the files' bytes");

        Listener l("127.0.0.1:0");
        std::vector<float> x(6144 * 3);
        for (size_t i = 0; i < x.size(); ++i) x[i] = (float) i * 0.5f;
        std::vector<Entry> ents = {{5, 0, 0.25f}, {7, 2, 0.75f}};

        // the helper's side: echoes a Compute's input back as its output, refuses anything else
        std::thread helper([&] {
            Conn c = l.accept();
            std::vector<uint8_t> p;
            for (;;) {
                uint32_t t = 0;
                try {
                    t = c.recv(p);
                } catch (const std::runtime_error&) {
                    return;   // the engine closed
                }
                if (t == kCompute) {
                    ComputeHead h;
                    std::memcpy(&h, p.data(), sizeof(h));
                    const size_t xb = (size_t) h.rows * 6144 * 4;
                    const bool sizes = p.size() == sizeof(h) + xb + h.n * sizeof(Entry);
                    Entry e1;
                    std::memcpy(&e1, p.data() + sizeof(h) + xb + sizeof(Entry), sizeof(e1));
                    if (sizes && e1.expert == 7 && e1.row == 2) c.send(kOk, p.data() + sizeof(h), xb);
                    else c.send(kErr, "bad compute", 11);
                } else {
                    c.send(kErr, "no", 2);
                }
            }
        });

        Conn c = Conn::dial("127.0.0.1:" + std::to_string(l.port()), 2000);
        ComputeHead h{3, 3, 10.f, 0, (uint32_t) ents.size(), 0};
        c.send(kCompute, &h, sizeof(h), x.data(), x.size() * 4, ents.data(), ents.size() * sizeof(Entry));
        std::vector<uint8_t> reply;
        check(c.recv(reply) == kOk, "a compute is answered Ok");
        check(reply.size() == x.size() * 4 && std::memcmp(reply.data(), x.data(), reply.size()) == 0,
              "the three parts arrive whole and in order");
        c.send(kHold, nullptr, 0);
        check(c.recv(reply) == kErr && Conn::text(reply) == "no", "an Err carries its text");
        // a large message, past the socket buffers
        std::vector<float> big((size_t) 6144 * 512);
        h.rows = 512;
        h.n = 2;
        c.send(kCompute, &h, sizeof(h), big.data(), big.size() * 4, ents.data(), ents.size() * sizeof(Entry));
        check(c.recv(reply) == kOk && reply.size() == big.size() * 4, "a 12 MiB message arrives whole");
        c.close();
        helper.join();

        bool threw = false;
        try { Conn::dial("127.0.0.1:1", 500); } catch (const std::runtime_error&) { threw = true; }
        check(threw, "dialling a closed port throws");
        threw = false;
        try { Conn::dial("no-port", 500); } catch (const std::runtime_error&) { threw = true; }
        check(threw, "an address without a port is refused");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "unexpected: %s\n", e.what());
        ++failures;
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("expert_wire_test: ok\n");
    return 0;
}
