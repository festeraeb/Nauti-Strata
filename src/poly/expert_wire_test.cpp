// src/poly/expert_wire_test.cpp - the expert helper's messages over a loopback TCP connection, the engine's client
// (RemoteHelper) against a helper that answers by the protocol, the RAM window's places, and RDMA port names.
#include "expert_rdma.hpp"
#include "expert_wire.hpp"
#include "ram_window.hpp"
#include "remote_helper.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace strata::poly::wire;
using strata::poly::RamWindow;
using strata::poly::RemoteHelper;

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
                } else if (t == kRelease) {   // gives up what it is asked to: answers the count
                    uint32_t n = 0;
                    if (p.size() >= 4) std::memcpy(&n, p.data(), 4);
                    const bool sizes = p.size() == 4 + (size_t) n * sizeof(Pair);
                    if (sizes) c.send(kOk, &n, 4);
                    else c.send(kErr, "bad release", 11);
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
        const std::vector<Pair> drop = {{3, 5}, {9, 7}};
        const uint32_t nd = (uint32_t) drop.size();
        c.send(kRelease, &nd, 4, drop.data(), drop.size() * sizeof(Pair));
        uint32_t freed = 0;
        check(c.recv(reply) == kOk && reply.size() == 4 && (std::memcpy(&freed, reply.data(), 4), freed == 2),
              "a Release names its pairs and is answered with a count");
        // a large message, past the socket buffers
        std::vector<float> big((size_t) 6144 * 512);
        h.rows = 512;
        h.n = 2;
        c.send(kCompute, &h, sizeof(h), big.data(), big.size() * 4, ents.data(), ents.size() * sizeof(Entry));
        check(c.recv(reply) == kOk && reply.size() == big.size() * 4, "a 12 MiB message arrives whole");
        c.close();
        helper.join();

        // version 2: the client against a helper that keeps a RAM window.  Experts of layer 1 fit its "device",
        // the rest go to RAM when the connection asked for it and the Hold allows; an Rdma is answered with the
        // window.  The helper checks every field it is sent, so a wrong encoding shows as a refusal.
        {
            Listener l2("127.0.0.1:0");
            const uint64_t key = model_key(1234, 128, 6144, 2048);
            std::thread fake([&] {
                for (int conn = 0; conn < 2; ++conn) {
                    Conn c = l2.accept();
                    std::vector<uint8_t> p;
                    bool want_ram = false;
                    for (;;) {
                        uint32_t t = 0;
                        try {
                            t = c.recv(p);
                        } catch (const std::runtime_error&) {
                            break;
                        }
                        if (t == kHello) {
                            Hello h{};
                            std::memcpy(&h, p.data(), sizeof(h));
                            if (p.size() != sizeof(h) || h.model_key != key || h.n_expert != 128) {
                                c.send(kErr, "bad hello", 9);
                                continue;
                            }
                            want_ram = (h.flags & kWantRam) != 0;
                            const HelloOk ok{2, kHasRdma, 3ull << 30, 16ull << 30};
                            c.send(kOk, &ok, sizeof(ok));
                        } else if (t == kHold) {
                            uint32_t n = 0, where = 99;
                            std::memcpy(&n, p.data(), 4);
                            std::memcpy(&where, p.data() + 4, 4);
                            if (p.size() != 8 + (size_t) n * sizeof(Pair) || where > kRamOnly) {
                                c.send(kErr, "bad hold", 8);
                                continue;
                            }
                            std::vector<Held> got;
                            for (uint32_t i = 0; i < n; ++i) {
                                Pair q;
                                std::memcpy(&q, p.data() + 8 + i * sizeof(Pair), sizeof(q));
                                const bool dev = q.layer == 1 && where != kRamOnly;
                                const bool ram = !dev && want_ram && where != kDeviceOnly;
                                if (dev) got.push_back({q.layer, q.expert, kOnDevice, 0, 0});
                                else if (ram) got.push_back({q.layer, q.expert, kInRam, 0, 0x10000ull + 4096ull * q.expert});
                            }
                            const uint32_t m = (uint32_t) got.size();
                            c.send(kOk, &m, 4, got.data(), got.size() * sizeof(Held));
                        } else if (t == kRdma) {
                            RdmaQp q{};
                            std::memcpy(&q, p.data(), sizeof(q));
                            if (p.size() != sizeof(q) || q.qpn != 77 || q.gid[15] != 0x3d || q.mtu != 5) {
                                c.send(kErr, "bad rdma", 8);
                                continue;
                            }
                            RdmaOk ok{};
                            ok.qp.qpn = 99;
                            ok.rkey = 0xabcd;
                            ok.base = 0x10000;
                            ok.bytes = 16ull << 30;
                            c.send(kOk, &ok, sizeof(ok));
                        } else {
                            c.send(kErr, "no", 2);
                        }
                    }
                }
            });
            const std::string at = "127.0.0.1:" + std::to_string(l2.port());
            {
                RemoteHelper r(at, key, 128, 6144, 2048, true);
                check(r.capacity() == 3ull << 30 && r.ram_capacity() == 16ull << 30 && r.has_rdma() && r.devices() == 2,
                      "HelloOk carries the device and RAM capacities and the RDMA flag");
                const std::vector<std::pair<int32_t, int32_t>> want = {{1, 3}, {2, 4}, {1, 5}, {7, 9}};
                std::vector<Held> any = r.hold(want, kAny);
                check(any.size() == 4 && any[0].kind == kOnDevice && any[1].kind == kInRam &&
                          any[1].addr == 0x10000ull + 4096 * 4 && any[2].kind == kOnDevice && any[3].layer == 7 &&
                          any[3].expert == 9 && any[3].kind == kInRam,
                      "a Hold anywhere: the device ones and the RAM ones with their window addresses");
                std::vector<Held> dev = r.hold(want, kDeviceOnly);
                check(dev.size() == 2 && dev[0].kind == kOnDevice && dev[1].expert == 5, "a Hold on a device only");
                std::vector<Held> ram = r.hold(want, kRamOnly);
                check(ram.size() == 4 && ram[0].kind == kInRam, "a Hold in RAM only");
                RdmaQp mine{};
                mine.qpn = 77;
                mine.gid[15] = 0x3d;
                mine.mtu = 5;
                const RdmaOk ok = r.rdma(mine);
                check(ok.qp.qpn == 99 && ok.rkey == 0xabcd && ok.base == 0x10000 && ok.bytes == 16ull << 30,
                      "an Rdma names the helper's queue pair and the window");
            }
            {
                RemoteHelper r(at, key, 128, 6144, 2048, false);   // no RAM asked for: none given
                const std::vector<Held> any = r.hold({{1, 3}, {2, 4}}, kAny);
                check(any.size() == 1 && any[0].kind == kOnDevice, "a connection that did not ask for RAM gets none");
            }
            fake.join();
        }

        // the RAM window's places: a bump for new sizes, a freed place to the next of its size, a larger free
        // place when nothing else has room, and the sizes add up
        {
            RamWindow w(10 * 4096);
            uint64_t a, as, b, bs, c, cs, d, ds;
            check(w.take(5000, a, as) && a == 0 && as == 8192, "a place is whole pages");
            check(w.take(4096, b, bs) && b == 8192 && bs == 4096, "the next place follows it");
            check(w.take(6 * 4096, c, cs) && c == 3 * 4096 && w.used() == 9 * 4096, "a large one fits the rest");
            check(!w.take(8192, d, ds), "a full window has no room");
            w.give(a, as);
            check(w.used() == 7 * 4096, "a place given back is free");
            check(w.take(4096, d, ds) && d == 9 * 4096 && ds == 4096, "a small one takes the last unused page first");
            check(w.take(4096, d, ds) && d == 0 && ds == 8192, "then a larger free place, whole");
            w.give(d, ds);
            check(w.take(8000, d, ds) && d == 0 && ds == 8192, "a freed place goes to the next of its size");
            check(!w.take(0, d, ds), "nothing asked: nothing given");
        }

        // RDMA port names
        {
            const strata::poly::rdma::Port p = strata::poly::rdma::parse_port("rocep135s0f1:1:5");
            check(p.dev == "rocep135s0f1" && p.port == 1 && p.gid == 5, "DEV:PORT:GID");
            const strata::poly::rdma::Port q = strata::poly::rdma::parse_port("rocep1s0f1");
            check(q.dev == "rocep1s0f1" && q.port == 1 && q.gid == 0, "DEV alone: port 1, GID 0");
            bool bad = false;
            try { strata::poly::rdma::parse_port("dev:x:1"); } catch (const std::runtime_error&) { bad = true; }
            check(bad, "a port that is not a number is refused");
            bad = false;
            try { strata::poly::rdma::parse_port(":1:1"); } catch (const std::runtime_error&) { bad = true; }
            check(bad, "no device name is refused");
        }

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
