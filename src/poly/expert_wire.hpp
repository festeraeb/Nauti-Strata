// src/poly/expert_wire.hpp - the expert helper's messages: an engine has another machine's GPUs compute some of
// its routed experts.
//
// The engine keeps the hottest experts in its own VRAM and computes the misses on its CPU from RAM, the Optane
// tier and the model's file.  A helper (strata-expert-helper) is a process on another machine with GPUs to
// spare - t440's P100s, a 25 GbE link from c4 - that holds the next warmest experts in its VRAM.  Per layer the
// engine sends the token's input (n_embd floats) and the helper's experts it routed to, with their weights, and
// gets back their weighted sum; the round trip hides behind the CPU's own experts.
//
// A message is a Head and `bytes` of payload, over TCP.  Little-endian, as every machine this runs on is.
//
//   engine -> helper                              helper -> engine
//   Hello  {Hello}                                Ok {HelloOk}                  or Err {text}
//   Hold   {u32 n}{u32 where}{Pair x n}           Ok {u32 n}{Held x n} (held)   or Err
//   Release {u32 n}{Pair x n}                     Ok {u32 n} (of this connection's holds, given up)  or Err
//   Compute {ComputeHead}{f32 x rows*n_embd}      Ok {f32 rows*n_embd}          or Err
//           {Entry x n}
//   Rdma   {RdmaQp}                               Ok {RdmaOk}                   or Err
//
// Version 2 (2026-11-08): the helper may also hold experts in a RAM window of its own (--ram-mib), pinned and
// registered for RDMA READ.  On a miss the engine reads such an expert's bytes over RoCE into its own memory and
// computes it on its CPU as one from its RAM tier: t440's SSD served Le Chonk's file misses at 72 ms each, its
// RAM serves the same 23 MiB in 8.4 ms (docs/REMOTE_TIER_PLAN.md).  A Hold says where it may put them (Where); a
// Held says where each one went, and the window address of a RAM one.  Rdma sets up one reliable-connected queue
// pair per connection with the window: control over TCP, bytes over RoCE.  Only a connection whose Hello asked
// for RAM (kWantRam) is ever given a RAM hold.
//
// A connection's holds end with it: an engine that restarts asks again, and the helper frees what no
// connection holds.  An engine keeps two connections: one that holds and releases while an answer is written,
// so that the helper's set follows the answer (Engine::follow()), and one that computes, which a Hold's file
// reads then never delay.  The queue pair is the holding connection's.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::poly::wire {

constexpr uint32_t kMagic = 0x32485853;   // "SXH2": version 1 had no RAM window, and its Hold answer no Held
constexpr uint32_t kGelu = 1;             // ComputeHead::flags: gelu(gate) * up, not silu with the clamp

enum Type : uint32_t { kHello = 1, kHold = 2, kCompute = 3, kRelease = 4, kRdma = 5, kOk = 8, kErr = 9 };

constexpr uint32_t kWantRam = 1;   ///< Hello::flags: this connection may be given RAM holds (it reads them by RDMA)
constexpr uint32_t kHasRdma = 1;   ///< HelloOk::flags: the RAM window is registered, an Rdma can be set up

/// Hold's `where`: where the helper may put what it is asked to hold.
enum Where : uint32_t {
    kAny = 0,          ///< a device if one has room, else the RAM window (for a connection that wants RAM)
    kDeviceOnly = 1,   ///< a device or nothing: what a version 1 helper did
    kRamOnly = 2,      ///< the RAM window or nothing
};
/// Held::kind
enum Kind : uint32_t { kOnDevice = 0, kInRam = 1 };

struct Head {
    uint32_t magic;
    uint32_t type;
    uint64_t bytes;   ///< the payload's
};
static_assert(sizeof(Head) == 16, "Head is 16 bytes");

struct Hello {
    uint64_t model_key;   ///< model_key() of the engine's model: both ends must have the same file
    uint32_t n_expert, n_embd, n_ff;
    uint32_t flags;       ///< kWantRam
};
static_assert(sizeof(Hello) == 24, "Hello is 24 bytes");

struct HelloOk {
    uint32_t devices;
    uint32_t flags;          ///< kHasRdma
    uint64_t capacity;       ///< expert bytes the helper's GPUs hold together, for all its connections
    uint64_t ram_capacity;   ///< expert bytes its RAM window holds, for all its connections; 0: none
};
static_assert(sizeof(HelloOk) == 24, "HelloOk is 24 bytes");

struct Pair {
    int32_t layer, expert;
};
static_assert(sizeof(Pair) == 8, "Pair is 8 bytes");

/// An expert the helper holds for the connection, and where.
struct Held {
    int32_t layer, expert;
    uint32_t kind;       ///< kOnDevice: the helper computes it; kInRam: the engine reads it at `addr`
    uint32_t reserved;
    uint64_t addr;       ///< kInRam: gate | up | down, back to back, in the window (RdmaOk::rkey); else 0
};
static_assert(sizeof(Held) == 24, "Held is 24 bytes");

/// One side's queue pair, as the other side needs it to connect (RoCE: the GID names the port; lid is 0).
struct RdmaQp {
    uint32_t qpn, psn;
    uint8_t gid[16];
    uint16_t lid;
    uint8_t mtu;         ///< enum ibv_mtu of the port
    uint8_t reserved[5];
};
static_assert(sizeof(RdmaQp) == 32, "RdmaQp is 32 bytes");

struct RdmaOk {
    RdmaQp qp;           ///< the helper's queue pair, connected to the engine's
    uint32_t rkey;       ///< the RAM window's
    uint32_t reserved;
    uint64_t base, bytes;   ///< the window's address and size, which every Held::addr falls in
};
static_assert(sizeof(RdmaOk) == 56, "RdmaOk is 56 bytes");

struct ComputeHead {
    int32_t layer, rows;   ///< rows: the tokens, each with n_embd input floats and as many output floats
    float limit;           ///< the swiglu clamp
    uint32_t flags;        ///< kGelu
    uint32_t n;            ///< Entry records after the input
    uint32_t reserved;
};
static_assert(sizeof(ComputeHead) == 24, "ComputeHead is 24 bytes");

struct Entry {
    int32_t expert, row;
    float weight;
};
static_assert(sizeof(Entry) == 12, "Entry is 12 bytes");

/// What names a model to both ends without its path, which differs between machines: its files' bytes together
/// and its experts' shape.
uint64_t model_key(uint64_t file_bytes, int n_expert, int n_embd, int n_ff);

/// One TCP connection, blocking, a whole message at a time.  Throws std::runtime_error on any failure, the peer
/// closing included; the connection is then of no further use.
class Conn {
public:
    Conn() = default;
    explicit Conn(int fd);
    ~Conn();
    Conn(Conn&& o) noexcept;
    Conn& operator=(Conn&& o) noexcept;
    Conn(const Conn&) = delete;
    Conn& operator=(const Conn&) = delete;

    /// Connects to "host:port" within timeout_ms.
    static Conn dial(const std::string& host_port, int timeout_ms);

    /// One message: the head, then a, b and c back to back (any may be empty).
    void send(uint32_t type, const void* a, size_t an, const void* b = nullptr, size_t bn = 0,
              const void* c = nullptr, size_t cn = 0);
    /// The next message's type, its payload in `payload` (resized to fit).  max_bytes refuses a larger one.
    uint32_t recv(std::vector<uint8_t>& payload, uint64_t max_bytes = uint64_t(1) << 31);
    /// An Err reply's text, or a generic one.
    static std::string text(const std::vector<uint8_t>& payload);

    bool open() const { return fd_ >= 0; }
    void close();
    /// From another thread: a recv() blocked on this connection returns with an error, and it is of no further use.
    void shutdown();

private:
    int fd_ = -1;
};

/// A listening TCP socket.
class Listener {
public:
    /// "port" or "addr:port"; all addresses when none is given.
    explicit Listener(const std::string& where);
    ~Listener();
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    Conn accept();
    int port() const { return port_; }   ///< the one bound: "0" asks the kernel for a free one

private:
    int fd_ = -1;
    int port_ = 0;
};

/// Low latency and large buffers on a connected socket.
void tune(int fd);

}  // namespace strata::poly::wire
