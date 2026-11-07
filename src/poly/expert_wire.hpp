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
//   Hold   {u32 n}{Pair x n}                      Ok {u32 n}{Pair x n} (held)   or Err
//   Compute {ComputeHead}{f32 x rows*n_embd}      Ok {f32 rows*n_embd}          or Err
//           {Entry x n}
//
// A connection's holds end with it: an engine that restarts asks again, and the helper frees what no
// connection holds.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::poly::wire {

constexpr uint32_t kMagic = 0x31485853;   // "SXH1"
constexpr uint32_t kGelu = 1;             // ComputeHead::flags: gelu(gate) * up, not silu with the clamp

enum Type : uint32_t { kHello = 1, kHold = 2, kCompute = 3, kOk = 8, kErr = 9 };

struct Head {
    uint32_t magic;
    uint32_t type;
    uint64_t bytes;   ///< the payload's
};
static_assert(sizeof(Head) == 16, "Head is 16 bytes");

struct Hello {
    uint64_t model_key;   ///< model_key() of the engine's model: both ends must have the same file
    uint32_t n_expert, n_embd, n_ff, reserved;
};
static_assert(sizeof(Hello) == 24, "Hello is 24 bytes");

struct HelloOk {
    uint32_t devices, reserved;
    uint64_t capacity;   ///< expert bytes the helper's GPUs hold together, for all its connections
};
static_assert(sizeof(HelloOk) == 16, "HelloOk is 16 bytes");

struct Pair {
    int32_t layer, expert;
};
static_assert(sizeof(Pair) == 8, "Pair is 8 bytes");

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
