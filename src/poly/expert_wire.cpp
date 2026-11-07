// src/poly/expert_wire.cpp - see expert_wire.hpp.
#include "expert_wire.hpp"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>

namespace strata::poly::wire {

uint64_t model_key(uint64_t file_bytes, int n_expert, int n_embd, int n_ff) {
    const std::string s = std::to_string(file_bytes) + "|" + std::to_string(n_expert) + "|" + std::to_string(n_embd) +
                          "|" + std::to_string(n_ff);
    uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : s) {
        h ^= ch;
        h *= 1099511628211ull;
    }
    return h;
}

void tune(int fd) {
    const int one = 1, buf = 8 << 20;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
}

Conn::Conn(int fd) : fd_(fd) {}
Conn::~Conn() { close(); }
Conn::Conn(Conn&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
Conn& Conn::operator=(Conn&& o) noexcept {
    if (this != &o) {
        close();
        fd_ = o.fd_;
        o.fd_ = -1;
    }
    return *this;
}

void Conn::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

namespace {

std::pair<std::string, std::string> split_host_port(const std::string& s) {
    const size_t c = s.rfind(':');
    if (c == std::string::npos) return {"", s};
    return {s.substr(0, c), s.substr(c + 1)};
}

}  // namespace

Conn Conn::dial(const std::string& host_port, int timeout_ms) {
    const auto [host, port] = split_host_port(host_port);
    if (host.empty() || port.empty()) throw std::runtime_error("expert helper: \"" + host_port + "\" is not host:port");
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (const int rc = getaddrinfo(host.c_str(), port.c_str(), &hints, &res); rc != 0)
        throw std::runtime_error("expert helper: " + host_port + ": " + gai_strerror(rc));
    std::string why = "no address";
    for (addrinfo* a = res; a; a = a->ai_next) {
        const int fd = socket(a->ai_family, a->ai_socktype | SOCK_CLOEXEC, a->ai_protocol);
        if (fd < 0) continue;
        const int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        int rc = connect(fd, a->ai_addr, a->ai_addrlen);
        if (rc != 0 && errno == EINPROGRESS) {
            pollfd p{fd, POLLOUT, 0};
            rc = poll(&p, 1, timeout_ms) == 1 ? 0 : -1;
            int err = 0;
            socklen_t len = sizeof(err);
            if (rc == 0 && (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0 || err != 0)) {
                errno = err;
                rc = -1;
            } else if (rc != 0) {
                errno = ETIMEDOUT;
            }
        }
        if (rc == 0) {
            fcntl(fd, F_SETFL, flags);
            tune(fd);
            freeaddrinfo(res);
            return Conn(fd);
        }
        why = std::strerror(errno);
        ::close(fd);
    }
    freeaddrinfo(res);
    throw std::runtime_error("expert helper: cannot connect to " + host_port + ": " + why);
}

void Conn::send(uint32_t type, const void* a, size_t an, const void* b, size_t bn, const void* c, size_t cn) {
    if (fd_ < 0) throw std::runtime_error("expert helper: the connection is closed");
    Head h{kMagic, type, (uint64_t) (an + bn + cn)};
    iovec v[4] = {{&h, sizeof(h)}, {const_cast<void*>(a), an}, {const_cast<void*>(b), bn}, {const_cast<void*>(c), cn}};
    int first = 0;
    while (first < 4) {
        while (first < 4 && v[first].iov_len == 0) ++first;
        if (first == 4) break;
        msghdr m{};
        m.msg_iov = v + first;
        m.msg_iovlen = (size_t) (4 - first);
        const ssize_t n = sendmsg(fd_, &m, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            const std::string why = std::strerror(errno);
            close();
            throw std::runtime_error("expert helper: send: " + why);
        }
        size_t left = (size_t) n;
        while (first < 4 && left >= v[first].iov_len) left -= v[first++].iov_len;
        if (first < 4) {
            v[first].iov_base = (char*) v[first].iov_base + left;
            v[first].iov_len -= left;
        }
    }
}

namespace {

void read_all(int fd, void* dst, size_t n) {
    char* p = (char*) dst;
    while (n > 0) {
        const ssize_t r = ::recv(fd, p, n, MSG_WAITALL);
        if (r == 0) throw std::runtime_error("expert helper: the peer closed the connection");
        if (r < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("expert helper: recv: ") + std::strerror(errno));
        }
        p += r;
        n -= (size_t) r;
    }
}

}  // namespace

uint32_t Conn::recv(std::vector<uint8_t>& payload, uint64_t max_bytes) {
    if (fd_ < 0) throw std::runtime_error("expert helper: the connection is closed");
    try {
        Head h{};
        read_all(fd_, &h, sizeof(h));
        if (h.magic != kMagic) throw std::runtime_error("expert helper: not a helper message (bad magic)");
        if (h.bytes > max_bytes) throw std::runtime_error("expert helper: a message of " + std::to_string(h.bytes) +
                                                          " bytes is larger than allowed");
        payload.resize((size_t) h.bytes);
        if (h.bytes) read_all(fd_, payload.data(), (size_t) h.bytes);
        return h.type;
    } catch (...) {
        close();
        throw;
    }
}

void Conn::shutdown() {
    if (fd_ >= 0) ::shutdown(fd_, SHUT_RDWR);
}

std::string Conn::text(const std::vector<uint8_t>& payload) {
    return payload.empty() ? std::string("the helper refused") : std::string(payload.begin(), payload.end());
}

Listener::Listener(const std::string& where) {
    const auto [host, port] = split_host_port(where);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = host.empty() ? AF_INET6 : AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    if (const int rc = getaddrinfo(host.empty() ? nullptr : host.c_str(), port.c_str(), &hints, &res); rc != 0)
        throw std::runtime_error("expert helper: listen " + where + ": " + gai_strerror(rc));
    fd_ = socket(res->ai_family, res->ai_socktype | SOCK_CLOEXEC, res->ai_protocol);
    const int one = 1, zero = 0;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    if (res->ai_family == AF_INET6) setsockopt(fd_, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof(zero));
    const bool ok = fd_ >= 0 && bind(fd_, res->ai_addr, res->ai_addrlen) == 0 && listen(fd_, 16) == 0;
    freeaddrinfo(res);
    if (!ok) {
        const std::string why = std::strerror(errno);
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("expert helper: listen " + where + ": " + why);
    }
    sockaddr_storage a{};
    socklen_t len = sizeof(a);
    getsockname(fd_, (sockaddr*) &a, &len);
    port_ = a.ss_family == AF_INET6 ? ntohs(((sockaddr_in6*) &a)->sin6_port) : ntohs(((sockaddr_in*) &a)->sin_port);
}

Listener::~Listener() {
    if (fd_ >= 0) ::close(fd_);
}

Conn Listener::accept() {
    for (;;) {
        const int fd = ::accept4(fd_, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd >= 0) {
            tune(fd);
            return Conn(fd);
        }
        if (errno != EINTR && errno != ECONNABORTED)
            throw std::runtime_error(std::string("expert helper: accept: ") + std::strerror(errno));
    }
}

}  // namespace strata::poly::wire
