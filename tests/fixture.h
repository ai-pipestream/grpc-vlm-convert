#pragma once

// Shared helpers for the plain-main test suite: an assertion that throws,
// and in-memory fixtures authored by the tests (no committed binaries).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <print>
#include <stdexcept>
#include <string>

inline void require(bool condition, const std::string& what) {
    if (!condition) {
        throw std::runtime_error("FAIL: " + what);
    }
}

// A minimal "PNG" for the wire: the real 8-byte signature (the server
// validates it) followed by a marker payload the fake VLM reads back.
// The server never rasterizes — PNG bytes are forwarded verbatim.
inline std::string make_png(const std::string& marker) {
    std::string png("\x89PNG\r\n\x1a\n", 8);
    png += marker;
    return png;
}

// A real 8-bit grayscale PNG of width x height mid-gray pixels, stored
// without compression so the test needs no encoder. stb checks neither
// chunk CRCs nor the zlib checksum, so those are zero.
inline std::string gray_png(uint32_t width, uint32_t height) {
    std::string png("\x89PNG\r\n\x1a\n", 8);
    auto be32 = [](std::string* out, uint32_t value) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            *out += static_cast<char>((value >> shift) & 0xFF);
        }
    };
    auto chunk = [&](const std::string& type, const std::string& data) {
        be32(&png, static_cast<uint32_t>(data.size()));
        png += type + data;
        be32(&png, 0);
    };
    std::string header;
    be32(&header, width);
    be32(&header, height);
    header += std::string("\x08\x00\x00\x00\x00", 5);  // 8-bit gray, no interlace
    chunk("IHDR", header);

    std::string row(1 + width, '\x80');
    row[0] = '\0';  // filter: none
    std::string raw;
    for (uint32_t y = 0; y < height; y++) {
        raw += row;
    }
    std::string zlib("\x78\x01", 2);
    for (size_t offset = 0; offset < raw.size(); offset += 65535) {
        const size_t length = std::min<size_t>(65535, raw.size() - offset);
        zlib += static_cast<char>(offset + length == raw.size() ? 1 : 0);  // stored block
        zlib += static_cast<char>(length & 0xFF);
        zlib += static_cast<char>(length >> 8);
        zlib += static_cast<char>(~length & 0xFF);
        zlib += static_cast<char>((~length >> 8) & 0xFF);
        zlib += raw.substr(offset, length);
    }
    zlib += std::string(4, '\0');  // adler32, unchecked
    chunk("IDAT", zlib);
    chunk("IEND", "");
    return png;
}

inline const char* env_or_null(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

// Standard skip: exit 77 so CTest reports SKIP, not PASS.
inline int skip(const std::string& why) {
    std::println(stderr, "SKIP: {}", why);
    return 77;
}
