#pragma once

// Shared helpers for the plain-main test suite: an assertion that throws,
// and in-memory fixtures authored by the tests (no committed binaries).

#include <cstdio>
#include <cstdlib>
#include <iostream>
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

inline const char* env_or_null(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

// Standard skip: exit 77 so CTest reports SKIP, not PASS.
inline int skip(const std::string& why) {
    std::cerr << "SKIP: " << why << '\n';
    return 77;
}
