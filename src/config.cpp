#include "config.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace vlm {

namespace {

size_t configured_size(const char* name, size_t fallback, size_t minimum, size_t maximum) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    char* end = nullptr;
    unsigned long long value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || value < minimum || value > maximum) {
        throw std::invalid_argument(std::string(name) + " must be an integer between " +
                                    std::to_string(minimum) + " and " + std::to_string(maximum));
    }
    return static_cast<size_t>(value);
}

std::string configured_string(const char* name, const std::string& fallback) {
    const char* raw = std::getenv(name);
    return raw == nullptr || *raw == '\0' ? fallback : raw;
}

std::vector<std::string> configured_list(const char* name) {
    std::vector<std::string> values;
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return values;
    }
    std::string buffer(raw);
    size_t start = 0;
    while (start <= buffer.size()) {
        size_t comma = buffer.find(',', start);
        if (comma == std::string::npos) {
            comma = buffer.size();
        }
        std::string item = buffer.substr(start, comma - start);
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            values.push_back(item);
        }
        start = comma + 1;
    }
    return values;
}

}  // namespace

Config load_config_from_env() {
    Config config;
    config.listen_address = configured_string("GRPC_VLM_LISTEN_ADDRESS", config.listen_address);
    config.endpoint = configured_string("GRPC_VLM_ENDPOINT", config.endpoint);
    config.presets = configured_list("GRPC_VLM_PRESETS");
    config.concurrency = configured_size("GRPC_VLM_CONCURRENCY", config.concurrency, 1, 64);
    config.max_page_bytes = configured_size("GRPC_VLM_MAX_PAGE_BYTES", config.max_page_bytes,
                                            1024, 1024ULL * 1024 * 1024);
    config.max_pages = configured_size("GRPC_VLM_MAX_PAGES", config.max_pages, 1, 100000);
    config.vlm_timeout_seconds =
        configured_size("GRPC_VLM_VLM_TIMEOUT_SECONDS", config.vlm_timeout_seconds, 1, 86400);
    config.metrics_interval_seconds = configured_size(
        "GRPC_VLM_METRICS_INTERVAL_SECONDS", config.metrics_interval_seconds, 0, 86400);
    return config;
}

}  // namespace vlm
