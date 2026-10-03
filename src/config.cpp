#include "config.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vlm_client.h"

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

// "true"/"1" or "false"/"0"; unset or empty is the fallback. Anything else
// throws: a typo must not silently keep (or flip) a security default.
bool configured_bool(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    const std::string value(raw);
    if (value == "true" || value == "1") {
        return true;
    }
    if (value == "false" || value == "0") {
        return false;
    }
    throw std::invalid_argument(std::string(name) + " must be true, false, 1 or 0");
}

// A credential, verbatim (a key is exactly what the operator wrote).
Secret configured_secret(const char* name) {
    const char* raw = std::getenv(name);
    return raw == nullptr ? Secret() : Secret(raw);
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
    if (!config.endpoint.empty()) {
        // Fail at startup rather than on every RPC, and without echoing
        // the value: a token in its path would land in the log.
        if (const std::string problem = endpoint_error(config.endpoint); !problem.empty()) {
            throw std::invalid_argument("GRPC_VLM_ENDPOINT: " + problem);
        }
    }
    config.vlm_api_key = configured_secret("GRPC_VLM_API_KEY");
    config.allow_endpoint_override =
        configured_bool("GRPC_VLM_ALLOW_ENDPOINT_OVERRIDE", config.allow_endpoint_override);
    config.presets = configured_list("GRPC_VLM_PRESETS");
    config.concurrency = configured_size("GRPC_VLM_CONCURRENCY", config.concurrency, 1, 64);
    config.max_page_bytes = configured_size("GRPC_VLM_MAX_PAGE_BYTES", config.max_page_bytes,
                                            1024, 1024ULL * 1024 * 1024);
    config.max_pages = configured_size("GRPC_VLM_MAX_PAGES", config.max_pages, 1, 100000);
    config.vlm_timeout_seconds =
        configured_size("GRPC_VLM_VLM_TIMEOUT_SECONDS", config.vlm_timeout_seconds, 1, 86400);
    config.metrics_interval_seconds = configured_size(
        "GRPC_VLM_METRICS_INTERVAL_SECONDS", config.metrics_interval_seconds, 0, 86400);
    // 0 or empty disables the HTTP listener; an explicit value must be a
    // real port.
    const char* http_port = std::getenv("GRPC_VLM_HTTP_PORT");
    if (http_port != nullptr && (*http_port == '\0' || std::string(http_port) == "0")) {
        config.http_port = 0;
    } else {
        config.http_port = configured_size("GRPC_VLM_HTTP_PORT", config.http_port, 1, 65535);
    }
    return config;
}

std::string startup_banner(const Config& config) {
    std::string banner = "grpc-vlm-convert listening on " + config.listen_address;
    if (config.http_port != 0) {
        banner += " (HTTP on 0.0.0.0:" + std::to_string(config.http_port) + ")";
    }
    banner += " (endpoint ";
    if (config.endpoint.empty()) {
        banner += config.allow_endpoint_override
                      ? "<none — per-request override required>"
                      : "<none — ConvertPages fails until GRPC_VLM_ENDPOINT is set>";
    } else {
        banner += endpoint_origin(config.endpoint);
    }
    // Whether a key is configured, never the key.
    if (!config.vlm_api_key.empty()) {
        banner += ", API key set";
    }
    banner += ")";
    if (config.allow_endpoint_override) {
        banner += " (per-request endpoint overrides allowed)";
    }
    return banner;
}

}  // namespace vlm
