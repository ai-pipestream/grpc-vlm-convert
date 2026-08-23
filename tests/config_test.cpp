// Unit tests for load_config_from_env: defaults on a clean environment,
// overrides, comma-list parsing with whitespace, range validation that
// names the offending variable, and the HTTP-port disable spellings.

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "config.h"
#include "fixture.h"

namespace {

constexpr const char* kAllVars[] = {
    "GRPC_VLM_LISTEN_ADDRESS",       "GRPC_VLM_ENDPOINT",
    "GRPC_VLM_PRESETS",              "GRPC_VLM_CONCURRENCY",
    "GRPC_VLM_MAX_PAGE_BYTES",       "GRPC_VLM_MAX_PAGES",
    "GRPC_VLM_VLM_TIMEOUT_SECONDS",  "GRPC_VLM_METRICS_INTERVAL_SECONDS",
    "GRPC_VLM_HTTP_PORT",
};

void clear_env() {
    for (const char* name : kAllVars) {
        ::unsetenv(name);
    }
}

// True when the value makes load_config_from_env throw and the message
// names the variable (operators must see WHICH knob is malformed).
bool rejects(const char* name, const char* value) {
    ::setenv(name, value, 1);
    bool threw = false;
    try {
        (void)vlm::load_config_from_env();
    } catch (const std::invalid_argument& error) {
        threw = std::string(error.what()).contains(name);
    }
    ::unsetenv(name);
    return threw;
}

void verify_defaults() {
    clear_env();
    const vlm::Config config = vlm::load_config_from_env();
    require(config.listen_address == "0.0.0.0:50058", "default listen address");
    require(config.endpoint.empty(), "no default endpoint");
    require(config.presets.empty(), "no default preset list");
    require(config.concurrency == 2, "default concurrency");
    require(config.max_page_bytes == 32ULL * 1024 * 1024, "default page byte cap");
    require(config.max_pages == 512, "default page cap");
    require(config.vlm_timeout_seconds == 300, "default VLM timeout");
    require(config.metrics_interval_seconds == 60, "default metrics interval");
    require(config.http_port == 50059, "default HTTP port");
}

void verify_overrides_and_lists() {
    clear_env();
    ::setenv("GRPC_VLM_LISTEN_ADDRESS", "127.0.0.1:9", 1);
    ::setenv("GRPC_VLM_ENDPOINT", "http://vlm:8080", 1);
    ::setenv("GRPC_VLM_CONCURRENCY", "64", 1);
    ::setenv("GRPC_VLM_PRESETS", " granite-docling , smoldocling ,,custom-ocr ", 1);
    const vlm::Config config = vlm::load_config_from_env();
    require(config.listen_address == "127.0.0.1:9", "listen address override");
    require(config.endpoint == "http://vlm:8080", "endpoint override");
    require(config.concurrency == 64, "concurrency override at the range edge");
    require(config.presets.size() == 3, "list drops empty entries");
    require(config.presets[0] == "granite-docling" && config.presets[1] == "smoldocling" &&
                config.presets[2] == "custom-ocr",
            "list entries are trimmed in order");

    // Empty values fall back to the defaults rather than parsing as 0.
    ::setenv("GRPC_VLM_CONCURRENCY", "", 1);
    ::setenv("GRPC_VLM_PRESETS", " , ", 1);
    const vlm::Config fallback = vlm::load_config_from_env();
    require(fallback.concurrency == 2, "empty value falls back to the default");
    require(fallback.presets.empty(), "whitespace-only list is empty");
    clear_env();
}

void verify_range_validation() {
    clear_env();
    require(rejects("GRPC_VLM_CONCURRENCY", "0"), "concurrency below the minimum");
    require(rejects("GRPC_VLM_CONCURRENCY", "65"), "concurrency above the maximum");
    require(rejects("GRPC_VLM_CONCURRENCY", "abc"), "non-numeric value");
    require(rejects("GRPC_VLM_CONCURRENCY", "12x"), "trailing garbage");
    require(rejects("GRPC_VLM_MAX_PAGE_BYTES", "1023"), "page bytes below the minimum");
    require(rejects("GRPC_VLM_MAX_PAGES", "100001"), "pages above the maximum");
    require(rejects("GRPC_VLM_VLM_TIMEOUT_SECONDS", "0"), "timeout below the minimum");

    // Boundary values pass.
    ::setenv("GRPC_VLM_MAX_PAGE_BYTES", "1024", 1);
    ::setenv("GRPC_VLM_METRICS_INTERVAL_SECONDS", "0", 1);
    const vlm::Config config = vlm::load_config_from_env();
    require(config.max_page_bytes == 1024, "page bytes at the minimum");
    require(config.metrics_interval_seconds == 0, "metrics interval 0 (disabled) is legal");
    clear_env();
}

void verify_http_port_disable() {
    clear_env();
    ::setenv("GRPC_VLM_HTTP_PORT", "0", 1);
    require(vlm::load_config_from_env().http_port == 0, "port 0 disables the listener");
    ::setenv("GRPC_VLM_HTTP_PORT", "", 1);
    require(vlm::load_config_from_env().http_port == 0, "empty port disables the listener");
    ::setenv("GRPC_VLM_HTTP_PORT", "50060", 1);
    require(vlm::load_config_from_env().http_port == 50060, "explicit port");
    require(rejects("GRPC_VLM_HTTP_PORT", "70000"), "port above 65535");
    clear_env();
}

}  // namespace

int main() {
    try {
        verify_defaults();
        verify_overrides_and_lists();
        verify_range_validation();
        verify_http_port_disable();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("config-test passed");
    return 0;
}
