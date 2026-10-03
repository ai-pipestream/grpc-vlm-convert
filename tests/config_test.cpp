// Unit tests for load_config_from_env: defaults on a clean environment,
// overrides, comma-list parsing with whitespace, range validation that
// names the offending variable, and the HTTP-port disable spellings.

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "config.h"
#include "fixture.h"
#include "mapping/image_crop.h"

namespace {

constexpr const char* kAllVars[] = {
    "GRPC_VLM_LISTEN_ADDRESS",       "GRPC_VLM_ENDPOINT",
    "GRPC_VLM_PRESETS",              "GRPC_VLM_CONCURRENCY",
    "GRPC_VLM_MAX_PAGE_BYTES",       "GRPC_VLM_MAX_PAGES",
    "GRPC_VLM_VLM_TIMEOUT_SECONDS",  "GRPC_VLM_METRICS_INTERVAL_SECONDS",
    "GRPC_VLM_HTTP_PORT",            "GRPC_VLM_API_KEY",
    "GRPC_VLM_ALLOW_ENDPOINT_OVERRIDE", "GRPC_VLM_MAX_INFLIGHT",
    "GRPC_VLM_MAX_STREAM_BUFFERED_BYTES", "GRPC_VLM_MAX_BUFFERED_BYTES",
    "GRPC_VLM_HTTP_HOST",            "GRPC_VLM_HTTP_MAX_BODY_BYTES",
    "GRPC_VLM_HTTP_TOKEN",           "GRPC_VLM_LOGPROBS",
    "GRPC_VLM_MAX_PAGE_CROP_BYTES",
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
    require(config.max_page_crop_bytes == 3ULL * 1024 * 1024 &&
                config.max_page_crop_bytes == vlm::mapping::kDefaultMaxInlineCropBytes,
            "default inline crop cap is 3 MiB, the mapper's own default");
    require(config.vlm_timeout_seconds == 300, "default VLM timeout");
    require(config.metrics_interval_seconds == 60, "default metrics interval");
    require(config.http_port == 50059, "default HTTP port");
    require(config.vlm_api_key.empty(), "no default API key");
    require(!config.allow_endpoint_override, "endpoint overrides are off by default");
    require(config.max_inflight == 8, "default process-wide VLM calls in flight");
    require(config.max_stream_buffered_bytes == 4 * config.max_page_bytes,
            "default stream buffer is four pages at the cap");
    require(config.max_buffered_bytes == 16 * config.max_page_bytes,
            "default process buffer is sixteen pages at the cap");
    require(config.http_host == "127.0.0.1", "the HTTP front end binds loopback by default");
    require(config.http_max_body_bytes == 64ULL * 1024 * 1024, "default HTTP body cap");
    require(config.http_token.empty(), "no default HTTP token");
    require(config.request_logprobs, "logprobs are requested by default");
}

// Off loopback the HTTP front end needs a token, or the process does not
// start; the token is read verbatim and never shown.
void verify_http_exposure() {
    clear_env();
    require(rejects("GRPC_VLM_HTTP_HOST", "0.0.0.0"),
            "binding all interfaces without a token is refused");
    require(rejects("GRPC_VLM_HTTP_HOST", "10.1.2.3"), "a LAN address without a token is refused");
    ::setenv("GRPC_VLM_HTTP_HOST", "127.0.0.5", 1);
    require(vlm::load_config_from_env().http_host == "127.0.0.5",
            "any 127/8 address is loopback");
    ::setenv("GRPC_VLM_HTTP_HOST", "0.0.0.0", 1);
    ::setenv("GRPC_VLM_HTTP_PORT", "0", 1);
    require(vlm::load_config_from_env().http_port == 0,
            "with the front end off, its host needs no token");
    ::unsetenv("GRPC_VLM_HTTP_PORT");
    ::setenv("GRPC_VLM_HTTP_TOKEN", "http-secret-91", 1);
    const vlm::Config config = vlm::load_config_from_env();
    require(config.http_host == "0.0.0.0" && config.http_token.reveal() == "http-secret-91",
            "with a token, any host is allowed");
    const std::string banner = vlm::startup_banner(config);
    require(banner.contains("HTTP on 0.0.0.0:50059, token required"),
            "the banner names the bind and that a token is required: " + banner);
    require(!banner.contains("http-secret-91"), "the banner never shows the token");
    require(rejects("GRPC_VLM_HTTP_MAX_BODY_BYTES", "1000"), "a body cap below 1 KiB is refused");
    clear_env();
}

// The buffer caps follow the page cap unless set, and may not go below it:
// a page that fits no budget could never be admitted.
void verify_buffer_caps() {
    clear_env();
    ::setenv("GRPC_VLM_MAX_PAGE_BYTES", "1048576", 1);
    vlm::Config derived = vlm::load_config_from_env();
    require(derived.max_stream_buffered_bytes == 4 * 1048576ULL,
            "the stream buffer default follows the page cap");
    require(derived.max_buffered_bytes == 16 * 1048576ULL,
            "the process buffer default follows the page cap");

    ::setenv("GRPC_VLM_MAX_STREAM_BUFFERED_BYTES", "2097152", 1);
    ::setenv("GRPC_VLM_MAX_BUFFERED_BYTES", "1048576", 1);
    ::setenv("GRPC_VLM_MAX_INFLIGHT", "3", 1);
    const vlm::Config set = vlm::load_config_from_env();
    require(set.max_stream_buffered_bytes == 2097152, "explicit stream buffer");
    require(set.max_buffered_bytes == 1048576, "a buffer exactly at the page cap is legal");
    require(set.max_inflight == 3, "explicit in-flight cap");
    ::unsetenv("GRPC_VLM_MAX_STREAM_BUFFERED_BYTES");
    ::unsetenv("GRPC_VLM_MAX_BUFFERED_BYTES");

    require(rejects("GRPC_VLM_MAX_STREAM_BUFFERED_BYTES", "1048575"),
            "a stream buffer below the page cap is rejected");
    require(rejects("GRPC_VLM_MAX_BUFFERED_BYTES", "4096"),
            "a process buffer below the page cap is rejected");
    require(rejects("GRPC_VLM_MAX_INFLIGHT", "0"), "an in-flight cap of zero is rejected");
    clear_env();
}

// The API key is read verbatim and never shown; the override opt-in takes
// exactly true/false/1/0, so a typo cannot quietly open the SSRF door (or
// leave it shut when the operator meant to open it).
void verify_credentials_and_override_flag() {
    clear_env();
    ::setenv("GRPC_VLM_ENDPOINT", "http://vlm:8080", 1);
    ::setenv("GRPC_VLM_API_KEY", "sk-test-7781", 1);
    ::setenv("GRPC_VLM_ALLOW_ENDPOINT_OVERRIDE", "true", 1);
    const vlm::Config config = vlm::load_config_from_env();
    require(config.vlm_api_key.reveal() == "sk-test-7781", "API key read verbatim");
    require(config.allow_endpoint_override, "override opt-in read");
    const std::string banner = vlm::startup_banner(config);
    require(!banner.contains("sk-test-7781"), "the banner never shows the key: " + banner);
    require(banner.contains("API key set"), "the banner says a key is configured");
    require(banner.contains("overrides allowed"), "the banner says overrides are allowed");

    ::setenv("GRPC_VLM_ALLOW_ENDPOINT_OVERRIDE", "0", 1);
    require(!vlm::load_config_from_env().allow_endpoint_override, "0 turns overrides off");
    require(rejects("GRPC_VLM_ALLOW_ENDPOINT_OVERRIDE", "yes"), "a typo is rejected");
    ::setenv("GRPC_VLM_LOGPROBS", "false", 1);
    require(!vlm::load_config_from_env().request_logprobs, "logprobs can be turned off");
    require(rejects("GRPC_VLM_LOGPROBS", "off"), "a logprobs typo is rejected");
    clear_env();
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
    require(rejects("GRPC_VLM_MAX_PAGE_CROP_BYTES", "0"), "a zero crop byte cap");
    require(rejects("GRPC_VLM_MAX_PAGE_CROP_BYTES", "1023"), "crop bytes below the minimum");
    require(rejects("GRPC_VLM_MAX_PAGE_CROP_BYTES", "3MiB"), "a crop byte cap with a unit");
    require(rejects("GRPC_VLM_MAX_PAGE_CROP_BYTES", "-1"), "a negative crop byte cap");

    // Boundary values pass.
    ::setenv("GRPC_VLM_MAX_PAGE_BYTES", "1024", 1);
    ::setenv("GRPC_VLM_METRICS_INTERVAL_SECONDS", "0", 1);
    ::setenv("GRPC_VLM_MAX_PAGE_CROP_BYTES", "1024", 1);
    const vlm::Config config = vlm::load_config_from_env();
    require(config.max_page_bytes == 1024, "page bytes at the minimum");
    require(config.max_page_crop_bytes == 1024, "crop bytes at the minimum");
    require(config.metrics_interval_seconds == 0, "metrics interval 0 (disabled) is legal");
    clear_env();
}

// A malformed endpoint stops the process at startup, named by its variable
// and never quoted (a token in it would land in the log). The startup line
// shows the endpoint as scheme, host and port only.
void verify_endpoint_validation_and_banner() {
    clear_env();
    require(rejects("GRPC_VLM_ENDPOINT", "https://vlm:8080"), "https endpoint is refused");
    require(rejects("GRPC_VLM_ENDPOINT", "http://vlm:notaport"), "bad port is refused");
    ::setenv("GRPC_VLM_ENDPOINT", "http://alice:hunter2@vlm:8080", 1);
    try {
        (void)vlm::load_config_from_env();
        require(false, "userinfo endpoint is refused");
    } catch (const std::invalid_argument& error) {
        require(std::string(error.what()).contains("GRPC_VLM_ENDPOINT"),
                "the error names the variable");
        require(!std::string(error.what()).contains("hunter2"),
                "the error does not quote the credentials");
    }

    ::setenv("GRPC_VLM_ENDPOINT", "http://vlm:8080/tenant/PATH-SECRET?key=QUERY-SECRET", 1);
    const vlm::Config config = vlm::load_config_from_env();
    const std::string banner = vlm::startup_banner(config);
    require(banner.contains("grpc-vlm-convert listening on"),
            "the banner keeps the line the smoke test waits for");
    require(banner.contains("(endpoint http://vlm:8080)"), "the banner shows the origin: " +
                                                               banner);
    require(!banner.contains("SECRET"), "the banner drops the path and query: " + banner);
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
        verify_endpoint_validation_and_banner();
        verify_credentials_and_override_flag();
        verify_buffer_caps();
        verify_http_exposure();
        verify_http_port_disable();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("config-test passed");
    return 0;
}
