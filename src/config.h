#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace vlm {

// Process configuration, entirely from GRPC_VLM_* environment variables.
// Malformed values throw at startup; nothing is silently defaulted away
// from what the operator wrote.
struct Config {
    std::string listen_address = "0.0.0.0:50058";
    // OpenAI-compatible VLM endpoint, e.g. "http://vlm:8080". Empty is
    // legal at startup — ConvertPages then requires a per-request
    // endpoint override and fails with FAILED_PRECONDITION otherwise.
    std::string endpoint;
    // Preset names the configured endpoint claims to serve (comma list).
    // Empty means "every built-in preset" when an endpoint is set.
    std::vector<std::string> presets;
    // Pages in flight against the VLM per stream (server default and
    // clamp for ConvertOptions.concurrency).
    size_t concurrency = 2;
    size_t max_page_bytes = 32ULL * 1024 * 1024;
    size_t max_pages = 512;
    // Deadline for one page's HTTP call to the VLM.
    size_t vlm_timeout_seconds = 300;
    // 0 disables the stdout metrics line.
    size_t metrics_interval_seconds = 60;
};

// Reads and validates the environment. Throws std::invalid_argument with
// the variable name and accepted range on any malformed value.
Config load_config_from_env();

}  // namespace vlm
