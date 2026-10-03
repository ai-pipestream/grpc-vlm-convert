#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "secret.h"

namespace vlm {

// Process configuration, entirely from GRPC_VLM_* environment variables.
// Malformed values throw at startup; nothing is silently defaulted away
// from what the operator wrote.
struct Config {
    std::string listen_address = "0.0.0.0:50058";
    // OpenAI-compatible VLM endpoint, e.g. "http://vlm:8080" or a full
    // ".../v1/chat/completions" URL. Empty is legal at startup —
    // ConvertPages then requires a per-request endpoint override and fails
    // with FAILED_PRECONDITION otherwise. Never printed or returned whole:
    // only endpoint_origin() of it leaves the process.
    std::string endpoint;
    // Bearer key for the configured endpoint (GRPC_VLM_API_KEY), sent as
    // "Authorization: Bearer <key>" on every call to it. Never printed,
    // returned or echoed, and never sent to an endpoint a request named.
    Secret vlm_api_key;
    // Whether ConvertOptions.endpoint may point a stream at another VLM
    // endpoint (GRPC_VLM_ALLOW_ENDPOINT_OVERRIDE=true). Off by default: an
    // override lets any caller make this process POST page images to any
    // http host it can reach and read the answer back.
    bool allow_endpoint_override = false;
    // Whether calls ask for "logprobs": true (GRPC_VLM_LOGPROBS), the page
    // score. Some OpenAI-compatible servers reject the parameter with a
    // 400; turn it off for them.
    bool request_logprobs = true;
    // Preset names the configured endpoint claims to serve (comma list).
    // Empty means "every built-in preset" when an endpoint is set.
    std::vector<std::string> presets;
    // Pages in flight against the VLM per stream (server default and
    // clamp for ConvertOptions.concurrency).
    size_t concurrency = 2;
    // VLM calls in flight across every stream (GRPC_VLM_MAX_INFLIGHT). The
    // endpoint's capacity is per process, not per stream: N streams at
    // full concurrency would otherwise queue N × concurrency requests on
    // it, each with its timeout running. A page keeps its slot while its
    // answer is mapped, so this also bounds the rasters decoded at once.
    size_t max_inflight = 8;
    size_t max_page_bytes = 32ULL * 1024 * 1024;
    size_t max_pages = 512;
    // Bytes of picture crops one PageDocument may carry inline
    // (GRPC_VLM_MAX_PAGE_CROP_BYTES, as data URIs). Past it the page's
    // remaining pictures go without an image and a PageWarning says so;
    // the default keeps a page under gRPC's default 4 MiB receive limit.
    size_t max_page_crop_bytes = 3ULL * 1024 * 1024;
    // Bytes of page images one stream may hold read but not yet answered,
    // queued or in flight (GRPC_VLM_MAX_STREAM_BUFFERED_BYTES, default 4 ×
    // max_page_bytes). A stream at the cap stops reading, so gRPC flow
    // control pushes back on its client instead of the server buffering
    // whatever it is sent.
    size_t max_stream_buffered_bytes = 4 * (32ULL * 1024 * 1024);
    // The same bound across every stream (GRPC_VLM_MAX_BUFFERED_BYTES,
    // default 16 × max_page_bytes).
    size_t max_buffered_bytes = 16 * (32ULL * 1024 * 1024);
    // Wall-clock budget for one page's VLM call, every attempt and retry
    // backoff included.
    size_t vlm_timeout_seconds = 300;
    // 0 disables the stdout metrics line.
    size_t metrics_interval_seconds = 60;
    // HTTP/JSON front-end port (POST /v1/convert, /v1/convert/stream,
    // GET /healthz). 0 disables the HTTP listener.
    size_t http_port = 50059;
    // Address the HTTP front end binds (GRPC_VLM_HTTP_HOST). Loopback by
    // default: it converts pages (paid VLM calls) for whoever reaches it,
    // so binding anywhere else requires http_token.
    std::string http_host = "127.0.0.1";
    // Largest request body the HTTP front end reads
    // (GRPC_VLM_HTTP_MAX_BODY_BYTES); a larger one is 413 before more than
    // this is buffered. A request is held several times over (body, JSON,
    // protobuf), so this bounds the front end's memory per request.
    size_t http_max_body_bytes = 64ULL * 1024 * 1024;
    // Bearer token every HTTP convert request must carry
    // (GRPC_VLM_HTTP_TOKEN) as "Authorization: Bearer <token>"; /healthz
    // stays open. Required when http_host is not a loopback address.
    // Never printed or echoed.
    Secret http_token;
};

// Reads and validates the environment. Throws std::invalid_argument with
// the variable name and accepted range on any malformed value; a malformed
// endpoint is named by its variable, never quoted.
Config load_config_from_env();

// The line main prints once it listens: the listen addresses and the
// endpoint, reduced to scheme, host and port (endpoint_origin) because
// deployments put tokens in its path and query. It never names a
// credential.
std::string startup_banner(const Config& config);

}  // namespace vlm
