#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vlm {

// One page's call to the VLM endpoint.
struct VlmCall {
    // OpenAI-compatible base, e.g. "http://vlm:8080" (an optional path
    // prefix is honored). /v1/chat/completions is appended.
    std::string endpoint;
    // Model name forwarded verbatim ("model" field on the wire).
    std::string model;
    // Prompt text accompanying the page image.
    std::string prompt;
    // OpenAI "stop" parameter; empty means the parameter is omitted.
    std::vector<std::string> stop;
    // OpenAI "max_tokens" parameter (per-preset in Docling's specs).
    int max_tokens = 4096;
    // PNG-encoded page raster.
    std::string png;
    // Whole-call timeout in seconds.
    long timeout_seconds = 300;
};

// The endpoint's answer for one page.
struct VlmResult {
    // True when the endpoint answered 200 with a parseable chat
    // completion; false means error carries the reason (HTTP status or
    // transport failure).
    bool ok = false;
    // choices[0].message.content when ok.
    std::string text;
    // Failure detail when !ok.
    std::string error;
    // Mean token probability from logprobs when the endpoint reported
    // them; has_confidence is false when absent (skipped silently).
    bool has_confidence = false;
    double confidence = 0.0;
    // The endpoint's stop reason, verbatim ("stop", "length",
    // "content_filter", ...); empty when it reported none. A "length" stop
    // means the answer was cut at max_tokens and the page is short.
    std::string finish_reason;
    // The model the endpoint says answered, which is not always the one
    // the call asked for; empty when it echoed none.
    std::string model;
    // Token usage the endpoint reported; has_usage is false when absent.
    bool has_usage = false;
    uint64_t prompt_tokens = 0;
    uint64_t completion_tokens = 0;
};

// Calls {endpoint}/v1/chat/completions with the page image inline as a
// data URL. Blocking; meant for the worker pool. Retries like docling's
// api_image_request: up to 5 retries with exponential backoff (100ms
// base) on HTTP 429/500/502/503/504 and on connect-level transport
// failures (vLLM still starting); other statuses, and 200s that do not
// parse, fail without a retry. The configured timeout applies per
// attempt, so a worst-case call takes (1 + retries) × timeout.
VlmResult generate(const VlmCall& call);

// Test hook: overrides the retry backoff base delay in milliseconds.
// Tests set this to 0 so persistent-failure cases do not sleep ~3s.
void set_retry_backoff_base_ms(long ms);

// Validates an endpoint string enough to fail fast at RPC start
// (scheme://host[:port][/path], http only). Empty detail when valid.
std::string endpoint_error(const std::string& endpoint);

// Scheme and authority of an endpoint, dropping any path prefix: what a
// fragment records about who answered. The path is dropped on purpose —
// deployments put tokens in it. Returns the input unchanged when it does
// not parse as an endpoint.
std::string endpoint_origin(const std::string& endpoint);

}  // namespace vlm
