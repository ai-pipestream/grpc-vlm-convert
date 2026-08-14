#pragma once

#include <string>

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
};

// Calls {endpoint}/v1/chat/completions with the page image inline as a
// data URL. Blocking; meant for the worker pool.
VlmResult generate(const VlmCall& call);

// Validates an endpoint string enough to fail fast at RPC start
// (scheme://host[:port][/path], http only). Empty detail when valid.
std::string endpoint_error(const std::string& endpoint);

}  // namespace vlm
