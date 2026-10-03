#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "secret.h"

namespace vlm {

// One page's call to the VLM endpoint.
struct VlmCall {
    // OpenAI-compatible endpoint in one of three shapes: a base such as
    // "http://vlm:8080" (an optional path prefix is honored), which gets
    // /v1/chat/completions appended; an OpenAI-style base ending in /v1,
    // which gets /chat/completions; or a full chat completions URL ending
    // in /chat/completions (Docling's ApiVlmOptions.url), used as is. A
    // query string rides along on the request.
    std::string endpoint;
    // Model name forwarded verbatim ("model" field on the wire).
    std::string model;
    // Prompt text accompanying the page image.
    std::string prompt;
    // OpenAI "stop" parameter; empty means the parameter is omitted.
    std::vector<std::string> stop;
    // OpenAI "max_tokens" parameter (per-preset in Docling's specs).
    int max_tokens = 4096;
    // OpenAI "top_logprobs": how many alternates per generated token the
    // endpoint should return. Zero omits the parameter and asks for none.
    int top_logprobs = 0;
    // PNG-encoded page raster.
    std::string png;
    // Wall-clock budget for the whole call in seconds: every attempt and
    // every backoff sleep together, not each socket read.
    long timeout_seconds = 300;
    // Sent as "Authorization: Bearer <key>" when set. The service sets it
    // only for the endpoint the operator configured, never for one a
    // request named.
    Secret api_key{};
    // True once the caller no longer wants the answer (its stream was
    // cancelled or is aborting). Polled between attempts and, from a
    // watchdog thread, while an attempt is in flight, so it must be cheap
    // and thread-safe. Empty means the call is never cancelled.
    std::function<bool()> cancelled{};
};

// One alternate reading the endpoint offered for a generated token.
struct TokenAlternative {
    // The alternate token, verbatim.
    std::string token;
    // Its log-probability; has_logprob is false when the endpoint sent
    // the alternate without a usable score.
    bool has_logprob = false;
    double logprob = 0.0;
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
    // True when the caller's cancel probe cut the call short: nothing is
    // known about the page, and nobody is waiting for it.
    bool cancelled = false;
    // Mean token log-probability over the whole response, verbatim and
    // unrescaled, when the endpoint reported logprobs; has_logprobs is
    // false when it did not (skipped silently). It is a page-wide
    // statistic, never a per-item one, and it is not exponentiated here:
    // a probability is a claim the response does not support.
    bool has_logprobs = false;
    double mean_logprob = 0.0;
    // How many tokens the mean was taken over, so a three-token page and a
    // three-thousand-token page stay distinguishable.
    uint64_t scored_tokens = 0;
    // Alternates the endpoint offered, flattened in generation order:
    // every alternate of token 1, then every alternate of token 2. Empty
    // unless the call asked for top_logprobs.
    std::vector<TokenAlternative> alternatives;
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

// Posts one chat completion to the endpoint (VlmCall::endpoint lists the
// shapes it takes) with the page image inline as a data URL. Blocking;
// meant for the worker pool. Retries like docling's
// api_image_request: up to 5 retries with exponential backoff (100ms
// base) on HTTP 429/500/502/503/504 and on connect-level transport
// failures (vLLM still starting); other statuses, and 200s that do not
// parse, fail without a retry. timeout_seconds bounds the whole call,
// retries and backoff included, and an endpoint that drips bytes cannot
// stretch it. When the caller's probe turns true the call ends at once,
// mid-attempt included (the in-flight socket is shut down), and nothing
// more is sent.
VlmResult generate(const VlmCall& call);

// Test hook: overrides the retry backoff base delay in milliseconds.
// Tests set this to 0 so persistent-failure cases do not sleep ~3s.
void set_retry_backoff_base_ms(long ms);

// Validates an endpoint string enough to fail fast at startup and at RPC
// start: http://host[:port][/path][?query], a host of letters, digits,
// '.', '-', '_' (or a bracketed IPv6 literal), a port from 1 to 65535,
// and no user:password@ part. Empty detail when valid. The detail never
// quotes the endpoint: deployments put tokens in it.
std::string endpoint_error(const std::string& endpoint);

// The endpoint as it may be shown: scheme, host and port only. Userinfo,
// path, query and fragment are dropped on purpose, because deployments put
// tokens in all of them. Startup logs, GetServiceInfo, error text and the
// GenerationSource a fragment records use this, never the endpoint itself.
// Returns "<invalid endpoint>" when the string does not parse.
std::string endpoint_origin(const std::string& endpoint);

}  // namespace vlm
