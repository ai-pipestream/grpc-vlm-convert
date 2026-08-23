#include "vlm_client.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace vlm {

namespace {

// Docling's api_image_request retry policy (urllib3 Retry): 5 retries,
// exponential backoff with a 0.1s factor, on these statuses.
constexpr int kMaxRetries = 5;
std::atomic<long> g_backoff_base_ms{100};

bool retryable_status(int status) {
    return status == 429 || status == 500 || status == 502 || status == 503 ||
           status == 504;
}

// Connect-level failures retry (connection refused while vLLM starts);
// mid-request read/write/SSL failures do not — docling's Retry has
// connect=5 but read=0.
bool retryable_transport(httplib::Error error) {
    return error == httplib::Error::Connection || error == httplib::Error::ConnectionTimeout;
}

std::string base64_encode(const std::string& bytes) {
    static const char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t i = 0; i < bytes.size(); i += 3) {
        unsigned group = static_cast<unsigned char>(bytes[i]) << 16;
        if (i + 1 < bytes.size()) {
            group |= static_cast<unsigned char>(bytes[i + 1]) << 8;
        }
        if (i + 2 < bytes.size()) {
            group |= static_cast<unsigned char>(bytes[i + 2]);
        }
        out.push_back(kAlphabet[(group >> 18) & 63]);
        out.push_back(kAlphabet[(group >> 12) & 63]);
        out.push_back(i + 1 < bytes.size() ? kAlphabet[(group >> 6) & 63] : '=');
        out.push_back(i + 2 < bytes.size() ? kAlphabet[group & 63] : '=');
    }
    return out;
}

// Splits "http://host:port/base" into the origin httplib connects to and
// the path prefix requests go under.
bool split_endpoint(const std::string& endpoint, std::string* origin, std::string* path) {
    const std::string scheme = "http://";
    if (!endpoint.starts_with(scheme) || endpoint.size() == scheme.size()) {
        return false;
    }
    size_t slash = endpoint.find('/', scheme.size());
    if (slash == std::string::npos) {
        *origin = endpoint;
        *path = "";
    } else {
        *origin = endpoint.substr(0, slash);
        *path = endpoint.substr(slash);
        // Trailing slashes would produce "//v1/chat/completions".
        while (path->ends_with('/')) {
            path->pop_back();
        }
    }
    return true;
}

}  // namespace

void set_retry_backoff_base_ms(long ms) { g_backoff_base_ms.store(ms); }

std::string endpoint_error(const std::string& endpoint) {
    std::string origin, path;
    if (!split_endpoint(endpoint, &origin, &path)) {
        return "endpoint must be http://host[:port][/path], got: " + endpoint;
    }
    return "";
}

VlmResult generate(const VlmCall& call) {
    VlmResult result;
    std::string origin, path;
    if (!split_endpoint(call.endpoint, &origin, &path)) {
        result.error = endpoint_error(call.endpoint);
        return result;
    }

    nlohmann::json body = {
        {"model", call.model},
        {"messages",
         {{{"role", "user"},
           {"content",
            {{{"type", "text"}, {"text", call.prompt}},
             {{"type", "image_url"},
              {"image_url", {{"url", "data:image/png;base64," + base64_encode(call.png)}}}}}}}}},
        {"max_tokens", call.max_tokens},
        {"logprobs", true},
    };
    if (!call.stop.empty()) {
        body["stop"] = call.stop;
    }

    const std::string payload = body.dump();
    httplib::Result response;
    int retries = 0;
    for (;;) {
        // A fresh client per attempt: after a connect-level failure the
        // previous one's socket state is useless anyway.
        httplib::Client client(origin);
        client.set_connection_timeout(call.timeout_seconds, 0);
        client.set_read_timeout(call.timeout_seconds, 0);
        client.set_write_timeout(call.timeout_seconds, 0);
        response = client.Post(path + "/v1/chat/completions", payload, "application/json");
        const bool retryable = response ? retryable_status(response->status)
                                        : retryable_transport(response.error());
        if (!retryable || retries == kMaxRetries) {
            break;
        }
        retries++;
        // Exponential backoff like urllib3: base * 2^(retries-1) —
        // 0.1s, 0.2s, 0.4s, ... at the default base.
        const long delay_ms = g_backoff_base_ms.load() << (retries - 1);
        if (delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }
    if (!response) {
        result.error = "endpoint unreachable: " + origin;
        return result;
    }
    if (response->status != 200) {
        result.error = "endpoint returned HTTP " + std::to_string(response->status);
        if (retries > 0) {
            result.error += " after " + std::to_string(retries + 1) + " attempts";
        }
        return result;
    }

    nlohmann::json parsed = nlohmann::json::parse(response->body, nullptr, false);
    if (parsed.is_discarded()) {
        result.error = "endpoint returned non-JSON body";
        return result;
    }
    // Key access goes through contains(): const operator[] on a missing
    // key is undefined behavior, and endpoints omit fields freely.
    const auto& choices = parsed["choices"];
    if (!choices.is_array() || choices.empty() || !choices[0].is_object() ||
        !choices[0].contains("message") || !choices[0]["message"].is_object() ||
        !choices[0]["message"].contains("content") ||
        !choices[0]["message"]["content"].is_string()) {
        result.error = "chat completion has no message content";
        return result;
    }
    result.text = choices[0]["message"]["content"].get<std::string>();

    // Logprobs are optional (Docling's OpenAI VLM logprobs knob). Mean
    // token probability becomes the CollectorSource confidence; absent
    // means skipped silently.
    const auto& first = choices[0];
    if (first.contains("logprobs") && first["logprobs"].is_object() &&
        first["logprobs"].contains("content")) {
        const auto& logprobs = first["logprobs"]["content"];
        if (logprobs.is_array() && !logprobs.empty()) {
            double sum = 0.0;
            size_t count = 0;
            for (const auto& token : logprobs) {
                // Endpoints emit malformed logprob entries in the wild;
                // anything that is not an object with a numeric logprob
                // is skipped. (const operator[] here would throw — or
                // worse — on non-objects and missing keys.)
                if (!token.is_object()) {
                    continue;
                }
                if (const auto logprob = token.find("logprob");
                    logprob != token.end() && logprob->is_number()) {
                    sum += logprob->get<double>();
                    count++;
                }
            }
            if (count > 0) {
                result.has_confidence = true;
                // Positive logprobs are endpoint garbage; keep the
                // probability in [0, 1] anyway.
                result.confidence =
                    std::min(1.0, std::exp(sum / static_cast<double>(count)));
            }
        }
    }
    result.ok = true;
    return result;
}

}  // namespace vlm
