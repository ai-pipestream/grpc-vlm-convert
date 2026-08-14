#include "vlm_client.h"

#include <cmath>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace vlm {

namespace {

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
    if (endpoint.compare(0, scheme.size(), scheme) != 0 ||
        endpoint.size() == scheme.size()) {
        return false;
    }
    size_t slash = endpoint.find('/', scheme.size());
    if (slash == std::string::npos) {
        *origin = endpoint;
        *path = "";
    } else {
        *origin = endpoint.substr(0, slash);
        *path = endpoint.substr(slash);
    }
    return true;
}

}  // namespace

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
        {"max_tokens", 4096},
        {"logprobs", true},
    };

    httplib::Client client(origin);
    client.set_connection_timeout(call.timeout_seconds, 0);
    client.set_read_timeout(call.timeout_seconds, 0);
    client.set_write_timeout(call.timeout_seconds, 0);
    auto response = client.Post(path + "/v1/chat/completions", body.dump(), "application/json");
    if (!response) {
        result.error = "endpoint unreachable: " + origin;
        return result;
    }
    if (response->status != 200) {
        result.error = "endpoint returned HTTP " + std::to_string(response->status);
        return result;
    }

    nlohmann::json parsed = nlohmann::json::parse(response->body, nullptr, false);
    if (parsed.is_discarded()) {
        result.error = "endpoint returned non-JSON body";
        return result;
    }
    const auto& choices = parsed["choices"];
    if (!choices.is_array() || choices.empty() ||
        !choices[0]["message"]["content"].is_string()) {
        result.error = "chat completion has no message content";
        return result;
    }
    result.text = choices[0]["message"]["content"].get<std::string>();

    // Logprobs are optional (Docling's OpenAI VLM logprobs knob). Mean
    // token probability becomes the CollectorSource confidence; absent
    // means skipped silently.
    const auto& logprobs = choices[0]["logprobs"]["content"];
    if (logprobs.is_array() && !logprobs.empty()) {
        double sum = 0.0;
        size_t count = 0;
        for (const auto& token : logprobs) {
            if (token["logprob"].is_number()) {
                sum += token["logprob"].get<double>();
                count++;
            }
        }
        if (count > 0) {
            result.has_confidence = true;
            result.confidence = std::exp(sum / static_cast<double>(count));
        }
    }
    result.ok = true;
    return result;
}

}  // namespace vlm
