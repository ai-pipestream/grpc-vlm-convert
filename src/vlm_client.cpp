#include "vlm_client.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

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

// How often the watchdog asks the caller's probe whether it still wants
// the answer, and how often it re-stops an attempt after the call trips.
constexpr auto kWatchInterval = std::chrono::milliseconds(50);
// httplib connects while holding the lock Client::stop() needs, so a
// cancel cannot interrupt a connect: cap it well below the call budget.
constexpr auto kConnectTimeout = std::chrono::seconds(10);
// How far past the budget httplib's own timeouts sit, so the budget is
// what ends an attempt and they only back it up.
constexpr auto kTimeoutSlack = std::chrono::milliseconds(200);

// Cuts one call short from outside: the caller's cancel probe and the
// call's own wall-clock budget, which spans every attempt and backoff.
// httplib's read timeout bounds each recv rather than the response, and
// its progress hooks fire only while bytes move, so neither ends an
// attempt parked on a model that has not answered yet, or a caller that
// has gone away. A watchdog thread does: the moment the call trips it
// shuts the in-flight socket down (Client::stop), which fails the blocked
// attempt at once, and it repeats that every interval so an attempt that
// raced past the last check is stopped too.
class CallGuard {
  public:
    CallGuard(std::function<bool()> cancelled, std::chrono::steady_clock::time_point deadline)
        : cancelled_(std::move(cancelled)), deadline_(deadline), thread_([this] { watch(); }) {}

    ~CallGuard() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        wake_.notify_all();
        thread_.join();
    }

    CallGuard(const CallGuard&) = delete;
    CallGuard& operator=(const CallGuard&) = delete;

    // True once the budget is spent or the caller stopped wanting the
    // answer; sticky.
    bool tripped() {
        if (tripped_.load()) {
            return true;
        }
        if (cancelled_ && cancelled_()) {
            by_caller_ = true;
            tripped_ = true;
        } else if (std::chrono::steady_clock::now() >= deadline_) {
            tripped_ = true;
        }
        return tripped_.load();
    }

    // True when the trip came from the caller rather than the budget.
    bool by_caller() const { return by_caller_.load(); }

    // What is left of the budget, never negative.
    std::chrono::milliseconds remaining() const {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline_ - std::chrono::steady_clock::now());
        return std::max(left, std::chrono::milliseconds(0));
    }

    // Registers the attempt in flight (nullptr once it returns). False
    // when the call already tripped: that attempt must not start, and it
    // is not registered (its client is about to be destroyed).
    bool watch(httplib::Client* client) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (client != nullptr && tripped()) {
            client_ = nullptr;
            return false;
        }
        client_ = client;
        return true;
    }

    // A backoff sleep that ends early when the call trips; false then.
    bool sleep_for(std::chrono::milliseconds delay) {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait_for(lock, delay, [&] { return tripped(); });
        return !tripped();
    }

  private:
    void watch() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!done_) {
            const bool was_tripped = tripped_.load();
            if (tripped()) {
                if (!was_tripped) {
                    wake_.notify_all();  // ends a backoff sleep at once
                }
                if (client_ != nullptr) {
                    client_->stop();
                }
            }
            const auto now = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point next;
            if (tripped_.load()) {
                next = now + kWatchInterval;  // keep stopping a racing attempt
            } else if (cancelled_) {
                next = std::min(deadline_, now + kWatchInterval);  // poll the probe
            } else {
                next = deadline_;  // nothing else can trip the call
            }
            wake_.wait_until(lock, next);
        }
    }

    std::function<bool()> cancelled_;
    const std::chrono::steady_clock::time_point deadline_;
    std::atomic<bool> tripped_{false};
    std::atomic<bool> by_caller_{false};
    std::mutex mutex_;
    std::condition_variable wake_;
    httplib::Client* client_ = nullptr;
    bool done_ = false;
    // Last: the watchdog starts once everything above is initialized.
    std::thread thread_;
};

std::string base64_encode(std::string_view bytes) {
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

// An endpoint taken apart: the origin httplib connects to and the request
// target the chat completion posts to.
struct ParsedEndpoint {
    // "http://host[:port]", never with userinfo.
    std::string origin;
    // Path plus query, e.g. "/base/v1/chat/completions?tenant=a".
    std::string target;
};

bool host_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '.' || c == '-' || c == '_';
}

bool ipv6_char(char c) {
    return std::isxdigit(static_cast<unsigned char>(c)) != 0 || c == ':' || c == '.';
}

// Parses http://host[:port][/path][?query][#fragment] into what a request
// needs. Returns an empty string on success, else why the endpoint is
// unusable, phrased without quoting it. httplib's own URL parser takes
// "user:pw@host" apart as host "user", port "pw@host", and leaves the
// client null when the port does not parse, so everything it would choke
// on is refused here first.
std::string parse_endpoint(const std::string& endpoint, ParsedEndpoint* out) {
    constexpr std::string_view kScheme = "http://";
    if (!endpoint.starts_with(kScheme)) {
        return "endpoint must be an http:// URL (https is not supported; put a TLS proxy beside "
               "the VLM)";
    }
    for (const char c : endpoint) {
        const auto byte = static_cast<unsigned char>(c);
        if (byte <= 0x20 || byte == 0x7f) {
            return "endpoint must not contain whitespace or control characters";
        }
    }
    const std::string_view rest = std::string_view(endpoint).substr(kScheme.size());
    const size_t authority_end = rest.find_first_of("/?#");
    const std::string_view authority = rest.substr(0, authority_end);
    if (authority.contains('@')) {
        return "endpoint must not carry credentials (user:password@); set GRPC_VLM_API_KEY "
               "instead";
    }

    std::string_view host = authority;
    std::string_view port;
    bool has_port = false;
    if (authority.starts_with('[')) {
        const size_t close = authority.find(']');
        if (close == std::string_view::npos || close == 1 ||
            !std::ranges::all_of(authority.substr(1, close - 1), ipv6_char)) {
            return "endpoint host must be a name or a bracketed IPv6 literal";
        }
        host = authority.substr(0, close + 1);
        const std::string_view after = authority.substr(close + 1);
        if (!after.empty()) {
            if (after[0] != ':') {
                return "endpoint host must be a name or a bracketed IPv6 literal";
            }
            has_port = true;
            port = after.substr(1);
        }
    } else {
        const size_t colon = authority.find(':');
        if (colon != std::string_view::npos) {
            host = authority.substr(0, colon);
            has_port = true;
            port = authority.substr(colon + 1);
        }
        if (host.empty() || !std::ranges::all_of(host, host_char)) {
            return "endpoint needs a host of letters, digits, '.', '-' or '_'";
        }
    }
    if (has_port) {
        unsigned value = 0;
        const auto [end, failure] = std::from_chars(port.data(), port.data() + port.size(), value);
        if (port.empty() || failure != std::errc() || end != port.data() + port.size() ||
            value < 1 || value > 65535) {
            return "endpoint port must be a number from 1 to 65535";
        }
    }
    out->origin = "http://" + std::string(host) + (has_port ? ":" + std::string(port) : "");

    // Path and query; a fragment never leaves the client.
    std::string_view tail =
        authority_end == std::string_view::npos ? std::string_view() : rest.substr(authority_end);
    tail = tail.substr(0, tail.find('#'));
    const size_t question = tail.find('?');
    std::string path(tail.substr(0, question));
    const std::string query(question == std::string_view::npos ? std::string_view()
                                                                : tail.substr(question));
    // Trailing slashes would produce "//v1/chat/completions".
    while (path.ends_with('/')) {
        path.pop_back();
    }
    // A full completions URL (Docling's ApiVlmOptions.url) is used as is;
    // an OpenAI-style base ending in /v1 gets the rest of the route; any
    // other base gets the whole route.
    if (path.ends_with("/chat/completions")) {
        out->target = path;
    } else if (path.ends_with("/v1")) {
        out->target = path + "/chat/completions";
    } else {
        out->target = path + "/v1/chat/completions";
    }
    out->target += query;
    return "";
}

}  // namespace

void set_retry_backoff_base_ms(long ms) { g_backoff_base_ms.store(ms); }

std::string endpoint_error(const std::string& endpoint) {
    ParsedEndpoint parsed;
    return parse_endpoint(endpoint, &parsed);
}

std::string endpoint_origin(const std::string& endpoint) {
    ParsedEndpoint parsed;
    if (!parse_endpoint(endpoint, &parsed).empty()) {
        return "<invalid endpoint>";
    }
    return parsed.origin;
}

VlmResult generate(const VlmCall& call) {
    VlmResult result;
    ParsedEndpoint where;
    if (std::string problem = parse_endpoint(call.endpoint, &where); !problem.empty()) {
        result.error = std::move(problem);
        return result;
    }

    // A caller that already left gets nothing sent on its behalf.
    if (call.cancelled && call.cancelled()) {
        result.cancelled = true;
        result.error = "VLM call cancelled before it started";
        return result;
    }

    std::string payload;
    {
        // The request DOM holds a base64 copy of the page; it is freed
        // before the call, so only the serialized payload waits on the
        // endpoint.
        nlohmann::json body = {
            {"model", call.model},
            {"messages",
             {{{"role", "user"},
               {"content",
                {{{"type", "text"}, {"text", call.prompt}},
                 {{"type", "image_url"},
                  {"image_url",
                   {{"url", "data:image/png;base64," + base64_encode(call.png)}}}}}}}}},
            {"max_tokens", call.max_tokens},
        };
        // Some OpenAI-compatible servers reject the parameter outright, so
        // an operator can leave it off; alternates cannot do without it.
        if (call.logprobs || call.top_logprobs > 0) {
            body["logprobs"] = true;
        }
        if (!call.stop.empty()) {
            body["stop"] = call.stop;
        }
        // One key buys the alternates the model weighed per token; asking
        // for none is the default, so the parameter is omitted rather than
        // zeroed.
        if (call.top_logprobs > 0) {
            body["top_logprobs"] = call.top_logprobs;
        }
        payload = body.dump();
    }
    httplib::Headers headers;
    if (!call.api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + call.api_key.reveal());
    }

    // One budget for the whole call: every attempt and every backoff sleep
    // spend from it, and the caller's probe can end it at any point.
    CallGuard guard(call.cancelled, std::chrono::steady_clock::now() +
                                        std::chrono::seconds(std::max(call.timeout_seconds, 0L)));
    httplib::Result response;
    int retries = 0;
    while (!guard.tripped()) {
        {
            // A fresh client per attempt: after a connect-level failure the
            // previous one's socket state is useless anyway.
            httplib::Client client(where.origin);
            if (!client.is_valid()) {
                // parse_endpoint refuses what httplib cannot take apart, so
                // this is a backstop: a client that did not construct must
                // never be used (its calls dereference null).
                result.error = "endpoint is not usable: " + where.origin;
                return result;
            }
            // httplib's own timeouts sit just past what is left of the
            // budget: the budget (the watchdog) ends the attempt, and they
            // are the backstop. max_timeout bounds the whole response read,
            // so an endpoint dripping a byte at a time cannot stretch an
            // attempt even then.
            const std::chrono::milliseconds limit = guard.remaining() + kTimeoutSlack;
            client.set_connection_timeout(
                std::min<std::chrono::milliseconds>(limit, kConnectTimeout));
            client.set_read_timeout(limit);
            client.set_write_timeout(limit);
            client.set_max_timeout(limit);
            if (!guard.watch(&client)) {
                break;
            }
            // The upload hook stops a large page mid-send; the watchdog
            // covers the wait for the answer, where no bytes move.
            response = client.Post(where.target, headers, payload, "application/json",
                                   [&guard](size_t, size_t) { return !guard.tripped(); });
            guard.watch(nullptr);
        }
        // An answer that made it in is used even if the budget ran out
        // while it arrived.
        if (response && response->status == 200) {
            break;
        }
        if (guard.tripped()) {
            break;
        }
        const bool retryable = response ? retryable_status(response->status)
                                        : retryable_transport(response.error());
        if (!retryable || retries == kMaxRetries) {
            break;
        }
        retries++;
        // Exponential backoff like urllib3: base * 2^(retries-1) —
        // 0.1s, 0.2s, 0.4s, ... at the default base.
        const long delay_ms = g_backoff_base_ms.load() << (retries - 1);
        if (delay_ms > 0 && !guard.sleep_for(std::chrono::milliseconds(delay_ms))) {
            break;
        }
    }
    if (!(response && response->status == 200) && guard.tripped()) {
        if (guard.by_caller()) {
            result.cancelled = true;
            result.error = "VLM call cancelled";
        } else {
            result.error = "VLM call exceeded its " + std::to_string(call.timeout_seconds) +
                           " s budget";
            if (retries > 0) {
                result.error += " after " + std::to_string(retries + 1) + " attempts";
            }
        }
        return result;
    }
    if (!response) {
        result.error = "endpoint unreachable: " + where.origin;
        return result;
    }
    if (response->status != 200) {
        result.error = "endpoint returned HTTP " + std::to_string(response->status);
        if (retries > 0) {
            result.error += " after " + std::to_string(retries + 1) + " attempts";
        }
        return result;
    }

    const nlohmann::json parsed = nlohmann::json::parse(response->body, nullptr, false);
    if (parsed.is_discarded()) {
        result.error = "endpoint returned non-JSON body";
        return result;
    }
    // Anything but an object fails here: a keyed lookup on an array, a
    // string or a number throws, and a throw out of a worker thread ends
    // the process.
    if (!parsed.is_object()) {
        result.error = "endpoint returned JSON that is not a chat completion object";
        return result;
    }
    // Key access goes through find()/contains(): const operator[] on a
    // missing key is undefined behavior, and endpoints omit fields freely.
    const auto choices_entry = parsed.find("choices");
    if (choices_entry == parsed.end()) {
        result.error = "chat completion has no message content";
        return result;
    }
    const auto& choices = *choices_entry;
    if (!choices.is_array() || choices.empty() || !choices[0].is_object() ||
        !choices[0].contains("message") || !choices[0]["message"].is_object() ||
        !choices[0]["message"].contains("content") ||
        !choices[0]["message"]["content"].is_string()) {
        result.error = "chat completion has no message content";
        return result;
    }
    result.text = choices[0]["message"]["content"].get<std::string>();

    // Generation facts: how the answer terminated, which model produced
    // it, and what it cost. All optional on the wire, all recorded when
    // present — a "length" finish_reason is the only marker that a short
    // page is a truncation rather than a short page.
    if (const auto reason = choices[0].find("finish_reason");
        reason != choices[0].end() && reason->is_string()) {
        result.finish_reason = reason->get<std::string>();
    }
    if (const auto model = parsed.find("model");
        model != parsed.end() && model->is_string()) {
        result.model = model->get<std::string>();
    }
    if (const auto usage = parsed.find("usage");
        usage != parsed.end() && usage->is_object()) {
        if (const auto prompt = usage->find("prompt_tokens");
            prompt != usage->end() && prompt->is_number_unsigned()) {
            result.prompt_tokens = prompt->get<uint64_t>();
            result.has_usage = true;
        }
        if (const auto completion = usage->find("completion_tokens");
            completion != usage->end() && completion->is_number_unsigned()) {
            result.completion_tokens = completion->get<uint64_t>();
            result.has_usage = true;
        }
    }

    // Logprobs are optional. The mean token log-probability is kept as
    // the endpoint reported it: not exponentiated, not clamped. Rescaling
    // it into a probability destroys the only signal it carries, and
    // clamping hides an endpoint that reports nonsense. Absent means
    // skipped silently.
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
                // The alternates this token was chosen over, when the call
                // asked for them. Kept verbatim and in order: they are the
                // model's own n-best reading of the page.
                const auto top = token.find("top_logprobs");
                if (top == token.end() || !top->is_array()) {
                    continue;
                }
                for (const auto& alternate : *top) {
                    if (!alternate.is_object()) {
                        continue;
                    }
                    const auto text = alternate.find("token");
                    if (text == alternate.end() || !text->is_string()) {
                        continue;
                    }
                    TokenAlternative entry;
                    entry.token = text->get<std::string>();
                    if (const auto score = alternate.find("logprob");
                        score != alternate.end() && score->is_number()) {
                        entry.has_logprob = true;
                        entry.logprob = score->get<double>();
                    }
                    result.alternatives.push_back(std::move(entry));
                }
            }
            if (count > 0) {
                result.has_logprobs = true;
                result.mean_logprob = sum / static_cast<double>(count);
                result.scored_tokens = count;
            }
        }
    }
    result.ok = true;
    return result;
}

}  // namespace vlm
