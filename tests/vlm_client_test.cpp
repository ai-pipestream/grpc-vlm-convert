// Unit tests for the VLM HTTP client retry policy (docling's
// api_image_request: 5 retries, exponential backoff, on 429/500/502/503/
// 504 and connect-level transport failures) against an in-process fake
// server that counts attempts. Backoff is pinned to zero — the policy's
// attempt counts are the assertions, not the wall-clock delays.

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "fixture.h"
#include "vlm_client.h"

namespace {

// A scriptable /v1/chat/completions: failures_before_success requests get
// failure_status, then a valid chat completion (or an unparseable 200
// when garbage_200 is set).
struct ScriptableVlm {
    httplib::Server server;
    std::thread thread;
    int port = 0;
    std::atomic<long> attempts{0};
    std::atomic<int> failures_before_success{0};
    std::atomic<int> failure_status{503};
    std::atomic<bool> garbage_200{false};
    // When set, the 200 carries the generation facts an OpenAI-compatible
    // endpoint reports: the answering model, the stop reason, usage.
    std::atomic<bool> report_generation{false};
    // The top_logprobs value the last request asked for, -1 when the
    // request omitted the parameter.
    std::atomic<int> asked_top_logprobs{-1};
    // The request target (path and query) of the last call, as it arrived.
    std::mutex target_mutex;
    std::string last_target;
    // Set when a parked /hang call saw its caller hang up.
    std::atomic<bool> hang_saw_disconnect{false};

    std::string target() {
        std::lock_guard<std::mutex> lock(target_mutex);
        return last_target;
    }

    void start() {
        // The same completions handler under a base path: the client must
        // post to {prefix}/v1/chat/completions for prefixed endpoints, and
        // to {base}/chat/completions for an OpenAI-style base ending in /v1.
        const auto handler = [this](const httplib::Request& request,
                                    httplib::Response& response) {
            attempts++;
            {
                std::lock_guard<std::mutex> lock(target_mutex);
                last_target = request.target;
            }
            const nlohmann::json asked =
                nlohmann::json::parse(request.body, nullptr, false);
            asked_top_logprobs = asked.is_object() && asked.contains("top_logprobs")
                                     ? asked["top_logprobs"].get<int>()
                                     : -1;
            if (failures_before_success.load() > 0) {
                failures_before_success--;
                response.status = failure_status.load();
                response.set_content("{\"error\":\"model overloaded\"}", "application/json");
                return;
            }
            if (garbage_200.load()) {
                response.set_content("not json at all", "text/plain");
                return;
            }
            nlohmann::json reply = {
                {"choices",
                 {{{"message", {{"role", "assistant"}, {"content", "<doctag/>"}}}}}}};
            if (asked_top_logprobs.load() > 0) {
                // What an endpoint returns for "top_logprobs": N — the
                // chosen token first, then the runners-up, per token. The
                // last alternate carries no score, as endpoints do emit.
                reply["choices"][0]["logprobs"]["content"] = {
                    {{"token", "cat"},
                     {"logprob", -0.1},
                     {"top_logprobs",
                      {{{"token", "cat"}, {"logprob", -0.1}},
                       {{"token", "car"}, {"logprob", -2.5}}}}},
                    {{"token", "sat"},
                     {"logprob", -0.3},
                     {"top_logprobs",
                      {{{"token", "sat"}, {"logprob", -0.3}},
                       {{"token", "set"}}}}},
                };
            }
            if (report_generation.load()) {
                reply["model"] = "served-model-b";
                reply["choices"][0]["finish_reason"] = "length";
                reply["usage"] = {{"prompt_tokens", 1200},
                                  {"completion_tokens", 4096},
                                  {"total_tokens", 5296}};
            }
            response.set_content(reply.dump(), "application/json");
        };
        server.Post("/v1/chat/completions", handler);
        server.Post("/base/v1/chat/completions", handler);
        server.Post("/gw/v1/chat/completions", handler);

        // A slow drip: one byte every 100 ms keeps every socket read alive,
        // so only a whole-call budget ends the call.
        server.Post("/drip/v1/chat/completions",
                    [](const httplib::Request&, httplib::Response& response) {
                        auto started = std::make_shared<std::chrono::steady_clock::time_point>(
                            std::chrono::steady_clock::now());
                        response.set_chunked_content_provider(
                            "application/json", [started](size_t, httplib::DataSink& sink) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                                if (std::chrono::steady_clock::now() - *started >
                                    std::chrono::seconds(20)) {
                                    sink.done();
                                    return true;
                                }
                                return sink.write(" ", 1);
                            });
                    });
        // A model that never answers: the handler parks until the caller
        // hangs up, and records that it saw the connection close.
        server.Post("/hang/v1/chat/completions",
                    [this](const httplib::Request& request, httplib::Response& response) {
                        attempts++;
                        const auto started = std::chrono::steady_clock::now();
                        while (!request.is_connection_closed() &&
                               std::chrono::steady_clock::now() - started <
                                   std::chrono::seconds(20)) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        }
                        if (request.is_connection_closed()) {
                            hang_saw_disconnect = true;
                        }
                        response.status = 503;
                    });
        // A busy model: 400 ms per attempt, then 503 (retryable).
        server.Post("/slow503/v1/chat/completions",
                    [this](const httplib::Request&, httplib::Response& response) {
                        attempts++;
                        std::this_thread::sleep_for(std::chrono::milliseconds(400));
                        response.status = 503;
                    });
        port = server.bind_to_any_port("127.0.0.1");
        require(port > 0, "fake VLM bound");
        thread = std::thread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }

    void stop() {
        server.stop();
        thread.join();
    }

    std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port); }
};

vlm::VlmCall call_to(const std::string& endpoint) {
    return {.endpoint = endpoint,
            .model = "test-model",
            .prompt = "prompt",
            .stop = {},
            .max_tokens = 4096,
            .png = "fake-png-bytes",
            .timeout_seconds = 5};
}

bool wait_until(const std::function<bool()>& condition) {
    for (int i = 0; i < 300; i++) {  // 3 s budget, ms-scale in practice
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

std::chrono::milliseconds elapsed_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
}

// The timeout is a budget for the whole call and the caller's probe ends
// it at any point: mid-wait on a model that has not answered, mid-drip,
// between retries, or before anything is sent.
void verify_budget_and_cancellation(ScriptableVlm& fake) {
    {
        vlm::VlmCall drip = call_to(fake.endpoint() + "/drip");
        drip.timeout_seconds = 1;
        const auto started = std::chrono::steady_clock::now();
        const vlm::VlmResult dripped = vlm::generate(drip);
        require(!dripped.ok && dripped.error.contains("budget"),
                "a dripping endpoint fails on the budget: " + dripped.error);
        require(elapsed_since(started) < std::chrono::seconds(4),
                "a byte every 100 ms does not stretch the call past its budget");
    }
    {
        fake.attempts = 0;
        vlm::VlmCall slow = call_to(fake.endpoint() + "/slow503");
        slow.timeout_seconds = 1;
        const auto started = std::chrono::steady_clock::now();
        const vlm::VlmResult spent = vlm::generate(slow);
        require(!spent.ok && spent.error.contains("budget"),
                "retries end when the budget does: " + spent.error);
        require(fake.attempts.load() < 6, "the budget, not the retry count, ended the call");
        require(elapsed_since(started) < std::chrono::milliseconds(2500),
                "retries and backoff spend from one budget");
    }
    {
        // Parked on a model that has not answered: no bytes move, so only
        // the watchdog can end it, and the endpoint sees the hang-up.
        fake.hang_saw_disconnect = false;
        std::atomic<bool> leave{false};
        vlm::VlmCall hung = call_to(fake.endpoint() + "/hang");
        hung.timeout_seconds = 30;
        hung.cancelled = [&leave] { return leave.load(); };
        std::jthread timer([&leave] {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            leave = true;
        });
        const auto started = std::chrono::steady_clock::now();
        const vlm::VlmResult cut = vlm::generate(hung);
        require(!cut.ok && cut.cancelled, "a cancelled call says so: " + cut.error);
        require(elapsed_since(started) < std::chrono::seconds(3),
                "the cancel reaches an attempt parked on the model");
        require(wait_until([&] { return fake.hang_saw_disconnect.load(); }),
                "the endpoint sees the connection close");
    }
    {
        // Cancelled between attempts: no further attempt is made.
        fake.attempts = 0;
        std::atomic<bool> leave{false};
        vlm::VlmCall busy = call_to(fake.endpoint() + "/slow503");
        busy.timeout_seconds = 30;
        busy.cancelled = [&leave] { return leave.load(); };
        std::jthread timer([&] {
            wait_until([&] { return fake.attempts.load() >= 1; });
            leave = true;
        });
        const vlm::VlmResult cut = vlm::generate(busy);
        require(!cut.ok && cut.cancelled, "a cancel between retries ends the call");
        require(fake.attempts.load() <= 2, "no attempt starts after the cancel");
    }
    {
        // A caller that already left gets nothing sent on its behalf.
        fake.attempts = 0;
        vlm::VlmCall gone = call_to(fake.endpoint());
        gone.cancelled = [] { return true; };
        const vlm::VlmResult nothing = vlm::generate(gone);
        require(!nothing.ok && nothing.cancelled, "an already-cancelled call is cancelled");
        require(fake.attempts.load() == 0, "nothing reaches the endpoint");
    }
}

}  // namespace

int main() {
    vlm::set_retry_backoff_base_ms(0);
    ScriptableVlm fake;
    fake.start();
    try {
        // Endpoint validation: plaintext http only, host required, an
        // optional path prefix allowed.
        require(vlm::endpoint_error("http://vlm:8080").empty(), "plain origin validates");
        require(vlm::endpoint_error("http://vlm:8080/base").empty(), "path prefix validates");
        require(vlm::endpoint_error("http://vlm").empty(), "a port is optional");
        require(vlm::endpoint_error("http://[::1]:8080").empty(), "bracketed IPv6 validates");
        require(vlm::endpoint_error("http://vlm:8080/v1/chat/completions?api-version=2").empty(),
                "a full completions URL with a query validates");
        require(!vlm::endpoint_error("https://vlm:8080").empty(), "https is rejected");
        require(!vlm::endpoint_error("http://").empty(), "scheme without a host is rejected");
        require(!vlm::endpoint_error("vlm:8080").empty(), "missing scheme is rejected");
        // What httplib would take apart wrongly (and then dereference a null
        // client for) is refused up front.
        for (const char* broken : {"http://vlm:abc", "http://vlm:0", "http://vlm:99999",
                                   "http://vlm:", "http://[::1", "http://vl m:8080",
                                   "http://vlm:8080/a b", "http://vlm\n:8080"}) {
            require(!vlm::endpoint_error(broken).empty(),
                    std::string("malformed endpoint is rejected: ") + broken);
        }
        // Credentials in the URL are refused, and the reason never repeats
        // them.
        const std::string with_userinfo = "http://alice:hunter2@vlm:8080/v1";
        const std::string userinfo_error = vlm::endpoint_error(with_userinfo);
        require(!userinfo_error.empty(), "userinfo is rejected");
        require(!userinfo_error.contains("hunter2") && !userinfo_error.contains("alice"),
                "the rejection does not echo the credentials: " + userinfo_error);
        require(!vlm::endpoint_error("http://vlm:8080/tenant/SECRET-TOKEN x").contains(
                    "SECRET-TOKEN"),
                "a rejection never quotes the endpoint");

        // A bad port is an error result, not a crash (httplib leaves its
        // client null when the port does not parse).
        vlm::VlmResult bad_port = vlm::generate(call_to("http://127.0.0.1:99999"));
        require(!bad_port.ok && !bad_port.error.empty(), "a bad port fails cleanly");

        // A path-prefixed endpoint posts under the prefix.
        vlm::VlmResult prefixed = vlm::generate(call_to(fake.endpoint() + "/base"));
        require(prefixed.ok, "prefixed endpoint resolves: " + prefixed.error);
        require(prefixed.text == "<doctag/>", "prefixed endpoint answer");
        require(fake.target() == "/base/v1/chat/completions", "prefix gets the whole route");

        // A Docling-style full completions URL is used as is, never with
        // a second /v1/chat/completions on the end.
        vlm::VlmResult full = vlm::generate(call_to(fake.endpoint() + "/v1/chat/completions"));
        require(full.ok, "a full completions URL resolves: " + full.error);
        require(fake.target() == "/v1/chat/completions", "a full URL is not doubled");
        full = vlm::generate(call_to(fake.endpoint() + "/base/v1/chat/completions/"));
        require(full.ok && fake.target() == "/base/v1/chat/completions",
                "a full URL with a prefix and a trailing slash is used as is");

        // An OpenAI-style base (ending in /v1) gets only the rest of the
        // route.
        vlm::VlmResult openai_base = vlm::generate(call_to(fake.endpoint() + "/gw/v1"));
        require(openai_base.ok, "an OpenAI-style base resolves: " + openai_base.error);
        require(fake.target() == "/gw/v1/chat/completions", "a /v1 base is not doubled");

        // A query string rides along on the request.
        vlm::VlmResult queried = vlm::generate(call_to(fake.endpoint() + "/base?tenant=a"));
        require(queried.ok, "an endpoint with a query resolves: " + queried.error);
        require(fake.target() == "/base/v1/chat/completions?tenant=a",
                "the query follows the completions route: " + fake.target());

        // Unreachable endpoints name the origin only: a token in the path
        // or the query never reaches the error text.
        vlm::VlmResult unreachable =
            vlm::generate(call_to("http://127.0.0.1:1/tenant/PATH-TOKEN?key=QUERY-TOKEN"));
        require(!unreachable.ok, "nothing listens on port 1");
        require(!unreachable.error.contains("PATH-TOKEN") &&
                    !unreachable.error.contains("QUERY-TOKEN"),
                "the unreachable error is redacted: " + unreachable.error);
        fake.attempts = 0;

        // Generation facts: the answering model, the stop reason verbatim,
        // and usage all survive the parse. An answer cut at max_tokens is
        // a plain 200 — "length" is the only thing that says so.
        fake.report_generation = true;
        vlm::VlmResult truncated = vlm::generate(call_to(fake.endpoint()));
        require(truncated.ok, "truncated answer is still a 200: " + truncated.error);
        require(truncated.finish_reason == "length", "finish_reason is kept verbatim");
        require(truncated.model == "served-model-b",
                "the model the endpoint says answered, not the one asked for");
        require(truncated.has_usage && truncated.prompt_tokens == 1200 &&
                    truncated.completion_tokens == 4096,
                "token usage is read from the response");
        fake.report_generation = false;

        // An endpoint that reports none of it leaves every field unset
        // rather than guessing.
        vlm::VlmResult silent = vlm::generate(call_to(fake.endpoint()));
        require(silent.ok && silent.finish_reason.empty() && silent.model.empty() &&
                    !silent.has_usage,
                "absent generation facts stay absent");

        // The recorded endpoint drops any path prefix: deployments put
        // tokens there and a fragment must not carry them.
        require(vlm::endpoint_origin("http://vlm:8080/base/secret") == "http://vlm:8080",
                "endpoint origin drops the path");
        require(vlm::endpoint_origin("http://vlm:8080") == "http://vlm:8080",
                "a bare origin is unchanged");
        require(vlm::endpoint_origin("http://vlm:8080?key=secret") == "http://vlm:8080",
                "endpoint origin drops a query that follows the authority directly");
        require(vlm::endpoint_origin("http://vlm:8080/v1#frag") == "http://vlm:8080",
                "endpoint origin drops the fragment");
        // An endpoint that does not parse is never shown as typed.
        require(vlm::endpoint_origin("http://alice:hunter2@vlm:8080") == "<invalid endpoint>",
                "a rejected endpoint is shown as a placeholder, not echoed");
        fake.attempts = 0;

        // Alternates: the parameter is omitted unless asked for, and what
        // comes back is kept verbatim, in generation order.
        vlm::VlmCall plain = call_to(fake.endpoint());
        vlm::VlmResult no_alternates = vlm::generate(plain);
        require(no_alternates.ok && no_alternates.alternatives.empty(),
                "no top_logprobs asked, no alternates carried");
        require(fake.asked_top_logprobs == -1,
                "the parameter is omitted rather than sent as zero");

        vlm::VlmCall nbest = call_to(fake.endpoint());
        nbest.top_logprobs = 2;
        vlm::VlmResult alternates = vlm::generate(nbest);
        require(alternates.ok, "n-best call succeeds: " + alternates.error);
        require(fake.asked_top_logprobs == 2, "top_logprobs reaches the wire");
        require(alternates.alternatives.size() == 4,
                "two alternates for each of two tokens");
        require(alternates.alternatives[0].token == "cat" &&
                    alternates.alternatives[1].token == "car" &&
                    alternates.alternatives[2].token == "sat" &&
                    alternates.alternatives[3].token == "set",
                "alternates keep generation order");
        require(alternates.alternatives[1].has_logprob &&
                    std::fabs(alternates.alternatives[1].logprob + 2.5) < 1e-9,
                "an alternate keeps its own score");
        require(!alternates.alternatives[3].has_logprob,
                "an alternate sent without a score claims none");
        require(alternates.has_logprobs && alternates.scored_tokens == 2,
                "the chosen tokens still drive the page score");
        fake.attempts = 0;

        // Transient failure: 503 once, then 200 — the page succeeds.
        fake.failures_before_success = 1;
        vlm::VlmResult result = vlm::generate(call_to(fake.endpoint()));
        require(result.ok, "transient 503 recovers: " + result.error);
        require(result.text == "<doctag/>", "response text after the retry");
        require(fake.attempts == 2, "one retry after the first 503");

        // Persistent 503: the call fails after 1 initial + 5 retries.
        fake.attempts = 0;
        fake.failures_before_success = 100;
        vlm::VlmResult failed = vlm::generate(call_to(fake.endpoint()));
        require(!failed.ok, "persistent 503 fails the call");
        require(failed.error.contains("503"), "the error names HTTP 503");
        require(fake.attempts > 1, "persistent 503 is retried");
        require(fake.attempts == 6, "1 initial attempt + 5 retries");

        // 429 and 504 are on docling's forcelist too.
        for (const int status : {429, 504}) {
            fake.attempts = 0;
            fake.failure_status = status;
            failed = vlm::generate(call_to(fake.endpoint()));
            require(!failed.ok && fake.attempts == 6,
                    "status " + std::to_string(status) + " is retried");
        }
        fake.failure_status = 503;

        // Other 4xx: no retry.
        fake.attempts = 0;
        fake.failure_status = 400;
        failed = vlm::generate(call_to(fake.endpoint()));
        require(!failed.ok, "400 fails");
        require(fake.attempts == 1, "400 is not retried");
        fake.failure_status = 503;

        // A 200 that does not parse is a failure, not a retry trigger.
        fake.attempts = 0;
        fake.failures_before_success = 0;
        fake.garbage_200 = true;
        failed = vlm::generate(call_to(fake.endpoint()));
        require(!failed.ok, "unparseable 200 fails");
        require(failed.error.contains("non-JSON"), "names the parse failure");
        require(fake.attempts == 1, "unparseable 200 is not retried");
        fake.garbage_200 = false;

        // Connect-level failure (nothing listening, as while vLLM starts):
        // retried, then surfaces as unreachable.
        failed = vlm::generate(call_to("http://127.0.0.1:1"));
        require(!failed.ok, "connection refused fails");
        require(failed.error.contains("unreachable"),
                "connection failure surfaces as unreachable");

        verify_budget_and_cancellation(fake);
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        fake.stop();
        return 1;
    }
    fake.stop();
    std::println("vlm-client-test passed");
    return 0;
}
