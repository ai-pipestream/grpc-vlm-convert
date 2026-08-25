// Unit tests for the VLM HTTP client retry policy (docling's
// api_image_request: 5 retries, exponential backoff, on 429/500/502/503/
// 504 and connect-level transport failures) against an in-process fake
// server that counts attempts. Backoff is pinned to zero — the policy's
// attempt counts are the assertions, not the wall-clock delays.

#include <atomic>
#include <cmath>
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

    void start() {
        // The same completions handler under a base path: split_endpoint
        // must post to {prefix}/v1/chat/completions for prefixed endpoints.
        const auto handler = [this](const httplib::Request& request,
                                    httplib::Response& response) {
            attempts++;
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
        require(!vlm::endpoint_error("https://vlm:8080").empty(), "https is rejected");
        require(!vlm::endpoint_error("http://").empty(), "scheme without a host is rejected");
        require(!vlm::endpoint_error("vlm:8080").empty(), "missing scheme is rejected");

        // A path-prefixed endpoint posts under the prefix.
        vlm::VlmResult prefixed = vlm::generate(call_to(fake.endpoint() + "/base"));
        require(prefixed.ok, "prefixed endpoint resolves: " + prefixed.error);
        require(prefixed.text == "<doctag/>", "prefixed endpoint answer");
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
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        fake.stop();
        return 1;
    }
    fake.stop();
    std::println("vlm-client-test passed");
    return 0;
}
