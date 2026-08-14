// Unit tests for the VLM HTTP client retry policy (docling's
// api_image_request: 5 retries, exponential backoff, on 429/500/502/503/
// 504 and connect-level transport failures) against an in-process fake
// server that counts attempts. Backoff is pinned to zero — the policy's
// attempt counts are the assertions, not the wall-clock delays.

#include <atomic>
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

    void start() {
        server.Post("/v1/chat/completions",
                    [this](const httplib::Request&, httplib::Response& response) {
                        attempts++;
                        if (failures_before_success.load() > 0) {
                            failures_before_success--;
                            response.status = failure_status.load();
                            response.set_content("{\"error\":\"model overloaded\"}",
                                                 "application/json");
                            return;
                        }
                        if (garbage_200.load()) {
                            response.set_content("not json at all", "text/plain");
                            return;
                        }
                        nlohmann::json reply = {
                            {"choices",
                             {{{"message",
                                {{"role", "assistant"}, {"content", "<doctag/>"}}}}}}};
                        response.set_content(reply.dump(), "application/json");
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
    vlm::VlmCall call;
    call.endpoint = endpoint;
    call.model = "test-model";
    call.prompt = "prompt";
    call.png = "fake-png-bytes";
    call.timeout_seconds = 5;
    return call;
}

}  // namespace

int main() {
    vlm::set_retry_backoff_base_ms(0);
    ScriptableVlm fake;
    fake.start();
    try {
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
        require(failed.error.find("503") != std::string::npos, "the error names HTTP 503");
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
        require(failed.error.find("non-JSON") != std::string::npos, "names the parse failure");
        require(fake.attempts == 1, "unparseable 200 is not retried");
        fake.garbage_200 = false;

        // Connect-level failure (nothing listening, as while vLLM starts):
        // retried, then surfaces as unreachable.
        failed = vlm::generate(call_to("http://127.0.0.1:1"));
        require(!failed.ok, "connection refused fails");
        require(failed.error.find("unreachable") != std::string::npos,
                "connection failure surfaces as unreachable");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        fake.stop();
        return 1;
    }
    fake.stop();
    std::cout << "vlm-client-test passed\n";
    return 0;
}
