// Health follows the VLM endpoint, not the process. With no endpoint a
// stream can use, or one that does not answer, every page of every stream
// fails, so the service must not report SERVING. The probe is one GET of
// the endpoint's model list; the monitor repeats it and reports each
// change.

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <grpcpp/health_check_service_interface.h>
#include <httplib.h>

#include "config.h"
#include "endpoint_health.h"
#include "fixture.h"

namespace {

using std::chrono::milliseconds;

// A VLM endpoint that answers GET <prefix>/v1/models with `status`.
struct FakeEndpoint {
    httplib::Server server;
    std::jthread thread;
    int port = 0;
    std::atomic<int> status{200};
    std::mutex mutex;
    std::vector<std::string> paths;
    std::string authorization;

    void start(int on_port = 0) {
        server.Get(R"(.*)", [this](const httplib::Request& request, httplib::Response& response) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                paths.push_back(request.path);
                authorization = request.get_header_value("Authorization");
            }
            response.status = status.load();
            response.set_content(R"({"object":"list","data":[]})", "application/json");
        });
        if (on_port == 0) {
            port = server.bind_to_any_port("127.0.0.1");
        } else {
            require(server.bind_to_port("127.0.0.1", on_port), "fake endpoint rebinds its port");
            port = on_port;
        }
        require(port > 0, "fake endpoint bound");
        thread = std::jthread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }

    void stop() {
        server.stop();
        if (thread.joinable()) {
            thread.join();
        }
    }

    // A test that fails mid-way must not hang on the listener thread.
    ~FakeEndpoint() { stop(); }

    std::string last_path() {
        std::lock_guard<std::mutex> lock(mutex);
        return paths.empty() ? "" : paths.back();
    }

    std::string base() const { return "http://127.0.0.1:" + std::to_string(port); }
};

bool wait_until(const std::function<bool()>& condition) {
    for (int i = 0; i < 500; i++) {  // 5 s budget, ms-scale in practice
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(milliseconds(10));
    }
    return false;
}

void verify_probe_reads_the_model_list() {
    FakeEndpoint fake;
    fake.start();
    const vlm::Secret no_key;

    // Each endpoint shape VlmCall::endpoint accepts lands on its own
    // /v1/models.
    for (const auto& [endpoint, path] : std::vector<std::pair<std::string, std::string>>{
             {fake.base(), "/v1/models"},
             {fake.base() + "/", "/v1/models"},
             {fake.base() + "/v1", "/v1/models"},
             {fake.base() + "/v1/chat/completions", "/v1/models"},
             {fake.base() + "/gateway/v1/chat/completions?tenant=a", "/gateway/v1/models"},
         }) {
        const vlm::EndpointHealth health = vlm::probe_endpoint(endpoint, no_key, milliseconds(2000));
        require(health.serving, "an endpoint that lists its models is serving: " + health.detail);
        require(fake.last_path() == path,
                "the probe of <" + endpoint + "> asks " + path + ", asked " + fake.last_path());
    }

    // The operator's key goes with the probe: an endpoint that needs it
    // would otherwise answer 401 to the probe and 200 to the pages.
    vlm::probe_endpoint(fake.base(), vlm::Secret("sk-probe"), milliseconds(2000));
    {
        std::lock_guard<std::mutex> lock(fake.mutex);
        require(fake.authorization == "Bearer sk-probe", "the probe carries the API key");
    }

    // A model still loading, a gateway error, credentials refused: pages
    // would fail, so the endpoint is not serving.
    for (const int status : {500, 502, 503, 401, 403}) {
        fake.status = status;
        const vlm::EndpointHealth health =
            vlm::probe_endpoint(fake.base(), no_key, milliseconds(2000));
        require(!health.serving && health.detail.contains(std::to_string(status)),
                "HTTP " + std::to_string(status) + " is not serving: " + health.detail);
    }
    // An endpoint with no model list still answered: it is reachable, and
    // that is all this probe can know about it.
    fake.status = 404;
    require(vlm::probe_endpoint(fake.base(), no_key, milliseconds(2000)).serving,
            "an endpoint without a model list counts as reachable");

    const int port = fake.port;
    fake.stop();
    const vlm::EndpointHealth down = vlm::probe_endpoint(
        "http://127.0.0.1:" + std::to_string(port), no_key, milliseconds(2000));
    require(!down.serving && down.detail.contains("unreachable"),
            "a closed port is not serving: " + down.detail);
    // The detail never quotes more than the origin: endpoints hide tokens
    // in their path and query.
    const vlm::EndpointHealth secret = vlm::probe_endpoint(
        "http://127.0.0.1:" + std::to_string(port) + "/t0ken/v1?key=s3cret", no_key,
        milliseconds(2000));
    require(!secret.detail.contains("t0ken") && !secret.detail.contains("s3cret"),
            "the detail never quotes the endpoint's path or query: " + secret.detail);
}

struct Changes {
    std::mutex mutex;
    std::vector<bool> seen;
    void record(bool serving) {
        std::lock_guard<std::mutex> lock(mutex);
        seen.push_back(serving);
    }
    std::vector<bool> all() {
        std::lock_guard<std::mutex> lock(mutex);
        return seen;
    }
};

void verify_no_endpoint() {
    vlm::Config config;  // no endpoint, no override
    vlm::EndpointMonitor refused(config, milliseconds(20), {});
    refused.start();
    require(!refused.health().serving && refused.health().detail.contains("GRPC_VLM_ENDPOINT"),
            "no endpoint and no override: not serving, and it says which variable: " +
                refused.health().detail);

    config.allow_endpoint_override = true;
    vlm::EndpointMonitor overridable(config, milliseconds(20), {});
    overridable.start();
    require(overridable.health().serving,
            "no endpoint but overrides allowed: a request can bring its own, so serving");
}

void verify_monitor_follows_the_endpoint() {
    FakeEndpoint fake;
    fake.start();
    const int port = fake.port;
    fake.stop();

    vlm::Config config;
    config.endpoint = "http://127.0.0.1:" + std::to_string(port);
    Changes changes;
    vlm::EndpointMonitor monitor(config, milliseconds(20),
                                 [&](const vlm::EndpointHealth& health) {
                                     changes.record(health.serving);
                                 });
    require(!monitor.health().serving, "not serving before the first probe has passed");
    monitor.start();
    require(!monitor.health().serving, "an endpoint that is down: not serving");

    FakeEndpoint up;
    up.start(port);
    require(wait_until([&] { return monitor.health().serving; }),
            "the endpoint came up: serving within a few probes");
    up.status = 503;
    require(wait_until([&] { return !monitor.health().serving; }),
            "the endpoint started answering 503: not serving again");
    up.status = 200;
    require(wait_until([&] { return monitor.health().serving; }), "and serving once it recovers");
    monitor.stop();
    require(changes.all() == std::vector<bool>({false, true, false, true}),
            "every change is reported once, in order, starting with the first verdict");
}

void verify_probe_can_be_turned_off() {
    vlm::Config config;
    config.endpoint = "http://127.0.0.1:1";  // nothing listens there
    vlm::EndpointMonitor monitor(config, milliseconds(0), {});
    monitor.start();
    require(monitor.health().serving && monitor.health().detail.contains("not probed"),
            "interval 0: health reflects configuration only, and says so: " +
                monitor.health().detail);
}

// What gRPC health is told: the overall status and this service's own.
struct RecordingHealth final : grpc::HealthCheckServiceInterface {
    std::vector<std::pair<std::string, bool>> named;
    std::vector<bool> all;
    void SetServingStatus(const std::string& service_name, bool serving) override {
        named.emplace_back(service_name, serving);
    }
    void SetServingStatus(bool serving) override { all.push_back(serving); }
};

void verify_grpc_health_is_published() {
    RecordingHealth health;
    vlm::publish_health(&health, {.serving = false, .detail = "endpoint unreachable"});
    vlm::publish_health(&health, {.serving = true, .detail = "ok"});
    const std::vector<std::pair<std::string, bool>> want = {
        {"", false},
        {"ai.pipestream.vlm.v1.VlmConvertService", false},
        {"", true},
        {"ai.pipestream.vlm.v1.VlmConvertService", true},
    };
    require(health.named == want,
            "the overall status and the service's own both follow the endpoint");
    vlm::publish_health(nullptr, {.serving = true, .detail = "ok"});  // no health service: a no-op
}

}  // namespace

int main() {
    try {
        verify_probe_reads_the_model_list();
        verify_no_endpoint();
        verify_monitor_follows_the_endpoint();
        verify_probe_can_be_turned_off();
        verify_grpc_health_is_published();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("endpoint-health-test passed");
    return 0;
}
