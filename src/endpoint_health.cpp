#include "endpoint_health.h"

#include <utility>

#include <httplib.h>

#include "ai/pipestream/vlm/v1/vlm_convert.grpc.pb.h"
#include "vlm_client.h"

namespace vlm {

namespace {

// How long one probe waits for the endpoint. A model list is a cheap
// answer; an endpoint that takes longer is not one to send pages to.
constexpr auto kProbeTimeout = std::chrono::seconds(5);

}  // namespace

EndpointHealth probe_endpoint(const std::string& endpoint, const Secret& api_key,
                              std::chrono::milliseconds timeout) {
    const std::optional<ModelsRoute> route = models_route(endpoint);
    if (!route) {
        return {.serving = false, .detail = "the endpoint is not a usable http:// URL"};
    }
    httplib::Client client(route->origin);
    if (!client.is_valid()) {
        return {.serving = false, .detail = "the endpoint is not usable: " + route->origin};
    }
    client.set_connection_timeout(timeout);
    client.set_read_timeout(timeout);
    client.set_write_timeout(timeout);
    client.set_max_timeout(timeout);
    httplib::Headers headers;
    if (!api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + api_key.reveal());
    }
    const httplib::Result response = client.Get(route->target, headers);
    if (!response) {
        return {.serving = false,
                .detail = "endpoint unreachable: " + route->origin + " (" +
                          httplib::to_string(response.error()) + ")"};
    }
    const int status = response->status;
    if (status >= 500 || status == 401 || status == 403) {
        return {.serving = false,
                .detail = "endpoint " + route->origin + " answered HTTP " +
                          std::to_string(status) + " to its model list" +
                          (status >= 500 ? "" : "; it refuses this server's credentials")};
    }
    return {.serving = true,
            .detail = "endpoint " + route->origin + " answered HTTP " + std::to_string(status)};
}

EndpointMonitor::EndpointMonitor(const Config& config, std::chrono::milliseconds interval,
                                 OnChange on_change)
    : config_(config), interval_(interval), on_change_(std::move(on_change)) {}

EndpointMonitor::~EndpointMonitor() { stop(); }

EndpointHealth EndpointMonitor::health() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return health_;
}

void EndpointMonitor::decide(EndpointHealth next) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool changed = !decided_ || next.serving != health_.serving;
        decided_ = true;
        health_ = next;
        if (!changed) {
            return;
        }
    }
    if (on_change_) {
        on_change_(next);
    }
}

void EndpointMonitor::start() {
    if (config_.endpoint.empty()) {
        decide(config_.allow_endpoint_override
                   ? EndpointHealth{.serving = true,
                                    .detail = "no endpoint configured; each request names its own"}
                   : EndpointHealth{.serving = false,
                                    .detail = "no VLM endpoint: GRPC_VLM_ENDPOINT is not set and "
                                              "per-request overrides are not allowed, so every "
                                              "ConvertPages stream fails"});
        return;
    }
    if (interval_ <= std::chrono::milliseconds(0)) {
        decide({.serving = true,
                .detail = "endpoint " + endpoint_origin(config_.endpoint) +
                          " is configured and not probed (GRPC_VLM_HEALTH_PROBE_SECONDS=0)"});
        return;
    }
    decide(probe_endpoint(config_.endpoint, config_.vlm_api_key, kProbeTimeout));
    thread_ = std::thread([this] { run(); });
}

void EndpointMonitor::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!wake_.wait_for(lock, interval_, [this] { return stopping_; })) {
        lock.unlock();
        decide(probe_endpoint(config_.endpoint, config_.vlm_api_key, kProbeTimeout));
        lock.lock();
    }
}

void EndpointMonitor::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void publish_health(grpc::HealthCheckServiceInterface* health, const EndpointHealth& verdict) {
    if (health == nullptr) {
        return;
    }
    health->SetServingStatus("", verdict.serving);
    health->SetServingStatus(ai::pipestream::vlm::v1::VlmConvertService::service_full_name(),
                             verdict.serving);
}

}  // namespace vlm
