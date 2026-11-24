#pragma once

// Health follows the VLM endpoint. This process holds no model: with no
// endpoint a stream can use, or one that does not answer, every page of
// every stream fails, and a health check that still says SERVING sends an
// orchestrator's traffic to a service that cannot do its one job.

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include <grpcpp/health_check_service_interface.h>

#include "config.h"
#include "secret.h"

namespace vlm {

// Whether pages sent now could be answered, and why or why not. The detail
// names the endpoint by scheme, host and port only.
struct EndpointHealth {
    bool serving = false;
    std::string detail;
    bool operator==(const EndpointHealth&) const = default;
};

// One GET of the endpoint's model list (models_route), with the API key
// when one is set. Serving on any answer below 500 other than 401 and 403:
// a 2xx lists the models, and an endpoint without that route (404) still
// proved it is reachable, which is all this probe can know about it. Not
// serving when nothing answers within `timeout`, on 5xx (a model still
// loading answers 503), and on 401 or 403 (the endpoint refuses this
// server's credentials, so it would refuse its pages).
EndpointHealth probe_endpoint(const std::string& endpoint, const Secret& api_key,
                              std::chrono::milliseconds timeout);

// Keeps the verdict current: probes the configured endpoint once at
// start() and then every `interval`, and calls `on_change` with the first
// verdict and each one that differs from the last.
//
// With no endpoint configured there is nothing to probe: serving only when
// per-request endpoint overrides are allowed (a request can then bring its
// own), otherwise not serving for the life of the process. An interval of
// zero turns the probe off; health then reflects configuration alone, and
// the detail says it was not probed.
class EndpointMonitor {
  public:
    using OnChange = std::function<void(const EndpointHealth&)>;

    EndpointMonitor(const Config& config, std::chrono::milliseconds interval, OnChange on_change);
    ~EndpointMonitor();

    EndpointMonitor(const EndpointMonitor&) = delete;
    EndpointMonitor& operator=(const EndpointMonitor&) = delete;

    // Runs the first probe on the calling thread, so health is decided
    // before the caller goes on, then starts the repeating one.
    void start();
    void stop();

    // The latest verdict; not serving until start() has decided one.
    EndpointHealth health() const;

  private:
    void decide(EndpointHealth next);
    void run();

    const Config& config_;
    const std::chrono::milliseconds interval_;
    const OnChange on_change_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    EndpointHealth health_{.serving = false, .detail = "the endpoint has not been probed yet"};
    bool decided_ = false;
    bool stopping_ = false;
    std::thread thread_;
};

// Tells gRPC health what the endpoint verdict is: the overall status ("")
// and this service's own name. A null `health` (no health service
// registered) is a no-op.
void publish_health(grpc::HealthCheckServiceInterface* health, const EndpointHealth& verdict);

}  // namespace vlm
