#pragma once

#include <atomic>
#include <functional>

#include "ai/pipestream/vlm/v1/vlm_convert.grpc.pb.h"
#include "config.h"

namespace vlm {

// Synchronous gRPC service. Each ConvertPages stream runs its read loop on
// the RPC thread; page work (VLM HTTP call + mapping) runs on a per-stream
// worker pool bounded by the effective concurrency, and a single writer
// thread drains the event queue onto the wire. Events are emitted in
// completion order — a page's PageDocument goes out the moment its VLM
// call returns, never held for an earlier page.
class VlmConvertServiceImpl final
    : public ai::pipestream::vlm::v1::VlmConvertService::Service {
  public:
    explicit VlmConvertServiceImpl(const Config& config);

    // Transport-independent ConvertPages pipeline. `read` yields the next
    // client message (options first, then pages; false = half-close),
    // `write` consumes one stream event (false = consumer gone, mirrors a
    // failed gRPC Write), `cancelled` mirrors ServerContext::IsCancelled.
    // The gRPC override below is a thin adapter over this; the HTTP front
    // end drives it directly, so both transports share one pipeline.
    using ConvertRead =
        std::function<bool(ai::pipestream::vlm::v1::ConvertPagesRequest*)>;
    using ConvertWrite =
        std::function<bool(const ai::pipestream::vlm::v1::ConvertPagesResponse&)>;
    grpc::Status ConvertPagesCore(const ConvertRead& read, const ConvertWrite& write,
                                  const std::function<bool()>& cancelled);

    grpc::Status ConvertPages(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<ai::pipestream::vlm::v1::ConvertPagesResponse,
                                 ai::pipestream::vlm::v1::ConvertPagesRequest>* stream) override;

    grpc::Status GetServiceInfo(
        grpc::ServerContext* context,
        const ai::pipestream::vlm::v1::GetServiceInfoRequest* request,
        ai::pipestream::vlm::v1::GetServiceInfoResponse* response) override;

    // Metrics counters, printed by the interval line in main. converted
    // counts OK streams; rejected counts client-caused failures; failed
    // counts server-caused ones.
    std::atomic<long> converted{0};
    std::atomic<long> rejected{0};
    std::atomic<long> failed{0};
    std::atomic<long> pages_ok{0};
    std::atomic<long> pages_failed{0};

  private:
    const Config& config_;
};

}  // namespace vlm
