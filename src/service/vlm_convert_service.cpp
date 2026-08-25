#include "vlm_convert_service.h"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#include "mapping/mapper.h"
#include "presets.h"
#include "vlm_client.h"

namespace vlm {

namespace {

namespace docv1 = ai::pipestream::document::v1;
namespace vlmv1 = ai::pipestream::vlm::v1;

// Closed FIFO: producers push, the consumer pops until closed and drained.
template <typename T>
class Channel {
  public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(std::move(value));
        }
        ready_.notify_one();
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        ready_.notify_all();
    }

    // False only once closed and drained.
    bool pop(T* value) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        *value = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

  private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<T> queue_;
    bool closed_ = false;
};

// PNG signature: 8 bytes, 89 50 4E 47 0D 0A 1A 0A.
bool is_png(const std::string& bytes) {
    static constexpr char kMagic[8] = {'\x89', 'P', 'N', 'G', '\x0d', '\x0a', '\x1a', '\x0a'};
    return bytes.starts_with(std::string_view(kMagic, sizeof(kMagic)));
}

// One page through the model queue: the image plus everything the worker
// needs to build the call and stamp the fragment.
struct PageJob {
    vlmv1::PageImage image;
    VlmCall call;
    vlmv1::ResponseFormat format;
    std::string model;
};

}  // namespace

VlmConvertServiceImpl::VlmConvertServiceImpl(const Config& config) : config_(config) {}

grpc::Status VlmConvertServiceImpl::ConvertPages(
    grpc::ServerContext* context,
    grpc::ServerReaderWriter<vlmv1::ConvertPagesResponse, vlmv1::ConvertPagesRequest>* stream) {
    return ConvertPagesCore(
        [&](vlmv1::ConvertPagesRequest* request) { return stream->Read(request); },
        [&](const vlmv1::ConvertPagesResponse& event) { return stream->Write(event); },
        [&] { return context->IsCancelled(); });
}

grpc::Status VlmConvertServiceImpl::ConvertPagesCore(
    const ConvertRead& read, const ConvertWrite& write,
    const std::function<bool()>& cancelled) {
    auto is_cancelled = [&] { return cancelled != nullptr && cancelled(); };
    auto client_error = [&](grpc::StatusCode code, const std::string& message) {
        rejected++;
        return grpc::Status(code, message);
    };

    // First message is the options; it chooses the input style.
    vlmv1::ConvertPagesRequest request;
    if (!read(&request) || !request.has_options()) {
        return client_error(grpc::StatusCode::INVALID_ARGUMENT,
                            "first stream message must be ConvertOptions");
    }
    const vlmv1::ConvertOptions options = request.options();

    const std::string endpoint =
        options.endpoint().empty() ? config_.endpoint : options.endpoint();
    if (endpoint.empty()) {
        return client_error(grpc::StatusCode::FAILED_PRECONDITION,
                            "no VLM endpoint configured (GRPC_VLM_ENDPOINT) and no per-request "
                            "endpoint override");
    }
    if (!endpoint_error(endpoint).empty()) {
        return client_error(grpc::StatusCode::INVALID_ARGUMENT, endpoint_error(endpoint));
    }

    std::string model, prompt;
    std::vector<std::string> stop;
    int max_tokens = 4096;
    vlmv1::ResponseFormat format;
    if (!resolve_request(options, &model, &prompt, &format, &stop, &max_tokens)) {
        return client_error(grpc::StatusCode::INVALID_ARGUMENT,
                            "preset resolves to no model name (set preset or preset_raw)");
    }
    // Proto3 keeps unknown enum ints; an unresolvable format would surface
    // as a per-page mapping failure only after paying for the VLM call.
    if (!vlmv1::ResponseFormat_IsValid(format)) {
        return client_error(grpc::StatusCode::INVALID_ARGUMENT,
                            "unknown response_format value: " +
                                std::to_string(static_cast<int>(format)));
    }
    const size_t concurrency =
        options.concurrency() == 0
            ? config_.concurrency
            : std::min<size_t>(config_.concurrency, options.concurrency());

    // The page pipeline: reader → jobs → workers → events → writer.
    Channel<PageJob> jobs;
    Channel<vlmv1::ConvertPagesResponse> events;
    std::atomic<uint32_t> started{0};
    std::atomic<uint32_t> ok{0};
    std::atomic<uint32_t> page_failed{0};

    std::thread writer([&] {
        vlmv1::ConvertPagesResponse event;
        while (events.pop(&event)) {
            if (!write(event)) {
                return;  // consumer gone; producers see the cancelled flag
            }
        }
    });

    std::vector<std::thread> workers;
    workers.reserve(concurrency);
    for (size_t i = 0; i < concurrency; i++) {
        workers.emplace_back([&] {
            PageJob job;
            while (jobs.pop(&job)) {
                if (is_cancelled()) {
                    continue;
                }
                vlmv1::ConvertPagesResponse event;
                mapping::PageContext page;
                page.page_no = job.image.page_no();
                page.width = job.image.width();
                page.height = job.image.height();
                page.png = job.image.png();
                page.source.set_collector("vlm-convert");
                page.source.set_model(job.model);
                page.source.set_version(GRPC_VLM_VERSION);
                // The raster then moves into the call rather than riding
                // the queue twice (once on the image, once on the call).
                job.call.png = std::move(*job.image.mutable_png());
                VlmResult result = generate(job.call);
                if (result.has_logprobs) {
                    // The response's mean token log-probability is a
                    // page-wide statistic. It rides the source as the raw
                    // uncalibrated score it is, with its kind naming the
                    // statistic and its scope, and never as `confidence`:
                    // one number copied onto every item would report a
                    // crisp heading and a hallucinated table as equally
                    // trustworthy. Per-item confidence needs per-item
                    // token spans, which the response does not carry.
                    page.source.set_raw_score(result.mean_logprob);
                    page.source.set_raw_score_kind("page_mean_token_logprob");
                }
                // Generation provenance rides every item next to the
                // collector source: the model that actually answered (not
                // only the one asked for), the endpoint origin, the stop
                // reason verbatim, and the token cost. Without the stop
                // reason a page cut off at max_tokens is indistinguishable
                // from a page that simply ended there.
                page.has_generation = true;
                page.generation.set_model(result.model.empty() ? job.model : result.model);
                page.generation.set_endpoint(endpoint_origin(job.call.endpoint));
                if (!result.finish_reason.empty()) {
                    page.generation.set_finish_reason(result.finish_reason);
                }
                if (result.has_usage) {
                    page.generation.set_prompt_tokens(result.prompt_tokens);
                    page.generation.set_completion_tokens(result.completion_tokens);
                }
                if (!result.ok) {
                    auto* raw = event.mutable_page_raw();
                    raw->set_page_no(job.image.page_no());
                    raw->set_error(result.error);
                    page_failed++;
                } else {
                    std::string map_error;
                    docv1::Document fragment;
                    if (mapping::map_response(job.format, result.text, page, &fragment,
                                              &map_error)) {
                        auto* document = event.mutable_page_document();
                        document->set_page_no(job.image.page_no());
                        *document->mutable_document() = std::move(fragment);
                        ok++;
                    } else {
                        // The model answered but not in its declared
                        // format: keep the raw text, tag the reason.
                        auto* raw = event.mutable_page_raw();
                        raw->set_page_no(job.image.page_no());
                        raw->set_text(result.text);
                        raw->set_error(map_error);
                        page_failed++;
                    }
                }
                events.push(std::move(event));
            }
        });
    }

    // Read loop: pages are queued the moment they arrive so page 1's
    // events reach the client while later pages are still uploading.
    grpc::Status status = grpc::Status::OK;
    bool saw_input = false;
    while (read(&request)) {
        if (request.has_options()) {
            status = client_error(grpc::StatusCode::INVALID_ARGUMENT,
                                  "ConvertOptions must not repeat on the stream");
            break;
        }
        if (request.has_pdf_chunk()) {
            // Rasterizing PDFs is the fallback path; v1 ships no
            // rasterizer — send PNG pages instead (the coordinator's CV
            // path already renders them).
            status = client_error(grpc::StatusCode::UNIMPLEMENTED,
                                  "PDF input needs a rasterizer this build does not carry; send "
                                  "PageImage PNGs instead");
            break;
        }
        if (!request.has_page_image()) {
            continue;  // payload-less message: ignore, keep the stream alive
        }
        saw_input = true;
        const vlmv1::PageImage& image = request.page_image();
        if (image.page_no() == 0) {
            status = client_error(grpc::StatusCode::INVALID_ARGUMENT,
                                  "page_no is 1-based; got 0");
            break;
        }
        if (image.png().size() > config_.max_page_bytes) {
            status = client_error(grpc::StatusCode::RESOURCE_EXHAUSTED,
                                  "page PNG above the " +
                                      std::to_string(config_.max_page_bytes) + " byte cap");
            break;
        }
        if (!is_png(image.png())) {
            status = client_error(grpc::StatusCode::INVALID_ARGUMENT,
                                  "page bytes are not a PNG");
            break;
        }
        if (started.load() >= config_.max_pages) {
            status = client_error(grpc::StatusCode::RESOURCE_EXHAUSTED,
                                  "stream above the " + std::to_string(config_.max_pages) +
                                      " page cap");
            break;
        }

        started++;
        vlmv1::ConvertPagesResponse event;
        event.mutable_page_started()->set_page_no(image.page_no());
        events.push(std::move(event));

        jobs.push(PageJob{
            .image = image,
            .call = {.endpoint = endpoint,
                     .model = model,
                     .prompt = prompt,
                     .stop = stop,
                     .max_tokens = max_tokens,
                     .png = {},
                     .timeout_seconds = static_cast<long>(config_.vlm_timeout_seconds)},
            .format = format,
            .model = model,
        });
    }
    jobs.close();
    for (std::thread& worker : workers) {
        worker.join();
    }
    events.close();
    writer.join();

    if (!status.ok()) {
        return status;
    }
    if (is_cancelled()) {
        return grpc::Status(grpc::StatusCode::CANCELLED, "client cancelled the stream");
    }
    if (!saw_input) {
        return client_error(grpc::StatusCode::INVALID_ARGUMENT, "stream carried no pages");
    }
    if (options.abort_on_error() && page_failed.load() > 0) {
        failed++;
        return grpc::Status(grpc::StatusCode::ABORTED,
                            "abort_on_error set and " + std::to_string(page_failed.load()) +
                                " page(s) failed");
    }

    // The writer is joined; the trailer goes out on this thread, last.
    vlmv1::ConvertPagesResponse complete;
    auto* trailer = complete.mutable_complete();
    trailer->set_pages_started(started.load());
    trailer->set_pages_ok(ok.load());
    trailer->set_pages_failed(page_failed.load());
    write(complete);

    converted++;
    pages_ok += ok.load();
    pages_failed += page_failed.load();
    return grpc::Status::OK;
}

grpc::Status VlmConvertServiceImpl::GetServiceInfo(
    grpc::ServerContext* /*context*/, const vlmv1::GetServiceInfoRequest* /*request*/,
    vlmv1::GetServiceInfoResponse* response) {
    response->set_version(GRPC_VLM_VERSION);
    response->set_endpoint(config_.endpoint);
    // Report what the configured endpoint claims to serve; with no
    // endpoint nothing claims anything.
    if (!config_.endpoint.empty()) {
        if (config_.presets.empty()) {
            for (const PresetSpec& spec : all_presets()) {
                response->add_presets(spec.preset);
            }
        } else {
            for (const std::string& name : config_.presets) {
                const PresetSpec* spec = find_preset_by_name(name);
                if (spec != nullptr) {
                    response->add_presets(spec->preset);
                } else {
                    response->add_raw_presets(name);
                }
            }
        }
    }
    response->set_concurrency(static_cast<uint32_t>(config_.concurrency));
    response->set_max_page_bytes(config_.max_page_bytes);
    response->set_max_pages(static_cast<uint32_t>(config_.max_pages));
    // Shared-shell frontend advertisement; the values are fixed for this
    // service, not configuration.
    auto* ui = response->mutable_ui();
    ui->set_title("VLM Convert");
    ui->set_path("/ui/vlm-convert");
    ui->set_description("Calls an external VLM server for document conversion");
    return grpc::Status::OK;
}

}  // namespace vlm
