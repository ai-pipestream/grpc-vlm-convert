#include "vlm_convert_service.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
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

    enum class Popped { kValue, kClosed, kTimeout };

    // pop() that gives up after `timeout`, so the caller can look around.
    Popped pop_for(T* value, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!ready_.wait_for(lock, timeout, [&] { return closed_ || !queue_.empty(); })) {
            return Popped::kTimeout;
        }
        if (queue_.empty()) {
            return Popped::kClosed;
        }
        *value = std::move(queue_.front());
        queue_.pop_front();
        return Popped::kValue;
    }

  private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<T> queue_;
    bool closed_ = false;
};

// How often a stream asks its transport whether the client is still there.
constexpr auto kCancelPollInterval = std::chrono::milliseconds(50);

// The alternates-per-token ceiling OpenAI-compatible endpoints impose.
// Asking for more is a 400 from the endpoint, so it fails here first,
// before a page is paid for.
constexpr uint32_t kMaxTopLogprobs = 20;

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
    // The page's share of the stream's and the process's buffer budgets,
    // given back when the job is destroyed: once its answer is mapped, or
    // when a halted stream skips it.
    Lease stream_bytes;
    Lease process_bytes;
};

}  // namespace

VlmConvertServiceImpl::VlmConvertServiceImpl(const Config& config)
    : config_(config),
      buffered_bytes_(config.max_buffered_bytes),
      vlm_slots_(config.max_inflight) {}

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

    // A request naming the configured endpoint is not an override. Any
    // other endpoint is, and is refused unless the operator opted in: it
    // would let every caller make this process POST page images to any
    // http host it can reach and read the answer back (SSRF).
    const bool endpoint_override =
        !options.endpoint().empty() && options.endpoint() != config_.endpoint;
    if (endpoint_override && !config_.allow_endpoint_override) {
        return client_error(grpc::StatusCode::PERMISSION_DENIED,
                            "per-request endpoint overrides are disabled on this server (the "
                            "operator enables them with GRPC_VLM_ALLOW_ENDPOINT_OVERRIDE=true)");
    }
    const std::string endpoint = endpoint_override ? options.endpoint() : config_.endpoint;
    if (endpoint.empty()) {
        return client_error(grpc::StatusCode::FAILED_PRECONDITION,
                            config_.allow_endpoint_override
                                ? "no VLM endpoint configured (GRPC_VLM_ENDPOINT) and no "
                                  "per-request endpoint override"
                                : "no VLM endpoint configured (GRPC_VLM_ENDPOINT)");
    }
    if (const std::string problem = endpoint_error(endpoint); !problem.empty()) {
        return client_error(grpc::StatusCode::INVALID_ARGUMENT, problem);
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
    if (options.top_logprobs() > kMaxTopLogprobs) {
        return client_error(grpc::StatusCode::INVALID_ARGUMENT,
                            "top_logprobs must be 0.." +
                                std::to_string(kMaxTopLogprobs) + ", got " +
                                std::to_string(options.top_logprobs()));
    }
    // Alternates ride on logprobs; an endpoint the operator keeps them off
    // for would answer 400, so fail before a page is paid for.
    if (options.top_logprobs() > 0 && !config_.request_logprobs) {
        return client_error(grpc::StatusCode::FAILED_PRECONDITION,
                            "top_logprobs needs logprobs, which this server does not request "
                            "from its endpoint (GRPC_VLM_LOGPROBS=false)");
    }
    const size_t concurrency =
        options.concurrency() == 0
            ? config_.concurrency
            : std::min<size_t>(config_.concurrency, options.concurrency());

    // The page pipeline: reader → jobs → workers → events → writer.
    //
    // Pages are bounded by bytes: the read loop admits a page only when it
    // fits this stream's budget and the process-wide one, and holds it
    // (reading nothing more) until it does. The job queue is therefore
    // bounded by the leases its jobs carry. The event queue is not bounded
    // on purpose: a client that uploads every page before it reads a single
    // event (gRParse does) would otherwise deadlock against a full event
    // queue, and what it can hold is capped anyway at two events per page
    // for at most max_pages pages.
    Budget stream_bytes(config_.max_stream_buffered_bytes);
    Channel<PageJob> jobs;
    Channel<vlmv1::ConvertPagesResponse> events;
    std::atomic<uint32_t> started{0};
    std::atomic<uint32_t> ok{0};
    std::atomic<uint32_t> page_failed{0};
    // Set once nobody wants this stream's answers any more: the client
    // cancelled, its deadline passed, the consumer stopped taking events,
    // the input turned out bad, or abort_on_error met a failed page.
    // Workers then skip queued pages, and every VLM call in flight is cut
    // short (it is each call's cancel probe).
    std::atomic<bool> halt{false};
    std::atomic<bool> client_gone{false};
    auto halted = [&] { return halt.load(); };

    std::thread writer([&] {
        // The writer is the one thread that asks the transport whether the
        // client is still there. IsCancelled plucks the call's completion
        // queue, which tolerates only a handful of concurrent pluckers (a
        // pluck turned away can fail a real Read), so the workers and the
        // watchdogs of their VLM calls read `halt` instead.
        auto next_check = std::chrono::steady_clock::now();
        vlmv1::ConvertPagesResponse event;
        for (;;) {
            if (cancelled != nullptr && std::chrono::steady_clock::now() >= next_check) {
                if (cancelled()) {
                    client_gone = true;
                    halt = true;
                    return;
                }
                next_check = std::chrono::steady_clock::now() + kCancelPollInterval;
            }
            const auto popped = events.pop_for(&event, kCancelPollInterval);
            if (popped == Channel<vlmv1::ConvertPagesResponse>::Popped::kClosed) {
                return;
            }
            if (popped == Channel<vlmv1::ConvertPagesResponse>::Popped::kValue &&
                !write(event)) {
                client_gone = true;  // consumer gone
                halt = true;
                return;
            }
        }
    });

    // One page through the model and the mapper. False when the page
    // emits nothing: the stream stopped while it waited for a VLM slot or
    // while its call was in flight.
    auto convert_page = [&](PageJob& job, vlmv1::ConvertPagesResponse* event) {
        mapping::PageContext page;
        page.page_no = job.image.page_no();
        page.width = job.image.width();
        page.height = job.image.height();
        // The one copy of the raster: the call reads it in place, and the
        // DocTags mapper crops pictures out of it afterwards.
        page.png = std::move(*job.image.mutable_png());
        page.source.set_collector("vlm-convert");
        page.source.set_model(job.model);
        page.source.set_version(GRPC_VLM_VERSION);
        job.call.png = page.png;
        job.call.cancelled = halted;
        VlmResult result;
        {
            // Every call holds one of the process's in-flight slots, so
            // all streams together never have more than
            // GRPC_VLM_MAX_INFLIGHT requests open on the endpoint.
            const Lease slot = vlm_slots_.acquire(1, halted);
            if (!slot) {
                return false;
            }
            result = generate(job.call);
        }
        if (result.cancelled) {
            return false;  // cut short because nobody is waiting for it
        }
        if (result.has_logprobs) {
            // The response's mean token log-probability is a page-wide
            // statistic. It rides the source as the raw uncalibrated score
            // it is, with its kind naming the statistic and its scope, and
            // never as `confidence`: one number copied onto every item
            // would report a crisp heading and a hallucinated table as
            // equally trustworthy. Per-item confidence needs per-item token
            // spans, which the response does not carry.
            page.source.set_raw_score(result.mean_logprob);
            page.source.set_raw_score_kind("page_mean_token_logprob");
            page.source.set_raw_score_samples(result.scored_tokens);
        }
        // Generation provenance rides every item next to the collector
        // source: the model that actually answered (not only the one asked
        // for), the endpoint origin, the stop reason verbatim, and the
        // token cost. Without the stop reason a page cut off at max_tokens
        // is indistinguishable from a page that simply ended there.
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
        if (!result.alternatives.empty()) {
            // The readings the model weighed and did not take, verbatim and
            // in generation order. They describe the page, not any one
            // item: nothing in the response maps a token back to the item
            // it ended up in.
            page.has_alternatives = true;
            page.alternatives.set_created_by(page.generation.model());
            for (const TokenAlternative& alternate : result.alternatives) {
                auto* hypothesis = page.alternatives.add_hypotheses();
                hypothesis->set_text(alternate.token);
                if (alternate.has_logprob) {
                    hypothesis->set_raw_score(alternate.logprob);
                    hypothesis->set_raw_score_kind("token_logprob");
                }
            }
        }
        if (!result.ok) {
            auto* raw = event->mutable_page_raw();
            raw->set_page_no(page.page_no);
            raw->set_error(result.error);
            return true;
        }
        std::string map_error;
        docv1::Document fragment;
        std::vector<vlmv1::PageWarning> warnings;
        if (mapping::map_response(job.format, result.text, page, &fragment, &map_error,
                                  &warnings)) {
            auto* document = event->mutable_page_document();
            document->set_page_no(page.page_no);
            *document->mutable_document() = std::move(fragment);
            // What the mapper cut to a server cap rides beside the
            // fragment, so a short table never passes for a complete one.
            for (vlmv1::PageWarning& warning : warnings) {
                *document->add_warnings() = std::move(warning);
            }
        } else {
            // The model answered but not in its declared format: keep the
            // raw text, tag the reason.
            auto* raw = event->mutable_page_raw();
            raw->set_page_no(page.page_no);
            raw->set_text(result.text);
            raw->set_error(map_error);
        }
        return true;
    };

    std::vector<std::thread> workers;
    workers.reserve(concurrency);
    for (size_t i = 0; i < concurrency; i++) {
        workers.emplace_back([&] {
            for (;;) {
                // A fresh job per page: its budget leases and its raster
                // are given back the moment the page is done or skipped,
                // not when the next page arrives.
                PageJob job;
                if (!jobs.pop(&job)) {
                    return;
                }
                if (halted()) {
                    continue;  // nobody wants this page any more
                }
                vlmv1::ConvertPagesResponse event;
                try {
                    if (!convert_page(job, &event)) {
                        continue;
                    }
                } catch (const std::exception& error) {
                    // A throw out of this thread would end the process; it
                    // fails this page instead.
                    event.Clear();
                    auto* raw = event.mutable_page_raw();
                    raw->set_page_no(job.image.page_no());
                    raw->set_error(std::string("internal error converting the page: ") +
                                   error.what());
                }
                const bool failed_page = event.has_page_raw();
                if (failed_page) {
                    page_failed++;
                } else {
                    ok++;
                }
                events.push(std::move(event));
                if (failed_page && options.abort_on_error()) {
                    // The stream fails ABORTED now whatever happens to the
                    // rest, so stop paying for pages nobody will receive.
                    halt = true;
                }
            }
        });
    }

    // Read loop: pages are queued the moment they are admitted, so page 1's
    // events reach the client while later pages are still uploading.
    grpc::Status status = grpc::Status::OK;
    bool saw_input = false;
    while (!halted() && read(&request)) {
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
        // Provenance and the Document's pages map key pages as int32: a
        // larger page_no would come out negative.
        if (image.page_no() > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            status = client_error(grpc::StatusCode::INVALID_ARGUMENT,
                                  "page_no above 2147483647: " + std::to_string(image.page_no()));
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

        // Admission: the page waits here, off the queue, until its bytes
        // fit this stream's budget and then the process's. While it waits
        // nothing more is read, so the transport's flow control holds the
        // client back.
        const size_t bytes = image.png().size();
        Lease stream_lease = stream_bytes.acquire(bytes, halted);
        Lease process_lease = stream_lease ? buffered_bytes_.acquire(bytes, halted) : Lease();
        if (!process_lease) {
            break;  // the stream stopped while the page waited
        }

        started++;
        vlmv1::ConvertPagesResponse event;
        event.mutable_page_started()->set_page_no(image.page_no());
        events.push(std::move(event));

        jobs.push(PageJob{
            .image = std::move(*request.mutable_page_image()),
            .call = {.endpoint = endpoint,
                     .model = model,
                     .prompt = prompt,
                     .stop = stop,
                     .max_tokens = max_tokens,
                     .top_logprobs = static_cast<int>(options.top_logprobs()),
                     .logprobs = config_.request_logprobs,
                     .png = {},
                     .timeout_seconds = static_cast<long>(config_.vlm_timeout_seconds),
                     // The operator's key goes to the operator's endpoint
                     // only, never to one a request named.
                     .api_key = endpoint_override ? Secret() : config_.vlm_api_key,
                     .cancelled = {}},
            .format = format,
            .model = model,
            .stream_bytes = std::move(stream_lease),
            .process_bytes = std::move(process_lease),
        });
    }
    if (!status.ok()) {
        // A bad page fails the whole stream: pages already queued would be
        // paid for and thrown away, and calls in flight answer nobody.
        halt = true;
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
    if (client_gone.load() || is_cancelled()) {
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
    // Scheme, host and port only: this RPC is unauthenticated, and
    // deployments put tokens in the endpoint's path and query.
    if (!config_.endpoint.empty()) {
        response->set_endpoint(endpoint_origin(config_.endpoint));
    }
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
